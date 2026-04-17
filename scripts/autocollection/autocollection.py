# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Automatic Data Collection Pipeline.

This script launches a task environment and drives it using a task-specific Skill class
instead of human teleoperation. It follows the same structure as teleop_record.py.

Usage:
    python scripts/AutoCollection/AutoCollection.py --task Isaac-Tableware-v0 --num_episodes 50
"""

import argparse
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. Parse Command-Line Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description="Automatic data collection pipeline using task-specific skills.")
parser.add_argument("--task", type=str, default=None, help="Name of the registered gym task.")
parser.add_argument("--num_episodes", type=int, default=20, help="Number of episodes to collect.")
parser.add_argument("--step_hz", type=int, default=60, help="Control loop rate in Hz (must match task decimation).")
parser.add_argument("--record", action="store_true", default=False, help="Enable dataset recording.")
parser.add_argument("--disable_depth", action="store_true", default=False, help="Disable depth recording.")
parser.add_argument("--enable_pointcloud", action="store_true", default=False, help="Enable pointcloud recording.")
parser.add_argument("--task_description", type=str, default="", help="Language description of the task.")
parser.add_argument("--max_steps_per_episode", type=int, default=3600, help="Max steps before force-resetting an episode.")
parser.add_argument("--settle_steps", type=int, default=30, help="Physics warmup steps after each reset.")

# AppLauncher args (e.g. --headless --device cuda:0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"

# Launch Omniverse application BEFORE any isaaclab/torch imports
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import os
import sys
import torch
import numpy as np
import logging
import traceback
from pathlib import Path
from contextlib import contextmanager

import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from WorldComposer.utils.record import get_next_experiment_path_with_gap, RateLimiter, append_episode_initial_pose

LeRobotDataset = None

logger = logging.getLogger(__name__)

# ==============================================================================
# 2. Utilities (shared with teleop_record.py style)
# ==============================================================================
@contextmanager
def suppress_video_encoder_output():
    """Suppress ffmpeg / SVT-AV1 encoder noise in the terminal."""
    original_stderr_fd = sys.stderr.fileno()
    original_stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    saved_stdout_fd = os.dup(original_stdout_fd)
    original_stderr, original_stdout = sys.stderr, sys.stdout
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        os.dup2(devnull.fileno(), original_stderr_fd)
        os.dup2(devnull.fileno(), original_stdout_fd)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.dup2(saved_stdout_fd, original_stdout_fd)
        except OSError:
            sys.stderr, sys.stdout = original_stderr, original_stdout
        finally:
            try:
                os.close(saved_stderr_fd)
                os.close(saved_stdout_fd)
            except OSError:
                pass
        if devnull and not devnull.closed:
            devnull.close()


# ==============================================================================
# 3. Environment Initialization & Physics Warmup
# ==============================================================================
def warmup_and_reset(env: DirectRLEnv, device: str, settle_steps: int = 10) -> dict | None:
    """
    Reset the environment and run physics for `settle_steps` steps so that objects
    (especially deformable / fluid) reach a stable resting state before data collection.

    Returns the ground-truth initial pose of all objects after settling.
    """
    env.reset()

    # Force robot to its default pose and lock target to prevent actuator drift
    env_ids = torch.tensor([0], dtype=torch.int32, device=device)
    initial_joint_pos = env.robot.data.default_joint_pos[env_ids]
    env.robot.write_joint_position_to_sim(initial_joint_pos[0], env_ids=env_ids)
    env.robot.set_joint_position_target(initial_joint_pos)

    if hasattr(env, "scene") and hasattr(env.scene, "write_data_to_sim"):
        env.scene.write_data_to_sim()

    # Let physics run until objects are truly at rest
    for _ in range(settle_steps):
        env.sim.step(render=False)
        env.robot.set_joint_position_target(initial_joint_pos)
        simulation_app.update()

    # Capture ground-truth initial pose for replay / logging
    initial_pose = env.get_all_pose() if hasattr(env, "get_all_pose") else None

    if hasattr(env, "initialize_obs"):
        env.initialize_obs()

    return initial_pose


# ==============================================================================
# 4. Dynamic Dataset Creation (camera-agnostic, future-proof)
# ==============================================================================
def create_dataset_dynamically(args: argparse.Namespace, sample_obs: dict):
    """
    Infer dataset feature shapes from a sample observation dict.
    This decouples the recording script from task-specific camera configs.
    """
    if not args.record:
        return None, None

    global LeRobotDataset
    if LeRobotDataset is None:
        try:
            from WorldComposer.datasets.lerobot_dataset import LeRobotDataset as _LRD
            LeRobotDataset = _LRD
        except ImportError:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LRD
                LeRobotDataset = _LRD
            except ImportError:
                raise ImportError(
                    "LeRobotDataset is not available. Please install WorldComposer.datasets or lerobot correctly, "
                    "or run without --record first."
                )

    is_bi_arm = "Bi" in (args.task or "") or (args.task or "").startswith("bi-")
    action_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    joint_names = ([f"left_{n}" for n in action_names] + [f"right_{n}" for n in action_names]) if is_bi_arm else action_names
    dim = len(joint_names)

    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": joint_names},
        "action":            {"dtype": "float32", "shape": (dim,), "names": joint_names},
    }

    # Auto-discover cameras and depth maps from sample observation
    for key, value in sample_obs.items():
        if key.startswith("observation.images."):
            features[key] = {
                "dtype": "video",
                "shape": tuple(value.shape),
                "names": ["height", "width", "channels"],
            }
        elif "depth" in key and not args.disable_depth:
            features[key] = {
                "dtype": "float32",
                "shape": tuple(value.shape),
                "names": ["height", "width"],
            }

    task_name = args.task or "default_task"
    root_path = Path("Datasets/auto") / task_name
    dataset = LeRobotDataset.create(
        repo_id="abc",
        fps=30,
        root=get_next_experiment_path_with_gap(root_path),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )
    try:
        dataset.meta.update_chunk_settings(video_files_size_in_mb=0.1)
    except Exception as e:
        logger.warning(f"Failed to set video size limit: {e}")

    jsonl_path = dataset.root / "meta" / "object_initial_pose.jsonl"
    return dataset, jsonl_path


# ==============================================================================
# 5. Skill Loading (generic, task-agnostic)
# ==============================================================================
def load_skill(env: DirectRLEnv, env_cfg) -> object:
    """
    Load the task-specific Skill class defined in the task config.

    Each task's *Cfg class should declare:
        skill_class: str = "WorldComposer.tasks.Task01_Tableware.Tableware_Skill.TablewareSkill"

    Returns an instantiated Skill object bound to the environment.
    """
    skill_class_path: str | None = getattr(env_cfg, "skill_class", None)
    if skill_class_path is None:
        raise AttributeError(
            "env_cfg has no `skill_class` attribute. "
            "Please add `skill_class: str = 'module.path.SkillClass'` to your task config."
        )

    # Dynamic import: "a.b.c.ClassName" -> import "a.b.c", getattr "ClassName"
    module_path, class_name = skill_class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    SkillClass = getattr(module, class_name)

    logger.info(f"Loaded skill class: {skill_class_path}")
    return SkillClass(env)


# ==============================================================================
# 6. Collection Loop
# ==============================================================================
def run_collection_loop(
    env: DirectRLEnv,
    skill,
    rate_limiter: RateLimiter,
    args: argparse.Namespace,
    dataset,
    jsonl_path,
    initial_pose: dict | None,
):
    """
    Core automatic data collection loop.

    Each episode:
      1. skill.reset() - let the skill know a new episode started.
      2. skill.step()  - returns (action_tensor, done_flag, info_dict) each control tick.
      3. Save observations to dataset.
      4. On episode end: save_episode(), reset env, proceed to next.
    """
    episode_index = 0
    object_initial_pose = initial_pose

    while episode_index < args.num_episodes:
        logger.info(f"--- Episode {episode_index + 1} / {args.num_episodes} ---")

        skill.reset()
        step_counter = 0
        episode_success = False

        while step_counter < args.max_steps_per_episode:
            # ----------------------------------------------------------------
            # skill.step() is the ONLY entry point for control logic.
            # It should return:
            #   action      : torch.Tensor  - joint position targets, shape [1, dof]
            #   done        : bool          - True when skill declares episode finished
            #   success     : bool          - True if task was accomplished
            #   info        : dict          - arbitrary extra info (ignored here)
            # ----------------------------------------------------------------
            action, done, success, info = skill.step()

            env.step(action)
            step_counter += 1

            # Save frame if recording
            if args.record and dataset is not None:
                observations = env._get_observations()

                if args.disable_depth:
                    for k in [k for k in observations if "depth" in k]:
                        observations.pop(k)

                if args.enable_pointcloud:
                    pc = env._get_workspace_pointcloud(num_points=4096, use_fps=True) if hasattr(env, "_get_workspace_pointcloud") else None
                    if pc is not None:
                        pc_np = pc.cpu().detach().numpy() if hasattr(pc, "cpu") else np.array(pc)
                        pc_dir = Path(dataset.root) / "pointclouds" / f"episode_{episode_index:03d}"
                        pc_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(str(pc_dir / f"frame_{step_counter:06d}.npz"), pointcloud=pc_np)

                frame = {**observations, "task": args.task_description}
                dataset.add_frame(frame)

                if rate_limiter:
                    rate_limiter.sleep(env)

            if done:
                episode_success = success
                break

        # Episode finished (by skill or timeout)
        if episode_success:
            if args.record and dataset is not None:
                with suppress_video_encoder_output():
                    dataset.save_episode()
                if jsonl_path is not None:
                    append_episode_initial_pose(jsonl_path, episode_index, object_initial_pose)
            logger.info(f"Episode {episode_index} SUCCESS.")
            episode_index += 1
        else:
            # Discard failed episode
            if args.record and dataset is not None:
                dataset.clear_episode_buffer()
            logger.warning(f"Episode {episode_index} FAILED (success={episode_success}, steps={step_counter}). Discarding.")

        # Reset & re-settle for next episode
        object_initial_pose = warmup_and_reset(env, args.device, args.settle_steps)

    if args.record and dataset is not None:
        dataset.clear_episode_buffer()
        with suppress_video_encoder_output():
            dataset.finalize()
    logger.info(f"Collection complete. Saved {episode_index} episodes.")


# ==============================================================================
# 7. Main
# ==============================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.propagate = False

    if args_cli.task is None:
        raise ValueError("Please specify --task.")

    # Build env config (reads from task-specific *Cfg class)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    # Create environment
    env: DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Warmup
    initial_pose = warmup_and_reset(env, args_cli.device, is_initial := args_cli.settle_steps * 2)

    # Load task-specific skill (reads skill_class from env_cfg)
    skill = load_skill(env, env_cfg)

    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.record else None

    # Create dataset by probing what observations the env actually returns
    sample_obs = env._get_observations()
    dataset, jsonl_path = create_dataset_dynamically(args_cli, sample_obs)

    try:
        run_collection_loop(
            env=env,
            skill=skill,
            rate_limiter=rate_limiter,
            args=args_cli,
            dataset=dataset,
            jsonl_path=jsonl_path,
            initial_pose=initial_pose,
        )
    except KeyboardInterrupt:
        logger.warning("[Ctrl+C] Interrupted.")
        if args_cli.record and dataset is not None:
            dataset.clear_episode_buffer()
            with suppress_video_encoder_output():
                dataset.finalize()
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
