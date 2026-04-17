# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a WorldComposer teleoperation with WorldComposer manipulation environments."""

import argparse
import yaml
from isaaclab.app import AppLauncher
import os
import sys
import torch
import numpy as np
from pathlib import Path
import gymnasium as gym
import logging
import traceback
import datetime
from contextlib import contextmanager
import random

# ==============================================================================
# 1. Parse Command-Line Arguments & Launch App
# ==============================================================================
parser = argparse.ArgumentParser(
    description="WorldComposer teleoperation script for manipulation environments."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
    help="Device for interacting with environment",
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Port for the single teleop device")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0", help="Port for left teleop device")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1", help="Port for right teleop device")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# Recording parameters
parser.add_argument("--record", action="store_true", default=False, help="Enable dataset recording")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--recalibrate", action="store_true", default=False, help="Recalibrate devices")
parser.add_argument("--num_episode", type=int, default=20, help="Maximum number of episodes to record")
parser.add_argument("--disable_depth", action="store_true", default=False, help="Disable recording depth")
parser.add_argument("--enable_pointcloud", action="store_true", default=False, help="Enable pointcloud recording")
parser.add_argument("--task_description", type=str, default="fold the garment on the table", help="Task description")

# Append AppLauncher CLI arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"

# Pre-parse YAML to get enable_cameras flag BEFORE AppLauncher
if args_cli.config and os.path.exists(args_cli.config):
    with open(args_cli.config, "r") as f:
        yaml_config = yaml.safe_load(f)
        if yaml_config.get("teleop", {}).get("enable_cameras", False):
            # We must set this directly in sys.argv so AppLauncher picks it up
            if "--enable_cameras" not in sys.argv:
                sys.argv.append("--enable_cameras")
                # Re-parse args so args_cli has it too
                args_cli = parser.parse_args()

# Launch Omniverse application (MUST HAPPEN BEFORE OTHER ISAACLAB IMPORTS)
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ==============================================================================
# 2. Imports that require SimulationApp to be initialized
# ==============================================================================
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import WorldComposer.tasks  # Register tasks
from WorldComposer.devices import Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard
from WorldComposer.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from WorldComposer.utils.record import (
    RateLimiter,
    append_episode_initial_pose,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Create logger
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. Context Managers & Utilities
# ==============================================================================


@contextmanager
def suppress_video_encoder_output():
    """Temporarily suppress video encoder output (stderr and stdout)."""
    original_stderr_fd = sys.stderr.fileno()
    original_stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    saved_stdout_fd = os.dup(original_stdout_fd)
    original_stderr = sys.stderr
    original_stdout = sys.stdout
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        devnull_fd = devnull.fileno()
        os.dup2(devnull_fd, original_stderr_fd)
        os.dup2(devnull_fd, original_stdout_fd)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.dup2(saved_stdout_fd, original_stdout_fd)
        except OSError:
            sys.stderr = original_stderr
            sys.stdout = original_stdout
        finally:
            try:
                os.close(saved_stderr_fd)
                os.close(saved_stdout_fd)
            except OSError:
                pass
        if devnull is not None and not devnull.closed:
            devnull.close()

def validate_task_and_device(args: argparse.Namespace) -> None:
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert args.teleop_device in ["bi-so101leader", "bi-keyboard"], "Only support bi-arm devices for bi-arm tasks"
    else:
        assert args.teleop_device in ["so101leader", "keyboard"], "Only support single-arm devices for single-arm tasks"

def create_teleop_interface(env: DirectRLEnv, args: argparse.Namespace):
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(env, left_port=args.left_arm_port, right_port=args.right_arm_port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(f"Invalid device interface '{args.teleop_device}'.")


def cleanup_teleop_interface(teleop_interface) -> None:
    """Best-effort cleanup for teleop devices when rebuilding the env."""
    listener = getattr(teleop_interface, "listener", None)
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass

    disconnect_fn = getattr(teleop_interface, "disconnect", None)
    if callable(disconnect_fn):
        try:
            disconnect_fn()
        except Exception:
            pass


def mark_recording_started(teleop_interface, flags: dict) -> None:
    """Keep teleop state consistent when the env is rebuilt mid-recording."""
    flags["start"] = True
    if hasattr(teleop_interface, "_started"):
        teleop_interface._started = True
    if hasattr(teleop_interface, "other_key_enable"):
        teleop_interface.other_key_enable = True
    if hasattr(teleop_interface, "b_disable"):
        teleop_interface.b_disable = True

def register_teleop_callbacks(teleop_interface):
    flags = {"start": False, "success": False, "remove": False, "abort": False}
    def on_start():
        flags["start"] = True
        logger.info("[S] Recording started!")
    def on_success():
        flags["success"] = True
        logger.info("[N] Mark the current episode as successful.")
    def on_remove():
        flags["remove"] = True
        logger.info("[D] Discard the current episode and re-record.")
    def on_abort():
        flags["abort"] = True
        logger.warning("[ESC] Abort recording, clearing the current episode buffer...")
        
    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)
    return flags

# ==============================================================================
# 3. Dynamic Dataset Creation & Initialization Flow
# ==============================================================================
def warmup_and_reset(env: DirectRLEnv, device: str, is_initial: bool = False):
    """
    Reset environment and run physics for a few steps to let objects settle.
    Crucial for reproducible Replays (especially for soft bodies / fluids).
    """
    logger.info("Resetting environment and warming up physics...")
    env.reset()
    
    # Force the robot into its default joint position to prevent drift
    env_ids = torch.tensor([0], dtype=torch.int32, device=device)
    initial_joint_pos = env.robot.data.default_joint_pos[env_ids]
    
    env.robot.write_joint_position_to_sim(initial_joint_pos[0], env_ids=env_ids)
    env.robot.set_joint_position_target(initial_joint_pos)
    
    if hasattr(env, "scene") and hasattr(env.scene, "write_data_to_sim"):
        env.scene.write_data_to_sim()
        
    # Step physics to let deformable objects or fluids settle down
    settle_steps = 30 if is_initial else 10
    for _ in range(settle_steps):
        env.sim.step(render=False)
        # Keep enforcing target so arms don't drop during settling
        env.robot.set_joint_position_target(initial_joint_pos)
        simulation_app.update()
        
    # Obtain true initial pose of objects after settling
    object_initial_pose = env.get_all_pose() if hasattr(env, "get_all_pose") else None
    
    if hasattr(env, "initialize_obs"):
        env.initialize_obs()
        
    return object_initial_pose


def apply_task_yaml_to_env_cfg(env_cfg, yaml_config: dict) -> None:
    """Apply task-level YAML overrides before the environment is created."""
    env_cfg.task_config = yaml_config

    env_section = yaml_config.get("env", {})
    if "episode_length_s" in env_section:
        env_cfg.episode_length_s = env_section["episode_length_s"]
    if "action_scale" in env_section:
        env_cfg.action_scale = env_section["action_scale"]

    scene_section = yaml_config.get("scene", {})
    if "path_scene" in scene_section:
        scene_path = scene_section["path_scene"]
        env_cfg.path_scene = (
            scene_path if os.path.isabs(scene_path) else os.path.join(os.getcwd(), scene_path)
        )

    # Object instance randomization: pick a pool of variants before gym.make().
    # All variants are spawned at startup (hidden underground) and switched each episode.
    rand_cfg = yaml_config.get("randomization", {})
    object_cfg = rand_cfg.get("object_instance", {})
    if object_cfg.get("enable", False):
        max_variants = int(object_cfg.get("max_object_variants", 5))

        def _find_all_usds(folder_path: str) -> list[str]:
            result = []
            if not folder_path or not os.path.exists(folder_path):
                return result
            for root, _, files in os.walk(folder_path):
                for f in files:
                    if f.endswith(".usd") or f.endswith(".usda"):
                        result.append(os.path.join(root, f))
            return result

        # Generically resolve any key ending with "_folder"
        for key, value in object_cfg.items():
            if key.endswith("_folder") and isinstance(value, str):
                obj_name = key[:-7]  # e.g., "object_A_folder" -> "object_A"
                folder_path = value
                if folder_path and not os.path.isabs(folder_path):
                    folder_path = os.path.join(os.getcwd(), folder_path)
                
                all_usds = _find_all_usds(folder_path)
                pool = random.sample(all_usds, min(max_variants, len(all_usds))) if all_usds else []
                
                # Store the resolved pools in task_config so task envs can read them
                yaml_config[f"_resolved_{obj_name}_usds"] = pool
                
                # Set the first variant as the default USD path for cfg validation
                if pool and hasattr(env_cfg, obj_name):
                    obj_cfg_instance = getattr(env_cfg, obj_name)
                    if hasattr(obj_cfg_instance, "spawn") and hasattr(obj_cfg_instance.spawn, "usd_path"):
                        obj_cfg_instance.spawn.usd_path = pool[0]


def build_runtime(args: argparse.Namespace, yaml_config: dict):
    """Create env + teleop runtime using the current YAML configuration."""
    env_cfg = parse_env_cfg(args.task, device=args.device)
    apply_task_yaml_to_env_cfg(env_cfg, yaml_config)
    env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(teleop_interface)
    rate_limiter = RateLimiter(args.step_hz)
    return env, teleop_interface, flags, rate_limiter

def create_dataset_dynamically(args: argparse.Namespace, sample_obs: dict):
    """
    Dynamically infer dataset feature shapes from sample observation.
    Separates task-specific logic (cameras/resolutions) from teleop code.
    """
    if not args.record:
        return None, None, None, False

    is_bi_arm = ("Bi" in (args.task or "")) or (args.teleop_device or "").startswith("bi-")
    
    action_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    if is_bi_arm:
        joint_names = [f"left_{n}" for n in action_names] + [f"right_{n}" for n in action_names]
    else:
        joint_names = action_names

    dim = len(joint_names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (dim,), "names": joint_names},
    }

    # Dynamically extract image and depth shapes from observations
    for key, value in sample_obs.items():
        if key.startswith("observation.images."):
            features[key] = {
                "dtype": "video",
                "shape": tuple(value.shape),
                "names": ["height", "width", "channels"]
            }
        elif "depth" in key and not args.disable_depth:
            features[key] = {
                "dtype": "float32",
                "shape": tuple(value.shape),
                "names": ["height", "width"]
            }

    task_name = args.task or "default_task"
    
    # Create Output/Record/{task_name}/Record_XXX_YYYYMMDDHHMMSS
    base_path = Path(os.getcwd()) / "Output" / "Record" / task_name
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing_records = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Record_")]
    indices = []
    for name in existing_records:
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            indices.append(int(parts[1]))
            
    next_idx = 1
    while next_idx in indices:
        next_idx += 1
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    record_dir_name = f"Record_{next_idx:03d}_{timestamp}"
    root_path = base_path / record_dir_name
    
    dataset = LeRobotDataset.create(
        repo_id=f"local/{task_name}",
        fps=30,
        root=root_path,
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

    return dataset, jsonl_path, None, is_bi_arm

# ==============================================================================
# 4. Core Phases
# ==============================================================================
def run_idle_phase(env, teleop_interface, rate_limiter, args):
    """Wait for S key to start recording, while keeping physics alive."""
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()
    
    if actions is None:
        env.render()
    else:
        env.step(actions)

def run_recording_phase(
    env,
    teleop_interface,
    rate_limiter,
    args,
    flags,
    dataset,
    jsonl_path,
    object_initial_pose,
    ee_solver,
    is_bi_arm,
    yaml_config,
):
    """Execute recording loop."""
    episode_index = 0
    step_counter = 0
    teleop_cfg = yaml_config.get("teleop", {})
    rebuild_every = int(teleop_cfg.get("rebuild_env_every_n_success", 0) or 0)

    while episode_index < args.num_episode:
        if flags["abort"]:
            dataset.clear_episode_buffer()
            with suppress_video_encoder_output():
                dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return env, teleop_interface, flags, rate_limiter, object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        step_counter = 0

        while not flags["success"]:
            if flags["abort"]:
                dataset.clear_episode_buffer()
                with suppress_video_encoder_output():
                    dataset.finalize()
                return env, teleop_interface, flags, rate_limiter, object_initial_pose

            dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
            actions = teleop_interface.advance()

            if actions is None:
                env.render()
            else:
                env.step(actions)

            observations = env._get_observations()
            
            # Remove excluded depths
            if args.disable_depth:
                keys_to_remove = [k for k in observations.keys() if "depth" in k]
                for k in keys_to_remove:
                    observations.pop(k)

            # Record pointclouds
            if args.enable_pointcloud and dataset is not None:
                pointcloud = env._get_workspace_pointcloud(num_points=4096, use_fps=True)
                if pointcloud is not None:
                    pc_np = pointcloud.cpu().detach().numpy() if hasattr(pointcloud, "cpu") else np.array(pointcloud)
                    pc_dir = Path(dataset.root) / "pointclouds" / f"episode_{episode_index:03d}"
                    pc_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(str(pc_dir / f"frame_{step_counter:06d}.npz"), pointcloud=pc_np)

            _, truncated = env._get_dones()
            frame = {**observations, "task": args.task_description}

            dataset.add_frame(frame)
            step_counter += 1

            # Sync time with physics dt
            if rate_limiter:
                rate_limiter.sleep(env)

            if truncated or flags["remove"]:
                dataset.clear_episode_buffer()
                logger.info(f"Re-recording episode {episode_index}")
                object_initial_pose = warmup_and_reset(env, args.device, is_initial=False)
                flags["remove"] = False
                break  # Break inner loop to restart episode

        if flags["success"]:
            with suppress_video_encoder_output():
                dataset.save_episode()
            append_episode_initial_pose(jsonl_path, episode_index, object_initial_pose)
            
            episode_index += 1
            logger.info(f"Episode {episode_index - 1} completed, progress: {episode_index}/{args.num_episode}")
            object_initial_pose = warmup_and_reset(env, args.device, is_initial=False)

    dataset.clear_episode_buffer()
    with suppress_video_encoder_output():
        dataset.finalize()
    logger.info(f"All {args.num_episode} episodes recording completed!")
    return env, teleop_interface, flags, rate_limiter, object_initial_pose

# ==============================================================================
# 5. Main Execution
# ==============================================================================
def main():
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False

    validate_task_and_device(args_cli)
    
    # Load YAML config if provided
    yaml_config = {}
    if args_cli.config and os.path.exists(args_cli.config):
        with open(args_cli.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        logger.info(f"Loaded YAML config from {args_cli.config}")
        
        # Override args_cli with YAML teleop settings
        teleop_cfg = yaml_config.get("teleop", {})
        if "device" in teleop_cfg: args_cli.teleop_device = teleop_cfg["device"]
        if "port" in teleop_cfg: args_cli.port = teleop_cfg["port"]
        if "num_envs" in teleop_cfg: args_cli.num_envs = teleop_cfg["num_envs"]
        if "device_target" in teleop_cfg: args_cli.device = teleop_cfg["device_target"]
        if "record" in teleop_cfg: args_cli.record = teleop_cfg["record"]
        if "disable_depth" in teleop_cfg: args_cli.disable_depth = teleop_cfg["disable_depth"]
        if "enable_cameras" in teleop_cfg: args_cli.enable_cameras = teleop_cfg["enable_cameras"]
        if "num_episode" in teleop_cfg: args_cli.num_episode = teleop_cfg["num_episode"]

    # If enable_cameras is true in YAML, we MUST set it in AppLauncher config 
    # BEFORE AppLauncher is initialized. But AppLauncher is already initialized above.
    # So we need to update the simulation_app config directly, or ensure args_cli.enable_cameras
    # is passed to AppLauncher.
    
    env, teleop_interface, flags, rate_limiter = build_runtime(args_cli, yaml_config)
    
    # 1. Warm up physics & obtain precise initial pose
    object_initial_pose = warmup_and_reset(env, args_cli.device, is_initial=True)
    teleop_interface.reset()
    
    # 2. Dynamically create dataset based on exact observations returned
    sample_obs = env._get_observations()
    dataset, jsonl_path, ee_solver, is_bi_arm = create_dataset_dynamically(args_cli, sample_obs)

    printed_instructions = False
    idle_frame_counter = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    run_idle_phase(env, teleop_interface, rate_limiter, args_cli)
                    
                    idle_frame_counter += 1
                    if idle_frame_counter == 100 and not printed_instructions:
                        logger.info("=" * 60 + "\n🎮 CONTROL INSTRUCTIONS 🎮\n" + "=" * 60)
                        logger.info(str(teleop_interface))
                        logger.info("=" * 60 + "\n")
                        printed_instructions = True

                elif args_cli.record and dataset is not None:
                    env, teleop_interface, flags, rate_limiter, object_initial_pose = run_recording_phase(
                        env, teleop_interface, rate_limiter, args_cli, flags,
                        dataset, jsonl_path, object_initial_pose, ee_solver, is_bi_arm, yaml_config
                    )
                    break
                else:
                    run_idle_phase(env, teleop_interface, rate_limiter, args_cli)

    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        if args_cli.record and dataset is not None and flags["start"]:
            logger.info("Clearing current episode buffer and saving dataset...")
            dataset.clear_episode_buffer()
            with suppress_video_encoder_output():
                dataset.finalize()
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
