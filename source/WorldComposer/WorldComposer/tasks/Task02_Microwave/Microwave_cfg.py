from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_CFG

@configclass
class MicrowaveEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 6
    observation_space = 6
    state_space = 0
    
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # robot
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.051, -0.38, 0.49844),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": -1.5,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            }
        ),
    )

    # Cameras
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=960,
        height=720,
        update_period=1 / 30.0,
    )
    
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.02, -1.116, 1.070),
            rot=[0.55917, -0.82905, 0, 0],
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=960,
        height=720,
    )

    # Microwave Base Config (Used as template for pool)
    microwave: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd() + "/Assets/objects/microwave/Microwave011/Microwave011.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
            scale=(0.8, 0.8, 0.8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, -10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={},
        soft_joint_pos_limit_factor=1.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
    path_scene: str = os.getcwd() + "/Assets/scenes/Marble/Scene_01_LoftwithKitchen/Scene_01_LoftwithKitchen.usd"

    # This will be populated dynamically from task_config.yaml
    task_config: dict = None
