from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_45_CFG


@configclass
class TablewareEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, render=render_cfg)

    robot: ArticulationCfg = SO101_FOLLOWER_45_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_45_CFG.init_state.replace(
            pos=(-0.13, -0.15, 0.49844),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": -1.5,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ),
    )

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

    object_A: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/object_A",
        spawn=sim_utils.UsdFileCfg(usd_path=os.getcwd() + "/Assets/objects/cups/b_cups/b_cups.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object_B: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/object_B",
        spawn=sim_utils.UsdFileCfg(usd_path=os.getcwd() + "/Assets/objects/plates/plate/plate_1.2.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1493, -0.06069, 0.56771),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(-0.06, 1.15, 1), lookat=(0.1, -0.1, 0.4))
    path_scene: str = os.getcwd() + "/Assets/scenes/Marble/Scene_04_WarmStudio.usd"
    task_config: dict = None
