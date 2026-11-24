from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_CYLINDER_CFG
from WorldComposer.utils.constant import ASSETS_ROOT


@configclass
class FoldClothEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 12
    observation_space = 12
    state_space = 0

    render_cfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA",
        carb_settings={"/physics/updateToUsd": True, "/physics/updateParticlesToUsd": True, "/physics/updateVelocitiesToUsd": True},
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg, device="cuda:0", use_fabric=True,
    )


    left_robot: ArticulationCfg = SO101_FOLLOWER_CYLINDER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CYLINDER_CFG.init_state.replace(
            pos=(-0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )
    right_robot: ArticulationCfg = SO101_FOLLOWER_CYLINDER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CYLINDER_CFG.init_state.replace(
            pos=(0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
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
        width=640,
        height=480,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    viewer = ViewerCfg(eye=(1.0, -1.3, 1.1), lookat=(0.0, 0.0, 0.6))
    path_scene: str = str(Path(ASSETS_ROOT) / "scenes/InternetScenes/Scene_04_WarmStudio.usd")
    garment_index: int | None = None
    task_config: dict = None
