from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_CFG
from WorldComposer.utils.constant import ASSETS_ROOT


@configclass
class AssembleBurgerEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg, device="cuda:0"
    )

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.0, 0.20546, 0.49903),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": -0.01176,
            },
        ),
    )

    burger_beef: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_beef",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/burger/Assets/Burger_Beef_Patties001/Burger_Beef_Patties001_Def.usd"),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.09664, -0.11681, 0.55093)),
    )
    burger_cheese: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_cheese",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/burger/Assets/Burger_Cheese001/Burger_Cheese001_Def.usd"),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.16834, -0.0919, 0.56)),
    )
    burger_bread_Top: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_bread_Top",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/burger/Assets/Burger_Bread002/Burger_Bread002.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.16834, -0.0919, 0.55)),
    )
    burger_board: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/burger/Assets/Burger_ChoppingBlock/Burger_ChoppingBlock.usd"),
            scale=(1.0, 0.9, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.29479, -0.0919, 0.534), rot=(0.70711, 0.0, 0.0, -0.70711),
        ),
    )
    burger_plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/plates/Plate2/Burger_Plate.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.16834, -0.0919, 0.54)),
    )

    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.08, -0.5, 0.6), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7, focus_distance=400.0, horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0), lock_camera=True,
        ),
        width=640, height=480,
    )

    path_scene: str = str(Path(ASSETS_ROOT) / "scenes/InternetScenes/Scene_04_WarmStudio.usd")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    viewer = ViewerCfg(eye=(1.0, -1.3, 1.0), lookat=(0.0, -0.05, 0.55))
    task_config: dict = None
