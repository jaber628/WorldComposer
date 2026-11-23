from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from WorldComposer.assets.robots.lerobot import SO101_FOLLOWER_CFG
from WorldComposer.utils.constant import ASSETS_ROOT


@configclass
class PourWaterEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    render_cfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA",
        enable_translucency=True, enable_reflections=True,
        carb_settings={"/rtx/translucency/worldEps": 0.0001},
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg, use_fabric=False, device="cpu"
    )

    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-0.13, -0.15, 0.49844),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ),
    )

    cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/mugs/mug/mug.usd"),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.16, 0.17, 0.568)),
    )
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/bowls/Bowl016/Bowl016.usd"),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.17, 0.56)),
    )

    camera_spawn = sim_utils.PinholeCameraCfg(
        focal_length=40, focus_distance=400.0, horizontal_aperture=38.11,
        clipping_range=(0.01, 50.0), lock_camera=True,
    )
    behind_camera = TiledCameraCfg(
        prim_path="/World/camera_behind",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.91884, 1.1839, 0.84114),
            rot=(0.26669, 0.23745, 0.62114, 0.69762), convention="opengl",
        ),
        data_types=["rgb"], spawn=camera_spawn, width=640, height=480,
    )
    front_camera = TiledCameraCfg(
        prim_path="/World/camera_front",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-1.28949, 1.26392, 0.8215),
            rot=(-0.28394, -0.25893, 0.62207, 0.68218), convention="opengl",
        ),
        data_types=["rgb"], spawn=camera_spawn, width=640, height=480,
    )

    fluid_usd_path: str = str(Path(ASSETS_ROOT) / "objects/fluid/water.usdc")
    path_scene: str = str(Path(ASSETS_ROOT) / "scenes/InternetScenes/Scene_04_WarmStudio.usd")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
    viewer = ViewerCfg(eye=(-0.8, -1.2, 1.1), lookat=(0.0, 0.15, 0.55))
    task_config: dict = None
