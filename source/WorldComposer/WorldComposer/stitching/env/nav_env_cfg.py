from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from WorldComposer.utils.constant import ASSETS_ROOT


def get_rotation(r: float, p: float, y: float):
    rad = torch.tensor([r, p, y], dtype=torch.float32) * (np.pi / 180.0)
    q = quat_from_euler_xyz(rad[0:1], rad[1:2], rad[2:3]).squeeze(0)
    return tuple(q.tolist())


SCENE_DIR = Path(ASSETS_ROOT) / "scenes/MultiroomScenes/MultiRoom01"


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 300.0
    action_space = 4
    observation_space = 3
    use_grid_obstacle = True
    obs_rgb_source: str = "sim"
    obs_depth_source: str = "merged"
    gs_ply_path: str = str(SCENE_DIR / "multi_room.ply")
    camera_relative_height_m: float = 1.4
    grid_map_path: str = str(SCENE_DIR / "grid_map.png")
    grid_meta_path: str = str(SCENE_DIR / "grid_meta.json")
    grid_resolution_m: float = 0.05
    grid_z_min_m: float = 0.1
    grid_z_max_m: float = 0.8
    grid_margin_ratio: float = 1.1

    marker_objects: dict[str, tuple[float, float, float]] = {
        "refrigerator": (-11.71497574751114, -5.034363007897863, 0.7263209023375464),
        "chair": (2.978491408869553, -5.985242144378995, 0.9763456365481269),
        "toilet": (2.898488996678908, -1.2987390655525088, 0.3578256175643223),
    }

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        device="cuda:0",
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0)

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=800.0, exposure=0.0),
    )

    visual_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/visual_scene",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_DIR / "default.usda")),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/xlerobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "robots/xlerobot/xlerobot/xlerobot.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.703, -1.717, 0.1),
            rot=get_rotation(0.0, 0.0, 180.0),
        ),
        actuators={
            "base_drive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
        },
    )

    microwave = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/microwave/Microwave081/Microwave081.usd"),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-10.084, -7.997, 1.443),
            rot=get_rotation(0.0, 0.0, 176.0),
        ),
    )

    oven = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/oven",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "objects/oven/Oven037/Oven037.usd"),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-11.961790905897422, -7.233579507261785, 1.5508209694653439),
            rot=get_rotation(0.0, 0.0, 83.0),
        ),
    )

    def __post_init__(self):
        self.scene.dome_light = self.dome_light
        self.scene.visual_scene = self.visual_scene
        self.scene.robot = self.robot
        self.scene.microwave = self.microwave
        self.scene.oven = self.oven
        super().__post_init__()
