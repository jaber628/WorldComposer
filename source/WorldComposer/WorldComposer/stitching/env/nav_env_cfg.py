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
    obs_rgb_source: str = "gs"
    obs_depth_source: str = "gs"
    gs_ply_path: str = str(SCENE_DIR / "stitched/merged.ply")
    model_to_world: tuple = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    opacity_threshold: float = 0.1
    camera_relative_height_m: float = 1.4
    grid_map_path: str = str(SCENE_DIR / "stitched/grid_map.png")
    grid_meta_path: str = str(SCENE_DIR / "stitched/grid_meta.json")
    grid_resolution_m: float = 0.05

    marker_objects: dict[str, tuple[float, float, float]] = {}

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

    visual_scene = None

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/xlerobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(ASSETS_ROOT) / "robots/xlerobot/xlerobot/xlerobot.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=get_rotation(0.0, 0.0, 0.0),
        ),
        actuators={
            "base_drive": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0
            ),
        },
    )

    def __post_init__(self):
        self.scene.dome_light = self.dome_light
        self.scene.visual_scene = self.visual_scene
        self.scene.robot = self.robot
        super().__post_init__()

    @classmethod
    def from_scene(cls, scene_dir, device="cuda:0"):
        from WorldComposer.stitching.utils.navigation import load_scene_config

        data = load_scene_config(scene_dir)
        cfg = cls()
        cfg.sim.device = device
        cfg.gs_ply_path = data["model"]
        cfg.model_to_world = data["model_to_world"]
        cfg.opacity_threshold = float(data.get("opacity_threshold", 0.1))
        cfg.grid_map_path = data["grid_map"]
        cfg.grid_meta_path = data["grid_meta"]
        cfg.obs_rgb_source = data.get("rgb_source", "gs")
        cfg.obs_depth_source = data.get("depth_source", "gs")
        cfg.camera_relative_height_m = float(data.get("camera_height", 1.4))
        cfg.robot.init_state.pos = tuple(data["robot_position"])
        cfg.robot.init_state.rot = get_rotation(0.0, 0.0, float(data["robot_yaw"]))
        cfg.marker_objects = data.get("markers", {})
        if data.get("visual_scene"):
            cfg.visual_scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/visual_scene",
                spawn=sim_utils.UsdFileCfg(usd_path=data["visual_scene"]),
            )
        cfg.scene.visual_scene = cfg.visual_scene
        cfg.scene.robot = cfg.robot
        return cfg
