from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import quat_apply_inverse

from .PourWater_cfg import PourWaterEnvCfg
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.constant import ASSETS_ROOT
from WorldComposer.utils.fluid import FluidSystem
from WorldComposer.utils.rendering import apply_default_render_settings


class PourWaterEnv(DirectRLEnv):
    cfg: PourWaterEnvCfg

    def __init__(self, cfg: PourWaterEnvCfg, render_mode: str | None = None, **kwargs):
        self._is_closed = True
        if cfg.sim.device != "cpu":
            raise ValueError("PourWater requires the CPU tensor interface; use --device cpu.")
        if cfg.scene.num_envs != 1:
            raise ValueError("PourWater supports a single environment; use num_envs=1.")
        with Path(__file__).with_name("task_config.yaml").open() as f:
            self.task_config = yaml.safe_load(f)
        self.task_config.update(cfg.task_config or {})
        cfg.task_config = self.task_config
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.actions = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self):
        apply_default_render_settings()
        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func("/World/Scene", scene_cfg)
        self._resolve_table_texture_paths()

        self.robot = Articulation(self.cfg.robot)
        self.cup = RigidObject(self.cfg.cup)
        self.bowl = RigidObject(self.cfg.bowl)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cup"] = self.cup
        self.scene.rigid_objects["bowl"] = self.bowl

        self.enable_cameras = self.task_config["teleop"].get("enable_cameras", False)
        if self.enable_cameras:
            self.behind_camera = TiledCamera(self.cfg.behind_camera)
            self.front_camera = TiledCamera(self.cfg.front_camera)
            self.scene.sensors["behind_camera"] = self.behind_camera
            self.scene.sensors["front_camera"] = self.front_camera

        fluid_cfg = self.task_config["fluid"]
        self.fluid = FluidSystem(
            prim_path="/World/Object/fluid", usd_path=self.cfg.fluid_usd_path,
            material_path=str(Path(__file__).parent / fluid_cfg["material_path"]), config=fluid_cfg,
        )
        UsdPhysics.MassAPI(self.fluid.point_instancer.GetPrim()).GetMassAttr().Set(fluid_cfg["mass"])
        UsdGeom.PrimvarsAPI(self.fluid.particle_system.GetPrim()).CreatePrimvar(
            "doNotCastShadows", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant,
        ).Set(not fluid_cfg.get("cast_shadows", True))
        offset = Gf.Vec3f(*fluid_cfg.get("offset", (0.0, 0.0, 0.0)))
        self.fluid.init_particle_positions = [point + offset for point in self.fluid.init_particle_positions]
        light = self.scene.stage.GetPrimAtPath(f"{self.fluid.prim_path}/env_light")
        if light:
            light.SetActive(False)
        self.fluid.particle_system.CreateSimulationOwnerRel().SetTargets([self.cfg.sim.physics_prim_path])
        self.fluid.reset(list(self.cfg.cup.init_state.pos), list(self.cfg.cup.init_state.rot))
        self.sim.get_physics_context().enable_ccd(False)
        self.sim.get_physics_context().enable_gpu_dynamics(True)
        self.sim.get_physics_context().set_broadphase_type("GPU")

    def _resolve_table_texture_paths(self):
        texture_dir = Path(ASSETS_ROOT) / "objects/table/Table038/texture"
        for prim in self.scene.stage.Traverse():
            if "Table038" not in str(prim.GetPath()) or not prim.IsA(UsdShade.Shader):
                continue
            texture_input = UsdShade.Shader(prim).GetInput("file")
            asset = texture_input.Get() if texture_input else None
            if isinstance(asset, Sdf.AssetPath):
                texture = texture_dir / Path(asset.path).name
                if texture.is_file():
                    texture_input.Set(Sdf.AssetPath(str(texture)))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        self.fluid.get_particle_positions_world()
        observations = {
            "action": self.actions[0].detach().cpu().numpy(),
            "observation.state": self.robot.data.joint_pos[0].detach().cpu().numpy(),
        }
        if self.enable_cameras:
            for name in ("behind", "front"):
                camera = getattr(self, f"{name}_camera")
                observations[f"observation.images.{name}_rgb"] = camera.data.output["rgb"][0].cpu().numpy()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.episode_length_buf, dtype=torch.float)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> torch.Tensor:
        particles = torch.as_tensor(
            self.fluid.get_particle_positions_world(), device=self.device, dtype=torch.float32,
        )
        success = False
        if len(particles):
            local = quat_apply_inverse(
                self.bowl.data.root_quat_w[0].expand(len(particles), -1),
                particles - self.bowl.data.root_pos_w[0],
            )
            success_cfg = self.task_config["success"]
            z_min, z_max = success_cfg["height_range"]
            inside = (
                (local[:, :2].norm(dim=-1) <= success_cfg["radius"])
                & (local[:, 2] >= z_min) & (local[:, 2] <= z_max)
            )
            success = bool(inside.float().mean() >= success_cfg["success_ratio"])
        return torch.full_like(self.episode_length_buf, success, dtype=torch.bool)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        for obj in (self.robot, self.cup, self.bowl):
            obj.write_root_state_to_sim(obj.data.default_root_state[env_ids].clone(), env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = self.robot.data.default_joint_pos.clone()
        cup_state = self.cup.data.default_root_state[0]
        self.fluid.reset(cup_state[:3].cpu().tolist(), cup_state[3:7].cpu().tolist())

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        poses = {name: obj.data.root_state_w[0].detach().cpu().numpy().copy()
                 for name, obj in (("robot", self.robot), ("cup", self.cup), ("bowl", self.bowl))}
        poses["joint_pos"] = self.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        return poses

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        for name, obj in (("robot", self.robot), ("cup", self.cup), ("bowl", self.bowl)):
            state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32)
            obj.write_root_state_to_sim(state.reshape(1, 13), env_ids=env_ids)
        joint_pos = torch.as_tensor(pose["joint_pos"], device=self.device, dtype=torch.float32).reshape(1, 6)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = joint_pos.clone()
        self.fluid.reset(list(pose["cup"][:3]), list(pose["cup"][3:7]))
