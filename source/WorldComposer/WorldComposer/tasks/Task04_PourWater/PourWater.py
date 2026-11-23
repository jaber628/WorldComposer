from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import random

import torch
import yaml
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .PourWater_cfg import PourWaterEnvCfg
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.constant import ASSETS_ROOT
from WorldComposer.utils.fluid import FluidSystem
from WorldComposer.utils.success_checker import success_checker_pour
from WorldComposer.utils.rendering import apply_default_render_settings, set_tone_mapping_fstop


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

        def merge_config(target, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge_config(target[key], value)
                else:
                    target[key] = value

        merge_config(self.task_config, cfg.task_config or {})
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
        self.scene.articulations["robot"] = self.robot
        self._create_object_pool("cup", self.cfg.cup)
        self._create_object_pool("bowl", self.cfg.bowl)

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
        self._fluid_base_positions = list(self.fluid.init_particle_positions)
        light = self.scene.stage.GetPrimAtPath(f"{self.fluid.prim_path}/env_light")
        if light:
            light.SetActive(False)
        self.fluid.particle_system.CreateSimulationOwnerRel().SetTargets([self.cfg.sim.physics_prim_path])
        cup_pos = list(self.cfg.cup.init_state.pos)
        cup_pos[2] = self.cup_variants[0].get("pos_z", cup_pos[2])
        self._reset_fluid(cup_pos, list(self.cfg.cup.init_state.rot))
        self.sim.get_physics_context().enable_ccd(False)
        self.sim.get_physics_context().enable_gpu_dynamics(True)
        self.sim.get_physics_context().set_broadphase_type("GPU")

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
        success_cfg = getattr(self, "_bowl_success", self.task_config["success"])
        success = success_checker_pour(self.fluid, self.bowl, **success_cfg)
        return torch.full_like(self.episode_length_buf, success, dtype=torch.bool)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        rand_cfg = self.task_config.get("randomization", {})
        replay_mode = getattr(self, "replay_mode", False)
        instance_cfg = rand_cfg.get("object_instance", {})
        pos_cfg = rand_cfg.get("position", {})
        global_xy = torch.zeros(len(env_ids), 2, device=self.device)
        randomize_position = not replay_mode and pos_cfg.get("enable", False)
        if randomize_position:
            global_xy.uniform_(*pos_cfg.get("global_xy_range", [-0.1, 0.1]))
        robot_state = self.robot.data.default_root_state[env_ids].clone()
        robot_state[:, :2] += global_xy
        self.robot.write_root_state_to_sim(robot_state, env_ids=env_ids)
        for name in ("cup", "bowl"):
            index = getattr(self, f"active_{name}_idx")
            if not replay_mode and instance_cfg.get("enable", False):
                index = random.randrange(len(getattr(self, f"{name}_pool")))
            self._select_object(name, index, env_ids)
            obj = getattr(self, name)
            variant = getattr(self, f"{name}_variants")[index]
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] = torch.tensor(getattr(self.cfg, name).init_state.pos, device=self.device)
            state[:, 2] = variant.get("pos_z", getattr(self.cfg, name).init_state.pos[2])
            state[:, 7:] = 0.0
            state[:, :2] += global_xy
            if randomize_position:
                state[:, :2] += torch.empty_like(global_xy).uniform_(*pos_cfg.get("relative_xy_range", [-0.02, 0.02]))
            obj.write_root_state_to_sim(state, env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = self.robot.data.default_joint_pos.clone()
        cup_state = self.cup.data.root_state_w[0]
        self._reset_fluid(cup_state[:3].cpu().tolist(), cup_state[3:7].cpu().tolist())
        tex_cfg = rand_cfg.get("texture", {})
        if not replay_mode and tex_cfg.get("enable", False):
            self._set_table_texture(random.randint(tex_cfg.get("min_id", 0), tex_cfg.get("max_id", 100)))
        light_cfg = rand_cfg.get("lighting", {})
        if not replay_mode and light_cfg.get("enable", False):
            self._fstop = random.uniform(*light_cfg.get("tone_range", [5.8, 8.8]))
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)

    def _create_object_pool(self, name, cfg):
        root = Path(ASSETS_ROOT).parent
        instance_cfg = self.task_config.get("randomization", {}).get("object_instance", {})
        variants = instance_cfg.get(f"{name}_variants") or [
            {"usd_path": str(Path(cfg.spawn.usd_path).relative_to(root))}
        ]
        pool = []
        for index, variant in enumerate(variants):
            path = Path(variant["usd_path"])
            if not path.is_absolute():
                path = root / path
            pos = list(cfg.init_state.pos)
            pos[2] = variant.get("pos_z", pos[2])
            if index:
                pos = (2.0 * index, 0.0 if name == "cup" else 2.0, -10.0)
            obj = RigidObject(cfg.replace(
                prim_path=cfg.prim_path if index == 0 else f"{cfg.prim_path}_{index}",
                spawn=cfg.spawn.replace(usd_path=str(path)),
                init_state=cfg.init_state.replace(pos=pos),
            ))
            pool.append(obj)
            self.scene.rigid_objects[f"{name}_{index}"] = obj
        setattr(self, f"{name}_pool", pool)
        setattr(self, f"{name}_variants", variants)
        setattr(self, f"active_{name}_idx", 0)
        setattr(self, name, pool[0])

    def _select_object(self, name, index, env_ids):
        for i, obj in enumerate(getattr(self, f"{name}_pool")):
            if i != index:
                hidden = obj.data.default_root_state[env_ids].clone()
                hidden[:, :3] = torch.tensor([2.0 * i, 0.0 if name == "cup" else 2.0, -10.0], device=self.device)
                hidden[:, 7:] = 0.0
                obj.write_root_state_to_sim(hidden, env_ids=env_ids)
        setattr(self, f"active_{name}_idx", index)
        setattr(self, name, getattr(self, f"{name}_pool")[index])
        if name == "bowl":
            self._bowl_success = {
                **self.task_config["success"], **self.bowl_variants[index].get("success", {})
            }

    def _reset_fluid(self, position, rotation):
        variant = self.cup_variants[self.active_cup_idx]
        offset = Gf.Vec3f(*variant.get("fluid_offset", self.task_config["fluid"].get("offset", (0.0, 0.0, 0.0))))
        self.fluid.init_particle_positions = [point + offset for point in self._fluid_base_positions]
        self.fluid.reset(position, rotation)

    def _set_table_texture(self, index):
        cfg = self.task_config["randomization"]["texture"]
        folder = Path(cfg["folder"])
        if not folder.is_absolute():
            folder = Path(ASSETS_ROOT).parent / folder
        texture = folder / f"{index}.png"
        shader = UsdShade.Shader(self.scene.stage.GetPrimAtPath(cfg["prim_path"]))
        texture_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not texture.is_file() or not texture_input:
            raise ValueError("PourWater table texture or shader is missing.")
        texture_input.Set(Sdf.AssetPath(str(texture)))
        self._texture_id = index

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

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        poses = {name: obj.data.root_state_w[0].detach().cpu().numpy().copy()
                 for name, obj in (("robot", self.robot), ("cup", self.cup), ("bowl", self.bowl))}
        poses["joint_pos"] = self.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        for name in ("cup", "bowl"):
            index = getattr(self, f"active_{name}_idx")
            poses[f"{name}_usd_path"] = getattr(self, f"{name}_variants")[index]["usd_path"]
        if hasattr(self, "_texture_id"):
            poses["texture_id"] = self._texture_id
        if hasattr(self, "_fstop"):
            poses["fstop"] = self._fstop
        return poses

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        for name in ("cup", "bowl"):
            paths = [v["usd_path"] for v in getattr(self, f"{name}_variants")]
            self._select_object(name, paths.index(pose.get(f"{name}_usd_path", paths[0])), env_ids)
        for name, obj in (("robot", self.robot), ("cup", self.cup), ("bowl", self.bowl)):
            state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32)
            obj.write_root_state_to_sim(state.reshape(1, 13), env_ids=env_ids)
        joint_pos = torch.as_tensor(pose["joint_pos"], device=self.device, dtype=torch.float32).reshape(1, 6)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = joint_pos.clone()
        self._reset_fluid(list(pose["cup"][:3]), list(pose["cup"][3:7]))
        if "texture_id" in pose:
            self._set_table_texture(int(pose["texture_id"]))
        if "fstop" in pose:
            self._fstop = float(pose["fstop"])
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)
