from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import random

import torch
import yaml
from pxr import Sdf, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .AssembleBurger_cfg import AssembleBurgerEnvCfg
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.constant import ASSETS_ROOT
from WorldComposer.utils.success_checker import success_checker_burger
from WorldComposer.utils.rendering import apply_default_render_settings, set_tone_mapping_fstop


class AssembleBurgerEnv(DirectRLEnv):
    cfg: AssembleBurgerEnvCfg

    def __init__(self, cfg: AssembleBurgerEnvCfg, render_mode: str | None = None, **kwargs):
        self._is_closed = True
        if not cfg.sim.device.startswith("cuda"):
            raise ValueError("AssembleBurger requires GPU dynamics; use --device cuda:0.")
        if cfg.scene.num_envs != 1:
            raise ValueError("AssembleBurger requires num_envs=1.")
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
        scene_cfg.func("/World/Scene", scene_cfg, orientation=(0.0, 0.0, 0.0, 1.0))
        self._resolve_table_texture_paths()

        self.robot = Articulation(self.cfg.robot)
        self.burger_beef = DeformableObject(self.cfg.burger_beef)
        self.burger_cheese = DeformableObject(self.cfg.burger_cheese)
        self.burger_board = RigidObject(self.cfg.burger_board)
        self.burger_plate = RigidObject(self.cfg.burger_plate)
        self.burger_bread_Top = RigidObject(self.cfg.burger_bread_Top)
        self.scene.articulations["robot"] = self.robot
        for name in ("burger_beef", "burger_cheese"):
            self.scene.deformable_objects[name] = getattr(self, name)
        for name in ("burger_board", "burger_plate", "burger_bread_Top"):
            self.scene.rigid_objects[name] = getattr(self, name)

        self.enable_cameras = self.task_config["teleop"].get("enable_cameras", False)
        if self.enable_cameras:
            self.top_camera = TiledCamera(self.cfg.top_camera)
            self.scene.sensors["top_camera"] = self.top_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        observations = {
            "action": self.actions[0].detach().cpu().numpy(),
            "observation.state": self.robot.data.joint_pos[0].detach().cpu().numpy(),
        }
        if self.enable_cameras:
            observations["observation.images.top_rgb"] = self.top_camera.data.output["rgb"][0].cpu().numpy()
            if not self.task_config["teleop"].get("disable_depth", True):
                observations["observation.images.top_depth"] = self.top_camera.data.output["depth"][0].cpu().numpy().squeeze(-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.episode_length_buf, dtype=torch.float)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> torch.Tensor:
        success = success_checker_burger(
            self.burger_beef, self.burger_bread_Top, **self.task_config["success"],
        )
        return torch.full_like(self.episode_length_buf, success, dtype=torch.bool)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        rand_cfg = self.task_config.get("randomization", {})
        replay_mode = getattr(self, "replay_mode", False)
        pos_cfg = rand_cfg.get("position", {})
        global_xy = torch.zeros(len(env_ids), 2, device=self.device)
        source_xy = global_xy.clone()
        target_xy = global_xy.clone()
        if not replay_mode and pos_cfg.get("enable", False):
            global_xy.uniform_(*pos_cfg.get("global_xy_range", [-0.1, 0.1]))
            source_xy = global_xy + torch.empty_like(global_xy).uniform_(*pos_cfg.get("relative_xy_range", [-0.02, 0.02]))
            target_xy = global_xy + torch.empty_like(global_xy).uniform_(*pos_cfg.get("relative_xy_range", [-0.02, 0.02]))
        for obj, offset in ((self.robot, global_xy), (self.burger_board, source_xy),
                            (self.burger_plate, target_xy), (self.burger_bread_Top, target_xy)):
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :2] += offset
            obj.write_root_state_to_sim(state, env_ids=env_ids)
        for obj, offset in ((self.burger_beef, source_xy), (self.burger_cheese, target_xy)):
            state = obj.data.default_nodal_state_w[env_ids].clone()
            state[:, :, :2] += offset[:, None, :]
            obj.write_nodal_state_to_sim(state, env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = self.robot.data.default_joint_pos.clone()
        tex_cfg = rand_cfg.get("texture", {})
        if not replay_mode and tex_cfg.get("enable", False):
            indices = {path: random.randint(tex_cfg.get("min_id", 0), tex_cfg.get("max_id", 100))
                       for path in (tex_cfg["prim_path"], tex_cfg["board_prim_path"])}
            self._set_textures(indices)
        light_cfg = rand_cfg.get("lighting", {})
        if not replay_mode and light_cfg.get("enable", False):
            self._fstop = random.uniform(*light_cfg.get("tone_range", [5.8, 8.8]))
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)

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

    def _set_textures(self, indices):
        folder = Path(self.task_config["randomization"]["texture"]["folder"])
        if not folder.is_absolute():
            folder = Path(ASSETS_ROOT).parent / folder
        for path, index in indices.items():
            shader = UsdShade.Shader(self.scene.stage.GetPrimAtPath(path))
            texture_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
            texture = folder / f"{index}.png"
            if not texture.is_file() or not texture_input:
                raise ValueError("AssembleBurger texture or shader is missing.")
            texture_input.Set(Sdf.AssetPath(str(texture)))
        self._texture_ids = dict(indices)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        poses = {name: getattr(self, name).data.root_state_w.detach().cpu().numpy().copy()
                 for name in ("burger_board", "burger_plate", "burger_bread_Top")}
        for name in ("burger_beef", "burger_cheese"):
            poses[name] = getattr(self, name).data.nodal_state_w.detach().cpu().numpy().copy()
        poses["robot"] = self.robot.data.root_state_w.detach().cpu().numpy().copy()
        poses["joint_pos"] = self.robot.data.joint_pos.detach().cpu().numpy().copy()
        if hasattr(self, "_texture_ids"):
            poses["texture_ids"] = dict(self._texture_ids)
        if hasattr(self, "_fstop"):
            poses["fstop"] = self._fstop
        return poses

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        for name in ("burger_board", "burger_plate", "burger_bread_Top"):
            if name in pose:
                state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32).reshape(1, 13)
                getattr(self, name).write_root_state_to_sim(state, env_ids=env_ids)
        for name in ("burger_beef", "burger_cheese"):
            if name in pose:
                state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32).reshape(1, -1, 6)
                getattr(self, name).write_nodal_state_to_sim(state, env_ids=env_ids)
        if "robot" in pose:
            state = torch.as_tensor(pose["robot"], device=self.device, dtype=torch.float32).reshape(1, 13)
            self.robot.write_root_state_to_sim(state, env_ids=env_ids)
        if "joint_pos" in pose:
            joint_pos = torch.as_tensor(pose["joint_pos"], device=self.device, dtype=torch.float32).reshape(1, 6)
            self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self.actions = joint_pos.clone()
        if "texture_ids" in pose:
            self._set_textures(pose["texture_ids"])
        if "fstop" in pose:
            self._fstop = float(pose["fstop"])
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)
