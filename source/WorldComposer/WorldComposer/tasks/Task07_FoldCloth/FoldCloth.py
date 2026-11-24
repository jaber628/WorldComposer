from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import random

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

from .FoldCloth_cfg import FoldClothEnvCfg
from WorldComposer.assets.object.Garment import GarmentObject
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.constant import ASSETS_ROOT
from WorldComposer.utils.rendering import apply_default_render_settings, set_tone_mapping_fstop
from WorldComposer.utils.success_checker import success_checker_fold


class FoldClothEnv(DirectRLEnv):
    cfg: FoldClothEnvCfg

    def __init__(self, cfg: FoldClothEnvCfg, render_mode: str | None = None, **kwargs):
        self._is_closed = True
        if cfg.scene.num_envs != 1 or not cfg.sim.device.startswith("cuda"):
            raise ValueError("FoldCloth requires num_envs=1 and a CUDA device.")
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
        self.actions = torch.cat((self.left_arm.data.default_joint_pos, self.right_arm.data.default_joint_pos), dim=1)
        self.object._cloth_prim_view.initialize(self.sim.physics_sim_view)
        self._rest_positions = self.object._cloth_prim_view.get_world_positions().clone()
        targets = torch.as_tensor(self._landmark_targets, device=self.device, dtype=self._rest_positions.dtype)
        self._landmarks = torch.cdist(targets, self._rest_positions[0]).argmin(dim=1).tolist()
        selected = self._rest_positions[0, self._landmarks]
        distances = torch.stack([(selected[a] - selected[b]).norm() for a, b in ((0, 4), (2, 3), (1, 5))])
        ratios = torch.as_tensor(self.task_config["success"]["distance_ratio"], device=self.device)
        self._distance_thresholds = (distances * ratios).cpu().tolist()
        self._reset_cloth((0.0, 0.0), 0.0)
        gravity_scale = self._gravity_scale
        self.object.particle_material.set_gravity_scale(gravity_scale)
        self.object.config.objects.particle_material.gravity_scale = gravity_scale

    def _setup_scene(self):
        apply_default_render_settings()
        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func("/World/Scene", scene_cfg, orientation=(0.0, 0.0, 0.0, 1.0))
        self._resolve_table_texture_paths()
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self._garment_models = self.task_config["randomization"]["object_instance"]["garments"]
        instance_cfg = self.task_config["randomization"]["object_instance"]
        self._garment_index = self.cfg.garment_index if self.cfg.garment_index is not None else instance_cfg.get("garment_index")
        if self._garment_index is None:
            self._garment_index = random.randrange(len(self._garment_models)) if instance_cfg.get("enable", False) else 0
        model = self._garment_models[self._garment_index]
        cloth_cfg = OmegaConf.merge(self.task_config["cloth"], model.get("cloth", {}))
        config = OmegaConf.create({"objects": cloth_cfg})
        self._gravity_scale = float(config.objects.particle_material.gravity_scale)
        config.objects.particle_material.gravity_scale = 0.0
        config.objects.common.scale = [model["scale"]] * 3
        config.objects.common.initial_pos_range = model["position"] * 2
        config.objects.common.soft_reset_pos_range = model["position"] * 2
        self.object = GarmentObject(
            prim_path="/World/Object/Cloth",
            usd_path=str(Path(ASSETS_ROOT).parent / model["path"]), visual_usd_path=None, config=config,
        )
        UsdGeom.Xformable(self.scene.stage.GetPrimAtPath(self.object.usd_prim_path)).ClearXformOpOrder()
        self.object.set_local_pose(translation=np.asarray(model["position"]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        for index, path in enumerate(model.get("materials", [])):
            material_path = f"{self.object.usd_prim_path}/visual_material_{index}"
            prim = self.scene.stage.DefinePrim(material_path)
            prim.GetReferences().AddReference(str(Path(ASSETS_ROOT).parent / path))
            material = next(UsdShade.Material(child) for child in Usd.PrimRange(prim) if child.IsA(UsdShade.Material))
            mesh_prim = self.scene.stage.GetPrimAtPath(self.object.mesh_prim_path)
            targets = [mesh_prim] if index == 0 else []
            name = "mesh" if index == 0 else f"mesh{index}"
            targets.extend(child for child in mesh_prim.GetChildren() if child.GetName() == name)
            for target in targets:
                UsdShade.MaterialBindingAPI.Apply(target).Bind(material)
        for prim in Usd.PrimRange(self.scene.stage.GetPrimAtPath(self.object.usd_prim_path)):
            if prim.IsA(UsdShade.Shader) and prim.GetName() == "normalTex":
                shader = UsdShade.Shader(prim)
                shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                shader.GetInput("scale").Set(Gf.Vec4f(2, 2, 2, 1))
                shader.GetInput("bias").Set(Gf.Vec4f(-1, -1, -1, 0))
        mesh = UsdGeom.Mesh(self.scene.stage.GetPrimAtPath(self.object.mesh_prim_path))
        points = np.asarray(mesh.GetPointsAttr().Get())[model["keypoints"]]
        transform = np.array(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        self._landmark_targets = points @ transform[:3, :3] + transform[3, :3]
        self.enable_cameras = self.task_config["teleop"].get("enable_cameras", False)
        if self.enable_cameras:
            for name in ("top_camera", "left_wrist", "right_wrist"):
                camera = TiledCamera(getattr(self.cfg, name))
                setattr(self, name, camera)
                self.scene.sensors[name] = camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])

    def _get_observations(self) -> dict:
        observations = {
            "action": self.actions[0].detach().cpu().numpy(),
            "observation.state": torch.cat((self.left_arm.data.joint_pos, self.right_arm.data.joint_pos), dim=1)[0].detach().cpu().numpy(),
        }
        if self.enable_cameras:
            for name, camera in (("top", self.top_camera), ("left", self.left_wrist), ("right", self.right_wrist)):
                observations[f"observation.images.{name}_rgb"] = camera.data.output["rgb"][0].cpu().numpy()
            if not self.task_config["teleop"].get("disable_depth", True):
                observations["observation.images.top_depth"] = self.top_camera.data.output["depth"][0].cpu().numpy().squeeze(-1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.episode_length_buf, dtype=torch.float)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> torch.Tensor:
        success = success_checker_fold(
            self.object, index_list=self._landmarks,
            distance_thresholds=self._distance_thresholds,
            max_speed=self.task_config["success"]["max_speed"],
        )
        return torch.full_like(self.episode_length_buf, success, dtype=torch.bool)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)
        rand_cfg = self.task_config["randomization"]
        replay_mode = getattr(self, "replay_mode", False)
        pos_cfg = rand_cfg["position"]
        global_xy = torch.zeros(len(env_ids), 2, device=self.device)
        cloth_xy = global_xy.clone()
        if not replay_mode and pos_cfg.get("enable", False):
            global_xy.uniform_(*pos_cfg["global_xy_range"])
            cloth_xy = global_xy + torch.empty_like(global_xy).uniform_(*pos_cfg["relative_xy_range"])
        for arm in (self.left_arm, self.right_arm):
            state = arm.data.default_root_state[env_ids].clone()
            state[:, :2] += global_xy
            arm.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            joints = arm.data.default_joint_pos[env_ids].clone()
            arm.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
            arm.set_joint_position_target(joints, env_ids=env_ids)
        self.actions = torch.cat((self.left_arm.data.default_joint_pos, self.right_arm.data.default_joint_pos), dim=1)
        rot_cfg = rand_cfg["rotation"]
        yaw = random.uniform(*rot_cfg["yaw_range"]) if not replay_mode and rot_cfg.get("enable", False) else 0.0
        drop_cfg = rand_cfg.get("drop", {})
        roll = pitch = height = 0.0
        if not replay_mode and drop_cfg.get("enable", False):
            roll, pitch = [random.uniform(*drop_cfg["tilt_range"]) for _ in range(2)]
            height = random.uniform(*drop_cfg["height_range"])
        self._reset_cloth(cloth_xy[0].cpu().tolist(), yaw, roll, pitch, height)
        tex_cfg = rand_cfg["texture"]
        if not replay_mode and tex_cfg.get("enable", False):
            self._set_texture(random.randint(tex_cfg["min_id"], tex_cfg["max_id"]))
        light_cfg = rand_cfg["lighting"]
        if not replay_mode and light_cfg.get("enable", False):
            self._fstop = random.uniform(*light_cfg["tone_range"])
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)

    def _reset_cloth(self, offset_xy, yaw, roll=0.0, pitch=0.0, height=0.0):
        self._offset_xy, self._yaw = list(offset_xy), float(yaw)
        self._tilt, self._drop_height = [float(roll), float(pitch)], float(height)
        points = self._rest_positions
        center = (points.amin(dim=1) + points.amax(dim=1)) / 2
        angles = torch.as_tensor(np.deg2rad([roll, pitch, yaw]), device=self.device, dtype=points.dtype)
        rotation = matrix_from_quat(quat_from_euler_xyz(*angles))
        positions = (points - center[:, None, :]) @ rotation.T + center[:, None, :]
        positions[:, :, :2] += torch.as_tensor(offset_xy, device=self.device)
        lift = (points[:, :, 2].amin(dim=1) - positions[:, :, 2].amin(dim=1)).clamp_min(0)
        positions[:, :, 2] += lift[:, None] + height
        view = self.object._cloth_prim_view
        view.set_world_positions(positions)
        view.set_velocities(torch.zeros_like(positions))

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

    def _set_texture(self, index):
        cfg = self.task_config["randomization"]["texture"]
        folder = Path(ASSETS_ROOT).parent / cfg["folder"]
        shader = UsdShade.Shader(self.scene.stage.GetPrimAtPath(cfg["prim_path"]))
        texture_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        texture = folder / f"{index}.png"
        if not texture.is_file() or not texture_input:
            raise ValueError("FoldCloth texture or shader is missing.")
        texture_input.Set(Sdf.AssetPath(str(texture)))
        self._texture_id = index

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        pose = {name: arm.data.root_state_w.cpu().numpy().copy()
                for name, arm in (("left_robot", self.left_arm), ("right_robot", self.right_arm))}
        pose.update({"joint_pos": torch.cat((self.left_arm.data.joint_pos, self.right_arm.data.joint_pos), dim=1).cpu().numpy().copy(),
                     "garment": self._garment_models[self._garment_index]["path"], "garment_index": self._garment_index,
                     "particle_positions": self.object._cloth_prim_view.get_world_positions().cpu().numpy(),
                     "particle_velocities": self.object._cloth_prim_view.get_velocities().cpu().numpy(),
                     "offset_xy": list(self._offset_xy), "yaw": self._yaw,
                     "tilt": list(self._tilt), "drop_height": self._drop_height})
        if hasattr(self, "_texture_id"):
            pose["texture_id"] = self._texture_id
        if hasattr(self, "_fstop"):
            pose["fstop"] = self._fstop
        return pose

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        for name, arm in (("left_robot", self.left_arm), ("right_robot", self.right_arm)):
            state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32).reshape(1, 13)
            arm.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        joints = torch.as_tensor(pose["joint_pos"], device=self.device, dtype=torch.float32).reshape(1, 12)
        for arm, values in ((self.left_arm, joints[:, :6]), (self.right_arm, joints[:, 6:])):
            arm.write_joint_state_to_sim(values, torch.zeros_like(values), env_ids=env_ids)
            arm.set_joint_position_target(values, env_ids=env_ids)
        self.actions = joints.clone()
        if pose["garment"] != self._garment_models[self._garment_index]["path"]:
            raise ValueError("Create FoldCloth with the recorded garment_index before restoring its state.")
        self._reset_cloth(pose["offset_xy"], pose["yaw"], *pose.get("tilt", [0.0, 0.0]), pose.get("drop_height", 0.0))
        self.object._cloth_prim_view.set_world_positions(torch.as_tensor(pose["particle_positions"], device=self.device, dtype=torch.float32))
        self.object._cloth_prim_view.set_velocities(torch.as_tensor(pose["particle_velocities"], device=self.device, dtype=torch.float32))
        if "texture_id" in pose:
            self._set_texture(pose["texture_id"])
        if "fstop" in pose:
            self._fstop = float(pose["fstop"])
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)
