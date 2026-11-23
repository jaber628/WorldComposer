from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import random

import omni.physx
import numpy as np
import torch
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .CutSausage_cfg import CutSausageEnvCfg
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.constant import ASSETS_ROOT
from WorldComposer.utils.cutMeshNode import cutMeshNode, generate_mesh, knife_intersects
from WorldComposer.utils.success_checker import success_checker_cut
from WorldComposer.utils.rendering import apply_default_render_settings, set_tone_mapping_fstop


class CutSausageEnv(DirectRLEnv):
    cfg: CutSausageEnvCfg
    sausage_path = "/World/CutSausage/sausage"
    knife_path = "/World/Robot/Robot/gripper/Knife/Knife/Knife002"

    def __init__(self, cfg: CutSausageEnvCfg, render_mode: str | None = None, **kwargs):
        self._is_closed = True
        if cfg.sim.device != "cpu" or cfg.sim.use_fabric:
            raise ValueError("CutSausage requires --device cpu and use_fabric=False.")
        if cfg.scene.num_envs != 1:
            raise ValueError("CutSausage requires num_envs=1.")
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
        self._last_collision = False

    def _setup_scene(self):
        apply_default_render_settings()
        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func("/World/Scene", scene_cfg, orientation=(0.0, 0.0, 0.0, 1.0))
        self._resolve_table_texture_paths()
        self.robot = Articulation(self.cfg.robot)
        knife_root = self.scene.stage.GetPrimAtPath("/World/Robot/Robot/gripper/Knife")
        for prim in Usd.PrimRange(knife_root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.IsA(UsdPhysics.FixedJoint):
                prim.SetActive(False)
        self.board = RigidObject(self.cfg.board)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["board"] = self.board
        self._spawn_sausage()
        self.enable_cameras = self.task_config["teleop"].get("enable_cameras", False)
        if self.enable_cameras:
            self.top_camera = TiledCamera(self.cfg.top_camera)
            self.scene.sensors["top_camera"] = self.top_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)
        self.sim.physics_sim_view.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True, False)
        stage = self.scene.stage
        knife = stage.GetPrimAtPath(self.knife_path)
        if self._last_collision:
            bounds = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide], False, True,
            )
            knife_low = bounds.ComputeWorldBound(knife).ComputeAlignedRange().GetMin()[2]
            food_high = max(bounds.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2]
                            for prim in self._sausage_meshes())
            if knife_low > food_high + 0.01:
                self._last_collision = False
        elif knife_intersects(knife, self._sausage_meshes()):
            self._last_collision = bool(cutMeshNode.compute(
                stage, self._sausage_mesh_path, self.knife_path,
            ))

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
        success = success_checker_cut(len(self._sausage_meshes()), **self.task_config["success"])
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
        if not replay_mode and pos_cfg.get("enable", False):
            global_xy.uniform_(*pos_cfg.get("global_xy_range", [-0.1, 0.1]))
            source_xy = global_xy + torch.empty_like(global_xy).uniform_(*pos_cfg.get("relative_xy_range", [-0.02, 0.02]))
        for obj, offset in ((self.robot, global_xy), (self.board, source_xy)):
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :2] += offset
            obj.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        self.scene.stage.RemovePrim(self.sausage_path)
        pos = np.array(self.cfg.sausage_pos)
        pos[:2] += source_xy[0].cpu().numpy()
        if not replay_mode and pos_cfg.get("enable", False):
            local_xy = np.array([random.uniform(*pos_cfg.get("sausage_xy_range", [-0.01, 0.01])) for _ in range(2)])
            board_rotation = Gf.Quatd(self.cfg.board.init_state.rot[0], Gf.Vec3d(*self.cfg.board.init_state.rot[1:])).GetNormalized()
            pos += np.array(board_rotation.Transform(Gf.Vec3d(*local_xy, 0.0)))
        instance_cfg = rand_cfg.get("object_instance", {})
        model_index = random.randrange(len(self._sausage_models())) if not replay_mode and instance_cfg.get("enable", False) else 0
        rot_cfg, scale_cfg = rand_cfg.get("rotation", {}), rand_cfg.get("scale", {})
        yaw = random.uniform(*rot_cfg.get("yaw_range", [-15.0, 15.0])) if not replay_mode and rot_cfg.get("enable", False) else 0.0
        scale = random.uniform(*scale_cfg.get("sausage_range", [0.9, 1.1])) if not replay_mode and scale_cfg.get("enable", False) else 1.0
        self._spawn_sausage(tuple(pos), model_index, yaw, scale)
        self._last_collision = False
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

    def _sausage_models(self):
        return self.task_config["randomization"]["object_instance"]["sausages"]

    def _spawn_sausage(self, position=None, model_index=0, yaw=0.0, scale=1.0):
        model = self._sausage_models()[model_index]
        rotation = Gf.Quatd(model["rotation"][0], Gf.Vec3d(*model["rotation"][1:])).GetNormalized()
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat() * rotation
        cfg = sim_utils.UsdFileCfg(usd_path=str(Path(ASSETS_ROOT).parent / model["path"]), scale=(scale,) * 3)
        cfg.func(self.sausage_path, cfg, translation=position or self.cfg.sausage_pos,
                 orientation=(rotation.GetReal(), *rotation.GetImaginary()))
        self._sausage_mesh_path = self.sausage_path + "/" + model["mesh_path"]
        self._sausage_model_index, self._sausage_scale, self._sausage_yaw = model_index, scale, yaw
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False, True)
        board_top = bounds.ComputeWorldBound(self.scene.stage.GetPrimAtPath(self.cfg.board.prim_path)).ComputeAlignedRange().GetMax()[2]
        root = self.scene.stage.GetPrimAtPath(self.sausage_path)
        food_low = bounds.ComputeWorldBound(root).ComputeAlignedRange().GetMin()[2]
        translate = root.GetAttribute("xformOp:translate")
        pos = Gf.Vec3d(translate.Get())
        pos[2] += board_top + 0.001 - food_low
        translate.Set(pos)
        for prim in self._sausage_meshes():
            mass = UsdPhysics.MassAPI(prim).GetMassAttr()
            mass.Set(mass.Get() * scale ** 3)

    def _sausage_meshes(self):
        prim = self.scene.stage.GetPrimAtPath(self._sausage_mesh_path)
        return [child for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh)]

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
                raise ValueError("CutSausage texture or shader is missing.")
            texture_input.Set(Sdf.AssetPath(str(texture)))
        self._texture_ids = dict(indices)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def get_all_pose(self):
        poses = {"board": self.board.data.root_state_w.cpu().numpy().copy(),
                 "robot": self.robot.data.root_state_w.cpu().numpy().copy(),
                 "joint_pos": self.robot.data.joint_pos.cpu().numpy().copy()}
        root = self.scene.stage.GetPrimAtPath(self.sausage_path)
        poses["sausage_model"] = self._sausage_models()[self._sausage_model_index]["path"]
        poses["sausage_scale"] = self._sausage_scale
        poses["sausage_yaw"] = self._sausage_yaw
        poses["sausage_transform"] = np.array(UsdGeom.Xformable(root).GetLocalTransformation())
        pieces = []
        for prim in self._sausage_meshes():
            mesh = UsdGeom.Mesh(prim)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            body = UsdPhysics.RigidBodyAPI(prim)
            pieces.append({"name": prim.GetName(), "vertices": np.array(mesh.GetPointsAttr().Get()),
                           "faces": np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                           "transform": np.array(mesh.GetLocalTransformation()), "material": str(material.GetPath()),
                           "mass": UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
                           "velocity": np.array(body.GetVelocityAttr().Get()),
                           "angular_velocity": np.array(body.GetAngularVelocityAttr().Get())})
            uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            if uv and uv.HasValue():
                pieces[-1]["uv"] = np.array(uv.ComputeFlattened())
            cap = prim.GetChild("cut_surface")
            if cap:
                cap_material, _ = UsdShade.MaterialBindingAPI(cap).ComputeBoundMaterial()
                pieces[-1]["cut_faces"] = np.array(UsdGeom.Subset(cap).GetIndicesAttr().Get())
                pieces[-1]["cut_material"] = str(cap_material.GetPath())
        poses["sausage_meshes"] = pieces
        if hasattr(self, "_texture_ids"):
            poses["texture_ids"] = dict(self._texture_ids)
        if hasattr(self, "_fstop"):
            poses["fstop"] = self._fstop
        return poses

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        for name in ("board", "robot"):
            state = torch.as_tensor(pose[name], device=self.device, dtype=torch.float32).reshape(1, 13)
            getattr(self, name).write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        joint_pos = torch.as_tensor(pose["joint_pos"], device=self.device, dtype=torch.float32).reshape(1, 6)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.actions = joint_pos.clone()
        self.scene.stage.RemovePrim(self.sausage_path)
        model_index = next((i for i, model in enumerate(self._sausage_models()) if model["path"] == pose.get("sausage_model")), 0)
        self._spawn_sausage(model_index=model_index, scale=pose.get("sausage_scale", 1.0), yaw=pose.get("sausage_yaw", 0.0))
        root = UsdGeom.Xformable(self.scene.stage.GetPrimAtPath(self.sausage_path))
        root.ClearXformOpOrder()
        root.AddTransformOp().Set(Gf.Matrix4d(pose["sausage_transform"].tolist() if isinstance(pose["sausage_transform"], np.ndarray) else pose["sausage_transform"]))
        for prim in self._sausage_meshes():
            prim.SetActive(False)
        for piece in pose["sausage_meshes"]:
            path = self._sausage_mesh_path + "/" + piece["name"]
            self.scene.stage.RemovePrim(path)
            prim = generate_mesh(self.scene.stage, path, piece["vertices"], piece["faces"],
                                 piece["transform"], piece["material"], piece["mass"],
                                 piece["velocity"], piece["angular_velocity"],
                                 piece.get("uv"), piece.get("cut_faces"), piece.get("cut_material"))
            prim.SetActive(True)
        self._last_collision = False
        if "texture_ids" in pose:
            self._set_textures(pose["texture_ids"])
        if "fstop" in pose:
            self._fstop = float(pose["fstop"])
            set_tone_mapping_fstop(fstop=self._fstop, enabled=True)
