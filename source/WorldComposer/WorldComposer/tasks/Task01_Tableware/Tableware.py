from __future__ import annotations
import torch
from typing import Sequence
import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera

from .Tableware_cfg import TablewareEnvCfg
from WorldComposer.utils.success_checker import success_checker_bowlinplate
from WorldComposer.utils.rendering import apply_default_render_settings, setup_default_lighting


class TablewareEnv(DirectRLEnv):
    cfg: TablewareEnvCfg

    def __init__(self, cfg: TablewareEnvCfg, render_mode: str | None = None, **kwargs):
        self.task_config = getattr(cfg, "task_config", {}) or {}
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        apply_default_render_settings(task_name="tableware")

        self.robot = Articulation(self.cfg.robot)

        teleop_cfg = self.task_config.get("teleop", {})
        self.enable_cameras = teleop_cfg.get("enable_cameras", False)
        
        if self.enable_cameras:
            self.top_camera = TiledCamera(self.cfg.top_camera)
            self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func(
            "/World/Scene",
            scene_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self._object_A_default_pos = list(self.cfg.object_A.init_state.pos)
        self._object_B_default_pos = list(self.cfg.object_B.init_state.pos)
        self._object_A_default_rot = list(self.cfg.object_A.init_state.rot)
        self._object_B_default_rot = list(self.cfg.object_B.init_state.rot)

        pool_a_usds = self.task_config.get("_resolved_object_A_usds", [])
        pool_b_usds = self.task_config.get("_resolved_object_B_usds", [])

        if not pool_a_usds:
            pool_a_usds = [self.cfg.object_A.spawn.usd_path]
        if not pool_b_usds:
            pool_b_usds = [self.cfg.object_B.spawn.usd_path]

        hidden_pos = (0.0, 0.0, -10.0)
        hidden_rot = (1.0, 0.0, 0.0, 0.0)

        self.object_A_pool: list[RigidObject] = []
        self.object_B_pool: list[RigidObject] = []

        for i, usd_path in enumerate(pool_a_usds):
            cfg = RigidObjectCfg(
                prim_path=f"/World/Object/object_A_{i}",
                spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
                init_state=RigidObjectCfg.InitialStateCfg(pos=hidden_pos, rot=hidden_rot),
            )
            obj = RigidObject(cfg)
            self.scene.rigid_objects[f"object_A_{i}"] = obj
            self.object_A_pool.append(obj)

        for i, usd_path in enumerate(pool_b_usds):
            cfg = RigidObjectCfg(
                prim_path=f"/World/Object/object_B_{i}",
                spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
                init_state=RigidObjectCfg.InitialStateCfg(pos=hidden_pos, rot=hidden_rot),
            )
            obj = RigidObject(cfg)
            self.scene.rigid_objects[f"object_B_{i}"] = obj
            self.object_B_pool.append(obj)

        self.active_A_idx = 0
        self.active_B_idx = 0
        self.object_A = self.object_A_pool[0]
        self.object_B = self.object_B_pool[0]

        self.scene.articulations["robot"] = self.robot
        if self.enable_cameras:
            self.scene.sensors["top_camera"] = self.top_camera
            self.scene.sensors["wrist_camera"] = self.wrist_camera

        setup_default_lighting()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
        }
        
        if self.enable_cameras:
            teleop_cfg = self.task_config.get("teleop", {})
            disable_depth = teleop_cfg.get("disable_depth", True)
            
            top_camera_rgb = self.top_camera.data.output["rgb"]
            wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
            
            observations["observation.images.top_rgb"] = top_camera_rgb.cpu().detach().numpy().squeeze()
            observations["observation.images.wrist_rgb"] = wrist_camera_rgb.cpu().detach().numpy().squeeze()
            
            if not disable_depth:
                top_camera_depth = self.top_camera.data.output["depth"].squeeze()
                wrist_camera_depth = self.wrist_camera.data.output["depth"].squeeze()
                observations["observation.images.top_depth"] = top_camera_depth.cpu().detach().numpy()
                observations["observation.images.wrist_depth"] = wrist_camera_depth.cpu().detach().numpy()

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = success_checker_bowlinplate(self.object_A, self.object_B)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES   
        super()._reset_idx(env_ids)

        # Replay mode: disable all reset-time randomization for strict replay.
        replay_mode = getattr(self, "replay_mode", False)
        rand_cfg = self.task_config.get("randomization", {})

        # ==========================================
        # 1. Object Instance Switch (show/hide pool)
        # ==========================================
        obj_inst_cfg = rand_cfg.get("object_instance", {})
        if not replay_mode and obj_inst_cfg.get("enable", False):
            self._switch_active_objects(env_ids)

        # ==========================================
        # 2. Scale Randomization
        # ==========================================
        scale_cfg = rand_cfg.get("scale", {})
        if not replay_mode and scale_cfg.get("enable", False):
            self._randomize_object_scale(scale_cfg)

        # ==========================================
        # 3. Position Randomization
        # ==========================================
        pos_cfg = rand_cfg.get("position", {})
        enable_pos_rand = not replay_mode and pos_cfg.get("enable", False)
        
        # Global XY shift
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        global_xy = torch.zeros(len(env_ids), 2, device=robot_root_state.device)
        
        if enable_pos_rand:
            g_range = pos_cfg.get("global_xy_range", [-0.1, 0.1])
            global_xy[:, 0].uniform_(g_range[0], g_range[1])
            
        robot_root_state[..., :2] += global_xy
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        
        # Reset joints
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos)
        
        self.robot_reset_state = np.array(robot_root_state.cpu().detach(), dtype=np.float32)

        # Object A & B positions: use stored default positions (not obj.data.default_root_state which is z=-10)
        dev = robot_root_state.device
        n = len(env_ids)

        object_A_pos = torch.zeros(n, 13, device=dev)
        object_A_pos[..., 0] = self._object_A_default_pos[0]
        object_A_pos[..., 1] = self._object_A_default_pos[1]
        object_A_pos[..., 2] = self._object_A_default_pos[2]
        object_A_pos[..., 3] = self._object_A_default_rot[0]
        object_A_pos[..., 4] = self._object_A_default_rot[1]
        object_A_pos[..., 5] = self._object_A_default_rot[2]
        object_A_pos[..., 6] = self._object_A_default_rot[3]

        object_B_pos = torch.zeros(n, 13, device=dev)
        object_B_pos[..., 0] = self._object_B_default_pos[0]
        object_B_pos[..., 1] = self._object_B_default_pos[1]
        object_B_pos[..., 2] = self._object_B_default_pos[2]
        object_B_pos[..., 3] = self._object_B_default_rot[0]
        object_B_pos[..., 4] = self._object_B_default_rot[1]
        object_B_pos[..., 5] = self._object_B_default_rot[2]
        object_B_pos[..., 6] = self._object_B_default_rot[3]

        object_A_pos[..., :2] += global_xy
        object_B_pos[..., :2] += global_xy

        # Relative XY shift
        if enable_pos_rand:
            r_range = pos_cfg.get("relative_xy_range", [-0.02, 0.02])
            rand_object_A = torch.empty(len(env_ids), 2, device=object_A_pos.device).uniform_(r_range[0], r_range[1])
            rand_object_B = torch.empty(len(env_ids), 2, device=object_B_pos.device).uniform_(r_range[0], r_range[1])
            object_A_pos[..., :2] += rand_object_A
            object_B_pos[..., :2] += rand_object_B

        self.object_A.write_root_state_to_sim(object_A_pos, env_ids=env_ids)
        self.object_B.write_root_state_to_sim(object_B_pos, env_ids=env_ids)

        self.object_A_reset_state = np.array(object_A_pos.cpu().detach(), dtype=np.float32)
        self.object_B_reset_state = np.array(object_B_pos.cpu().detach(), dtype=np.float32)

        # ==========================================
        # 3. Texture Randomization
        # ==========================================
        tex_cfg = rand_cfg.get("texture", {})
        if not replay_mode and tex_cfg.get("enable", False):
            self._randomize_table038_texture(tex_cfg)

        # ==========================================
        # 4. Lighting Randomization (Tone Mapping)
        # ==========================================
        light_cfg = rand_cfg.get("lighting", {})
        if not replay_mode and light_cfg.get("enable", False):
            self._randomize_lighting(light_cfg)

    def _switch_active_objects(self, env_ids):
        """Randomly pick one object from each pool; hide all others underground."""
        HIDDEN = torch.tensor([[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                              device=self.device)

        new_A = random.randint(0, len(self.object_A_pool) - 1)
        new_B = random.randint(0, len(self.object_B_pool) - 1)

        for i, obj in enumerate(self.object_A_pool):
            if i != new_A:
                obj.write_root_state_to_sim(HIDDEN.clone(), env_ids=env_ids)

        for i, obj in enumerate(self.object_B_pool):
            if i != new_B:
                obj.write_root_state_to_sim(HIDDEN.clone(), env_ids=env_ids)

        self.active_A_idx = new_A
        self.active_B_idx = new_B
        self.object_A = self.object_A_pool[new_A]
        self.object_B = self.object_B_pool[new_B]

    def _randomize_object_scale(self, cfg: dict):
        """Randomize scale of object A and B using direct XformOp (avoids XformCommonAPI incompatibility)."""
        stage = self.scene.stage
        scale_cfgs = {
            f"/World/Object/object_A_{self.active_A_idx}": cfg.get("object_A_range", cfg.get("range", [1.0, 1.2])),
            f"/World/Object/object_B_{self.active_B_idx}": cfg.get("object_B_range", cfg.get("range", [1.0, 1.2])),
        }

        for prim_path, scale_range in scale_cfgs.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            scale_val = random.uniform(scale_range[0], scale_range[1])
            xform = UsdGeom.Xformable(prim)
            # Reuse existing scale op if present, otherwise add one
            scale_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            try:
                if scale_op is None:
                    scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
                scale_op.Set(Gf.Vec3d(scale_val, scale_val, scale_val))
            except Exception:
                pass

    def _randomize_lighting(self, cfg: dict):
        """Randomize tone mapping f-stop to simulate lighting changes"""
        tone_range = cfg.get("tone_range", [4.8, 7.8])
        fstop = random.uniform(tone_range[0], tone_range[1])
        set_tone_mapping_fstop(fstop=fstop, enabled=True)

    def _randomize_table038_texture(self, cfg: dict):
        """Randomize Table038 texture based on config."""
        folder = cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(cfg.get("min_id", 1))
        max_id = int(cfg.get("max_id", 1))
        shader_path = cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            return
        if not shader_path:
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            return

        tex_input.Set(Sdf.AssetPath(tex_path))

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        robot_state = self.robot.data.default_root_state
        self.robot_reset_state = np.array(robot_state.cpu().detach(), dtype=np.float32)
        object_A_state = self.object_A.data.default_root_state
        self.object_A_reset_state = np.array(object_A_state.cpu().detach(), dtype=np.float32)
        object_B_state = self.object_B.data.default_root_state
        self.object_B_reset_state = np.array(object_B_state.cpu().detach(), dtype=np.float32)
        
    def get_all_pose(self):
        return {
            "robot": self.robot_reset_state,
            "object_A": self.object_A_reset_state,
            "object_B": self.object_B_reset_state,
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        pose_tensor = torch.tensor(pose["robot"], dtype=torch.float32, device=self.device)
        self.robot.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
        pose_tensor = torch.tensor(pose["object_A"], dtype=torch.float32, device=self.device)
        self.object_A.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
        pose_tensor = torch.tensor(pose["object_B"], dtype=torch.float32, device=self.device)
        self.object_B.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
