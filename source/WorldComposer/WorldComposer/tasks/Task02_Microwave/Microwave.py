from __future__ import annotations
import torch
import random
import os
import numpy as np
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from pxr import Usd, UsdShade, Sdf, UsdGeom, UsdPhysics

from .Microwave_cfg import MicrowaveEnvCfg
from WorldComposer.devices.action_process import preprocess_device_action
from WorldComposer.utils.rendering import apply_default_render_settings, setup_default_lighting, set_tone_mapping_fstop


class MicrowaveEnv(DirectRLEnv):
    cfg: MicrowaveEnvCfg

    def __init__(self, cfg: MicrowaveEnvCfg, render_mode: str | None = None, **kwargs):
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
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Build microwave pool
        pool_usds = self.task_config.get("_resolved_microwave_usds", [])
        if not pool_usds:
            pool_usds = [self.cfg.microwave.spawn.usd_path]

        self.microwave_pool: list[Articulation] = []
        for i, usd_path in enumerate(pool_usds):
            spawn_cfg = self.cfg.microwave.spawn.replace(usd_path=usd_path)
            cfg = self.cfg.microwave.replace(prim_path=f"/World/Object/microwave_{i}", spawn=spawn_cfg)
            mw = Articulation(cfg)
            self.scene.articulations[f"microwave_{i}"] = mw
            self.microwave_pool.append(mw)

        # Initially select index 0
        self.active_mw_idx = 0
        self.microwave = self.microwave_pool[0]

        self.scene.articulations["robot"] = self.robot
        if self.enable_cameras:
            self.scene.sensors["top_camera"] = self.top_camera
            self.scene.sensors["wrist_camera"] = self.wrist_camera

        setup_default_lighting()

        # Calculate Z offsets for all microwaves using BBoxCache
        self.microwave_z_offsets = []
        stage = self.scene.stage
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for mw in self.microwave_pool:
            prim = stage.GetPrimAtPath(mw.cfg.prim_path)
            if prim.IsValid():
                bbox = bbox_cache.ComputeWorldBound(prim)
                z_min = bbox.ComputeAlignedRange().GetMin()[2]
                # Spawned at z=-10, so offset to make bottom at 0 is -10 - z_min
                self.microwave_z_offsets.append(-10.0 - z_min)
            else:
                self.microwave_z_offsets.append(10.0) # Fallback

        # Apply damping to all microwaves
        for mw in self.microwave_pool:
            self._set_microwave_joint_damping(mw)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        if hasattr(self, 'actions'):
            action = self.actions.squeeze(0)
        else:
            action = torch.zeros(self.cfg.action_space, device=self.device)
            
        joint_pos = torch.cat([self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1)
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
        return torch.zeros_like(self.episode_length_buf, dtype=torch.float)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        joint_pos = self.microwave.data.joint_pos
        if joint_pos.numel() > 0 and joint_pos.shape[-1] > 0:
            success_tensor = torch.abs(joint_pos[:, 0]) > 0.01
        return success_tensor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES   
        super()._reset_idx(env_ids)

        replay_mode = getattr(self, "replay_mode", False)
        rand_cfg = self.task_config.get("randomization", {})

        # 1. Object Instance Switch
        obj_inst_cfg = rand_cfg.get("object_instance", {})
        if not replay_mode and obj_inst_cfg.get("enable", False):
            self._switch_active_microwave(env_ids)
        else:
            self._switch_active_microwave(env_ids, force_idx=self.active_mw_idx)

        # 2. Position Randomization
        pos_cfg = rand_cfg.get("position", {})
        enable_pos_rand = not replay_mode and pos_cfg.get("enable", False)
        
        # Global XY shift
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        global_xy = torch.zeros(len(env_ids), 2, device=robot_root_state.device)
        
        if enable_pos_rand:
            g_range = pos_cfg.get("global_xy_range", [-0.1, 0.1])
            global_xy[:, 0].uniform_(g_range[0], g_range[1])
            global_xy[:, 1].uniform_(g_range[0], g_range[1])
            
        robot_root_state[..., :2] += global_xy
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos)
        self.robot_reset_state = np.array(robot_root_state.cpu().detach(), dtype=np.float32)

        # Microwave Position
        TABLE_HEIGHT = 0.53
        z_offset = self.microwave_z_offsets[self.active_mw_idx]
        target_z = TABLE_HEIGHT + z_offset

        mw_pos = torch.zeros(len(env_ids), 13, device=self.device)
        mw_pos[..., 0] = -0.04      # default X (inverted)
        mw_pos[..., 1] = 0.18     # default Y (inverted)
        mw_pos[..., 2] = target_z
        mw_pos[..., 3] = 1.0       # rot W
        mw_pos[..., 4:7] = 0.0     # rot XYZ

        mw_pos[..., :2] += global_xy

        # Relative XY shift
        if enable_pos_rand:
            r_range = pos_cfg.get("relative_xy_range", [-0.02, 0.02])
            rel_xy = torch.empty(len(env_ids), 2, device=self.device).uniform_(r_range[0], r_range[1])
            mw_pos[..., :2] += rel_xy

        self.microwave.write_root_state_to_sim(mw_pos, env_ids=env_ids)
        self.microwave_reset_state = np.array(mw_pos.cpu().detach(), dtype=np.float32)

        # Reset microwave joints
        mw_joint_pos = self.microwave.data.default_joint_pos[env_ids]
        mw_joint_pos.fill_(0.0)
        self.microwave.write_joint_position_to_sim(mw_joint_pos, joint_ids=None, env_ids=env_ids)
        
        mw_joint_vel = self.microwave.data.default_joint_vel[env_ids]
        mw_joint_vel.fill_(0.0)
        self.microwave.write_joint_velocity_to_sim(mw_joint_vel, joint_ids=None, env_ids=env_ids)
        
        self.microwave_joint_reset_state = np.array(mw_joint_pos.cpu().detach(), dtype=np.float32)

        # 3. Texture Randomization
        tex_cfg = rand_cfg.get("texture", {})
        if not replay_mode and tex_cfg.get("enable", False):
            self._randomize_table038_texture(tex_cfg)

        # 4. Lighting Randomization
        light_cfg = rand_cfg.get("lighting", {})
        if not replay_mode and light_cfg.get("enable", False):
            self._randomize_lighting(light_cfg)

    def _switch_active_microwave(self, env_ids, force_idx=None):
        HIDDEN = torch.tensor([[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=self.device)
        
        if force_idx is not None:
            new_idx = force_idx
        else:
            new_idx = random.randint(0, len(self.microwave_pool) - 1)
            
        for i, mw in enumerate(self.microwave_pool):
            if i != new_idx:
                mw.write_root_state_to_sim(HIDDEN.clone(), env_ids=env_ids)
                
        self.active_mw_idx = new_idx
        self.microwave = self.microwave_pool[new_idx]

    def _set_microwave_joint_damping(self, microwave: Articulation):
        stage = self.scene.stage
        microwave_prim = stage.GetPrimAtPath(microwave.cfg.prim_path)
        if not microwave_prim.IsValid():
            return
        
        door_damping_value = 0.00015
        door_joint_friction_value = 0.0
        door_drive_stiffness_value = 0.0
        door_drive_max_force_value = 0.005
        
        def set_joint_damping(prim, damping_value):
            joint = UsdPhysics.Joint(prim)
            if joint:
                joint_type = prim.GetTypeName()
                drive_type = "linear" if "Prismatic" in joint_type or "Linear" in joint_type else "angular"
                
                try:
                    if "FixedJoint" in joint_type:
                        return
                    if "microjoint" not in str(prim.GetPath()):
                        return

                    drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
                    
                    try:
                        stiff_attr = drive.GetStiffnessAttr()
                        if stiff_attr: stiff_attr.Set(door_drive_stiffness_value)
                    except: pass

                    try:
                        maxf_attr = drive.GetMaxForceAttr()
                        if maxf_attr: maxf_attr.Set(door_drive_max_force_value)
                    except: pass

                    try:
                        fr_attr = prim.GetAttribute("physxJoint:friction")
                        if fr_attr and fr_attr.IsValid():
                            fr_attr.Set(door_joint_friction_value)
                    except: pass
                    
                    damping_attr = drive.GetDampingAttr()
                    if damping_attr:
                        damping_attr.Set(door_damping_value)
                    else:
                        drive.CreateDampingAttr(door_damping_value)
                        
                except Exception:
                    pass
            
            for child in prim.GetChildren():
                set_joint_damping(child, damping_value)
        
        set_joint_damping(microwave_prim, door_damping_value)

    def _randomize_lighting(self, cfg: dict):
        tone_range = cfg.get("tone_range", [4.8, 7.8])
        fstop = random.uniform(tone_range[0], tone_range[1])
        set_tone_mapping_fstop(fstop=fstop, enabled=True)

    def _randomize_table038_texture(self, cfg: dict):
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
        
        mw_state = self.microwave.data.default_root_state
        self.microwave_reset_state = np.array(mw_state.cpu().detach(), dtype=np.float32)
        
        mw_joint_state = self.microwave.data.default_joint_pos
        self.microwave_joint_reset_state = np.array(mw_joint_state.cpu().detach(), dtype=np.float32)

    def get_all_pose(self):
        if not hasattr(self, 'robot_reset_state'):
            self.initialize_obs()
        return {
            "robot": self.robot_reset_state,
            "microwave_root": self.microwave_reset_state,
            "microwave_joint": self.microwave_joint_reset_state,
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        if pose and "robot" in pose and pose["robot"] is not None:
            robot_tensor = torch.tensor(pose["robot"], dtype=torch.float32, device=self.device)
            self.robot.write_root_state_to_sim(robot_tensor, env_ids=env_ids)
            
        if pose and "microwave_root" in pose and pose["microwave_root"] is not None:
            mw_tensor = torch.tensor(pose["microwave_root"], dtype=torch.float32, device=self.device)
            self.microwave.write_root_state_to_sim(mw_tensor, env_ids=env_ids)
            
        if pose and "microwave_joint" in pose and pose["microwave_joint"] is not None:
            mw_joint_tensor = torch.tensor(pose["microwave_joint"], dtype=torch.float32, device=self.device)
            if mw_joint_tensor.ndim == 1:
                mw_joint_tensor = mw_joint_tensor.unsqueeze(0)
            
            sim_num_joints = self.microwave.data.joint_pos.shape[1]
            if mw_joint_tensor.shape[1] != sim_num_joints:
                mw_joint_tensor = mw_joint_tensor[:, :sim_num_joints]

            self.microwave.write_joint_position_to_sim(mw_joint_tensor, joint_ids=None, env_ids=env_ids)
        else:
            mw_joint_pos = self.microwave.data.default_joint_pos[env_ids]
            mw_joint_pos.fill_(0.0)
            self.microwave.write_joint_position_to_sim(mw_joint_pos, joint_ids=None, env_ids=env_ids)
