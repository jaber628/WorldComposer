"""
TablewareSkill task skill implementation.

State machine:
    IDLE
      -> APPROACH_A
      -> GRASP_DESCEND
      -> CLOSE_GRIPPER
      -> LIFT
      -> APPROACH_B
      -> PLACE_DESCEND
      -> OPEN_GRIPPER
      -> RETREAT
      -> DONE

IK method:
    Hand-written DLS position IK using PhysX Jacobians directly in world frame.

Grasp point rule:
    1. Collect all world-space vertices from the USD mesh of object_A.
    2. Keep the top band with z >= max_z - TOP_BAND_Z.
    3. Select the point in that band that is closest to the robot base in XY.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 通用基类
# ══════════════════════════════════════════════════════════════════════════════
class BaseSkill:
    """
    All task-specific skills must inherit this class.
    AutoCollection.py calls only reset() and step().
    """

    def reset(self) -> None:
        raise NotImplementedError

    def step(self) -> tuple[torch.Tensor, bool, bool, dict]:
        """
        Returns
        -------
        action  : torch.Tensor  shape [1, dof]
        done    : bool
        success : bool
        info    : dict
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Tableware Skill
# ══════════════════════════════════════════════════════════════════════════════
class TablewareSkill(BaseSkill):
    """
    Pick object_A from its topmost-nearest grasp point and place it on object_B.

    IK uses a hand-written DLS position controller with PhysX Jacobians.
    """

    STAGE_IDLE = "IDLE"
    STAGE_APPROACH_A = "APPROACH_A"
    STAGE_GRASP_DESCEND = "GRASP_DESCEND"
    STAGE_CLOSE_GRIPPER = "CLOSE_GRIPPER"
    STAGE_LIFT = "LIFT"
    STAGE_APPROACH_B = "APPROACH_B"
    STAGE_PLACE_DESCEND = "PLACE_DESCEND"
    STAGE_OPEN_GRIPPER = "OPEN_GRIPPER"
    STAGE_RETREAT = "RETREAT"
    STAGE_DONE = "DONE"

    DOF = 6

    GRIPPER_OPEN = 0.5
    GRIPPER_CLOSED = -0.2

    HOVER_HEIGHT = 0.12
    LIFT_HEIGHT = 0.18
    PLACE_DROP = 0.12
    TOP_BAND_Z = 0.008
    GRASP_Z_OFFSET = 0.08

    WRIST_DOWN_ANGLE = 1.3
    WRIST_STEP_RAD    = 0.15
    WRIST_TOL         = 0.10

    ARRIVE_TOL        = 0.025
    MAX_STAGE_STEPS   = 500

    GRASP_HOLD_STEPS  = 20
    RELEASE_HOLD_STEPS = 15
    RETREAT_HOLD_STEPS = 80

    HOME_JOINTS = np.array([-0.0363, -0.8, 0.8, -0.8, 0.0, GRIPPER_OPEN], dtype=np.float32)
    GRASP_QUAT_WXYZ_WORLD = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    JOINT_STEP_RAD = 0.09

    def __init__(self, env):
        self.env = env
        self.device = env.device

        self._ik_ready = False
        self._arm_joint_ids = None
        self._ee_body_id = None
        self._ee_jacobi_idx = None

        self._stage         = self.STAGE_IDLE
        self._step_in_stage = 0

        self._grasp_point: np.ndarray | None = None
        self._place_point: np.ndarray | None = None

    def _ensure_ik_ready(self) -> None:
        if not self._ik_ready:
            self._setup_ik()
            self._ik_ready = True

    def _setup_ik(self) -> None:
        """
        Resolve joint IDs and end-effector body IDs through SceneEntityCfg.
        IK is computed with a hand-written DLS solver.
        """
        from isaaclab.managers import SceneEntityCfg

        robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_names=["gripper"],
        )
        robot_entity_cfg.resolve(self.env.scene)

        self._arm_joint_ids = list(robot_entity_cfg.joint_ids)
        self._ee_body_id    = int(robot_entity_cfg.body_ids[0])

        robot = self.env.robot
        self._ee_jacobi_idx = (
            int(robot_entity_cfg.body_ids[0]) - 1
            if robot.is_fixed_base
            else int(robot_entity_cfg.body_ids[0])
        )

        logger.info(
            f"[IK] Setup complete. "
            f"arm_joint_ids={self._arm_joint_ids}, "
            f"ee_body_id={self._ee_body_id}, "
            f"ee_jacobi_idx={self._ee_jacobi_idx}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Called once per episode. Pre-compute grasp & place targets."""
        self._ensure_ik_ready()

        self._stage         = self.STAGE_IDLE
        self._step_in_stage = 0

        raw_grasp         = self._compute_grasp_point()
        self._grasp_point = raw_grasp + np.array([0.0, 0.0, self.GRASP_Z_OFFSET], dtype=np.float32)
        self._place_point = self._get_object_B_pos().copy()

        logger.info(
            f"[TablewareSkill] reset. "
            f"grasp_raw={np.round(raw_grasp, 4)}, "
            f"grasp(+offset)={np.round(self._grasp_point, 4)}, "
            f"place={np.round(self._place_point, 4)}"
        )

    def step(self) -> tuple[torch.Tensor, bool, bool, dict]:
        """One control tick. Returns (action, done, success, info)."""
        self._ensure_ik_ready()

        q_now = self.env.robot.data.joint_pos[0].cpu().numpy()
        action_np = q_now.copy()
        done = False
        success = False
        info: dict[str, Any] = {
            "stage": self._stage,
            "step_in_stage": self._step_in_stage,
        }

        if self._stage == self.STAGE_IDLE:
            self._advance_stage(self.STAGE_APPROACH_A)

        elif self._stage == self.STAGE_APPROACH_A:
            hover_target = self._grasp_point + np.array([0.0, 0.0, self.HOVER_HEIGHT], dtype=np.float32)
            action_np, pos_ok = self._move_to(hover_target, q_now, keep_gripper=False)
            action_np = self._apply_wrist_correction(action_np, q_now)

            wrist_ok = abs(self.WRIST_DOWN_ANGLE - q_now[3]) < self.WRIST_TOL
            if (pos_ok and wrist_ok) or self._step_in_stage >= self.MAX_STAGE_STEPS:
                logger.info(
                    f"[Skill] APPROACH_A done "
                    f"(step={self._step_in_stage}, wrist={q_now[3]:.3f}rad) → GRASP_DESCEND"
                )
                self._advance_stage(self.STAGE_GRASP_DESCEND)

        elif self._stage == self.STAGE_GRASP_DESCEND:
            action_np, arrived = self._move_to(self._grasp_point, q_now, keep_gripper=False)
            action_np = self._apply_wrist_correction(action_np, q_now)

            if arrived or self._step_in_stage >= self.MAX_STAGE_STEPS:
                logger.info(f"[Skill] GRASP_DESCEND done (step={self._step_in_stage}) → CLOSE_GRIPPER")
                self._advance_stage(self.STAGE_CLOSE_GRIPPER)

        elif self._stage == self.STAGE_CLOSE_GRIPPER:
            action_np = self.close_gripper(q_now)
            if self._step_in_stage >= self.GRASP_HOLD_STEPS:
                logger.info("[Skill] CLOSE_GRIPPER done → LIFT")
                self._advance_stage(self.STAGE_LIFT)

        elif self._stage == self.STAGE_LIFT:
            target = self._grasp_point + np.array([0.0, 0.0, self.LIFT_HEIGHT], dtype=np.float32)
            action_np, arrived = self._move_to(target, q_now, keep_gripper=True)
            if arrived or self._step_in_stage >= self.MAX_STAGE_STEPS:
                logger.info(f"[Skill] LIFT done (step={self._step_in_stage}) → APPROACH_B")
                self._advance_stage(self.STAGE_APPROACH_B)

        elif self._stage == self.STAGE_APPROACH_B:
            target = self._place_point + np.array([0.0, 0.0, self.LIFT_HEIGHT], dtype=np.float32)
            action_np, arrived = self._move_to(target, q_now, keep_gripper=True)
            if arrived or self._step_in_stage >= self.MAX_STAGE_STEPS:
                logger.info(f"[Skill] APPROACH_B done (step={self._step_in_stage}) → PLACE_DESCEND")
                self._advance_stage(self.STAGE_PLACE_DESCEND)

        elif self._stage == self.STAGE_PLACE_DESCEND:
            target = self._place_point + np.array([0.0, 0.0, self.PLACE_DROP], dtype=np.float32)
            action_np, arrived = self._move_to(target, q_now, keep_gripper=True)
            action_np = self._apply_wrist_correction(action_np, q_now)

            if arrived or self._step_in_stage >= self.MAX_STAGE_STEPS:
                logger.info(f"[Skill] PLACE_DESCEND done (step={self._step_in_stage}) → OPEN_GRIPPER")
                self._advance_stage(self.STAGE_OPEN_GRIPPER)

        elif self._stage == self.STAGE_OPEN_GRIPPER:
            action_np = self.open_gripper(q_now)
            if self._step_in_stage >= self.RELEASE_HOLD_STEPS:
                logger.info("[Skill] OPEN_GRIPPER done → RETREAT")
                self._advance_stage(self.STAGE_RETREAT)

        elif self._stage == self.STAGE_RETREAT:
            action_np, arrived = self._interp_toward(self.HOME_JOINTS, q_now)
            if arrived or self._step_in_stage >= self.RETREAT_HOLD_STEPS:
                logger.info("[Skill] RETREAT done → DONE")
                self._advance_stage(self.STAGE_DONE)

        elif self._stage == self.STAGE_DONE:
            done    = True
            success = True

        self._step_in_stage += 1
        action_t = torch.tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return action_t, done, success, info

    def _advance_stage(self, next_stage: str) -> None:
        """Advance to the next stage and reset the local step counter."""
        self._stage         = next_stage
        self._step_in_stage = 0

    def _apply_wrist_correction(self, action_np: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """Drive `wrist_flex` toward `WRIST_DOWN_ANGLE` on top of the IK output."""
        wrist_err = self.WRIST_DOWN_ANGLE - q_now[3]
        if abs(wrist_err) > self.WRIST_TOL:
            action_np[3] = q_now[3] + float(
                np.clip(wrist_err, -self.WRIST_STEP_RAD, self.WRIST_STEP_RAD)
            )
        return action_np

    DLS_LAMBDA = 0.005

    def _move_to(
        self,
        target_world_pos: np.ndarray,
        q_current: np.ndarray,
        keep_gripper: bool = False,
        target_quat_wxyz_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Run one DLS position-IK update step in world frame."""
        robot = self.env.robot

        ee_pose_w = robot.data.body_pose_w[:, self._ee_body_id]
        joint_pos_arm = robot.data.joint_pos[:, self._arm_joint_ids]
        ee_pos_w = ee_pose_w[:, :3]

        tgt_pos_t = torch.tensor(
            target_world_pos, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        pos_error_w = tgt_pos_t - ee_pos_w

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._arm_joint_ids
        ]
        J_pos = jacobian[:, :3, :]

        lam_I = self.DLS_LAMBDA * torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0)
        A = J_pos @ J_pos.transpose(-2, -1) + lam_I
        q_delta = (
            J_pos.transpose(-2, -1) @
            torch.linalg.solve(A, pos_error_w.unsqueeze(-1))
        ).squeeze(-1)

        arm_ids = self._arm_joint_ids
        q_cur_arm = q_current[arm_ids]
        q_des_arm = (joint_pos_arm + q_delta)[0].cpu().numpy()
        delta = q_des_arm - q_cur_arm
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self.JOINT_STEP_RAD:
            delta = delta * (self.JOINT_STEP_RAD / max_delta)

        action_np = q_current.copy()
        action_np[arm_ids] = q_cur_arm + delta
        action_np[-1] = float(q_current[-1]) if keep_gripper else self.GRIPPER_OPEN

        ee_pos_world = ee_pos_w[0].cpu().numpy()
        dist = float(np.linalg.norm(ee_pos_world - target_world_pos.astype(np.float32)))
        arrived = dist < self.ARRIVE_TOL

        if self._step_in_stage % 20 == 0:
            err_w = pos_error_w[0].cpu().numpy()
            logger.info(
                f"[IK] {self._stage}  step={self._step_in_stage}  "
                f"dist_w={dist:.4f}m  "
                f"err_w={np.round(err_w, 3)}  "
                f"ee_w={np.round(ee_pos_world, 3)}  "
                f"tgt_w={np.round(target_world_pos, 3)}"
            )

        return action_np, arrived

    JOINT_STEP_RAD = 0.09

    def _interp_toward(
        self,
        q_target: np.ndarray,
        q_current: np.ndarray,
        tol: float = 0.02,
    ) -> tuple[np.ndarray, bool]:
        """Interpolate in joint space toward `q_target` with bounded per-step motion."""
        delta     = q_target - q_current
        max_delta = float(np.max(np.abs(delta)))
        if max_delta < tol:
            return q_target.copy(), True
        scale  = min(1.0, self.JOINT_STEP_RAD / max_delta)
        q_next = q_current + delta * scale
        return q_next, False

    # ──────────────────────────────────────────────────────────────────────────
    # Gripper 控制
    # ──────────────────────────────────────────────────────────────────────────
    def open_gripper(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[-1] = self.GRIPPER_OPEN
        return q

    def close_gripper(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[-1] = self.GRIPPER_CLOSED
        return q

    # ──────────────────────────────────────────────────────────────────────────
    # 抓取点计算（与物体 B 中心获取）
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_grasp_point(self) -> np.ndarray:
        """
        Compute a grasp point from the USD mesh of object_A.

        1. Read world-space mesh vertices through XformCache.
        2. Keep the top band with z >= max_z - TOP_BAND_Z.
        3. Select the point nearest to the robot base in XY.
        Falls back to the object physics center if mesh extraction fails.
        """
        obj_pos_w  = self.env.object_A.data.root_pos_w[0].cpu().numpy().astype(np.float64)
        obj_quat_w = self.env.object_A.data.root_quat_w[0].cpu().numpy().astype(np.float64)

        template_path = self.env.object_A.cfg.prim_path
        instance_path = template_path.replace("/World/", "/World/envs/env_0/", 1)

        world_pts = self._get_mesh_world_points_xform(instance_path)
        if world_pts is None or len(world_pts) == 0:
            world_pts = self._get_mesh_world_points_xform(template_path)

        if world_pts is None or len(world_pts) == 0:
            logger.warning("[Skill] Cannot read mesh. Falling back to physics center.")
            return obj_pos_w.astype(np.float32)

        # 验证 XformCache 结果与 physics pos 偏差
        mesh_center = world_pts.mean(axis=0)
        xform_err = float(np.linalg.norm(mesh_center[:2] - obj_pos_w[:2]))
        if xform_err > 0.1:
            logger.warning(
                f"[Skill] XformCache XY err={xform_err:.3f}m → fallback to manual transform"
            )
            local_pts = (
                self._get_mesh_local_points(instance_path)
                or self._get_mesh_local_points(template_path)
            )
            if local_pts is None:
                return obj_pos_w.astype(np.float32)
            world_pts = self._apply_rigid_transform(local_pts, obj_pos_w, obj_quat_w)

        max_z    = world_pts[:, 2].max()
        top_mask = world_pts[:, 2] >= max_z - self.TOP_BAND_Z
        top_pts  = world_pts[top_mask]

        base_pos = self.env.robot.data.root_pos_w[0].cpu().numpy()
        xy_dist  = np.linalg.norm(top_pts[:, :2] - base_pos[:2], axis=1)
        grasp_pt = top_pts[int(np.argmin(xy_dist))].copy().astype(np.float32)

        logger.info(
            f"[Skill] grasp_point={np.round(grasp_pt, 4)}  "
            f"(verts={len(world_pts)}, top-band={len(top_pts)}, "
            f"obj_center={np.round(obj_pos_w, 4)})"
        )
        return grasp_pt

    def _get_mesh_world_points_xform(self, root_prim_path: str) -> np.ndarray | None:
        """Read world-space mesh vertices under `root_prim_path` via `UsdGeom.XformCache`."""
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            xfc   = UsdGeom.XformCache()
            all_pts: list[np.ndarray] = []

            def _visit(prim):
                if prim.IsA(UsdGeom.Mesh):
                    pts_attr = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    if pts_attr is None:
                        return
                    W     = np.array(xfc.GetLocalToWorldTransform(prim)).reshape(4, 4).T
                    local = np.array([[p[0], p[1], p[2]] for p in pts_attr], dtype=np.float64)
                    ones  = np.ones((len(local), 1))
                    world = (np.hstack([local, ones]) @ W.T)[:, :3]
                    all_pts.append(world.astype(np.float32))
                for child in prim.GetChildren():
                    _visit(child)

            root = stage.GetPrimAtPath(root_prim_path)
            if not root.IsValid():
                return None
            _visit(root)
            return np.concatenate(all_pts, axis=0) if all_pts else None
        except Exception as e:
            logger.warning(f"[Skill] XformCache read failed ({root_prim_path}): {e}")
            return None

    def _get_mesh_local_points(self, root_prim_path: str) -> np.ndarray | None:
        """Recursively collect local-space vertices from all `UsdGeom.Mesh` prims under the root path."""
        try:
            import omni.usd
            from pxr import UsdGeom

            stage    = omni.usd.get_context().get_stage()
            all_pts: list[np.ndarray] = []

            def _visit(prim):
                if prim.IsA(UsdGeom.Mesh):
                    pts_attr = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                    if pts_attr is not None:
                        all_pts.append(
                            np.array([[p[0], p[1], p[2]] for p in pts_attr], dtype=np.float32)
                        )
                for child in prim.GetChildren():
                    _visit(child)

            root = stage.GetPrimAtPath(root_prim_path)
            if not root.IsValid():
                logger.warning(f"[Skill] Prim not found: {root_prim_path}")
                return None
            _visit(root)
            return np.concatenate(all_pts, axis=0) if all_pts else None
        except Exception as e:
            logger.warning(f"[Skill] Mesh local read failed ({root_prim_path}): {e}")
            return None

    @staticmethod
    def _apply_rigid_transform(
        local_pts: np.ndarray,
        pos_w: np.ndarray,
        quat_wxyz_w: np.ndarray,
    ) -> np.ndarray:
        """Transform local points into world space using a rigid transform."""
        qw, qx, qy, qz = [float(v) for v in quat_wxyz_w]
        R = np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),    2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),    1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)
        world_pts = (R @ local_pts.astype(np.float64).T).T + pos_w
        return world_pts.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # 坐标工具
    # ──────────────────────────────────────────────────────────────────────────
    def _get_object_A_pos(self) -> np.ndarray:
        return self.env.object_A.data.root_pos_w[0].cpu().numpy()

    def _get_object_B_pos(self) -> np.ndarray:
        return self.env.object_B.data.root_pos_w[0].cpu().numpy()

    def _get_robot_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pos [3], quat_wxyz [4]) of robot base in world frame."""
        s = self.env.robot.data.root_state_w[0].cpu().numpy()
        return s[:3], s[3:7]

    @property
    def _gripper_body_id(self) -> int | None:
        """Compatibility alias for _ee_body_id (used by test scripts)."""
        return self._ee_body_id
