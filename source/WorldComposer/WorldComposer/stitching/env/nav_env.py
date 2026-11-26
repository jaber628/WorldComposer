from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from pxr import UsdGeom
from scipy.spatial.transform import Rotation as ScipyRot

from WorldComposer.stitching.env.gs_manager import GSManager
from WorldComposer.stitching.env.nav_env_cfg import NavEnvCfg

MOVE_FORWARD_DIST = 0.25
TURN_ANGLE_DEG = 30
ACTION_MOVE_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_STOP = 3
MIN_DEPTH_M = 0.02
MAX_DEPTH_M = 5.0


def _depth_to_2d_numpy(depth: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(
        depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth,
        dtype=np.float64,
    )
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def merge_depths_min(
    sim_depth: np.ndarray | torch.Tensor, render_depth: np.ndarray
) -> np.ndarray:
    sim_depth = _depth_to_2d_numpy(sim_depth)
    render_depth = _depth_to_2d_numpy(render_depth)
    if sim_depth.shape != render_depth.shape:
        from scipy.ndimage import zoom

        h, w = sim_depth.shape
        render_depth = zoom(
            render_depth,
            (h / render_depth.shape[0], w / render_depth.shape[1]),
            order=1,
        )
    sim_valid = np.isfinite(sim_depth) & (sim_depth > 0)
    render_valid = np.isfinite(render_depth) & (render_depth > 0)
    out = np.full(sim_depth.shape, np.nan, dtype=np.float64)
    both = sim_valid & render_valid
    out[both] = np.minimum(sim_depth[both], render_depth[both])
    out[sim_valid & ~render_valid] = sim_depth[sim_valid & ~render_valid]
    out[render_valid & ~sim_valid] = render_depth[render_valid & ~sim_valid]
    out = np.clip(out, MIN_DEPTH_M, MAX_DEPTH_M)
    norm = np.zeros_like(out, dtype=np.float32)
    valid = np.isfinite(out)
    norm[valid] = (out[valid] - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)
    return norm


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.scene.num_envs != 1 or not str(cfg.sim.device).startswith("cuda"):
            raise ValueError("Navigation requires num_envs=1 and a CUDA device.")
        gs_ply_path = kwargs.pop("gs_ply_path", None)
        self.cam_height = float(getattr(cfg, "camera_relative_height_m", 1.4))
        self._marker_objects = dict(cfg.marker_objects)
        self._camera_prim_path: str = ""
        self._viewport_camera_prim_path: str = ""
        resolved_gs_ply = gs_ply_path if gs_ply_path else cfg.gs_ply_path
        if not resolved_gs_ply or not os.path.isfile(resolved_gs_ply):
            raise FileNotFoundError(
                "A valid stitched GS PLY is required for depth alignment. "
                "Prepare the scene with scripts/stitching/prepare_scene.py first."
            )
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self._target_root_pose: torch.Tensor | None = None
        self._asset_registry = self._build_asset_registry()
        cam_intrinsic = (
            self.scene["head_camera"]
            .data.intrinsic_matrices[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        self.gs_manager = GSManager(
            resolved_gs_ply,
            cam_intrinsic,
            640,
            480,
            device=self.device,
            model_to_world=cfg.model_to_world,
            alpha_threshold=cfg.opacity_threshold,
        )
        self._grid = None
        self._grid_size = 0
        self._grid_origin = 0
        self._grid_resolution = float(getattr(cfg, "grid_resolution_m", 0.05))
        if bool(getattr(cfg, "use_grid_obstacle", True)):
            self._prepare_grid_collision_map()
            if not self._is_valid_world_xy(*cfg.robot.init_state.pos[:2]):
                raise ValueError("The robot spawn must lie in navigable space.")

    def _build_asset_registry(self) -> dict[str, dict[str, object]]:
        reg: dict[str, dict[str, object]] = {
            "xlerobot": {
                "mode": "robot",
                "prim_path": "/World/envs/env_0/xlerobot",
                "xform_tag": "robot_root",
            },
        }
        for k, v in self._marker_objects.items():
            reg[k] = {
                "mode": "xform_marker",
                "prim_path": f"/World/envs/env_0/markers/{k}",
                "xform_tag": "manual_marker_xform",
                "position_xyz": v,
            }
        return reg

    def list_assets(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self._asset_registry.items()]

    def _prepare_grid_collision_map(self) -> None:
        grid_map_path = str(getattr(self.cfg, "grid_map_path", self.cfg.grid_map_path))
        grid_meta_path = str(
            getattr(self.cfg, "grid_meta_path", self.cfg.grid_meta_path)
        )

        from WorldComposer.stitching.utils.navigation import model_digest

        if not (os.path.isfile(grid_map_path) and os.path.isfile(grid_meta_path)):
            raise FileNotFoundError(
                "Run prepare_scene.py to generate the navigation map."
            )
        meta = json.loads(Path(grid_meta_path).read_text())
        if meta.get("model_sha256") != model_digest(self.gs_manager.ply_path):
            raise ValueError("The navigation map does not match the Gaussian model.")
        if not np.allclose(meta["model_to_world"], self.cfg.model_to_world):
            raise ValueError("The navigation map uses a different coordinate system.")

        self._load_grid_map(grid_map_path, grid_meta_path)

    def _load_grid_map(self, grid_map_path: str, grid_meta_path: str) -> None:
        if not os.path.isfile(grid_map_path):
            return
        grid = cv2.imread(grid_map_path, cv2.IMREAD_GRAYSCALE)
        if grid is None or grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
            raise ValueError("The navigation map must be a square grayscale image.")

        self._grid = grid
        self._grid_size = int(grid.shape[0])
        self._grid_origin = self._grid_size // 2

        if os.path.isfile(grid_meta_path):
            with open(grid_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._grid_resolution = float(meta.get("resolution", self._grid_resolution))
            origin = meta.get("origin_pixel", [self._grid_origin, self._grid_origin])
            self._grid_origin = int(origin[0])
            size = meta.get("size", [self._grid_size, self._grid_size])
            self._grid_size = int(size[0])

    def _world_xy_to_grid(self, x: float, y: float) -> tuple[int, int]:
        if self._grid is None:
            return (0, 0)
        col = int(round(x / self._grid_resolution)) + self._grid_origin
        row = (
            self._grid_size - int(round(y / self._grid_resolution)) - self._grid_origin
        )
        return row, col

    def _is_valid_world_xy(self, x: float, y: float) -> bool:
        if self._grid is None:
            return True
        row, col = self._world_xy_to_grid(x, y)
        if row < 0 or row >= self._grid_size or col < 0 or col >= self._grid_size:
            return False
        return bool(self._grid[row, col] == 255)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        ns = (
            "/World/envs/env_0" if self.scene.num_envs == 1 else self.scene.env_regex_ns
        )
        stage = self.scene.stage
        for name, pos in self._marker_objects.items():
            marker_path = f"{ns}/markers/{name}"
            marker_xform = UsdGeom.Xform.Define(stage, marker_path)
            translate_ops = marker_xform.GetOrderedXformOps()
            if translate_ops:
                translate_ops[0].Set((float(pos[0]), float(pos[1]), float(pos[2])))
            else:
                marker_xform.AddTranslateOp().Set(
                    (float(pos[0]), float(pos[1]), float(pos[2]))
                )
        camera_parent_candidates = [
            f"{ns}/xlerobot/base_link",
            f"{ns}/xlerobot/xlerobot/root",
            f"{ns}/xlerobot",
        ]
        camera_parent = f"{ns}/xlerobot"
        for p in camera_parent_candidates:
            if stage.GetPrimAtPath(p).IsValid():
                camera_parent = p
                break
        camera_prim_path = f"{camera_parent}/camera"
        viewport_camera_prim_path = f"{camera_parent}/follow_camera"
        self._camera_prim_path = camera_prim_path
        self._viewport_camera_prim_path = viewport_camera_prim_path
        prim = stage.GetPrimAtPath(camera_prim_path)
        if not prim.IsValid():
            import isaaclab.sim as sim_utils

            R_cam = np.array(
                [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
            )
            q_xyzw = ScipyRot.from_matrix(R_cam).as_quat()
            quat_wxyz = (
                float(q_xyzw[3]),
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
            )
            sim_utils.spawn_camera(
                camera_prim_path,
                sim_utils.PinholeCameraCfg(
                    focal_length=8.7362,
                    horizontal_aperture=20.955,
                    vertical_aperture=20.955,
                    clipping_range=(0.01, 10.0),
                ),
                translation=(0.0, 0.0, self.cam_height),
                orientation=quat_wxyz,
            )

        viewport_prim = stage.GetPrimAtPath(viewport_camera_prim_path)
        if not viewport_prim.IsValid():
            import isaaclab.sim as sim_utils

            viewport_rot = ScipyRot.from_euler(
                "xyz", [55.0, 0.0, -90.0], degrees=True
            ).as_quat()
            viewport_quat_wxyz = (
                float(viewport_rot[3]),
                float(viewport_rot[0]),
                float(viewport_rot[1]),
                float(viewport_rot[2]),
            )
            sim_utils.spawn_camera(
                viewport_camera_prim_path,
                sim_utils.PinholeCameraCfg(
                    focal_length=8.7362,
                    horizontal_aperture=20.955,
                    vertical_aperture=20.955,
                    clipping_range=(0.01, 30.0),
                ),
                translation=(-0.8, 0.0, self.cam_height + 0.75),
                orientation=viewport_quat_wxyz,
            )
        head_camera_cfg = TiledCameraCfg(
            prim_path=camera_prim_path,
            spawn=None,
            data_types=["rgb", "depth"],
            width=640,
            height=480,
            depth_clipping_behavior="zero",
        )
        self.scene.sensors["head_camera"] = TiledCamera(head_camera_cfg)

    @property
    def camera_prim_path(self) -> str:
        return self._camera_prim_path

    @property
    def viewport_camera_prim_path(self) -> str:
        return self._viewport_camera_prim_path

    def get_object_position_by_name(
        self, target_name: str
    ) -> tuple[float, float, float]:
        if target_name in self._marker_objects:
            p = self._marker_objects[target_name]
            return (float(p[0]), float(p[1]), float(p[2]))
        ns = (
            "/World/envs/env_0" if self.scene.num_envs == 1 else self.scene.env_regex_ns
        )
        prim_path = f"{ns}/{target_name}"
        prim = self.scene.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {prim_path}")
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    def get_robot_pose(self) -> tuple[float, float, float]:
        pos = self.robot.data.root_pos_w[0].cpu()
        quat = self.robot.data.root_quat_w[0].cpu().unsqueeze(0)
        _, _, yaw_t = euler_xyz_from_quat(quat)
        return (float(pos[0].item()), float(pos[1].item()), float(yaw_t[0].item()))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is None or self.robot.data.root_pos_w.shape[0] == 0:
            return
        pos = self.robot.data.root_pos_w.clone()
        quat = self.robot.data.root_quat_w.clone()
        turn_rad = np.deg2rad(TURN_ANGLE_DEG)
        for i in range(self.num_envs):
            r, p, yaw_t = euler_xyz_from_quat(quat[i : i + 1])
            yaw = float(yaw_t[0].item())
            act = int(actions[i].item() if actions.ndim == 1 else actions[i, 0].item())
            if act == ACTION_MOVE_FORWARD:
                new_x = float(pos[i, 0].item() + MOVE_FORWARD_DIST * np.cos(yaw))
                new_y = float(pos[i, 1].item() + MOVE_FORWARD_DIST * np.sin(yaw))
                start = pos[i, :2].cpu().numpy()
                samples = max(
                    2, int(np.ceil(MOVE_FORWARD_DIST / self._grid_resolution * 2)) + 1
                )
                if all(
                    self._is_valid_world_xy(x, y)
                    for x, y in np.linspace(start, [new_x, new_y], samples)
                ):
                    pos[i, 0] = new_x
                    pos[i, 1] = new_y
            elif act == ACTION_TURN_LEFT:
                quat[i] = quat_from_euler_xyz(
                    r,
                    p,
                    torch.tensor(
                        [yaw + turn_rad], device=quat.device, dtype=quat.dtype
                    ),
                ).squeeze(0)
            elif act == ACTION_TURN_RIGHT:
                quat[i] = quat_from_euler_xyz(
                    r,
                    p,
                    torch.tensor(
                        [yaw - turn_rad], device=quat.device, dtype=quat.dtype
                    ),
                ).squeeze(0)
        self._target_root_pose = torch.cat([pos, quat], dim=-1)

    def _apply_action(self) -> None:
        if self._target_root_pose is not None:
            self.robot.write_root_pose_to_sim(self._target_root_pose)
        self.robot.write_data_to_sim()

    def _get_observations(
        self,
    ) -> dict[str, tuple[float, float, float] | torch.Tensor | np.ndarray]:
        robot_pose = self.get_robot_pose()
        rz = float(self.robot.data.root_pos_w[0, 2].item())
        camera_pose_xyz_yaw = (
            robot_pose[0],
            robot_pose[1],
            rz + self.cam_height,
            robot_pose[2],
        )
        sim_rgb = self.scene["head_camera"].data.output["rgb"]
        sim_depth = self.scene["head_camera"].data.output["depth"]
        render_rgb, render_depth = self.gs_manager.render_rgbd(camera_pose_xyz_yaw)

        rgb_src = getattr(self.cfg, "obs_rgb_source", "sim")
        depth_src = getattr(self.cfg, "obs_depth_source", "merged")
        sim_rgb_np = np.squeeze(sim_rgb.cpu().numpy(), axis=0)

        if depth_src == "merged":
            depth = merge_depths_min(sim_depth, render_depth)
        elif depth_src == "sim":
            depth = _depth_to_2d_numpy(sim_depth)
            depth = np.clip(depth, MIN_DEPTH_M, MAX_DEPTH_M)
            depth = ((depth - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)).astype(
                np.float32
            )
        elif depth_src == "gs":
            depth = np.clip(render_depth, MIN_DEPTH_M, MAX_DEPTH_M)
            depth = ((depth - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown obs_depth_source: {depth_src!r}")

        if rgb_src == "sim":
            rgb = sim_rgb_np
        elif rgb_src == "gs":
            rgb = (np.clip(render_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Unknown obs_rgb_source: {rgb_src!r}")
        return {
            "robot_xy_yaw": robot_pose,
            "sim_rgb": sim_rgb,
            "sim_depth": sim_depth,
            "render_rgb": render_rgb,
            "render_depth": render_depth,
            "rgb": rgb,
            "depth": depth,
            "depth_sim": _depth_to_2d_numpy(sim_depth),
            "depth_merged": depth,
        }

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        state = self.robot.data.default_root_state[env_ids].clone()
        self.robot.write_root_state_to_sim(state, env_ids=env_ids)
        self._target_root_pose = None
