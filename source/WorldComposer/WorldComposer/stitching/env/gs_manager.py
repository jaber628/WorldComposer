from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from gsplat import rasterization
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R


class GSManager:
    def __init__(self, ply_path: str, cam_intrinsic: np.ndarray, img_width: int, img_height: int, device: str = "cuda"):
        self.ply_path = ply_path
        self.cam_intrinsic = cam_intrinsic
        self.img_width = img_width
        self.img_height = img_height
        self.device = torch.device(device)
        self._rot_R = R.from_euler("xz", [-90, 180], degrees=True)
        self.cloud: dict[str, np.ndarray] = {}
        self.load_ply()
        self.restore_color()
        self.rotate_cloud()
        self.ready4gsplat()

    def load_ply(self) -> None:
        plydata = PlyData.read(self.ply_path)
        v = plydata["vertex"]
        self.cloud = {
            "positions": np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32),
            "colors": np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32),
            "scales": np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32),
            "rotations": np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32),
            "alphas": v["opacity"].astype(np.float32),
        }

    def restore_color(self) -> None:
        sh_c0 = 0.5 * np.sqrt(1 / np.pi)
        self.cloud["colors"] = np.clip(self.cloud["colors"] * sh_c0 + 0.5, 0.0, 1.0).astype(np.float32)

    def rotate_cloud(self) -> None:
        self.cloud["positions"] = self._rot_R.apply(self.cloud["positions"])
        q_old_raw = self.cloud["rotations"]
        q_old_scipy = q_old_raw[:, [1, 2, 3, 0]]
        r_old = R.from_quat(q_old_scipy)
        q_new_scipy = (self._rot_R * r_old).as_quat()
        q_new_scipy /= np.linalg.norm(q_new_scipy, axis=-1, keepdims=True)
        self.cloud["rotations"] = q_new_scipy[:, [3, 0, 1, 2]]

    def ready4gsplat(self, alpha_threshold: float = 0.5) -> None:
        raw_alphas = torch.from_numpy(self.cloud["alphas"].reshape(-1)).to(self.device).float()
        opacities = torch.sigmoid(raw_alphas)
        mask = opacities > alpha_threshold
        self.means = torch.from_numpy(self.cloud["positions"].reshape(-1, 3)).to(self.device).float()[mask]
        self.scales = torch.exp(torch.from_numpy(self.cloud["scales"].reshape(-1, 3)).to(self.device).float())[mask]
        self.quats = torch.from_numpy(self.cloud["rotations"].reshape(-1, 4)).to(self.device).float()[mask]
        self.opacities = opacities[mask]
        self.colors = torch.from_numpy(self.cloud["colors"].reshape(-1, 3)).to(self.device).float()[mask]

    def _build_c2w_from_xyz_yaw(self, position: np.ndarray, yaw_rad: float) -> np.ndarray:
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        n = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32) if n < 1e-6 else right / n
        down = np.cross(forward, right)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.stack([right, down, forward], axis=1)
        c2w[:3, 3] = position
        return c2w

    def render_rgbd(self, xyz_yaw: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        position = np.array(xyz_yaw[:3], dtype=np.float32)
        yaw_rad = float(xyz_yaw[3])
        w2c = np.linalg.inv(self._build_c2w_from_xyz_yaw(position, yaw_rad))
        return self._render_from_w2c(w2c)

    def render_rgbd_from_c2w(self, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w2c = np.linalg.inv(c2w.astype(np.float32))
        return self._render_from_w2c(w2c)

    def _render_from_w2c(self, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        viewmat = torch.from_numpy(w2c).to(self.device).float().unsqueeze(0)
        K = torch.from_numpy(self.cam_intrinsic).to(self.device).float().unsqueeze(0)
        color, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=viewmat,
            Ks=K,
            width=self.img_width,
            height=self.img_height,
            render_mode="RGB+ED",
            packed=True,
        )
        color = color.squeeze(0)
        rgb = color[..., :3].cpu().numpy()
        depth = color[..., 3].cpu().numpy()
        return rgb, depth

    def generate_grid_map(
        self,
        grid_image_path: str,
        grid_meta_path: str,
        z_range: tuple[float, float],
        resolution: float = 0.05,
        margin_ratio: float = 1.1,
    ) -> None:
        """Create obstacle grid map from 3DGS points if map does not exist."""
        if os.path.isfile(grid_image_path) and os.path.isfile(grid_meta_path):
            return

        os.makedirs(os.path.dirname(grid_image_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(grid_meta_path) or ".", exist_ok=True)

        pts = self.means.detach().cpu().numpy()
        min_z, max_z = float(z_range[0]), float(z_range[1])
        mask = (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)
        xy = pts[mask, :2]
        if xy.shape[0] == 0:
            size = 512
            origin = size // 2
            grid = np.full((size, size), 255, dtype=np.uint8)
            cv2.imwrite(grid_image_path, grid)
            with open(grid_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "resolution": resolution,
                        "origin_pixel": [origin, origin],
                        "size": [size, size],
                        "z_range": [min_z, max_z],
                        "margin_ratio": margin_ratio,
                    },
                    f,
                    indent=2,
                )
            return

        ppm = 1.0 / float(resolution)
        half_extent = max(np.abs(xy[:, 0]).max(), np.abs(xy[:, 1]).max()) * float(margin_ratio)
        size = int(np.ceil(2 * half_extent / float(resolution)))
        size = max(size, 1)
        origin = size // 2

        col = np.rint(xy[:, 0] * ppm).astype(np.int32) + origin
        row = size - np.rint(xy[:, 1] * ppm).astype(np.int32) - origin
        valid = (row >= 0) & (row < size) & (col >= 0) & (col < size)
        row, col = row[valid], col[valid]

        grid = np.full((size, size), 255, dtype=np.uint8)
        grid[row, col] = 0

        obstacle = (grid == 0).astype(np.uint8)
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        grid = np.where(obstacle > 0, 0, 255).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((grid == 0).astype(np.uint8))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 15:
                grid[labels == i] = 255

        cv2.imwrite(grid_image_path, grid)
        with open(grid_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "resolution": float(resolution),
                    "origin_pixel": [int(origin), int(origin)],
                    "size": [int(size), int(size)],
                    "z_range": [min_z, max_z],
                    "margin_ratio": float(margin_ratio),
                },
                f,
                indent=2,
            )
