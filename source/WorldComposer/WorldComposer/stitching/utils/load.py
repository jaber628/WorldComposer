from __future__ import annotations

import numpy as np
import open3d as o3d
from plyfile import PlyData


def load_pcd_from_ply(
    ply_path: str,
    opacity_threshold: float = 0.15,
    scale_threshold: float = 0.05,
    voxel_size: float = 0.05,
    enable_denoising: bool = True,
) -> o3d.geometry.PointCloud:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = set(v.dtype.names or [])

    x = np.asarray(v["x"], dtype=np.float32)
    y = np.asarray(v["y"], dtype=np.float32)
    z = np.asarray(v["z"], dtype=np.float32)
    positions = np.stack([x, y, z], axis=1)

    if {"red", "green", "blue"}.issubset(names):
        colors = np.stack(
            [
                np.asarray(v["red"], dtype=np.float32),
                np.asarray(v["green"], dtype=np.float32),
                np.asarray(v["blue"], dtype=np.float32),
            ],
            axis=1,
        )
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    elif {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_c0 = 0.5 * np.sqrt(1.0 / np.pi)
        colors = np.stack(
            [
                np.asarray(v["f_dc_0"], dtype=np.float32),
                np.asarray(v["f_dc_1"], dtype=np.float32),
                np.asarray(v["f_dc_2"], dtype=np.float32),
            ],
            axis=1,
        )
        colors = np.clip(colors * sh_c0 + 0.5, 0.0, 1.0)
    else:
        colors = np.full((positions.shape[0], 3), 0.7, dtype=np.float32)

    valid = np.ones(positions.shape[0], dtype=bool)
    if "opacity" in names:
        opacity_raw = np.asarray(v["opacity"], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
        valid &= opacity > float(opacity_threshold)
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scales = np.stack(
            [
                np.asarray(v["scale_0"], dtype=np.float32),
                np.asarray(v["scale_1"], dtype=np.float32),
                np.asarray(v["scale_2"], dtype=np.float32),
            ],
            axis=1,
        )
        max_scale = np.exp(scales).max(axis=1)
        valid &= max_scale < float(scale_threshold)

    positions = positions[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if enable_denoising and len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd.voxel_down_sample(voxel_size)
