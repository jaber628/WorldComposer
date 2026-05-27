from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def pcd(points: np.ndarray, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if points.size == 0:
        return cloud
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if color is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64), (points.shape[0], 1)))
    return cloud


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Table point cloud not found: {path}')
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError(f'Table point cloud is empty: {path}')
    return points


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape((-1, 3)).astype(np.float32)
    points_h = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (np.asarray(transform, dtype=np.float64) @ points_h.T).T[:, :3].astype(np.float32)


def usd_matrix_to_numpy_column_major(matrix) -> np.ndarray:
    # USD/Gf matrices are conventionally applied to row vectors in Python-facing
    # APIs, while this script uses NumPy column vectors: p' = H @ p. Transpose
    # the raw Gf matrix to get the equivalent column-vector transform.
    return np.asarray(matrix, dtype=np.float64).T


def numpy_column_major_to_gf_matrix(transform: np.ndarray):
    from pxr import Gf  # type: ignore

    raw = np.asarray(transform, dtype=np.float64).T
    matrix = Gf.Matrix4d(1.0)
    for row in range(4):
        for col in range(4):
            matrix[row][col] = float(raw[row, col])
    return matrix


def resolve_table_points_path(phase1_output_dir: Path, table_points: str | None) -> Path:
    if table_points is not None:
        return Path(table_points)

    summary_path = phase1_output_dir / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for key in ('table_points', 'reference_table_points'):
            rel_path = summary.get(key)
            if rel_path:
                candidate = phase1_output_dir / rel_path
                if candidate.exists():
                    return candidate

    return phase1_output_dir / 'table_main_cluster.ply'


def resolve_summary_points_path(phase1_output_dir: Path, key: str) -> Path:
    summary_path = phase1_output_dir / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Phase1 summary not found: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    rel_path = summary.get(key)
    if not rel_path:
        raise KeyError(f'Phase1 summary has no points entry: {key}')
    candidate = phase1_output_dir / rel_path
    if not candidate.exists():
        raise FileNotFoundError(f'Phase1 points file for {key} not found: {candidate}')
    return candidate


def fit_table_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    if points.shape[0] < ransac_n:
        raise RuntimeError(f'Not enough table points for plane fitting: {points.shape[0]} < {ransac_n}')

    cloud = pcd(points)
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = [float(x) for x in plane_model]
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    if normal[2] < 0:
        normal = -normal
        a, b, c, d = [-a, -b, -c, -d]
    return normal, list(inliers), [a, b, c, d]


def project_world_points_to_plane(points_world: np.ndarray, plane_normal: np.ndarray, plane_model: list[float]) -> np.ndarray:
    normal = plane_normal.astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    d = float(plane_model[3])
    signed_dist = points_world.astype(np.float64) @ normal + d
    return (points_world.astype(np.float64) - signed_dist[:, None] * normal[None, :]).astype(np.float32)


def shift_plane_to_z_at_xy(plane_model: list[float], xy: np.ndarray, target_z: float) -> list[float]:
    a, b, c, _ = [float(x) for x in plane_model]
    x, y = [float(v) for v in np.asarray(xy, dtype=np.float64)]
    d = -(a * x + b * y + c * float(target_z))
    return [a, b, c, d]


def compress_world_points_to_plane_band(
    points_world: np.ndarray,
    plane_normal: np.ndarray,
    plane_model: list[float],
    band: float,
) -> np.ndarray:
    if points_world.size == 0:
        return points_world.reshape((-1, 3)).astype(np.float32)
    normal = plane_normal.astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    d = float(plane_model[3])
    points = points_world.astype(np.float64)
    signed_dist = points @ normal + d
    clipped_dist = np.clip(signed_dist, -float(band), float(band))
    return (points - (signed_dist - clipped_dist)[:, None] * normal[None, :]).astype(np.float32)


def find_mesh_prims(stage):
    from pxr import UsdGeom  # type: ignore

    meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    return meshes


def find_composed_mesh_transforms(scene_usd_path: Path) -> list[dict]:
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.Open(str(scene_usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f'Failed to open scene USD: {scene_usd_path}')

    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        local_to_scene = usd_matrix_to_numpy_column_major(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts_scene = transform_points(np.asarray(points, dtype=np.float32), local_to_scene)
        meshes.append({
            'prim_path': str(prim.GetPath()),
            'local_to_scene': local_to_scene,
            'num_vertices': int(pts_scene.shape[0]),
            'scene_bounds': {'min': pts_scene.min(axis=0).tolist(), 'max': pts_scene.max(axis=0).tolist()},
        })
    return meshes


def flatten_table_vertices(
    *,
    mesh_usd_path: Path,
    output_mesh_usd_path: Path,
    table_points_world: np.ndarray,
    plane_normal: np.ndarray,
    plane_model: list[float],
    composed_mesh_transforms: list[dict],
    z_tolerance: float,
    xy_margin: float,
    nearest_xy_distance: float,
    debug_dir: Path | None,
    selection_points_world: np.ndarray | None = None,
    project_selection_points_to_plane: bool = False,
    flatten_target_plane_model: list[float] | None = None,
    flatten_plane_band: float | None = None,
) -> dict:
    from pxr import Sdf, Usd, UsdGeom  # type: ignore

    if not mesh_usd_path.exists():
        raise FileNotFoundError(f'Mesh USD not found: {mesh_usd_path}')

    output_mesh_usd_path.parent.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    layer = Sdf.Layer.FindOrOpen(str(mesh_usd_path.resolve()))
    if layer is None:
        raise RuntimeError(f'Failed to open mesh USD layer: {mesh_usd_path}')
    layer.Export(str(output_mesh_usd_path.resolve()))

    stage = Usd.Stage.Open(str(output_mesh_usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f'Failed to open output mesh USD: {output_mesh_usd_path}')

    mesh_prims = find_mesh_prims(stage)
    if not mesh_prims:
        raise RuntimeError(f'No UsdGeom.Mesh prims found in: {output_mesh_usd_path}')

    raw_selection_points_world = table_points_world if selection_points_world is None else selection_points_world
    if project_selection_points_to_plane:
        selection_points_for_region = project_world_points_to_plane(raw_selection_points_world, plane_normal, plane_model)
    else:
        selection_points_for_region = raw_selection_points_world

    xy_min = selection_points_for_region[:, :2].min(axis=0) - float(xy_margin)
    xy_max = selection_points_for_region[:, :2].max(axis=0) + float(xy_margin)
    table_z = float(np.median(selection_points_for_region[:, 2]))

    selection_xy = selection_points_for_region[:, :2].astype(np.float64)
    selection_xy_cloud = o3d.geometry.PointCloud()
    selection_xy_cloud.points = o3d.utility.Vector3dVector(np.column_stack([selection_xy, np.zeros(selection_xy.shape[0])]))
    kdtree = o3d.geometry.KDTreeFlann(selection_xy_cloud)

    total_vertices = 0
    flattened_vertices = 0
    mesh_stats = []
    selected_before_all = []
    selected_after_all = []
    if not composed_mesh_transforms:
        raise RuntimeError('No composed mesh transforms found in scene USD')

    for mesh_idx, prim in enumerate(mesh_prims):
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points_value = points_attr.Get()
        if not points_value:
            continue

        points_local = np.asarray(points_value, dtype=np.float64)
        composed = composed_mesh_transforms[min(mesh_idx, len(composed_mesh_transforms)-1)]
        mesh_local_to_scene = np.asarray(composed['local_to_scene'], dtype=np.float64)
        mesh_scene_to_local = np.linalg.inv(mesh_local_to_scene)
        points_scene = transform_points(points_local.astype(np.float32), mesh_local_to_scene)
        total_vertices += int(points_scene.shape[0])

        in_xy_bounds = (
            (points_scene[:, 0] >= xy_min[0]) & (points_scene[:, 0] <= xy_max[0]) &
            (points_scene[:, 1] >= xy_min[1]) & (points_scene[:, 1] <= xy_max[1])
        )
        near_z = np.abs(points_scene[:, 2] - table_z) <= float(z_tolerance)
        candidate_idx = np.nonzero(in_xy_bounds & near_z)[0]

        selected = []
        for idx in candidate_idx:
            query = np.array([points_scene[idx, 0], points_scene[idx, 1], 0.0], dtype=np.float64)
            _, _, dist2 = kdtree.search_knn_vector_3d(query, 1)
            if dist2 and float(dist2[0]) <= float(nearest_xy_distance) ** 2:
                selected.append(int(idx))
        selected_idx = np.asarray(selected, dtype=np.int64)

        before_world = points_scene[selected_idx].copy() if selected_idx.size > 0 else np.empty((0, 3), dtype=np.float32)
        if before_world.size and flatten_target_plane_model is not None and flatten_plane_band is not None:
            after_world = compress_world_points_to_plane_band(before_world, plane_normal, flatten_target_plane_model, flatten_plane_band)
        elif before_world.size:
            after_world = project_world_points_to_plane(before_world, plane_normal, plane_model)
        else:
            after_world = before_world

        if selected_idx.size > 0:
            after_local = transform_points(after_world, mesh_scene_to_local)
            points_local[selected_idx] = after_local.astype(np.float64)
            points_attr.Set(points_local.tolist())
            flattened_vertices += int(selected_idx.size)
            selected_before_all.append(before_world)
            selected_after_all.append(after_world)

        mesh_stats.append({
            'prim_path': str(prim.GetPath()),
            'num_vertices': int(points_scene.shape[0]),
            'num_candidates': int(candidate_idx.size),
            'num_flattened': int(selected_idx.size),
        })

    stage.GetRootLayer().Save()

    selected_before = np.concatenate(selected_before_all, axis=0) if selected_before_all else np.empty((0, 3), dtype=np.float32)
    selected_after = np.concatenate(selected_after_all, axis=0) if selected_after_all else np.empty((0, 3), dtype=np.float32)
    debug_files = {}
    if debug_dir is not None:
        o3d.io.write_point_cloud(str(debug_dir / 'table_points_input.ply'), pcd(table_points_world, (0.1, 0.7, 1.0)))
        o3d.io.write_point_cloud(str(debug_dir / 'selection_points_input.ply'), pcd(raw_selection_points_world, (0.9, 0.7, 0.1)))
        o3d.io.write_point_cloud(str(debug_dir / 'selection_points_for_region.ply'), pcd(selection_points_for_region, (0.9, 0.4, 0.1)))
        o3d.io.write_point_cloud(str(debug_dir / 'selected_vertices_before.ply'), pcd(selected_before, (1.0, 0.2, 0.1)))
        o3d.io.write_point_cloud(str(debug_dir / 'selected_vertices_after.ply'), pcd(selected_after, (0.1, 0.9, 0.2)))
        debug_files = {
            'table_points_input': str((debug_dir / 'table_points_input.ply').resolve()),
            'selection_points_input': str((debug_dir / 'selection_points_input.ply').resolve()),
            'selection_points_for_region': str((debug_dir / 'selection_points_for_region.ply').resolve()),
            'selected_vertices_before': str((debug_dir / 'selected_vertices_before.ply').resolve()),
            'selected_vertices_after': str((debug_dir / 'selected_vertices_after.ply').resolve()),
        }

    before_z = selected_before[:, 2] if selected_before.size else np.array([], dtype=np.float32)
    distance_plane_model = flatten_target_plane_model if flatten_target_plane_model is not None else plane_model
    after_plane_dist = selected_after.astype(np.float64) @ plane_normal.astype(np.float64) + float(distance_plane_model[3]) if selected_after.size else np.array([], dtype=np.float64)
    after_z = selected_after[:, 2] if selected_after.size else np.array([], dtype=np.float32)
    stats = {
        'source_mesh_usd': str(mesh_usd_path.resolve()),
        'flattened_mesh_usd': str(output_mesh_usd_path.resolve()),
        'composed_mesh_transforms': [dict(item, local_to_scene=np.asarray(item['local_to_scene'], dtype=np.float64).tolist()) for item in composed_mesh_transforms],
        'num_mesh_prims': len(mesh_prims),
        'num_total_vertices': total_vertices,
        'num_flattened_vertices': flattened_vertices,
        'table_z_median': table_z,
        'z_tolerance': float(z_tolerance),
        'xy_margin': float(xy_margin),
        'nearest_xy_distance': float(nearest_xy_distance),
        'flatten_target': 'shifted_plane_band' if flatten_target_plane_model is not None else 'fitted_plane',
        'flatten_target_plane_model': None if flatten_target_plane_model is None else [float(v) for v in flatten_target_plane_model],
        'flatten_plane_band': None if flatten_plane_band is None else float(flatten_plane_band),
        'selection_points_source': 'table_points_world' if selection_points_world is None else 'selection_points_world',
        'project_selection_points_to_plane': bool(project_selection_points_to_plane),
        'num_plane_fit_points': int(table_points_world.shape[0]),
        'num_selection_points': int(raw_selection_points_world.shape[0]),
        'plane_fit_xy_bounds': {
            'min': table_points_world[:, :2].min(axis=0).tolist(),
            'max': table_points_world[:, :2].max(axis=0).tolist(),
        },
        'selection_xy_bounds_before_projection': {
            'min': raw_selection_points_world[:, :2].min(axis=0).tolist(),
            'max': raw_selection_points_world[:, :2].max(axis=0).tolist(),
        },
        'selection_xy_bounds_for_region': {
            'min': selection_points_for_region[:, :2].min(axis=0).tolist(),
            'max': selection_points_for_region[:, :2].max(axis=0).tolist(),
        },
        'table_xy_bounds': {'min': xy_min.tolist(), 'max': xy_max.tolist()},
        'selected_before_z_min': None if before_z.size == 0 else float(before_z.min()),
        'selected_before_z_max': None if before_z.size == 0 else float(before_z.max()),
        'selected_before_z_std': None if before_z.size == 0 else float(before_z.std()),
        'selected_after_z_min': None if after_z.size == 0 else float(after_z.min()),
        'selected_after_z_max': None if after_z.size == 0 else float(after_z.max()),
        'selected_after_z_std': None if after_z.size == 0 else float(after_z.std()),
        'selected_after_plane_distance_abs_max': None if after_plane_dist.size == 0 else float(np.max(np.abs(after_plane_dist))),
        'selected_after_plane_distance_std': None if after_plane_dist.size == 0 else float(np.std(after_plane_dist)),
        'debug_files': debug_files,
        'mesh_stats': mesh_stats,
    }
    return stats


def default_mesh_usd_path(scene_usd_path: Path) -> Path:
    return scene_usd_path.with_name(f'{scene_usd_path.stem}_mesh.usd')


def default_flattened_mesh_usd_path(scene_usd_path: Path) -> Path:
    return scene_usd_path.with_name(f'{scene_usd_path.stem}_mesh_flattened.usd')


def default_ready_usd_path(scene_usd_path: Path) -> Path:
    return scene_usd_path.with_name(f'{scene_usd_path.stem}_Ready.usd')



def z_on_plane_at_xy(plane_model: list[float], xy: np.ndarray) -> float:
    a, b, c, d = [float(x) for x in plane_model]
    x, y = [float(v) for v in np.asarray(xy, dtype=np.float64)]
    return float(-(a * x + b * y + d) / max(c, 1e-8))


def fit_tabletop_oriented_rectangle(points_world: np.ndarray, plane_normal: np.ndarray, plane_model: list[float]) -> dict:
    if points_world.shape[0] < 3:
        raise RuntimeError(f'Not enough tabletop points for oriented rectangle fitting: {points_world.shape[0]}')

    projected = project_world_points_to_plane(points_world, plane_normal, plane_model)
    xy = projected[:, :2].astype(np.float64)
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy[None, :]
    covariance = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    long_axis = eigvecs[:, 0]
    long_axis /= np.linalg.norm(long_axis) + 1e-12
    short_axis = eigvecs[:, 1]
    short_axis -= long_axis * float(np.dot(short_axis, long_axis))
    short_axis /= np.linalg.norm(short_axis) + 1e-12

    axes = np.column_stack([long_axis, short_axis])
    if np.linalg.det(axes) < 0:
        short_axis = -short_axis
        axes = np.column_stack([long_axis, short_axis])

    local = centered @ axes
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    local_extent = local_max - local_min

    if local_extent[1] > local_extent[0]:
        long_axis, short_axis = short_axis, long_axis
        axes = np.column_stack([long_axis, short_axis])
        if np.linalg.det(axes) < 0:
            short_axis = -short_axis
            axes = np.column_stack([long_axis, short_axis])
        local = centered @ axes
        local_min = local.min(axis=0)
        local_max = local.max(axis=0)
        local_extent = local_max - local_min

    local_center = 0.5 * (local_min + local_max)
    center_xy = mean_xy + axes @ local_center
    raw_yaw_rad = float(np.arctan2(long_axis[1], long_axis[0]))
    yaw_rad = raw_yaw_rad
    if yaw_rad > np.pi / 2:
        yaw_rad -= np.pi
    elif yaw_rad < -np.pi / 2:
        yaw_rad += np.pi
    center_z = z_on_plane_at_xy(plane_model, center_xy)

    corners_local = np.array([
        [local_min[0], local_min[1]],
        [local_max[0], local_min[1]],
        [local_max[0], local_max[1]],
        [local_min[0], local_max[1]],
    ], dtype=np.float64)
    corners_xy = mean_xy[None, :] + corners_local @ axes.T

    return {
        'center_xy': center_xy.tolist(),
        'center_z': float(center_z),
        'long_axis_xy': long_axis.tolist(),
        'short_axis_xy': short_axis.tolist(),
        'raw_yaw_rad': raw_yaw_rad,
        'raw_yaw_deg': float(np.degrees(raw_yaw_rad)),
        'yaw_rad': yaw_rad,
        'yaw_deg': float(np.degrees(yaw_rad)),
        'length': float(local_extent[0]),
        'width': float(local_extent[1]),
        'local_min': local_min.tolist(),
        'local_max': local_max.tolist(),
        'corners_xy': corners_xy.tolist(),
        'pca_eigenvalues': eigvals.tolist(),
        'num_points': int(points_world.shape[0]),
    }


def make_table_normalization_matrix(
    *,
    table_center_xy: np.ndarray,
    table_plane_z: float,
    target_table_height: float,
    table_yaw_rad: float,
) -> np.ndarray:
    cx, cy = [float(v) for v in np.asarray(table_center_xy, dtype=np.float64)]
    cz = float(table_plane_z)
    target_z = float(target_table_height)
    yaw = float(table_yaw_rad)

    translate_to_origin = np.eye(4, dtype=np.float64)
    translate_to_origin[:3, 3] = [-cx, -cy, -cz]

    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    rotate_to_x = np.array([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    translate_to_target = np.eye(4, dtype=np.float64)
    translate_to_target[:3, 3] = [0.0, 0.0, target_z]
    return translate_to_target @ rotate_to_x @ translate_to_origin


def apply_world_table_normalization(
    ready_usd_path: Path,
    *,
    table_center_xy: np.ndarray,
    table_plane_z: float,
    target_table_height: float,
    table_yaw_rad: float = 0.0,
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.Open(str(ready_usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f'Failed to open ready USD for world normalization: {ready_usd_path}')

    world_prim = stage.GetPrimAtPath('/World')
    if not world_prim.IsValid():
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            world_prim = default_prim
    if not world_prim.IsValid():
        raise RuntimeError(f'No /World or valid default prim found in ready USD: {ready_usd_path}')

    cx, cy = [float(v) for v in np.asarray(table_center_xy, dtype=np.float64)]
    cz = float(table_plane_z)
    target_z = float(target_table_height)
    yaw = float(table_yaw_rad)
    yaw_deg = float(np.degrees(yaw))
    wrapper_path = Sdf.Path('/World/Real2SimNormalized')
    yaw_path = wrapper_path.AppendChild('YawRotate')
    center_path = yaw_path.AppendChild('TableCenterToOrigin')
    content_path = center_path.AppendChild('Content')

    existing_wrapper = stage.GetPrimAtPath(wrapper_path)
    if existing_wrapper.IsValid():
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(wrapper_path, Sdf.Path.emptyPath)
        if not stage.GetRootLayer().Apply(edit):
            raise RuntimeError(f'Failed to remove existing normalization wrapper: {wrapper_path}')
        stage = Usd.Stage.Open(str(ready_usd_path.resolve()))
        if stage is None:
            raise RuntimeError(f'Failed to reopen ready USD after removing wrapper: {ready_usd_path}')
        world_prim = stage.GetPrimAtPath('/World')

    movable_children = []
    for child in world_prim.GetChildren():
        name = child.GetName()
        if name in {'physicsScene', 'DefaultLight', 'Real2SimNormalized'}:
            continue
        if child.IsA(UsdGeom.Xformable):
            movable_children.append(child)

    if not movable_children:
        raise RuntimeError(f'No top-level scene prims found under /World to normalize in: {ready_usd_path}')

    wrapper = UsdGeom.Xform.Define(stage, str(wrapper_path))
    yaw_xform = UsdGeom.Xform.Define(stage, str(yaw_path))
    center_xform = UsdGeom.Xform.Define(stage, str(center_path))
    content_xform = UsdGeom.Xform.Define(stage, str(content_path))

    wrapper_xform = UsdGeom.Xformable(wrapper.GetPrim())
    wrapper_xform.ClearXformOpOrder()
    wrapper_xform.AddTranslateOp(opSuffix='target_table_pose').Set(Gf.Vec3d(0.0, 0.0, target_z))

    yaw_xformable = UsdGeom.Xformable(yaw_xform.GetPrim())
    yaw_xformable.ClearXformOpOrder()
    yaw_xformable.AddRotateZOp(opSuffix='table_long_axis_to_x').Set(-yaw_deg)

    center_xformable = UsdGeom.Xformable(center_xform.GetPrim())
    center_xformable.ClearXformOpOrder()
    center_xformable.AddTranslateOp(opSuffix='table_center_to_origin').Set(Gf.Vec3d(-cx, -cy, -cz))

    UsdGeom.Xformable(content_xform.GetPrim()).ClearXformOpOrder()

    moved_prims = []
    edit = Sdf.BatchNamespaceEdit()
    for child in movable_children:
        old_path = child.GetPath()
        new_path = content_path.AppendChild(child.GetName())
        edit.Add(old_path, new_path)
        moved_prims.append({'old': str(old_path), 'new': str(new_path)})
    if not stage.GetRootLayer().Apply(edit):
        raise RuntimeError(f'Failed to move top-level scene prims under normalization wrapper: {ready_usd_path}')

    UsdGeom.Xformable(world_prim).ClearXformOpOrder()
    stage.GetRootLayer().Save()

    return {
        'world_prim': str(world_prim.GetPath()),
        'normalization_mode': 'nested_xforms_rotate_around_table_center',
        'wrapper_prim': str(wrapper_path),
        'yaw_prim': str(yaw_path),
        'center_prim': str(center_path),
        'content_prim': str(content_path),
        'moved_prims': moved_prims,
        'table_center_xy': [cx, cy],
        'table_plane_z_at_center_before': cz,
        'table_yaw_rad_before': yaw,
        'table_yaw_deg_before': yaw_deg,
        'target_table_center': [0.0, 0.0, target_z],
        'target_table_height': target_z,
        'xform_ops': [
            {'type': 'translate', 'suffix': 'table_center_to_origin', 'value': [-cx, -cy, -cz]},
            {'type': 'rotateZ', 'suffix': 'table_long_axis_to_x', 'value_deg': -yaw_deg},
            {'type': 'translate', 'suffix': 'target_table_pose', 'value': [0.0, 0.0, target_z]},
        ],
    }


def write_ready_scene_usd(scene_usd_path: Path, source_mesh_usd_path: Path, flattened_mesh_usd_path: Path, ready_usd_path: Path) -> dict:
    from pxr import Sdf, Usd  # type: ignore

    ready_usd_path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(scene_usd_path.resolve()))
    if layer is None:
        raise RuntimeError(f'Failed to open scene USD layer: {scene_usd_path}')
    layer.Export(str(ready_usd_path.resolve()))

    stage = Usd.Stage.Open(str(ready_usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f'Failed to open ready USD: {ready_usd_path}')

    original_mesh_name = source_mesh_usd_path.name
    replacement_path = Path(os.path.relpath(flattened_mesh_usd_path.resolve(), ready_usd_path.parent.resolve()))
    replacement = './' + replacement_path.as_posix() if not replacement_path.is_absolute() else str(replacement_path)
    scene_dir = scene_usd_path.parent.resolve()
    ready_dir = ready_usd_path.parent.resolve()
    replaced_mesh = 0
    rewritten_assets = []

    for prim in stage.Traverse():
        refs = prim.GetMetadata('references')
        if refs is None:
            continue
        items = list(refs.GetAddedOrExplicitItems())
        if not items:
            continue
        explicit = []
        changed = False
        for ref in items:
            asset_path = ref.assetPath
            if asset_path and Path(asset_path).name == original_mesh_name:
                explicit.append(Sdf.Reference(replacement, ref.primPath, ref.layerOffset, ref.customData))
                replaced_mesh += 1
                changed = True
                continue

            if asset_path:
                asset = Path(asset_path)
                candidate = asset if asset.is_absolute() else (scene_dir / asset)
                if not candidate.exists():
                    same_name_in_scene_dir = scene_dir / asset.name
                    same_name_in_ready_intermediate = ready_dir / 'real2sim_intermediate' / asset.name
                    if same_name_in_scene_dir.exists():
                        candidate = same_name_in_scene_dir
                    elif same_name_in_ready_intermediate.exists():
                        candidate = same_name_in_ready_intermediate
                if candidate.exists():
                    rel = Path(os.path.relpath(candidate.resolve(), ready_dir))
                    rel_asset = './' + rel.as_posix()
                    if rel_asset != asset_path:
                        explicit.append(Sdf.Reference(rel_asset, ref.primPath, ref.layerOffset, ref.customData))
                        rewritten_assets.append({'prim_path': str(prim.GetPath()), 'old': asset_path, 'new': rel_asset})
                        changed = True
                        continue

            explicit.append(ref)
        if changed:
            new_refs = Sdf.ReferenceListOp()
            new_refs.explicitItems = explicit
            prim.SetMetadata('references', new_refs)
    stage.GetRootLayer().Save()
    if replaced_mesh == 0:
        raise RuntimeError(f'No mesh reference named {original_mesh_name} was replaced in {ready_usd_path}')
    return {
        'ready_usd': str(ready_usd_path.resolve()),
        'replaced_mesh_references': replaced_mesh,
        'replacement_mesh': replacement,
        'rewritten_asset_references': rewritten_assets,
    }



def write_recognized_points_debug_usd(
    output_usd_path: Path,
    scene_usd_path: Path,
    recognized_points_world: np.ndarray,
    title: str,
    *,
    projected_points_world: np.ndarray | None = None,
    plane_points_world: np.ndarray | None = None,
):
    from pxr import Gf, Usd, UsdGeom, UsdLux  # type: ignore

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd_path.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())

    scene_prim = stage.DefinePrim('/World/SceneDebug', 'Xform')
    scene_prim.GetReferences().AddReference(str(scene_usd_path.resolve()))

    def add_points(name: str, points: np.ndarray, color: tuple[float, float, float], width: float):
        if points is None or points.size == 0:
            return None
        pts = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points.astype(np.float64)]
        geom = UsdGeom.Points.Define(stage, f'/World/{name}')
        geom.CreatePointsAttr(pts)
        geom.CreateWidthsAttr([float(width)] * len(pts))
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return geom

    add_points('RecognizedReferencePoints_Red', recognized_points_world, (1.0, 0.02, 0.02), 0.018)
    add_points('ProjectedReferencePoints_Orange', projected_points_world, (1.0, 0.45, 0.02), 0.012)
    add_points('PlaneFitPoints_Cyan', plane_points_world, (0.05, 0.75, 1.0), 0.012)

    all_for_camera = [recognized_points_world]
    if projected_points_world is not None and projected_points_world.size:
        all_for_camera.append(projected_points_world)
    if plane_points_world is not None and plane_points_world.size:
        all_for_camera.append(plane_points_world)
    merged = np.concatenate(all_for_camera, axis=0)
    center = merged.mean(axis=0)
    xy_min = merged[:, :2].min(axis=0)
    xy_max = merged[:, :2].max(axis=0)
    extent = np.maximum(xy_max - xy_min, 0.5)
    cam_dist = float(max(extent[0], extent[1]) * 2.8)

    camera = UsdGeom.Camera.Define(stage, '/World/DebugCamera')
    camera.CreateFocalLengthAttr(18.0)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1] - cam_dist), float(center[2] + cam_dist * 0.8)))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3d(60.0, 0.0, 0.0))

    light = UsdLux.DistantLight.Define(stage, '/World/KeyLight')
    light.CreateIntensityAttr(3500.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 30.0))
    stage.SetMetadata('customLayerData', {'real2simDebugTitle': title})
    stage.GetRootLayer().Save()
    return str(output_usd_path.resolve())


def write_mesh_plane_debug_usd(
    output_usd_path: Path,
    scene_usd_path: Path,
    plane_model: list[float],
    table_points_world: np.ndarray,
    title: str,
):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # type: ignore

    output_usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd_path.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())

    scene_prim = stage.DefinePrim('/World/SceneDebug', 'Xform')
    scene_prim.GetReferences().AddReference(str(scene_usd_path.resolve()))

    xy_min = table_points_world[:, :2].min(axis=0)
    xy_max = table_points_world[:, :2].max(axis=0)
    margin = 0.08
    x0, y0 = xy_min - margin
    x1, y1 = xy_max + margin
    a, b, c, d = [float(x) for x in plane_model]

    def z_on_plane(x: float, y: float) -> float:
        return float(-(a * x + b * y + d) / max(c, 1e-8))

    plane = UsdGeom.Mesh.Define(stage, '/World/FittedTablePlane')
    pts = [
        Gf.Vec3f(float(x0), float(y0), z_on_plane(float(x0), float(y0))),
        Gf.Vec3f(float(x1), float(y0), z_on_plane(float(x1), float(y0))),
        Gf.Vec3f(float(x1), float(y1), z_on_plane(float(x1), float(y1))),
        Gf.Vec3f(float(x0), float(y1), z_on_plane(float(x0), float(y1))),
    ]
    plane.CreatePointsAttr(pts)
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
    plane.CreateDoubleSidedAttr(True)

    center = table_points_world.mean(axis=0)
    extent = np.maximum(xy_max - xy_min, 0.5)
    cam_dist = float(max(extent[0], extent[1]) * 2.4)
    camera = UsdGeom.Camera.Define(stage, '/World/DebugCamera')
    camera.CreateFocalLengthAttr(18.0)
    cam_x = float(center[0])
    cam_y = float(center[1] - cam_dist)
    cam_z = float(center[2] + cam_dist * 0.8)
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3d(60.0, 0.0, 0.0))
    stage.SetMetadata('customLayerData', {'real2simDebugTitle': title})

    light = UsdLux.DistantLight.Define(stage, '/World/KeyLight')
    light.CreateIntensityAttr(3500.0)
    UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 30.0))
    stage.GetRootLayer().Save()
    return str(output_usd_path.resolve())


def main():
    parser = argparse.ArgumentParser(description='Real2Sim phase 3: flatten tabletop mesh vertices and normalize the final scene pose.')
    parser.add_argument('--scene_usd', type=Path, required=True, help='Original scene USD, used for default path resolution')
    parser.add_argument('--phase1_output_dir', type=Path, required=True, help='Phase1 output directory containing table point cloud')
    parser.add_argument('--table_points', type=str, default=None, help='Optional explicit table point cloud path')
    parser.add_argument('--selection_points', type=str, default=None, help='Optional point cloud used only for XY/nearest-neighbor mesh vertex selection')
    parser.add_argument('--selection_points_from_summary', type=str, default='reference_table_points', choices=['table_points', 'reference_table_points'], help='Use a Phase1 summary point cloud only for mesh vertex selection')
    parser.add_argument('--project_selection_points_to_plane', action='store_true', default=True, help='Project selection points to the fitted plane before computing XY bounds and nearest-neighbor selection')
    parser.add_argument('--mesh_usd', type=Path, default=None, help='Source mesh USD, default: <scene_stem>_mesh.usd next to scene_usd')
    parser.add_argument('--flattened_mesh_usd', type=Path, default=None, help='Output flattened mesh USD, default: <scene_stem>_mesh_flattened.usd')
    parser.add_argument('--ready_usd', type=Path, default=None, help='Output ready USD referencing the flattened mesh, default: <scene_stem>_Ready.usd')
    parser.add_argument('--output_dir', type=Path, default=None, help='Phase3 metadata output dir')
    parser.add_argument('--save_debug_artifacts', action='store_true', default=False, help='Save optional Phase3 diagnostic PLY/USD visualizations')
    parser.add_argument('--plane_distance_threshold', type=float, default=0.01)
    parser.add_argument('--plane_ransac_n', type=int, default=3)
    parser.add_argument('--plane_num_iterations', type=int, default=2000)
    parser.add_argument('--flatten_z_tolerance', type=float, default=0.04)
    parser.add_argument('--flatten_xy_margin', type=float, default=0.04)
    parser.add_argument('--flatten_nearest_xy_distance', type=float, default=0.04)
    parser.add_argument('--flatten_plane_z', type=float, default=None, help='Optional later-stage height shift: move the fitted plane to z=<value> at the selection-region center before band compression')
    parser.add_argument('--flatten_plane_band', type=float, default=0.0, help='Compress selected vertices into +/- this signed-distance band around the target plane; by default the target is the original fitted plane')
    parser.add_argument('--target_table_height', type=float, default=0.464, help='After flattening, normalize the whole ready scene so the fitted tabletop rectangle center is at XY=(0,0), its long edge is aligned with +X, and its height is this value')
    parser.add_argument('--table_yaw_offset_deg', type=float, default=180.0, help='Additional horizontal Z-up yaw offset after aligning the table long edge; use 180 when the robot faces the wrong side of the table')
    parser.add_argument('--disable_world_normalization', action='store_true', default=False, help='Disable final whole-scene pose normalization')
    args = parser.parse_args()

    scene_usd = args.scene_usd
    if not scene_usd.exists():
        raise FileNotFoundError(f'Scene USD not found: {scene_usd}')

    phase2_dir = args.output_dir or (args.phase1_output_dir / 'phase2_table_flatten')
    phase2_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = phase2_dir / 'debug' if args.save_debug_artifacts else None

    table_points_path = resolve_table_points_path(args.phase1_output_dir, args.table_points)
    selection_points_path = None
    if args.selection_points is not None:
        selection_points_path = Path(args.selection_points)
    elif args.selection_points_from_summary is not None:
        selection_points_path = resolve_summary_points_path(args.phase1_output_dir, args.selection_points_from_summary)
    mesh_usd = args.mesh_usd or default_mesh_usd_path(scene_usd)
    flattened_mesh_usd = args.flattened_mesh_usd or default_flattened_mesh_usd_path(scene_usd)
    ready_usd = args.ready_usd or default_ready_usd_path(scene_usd)

    table_points = load_points(table_points_path)
    selection_points = load_points(selection_points_path) if selection_points_path is not None else None
    composed_mesh_transforms = find_composed_mesh_transforms(scene_usd)
    print(f'[Phase3] composed_mesh_count={len(composed_mesh_transforms)}')
    for item in composed_mesh_transforms:
        print(f"[Phase3] composed_mesh={item['prim_path']} bounds={item['scene_bounds']}")
    plane_normal, inliers, plane_model = fit_table_plane(
        table_points,
        distance_threshold=args.plane_distance_threshold,
        ransac_n=args.plane_ransac_n,
        num_iterations=args.plane_num_iterations,
    )
    table_plane_inliers_path = None
    if args.save_debug_artifacts:
        inlier_points = table_points[np.asarray(inliers, dtype=np.int64)]
        table_plane_inliers_path = phase2_dir / 'table_plane_inliers.ply'
        o3d.io.write_point_cloud(str(table_plane_inliers_path), pcd(inlier_points, (0.1, 0.9, 0.2)))

    recognized_points_for_debug = selection_points if selection_points is not None else table_points
    projected_points_for_debug = (
        project_world_points_to_plane(recognized_points_for_debug, plane_normal, plane_model)
        if args.project_selection_points_to_plane
        else None
    )
    plane_debug_points = projected_points_for_debug if projected_points_for_debug is not None else recognized_points_for_debug
    target_plane_model = (
        shift_plane_to_z_at_xy(plane_model, plane_debug_points[:, :2].mean(axis=0), args.flatten_plane_z)
        if args.flatten_plane_z is not None
        else plane_model
    )
    plane_debug_model = target_plane_model
    plane_debug_usd = None
    recognized_points_debug_usd = None
    if args.save_debug_artifacts:
        plane_debug_usd = write_mesh_plane_debug_usd(
            phase2_dir / 'debug_plane_mesh_only.usd',
            scene_usd,
            plane_debug_model,
            plane_debug_points,
            'mesh-only target table plane debug',
        )
        recognized_points_debug_usd = write_recognized_points_debug_usd(
            phase2_dir / 'debug_recognized_reference_points.usd',
            scene_usd,
            recognized_points_for_debug,
            'recognized reference points debug',
            projected_points_world=projected_points_for_debug,
            plane_points_world=table_points,
        )

    flatten_stats = flatten_table_vertices(
        mesh_usd_path=mesh_usd,
        output_mesh_usd_path=flattened_mesh_usd,
        table_points_world=table_points,
        plane_normal=plane_normal,
        plane_model=plane_model,
        composed_mesh_transforms=composed_mesh_transforms,
        z_tolerance=args.flatten_z_tolerance,
        xy_margin=args.flatten_xy_margin,
        nearest_xy_distance=args.flatten_nearest_xy_distance,
        debug_dir=debug_dir,
        selection_points_world=selection_points,
        project_selection_points_to_plane=args.project_selection_points_to_plane,
        flatten_target_plane_model=target_plane_model,
        flatten_plane_band=args.flatten_plane_band,
    )
    ready_stats = write_ready_scene_usd(scene_usd, mesh_usd, flattened_mesh_usd, ready_usd)
    rectangle_points_for_normalization = plane_debug_points
    table_rectangle = fit_tabletop_oriented_rectangle(rectangle_points_for_normalization, plane_normal, target_plane_model)
    yaw_offset_rad = float(np.radians(args.table_yaw_offset_deg))
    table_yaw_rad = float(table_rectangle['yaw_rad']) + yaw_offset_rad
    if table_yaw_rad > np.pi:
        table_yaw_rad -= 2.0 * np.pi
    elif table_yaw_rad < -np.pi:
        table_yaw_rad += 2.0 * np.pi
    table_rectangle['yaw_offset_deg'] = float(args.table_yaw_offset_deg)
    table_rectangle['final_yaw_rad'] = table_yaw_rad
    table_rectangle['final_yaw_deg'] = float(np.degrees(table_yaw_rad))
    table_center_xy = np.asarray(table_rectangle['center_xy'], dtype=np.float64)
    table_plane_z_at_center = float(table_rectangle['center_z'])
    world_normalization_stats = None
    if not args.disable_world_normalization:
        world_normalization_stats = apply_world_table_normalization(
            ready_usd,
            table_center_xy=table_center_xy,
            table_plane_z=table_plane_z_at_center,
            target_table_height=args.target_table_height,
            table_yaw_rad=table_yaw_rad,
        )
    flatten_debug_usd = None
    if args.save_debug_artifacts:
        flatten_debug_usd = write_mesh_plane_debug_usd(
            phase2_dir / 'debug_flattened_mesh_only.usd',
            ready_usd,
            plane_debug_model,
            plane_debug_points,
            'mesh-only flattened tabletop debug',
        )

    plane_dist = table_points.astype(np.float64) @ plane_normal.astype(np.float64) + float(plane_model[3])
    stats = {
        'scene_usd': str(scene_usd.resolve()),
        'ready_usd': ready_stats,
        'world_normalization': world_normalization_stats,
        'table_oriented_rectangle': table_rectangle,
        'table_points': str(table_points_path.resolve()),
        'selection_points': None if selection_points_path is None else str(selection_points_path.resolve()),
        'project_selection_points_to_plane': bool(args.project_selection_points_to_plane),
        'flatten_plane_z': None if args.flatten_plane_z is None else float(args.flatten_plane_z),
        'flatten_plane_band': float(args.flatten_plane_band),
        'target_plane_model': [float(v) for v in target_plane_model],
        'mesh_flattening': flatten_stats,
        'plane_model': plane_model,
        'plane_normal': plane_normal.tolist(),
        'num_table_points': int(table_points.shape[0]),
        'num_plane_inliers': int(len(inliers)),
        'plane_inlier_ratio': float(len(inliers) / max(table_points.shape[0], 1)),
        'table_point_plane_distance_abs_max': float(np.max(np.abs(plane_dist))),
        'table_point_plane_distance_std': float(np.std(plane_dist)),
        'table_plane_inliers': None if table_plane_inliers_path is None else str(table_plane_inliers_path.resolve()),
        'debug_usd': {
            'plane_mesh_only': plane_debug_usd,
            'recognized_reference_points': recognized_points_debug_usd,
            'flattened_mesh_only': flatten_debug_usd,
        },
    }
    stats_path = phase2_dir / 'table_flatten_stats.json'
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[Phase3] table_points={table_points_path}')
    print(f'[Phase3] selection_points={selection_points_path}')
    print(f"[Phase3] plane={plane_model} inliers={len(inliers)}/{table_points.shape[0]} dist_std={stats['table_point_plane_distance_std']:.6f}")
    print(f"[Phase3] flattened_vertices={flatten_stats['num_flattened_vertices']}/{flatten_stats['num_total_vertices']}")
    print(f"[Phase3] flattened_mesh={flatten_stats['flattened_mesh_usd']}")
    print(f"[Phase3] ready_usd={ready_stats['ready_usd']} replaced_refs={ready_stats['replaced_mesh_references']}")
    if world_normalization_stats is not None:
        print(
            "[Phase3] world_normalization="
            f"mode={world_normalization_stats['normalization_mode']} "
            f"center={world_normalization_stats['table_center_xy']} "
            f"yaw_deg={world_normalization_stats['table_yaw_deg_before']:.3f} "
            f"target_table_height={world_normalization_stats['target_table_height']} "
            f"wrapper={world_normalization_stats['wrapper_prim']}"
        )
    print(
        "[Phase3] table_rectangle="
        f"center={table_rectangle['center_xy']} "
        f"length={table_rectangle['length']:.4f} width={table_rectangle['width']:.4f} "
        f"yaw_deg={table_rectangle['yaw_deg']:.3f}"
    )
    print(f'[Phase3] stats={stats_path}')


if __name__ == '__main__':
    main()
