from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _positions(vertices):
    return np.column_stack([vertices[name] for name in ("x", "y", "z")])


def _voxel_indices(points, voxel_size):
    cells = np.floor(points / voxel_size).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return indices


def _shape(vertices):
    scales = np.exp(np.column_stack([vertices[f"scale_{i}"] for i in range(3)]))
    order = np.argsort(scales, axis=1)
    sorted_scales = np.take_along_axis(scales, order, axis=1)
    quaternions = np.column_stack([vertices[f"rot_{i}"] for i in (1, 2, 3, 0)])
    normals = Rotation.from_quat(quaternions).apply(np.eye(3)[order[:, 0]])
    flat = sorted_scales[:, 0] < 0.35 * sorted_scales[:, 1]
    return sorted_scales, normals, flat


def _quality_mask(vertices):
    positions = _positions(vertices)
    scales, normals, flat = _shape(vertices)
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(scales).all(axis=1)
    valid &= vertices["opacity"] > np.log(0.1 / 0.9)
    if valid.sum() < 4:
        return valid
    compact_limit = 4 * np.percentile(scales[valid, 2], 75)
    support = valid & (scales[:, 2] <= compact_limit)
    anchors = positions[support]
    sample = anchors[
        np.linspace(0, len(anchors) - 1, min(4096, len(anchors))).astype(int)
    ]
    distances, _ = cKDTree(anchors).query(sample, k=2, workers=4)
    positive = distances[:, 1][distances[:, 1] > 1e-8]
    spacing = (
        float(np.median(positive))
        if len(positive)
        else float(np.median(scales[valid, 1]))
    )
    spacing = max(spacing, 1e-8)
    anchors = anchors[_voxel_indices(anchors, 2 * spacing)]
    if len(anchors) < 4:
        return valid
    tree = cKDTree(anchors)
    distances, _ = tree.query(anchors[:: max(1, len(anchors) // 4096)], k=2, workers=4)
    spacing = max(spacing, float(np.median(distances[:, 1])))
    thickness_limit = max(4 * np.percentile(scales[valid, 0], 75), 2 * spacing)
    kept = np.zeros(len(vertices), dtype=bool)
    indices = np.flatnonzero(valid)
    for start in range(0, len(indices), 65536):
        rows = indices[start : start + 65536]
        distance, nearest = tree.query(
            positions[rows], k=min(8, len(anchors)), workers=4
        )
        offsets = anchors[nearest] - positions[rows, None]
        mean = offsets.mean(axis=1)
        spread = np.sqrt(
            np.mean(np.sum((offsets - mean[:, None]) ** 2, axis=2), axis=1)
        )
        normal = normals[rows]
        plane_error = np.median(
            np.abs(np.einsum("nki,ni->nk", offsets, normal)), axis=1
        )
        tangent = mean - np.sum(mean * normal, axis=1)[:, None] * normal
        balanced = np.linalg.norm(tangent, axis=1) <= 2 * spread + 2 * spacing
        thin = scales[rows, 0] <= thickness_limit
        compact = scales[rows, 2] <= compact_limit
        compact &= (
            distance[:, min(3, distance.shape[1] - 1)]
            <= 6 * spacing + 4 * scales[rows, 2]
        )
        planar = thin & flat[rows] & balanced
        planar &= plane_error <= 3 * scales[rows, 0] + 2 * spacing
        planar &= distance[:, 0] <= 6 * spacing + 4 * scales[rows, 1]
        kept[rows] = compact | planar
    return kept


def _surface_cloud(vertices, mask, voxel_size):
    points = _positions(vertices)[mask]
    rows = np.flatnonzero(mask)[_voxel_indices(points, voxel_size)]
    points = _positions(vertices)[rows]
    scales, normals, reliable = _shape(vertices[rows])
    tree = cKDTree(points)
    distance, nearest = tree.query(
        points, k=list(range(1, min(12, len(points)) + 1)), workers=4
    )
    neighbors = points[nearest]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / nearest.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    planar = (eigenvalues[:, 0] < 0.25 * eigenvalues[:, 1]) & (
        eigenvalues[:, 1] > 0.05 * eigenvalues[:, 2]
    )
    normals[planar] = eigenvectors[planar, :, 0]
    thickness = scales[:, 0].copy()
    thickness[planar] = np.minimum(
        thickness[planar], np.sqrt(np.maximum(eigenvalues[planar, 0], 0))
    )
    radius = distance[:, min(3, distance.shape[1] - 1)]
    supported = (radius <= 3 * max(float(np.median(radius)), voxel_size)) & (
        len(points) >= 3
    )
    colors = np.column_stack([vertices[f"f_dc_{i}"][rows] for i in range(3)])
    colors = np.clip(colors * (0.5 / np.sqrt(np.pi)) + 0.5, 0, 1)
    return {
        "points": points,
        "tree": tree,
        "normal": normals,
        "reliable": reliable | planar,
        "thickness": thickness,
        "supported": supported,
        "colors": colors,
    }


def _overlap_masks(target, source, masks, overlap_distance):
    voxel_size = overlap_distance / 5
    clouds = [
        _surface_cloud(vertices, mask, voxel_size)
        for vertices, mask in zip((target, source), masks)
    ]
    matches = [
        clouds[1 - i]["tree"].query(cloud["points"], workers=4)
        for i, cloud in enumerate(clouds)
    ]
    result = []
    for index, cloud in enumerate(clouds):
        other = clouds[1 - index]
        distance, nearest = matches[index]
        reverse = matches[1 - index][1][nearest]
        cycle = np.linalg.norm(cloud["points"] - cloud["points"][reverse], axis=1)
        delta = cloud["points"] - other["points"][nearest]
        agreement = np.abs(np.sum(cloud["normal"] * other["normal"][nearest], axis=1))
        plane_error = np.maximum(
            np.abs(np.sum(delta * cloud["normal"], axis=1)),
            np.abs(np.sum(delta * other["normal"][nearest], axis=1)),
        )
        tolerance = np.minimum(
            overlap_distance / 2,
            np.maximum(
                overlap_distance / 5,
                3 * (cloud["thickness"] + other["thickness"][nearest]),
            ),
        )
        reliable = cloud["reliable"] & other["reliable"][nearest]
        surface = reliable & (agreement > 0.85) & (plane_error < tolerance)
        surface |= ~reliable & (distance < overlap_distance / 10)
        color_error = np.linalg.norm(cloud["colors"] - other["colors"][nearest], axis=1)
        strong = reliable & (agreement > 0.98) & (plane_error < overlap_distance / 10)
        strong &= cycle < voxel_size
        overlap = (distance < overlap_distance) & (cycle <= voxel_size * 1.5)
        overlap &= (
            surface
            & ((color_error < 0.35) | strong)
            & cloud["supported"]
            & other["supported"][nearest]
        )
        _, own = cloud["tree"].query(
            _positions((target, source)[index])[masks[index]], workers=4
        )
        result.append(overlap[own])
    return result


def merge_gaussian_plys(
    target_path,
    source_path,
    transform,
    output_path,
    clean_overlap=False,
    camera_height=1.5,
    overlap_distance=0.15,
    surface_fusion=False,
    camera_poses=None,
    panoramas=None,
    device="cuda:0",
):
    target_path, source_path, output_path = map(
        Path, (target_path, source_path, output_path)
    )
    if output_path.resolve() in (target_path.resolve(), source_path.resolve()):
        raise ValueError("The output must differ from the input PLY files.")
    target = PlyData.read(target_path)["vertex"].data.copy()
    source = PlyData.read(source_path)["vertex"].data.copy()
    if target.dtype != source.dtype:
        raise ValueError("The Gaussian PLY properties must match.")
    if any(name.startswith("f_rest_") for name in source.dtype.names):
        raise ValueError(
            "Higher-order SH coefficients require rotation before merging."
        )
    transform = np.asarray(transform, dtype=np.float64)
    linear = transform[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if scale <= 0 or not np.allclose(
        linear.T @ linear, np.eye(3) * scale**2, atol=1e-6
    ):
        raise ValueError(
            "The transform must contain a rotation and uniform positive scale."
        )
    rotation = linear / scale
    moved = _positions(source) @ linear.T + transform[:3, 3]
    for index, name in enumerate(("x", "y", "z")):
        source[name] = moved[:, index]
    if all(name in source.dtype.names for name in ("nx", "ny", "nz")):
        normals = (
            np.column_stack([source[name] for name in ("nx", "ny", "nz")]) @ rotation.T
        )
        for index, name in enumerate(("nx", "ny", "nz")):
            source[name] = normals[:, index]
    quaternions = np.column_stack([source[f"rot_{i}"] for i in range(4)])
    quaternions = (
        Rotation.from_matrix(rotation)
        * Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    ).as_quat()[:, [3, 0, 1, 2]]
    for index in range(4):
        source[f"rot_{index}"] = quaternions[:, index]
    for index in range(3):
        source[f"scale_{index}"] += np.log(scale)
    masks = [
        _quality_mask(vertices) if clean_overlap else np.ones(len(vertices), dtype=bool)
        for vertices in (target, source)
    ]
    quality_counts = [int(mask.sum()) for mask in masks]
    fusion_info = None
    if surface_fusion:
        if not clean_overlap or camera_poses is None:
            raise ValueError("Surface fusion requires cleanup and two camera poses.")
        from .surface_fusion import surface_ownership

        masks, fusion_info = surface_ownership(
            (target, source), masks, camera_poses, panoramas, device
        )
    elif clean_overlap and all(quality_counts):
        overlap_masks = _overlap_masks(target, source, masks, overlap_distance)
        clouds = [
            _positions(vertices)[mask]
            for vertices, mask in zip((target, source), masks)
        ]
        target_camera = np.array([0, camera_height, 0])
        source_camera = linear @ target_camera + transform[:3, 3]
        for index, points in enumerate(clouds):
            d_target = np.linalg.norm(points - target_camera, axis=1)
            d_source = np.linalg.norm(points - source_camera, axis=1)
            owned = d_target <= d_source if index == 0 else d_source < d_target
            masks[index][np.flatnonzero(masks[index])] &= (
                ~overlap_masks[index]
            ) | owned
    merged = np.concatenate(
        [vertices[mask] for vertices, mask in zip((target, source), masks)]
    )
    if not len(merged):
        raise RuntimeError("No Gaussian remains after cleanup.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False, byte_order="<").write(
        output_path
    )
    report = {
        "file": output_path.name,
        "input_counts": [len(target), len(source)],
        "quality_counts": quality_counts,
        "kept_counts": [int(mask.sum()) for mask in masks],
        "output_count": len(merged),
        "clean_overlap": clean_overlap,
        "relative_scale": scale,
    }
    if fusion_info is not None:
        report["surface_fusion"] = fusion_info
    return report
