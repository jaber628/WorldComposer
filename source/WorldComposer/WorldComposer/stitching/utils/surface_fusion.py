import numpy as np
from scipy.spatial import cKDTree

from .gaussian import _positions, _surface_cloud


def _cube_rotations():
    rotations = []
    for forward in np.r_[np.eye(3), -np.eye(3)]:
        up = (
            np.array([0.0, 1.0, 0.0])
            if abs(forward[1]) < 0.5
            else np.array([0.0, 0.0, 1.0])
        )
        right = np.cross(forward, up)
        rotations.append(np.column_stack([right, np.cross(forward, right), forward]))
    return np.asarray(rotations)


def _capture(vertices, mask, pose, panorama, device):
    import cv2
    import torch
    from gsplat import rasterization

    size = 256
    intrinsic = torch.tensor(
        [[size / 2, 0, size / 2], [0, size / 2, size / 2], [0, 0, 1]], device=device
    )
    rotations = pose[:3, :3] @ _cube_rotations()
    positions = _positions(vertices)[mask]
    means = torch.as_tensor(positions, device=device, dtype=torch.float32)
    quats = torch.as_tensor(
        np.column_stack([vertices[f"rot_{i}"][mask] for i in range(4)]), device=device
    )
    scales = torch.exp(
        torch.as_tensor(
            np.column_stack([vertices[f"scale_{i}"][mask] for i in range(3)]),
            device=device,
        )
    )
    opacity = torch.sigmoid(
        torch.as_tensor(vertices["opacity"][mask].copy(), device=device)
    )
    colors = np.column_stack([vertices[f"f_dc_{i}"][mask] for i in range(3)])
    colors = torch.as_tensor(
        np.clip(colors * (0.5 / np.sqrt(np.pi)) + 0.5, 0, 1), device=device
    )
    faces = []
    yy, xx = np.mgrid[:size, :size]
    rays = np.stack(
        [(xx - size / 2) / (size / 2), (yy - size / 2) / (size / 2), np.ones_like(xx)],
        -1,
    )
    image = cv2.imread(str(panorama)) if panorama else None
    if panorama and image is None:
        raise FileNotFoundError(panorama)
    for rotation in rotations:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3], c2w[:3, 3] = rotation, pose[:3, 3]
        view = torch.as_tensor(np.linalg.inv(c2w), device=device)[None]
        with torch.no_grad():
            render, alpha, _ = rasterization(
                means,
                quats,
                scales,
                opacity,
                colors,
                view,
                intrinsic[None],
                size,
                size,
                render_mode="RGB+ED",
                packed=True,
            )
        render, alpha = render[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()
        confidence = np.ones((size, size), np.float32)
        if image is not None:
            local = rays @ (pose[:3, :3].T @ rotation).T
            u = (
                np.arctan2(local[..., 0], -local[..., 2]) / (2 * np.pi) + 0.5
            ) * image.shape[1]
            v = (
                0.5 - np.arcsin(local[..., 1] / np.linalg.norm(local, axis=-1)) / np.pi
            ) * image.shape[0]
            reference = cv2.remap(
                image,
                u.astype("f4"),
                v.astype("f4"),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB) / 255
            error = np.abs(
                cv2.GaussianBlur(reference, (5, 5), 0)
                - cv2.GaussianBlur(render[..., :3], (5, 5), 0)
            ).mean(axis=-1)
            confidence = 0.2 + 0.8 * np.exp(-error / 0.12)
        faces.append((render[..., 3], alpha, confidence))
    return {"pose": pose, "rotations": rotations, "faces": faces, "size": size}


def _observe(points, observation):
    delta = points - observation["pose"][:3, 3]
    rotations = observation["rotations"]
    face = np.argmax(delta @ rotations[:, :, 2].T, axis=1)
    distance = np.linalg.norm(delta, axis=1)
    visible = np.zeros(len(points), bool)
    foreground = np.zeros(len(points), bool)
    confidence = np.zeros(len(points))
    size = observation["size"]
    for index, rotation in enumerate(rotations):
        rows = np.flatnonzero(face == index)
        local = delta[rows] @ rotation
        xy = np.rint(
            local[:, :2] / np.maximum(local[:, 2:], 1e-8) * (size / 2) + size / 2
        ).astype(int)
        xy = np.clip(xy, 0, size - 1)
        depth, alpha, quality = [
            p[xy[:, 1], xy[:, 0]] for p in observation["faces"][index]
        ]
        tolerance = np.maximum(0.08, 0.03 * depth)
        visible[rows] = (alpha > 0.7) & (np.abs(local[:, 2] - depth) < tolerance)
        foreground[rows] = (alpha > 0.9) & (local[:, 2] < depth + tolerance)
        confidence[rows] = quality
    return distance, visible, foreground, confidence


def surface_ownership(
    vertices, masks, camera_poses, panoramas=None, device="cuda:0", patch_size=0.2
):
    camera_poses = np.asarray(camera_poses, dtype=float)
    if camera_poses.shape != (2, 4, 4) or not np.isfinite(camera_poses).all():
        raise ValueError("Two finite camera poses are required.")
    if not all(np.any(mask) for mask in masks):
        raise ValueError("Both scenes require valid surface points.")
    panoramas = panoramas or [None, None]
    clouds = [_surface_cloud(v, m, 0.06) for v, m in zip(vertices, masks)]
    observations = [
        _capture(v, m, p, image, device)
        for v, m, p, image in zip(vertices, masks, camera_poses, panoramas)
    ]
    separation = max(np.linalg.norm(camera_poses[0][:3, 3] - camera_poses[1][:3, 3]), 1)
    quality, conflicts = [], []
    for index, cloud in enumerate(clouds):
        own = _observe(cloud["points"], observations[index])
        other = _observe(cloud["points"], observations[1 - index])
        score = own[3] * np.where(own[1], 1.0, 0.25) / (1 + own[0] / separation) ** 2
        other_score = other[3] / (1 + other[0] / separation) ** 2
        conflict = other[2] & (other[0] < own[0]) & (score < 0.75 * other_score)
        quality.append(score)
        conflicts.append(conflict)

    points = np.concatenate([c["points"] for c in clouds])
    normals = np.concatenate([c["normal"] for c in clouds])
    keys = np.column_stack(
        [np.floor(points / patch_size).astype(int), np.argmax(np.abs(normals), axis=1)]
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    anchors, normal = points[first], normals[first]
    costs, available = [], []
    for index, cloud in enumerate(clouds):
        distance, nearest = cloud["tree"].query(anchors, workers=4)
        delta = anchors - cloud["points"][nearest]
        agreement = np.abs(np.sum(normal * cloud["normal"][nearest], axis=1))
        plane = np.abs(np.sum(delta * normal, axis=1))
        valid = (distance < patch_size * 1.5) & (agreement > 0.9) & (plane < 0.04)
        value = -np.log(np.maximum(quality[index][nearest], 1e-6))
        value += 2 * conflicts[index][nearest]
        costs.append(np.where(valid, value, 1e6))
        available.append(valid)
    costs = np.column_stack(costs)
    available = np.column_stack(available)
    labels = np.argmin(costs, axis=1)
    distance, neighbors = cKDTree(anchors).query(
        anchors, k=min(12, len(anchors)), workers=4
    )
    if neighbors.ndim == 1:
        distance, neighbors = distance[:, None], neighbors[:, None]
    agreement = np.abs(np.sum(normal[:, None] * normal[neighbors], axis=-1))
    plane = np.abs(
        np.sum((anchors[:, None] - anchors[neighbors]) * normal[:, None], axis=-1)
    )
    weight = (
        (distance > 1e-8)
        & (distance < patch_size * 1.8)
        & (agreement > 0.9)
        & (plane < 0.06)
    )
    weight = (
        weight.astype(float) * 0.6 / np.maximum(weight.sum(axis=1, keepdims=True), 1)
    )
    for _ in range(12):
        changed = 0
        for parity in (0, 1):
            rows = np.arange(parity, len(labels), 2)
            energy = costs[rows].copy()
            for index in (0, 1):
                energy[:, index] += np.sum(
                    weight[rows] * (labels[neighbors[rows]] != index), axis=1
                )
            selected = np.argmin(energy, axis=1)
            changed += int(np.sum(labels[rows] != selected))
            labels[rows] = selected
        if not changed:
            break
    result, removed = [], []
    offset = 0
    for index, (v, mask, cloud) in enumerate(zip(vertices, masks, clouds)):
        nodes = inverse[offset : offset + len(cloud["points"])]
        alternate = available[nodes].all(axis=1)
        keep = ~conflicts[index] & (~alternate | (labels[nodes] == index))
        _, nearest = cloud["tree"].query(_positions(v)[mask], workers=4)
        selected = mask.copy()
        selected[np.flatnonzero(mask)] &= keep[nearest]
        result.append(selected)
        removed.append(
            {
                "visibility": int(conflicts[index].sum()),
                "removed_gaussians": int(mask.sum() - selected.sum()),
            }
        )
        offset += len(cloud["points"])
    return result, {
        "patch_size": patch_size,
        "patches": len(anchors),
        "removed": removed,
        "static_ownership": True,
        "panorama_guided": all(p is not None for p in panoramas),
    }
