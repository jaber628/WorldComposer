from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares

R_180_Y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


def equirectangular_to_sphere(
    u: float, v: float, width: int, height: int
) -> np.ndarray:
    theta = (u / width - 0.5) * 2 * np.pi
    phi = (0.5 - v / height) * np.pi
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = -np.cos(phi) * np.cos(theta)
    return np.array([x, y, z])


def pixels_to_bearing_vectors(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    bearings = []
    for pt in pts:
        bearing = equirectangular_to_sphere(pt[0], pt[1], width, height)
        bearings.append(bearing)
    return np.array(bearings)


def triangulate_point(
    b1: np.ndarray, b2: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    rays = np.column_stack((R @ b1, -b2))
    depths = np.linalg.lstsq(rays, -t, rcond=None)[0]
    return b1 * depths[0]


def pano_pose_to_connection_convention(
    R_pano: np.ndarray, t_pano: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R_cc = R_pano.T
    t_cc = -R_cc @ t_pano
    return R_cc, t_cc


def rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    is_y_rotation = (
        abs(rotation[1, 1] - 1.0) < 1e-6
        and abs(rotation[1, 0]) < 1e-6
        and abs(rotation[1, 2]) < 1e-6
        and abs(rotation[0, 1]) < 1e-6
        and abs(rotation[2, 1]) < 1e-6
    )
    if is_y_rotation:
        yaw = np.arctan2(rotation[0, 2], rotation[0, 0]) * 180.0 / np.pi
        return np.array([0.0, yaw, 0.0])
    r = R_scipy.from_matrix(rotation)
    return r.as_euler("xyz", degrees=True)


def estimate_yaw_only_pose(
    bearings1,
    bearings2,
    threshold=0.01,
    max_iterations=10000,
    constrain_y_translation=True,
):
    """Estimate an upright pose from spherical epipolar constraints."""
    first = np.array(bearings1, dtype=np.float64, copy=True)
    second = np.array(bearings2, dtype=np.float64, copy=True)
    if len(first) < 5 or first.shape != second.shape:
        raise ValueError("At least five paired bearings are required.")
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(np.linalg.norm(first, axis=1) == 0)
        or np.any(np.linalg.norm(second, axis=1) == 0)
    ):
        raise ValueError("Bearings must be finite and nonzero.")
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    yaw_grid = np.deg2rad(np.arange(-180, 180, 2))
    translation_grid = np.deg2rad(np.arange(0, 180, 2))
    rotated = np.stack(
        [
            np.cos(yaw_grid)[:, None] * first[:, 0]
            + np.sin(yaw_grid)[:, None] * first[:, 2],
            np.broadcast_to(first[:, 1], (len(yaw_grid), len(first))),
            -np.sin(yaw_grid)[:, None] * first[:, 0]
            + np.cos(yaw_grid)[:, None] * first[:, 2],
        ],
        axis=-1,
    )
    directions = np.stack(
        [
            np.cos(translation_grid),
            np.zeros(len(translation_grid)),
            np.sin(translation_grid),
        ],
        axis=1,
    )
    cross = np.cross(rotated, second)
    numerator = np.einsum("yni,ti->ytn", cross, directions)
    p_dot = np.einsum("yni,ti->ytn", rotated, directions)
    q_dot = second @ directions.T
    denominator = np.sqrt(
        np.maximum(2 - p_dot**2 - q_dot.T[None] ** 2 - 2 * numerator**2, 1e-12)
    )
    errors = np.abs(numerator) / denominator
    scores = np.sum(errors < threshold, axis=-1) + 0.5 * np.mean(
        np.maximum(1 - errors / threshold, 0), axis=-1
    )

    def pose(parameters):
        yaw, angle = parameters[:2]
        elevation = 0 if len(parameters) == 2 else parameters[2]
        rotation = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )
        translation = np.array(
            [
                np.cos(angle) * np.cos(elevation),
                np.sin(elevation),
                np.sin(angle) * np.cos(elevation),
            ]
        )
        return rotation, translation

    def residual(parameters, mask):
        rotation, translation = pose(parameters)
        p = first[mask] @ rotation.T
        q = second[mask]
        error = np.cross(p, q) @ translation
        norm = 2 - (p @ translation) ** 2 - (q @ translation) ** 2 - 2 * error**2
        return error / np.sqrt(np.maximum(norm, 1e-12))

    candidates = []
    best = None
    for index in np.argsort(scores.ravel())[::-1]:
        yi, ti = np.unravel_index(index, scores.shape)
        parameters = np.array(
            [yaw_grid[yi], translation_grid[ti]]
            + ([] if constrain_y_translation else [0.0])
        )
        if any(
            abs(np.angle(np.exp(1j * (parameters[0] - item[0])))) < np.deg2rad(10)
            and abs(np.sin(parameters[1] - item[1])) < np.sin(np.deg2rad(10))
            for item in candidates
        ):
            continue
        candidates.append(parameters.copy())
        mask = errors[yi, ti] < threshold
        for _ in range(3):
            if mask.sum() < 5:
                break
            parameters = least_squares(
                residual,
                parameters,
                args=(mask,),
                loss="soft_l1",
                f_scale=threshold / 2,
                max_nfev=100,
            ).x
            mask = (
                np.abs(residual(parameters, np.ones(len(first), dtype=bool)))
                < threshold
            )
        rotation, translation = pose(parameters)
        p = first @ rotation.T
        cosine = np.sum(p * second, axis=1)
        divisor = 1 - cosine**2
        depth1 = (-(p @ translation) + cosine * (second @ translation)) / np.maximum(
            divisor, 1e-12
        )
        depth2 = (-cosine * (p @ translation) + second @ translation) / np.maximum(
            divisor, 1e-12
        )
        positive = mask & (divisor > 1e-6) & (depth1 > 0) & (depth2 > 0)
        negative = mask & (divisor > 1e-6) & (depth1 < 0) & (depth2 < 0)
        if negative.sum() > positive.sum():
            translation = -translation
            positive = negative
        count = int(positive.sum())
        score = (
            count,
            (
                -float(np.median(np.abs(residual(parameters, positive))))
                if count
                else -np.inf
            ),
        )
        if best is None or score > best[0]:
            best = score, rotation, translation, positive
        if len(candidates) >= min(16, max_iterations):
            break
    if best is None or best[0][0] < 5:
        raise RuntimeError(
            "Not enough consistent panorama matches with positive depth."
        )
    print(f"Upright pose estimation: {best[0][0]}/{len(first)} inliers.")
    return best[1], best[2], best[3]
