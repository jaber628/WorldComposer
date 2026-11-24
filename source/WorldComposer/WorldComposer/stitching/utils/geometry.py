from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

R_180_Y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])


def equirectangular_to_sphere(u: float, v: float, width: int, height: int) -> np.ndarray:
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
    a = np.zeros((4, 4))
    a[0] = np.array([b1[2], 0.0, -b1[0], 0.0])
    a[1] = np.array([0.0, b1[2], -b1[1], 0.0])
    p2 = np.hstack([R, t.reshape(3, 1)])
    a[2] = b2[2] * p2[0] - b2[0] * p2[2]
    a[3] = b2[2] * p2[1] - b2[1] * p2[2]
    _, _, vt = np.linalg.svd(a)
    x_h = vt[-1]
    return x_h[:3] / x_h[3] if abs(x_h[3]) > 1e-9 else x_h[:3]


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
    bearings1: np.ndarray,
    bearings2: np.ndarray,
    threshold: float = 0.01,
    max_iterations: int = 10000,
    constrain_y_translation: bool = True,
) -> tuple:
    n_points = len(bearings1)
    if n_points < 5:
        print("Not enough points to estimate pose.")
        return np.eye(3), np.array([0, 0, -1]), np.zeros(n_points, dtype=bool)

    best_R = np.eye(3)
    best_t = np.array([0, 0, -1])
    best_inliers = np.zeros(n_points, dtype=bool)
    best_num_inliers = 0

    for _ in range(max_iterations):
        indices = np.random.choice(n_points, min(5, n_points), replace=False)
        b1_xz = bearings1[indices].copy()
        b2_xz = bearings2[indices].copy()
        b1_xz_norm = np.linalg.norm(b1_xz[:, [0, 2]], axis=1, keepdims=True)
        b2_xz_norm = np.linalg.norm(b2_xz[:, [0, 2]], axis=1, keepdims=True)
        if np.any(b1_xz_norm < 0.1) or np.any(b2_xz_norm < 0.1):
            continue

        yaw_estimates = []
        for i in range(len(indices)):
            vec1 = b1_xz[i, [0, 2]]
            vec2 = b2_xz[i, [0, 2]]
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            cos_angle = np.dot(vec1, vec2)
            sin_angle = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            yaw = np.arctan2(sin_angle, cos_angle)
            yaw_estimates.append(yaw)

        yaw = np.median(yaw_estimates)
        R_y = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
        )

        A_list = []
        b_list = []
        for i in indices:
            b1_rot = R_y @ bearings1[i]
            b2 = bearings2[i]
            I_minus_bb = np.eye(3) - np.outer(b2, b2)
            if constrain_y_translation:
                I_minus_bb[1, :] = 0
                b_residual = b2 - b1_rot
                b_residual[1] = 0
            else:
                b_residual = b2 - b1_rot
            A_list.append(I_minus_bb)
            b_list.append(b_residual)

        A_mat = np.vstack(A_list)
        b_vec = np.concatenate(b_list)

        try:
            _, _, Vt = np.linalg.svd(A_mat)
            t_est = Vt[-1]
            if constrain_y_translation:
                t_est[1] = 0
            t_est = t_est / (np.linalg.norm(t_est) + 1e-10)
        except Exception:
            continue

        errors = []
        for i in range(n_points):
            b2 = bearings2[i]
            t_cross = np.array(
                [[0, -t_est[2], t_est[1]], [t_est[2], 0, -t_est[0]], [-t_est[1], t_est[0], 0]]
            )
            E = t_cross @ R_y
            error = abs(b2 @ E @ bearings1[i])
            errors.append(error)

        errors = np.array(errors)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_R = R_y
            best_t = t_est

    print(f"Yaw-Only Pose Estimation (RANSAC): {best_num_inliers}/{n_points} inliers.")

    if best_num_inliers >= 5:
        inlier_indices = np.where(best_inliers)[0]
        yaw_estimates = []
        for i in inlier_indices:
            vec1 = bearings1[i, [0, 2]]
            vec2 = bearings2[i, [0, 2]]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0.1 and norm2 > 0.1:
                vec1 = vec1 / norm1
                vec2 = vec2 / norm2
                cos_angle = np.dot(vec1, vec2)
                sin_angle = vec1[0] * vec2[1] - vec1[1] * vec2[0]
                yaw = np.arctan2(sin_angle, cos_angle)
                yaw_estimates.append(yaw)

        if len(yaw_estimates) > 0:
            yaw = np.median(yaw_estimates)
            best_R = np.array(
                [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
            )

        A_list = []
        b_list = []
        for i in inlier_indices:
            b1_rot = best_R @ bearings1[i]
            b2 = bearings2[i]
            I_minus_bb = np.eye(3) - np.outer(b2, b2)
            if constrain_y_translation:
                I_minus_bb[1, :] = 0
                b_residual = b2 - b1_rot
                b_residual[1] = 0
            else:
                b_residual = b2 - b1_rot
            A_list.append(I_minus_bb)
            b_list.append(b_residual)

        A_mat = np.vstack(A_list)
        b_vec = np.concatenate(b_list)
        _ = b_vec
        _, _, Vt = np.linalg.svd(A_mat)
        best_t = Vt[-1]
        if constrain_y_translation:
            best_t[1] = 0
        best_t = best_t / (np.linalg.norm(best_t) + 1e-10)

    return best_R, best_t, best_inliers
