import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _view(panorama, rotation, width, height):
    import cv2

    yy, xx = np.mgrid[:height, :width]
    rays = (
        np.stack(
            [
                (xx - width / 2) / (width / 2),
                (yy - height / 2) / (width / 2),
                np.ones_like(xx),
            ],
            -1,
        )
        @ rotation.T
    )
    u = (np.arctan2(rays[..., 0], -rays[..., 2]) / (2 * np.pi) + 0.5) * panorama.shape[
        1
    ]
    v = (
        0.5 - np.arcsin(rays[..., 1] / np.linalg.norm(rays, axis=-1)) / np.pi
    ) * panorama.shape[0]
    return cv2.remap(
        panorama,
        u.astype("f4"),
        v.astype("f4"),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def fit_camera_bearings(points, bearings, initial):
    points, bearings = np.asarray(points), np.asarray(bearings)
    if len(points) < 20 or points.shape != bearings.shape:
        raise ValueError("At least 20 image-to-surface correspondences are required.")
    if not np.isfinite(points).all() or not np.isfinite(bearings).all():
        raise ValueError("Camera correspondences must be finite.")

    def residual(parameters):
        rotation = Rotation.from_euler("y", parameters[3]).as_matrix()
        rays = (points - parameters[:3]) @ rotation
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-8)
        return (rays - bearings).ravel()

    initial = np.r_[initial, 0.0]
    span = np.array([initial[1] / 2, initial[1], initial[1] / 2, 0.2])
    result = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=0.01,
        bounds=(initial - span, initial + span),
        max_nfev=200,
    )
    rotation = Rotation.from_euler("y", result.x[3]).as_matrix()
    errors = np.linalg.norm(residual(result.x).reshape(-1, 3), axis=1)
    inliers = errors < 0.02
    if inliers.sum() < 20:
        raise RuntimeError("Panorama camera calibration has too few inliers.")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, result.x[:3]
    return (
        pose,
        inliers,
        {
            "matches": len(points),
            "inliers": int(inliers.sum()),
            "median_angle_deg": float(np.rad2deg(np.median(errors[inliers]))),
        },
    )


def fit_panorama_camera(ply_path, panorama_path, camera_height=1.5, device="cuda:0"):
    import cv2
    import torch
    from ..env.gs_manager import GSManager

    panorama = cv2.imread(str(panorama_path))
    if panorama is None:
        raise FileNotFoundError(panorama_path)
    panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    k = np.array([[160, 0, 160], [0, 160, 120], [0, 0, 1]], np.float32)
    model = GSManager(str(ply_path), k, 320, 240, device)
    model.ready4gsplat(0.1)
    undo = model._rot_R.inv()
    model.means = model.means @ torch.as_tensor(
        undo.as_matrix().T.copy(), device=model.device, dtype=torch.float32
    )
    q = (
        undo * Rotation.from_quat(model.quats.cpu().numpy()[:, [1, 2, 3, 0]])
    ).as_quat()[:, [3, 0, 1, 2]]
    model.quats = torch.as_tensor(q, device=model.device, dtype=torch.float32)
    rotations = []
    for angle in (0, 90, 180, 270):
        forward = np.array([np.sin(np.deg2rad(angle)), 0, -np.cos(np.deg2rad(angle))])
        right = np.cross(forward, [0, 1, 0])
        rotations.append(np.column_stack([right, np.cross(forward, right), forward]))
    references = [_view(panorama, r, 320, 240) / 255 for r in rotations]
    candidates = []
    for height in camera_height * np.array([0.8, 1.0, 1.2, 1.4, 1.6, 2.0]):
        errors = []
        for rotation, reference in zip(rotations, references):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3], pose[:3, 3] = rotation, [0, height, 0]
            rgb, _ = model.render_rgbd_from_c2w(pose)
            errors.append(
                np.abs(
                    cv2.GaussianBlur(rgb, (7, 7), 0)
                    - cv2.GaussianBlur(reference, (7, 7), 0)
                ).mean()
            )
        candidates.append(float(np.mean(errors)))
    height = camera_height * [0.8, 1.0, 1.2, 1.4, 1.6, 2.0][int(np.argmin(candidates))]
    model.img_width, model.img_height = 640, 480
    model.cam_intrinsic = k * 2
    model.cam_intrinsic[2, 2] = 1
    sift, matcher = cv2.SIFT_create(nfeatures=6000), cv2.BFMatcher()
    points, bearings, views = [], [], []
    for index, rotation in enumerate(rotations):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3], pose[:3, 3] = rotation, [0, height, 0]
        reference = _view(panorama, rotation, 640, 480)
        rgb, depth = model.render_rgbd_from_c2w(pose)
        rgb = (np.clip(rgb, 0, 1) * 255).astype("u1")
        kr, dr = sift.detectAndCompute(
            cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None
        )
        kg, dg = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        if dr is None or dg is None or len(dg) < 2:
            continue
        for first, second in matcher.knnMatch(dr, dg, k=2):
            if first.distance >= 0.7 * second.distance:
                continue
            a, b = np.array(kr[first.queryIdx].pt), np.array(kg[first.trainIdx].pt)
            z = depth[round(b[1]), round(b[0])]
            if not 0.2 < z < 20:
                continue
            points.append(
                (np.r_[(b - [320, 240]) / 320, 1] * z) @ rotation.T + [0, height, 0]
            )
            bearing = np.r_[(a - [320, 240]) / 320, 1] @ rotation.T
            bearings.append(bearing / np.linalg.norm(bearing))
            views.append(index)
    pose, inliers, report = fit_camera_bearings(points, bearings, [0, height, 0])
    if len(np.unique(np.array(views)[inliers])) < 3:
        raise RuntimeError(
            "Camera calibration requires matches in at least three panorama directions."
        )
    report["position"] = pose[:3, 3].tolist()
    report["yaw_deg"] = float(np.rad2deg(np.arctan2(pose[0, 2], pose[0, 0])))
    return pose, report
