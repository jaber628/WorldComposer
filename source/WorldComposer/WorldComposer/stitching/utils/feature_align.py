from __future__ import annotations

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint

from WorldComposer.stitching.utils.geometry import (
    estimate_yaw_only_pose,
    pixels_to_bearing_vectors,
    triangulate_point,
)

_extractor = None
_matcher = None


def get_lightglue_models(device: str = "cuda", max_num_keypoints: int = 4096):
    global _extractor, _matcher
    if _extractor is None or _matcher is None:
        print(
            f"Loading SuperPoint + LightGlue models to {device} (max_points={max_num_keypoints})..."
        )
        _extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        _matcher = LightGlue(features="superpoint").eval().to(device)
    elif _extractor.conf.max_num_keypoints != max_num_keypoints:
        _extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    return _extractor, _matcher


def extract_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 4096,
    device: str = "cuda",
    exclude_top_ratio: float = 0.2,
    exclude_bottom_ratio: float = 0.2,
) -> tuple:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU.")
        device = "cpu"
    extractor, matcher = get_lightglue_models(device, max_features)

    h1, _w1 = img1.shape[:2]
    h2, _w2 = img2.shape[:2]
    top_crop1 = int(h1 * exclude_top_ratio)
    bottom_crop1 = int(h1 * (1 - exclude_bottom_ratio))
    top_crop2 = int(h2 * exclude_top_ratio)
    bottom_crop2 = int(h2 * (1 - exclude_bottom_ratio))

    print(
        f"Image 1: Using region y=[{top_crop1}, {bottom_crop1}] (Height {h1}, excluded top {exclude_top_ratio * 100:.0f}% and bottom {exclude_bottom_ratio * 100:.0f}%)"
    )
    print(
        f"Image 2: Using region y=[{top_crop2}, {bottom_crop2}] (Height {h2}, excluded top {exclude_top_ratio * 100:.0f}% and bottom {exclude_bottom_ratio * 100:.0f}%)"
    )

    img1_cropped = img1[top_crop1:bottom_crop1, :]
    img2_cropped = img2[top_crop2:bottom_crop2, :]
    img1_rgb = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2RGB)
    image0 = torch.from_numpy(img1_rgb).permute(2, 0, 1).float() / 255.0
    image1 = torch.from_numpy(img2_rgb).permute(2, 0, 1).float() / 255.0
    image0 = image0.to(device)
    image1 = image1.to(device)

    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    from lightglue.utils import rbd

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01["matches"]
    kp1_cropped = feats0["keypoints"].cpu().numpy()
    kp2_cropped = feats1["keypoints"].cpu().numpy()
    kp1_all = kp1_cropped.copy()
    kp1_all[:, 1] += top_crop1
    kp2_all = kp2_cropped.copy()
    kp2_all[:, 1] += top_crop2
    print(f"Detected keypoints: Img1={len(kp1_all)}, Img2={len(kp2_all)}")

    if len(matches) == 0:
        print("No matches found.")
        return np.array([]), np.array([]), [], kp1_all, kp2_all

    matches_np = matches.cpu().numpy()
    pts1 = kp1_all[matches_np[:, 0]]
    pts2 = kp2_all[matches_np[:, 1]]
    print(f"Number of matches: {len(pts1)}")
    return pts1, pts2, matches_np, kp1_all, kp2_all


def run_panorama_matching_and_ground(
    pano1_path: str,
    pano2_path: str,
    cam_height: float = 1.5,
    exclude_top_ratio: float = 0.2,
    exclude_bottom_ratio: float = 0.1,
    ground_zone_bottom_ratio: float = 0.6,
):
    pano1 = cv2.imread(pano1_path)
    pano2 = cv2.imread(pano2_path)
    if pano1 is None or pano2 is None:
        raise FileNotFoundError(f"Cannot read panorama: {pano1_path} or {pano2_path}")

    h, w = pano1.shape[:2]
    pts1, pts2, _, _, _ = extract_and_match_features(
        pano1,
        pano2,
        exclude_top_ratio=exclude_top_ratio,
        exclude_bottom_ratio=exclude_bottom_ratio,
    )
    if len(pts1) < 8:
        raise RuntimeError("Not enough matches to estimate pose.")

    bearings1 = pixels_to_bearing_vectors(pts1, w, h)
    bearings2 = pixels_to_bearing_vectors(pts2, w, h)
    R, t, inliers = estimate_yaw_only_pose(bearings1, bearings2)
    inlier_idx = np.where(inliers)[0]

    points_3d_list = []
    inlier_idx_for_3d = []
    for i in inlier_idx:
        X = triangulate_point(bearings1[i], bearings2[i], R, t)
        depth1 = np.dot(X, bearings1[i])
        X_cam2 = R @ X + t
        depth2 = np.dot(X_cam2, bearings2[i])
        if depth1 > 0 and depth2 > 0:
            points_3d_list.append(X)
            inlier_idx_for_3d.append(i)
    points_3d = np.array(points_3d_list) if points_3d_list else np.zeros((0, 3))
    inlier_idx_for_3d = np.array(inlier_idx_for_3d, dtype=int)

    y_thresh = h * ground_zone_bottom_ratio if 0 < ground_zone_bottom_ratio < 1 else 0
    scale = 1.0
    scale_ratios = []
    for j in range(len(points_3d)):
        i = inlier_idx_for_3d[j]
        if pts1[i, 1] < y_thresh:
            continue
        by = bearings1[i][1]
        if abs(by) < 1e-6 or by > 0:
            continue
        real_depth = cam_height / abs(by)
        tri_depth = np.linalg.norm(points_3d[j])
        if tri_depth >= 1e-9:
            scale_ratios.append(real_depth / tri_depth)
    if len(scale_ratios) >= 1:
        scale = float(np.median(scale_ratios))

    ground_match_mask = np.zeros(len(pts1), dtype=bool)
    for i in range(len(pts1)):
        if inliers[i] and pts1[i, 1] >= y_thresh:
            ground_match_mask[i] = True

    return {
        "pano1": pano1,
        "pano2": pano2,
        "pts1": pts1,
        "pts2": pts2,
        "inliers": inliers,
        "ground_match_mask": ground_match_mask,
        "points_3d": points_3d,
        "scale": scale,
        "inlier_idx": inlier_idx,
        "R": R,
        "t": t,
    }
