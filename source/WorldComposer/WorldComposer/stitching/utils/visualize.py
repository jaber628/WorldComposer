from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d


def show_image(
    img: np.ndarray,
    window_name: str,
    max_width: int = 1920,
    max_height: int = 1080,
) -> None:
    h, w = img.shape[:2]
    scale_w = max_width / w if max_width > 0 else 1.0
    scale_h = max_height / h if max_height > 0 else 1.0
    scale = min(scale_w, scale_h, 1.0)
    display_img = img
    if scale < 1.0:
        display_img = cv2.resize(
            img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
        )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
    cv2.imshow(window_name, display_img)
    while True:
        key = cv2.waitKey(30)
        if key != -1:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(window_name)


def visualize_panorama_matches_and_ground(
    pano1: np.ndarray,
    pano2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inliers: np.ndarray,
    ground_match_mask: np.ndarray,
    max_draw_all: int = 80,
    max_draw_ground: int = 50,
    displacement_angle_tol_deg: float = 40.0,
) -> np.ndarray:
    _ = max_draw_ground, displacement_angle_tol_deg
    h, _w = pano1.shape[:2]
    combined = np.vstack([pano1, pano2]).copy()

    draw_idx = np.where(inliers)[0]
    if len(draw_idx) > max_draw_all:
        draw_idx = np.random.choice(draw_idx, min(max_draw_all, len(draw_idx)), replace=False)
    for i in draw_idx:
        p1 = (int(pts1[i, 0]), int(pts1[i, 1]))
        p2 = (int(pts2[i, 0]), int(pts2[i, 1] + h))
        cv2.line(combined, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(combined, p1, 5, (0, 255, 0), -1)
        cv2.circle(combined, p2, 5, (0, 255, 0), -1)

    ground_idx = np.where(ground_match_mask & inliers)[0]
    for i in ground_idx:
        p1 = (int(pts1[i, 0]), int(pts1[i, 1]))
        p2 = (int(pts2[i, 0]), int(pts2[i, 1] + h))
        cv2.circle(combined, p1, 4, (0, 0, 255), -1)
        cv2.circle(combined, p2, 4, (0, 0, 255), -1)
    return combined


def visualize_icp_result(
    pcd1: o3d.geometry.PointCloud,
    pcd2_aligned: o3d.geometry.PointCloud,
    red_offset: tuple[float, float, float] = (0.15, 0.0, 0.0),
    blue_offset: tuple[float, float, float] = (0.0, 0.0, 0.15),
    window_name: str = "ICP Alignment",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    def _copy_no_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        return out

    def _apply_offset(colors: np.ndarray, offset: tuple[float, float, float]) -> np.ndarray:
        return np.clip(colors + np.array(offset, dtype=colors.dtype), 0.0, 1.0)

    a = _copy_no_normals(pcd1)
    b = _copy_no_normals(pcd2_aligned)
    n1, n2 = len(a.points), len(b.points)
    c1 = np.asarray(a.colors) if a.has_colors() and len(a.colors) == n1 else np.tile([0.7, 0.7, 0.7], (n1, 1))
    c2 = np.asarray(b.colors) if b.has_colors() and len(b.colors) == n2 else np.tile([0.7, 0.7, 0.7], (n2, 1))
    a.colors = o3d.utility.Vector3dVector(_apply_offset(c1, red_offset))
    b.colors = o3d.utility.Vector3dVector(_apply_offset(c2, blue_offset))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720, visible=show)
    vis.add_geometry(a)
    vis.add_geometry(b)
    vis.poll_events()
    vis.update_renderer()
    if save_path is not None:
        vis.capture_screen_image(save_path, do_render=True)
    if show:
        vis.run()
    vis.destroy_window()
