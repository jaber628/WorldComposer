"""Estimate a scene-pair transform from panorama matches and Colored ICP."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from WorldComposer.stitching.utils.feature_align import run_panorama_matching_and_ground
from WorldComposer.stitching.utils.geometry import R_180_Y, pano_pose_to_connection_convention, rotation_matrix_to_euler
from WorldComposer.stitching.utils.icp_utils import run_colored_icp_from_pcds
from WorldComposer.stitching.utils.io_utils import save_json
from WorldComposer.stitching.utils.load import load_pcd_from_ply
from WorldComposer.stitching.utils.visualize import show_image, visualize_icp_result, visualize_panorama_matches_and_ground
from WorldComposer.utils.constant import ASSETS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene1", nargs="?", default="scene_01")
    parser.add_argument("scene2", nargs="?", default="scene_02")
    parser.add_argument("--scene-dir", type=Path, default=Path(ASSETS_ROOT) / "scenes/MultiroomScenes/MultiRoom01")
    parser.add_argument("--pano-dir", type=Path)
    parser.add_argument("--gs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cam-height", type=float, default=1.5)
    parser.add_argument("--ground-zone-bottom-ratio", type=float, default=0.6)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-debug-images", action="store_true")
    args = parser.parse_args()
    np.random.seed(42)
    pano_dir = args.pano_dir or args.scene_dir / "input/pano"
    gs_dir = args.gs_dir or args.scene_dir / "input/scenes"
    output_dir = args.output_dir or args.scene_dir / "stitching_intermediate" / f"{args.scene1}_{args.scene2}"
    data = run_panorama_matching_and_ground(
        str(pano_dir / f"{args.scene1}.png"), str(pano_dir / f"{args.scene2}.png"),
        cam_height=args.cam_height, ground_zone_bottom_ratio=args.ground_zone_bottom_ratio,
    )
    rotation, translation = pano_pose_to_connection_convention(data["R"] @ R_180_Y, -data["t"] * data["scale"])
    pcd1 = load_pcd_from_ply(str(gs_dir / f"{args.scene1}.ply"), voxel_size=args.voxel_size)
    pcd2 = load_pcd_from_ply(str(gs_dir / f"{args.scene2}.ply"), voxel_size=args.voxel_size)
    transform, info = run_colored_icp_from_pcds(pcd1, pcd2, rotation, translation, voxel_size=args.voxel_size)
    position = np.asarray(transform[:3, 3])
    euler = rotation_matrix_to_euler(np.asarray(transform[:3, :3]))
    result = {
        "source_scene": args.scene2, "target_scene": args.scene1,
        "transform_source_to_target": np.asarray(transform).tolist(),
        "position_xyz": position.tolist(), "rotation_xyz_deg": euler.tolist(),
        "translation_unit": "m", "ground_scale": float(data["scale"]),
        "panorama_inliers": int(np.sum(data["inliers"])),
        "fitness": info["fitness"], "inlier_rmse": info["rmse"],
    }
    save_json(result, output_dir / "pair_transform.json")
    print(f"[SuperSplat] position_xyz={tuple(position)}, rotation_xyz_deg={tuple(euler)}")
    print(f"[ICP] fitness={info['fitness']:.4f}, inlier_rmse={info['rmse']:.4f}")
    print(f"[Output] {output_dir / 'pair_transform.json'}")
    if args.visualize or args.save_debug_images:
        matches = visualize_panorama_matches_and_ground(
            data["pano1"], data["pano2"], data["pts1"], data["pts2"], data["inliers"], data["ground_match_mask"],
            max_draw_all=400, max_draw_ground=200,
        )
        if args.save_debug_images:
            cv2.imwrite(str(output_dir / "matches.png"), matches)
        if args.visualize:
            show_image(matches, "Panorama matches", max_width=1400, max_height=720)
            visualize_icp_result(pcd1, pcd2, window_name="Raw point clouds")
            coarse = np.eye(4)
            coarse[:3, :3], coarse[:3, 3] = rotation, translation
            visualize_icp_result(pcd1, o3d.geometry.PointCloud(pcd2).transform(coarse), window_name="Coarse alignment")
            visualize_icp_result(pcd1, o3d.geometry.PointCloud(pcd2).transform(transform), window_name="ICP alignment")


if __name__ == "__main__":
    main()
