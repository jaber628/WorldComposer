import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from WorldComposer.stitching.utils.gaussian import merge_gaussian_plys
from WorldComposer.stitching.utils.geometry import (
    estimate_yaw_only_pose,
    triangulate_point,
)
from WorldComposer.stitching.utils.icp_utils import run_colored_icp_from_pcds
from WorldComposer.stitching.utils.panorama import fit_camera_bearings
from WorldComposer.stitching.utils.surface_fusion import (
    _cube_rotations,
    _observe,
    surface_ownership,
)


class PoseTests(unittest.TestCase):
    def test_pose_with_outliers_on_full_sphere(self):
        for seed, yaw, translation, vertical in [
            (7, 47.3, [0.6, 0, -0.8], False),
            (19, -128.6, [-0.4, 0, 0.9165], False),
            (31, 83, [0.6, 0.2, -0.8], True),
        ]:
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                points = rng.normal(size=(100, 3))
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                points *= rng.uniform(3, 8, (100, 1))
                rotation = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
                translation = np.asarray(translation)
                moved = points @ rotation.T + translation
                first = points / np.linalg.norm(points, axis=1, keepdims=True)
                second = moved / np.linalg.norm(moved, axis=1, keepdims=True)
                second += rng.normal(0, 0.0005, second.shape)
                second /= np.linalg.norm(second, axis=1, keepdims=True)
                second[:35] = rng.normal(size=(35, 3))
                second[:35] /= np.linalg.norm(second[:35], axis=1, keepdims=True)
                estimated, direction, inliers = estimate_yaw_only_pose(
                    first, second, constrain_y_translation=not vertical
                )
                error = Rotation.from_matrix(estimated @ rotation.T).magnitude()
                direction_error = np.arccos(
                    np.clip(
                        direction @ translation / np.linalg.norm(translation), -1, 1
                    )
                )
                self.assertLess(error, np.deg2rad(0.5))
                self.assertLess(direction_error, np.deg2rad(2))
                self.assertGreaterEqual(int(inliers[35:].sum()), 60)
                self.assertLessEqual(int(inliers[:35].sum()), 2)

    def test_triangulation_in_front_and_behind(self):
        rng = np.random.default_rng(5)
        points = rng.normal(size=(30, 3)) * 3
        rotation = Rotation.from_euler("y", -73, degrees=True).as_matrix()
        translation = np.array([0.6, 0, 0.8])
        for point in points:
            moved = rotation @ point + translation
            actual = triangulate_point(
                point / np.linalg.norm(point),
                moved / np.linalg.norm(moved),
                rotation,
                translation,
            )
            np.testing.assert_allclose(actual, point, atol=1e-8)


class ICPTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D is required")
    def test_structural_sampling_retains_broad_supported_planes(self):
        from WorldComposer.stitching.utils.load import load_pcd_from_ply

        x, z = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))
        points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
        vertices = GaussianTests.vertices(points, scale=0.08)
        vertices["scale_1"] = np.log(0.003)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "surface.ply"
            GaussianTests.save(path, vertices)
            old = load_pcd_from_ply(str(path), voxel_size=0.04)
            structural = load_pcd_from_ply(str(path), voxel_size=0.04, structural=True)
        self.assertEqual(len(old.points), 0)
        self.assertGreater(len(structural.points), 300)
        self.assertTrue(structural.has_normals())

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "Open3D is required")
    def test_multiscale_registration_recovers_known_pose(self):
        import open3d as o3d

        rng = np.random.default_rng(11)
        u, v = np.meshgrid(np.linspace(-1, 1, 35), np.linspace(-1, 1, 35))
        points = np.concatenate(
            [
                np.column_stack([u.ravel(), np.zeros(u.size), v.ravel()]),
                np.column_stack([np.full(u.size, -1), u.ravel() + 1, v.ravel()]),
                np.column_stack([u.ravel(), v.ravel() + 1, np.full(u.size, -1)]),
            ]
        )
        colors = 0.5 + 0.25 * np.sin(points * [2, 3, 4])
        rotation = Rotation.from_euler("y", 17, degrees=True).as_matrix()
        translation = np.array([0.1, 0.02, -0.15])
        source_points = (points - translation) @ rotation + rng.normal(
            0, 0.001, points.shape
        )
        target, source = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        target.points, source.points = o3d.utility.Vector3dVector(
            points
        ), o3d.utility.Vector3dVector(source_points)
        target.colors = source.colors = o3d.utility.Vector3dVector(colors)
        initial = Rotation.from_euler("y", 16, degrees=True).as_matrix()
        transform, _ = run_colored_icp_from_pcds(
            target, source, initial, translation + [0.02, 0.01, 0.02], voxel_size=0.02
        )
        error = Rotation.from_matrix(transform[:3, :3] @ rotation.T).magnitude()
        self.assertLess(error, np.deg2rad(0.5))
        self.assertLess(float(np.linalg.norm(transform[:3, 3] - translation)), 0.01)
        robust, _ = run_colored_icp_from_pcds(
            target,
            source,
            initial,
            translation + [0.02, 0.01, 0.02],
            voxel_size=0.02,
            robust=True,
        )
        self.assertLess(
            Rotation.from_matrix(robust[:3, :3] @ rotation.T).magnitude(),
            np.deg2rad(0.5),
        )
        self.assertLess(np.linalg.norm(robust[:3, 3] - translation), 0.01)


class PanoramaCameraTests(unittest.TestCase):
    def test_camera_pose_from_noisy_bearings_and_outliers(self):
        rng = np.random.default_rng(22)
        points = rng.uniform([-3, -0.5, -3], [3, 4, 3], (200, 3))
        position = np.array([0.03, 2.2, -0.02])
        rotation = Rotation.from_euler("y", 1.2, degrees=True).as_matrix()
        bearings = (points - position) @ rotation
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        bearings += rng.normal(0, 0.0002, bearings.shape)
        bearings[:40] = rng.normal(size=(40, 3))
        bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
        pose, inliers, _ = fit_camera_bearings(points, bearings, [0, 2.4, 0])
        self.assertLess(np.linalg.norm(pose[:3, 3] - position), 0.01)
        self.assertLess(
            Rotation.from_matrix(pose[:3, :3] @ rotation.T).magnitude(), np.deg2rad(0.1)
        )
        self.assertGreaterEqual(int(inliers[40:].sum()), 155)
        self.assertLessEqual(int(inliers[:40].sum()), 1)


class SurfaceFusionTests(unittest.TestCase):
    def test_visibility_distinguishes_foreground_and_occluded_points(self):
        observation = {
            "pose": np.eye(4),
            "rotations": _cube_rotations(),
            "size": 16,
            "faces": [
                (np.full((16, 16), 2.0), np.ones((16, 16)), np.ones((16, 16)))
                for _ in range(6)
            ],
        }
        _, visible, foreground, _ = _observe(
            np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]]), observation
        )
        np.testing.assert_array_equal(visible, [False, True, False])
        np.testing.assert_array_equal(foreground, [True, True, False])

    def test_static_surface_selection_preserves_coverage_and_separate_planes(self):
        x, z = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))
        points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
        cameras = [np.eye(4), np.eye(4)]
        cameras[0][:3, 3], cameras[1][:3, 3] = [-1, 1.5, 0], [1, 1.5, 0]

        def observe(points, camera):
            distance = np.linalg.norm(points - camera[:3, 3], axis=1)
            return (
                distance,
                np.ones(len(points), bool),
                np.zeros(len(points), bool),
                np.ones(len(points)),
            )

        for separation in (0.0, 0.08):
            vertices = [
                GaussianTests.vertices(points),
                GaussianTests.vertices(points + [0, separation, 0]),
            ]
            for v in vertices:
                v["scale_1"] = np.log(0.003)
            masks = [np.ones(len(v), bool) for v in vertices]
            with patch(
                "WorldComposer.stitching.utils.surface_fusion._capture",
                side_effect=lambda v, m, p, i, d: p,
            ), patch(
                "WorldComposer.stitching.utils.surface_fusion._observe",
                side_effect=observe,
            ):
                selected, report = surface_ownership(vertices, masks, cameras)
                repeated, _ = surface_ownership(vertices, masks, cameras)
            for first, second in zip(selected, repeated):
                np.testing.assert_array_equal(first, second)
            kept = np.concatenate(
                [
                    np.column_stack([v[k][m] for k in ("x", "y", "z")])
                    for v, m in zip(vertices, selected)
                ]
            )
            expected = np.concatenate([points, points + [0, separation, 0]])
            distance, _ = cKDTree(kept).query(expected)
            self.assertLess(distance.max(), 1e-6)
            self.assertEqual(len(kept), len(points) * (1 if separation == 0 else 2))
            self.assertTrue(report["static_ownership"])


class GaussianTests(unittest.TestCase):
    @staticmethod
    def vertices(points, scale=0.02):
        names = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        vertices = np.zeros(len(points), dtype=[(name, "f4") for name in names])
        for index, name in enumerate(("x", "y", "z")):
            vertices[name] = np.asarray(points)[:, index]
        vertices["opacity"] = np.log(4)
        vertices["rot_0"] = 1
        for name in ("scale_0", "scale_1", "scale_2"):
            vertices[name] = np.log(scale)
        return vertices

    @staticmethod
    def save(path, vertices):
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)

    def test_transform_preserves_appearance_and_rotates_covariance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            target = self.vertices([[-1, 0, 0]])
            source = self.vertices([[1, 0, 2]])
            source["opacity"] = np.inf
            source["f_dc_0"] = 0.3
            source["scale_0"], source["scale_1"], source["scale_2"] = (
                np.log(0.01),
                np.log(0.02),
                np.log(0.04),
            )
            self.save(folder / "target.ply", target)
            self.save(folder / "source.ply", source)
            transform = np.eye(4)
            transform[:3, :3] = (
                Rotation.from_euler("y", 90, degrees=True).as_matrix() * 1.2
            )
            transform[:3, 3] = [0.3, 0.2, -0.4]
            merge_gaussian_plys(
                folder / "target.ply",
                folder / "source.ply",
                transform,
                folder / "merged.ply",
            )
            output = PlyData.read(folder / "merged.ply")["vertex"].data
            np.testing.assert_array_equal(output[:1], target)
            np.testing.assert_allclose(
                [output[1][name] for name in ("x", "y", "z")],
                [2.7, 0.2, -1.6],
                atol=1e-6,
            )
            self.assertTrue(np.isposinf(output[1]["opacity"]))
            self.assertEqual(output[1]["f_dc_0"], source[0]["f_dc_0"])
            quaternion = np.array([output[1][f"rot_{i}"] for i in (1, 2, 3, 0)])
            rotation = Rotation.from_quat(quaternion).as_matrix()
            scales = np.exp([output[1][f"scale_{i}"] for i in range(3)])
            covariance = rotation @ np.diag(scales**2) @ rotation.T
            np.testing.assert_allclose(
                covariance,
                np.diag(np.array([0.04, 0.02, 0.01]) ** 2 * 1.2**2),
                atol=1e-9,
            )

    def test_overlap_cleanup_keeps_surface_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, z = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-1, 1, 20))
            points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
            target = self.vertices(points)
            source = self.vertices(np.concatenate([points - [1, 0, 0], [[10, 10, 10]]]))
            for name in ("scale_0", "scale_1", "scale_2"):
                source[name][-1] = 0
            self.save(folder / "target.ply", target)
            self.save(folder / "source.ply", source)
            transform = np.eye(4)
            transform[0, 3] = 1
            report = merge_gaussian_plys(
                folder / "target.ply",
                folder / "source.ply",
                transform,
                folder / "merged.ply",
                clean_overlap=True,
            )
            output = PlyData.read(folder / "merged.ply")["vertex"].data
            merged_points = np.column_stack([output[name] for name in ("x", "y", "z")])
            self.assertEqual(len(output), len(points))
            self.assertEqual(report["quality_counts"], [400, 400])
            distance, _ = cKDTree(merged_points).query(points)
            self.assertLess(float(distance.max()), 1e-6)

    def test_large_planar_gaussian_survives_at_different_scales(self):
        for factor in (1, 10):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                x, y = np.meshgrid(
                    np.linspace(-0.4, 0.4, 17), np.linspace(-0.4, 0.4, 17)
                )
                points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
                target = self.vertices(
                    np.concatenate([points, [[0.012, 0.017, 0], [0.012, 0.017, 0.1]]])
                )
                target["scale_2"] = np.log(0.003)
                target["scale_0"][-2], target["scale_1"][-2] = np.log(0.6), np.log(0.4)
                for name in ("scale_0", "scale_1", "scale_2"):
                    target[name][-1] = 0
                source = self.vertices([[20, 0, 20]])
                for vertices in (target, source):
                    for name in ("x", "y", "z"):
                        vertices[name] *= factor
                    for name in ("scale_0", "scale_1", "scale_2"):
                        vertices[name] += np.log(factor)
                self.save(folder / "target.ply", target)
                self.save(folder / "source.ply", source)
                report = merge_gaussian_plys(
                    folder / "target.ply",
                    folder / "source.ply",
                    np.eye(4),
                    folder / "merged.ply",
                    clean_overlap=True,
                    camera_height=1.5 * factor,
                    overlap_distance=0.15 * factor,
                )
                output = PlyData.read(folder / "merged.ply")["vertex"].data
                largest = np.exp(
                    np.column_stack([output[f"scale_{i}"] for i in range(3)])
                ).max(axis=1)
                self.assertEqual(int(np.sum(largest > 0.3 * factor)), 1)
                self.assertEqual(report["quality_counts"], [len(points) + 1, 1])

    def test_close_parallel_surfaces_are_not_duplicates(self):
        self.check_distinct_surfaces("parallel")

    def test_perpendicular_surfaces_are_not_duplicates(self):
        self.check_distinct_surfaces("perpendicular")

    def test_conflicting_colors_keep_uncertain_layers(self):
        self.check_distinct_surfaces("color")

    def test_coincident_surfaces_merge_despite_exposure_difference(self):
        self.check_distinct_surfaces("exposure")

    def check_distinct_surfaces(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            x, z = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-1, 1, 20))
            points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
            target = self.vertices(points)
            target["scale_1"] = np.log(0.003)
            other = points.copy()
            if mode == "parallel":
                other[:, 1] = 0.08
            elif mode == "color":
                other[:, 1] = 0.02
            elif mode == "perpendicular":
                other[:, 1], other[:, 2] = other[:, 2].copy(), 0
            source = self.vertices(other - [1, 0, 0])
            source["scale_2" if mode == "perpendicular" else "scale_1"] = np.log(0.003)
            if mode in ("color", "exposure"):
                for name in ("f_dc_0", "f_dc_1", "f_dc_2"):
                    source[name] = 2
            self.save(folder / "target.ply", target)
            self.save(folder / "source.ply", source)
            transform = np.eye(4)
            transform[0, 3] = 1
            report = merge_gaussian_plys(
                folder / "target.ply",
                folder / "source.ply",
                transform,
                folder / "merged.ply",
                clean_overlap=True,
            )
            self.assertEqual(
                report["output_count"],
                len(points) if mode == "exposure" else 2 * len(points),
            )


if __name__ == "__main__":
    unittest.main()
