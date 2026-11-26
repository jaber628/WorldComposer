import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from WorldComposer.stitching.utils.navigation import load_scene_config, prepare_scene


class NavigationTests(unittest.TestCase):
    def test_map_requires_floor_clearance_and_matching_model(self):
        x, z = np.meshgrid(np.arange(-2, 2, 0.04), np.arange(-2, 2, 0.04))
        floor = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
        y, z = np.meshgrid(np.arange(0.2, 1.4, 0.04), np.arange(-2, 2, 0.04))
        wall = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
        xyz = np.concatenate([floor, wall])
        vertices = np.zeros(
            len(xyz), dtype=[(k, "f4") for k in ("x", "y", "z", "opacity")]
        )
        for i, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, i]
        vertices["opacity"] = 2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "merged.ply"
            PlyData([PlyElement.describe(vertices, "vertex")]).write(model)
            prepare_scene(model)
            config = load_scene_config(root)
            self.assertEqual(Path(config["model"]), model)
            self.assertGreaterEqual(abs(config["robot_position"][0]), 0.2)
            grid = cv2.imread(config["grid_map"], 0)
            meta = json.loads(Path(config["grid_meta"]).read_text())
            origin = meta["origin_pixel"][0]
            self.assertEqual(grid[grid.shape[0] - origin, origin], 0)
            self.assertEqual(grid[0, 0], 0)
            with model.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_scene_config(root)


if __name__ == "__main__":
    unittest.main()
