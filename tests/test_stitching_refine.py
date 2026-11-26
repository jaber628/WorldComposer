import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("gsplat"), "GSplat is required")
class RefinementTests(unittest.TestCase):
    def test_refinement_preserves_inputs_and_protected_regions(self):
        import cv2
        import torch

        from WorldComposer.stitching.utils.gaussian import merge_gaussian_plys
        from WorldComposer.stitching.utils.refine import (
            _read,
            _write,
            _refine_capture,
            refine_gaussian_pair,
        )

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        fields = ["x", "y", "z", "nx", "ny", "nz"]
        fields += [f"f_dc_{i}" for i in range(3)]
        fields += ["opacity"] + [f"scale_{i}" for i in range(3)]
        fields += [f"rot_{i}" for i in range(4)]
        x, y = np.meshgrid(np.linspace(-0.8, 0.8, 17), np.linspace(-0.5, 0.5, 11))
        vertices = np.zeros(x.size, dtype=[(name, "f4") for name in fields])
        vertices["x"], vertices["y"], vertices["z"] = x.ravel(), y.ravel(), -2
        vertices["rot_0"] = 1
        vertices["opacity"] = np.log(0.8 / 0.2)
        for i in range(3):
            vertices[f"scale_{i}"] = np.log(0.04)
        transform = np.eye(4)
        transform[0, 3] = 1
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scenes").mkdir()
            (root / "pano").mkdir()
            for name in ("left", "right"):
                _write(root / "scenes" / f"{name}.ply", vertices)
                cv2.imwrite(
                    str(root / "pano" / f"{name}.png"),
                    np.full((128, 256, 3), 128, np.uint8),
                )
            baseline = root / "merged.ply"
            merge_gaussian_plys(
                root / "scenes/left.ply", root / "scenes/right.ply", transform, baseline
            )
            pair = root / "pair.json"
            pair.write_text(
                json.dumps(
                    {
                        "target_scene": "left",
                        "source_scene": "right",
                        "transform_source_to_target": transform.tolist(),
                        "camera_poses": [np.eye(4).tolist(), transform.tolist()],
                    }
                )
            )
            paths = [pair, baseline, *sorted((root / "scenes").glob("*.ply"))]
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            for geometry in (False, True):
                with self.subTest(geometry=geometry):
                    output = root / f"refined-{geometry}"
                    report = _refine_capture(
                        root,
                        pair,
                        baseline,
                        output,
                        steps=8,
                        width=64,
                        refine_geometry=geometry,
                    )
                    self.assertGreater(report["protected_count"], 0)
                    self.assertTrue(report["protected_exact"])
                    self.assertTrue(
                        all(np.isfinite(row["loss"]) for row in report["history"])
                    )
                    provenance = np.load(output / "provenance.npz")
                    original = _read(baseline)[provenance["original_indices"]]
                    refined = _read(output / "merged.ply")
                    np.testing.assert_array_equal(
                        refined[provenance["protected"]],
                        original[provenance["protected"]],
                    )
                    for name in ("x", "y", "z"):
                        self.assertLessEqual(
                            float(np.abs(refined[name] - original[name]).max()),
                            0.03001 if geometry else 0,
                        )
                    actual = json.loads((output / "pair_transform.json").read_text())
                    self.assertEqual(
                        actual["camera_poses"],
                        json.loads(pair.read_text())["camera_poses"],
                    )
            output = root / "complete"
            report = refine_gaussian_pair(
                root,
                pair,
                baseline,
                output,
                steps=3,
                width=64,
                final_width=64,
            )
            self.assertEqual(set(report["stages"]), {"capture", "navigation", "detail"})
            self.assertTrue(report["stages"]["detail"]["frozen_exact"])
            self.assertTrue(np.isfinite(_read(output / "merged.ply")["opacity"]).all())
            self.assertEqual(
                report["model_sha256"],
                hashlib.sha256((output / "merged.ply").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                before,
                [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
            )


if __name__ == "__main__":
    unittest.main()
