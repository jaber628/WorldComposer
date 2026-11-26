"""Refine a stitched Gaussian scene using its original views."""

import argparse
from pathlib import Path

from WorldComposer.stitching.utils.refine import refine_gaussian_pair
from WorldComposer.utils.constant import ASSETS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(ASSETS_ROOT) / "scenes/MultiroomScenes/MultiRoom01/input",
    )
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--final-width", type=int, default=960)
    parser.add_argument("--navigation-height", type=float, default=1.5)
    args = parser.parse_args()
    refine_gaussian_pair(
        args.input_dir,
        args.pair,
        args.model or args.pair.parent / "merged.ply",
        args.output_dir,
        args.steps,
        args.width,
        args.final_width,
        args.navigation_height,
    )


if __name__ == "__main__":
    main()
