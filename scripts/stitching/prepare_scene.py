"""Prepare a stitched Gaussian scene for navigation."""

import argparse
from pathlib import Path

from WorldComposer.stitching.utils.navigation import prepare_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pair", type=Path)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--clearance", type=float, default=0.2)
    args = parser.parse_args()
    prepare_scene(
        args.model, args.output_dir, args.pair, args.resolution, args.clearance
    )
    print(f"[Output] {(args.output_dir or args.model.parent) / 'scene_config.yaml'}")


if __name__ == "__main__":
    main()
