#!/usr/bin/env bash
set -euo pipefail

SCENE_DIR=Assets/scenes/MultiroomScenes/MultiRoom01

python scripts/stitching/estimate_pair.py \
    --scene-dir "${SCENE_DIR}" \
    "$@"
