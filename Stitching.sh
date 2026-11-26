#!/usr/bin/env bash
set -euo pipefail

SCENE_DIR=${1:-Assets/scenes/MultiroomScenes/MultiRoom01}
OUTPUT_DIR=${2:-outputs/stitching}

python scripts/stitching/estimate_pair.py \
    --scene-dir "${SCENE_DIR}" --output-dir "${OUTPUT_DIR}/aligned"
python scripts/stitching/refine_pair.py \
    --input-dir "${SCENE_DIR}/input" \
    --pair "${OUTPUT_DIR}/aligned/pair_transform.json" \
    --output-dir "${OUTPUT_DIR}/stitched"
python scripts/stitching/prepare_scene.py \
    --model "${OUTPUT_DIR}/stitched/merged.ply"
