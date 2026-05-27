#!/usr/bin/env bash
set -euo pipefail

# Only change this scene directory for a new Real2Sim run.
SCENE_DIR=Assets/scenes/Real2SimScenes/RealScene01

SCENE_NAME=$(basename "${SCENE_DIR}")
PLY=${SCENE_DIR}/${SCENE_NAME}.ply
GLB=${SCENE_DIR}/${SCENE_NAME}.glb
INTERMEDIATE_DIR=${SCENE_DIR}/real2sim_intermediate
ORIN_USD=${INTERMEDIATE_DIR}/${SCENE_NAME}_Orin.usd
MESH_USD=${INTERMEDIATE_DIR}/${SCENE_NAME}_mesh.usd
PHASE1_DIR=${INTERMEDIATE_DIR}
FLATTENED_MESH_USD=${INTERMEDIATE_DIR}/${SCENE_NAME}_mesh_flattened.usd
READY_USD=${SCENE_DIR}/${SCENE_NAME}.usd

mkdir -p "${INTERMEDIATE_DIR}"

# Phase 1: convert reconstruction assets and assemble the aligned scene USD.
python source/WorldComposer/WorldComposer/real2sim/scene_assembler.py \
    --ply "${PLY}" \
    --glb "${GLB}" \
    --out_usd "${ORIN_USD}" \
    --mesh_usd "${MESH_USD}"

# Phase 2: recognize the tabletop. Release mode keeps only reference overlay,
# table point clouds, and summary metadata needed by Phase 3.
python source/WorldComposer/WorldComposer/real2sim/table_detection.py \
    --scene_usd "${ORIN_USD}" \
    --output_dir "${PHASE1_DIR}" \
    --pitch_deg -30 \
    --yaw_step_deg 30 \
    --sam_prompt tabletop \
    --headless

# Phase 3: fit the tabletop plane, flatten selected mesh vertices, and normalize the whole scene pose.
python source/WorldComposer/WorldComposer/real2sim/scene_ready_builder.py \
    --scene_usd "${ORIN_USD}" \
    --phase1_output_dir "${PHASE1_DIR}" \
    --mesh_usd "${MESH_USD}" \
    --flattened_mesh_usd "${FLATTENED_MESH_USD}" \
    --ready_usd "${READY_USD}" \
    --output_dir "${INTERMEDIATE_DIR}"

echo "[Real2Sim] Intermediate artifacts: ${INTERMEDIATE_DIR}"
echo "[Real2Sim] Ready scene: ${READY_USD}"
