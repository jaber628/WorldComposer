#!/bin/bash

TASK_NAME="WorldComposer-Tableware-v0"
CONFIG_PATH="source/WorldComposer/WorldComposer/tasks/Task01_Tableware/task_config.yaml"

python scripts/teleoperation/teleop_record.py \
    --task="$TASK_NAME" \
    --config="$CONFIG_PATH"
