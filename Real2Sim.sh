#!/bin/bash

conda run -n WorldComposerR2S python source/WorldComposer/WorldComposer/real2sim/scene_assembler.py \
    --ply /home/lightwheel/Projects/WorldComposer/Assets/scenes/Marble/RealScene/RealScene01/RealScene01.ply \
    --glb /home/lightwheel/Projects/WorldComposer/Assets/scenes/Marble/RealScene/RealScene01/RealScene01.glb \
    --out_usd /home/lightwheel/Projects/WorldComposer/Assets/scenes/Marble/RealScene/RealScene01/RealScene01.usd

