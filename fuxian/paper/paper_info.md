# Paper Info

## Title

**From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation** (WorldComposer)

## Authors

Jasper Lu, Zhenhao Shen, Yuanfei Wang, Shugao Liu, Shengqiang Xu, Shawn Xie, Jingkai Xu, Feng Jiang, Jade Yang, Chen Xie, Ruihai Wu

## Links

| Resource | URL |
|----------|-----|
| arXiv | https://arxiv.org/abs/2604.15805 |
| PDF (project) | https://stubborn111.github.io/WorldComposer/Assets/WorldComposer.pdf |
| Project page | https://stubborn111.github.io/WorldComposer/ |
| Official code | https://github.com/jaber628/WorldComposer |
| Assets (HF) | https://huggingface.co/datasets/Jaber628/WorldComposer_Assets |

## Repo correspondence

This repository implements (released portions):

1. **Real2Sim** — real capture → Marble (external) → `.ply`/`.glb` → USD scene with tabletop rectification
2. **Simulation** — Isaac Sim / Isaac Lab task (`Task01_Tableware`), teleoperation
3. **Planned / not fully released** — AutoCollection, full sim env, training/eval, multiroom stitching (per README update list)

## Input modality (released pipeline)

- Primary: **single panoramic / real-world capture** lifted by **Marble 1.1** to Gaussian point cloud + textured mesh
- Not in-repo: Marble itself; user provides `SceneName.ply` + `SceneName.glb`

## Output (released)

- Simulation-ready USD: `${SCENE_DIR}/RealScene01.usd` (example naming)
- Task-level robot demos / teleop on composed digital-cousin scenes (with HF assets)

## User research question (for section 8)

**如何从一张图片/一段视频出发，生成机器人操作任务中不同场景—物体对的操作数据，用于训练泛化性更好的操作模型？**

WorldComposer is directly adjacent: seeing → sim-ready scenes → (planned) auto manipulation data; current release emphasizes Real2Sim + one tabletop task, not full cross scene–object pair data factory yet.