<div align="center">
  <h1>WorldComposer</h1>
  <h3><em>From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation</em></h3>
</div>

<div align="center">
    <a href="https://stubborn111.github.io/WorldComposer/Paper/WorldComposer.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Paper-PDF-2ea043?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper PDF"></a>
    <a href="https://stubborn111.github.io/WorldComposer/" target="_blank">
    <img src="https://img.shields.io/badge/Page-Project_Page-e5534b?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
    <a href="https://github.com/jaber628/WorldComposer" target="_blank">
    <img src="https://img.shields.io/badge/Code-GitHub-1f6feb?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Code"></a>
</div>

<div align="center">
  <img src="Pic/Teaser.png" width="94%" alt="WorldComposer teaser"/>
</div>

<div align="center">
  <img src="Pic/Pipeline.png" width="94%" alt="WorldComposer pipeline"/>
</div>

## ✅ **Milestone**

<div align="center">
  <img src="https://img.shields.io/badge/Status-Official_Release-2ea043?style=flat-square&logo=checkmarx&logoColor=white" alt="Official Release" />
  <img src="https://img.shields.io/badge/Track-Simulation_Environment-7c3aed?style=flat-square&logo=unity&logoColor=white" alt="Simulation Environment" />
  <img src="https://img.shields.io/badge/Track-Real2Sim-ff8c00?style=flat-square&logo=opengl&logoColor=white" alt="Real2Sim" />
</div>

**2026.04.17 — WorldComposer Simulation Environment and Real2Sim pipeline release.**

## ✦ **Update List**

<div align="center">
  <img src="https://img.shields.io/badge/Simulation_Environment-Released-2ea043?style=flat-square" alt="Simulation Environment Released" />
  <img src="https://img.shields.io/badge/Real2Sim-Released-1f6feb?style=flat-square" alt="Real2Sim Released" />
  <img src="https://img.shields.io/badge/AutoCollection-In_Progress-f59e0b?style=flat-square" alt="AutoCollection In Progress" />
  <img src="https://img.shields.io/badge/Train_%26_Eval-In_Progress-db2777?style=flat-square" alt="Train and Eval In Progress" />
  <img src="https://img.shields.io/badge/Multiroom_Stitching-In_Progress-9333ea?style=flat-square" alt="Multiroom Stitching In Progress" />
</div>

- **Simulation Environment** — released.
- **Real2Sim Pipeline** — released.
- **AutoCollection Pipeline** — release in progress.
- **Training and Evaluation Environment** — release in progress.
- **Multiroom Stitching** — release in progress.

## ✦ **Simulation Environment**

### ◆ **Installation**

```bash
conda create -n WorldComposer python=3.11
conda activate WorldComposer

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[all]==0.4.1"
pip install --upgrade pip
pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com

cd /home/lightwheel/Projects
git clone <WORLD_COMPOSER_REPO_URL>
cd WorldComposer
python -m pip install -e source/WorldComposer
```

### ◆ **Task Code Structure**

Public tasks are organized under `source/WorldComposer/WorldComposer/tasks/`.

- `Task01_Tableware/`
  - `Tableware.py`: task environment logic, reset behavior, randomization, observations, and success checking.
  - `Tableware_cfg.py`: Isaac Lab environment, robot, object, scene, and camera configuration.
  - `task_config.yaml`: teleoperation and data-collection runtime configuration.
  - `__init__.py`: gym task registration.

Each task follows the same structure: an environment implementation file, a static configuration file, a runtime YAML config, and a registration entry.

### ◆ **Teleoperation**

Use `Teleoperation.sh` to launch the released teleoperation entry.

```bash
bash Teleoperation.sh
```

### ◆ **AutoCollection**

AutoCollection is the automatic data generation pipeline built on top of the same task interface and will be released progressively.

## ✦ **Real2Sim Pipeline**

### ◆ **Overview**

WorldComposer Real2Sim is a complete automated pipeline for point-cloud and mesh alignment, operation-region recognition, operation-surface rectification, and scene-scale / pose alignment, turning real captured scenes into simulation-ready assets.

### ◆ **Installation**

```bash
cd /home/lightwheel/Projects/WorldComposer
mkdir -p 3rd/cache/huggingface 3rd/cache/torch 3rd/cache/xdg

cd 3rd
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
git clone https://github.com/luca-medeiros/lang-segment-anything.git

cd /home/lightwheel/Projects/WorldComposer/3rd/3dgrut
CUDA_VERSION=12 ./scripts/create_conda.sh WorldComposerR2S
conda activate WorldComposerR2S
CUDA_VERSION=12.8.1 ./install_env.sh WorldComposerR2S WITH_GCC11

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
pip install 'isaacsim[all,extscache]' --extra-index-url https://pypi.nvidia.com
pip install open3d trimesh opencv-python segment-anything
pip install -e /home/lightwheel/Projects/WorldComposer/3rd/lang-segment-anything

cd /home/lightwheel/Projects/WorldComposer
python -m pip install -e source/WorldComposer
```

### ◆ **Released Entry**

The current public release exposes `source/WorldComposer/WorldComposer/real2sim/scene_assembler.py` and `Real2Sim.sh` for the scene assembly stage.

### ◆ **Run**

```bash
bash Real2Sim.sh
```


### ◆ **Current Public Scope**

The current release covers PLY-to-USDZ conversion, GLB-to-USD mesh conversion, and composed simulation scene generation.
