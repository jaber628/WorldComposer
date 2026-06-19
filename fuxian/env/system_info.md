# System Info (capture time: 2026-06-15)

## GPU

- **Model**: NVIDIA GeForce RTX 3090
- **VRAM**: 24576 MiB total (~1483 MiB in use at capture)
- **Driver**: 580.159.03
- **CUDA (driver report)**: 13.0

## Memory

- **RAM**: 62 Gi total, ~43 Gi available
- **Swap**: 61 Gi

## Disk (project partition)

- **/** : 873G total, 297G avail (65% used)

## Python / conda

- **python3**: 3.10.12 (system default at capture)
- **conda**: 26.3.2
- **README recommends**: `WorldComposer` env with Python 3.11 + Isaac Sim + torch 2.7 cu128

## Implication for reproduction

- Single RTX 3090 is **likely sufficient** for Real2Sim headless stages and single-env teleop/autocollection per Isaac Lab norms, but **Isaac Sim install is heavy** (disk + first-run ext cache).
- Full training pipeline **not in repo**; no multi-GPU requirement from this codebase today.