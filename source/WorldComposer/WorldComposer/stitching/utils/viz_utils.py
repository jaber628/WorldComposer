import numpy as np


def depth_to_u8(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    if d.max() > 1.0:
        d = (d - d.min()) / max(1e-6, d.max() - d.min())
    return (np.clip(d, 0.0, 1.0) * 255).astype(np.uint8)
