import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from plyfile import PlyData

MODEL_TO_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def model_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene_config(scene_dir):
    scene_dir = Path(scene_dir).resolve()
    config = yaml.safe_load((scene_dir / "scene_config.yaml").read_text())
    for key in ("model", "grid_map", "grid_meta"):
        config[key] = str((scene_dir / config[key]).resolve())
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if config.get("visual_scene"):
        config["visual_scene"] = str((scene_dir / config["visual_scene"]).resolve())
    rotation = np.asarray(config["model_to_world"], dtype=float)
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3))
        or not np.isclose(np.linalg.det(rotation), 1)
    ):
        raise ValueError("model_to_world must be a rotation matrix.")
    meta = json.loads(Path(config["grid_meta"]).read_text())
    if meta.get("model_sha256") != model_digest(config["model"]):
        raise ValueError(
            "The navigation map does not match the model. Run prepare_scene.py again."
        )
    if not np.allclose(meta["model_to_world"], rotation):
        raise ValueError("The navigation map uses a different coordinate system.")
    return config


def prepare_scene(
    model_path, output_dir=None, pair_path=None, resolution=0.05, clearance=0.2
):
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else model_path.parent
    if (
        not np.isfinite([resolution, clearance]).all()
        or resolution <= 0
        or clearance < 0
    ):
        raise ValueError("Use a positive map resolution and nonnegative clearance.")
    vertices = PlyData.read(model_path)["vertex"].data
    points = np.column_stack([vertices[k] for k in ("x", "y", "z")]) @ MODEL_TO_WORLD.T
    points = points[
        np.isfinite(points).all(1) & (vertices["opacity"] > np.log(0.1 / 0.9))
    ]
    if not len(points):
        raise ValueError("The model has no valid navigation points.")
    floor_points = points[np.abs(points[:, 2]) < 0.15, :2]
    obstacles = points[(points[:, 2] >= 0.15) & (points[:, 2] <= 1.6), :2]
    if not len(floor_points):
        raise ValueError("The model must have floor support near world height zero.")
    extent = (
        max(
            np.abs(floor_points).max(), np.abs(obstacles).max() if len(obstacles) else 0
        )
        * 1.1
    )
    size = max(32, int(np.ceil(2 * extent / resolution)))
    if size > 8192:
        raise ValueError(
            "The map exceeds 8192 cells per side; check model scale or increase resolution."
        )
    origin = size // 2

    def rasterize(xy):
        col = np.rint(xy[:, 0] / resolution).astype(int) + origin
        row = size - np.rint(xy[:, 1] / resolution).astype(int) - origin
        valid = (row >= 0) & (row < size) & (col >= 0) & (col < size)
        mask = np.zeros((size, size), np.uint8)
        mask[row[valid], col[valid]] = 1
        return mask

    obstacle = rasterize(obstacles)
    for width in (5, 3):
        obstacle = cv2.morphologyEx(
            obstacle, cv2.MORPH_CLOSE, np.ones((width, width), np.uint8)
        )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 15:
            obstacle[labels == index] = 0
    floor = cv2.morphologyEx(
        rasterize(floor_points), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    floor = cv2.dilate(floor, np.ones((3, 3), np.uint8))
    free = ((obstacle == 0) & (floor > 0)).astype(np.uint8)
    radius = int(np.ceil(clearance / resolution))
    free = cv2.erode(
        free,
        np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cells = np.argwhere(free > 0)
    if not len(cells):
        raise ValueError("No navigable floor remains with the requested clearance.")
    xy = np.column_stack(
        [
            (cells[:, 1] - origin) * resolution,
            (size - cells[:, 0] - origin) * resolution,
        ]
    )
    desired, heading = np.zeros(2), 0.0
    pair_path = (
        Path(pair_path) if pair_path else model_path.parent / "pair_transform.json"
    )
    if pair_path.is_file():
        pair = json.loads(pair_path.read_text())
        poses = np.asarray(pair["camera_poses"])
        centers = poses[:, :3, 3] @ MODEL_TO_WORLD.T
        desired = centers[0, :2]
        delta = centers[1, :2] - desired
        heading = float(np.degrees(np.arctan2(delta[1], delta[0])))
    spawn = xy[np.argmin(np.linalg.norm(xy - desired, axis=1))]
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "grid_map.png"), free * 255)
    meta = {
        "resolution": resolution,
        "origin_pixel": [origin, origin],
        "size": [size, size],
        "z_range": [0.15, 1.6],
        "robot_clearance_m": clearance,
        "floor_support_required": True,
        "model_sha256": model_digest(model_path),
        "model_to_world": MODEL_TO_WORLD.tolist(),
    }
    (output_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2))
    config = {
        "model": os.path.relpath(model_path, output_dir),
        "model_to_world": MODEL_TO_WORLD.tolist(),
        "opacity_threshold": 0.1,
        "grid_map": "grid_map.png",
        "grid_meta": "grid_meta.json",
        "rgb_source": "gs",
        "depth_source": "gs",
        "camera_height": 1.4,
        "robot_position": [float(spawn[0]), float(spawn[1]), 0.1],
        "robot_yaw": heading,
    }
    (output_dir / "scene_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    return config
