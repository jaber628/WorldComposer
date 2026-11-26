import hashlib
import json
import time
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from numpy.lib.recfunctions import repack_fields
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .gaussian import _positions, _voxel_indices, merge_gaussian_plys
from .panorama import _view

C0 = 0.5 / np.sqrt(np.pi)


def _read(path):
    return PlyData.read(path)["vertex"].data.copy()


def _write(path, vertices):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")], text=False, byte_order="<"
    ).write(path)


def _tensors(vertices, device="cuda:0"):
    fields = lambda names: torch.as_tensor(
        np.column_stack([vertices[n] for n in names]), device=device
    ).float()
    return (
        torch.as_tensor(_positions(vertices), device=device).float(),
        fields([f"rot_{i}" for i in range(4)]),
        fields([f"scale_{i}" for i in range(3)]).exp(),
        torch.as_tensor(vertices["opacity"].copy(), device=device).sigmoid(),
        (fields([f"f_dc_{i}" for i in range(3)]) * C0 + 0.5).clamp(0, 1),
    )


def _orientation(yaw, pitch=0):
    (yaw, pitch) = np.deg2rad([yaw, pitch])
    forward = np.array(
        [np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)]
    )
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def _camera(parent, yaw, pitch=0, offset=0):
    pose = parent.copy()
    pose[:3, 3] += parent[:3, :3] @ np.array([offset, 0, 0])
    pose[:3, :3] = parent[:3, :3] @ _orientation(yaw, pitch)
    return pose


def _render(
    parameters, pose, width=640, height=480, alpha=None, color=None, depth=False
):
    (means, quats, scales, opacity, colors) = parameters
    device = means.device
    k = torch.tensor(
        [[width / 2, 0, width / 2], [0, width / 2, height / 2], [0, 0, 1]],
        device=device,
    )[None]
    view = torch.as_tensor(np.linalg.inv(pose).astype("f4"), device=device)[None]
    (rgb, accumulation, _) = rasterization(
        means,
        quats,
        scales,
        opacity if alpha is None else alpha,
        colors if color is None else color,
        view,
        k,
        width,
        height,
        packed=True,
        render_mode="RGB+ED" if depth else "RGB",
    )
    return (rgb[0], accumulation[0])


def _photo(panorama, yaw, pitch, width, height):
    return (
        torch.as_tensor(
            _view(panorama, _orientation(yaw, pitch), width, height).copy(),
            device="cuda:0",
        ).float()
        / 255
    )


def _depth_loss(predicted, reference, valid):
    return (
        (predicted - reference).abs().clamp(max=0.3) * valid
    ).sum() / valid.sum().clamp_min(1)


def _refine_capture(
    input_dir,
    pair_path,
    model_path,
    output_dir,
    steps=12000,
    width=640,
    refine_geometry=False,
):
    input_dir, pair_path, model_path, output_dir = map(
        Path, (input_dir, pair_path, model_path, output_dir)
    )
    if steps < 1 or width < 64 or width % 8:
        raise ValueError(
            "Use positive steps and an image width divisible by eight, at least 64."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(41)
    torch.set_num_threads(8)
    config = json.loads(pair_path.read_text())
    if "camera_poses" not in config:
        raise ValueError("Estimate the pair first to calibrate its capture views.")
    poses = np.array(config["camera_poses"])
    names = [config["target_scene"], config["source_scene"]]
    paths = [input_dir / "scenes" / f"{name}.ply" for name in names]
    for path in [*paths, pair_path, model_path]:
        if path.resolve() in (
            (output_dir / "merged.ply").resolve(),
            (output_dir / "pair_transform.json").resolve(),
        ):
            raise ValueError("Refinement output must differ from its inputs.")
    with TemporaryDirectory() as temporary:
        union = Path(temporary) / "union.ply"
        merge_gaussian_plys(
            *paths, np.asarray(config["transform_source_to_target"]), union
        )
        full = _read(union)
    counts = [len(_read(p)) for p in paths]
    assert len(counts) == 2 and sum(counts) == len(full)
    original_sources = np.split(full, np.cumsum(counts)[:-1])
    valid = np.isfinite(_positions(full)).all(1)
    valid &= np.isfinite(np.column_stack([full[f"scale_{i}"] for i in range(3)])).all(1)
    valid &= full["opacity"] > np.log(0.1 / 0.9)
    vertices = full[valid].copy()
    source = np.repeat(np.arange(2), counts)[valid]
    baseline_vertices = _read(model_path)
    geometry_fields = [
        n
        for n in vertices.dtype.names
        if n != "opacity" and (not n.startswith("f_dc_"))
    ]
    (a, b) = (
        repack_fields(vertices[geometry_fields]),
        repack_fields(baseline_vertices[geometry_fields]),
    )
    geometry_record = np.dtype((np.void, a.dtype.itemsize))
    lookup = {
        row.tobytes(): i for (i, row) in enumerate(b.view(geometry_record).ravel())
    }
    indices = np.array(
        [lookup.get(row.tobytes(), -1) for row in a.view(geometry_record).ravel()]
    )
    selected = indices >= 0
    if selected.sum() != len(baseline_vertices):
        raise ValueError("The baseline must preserve the input Gaussian geometry.")
    positions = _positions(vertices)
    scales = np.exp(np.column_stack([vertices[f"scale_{i}"] for i in range(3)]))
    protected = np.zeros(len(vertices), bool)
    for index in range(2):
        other = positions[source != index]
        tree = cKDTree(other[_voxel_indices(other, 0.04)])
        rows = np.flatnonzero(source == index)
        (distance, _) = tree.query(positions[rows], workers=8)
        protected[rows] = selected[rows] & (
            distance > np.maximum(0.2, 3 * scales[rows].max(1))
        )
    cells = np.column_stack([source, np.floor(positions / 0.12).astype(np.int32)])
    (_, patch) = np.unique(cells, axis=0, return_inverse=True)
    patch = torch.as_tensor(patch, device="cuda:0")
    patch_count = torch.bincount(patch).clamp_min(1)
    (width, height) = (width, width * 3 // 4)
    views = []
    for index, original in enumerate(original_sources):
        panorama = cv2.cvtColor(
            cv2.imread(str(input_dir / "pano" / f"{names[index]}.png")),
            cv2.COLOR_BGR2RGB,
        )
        teacher = _tensors(original)
        teacher = (
            *teacher[:3],
            torch.where(teacher[3] > 0.1, teacher[3], 0),
            teacher[4],
        )
        for pitch in (-30, 0, 30):
            for yaw in range(0, 360, 45):
                pose = _camera(poses[index], yaw, pitch)
                with torch.no_grad():
                    (rgb, alpha) = _render(
                        teacher, pose, width, height, depth=refine_geometry
                    )
                    reference = _photo(panorama, yaw, pitch, width, height)
                views.append(
                    {
                        "pose": pose,
                        "teacher": rgb,
                        "photo": reference,
                        "alpha": alpha,
                        "name": f"capture{index + 1}-{yaw}-{pitch}",
                        "kind": "capture",
                    }
                )
        for offset in (-0.1, 0.1):
            for yaw in (0, 90, 180, 270):
                pose = _camera(poses[index], yaw, offset=offset)
                with torch.no_grad():
                    (rgb, alpha) = _render(
                        teacher, pose, width, height, depth=refine_geometry
                    )
                views.append(
                    {
                        "pose": pose,
                        "teacher": rgb,
                        "alpha": alpha,
                        "name": f"near{index + 1}-{offset}-{yaw}",
                        "kind": "near",
                    }
                )
        del teacher
        torch.cuda.empty_cache()
    parameters = _tensors(vertices)
    baseline_parameters = _tensors(baseline_vertices)
    for ratio in (0.25, 0.5, 0.75):
        for elevation in (1.5, float(poses[:, 1, 3].mean())):
            position = (1 - ratio) * poses[0, :3, 3] + ratio * poses[1, :3, 3]
            position[1] = elevation
            forward = poses[1, :3, 3] - poses[0, :3, 3]
            forward[1] = 0
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 1, 0])
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
            pose[:3, 3] = position
            with torch.no_grad():
                (rgbd, alpha) = _render(
                    baseline_parameters, pose, width, height, depth=True
                )
            views.append(
                {
                    "pose": pose,
                    "teacher": rgbd,
                    "alpha": alpha,
                    "name": f"transition-{ratio}-{elevation}",
                    "kind": "transition",
                }
            )
    del baseline_parameters
    initial_alpha = parameters[3].clone() * 0.05
    initial_color = parameters[4].clone()
    rows = np.flatnonzero(selected)
    base_alpha = torch.as_tensor(
        baseline_vertices["opacity"].copy(), device="cuda:0"
    ).sigmoid()
    initial_alpha[rows] = base_alpha[indices[rows]]
    base_color = torch.as_tensor(
        np.column_stack([baseline_vertices[f"f_dc_{i}"] for i in range(3)]),
        device="cuda:0",
    )
    base_color = (base_color * C0 + 0.5).clamp(0, 1)
    initial_color[rows] = base_color[indices[rows]]
    protected_gpu = torch.as_tensor(protected, device="cuda:0")
    initial_alpha[protected_gpu] = parameters[3][protected_gpu]
    initial_color[protected_gpu] = parameters[4][protected_gpu]
    ratio = (initial_alpha / parameters[3]).clamp(0.005, 0.995)
    gate = torch.nn.Parameter(torch.logit(ratio))
    color_delta = torch.nn.Parameter(
        torch.atanh(((initial_color - parameters[4]) / 0.06).clamp(-0.99, 0.99))
    )
    optimizer = torch.optim.Adam(
        [{"params": [gate], "lr": 0.05}, {"params": [color_delta], "lr": 0.02}]
    )
    if refine_geometry:
        position_delta = torch.nn.Parameter(torch.zeros_like(parameters[0]))
        scale_delta = torch.nn.Parameter(torch.zeros_like(parameters[2]))
        optimizer.add_param_group(
            {"params": [position_delta, scale_delta], "lr": 0.008}
        )

    def geometry():
        if not refine_geometry:
            return parameters
        moved = (
            parameters[0] + 0.03 * torch.tanh(position_delta) * ~protected_gpu[:, None]
        )
        scaled = parameters[2] * torch.exp(
            0.3 * torch.tanh(scale_delta) * ~protected_gpu[:, None]
        )
        return (moved, parameters[1], scaled, *parameters[3:])

    captures = [v for v in views if v["kind"] == "capture"]
    near = [v for v in views if v["kind"] == "near"]
    transitions = [v for v in views if v["kind"] == "transition"]
    history = []
    started = time.monotonic()
    for step in range(steps):
        if step % 8 == 7:
            view = transitions[step // 8 % len(transitions)]
        elif step % 8 == 6:
            view = near[step // 8 % len(near)]
        else:
            view = captures[(step // 8 * 6 + step % 8) % len(captures)]
        gates = torch.where(protected_gpu, 1.0, torch.sigmoid(gate))
        alpha = parameters[3] * gates
        color = (parameters[4] + 0.06 * torch.tanh(color_delta)).clamp(0, 1)
        color = torch.where(protected_gpu[:, None], parameters[4], color)
        (rgb, accumulation) = _render(
            geometry(),
            view["pose"],
            width,
            height,
            alpha,
            color,
            depth=refine_geometry or view["kind"] == "transition",
        )
        teacher = view["teacher"]
        if view["kind"] == "transition":
            reliable = view["alpha"][..., 0] > 0.9
            loss = 0.1 * (rgb[..., :3] - teacher[..., :3]).abs().mean()
            loss += 0.08 * _depth_loss(rgb[..., 3], teacher[..., 3], reliable)
            loss += 0.25 * (accumulation - view["alpha"]).abs().mean()
        else:
            if refine_geometry:
                reliable = view["alpha"][..., 0] > 0.9
                depth_loss = 0.03 * _depth_loss(rgb[..., 3], teacher[..., 3], reliable)
                (rgb, teacher) = (rgb[..., :3], teacher[..., :3])
            loss = (rgb - teacher).abs().mean()
            if refine_geometry:
                loss += depth_loss
            for stride in (1, 2, 4):
                loss += (
                    0.15
                    * (
                        rgb[stride:]
                        - rgb[:-stride]
                        - (teacher[stride:] - teacher[:-stride])
                    )
                    .abs()
                    .mean()
                )
                loss += (
                    0.15
                    * (
                        rgb[:, stride:]
                        - rgb[:, :-stride]
                        - (teacher[:, stride:] - teacher[:, :-stride])
                    )
                    .abs()
                    .mean()
                )
            if view["kind"] == "capture":
                loss += 0.15 * (rgb - view["photo"]).abs().mean()
            loss += 0.1 * (accumulation - view["alpha"]).abs().mean()
        patch_mean = (
            torch.zeros(len(patch_count), device="cuda:0").scatter_add_(0, patch, gates)
            / patch_count
        )
        loss += 0.003 * (gates - patch_mean[patch]).square().mean()
        loss += 0.004 * torch.tanh(color_delta).square().mean()
        if refine_geometry:
            loss += 0.002 * torch.tanh(position_delta).square().mean()
            loss += 0.002 * torch.tanh(scale_delta).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == steps - 1:
            item = {
                "step": step,
                "view": view["name"],
                "loss": float(loss),
                "seconds": time.monotonic() - started,
            }
            history.append(item)
            print(f"[Refine] {step + 1}/{steps} loss={float(loss):.5f}", flush=True)
    with torch.no_grad():
        alpha = (
            (parameters[3] * torch.where(protected_gpu, 1.0, torch.sigmoid(gate)))
            .cpu()
            .numpy()
        )
        color = (parameters[4] + 0.06 * torch.tanh(color_delta)).clamp(0, 1)
        color = torch.where(protected_gpu[:, None], parameters[4], color).cpu().numpy()
        if refine_geometry:
            optimized = geometry()
            (moved, scales) = (
                optimized[0].cpu().numpy(),
                optimized[2].log().cpu().numpy(),
            )
            for index, name in enumerate(("x", "y", "z")):
                vertices[name] = moved[:, index]
                vertices[f"scale_{index}"] = scales[:, index]
    vertices["opacity"] = np.log(
        np.clip(alpha, 1e-07, 1 - 1e-07) / np.clip(1 - alpha, 1e-07, 1)
    )
    for index in range(3):
        vertices[f"f_dc_{index}"] = (color[:, index] - 0.5) / C0
    vertices[protected] = full[valid][protected]
    kept = (alpha > 0.1) | protected
    _write(output_dir / "merged.ply", vertices[kept])
    np.savez_compressed(
        output_dir / "provenance.npz",
        original_indices=np.flatnonzero(valid)[kept],
        protected=protected[kept],
    )
    config["initial_merge"] = config.get("merged_gaussians")
    config["merged_gaussians"] = {
        "file": "merged.ply",
        "output_count": int(kept.sum()),
        "original_view_refined": True,
    }
    (output_dir / "pair_transform.json").write_text(json.dumps(config, indent=2))
    summary = {
        "steps": steps,
        "width": width,
        "height": height,
        "pose_fixed": True,
        "geometry_fixed": not refine_geometry,
        "position_bound_per_axis": 0.03 if refine_geometry else 0,
        "log_scale_bound": 0.3 if refine_geometry else 0,
        "candidate_count": len(vertices),
        "output_count": int(kept.sum()),
        "protected_count": int(protected.sum()),
        "protected_exact": bool(
            np.array_equal(vertices[protected], full[valid][protected])
        ),
        "restored_count": int((kept & ~selected).sum()),
        "capture_views": len(captures),
        "near_views": len(near),
        "transition_views": len(transitions),
        "history": history,
        "training_seconds": time.monotonic() - started,
        "model_sha256": hashlib.sha256(
            (output_dir / "merged.ply").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "refinement_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def _refine_residuals(
    input_dir,
    model_path,
    output_dir,
    steps,
    width,
    navigation_height,
    threshold,
    position_bound,
    scale_bound,
    color_bound,
    capture_weight,
    preservation_weight,
    lock_previous=False,
    unlock_foreground=False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(41)
    torch.set_num_threads(8)
    config = json.loads((model_path.parent / "pair_transform.json").read_text())
    poses = np.asarray(config["camera_poses"])
    names = [config["target_scene"], config["source_scene"]]
    paths = [input_dir / "scenes" / f"{name}.ply" for name in names]
    if model_path.resolve() == (output_dir / "merged.ply").resolve():
        raise ValueError("Output must differ from the baseline.")
    with TemporaryDirectory() as folder:
        union = Path(folder) / "union.ply"
        merge_gaussian_plys(
            *paths, np.asarray(config["transform_source_to_target"]), union
        )
        full = _read(union)
    counts = [len(_read(path)) for path in paths]
    previous_report = json.loads(
        (model_path.parent / "refinement_report.json").read_text()
    )
    sources = np.split(full, np.cumsum(counts)[:-1])
    valid = full["opacity"] > np.log(0.1 / 0.9)
    original_indices = np.flatnonzero(valid)
    vertices = full[valid].copy()
    provenance = np.load(model_path.parent / "provenance.npz")
    baseline = _read(model_path)
    rows = np.searchsorted(original_indices, provenance["original_indices"])
    if (rows >= len(original_indices)).any() or not np.array_equal(
        original_indices[rows], provenance["original_indices"]
    ):
        raise ValueError("The model provenance does not match the input scenes.")
    vertices["opacity"] = -7
    vertices[rows] = baseline
    selected = np.zeros(len(vertices), bool)
    selected[rows] = True
    protected = np.zeros(len(vertices), bool)
    protected[rows] = provenance["protected"]
    prior_protected = protected.copy()
    locked = protected.copy()
    if lock_previous:
        locked[rows] |= provenance["frozen"]
    baseline_indices = np.full(len(vertices), -1, np.int32)
    baseline_indices[rows] = np.arange(len(baseline))
    parameters = _tensors(vertices)
    original_parameters = _tensors(full[valid])
    score = torch.zeros(len(vertices), device="cuda:0")
    height = width * 3 // 4
    source = torch.as_tensor(original_indices >= counts[0], device="cuda:0")
    foreground_hits = torch.zeros(len(vertices), device="cuda:0", dtype=torch.int32)
    views = []

    def observe_residual(pose, target, current, accumulation, index):
        error = (target[..., :3] - current[..., :3]).abs().mean(-1)
        error = F.max_pool2d(error[None, None], 17, 1, 8)[0, 0]
        R = torch.as_tensor(pose[:3, :3], device="cuda:0").float()
        t = torch.as_tensor(pose[:3, 3], device="cuda:0").float()
        for start in range(0, len(vertices), 262144):
            stop = min(start + 262144, len(vertices))
            xyz = (parameters[0][start:stop] - t) @ R
            z = xyz[:, 2].clamp_min(1e-6)
            x = (xyz[:, 0] / z * (width / 2) + width / 2).long()
            y = (xyz[:, 1] / z * (width / 2) + height / 2).long()
            inside = (
                (xyz[:, 2] > 0.05) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            )
            x, y = x.clamp(0, width - 1), y.clamp(0, height - 1)
            tolerance = 0.12 + 2 * parameters[2][start:stop].max(1).values
            visible = ((z - target[y, x, 3]).abs() < tolerance) | (
                (z - current[y, x, 3]).abs() < tolerance
            )
            score[start:stop] = torch.maximum(
                score[start:stop], error[y, x] * inside * visible
            )
            occluding = (z < target[y, x, 3] - 0.15) & (
                (z - current[y, x, 3]).abs() < tolerance
            )
            evidence = (
                inside
                & occluding
                & (error[y, x] > 0.06)
                & (accumulation[y, x, 0] > 0.95)
                & (source[start:stop] != index)
            )
            foreground_hits[start:stop] += evidence.int()

    for index, original in enumerate(sources):
        pano = cv2.cvtColor(
            cv2.imread(str(input_dir / "pano" / f"{names[index]}.png")),
            cv2.COLOR_BGR2RGB,
        )
        teacher = _tensors(original)
        teacher = (
            *teacher[:3],
            torch.where(teacher[3] > 0.1, teacher[3], 0),
            teacher[4],
        )
        requests = []
        for pitch in (-30, 0, 30):
            for yaw in range(0, 360, 45):
                requests.append(
                    ("capture", _camera(poses[index], yaw, pitch), yaw, pitch, False)
                )
        for ratio in (0, 0.125, 0.25):
            parent = poses[index].copy()
            parent[:3, 3] = (1 - ratio) * poses[index, :3, 3] + ratio * poses[
                1 - index, :3, 3
            ]
            parent[1, 3] = navigation_height
            for yaw in range(0, 360, 45):
                requests.append(("low", _camera(parent, yaw), yaw, 0, True))
        parent = poses[index].copy()
        parent[1, 3] = (navigation_height + poses[index, 1, 3]) / 2
        for yaw in range(0, 360, 45):
            requests.append(("low", _camera(parent, yaw), yaw, 0, False))
        for number, (kind, pose, yaw, pitch, probe) in enumerate(requests):
            with torch.no_grad():
                target, alpha = _render(teacher, pose, width, height, depth=True)
                current, _ = _render(parameters, pose, width, height, depth=True)
                reference = (
                    _photo(pano, yaw, pitch, width, height)
                    if kind == "capture"
                    else None
                )
                if probe or (kind == "capture" and pitch == 0):
                    observe_residual(pose, target, current, alpha, index)
                weight = 1 + 2 * (
                    (target[..., :3] - current[..., :3]).abs().mean(-1, keepdim=True)
                    / 0.08
                ).clamp(0, 1)
            views.append(
                {
                    "kind": kind,
                    "pose": pose,
                    "teacher": target,
                    "alpha": alpha,
                    "baseline": current,
                    "photo": reference,
                    "weight": weight,
                    "name": f"{kind}{index+1}-{number}-{yaw}",
                }
            )
        del teacher
        torch.cuda.empty_cache()
    geometry_active = (score > threshold) & ~torch.as_tensor(locked, device="cuda:0")
    unlocked = (
        (foreground_hits >= 3) & torch.as_tensor(locked, device="cuda:0")
        if unlock_foreground
        else torch.zeros_like(geometry_active)
    )
    active = geometry_active | unlocked
    active_cpu = active.cpu().numpy()
    protected &= ~unlocked.cpu().numpy()
    captures = [v for v in views if v["kind"] == "capture"]
    low = [v for v in views if v["kind"] == "low"]
    gate = torch.nn.Parameter(
        torch.logit((parameters[3] / original_parameters[3]).clamp(0.001, 0.999))
    )
    color_delta = torch.nn.Parameter(torch.zeros_like(parameters[4]))
    position_delta = torch.nn.Parameter(torch.zeros_like(parameters[0]))
    scale_delta = torch.nn.Parameter(torch.zeros_like(parameters[2]))
    optimizer = torch.optim.Adam(
        [
            {"params": [gate], "lr": 0.03},
            {"params": [color_delta], "lr": 0.01},
            {"params": [position_delta, scale_delta], "lr": 0.005},
        ]
    )

    def updated():
        alpha = torch.where(
            active, original_parameters[3] * torch.sigmoid(gate), parameters[3]
        )
        color = (
            parameters[4]
            + color_bound * torch.tanh(color_delta) * geometry_active[:, None]
        ).clamp(0, 1)
        means = (
            parameters[0]
            + position_bound * torch.tanh(position_delta) * geometry_active[:, None]
        )
        scales = parameters[2] * torch.exp(
            scale_bound * torch.tanh(scale_delta) * geometry_active[:, None]
        )
        return (means, parameters[1], scales, alpha, color)

    started = time.monotonic()
    history = []
    for step in range(steps):
        view = (
            captures[(step // 3) % len(captures)]
            if step % 3 == 0
            else low[(step // 3 * 2 + step % 3 - 1) % len(low)]
        )
        current = updated()
        rgbd, alpha = _render(current, view["pose"], width, height, depth=True)
        target = view["teacher"]
        rgb = rgbd[..., :3]
        reference = target[..., :3]
        loss = ((rgb - reference).abs() * view["weight"]).mean()
        for stride in (1, 2, 4):
            loss += (
                0.15
                * (
                    (rgb[stride:] - rgb[:-stride])
                    - (reference[stride:] - reference[:-stride])
                )
                .abs()
                .mean()
            )
            loss += (
                0.15
                * (
                    (rgb[:, stride:] - rgb[:, :-stride])
                    - (reference[:, stride:] - reference[:, :-stride])
                )
                .abs()
                .mean()
            )
        loss += 0.03 * _depth_loss(
            rgbd[..., 3], target[..., 3], view["alpha"][..., 0] > 0.9
        )
        loss += 0.10 * (alpha - view["alpha"]).abs().mean()
        if view["kind"] == "capture":
            loss += 0.15 * (rgb - view["photo"]).abs().mean()
            loss += preservation_weight * (rgb - view["baseline"][..., :3]).abs().mean()
            loss *= capture_weight
        loss += 0.004 * torch.tanh(color_delta).square().mean()
        loss += 0.004 * torch.tanh(position_delta).square().mean()
        loss += 0.004 * torch.tanh(scale_delta).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0 or step == steps - 1:
            item = {
                "step": step,
                "view": view["name"],
                "loss": float(loss),
                "seconds": time.monotonic() - started,
            }
            history.append(item)
            print(f"[Refine] {step + 1}/{steps} loss={float(loss):.5f}", flush=True)
    with torch.no_grad():
        optimized = updated()
        means, scales, alpha, color = [optimized[i].cpu().numpy() for i in (0, 2, 3, 4)]
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = means[:, i]
        vertices[f"scale_{i}"] = np.log(scales[:, i])
        vertices[f"f_dc_{i}"] = (color[:, i] - 0.5) / C0
    vertices["opacity"] = np.log(
        np.clip(alpha, 1e-7, 1 - 1e-7) / np.clip(1 - alpha, 1e-7, 1)
    )
    locked_rows = locked & selected
    for name in vertices.dtype.names:
        if name != "opacity":
            vertices[name][locked_rows] = baseline[name][baseline_indices[locked_rows]]
    frozen = ~active_cpu & selected
    vertices[frozen] = baseline[baseline_indices[frozen]]
    kept = (alpha > 0.1) | frozen
    _write(output_dir / "merged.ply", vertices[kept])
    np.savez_compressed(
        output_dir / "provenance.npz",
        original_indices=original_indices[kept],
        protected=protected[kept],
        frozen=frozen[kept],
        baseline_indices=baseline_indices[kept],
        opacity_unlocked=unlocked.cpu().numpy()[kept],
        geometry_frozen=locked[kept],
        prior_protected=prior_protected[kept],
    )
    config["initial_merge"] = config["merged_gaussians"]
    config["merged_gaussians"] = {
        "file": "merged.ply",
        "output_count": int(kept.sum()),
        "residual_refined": True,
    }
    (output_dir / "pair_transform.json").write_text(json.dumps(config, indent=2))
    summary = {
        "steps": steps,
        "geometry_fixed": False,
        "pose_fixed": True,
        "output_count": int(kept.sum()),
        "active_count": int(active.sum()),
        "frozen_count": int(frozen.sum()),
        "protected_count": int(protected.sum()),
        "frozen_exact": bool(
            np.array_equal(vertices[frozen], baseline[baseline_indices[frozen]])
        ),
        "position_bound_per_axis": previous_report["position_bound_per_axis"]
        + position_bound,
        "log_scale_bound": previous_report["log_scale_bound"] + scale_bound,
        "incremental_position_bound": position_bound,
        "incremental_log_scale_bound": scale_bound,
        "capture_views": len(captures),
        "lower_views": len(low),
        "navigation_height": navigation_height,
        "residual_threshold": threshold,
        "width": width,
        "height": height,
        "opacity_unlocked": int(unlocked.sum()),
        "geometry_frozen": int(locked.sum()),
        "capture_weight": capture_weight,
        "preservation_weight": preservation_weight,
        "history": history,
        "training_seconds": time.monotonic() - started,
        "baseline_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(
            (output_dir / "merged.ply").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "refinement_report.json").write_text(json.dumps(summary, indent=2))

    return summary


def refine_gaussian_pair(
    input_dir,
    pair_path,
    model_path,
    output_dir,
    steps=12000,
    width=640,
    final_width=960,
    navigation_height=1.5,
):
    input_dir, pair_path, model_path, output_dir = map(
        Path, (input_dir, pair_path, model_path, output_dir)
    )
    if steps < 1 or any(w < 64 or w % 8 for w in (width, final_width)):
        raise ValueError(
            "Use positive steps and image widths divisible by eight, at least 64."
        )
    if not np.isfinite(navigation_height) or navigation_height <= 0:
        raise ValueError("Navigation height must be positive.")
    config = json.loads(pair_path.read_text())
    inputs = [pair_path, model_path]
    inputs += [
        input_dir / "scenes" / f"{config[k]}.ply"
        for k in ("target_scene", "source_scene")
    ]
    outputs = [output_dir / name for name in ("merged.ply", "pair_transform.json")]
    if {p.resolve() for p in inputs} & {p.resolve() for p in outputs}:
        raise ValueError("Refinement output must differ from its inputs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder:
        stages = [Path(folder) / name for name in ("capture", "navigation", "detail")]
        reports = [
            _refine_capture(
                input_dir, pair_path, model_path, stages[0], steps, width, True
            )
        ]
        torch.cuda.empty_cache()
        reports.append(
            _refine_residuals(
                input_dir,
                stages[0] / "merged.ply",
                stages[1],
                steps,
                width,
                navigation_height,
                threshold=0.035,
                position_bound=0.02,
                scale_bound=0.15,
                color_bound=0.03,
                capture_weight=2.5,
                preservation_weight=0.5,
            )
        )
        torch.cuda.empty_cache()
        reports.append(
            _refine_residuals(
                input_dir,
                stages[1] / "merged.ply",
                stages[2],
                steps,
                final_width,
                navigation_height,
                threshold=0.05,
                position_bound=0.01,
                scale_bound=0.08,
                color_bound=0.015,
                capture_weight=3.0,
                preservation_weight=0.75,
                lock_previous=True,
                unlock_foreground=True,
            )
        )
        for name in ("merged.ply", "pair_transform.json", "provenance.npz"):
            shutil.copy2(stages[-1] / name, output_dir / name)
    summary = {
        "stages": {
            name: {
                k: v
                for k, v in report.items()
                if k not in ("history", "baseline_sha256")
            }
            for name, report in zip(("capture", "navigation", "detail"), reports)
        },
        "model_sha256": reports[-1]["model_sha256"],
        "output_count": reports[-1]["output_count"],
    }
    (output_dir / "refinement_report.json").write_text(json.dumps(summary, indent=2))
    return summary
