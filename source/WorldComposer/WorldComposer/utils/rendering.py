from __future__ import annotations

"""Utilities to configure Isaac Sim and Omniverse Kit rendering defaults."""

from typing import Any

_TONEMAP_APPLIED = False
_LIGHTING_APPLIED = False


def _get_settings():
    try:
        import carb  # type: ignore
        return carb.settings.get_settings()
    except Exception:
        return None


def _safe_set_setting(settings, path: str, value: Any) -> bool:
    """Try to set a carb setting."""
    if settings is None:
        return False
    try:
        settings.set(path, value)
        return True
    except Exception:
        return False


def set_tone_mapping_fstop(fstop: float = 5.8, enabled: bool = True) -> bool:
    """Set RTX tone mapping fNumber to the desired value."""
    settings = _get_settings()
    if settings is None:
        return False

    ok1 = _safe_set_setting(settings, "/rtx/post/tonemap/enabled", bool(enabled))
    ok2 = _safe_set_setting(settings, "/rtx/post/tonemap/fNumber", float(fstop))
    return ok1 and ok2


def setup_default_lighting(task_name: str | None = None):
    """Create a three-point light rig under `/World/DefaultLight`."""
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom, UsdLux  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        base_path = "/World/DefaultLight"
        configs = [
            ("Key", 3500.0, (65.0, 25.0, 0.0)),
            ("Fill", 900.0, (80.0, -55.0, 0.0)),
            ("Rim", 600.0, (160.0, 160.0, 0.0)),
        ]

        for name, intensity, rot in configs:
            path = f"{base_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            light = UsdLux.DistantLight.Define(stage, path) if not prim.IsValid() else UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot))

        return True
    except Exception:
        return False


def setup_default_lighting_drawer(task_name: str | None = None):
    """Create the drawer-specific three-point light rig."""
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom, UsdLux  # type: ignore

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        base_path = "/World/DefaultLight"
        configs = [
            ("Key", 3500.0, (-75.0, 25.0, 20.0)),
            ("Fill", 900.0, (-60.0, -55.0, 20.0)),
            ("Rim", 600.0, (20.0, 160.0, 20.0)),
        ]

        for name, intensity, rot in configs:
            path = f"{base_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            light = UsdLux.DistantLight.Define(stage, path) if not prim.IsValid() else UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot))

        return True
    except Exception:
        return False


def apply_default_render_settings(
    *,
    tonemap_fstop: float = 5.8,
    enable_tonemap: bool = True,
    once_per_process: bool = True,
    task_name: str | None = None,
    **kwargs,
) -> None:
    """Apply default tone mapping and light rig settings."""
    global _TONEMAP_APPLIED, _LIGHTING_APPLIED

    if not (once_per_process and _TONEMAP_APPLIED):
        tm_ok = set_tone_mapping_fstop(fstop=tonemap_fstop, enabled=enable_tonemap)
        if tm_ok:
            _TONEMAP_APPLIED = True

    if not (once_per_process and _LIGHTING_APPLIED):
        if setup_default_lighting(task_name=task_name):
            _LIGHTING_APPLIED = True


def apply_default_render_settings_drawer(
    *,
    tonemap_fstop: float = 5.8,
    enable_tonemap: bool = True,
    once_per_process: bool = True,
    task_name: str | None = None,
    **kwargs,
) -> None:
    """Apply default tone mapping and drawer light rig settings."""
    global _TONEMAP_APPLIED, _LIGHTING_APPLIED

    if not (once_per_process and _TONEMAP_APPLIED):
        tm_ok = set_tone_mapping_fstop(fstop=tonemap_fstop, enabled=enable_tonemap)
        if tm_ok:
            _TONEMAP_APPLIED = True

    if not (once_per_process and _LIGHTING_APPLIED):
        if setup_default_lighting_drawer(task_name=task_name):
            _LIGHTING_APPLIED = True
