import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_DIR = PROJECT_ROOT / "3rd"


def default_mesh_usd_path(glb_path: str) -> str:
    glb = Path(glb_path)
    return str(glb.with_name(f"{glb.stem}_mesh.usd"))


def convert_ply_to_usdz(ply_path: str, output_usdz_path: str):
    """Convert a PLY Gaussian reconstruction to USDZ using 3dgrut."""
    print(f"[Info] Converting PLY to USDZ: {ply_path} -> {output_usdz_path}")

    env = os.environ.copy()
    threedgrut_path = THIRD_PARTY_DIR / "3dgrut"
    if not threedgrut_path.exists():
        print(f"[Warning] 3dgrut path not found: {threedgrut_path}")

    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{threedgrut_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(threedgrut_path)

    cmd = [
        "python",
        "-m",
        "threedgrut.export.scripts.ply_to_usd",
        ply_path,
        "--output_file",
        output_usdz_path,
    ]
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"[Success] USDZ created: {output_usdz_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] 3dgrut conversion failed: {e}")
        raise


async def _convert_glb_to_usd_async(glb_path: str, output_mesh_usd_path: str, *, timeout_sec: float = 180.0):
    import omni.kit.asset_converter  # type: ignore

    def _to_file_uri(path: str) -> str:
        return Path(path).resolve().as_uri()

    def _progress_callback(progress: int, total_steps: int) -> int:
        print(f"[AssetConverter] progress={progress} total_steps={total_steps}")
        return progress

    src_uri = _to_file_uri(glb_path)
    dst_uri = _to_file_uri(output_mesh_usd_path)
    print(f"[AssetConverter] src={src_uri}")
    print(f"[AssetConverter] dst={dst_uri}")
    print("[AssetConverter] mode=geometry_only (ignore_materials=True, merge_all_meshes=True)")

    converter_manager = omni.kit.asset_converter.get_instance()
    context = omni.kit.asset_converter.AssetConverterContext()
    # Only geometry is needed for depth, collision, and alignment.
    context.ignore_materials = True
    context.ignore_animations = True
    context.ignore_cameras = True
    context.ignore_lights = True
    context.merge_all_meshes = True
    context.use_meter_as_world_unit = True
    context.embed_textures = False

    task = converter_manager.create_converter_task(src_uri, dst_uri, _progress_callback, context)
    try:
        success = await asyncio.wait_for(task.wait_until_finished(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"GLB -> USD conversion timed out after {timeout_sec}s: {glb_path}") from exc
    if not success:
        raise RuntimeError(f"GLB -> USD conversion failed: {glb_path} -> {output_mesh_usd_path}")


def convert_glb_to_usd(glb_path: str, output_mesh_usd_path: str, timeout_sec: float = 180.0, simulation_app=None):
    """Convert a GLB mesh to a USD asset that can be referenced by the scene stage."""
    glb = Path(glb_path)
    out = Path(output_mesh_usd_path)
    if not glb.exists():
        raise FileNotFoundError(f"Mesh (GLB) file not found: {glb}")

    print(f"[Info] Converting GLB to USD: {glb} -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    owns_simulation_app = simulation_app is None
    if owns_simulation_app:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_convert_glb_to_usd_async(str(glb.resolve()), str(out.resolve()), timeout_sec=timeout_sec))
    finally:
        loop.close()
        asyncio.set_event_loop(None)
        if owns_simulation_app:
            simulation_app.close()

    if not out.exists():
        raise FileNotFoundError(f"GLB conversion finished but USD was not created: {out}")
    print(f"[Success] Mesh USD created: {out}")


def build_assembled_scene(
    ply_path: str,
    glb_path: str,
    out_usd_path: str,
    mesh_usd_path: str | None = None,
    timeout_sec: float = 180.0,
    scene_scale: float = 1.0,
):
    out_dir = os.path.dirname(out_usd_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    ply_basename = os.path.basename(ply_path)
    usdz_name = os.path.splitext(ply_basename)[0] + ".usdz"
    temp_usdz_path = os.path.join(out_dir, usdz_name)
    resolved_mesh_usd_path = mesh_usd_path or default_mesh_usd_path(glb_path)

    convert_ply_to_usdz(ply_path, temp_usdz_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        convert_glb_to_usd(glb_path, resolved_mesh_usd_path, timeout_sec=timeout_sec, simulation_app=simulation_app)
        align_mesh_to_usd(out_usd_path, temp_usdz_path, resolved_mesh_usd_path, scene_scale=scene_scale)
    finally:
        simulation_app.close()


def align_mesh_to_usd(output_usd_path: str, gauss_usdz_path: str, mesh_usd_path: str, scene_scale: float = 1.0):
    """Assemble the Gaussian USDZ and converted mesh USD into one aligned scene."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not os.path.exists(gauss_usdz_path):
        raise FileNotFoundError(f"Gauss (USDZ) file not found: {gauss_usdz_path}")
    if not os.path.exists(mesh_usd_path):
        raise FileNotFoundError(f"Mesh (USD) file not found: {mesh_usd_path}")

    print(f"[Info] Creating assembled USD scene: {output_usd_path}")
    if os.path.exists(output_usd_path):
        os.remove(output_usd_path)

    stage = Usd.Stage.CreateNew(output_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scale_vec = Gf.Vec3d(float(scene_scale), float(scene_scale), float(scene_scale))
    print(f"[Info] Scene scale: {scene_scale}")

    gauss_prim_path = "/World/gauss"
    gauss_prim = stage.DefinePrim(gauss_prim_path, "Xform")
    gauss_prim.GetReferences().AddReference(os.path.abspath(gauss_usdz_path))
    gauss_xform = UsdGeom.Xformable(gauss_prim)
    gauss_xform.ClearXformOpOrder()
    gauss_xform.AddRotateXOp().Set(-90.0)
    gauss_xform.AddScaleOp().Set(scale_vec)

    mesh_prim_path = "/World/Xform"
    mesh_prim = stage.DefinePrim(mesh_prim_path, "Xform")
    mesh_prim.GetReferences().AddReference(os.path.abspath(mesh_usd_path))
    mesh_xform = UsdGeom.Xformable(mesh_prim)
    mesh_xform.ClearXformOpOrder()
    mesh_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    mesh_xform.AddRotateXYZOp().Set(Gf.Vec3d(-90.0, 0.0, 0.0))
    mesh_xform.AddScaleOp().Set(scale_vec)

    mesh_imageable = UsdGeom.Imageable(mesh_prim)
    mesh_imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    rigid_api.CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("meshSimplification")

    stage.Save()
    print(f"[Success] Assembled scene created: {output_usd_path}")


def main():
    parser = argparse.ArgumentParser(description="Real2Sim phase 1: assemble PLY Gaussian and GLB mesh assets into a USD scene")
    parser.add_argument("--ply", type=str, required=True, help="Input .ply Gaussian reconstruction path")
    parser.add_argument("--glb", type=str, required=True, help="Input .glb mesh path")
    parser.add_argument("--out_usd", type=str, required=True, help="Output assembled .usd scene path")
    parser.add_argument("--mesh_usd", type=str, default=None, help="Optional output path for the converted mesh USD")
    parser.add_argument("--converter_timeout", type=float, default=300.0, help="GLB-to-USD conversion timeout in seconds")
    parser.add_argument("--scene_scale", type=float, default=1.0, help="Uniform scale applied to Gaussian and mesh assets")
    args = parser.parse_args()

    try:
        build_assembled_scene(
            args.ply,
            args.glb,
            args.out_usd,
            args.mesh_usd,
            timeout_sec=args.converter_timeout,
            scene_scale=args.scene_scale,
        )
    except Exception as exc:
        print(f"[Error] scene_assembler failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
