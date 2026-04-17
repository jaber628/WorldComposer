import os
import numpy as np
import omni.kit.commands
from scipy.spatial import Delaunay

from omni.physx.scripts import particleUtils, physicsUtils
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid

from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema

try:
    from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material as _create_mdl_material
except Exception:
    _create_mdl_material = None
    from pxr import UsdShade

def _safe_create_material(material_url: str, material_name: str, material_path: str, stage):
    if _create_mdl_material is not None:
        _create_mdl_material(material_url, material_name, material_path)
        return
    try:
        UsdShade.Material.Define(stage, Sdf.Path(material_path))
    except Exception:
        pass

class FluidSystem:
    """
    FluidSystem class for simulating fluid particles in Isaac Sim.
    Manages fluid particle systems, materials, and physics properties.
    Does NOT manage containers (cups/bowls).
    """
    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        material_path: str,
        config: dict,
    ):
        # Enable USD ↔ PhysX bidirectional sync for particles.
        # Must be set BEFORE sim.reset() (i.e., during _setup_scene), consistent
        # with the working Lehome_Marble reference implementation.
        import carb as _carb
        _carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        _carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", True)

        self.prim_path = prim_path
        self.usd_path = usd_path
        self.material_path = material_path
        self.config = config
        self.physics_cfg = config.get("physics", {})
        self.material_cfg = config.get("material", {})
        
        self.stage = get_current_stage()

        # Add reference to the fluid mesh USD
        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.prim_path)
        self.mesh_prim_path = self.prim_path + "/water/water"

        # --- Particle System Setup ---
        self.particle_system_path = f"/Particle_Attribute/fluid_particle_system"

        if not is_prim_path_valid(self.particle_system_path):
            self.particle_system = PhysxSchema.PhysxParticleSystem.Define(
                self.stage, self.particle_system_path
            )
        else:
            prim = self.stage.GetPrimAtPath(self.particle_system_path)
            self.particle_system = PhysxSchema.PhysxParticleSystem(prim)

        # Configure particle system properties
        self.particle_system.CreateParticleContactOffsetAttr().Set(
            self.physics_cfg.get("particle_contact_offset", 0.0025)
        )
        self.particle_system.CreateContactOffsetAttr().Set(
            self.physics_cfg.get("contact_offset", 0.0025)
        )
        self.particle_system.CreateRestOffsetAttr().Set(
            self.physics_cfg.get("rest_offset", 0.0024)
        )
        self.particle_system.CreateFluidRestOffsetAttr().Set(
            self.physics_cfg.get("fluid_rest_offset", 0.0015)
        )
        self.particle_system.CreateSolidRestOffsetAttr().Set(
            self.physics_cfg.get("solid_rest_offset", 0.0015)
        )
        self.particle_system.CreateMaxVelocityAttr().Set(
            self.physics_cfg.get("max_velocity", 0.5)
        )

        if self.physics_cfg.get("smoothing", False):
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
        if self.physics_cfg.get("anisotropy", False):
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(self.particle_system.GetPrim())
        if self.physics_cfg.get("isosurface", True):
            PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.GetPrim())

        # --- Particle Generation and Instancing ---
        fluid_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(self.mesh_prim_path))
        fluid_volumn_multiplier = self.physics_cfg.get("fluid_volumn", 1.0)
        cloud_points_base = np.array(fluid_mesh.GetPointsAttr().Get())
        cloud_points = cloud_points_base * fluid_volumn_multiplier
        fluid_rest_offset = self.physics_cfg.get("fluid_rest_offset", 0.0015)
        particleSpacing = 2.0 * fluid_rest_offset

        self.init_particle_positions, self.init_particle_velocities = generate_particles_in_convex_mesh(
            vertices=cloud_points, sphere_diameter=particleSpacing, visualize=False
        )
        
        print(f"[DEBUG FluidSystem] Generated {len(self.init_particle_positions)} particles.")
        
        self.stage.GetPrimAtPath(self.mesh_prim_path).SetActive(False)

        self.particle_point_instancer_path = Sdf.Path(self.prim_path).AppendChild("particles")

        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage,
            path=self.particle_point_instancer_path,
            positions=Vt.Vec3fArray(self.init_particle_positions),
            velocities=Vt.Vec3fArray(self.init_particle_velocities),
            particle_system_path=self.particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.001,
            density=0.0,
        )

        self.point_instancer = UsdGeom.PointInstancer.Get(
            self.stage, self.particle_point_instancer_path
        )

        proto_path = self.particle_point_instancer_path.AppendChild("particlePrototype0")
        self.particle_prototype_path = proto_path

        particle_prototype_sphere = UsdGeom.Sphere.Get(self.stage, proto_path)
        particle_prototype_sphere.CreateRadiusAttr().Set(fluid_rest_offset)
        if self.physics_cfg.get("isosurface", True):
            UsdGeom.Imageable(particle_prototype_sphere).MakeInvisible()

        # Pre-create XformOps on the PointInstancer at setup time (BEFORE sim.reset()
        # builds TensorViews). This ensures that reset() only ever SETs existing ops
        # rather than ADDing new ones, which would cause USD structural changes at runtime.
        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f(0.0, 0.0, 0.0),
            orient=Gf.Quatf(1.0, 0.0, 0.0, 0.0),
            scale=Gf.Vec3f(1.0, 1.0, 1.0),
        )

        # --- Material Setup ---
        self._apply_material()

    def enable_physics_sync(self):
        """No-op: carb settings are now set early in __init__ (before sim.reset()),
        matching the working Lehome_Marble reference implementation."""
        pass

    def _apply_material(self):
        project_root = os.getcwd()
        material_url = os.path.abspath(os.path.join(project_root, self.material_path))
        material_name = os.path.splitext(os.path.basename(material_url))[0]
        
        base_dir = os.path.dirname(self.prim_path)
        material_base_path = f"{base_dir}/Looks/material"
        
        # DO NOT DELETE PRIMS AT RUNTIME
        # if is_prim_path_valid(material_base_path):
        #     delete_prim(material_base_path)

        unique_material_name = find_unique_string_name(
            initial_name=material_base_path,
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        color_material_path = unique_material_name
        _safe_create_material(material_url, material_name, color_material_path, self.stage)

        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=color_material_path,
            adhesion=self.material_cfg.get("adhesion"),
            adhesion_offset_scale=self.material_cfg.get("adhesion_offset_scale"),
            cohesion=self.material_cfg.get("cohesion", 0.01),
            particle_adhesion_scale=self.material_cfg.get("particle_adhesion_scale"),
            particle_friction_scale=self.material_cfg.get("particle_friction_scale"),
            drag=self.material_cfg.get("drag"),
            lift=self.material_cfg.get("lift"),
            friction=self.material_cfg.get("friction", 0.1),
            damping=self.material_cfg.get("damping", 0.99),
            gravity_scale=self.material_cfg.get("gravity_scale", 1.0),
            viscosity=self.material_cfg.get("viscosity", 0.0091),
            vorticity_confinement=self.material_cfg.get("vorticity_confinement"),
            surface_tension=self.material_cfg.get("surface_tension", 0.0074),
            density=self.material_cfg.get("density"),
            cfl_coefficient=self.material_cfg.get("cfl_coefficient"),
        )

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.particle_system_path,
            material_path=color_material_path,
        )

        if hasattr(self, "particle_prototype_path") and is_prim_path_valid(self.particle_prototype_path):
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.particle_prototype_path,
                material_path=color_material_path,
            )

    def reset(self, cup_pos: list, cup_ori_quat: list):
        """
        Reset fluid to initial particle positions at the cup's world location.

        We move the PointInstancer's world Transform to cup_pos (only SETs
        pre-created XformOps — no structural USD changes at runtime), then
        reset particle LOCAL positions to their initial values. updateToUsd=True
        propagates this back to PhysX so gravity will act on the particles.
        """
        # Move the entire particle system to the cup position
        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f(float(cup_pos[0]), float(cup_pos[1]), float(cup_pos[2])),
            orient=Gf.Quatf(
                float(cup_ori_quat[0]),
                Gf.Vec3f(float(cup_ori_quat[1]), float(cup_ori_quat[2]), float(cup_ori_quat[3]))
            ),
            scale=Gf.Vec3f(1.0, 1.0, 1.0),
        )

        # Reset particle LOCAL positions and zero out velocities
        self.point_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray(self.init_particle_positions)
        )
        self.point_instancer.GetVelocitiesAttr().Set(
            Vt.Vec3fArray(self.init_particle_velocities)
        )

    def get_particle_positions_world(self) -> np.ndarray:
        """Read current particle world positions from USD (only valid when updateParticlesToUsd=True,
        otherwise use get_particle_positions_from_physx)."""
        pos_attr = self.point_instancer.GetPositionsAttr().Get()
        if not pos_attr:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pos_attr], dtype=np.float32)

def generate_particles_in_convex_mesh(vertices: np.ndarray, sphere_diameter: float, visualize: bool = False):
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    hull = Delaunay(vertices)

    x_vals = np.arange(min_bound[0], max_bound[0], sphere_diameter)
    y_vals = np.arange(min_bound[1], max_bound[1], sphere_diameter)
    z_vals = np.arange(min_bound[2], max_bound[2], sphere_diameter)

    samples = np.stack(np.meshgrid(x_vals, y_vals, z_vals, indexing="ij"), axis=-1).reshape(-1, 3)

    inside_mask = hull.find_simplex(samples) >= 0
    inside_points = samples[inside_mask]

    velocity = np.zeros_like(inside_points)

    return [Gf.Vec3f(*pt) for pt in inside_points], [Gf.Vec3f(*vel) for vel in velocity]
