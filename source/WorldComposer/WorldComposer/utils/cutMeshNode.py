import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def get_knife_plane_in_world(knife_prim):
    geometry = UsdGeom.Boundable(knife_prim)
    extent = geometry.GetExtentAttr().Get()
    transform = geometry.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    size = extent[1] - extent[0]
    axes = [transform.TransformDir(Gf.Vec3d(*(float(size[i]) * int(i == j) for i in range(3)))) for j in range(3)]
    normal = min(axes, key=lambda axis: axis.GetLength()).GetNormalized()
    return transform.Transform(Gf.Vec3d((extent[0] + extent[1]) / 2)), normal, extent, transform


def get_cutting_edge(knife_prim):
    points = np.asarray(UsdGeom.Mesh(knife_prim).GetPointsAttr().Get())
    lower, upper = points.min(0), points.max(0)
    edge = points[
        (np.abs(points[:, 0] - (lower[0] + upper[0]) / 2) < 1e-5)
        & (points[:, 1] > (lower[1] + upper[1]) / 2)
        & (points[:, 2] > lower[2] + 0.25 * (upper[2] - lower[2]))
    ]
    edge = edge[np.argsort(edge[:, 2])]
    return np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / .001) + 1))
                           for a, b in zip(edge[:-1], edge[1:])])


def knife_intersects(knife_prim, mesh_prim):
    import trimesh

    _, _, extent, knife_transform = get_knife_plane_in_world(knife_prim)
    prims = mesh_prim if isinstance(mesh_prim, (list, tuple)) else [mesh_prim]
    edge = get_cutting_edge(knife_prim)
    for prim in prims:
        mesh = UsdGeom.Mesh(prim)
        transform = mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        matrix = np.array(transform * knife_transform.GetInverse())
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        points = vertices @ matrix[:3, :3] + matrix[3, :3]
        if not (np.all(points.max(0) >= extent[0]) and np.all(points.min(0) <= extent[1])):
            continue
        matrix = np.array(knife_transform)
        samples = edge @ matrix[:3, :3] + matrix[3, :3]
        matrix = np.array(transform)
        vertices = vertices @ matrix[:3, :3] + matrix[3, :3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        solid = trimesh.Trimesh(vertices, faces)
        if np.any(trimesh.proximity.signed_distance(solid, samples) >= -.001):
            return True
    return False


def generate_mesh(stage, path, vertices, faces, transform, material, mass, velocity=(0., 0., 0.), angular_velocity=(0., 0., 0.),
                  uv=None, cut_faces=None, cut_material=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, np.min(vertices, axis=0))), Gf.Vec3f(*map(float, np.max(vertices, axis=0)))])
    mesh.AddTransformOp().Set(Gf.Matrix4d(np.asarray(transform).tolist()))
    prim = mesh.GetPrim()
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateVelocityAttr(Gf.Vec3f(*map(float, velocity)))
    body.CreateAngularVelocityAttr(Gf.Vec3f(*map(float, angular_velocity)))
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(.001)
    collision.CreateRestOffsetAttr(0.)
    if material:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(material)))
    if uv is not None:
        import trimesh

        solid = trimesh.Trimesh(vertices, faces, process=False)
        normals = solid.vertex_normals[solid.faces].copy()
        if cut_faces is not None and len(cut_faces):
            normals[cut_faces] = solid.face_normals[cut_faces, None, :]
        mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, value)) for value in normals.reshape(-1, 3)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
            [Gf.Vec2f(*map(float, point)) for point in np.asarray(uv).reshape(-1, 2)])
    if cut_material and cut_faces is not None and len(cut_faces):
        subset = UsdGeom.Subset.Define(stage, str(path) + "/cut_surface")
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(np.asarray(cut_faces).tolist())
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(cut_material)))
    return prim


def cut_prim(stage, prim, plane_origin, plane_normal):
    import trimesh

    mesh = UsdGeom.Mesh(prim)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    faces = []
    start = 0
    for count in counts:
        faces.extend([indices[start], indices[start + i], indices[start + i + 1]] for i in range(1, count - 1))
        start += count
    original = trimesh.Trimesh(vertices=np.asarray(mesh.GetPointsAttr().Get()), faces=faces, process=False)
    origin, normal = np.asarray(plane_origin), np.asarray(plane_normal)
    distances = (original.vertices - origin) @ normal
    if distances.min() >= -.003 or distances.max() <= .003:
        return False
    halves = [original.slice_plane(origin, sign * normal, cap=True) for sign in (1, -1)]
    if any(not part.is_watertight or abs(part.volume) < abs(original.volume) * .03 for part in halves):
        return False
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    body = UsdPhysics.RigidBodyAPI(prim)
    velocity = body.GetVelocityAttr().Get() or Gf.Vec3f(0.)
    angular_velocity = body.GetAngularVelocityAttr().Get() or Gf.Vec3f(0.)
    mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() or .12
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    source_uv = np.asarray(uv.ComputeFlattened()).reshape(-1, 3, 2) if uv and uv.HasValue() else None
    cut_material = material.GetPath().GetParentPath().AppendChild("CutSurface") if material else None
    cut_material = str(cut_material) if cut_material and stage.GetPrimAtPath(cut_material) else None
    old_cap = prim.GetChild("cut_surface")
    old_cap_faces = np.asarray(UsdGeom.Subset(old_cap).GetIndicesAttr().Get()) if old_cap else []
    for i, part in enumerate(halves, 1):
        mapped_uv, cap_faces = None, None
        if source_uv is not None:
            _, _, source_faces = trimesh.proximity.closest_point(original, part.triangles_center)
            weights = trimesh.triangles.points_to_barycentric(
                np.repeat(original.triangles[source_faces], 3, axis=0), part.triangles.reshape(-1, 3),
            ).reshape(-1, 3, 3)
            mapped_uv = np.einsum("fcv,fvd->fcd", weights, source_uv[source_faces])
            on_plane = np.all(np.abs((part.triangles - origin) @ normal) < 1e-6, axis=1)
            cap_faces = np.flatnonzero(on_plane | np.isin(source_faces, old_cap_faces))
        path = prim.GetParent().GetPath().AppendChild(prim.GetName() + f'_sub{i}')
        generate_mesh(stage, path, part.vertices, part.faces, transform, str(material.GetPath()),
                      mass * abs(part.volume / original.volume), velocity, angular_velocity, mapped_uv, cap_faces, cut_material)
    prim.SetActive(False)
    return True


class cutMeshNode:
    @staticmethod
    def compute(stage, mesh_path, knife_path):
        knife = stage.GetPrimAtPath(knife_path)
        center, normal, _, _ = get_knife_plane_in_world(knife)
        count = 0
        for prim in list(stage.GetPrimAtPath(mesh_path).GetChildren()):
            if not prim.IsA(UsdGeom.Mesh) or not knife_intersects(knife, prim):
                continue
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_origin = transform.GetInverse().Transform(center)
            local_normal = transform.GetTranspose().TransformDir(normal).GetNormalized()
            count += cut_prim(stage, prim, local_origin, local_normal)
        return count
