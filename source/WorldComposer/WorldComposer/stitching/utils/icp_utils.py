import numpy as np


def run_colored_icp_from_pcds(
    pcd1,
    pcd2,
    rotation: np.ndarray,
    translation: np.ndarray,
    voxel_size: float = 0.03,
    max_iteration: int = 100,
    robust: bool = False,
) -> tuple[np.ndarray, dict]:
    import open3d as o3d

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    levels = [
        (0.1, 0.5, 50),
        (0.05, 0.15, 50),
        (voxel_size, voxel_size * 2, max_iteration),
    ]
    for voxel, distance, iterations in levels:
        target = (
            o3d.geometry.PointCloud(pcd1)
            if voxel == voxel_size
            else pcd1.voxel_down_sample(voxel)
        )
        source = (
            o3d.geometry.PointCloud(pcd2)
            if voxel == voxel_size
            else pcd2.voxel_down_sample(voxel)
        )
        search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(0.1, 2 * voxel), max_nn=30
        )
        target.estimate_normals(search)
        source.estimate_normals(search)
        if robust:
            estimation = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=distance / 2)
                )
            )
            refined = o3d.pipelines.registration.registration_icp(
                source,
                target,
                distance,
                transform,
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )
            transform = refined.transformation
        reg = o3d.pipelines.registration.registration_colored_icp(
            source,
            target,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=iterations,
            ),
        )
        transform = reg.transformation
    return reg.transformation, {
        "fitness": float(reg.fitness),
        "rmse": float(reg.inlier_rmse),
    }
