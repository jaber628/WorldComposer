import numpy as np


def run_colored_icp_from_pcds(
    pcd1,
    pcd2,
    rotation: np.ndarray,
    translation: np.ndarray,
    voxel_size: float = 0.03,
    max_iteration: int = 100,
) -> tuple[np.ndarray, dict]:
    import open3d as o3d

    pcd1_icp = o3d.geometry.PointCloud(pcd1)
    pcd2_icp = o3d.geometry.PointCloud(pcd2)

    initial = np.eye(4, dtype=np.float64)
    initial[:3, :3] = rotation
    initial[:3, 3] = translation

    voxel_coarse = 0.1
    pcd1_c = pcd1_icp.voxel_down_sample(voxel_coarse)
    pcd2_c = pcd2_icp.voxel_down_sample(voxel_coarse)
    pcd1_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pcd2_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    reg_coarse = o3d.pipelines.registration.registration_colored_icp(
        pcd2_c,
        pcd1_c,
        voxel_coarse * 5.0,
        initial,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        ),
    )

    pcd1_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd2_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    reg = o3d.pipelines.registration.registration_colored_icp(
        pcd2_icp,
        pcd1_icp,
        voxel_size * 2.0,
        reg_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iteration
        ),
    )
    return reg.transformation, {"fitness": float(reg.fitness), "rmse": float(reg.inlier_rmse)}
