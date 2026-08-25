from dataclasses import dataclass, field
import numpy as np
import open3d as o3d

@dataclass
class ICPResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_count: int
    levels: list = field(default_factory=list)
    accepted: bool = True
    message: str = ''

def point_to_plane_icp(source_points, target_points, *, init=None, voxel_size=0.05,
                       max_correspondence_distance=None, max_iteration=40):
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError('点到面 ICP 至少需要两组各 3 个有效点')
    # The PLYs entering this function are already in the project global frame.
    # Only a refinement transform is estimated; never apply transformToGlobal here.
    scales = [float(voxel_size) * 4, float(voxel_size) * 2, float(voxel_size)] if voxel_size else [0.05]
    scales = [max(x, 1e-3) for x in scales]
    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)
    reports = []
    final = None
    for scale in scales:
        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)).voxel_down_sample(scale)
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt)).voxel_down_sample(scale)
        radius = max(scale * 2.5, 1e-3)
        for cloud in (source, target):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
            cloud.normalize_normals()
        distance = max_correspondence_distance or max(scale * 2.5, 0.01)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(max_iteration))
        final = o3d.pipelines.registration.registration_icp(
            source, target, distance, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
        T = np.asarray(final.transformation)
        reports.append({'voxel': scale, 'fitness': float(final.fitness),
                        'rmse': float(final.inlier_rmse),
                        'correspondences': len(final.correspondence_set)})
    fitness = float(final.fitness)
    rmse = float(final.inlier_rmse)
    accepted = fitness >= 0.05 and len(final.correspondence_set) >= 20 and np.isfinite(rmse)
    message = '' if accepted else '有效对应不足或配准质量未达到门限'
    return ICPResult(T, fitness, rmse, len(final.correspondence_set), reports, accepted, message)