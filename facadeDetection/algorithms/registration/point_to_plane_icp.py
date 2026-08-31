from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RegistrationConfig:
    """Parameters for residual ICP on already-globalized PLY clouds."""
    voxel_size: float = 0.05
    max_correspondence_distance: float = 0.12
    max_iteration: int = 30
    pyramid_scales: tuple[float, ...] = (2.0, 1.0)


# =============================================================================
# 1. 配准质量评估
# =============================================================================

def registration_metrics(source_points, target_points, transformation,
                          max_correspondence_distance):
    """Recompute metrics from the same fine registration-downsample domain."""
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    if len(source) == 0 or len(target) == 0:
        return {'metric_domain': 'registration_downsample', 'fitness': 0.0,
                'rmse': float('inf'), 'mean_error': float('inf'),
                'p95_error': float('inf'), 'max_error': float('inf'),
                'correspondence_count': 0}
    moved = source @ t[:3, :3].T + t[:3, 3]
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    moved_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moved))
    errors = np.asarray(moved_pcd.compute_point_cloud_distance(target_pcd),
                        dtype=np.float64)
    errors = errors[errors <= float(max_correspondence_distance)]
    if len(errors) == 0:
        rmse = mean = p95 = maximum = float('inf')
    else:
        rmse = float(np.sqrt(np.mean(errors * errors)))
        mean = float(np.mean(errors))
        p95 = float(np.percentile(errors, 95))
        maximum = float(np.max(errors))
    return {
        'metric_domain': 'registration_downsample',
        'fitness': float(len(errors) / len(source)),
        'rmse': rmse,
        'mean_error': mean,
        'p95_error': p95,
        'max_error': maximum,
        'correspondence_count': int(len(errors)),
    }


# =============================================================================
# 2. 配准域点云缓存
# =============================================================================

@dataclass
class RegistrationCloud:
    """Immutable registration-domain snapshot derived from a proxy cloud."""
    points: np.ndarray
    colors: np.ndarray | None
    voxel_size: float

    def as_open3d(self) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        if self.colors is not None and len(self.colors) == len(self.points):
            cloud.colors = o3d.utility.Vector3dVector(self.colors)
        return cloud


def build_registration_cloud(points, colors=None, voxel_size=0.05) -> RegistrationCloud:
    """Create the only point domain used by ICP and its quality metrics."""
    voxel_size = max(float(voxel_size), 1e-4)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = np.ascontiguousarray(pts[valid])
    cols = None
    if colors is not None:
        raw_cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(raw_cols) == len(valid):
            cols = np.ascontiguousarray(raw_cols[valid])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if cols is not None:
        cloud.colors = o3d.utility.Vector3dVector(cols)
    down = cloud.voxel_down_sample(voxel_size)
    down_pts = np.asarray(down.points, dtype=np.float64)
    down_cols = np.asarray(down.colors, dtype=np.float64) if down.has_colors() else None
    return RegistrationCloud(np.ascontiguousarray(down_pts),
                             np.ascontiguousarray(down_cols) if down_cols is not None else None,
                             voxel_size)


# =============================================================================
# 3. ICP 配准算法
# =============================================================================

@dataclass
class ICPResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_count: int
    levels: list = field(default_factory=list)
    accepted: bool = True
    message: str = ''


def rigid_transform_from_correspondences(source_points, target_points):
    """Solve the least-squares rigid transform without scale (Kabsch)."""
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(src) != len(tgt) or len(src) < 3:
        raise ValueError('人工对应点至少需要 3 对且数量必须一致')
    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise ValueError('人工对应点包含无效坐标')
    if (len(np.unique(src, axis=0)) < 3 or
            len(np.unique(tgt, axis=0)) < 3):
        raise ValueError('人工对应点包含重复点，无法稳定求解刚体变换')
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
        raise ValueError('源点近似共线，无法稳定求解刚体变换')
    if np.linalg.matrix_rank(tgt - tgt.mean(axis=0)) < 2:
        raise ValueError('目标点近似共线，无法稳定求解刚体变换')
    cs, ct = src.mean(axis=0), tgt.mean(axis=0)
    u, _, vt = np.linalg.svd((src - cs).T @ (tgt - ct))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = ct - r @ cs
    out = np.eye(4, dtype=np.float64)
    out[:3, :3], out[:3, 3] = r, t
    return out


def manual_seeded_icp(source_points, target_points, source_correspondences,
                       target_correspondences, *, voxel_size=0.05,
                       max_iteration=30, max_correspondence_distance=0.12):
    """Global-coordinate ICP seeded by optional operator correspondences."""
    src_pairs = np.asarray(source_correspondences, dtype=np.float64).reshape(-1, 3)
    tgt_pairs = np.asarray(target_correspondences, dtype=np.float64).reshape(-1, 3)
    init = rigid_transform_from_correspondences(src_pairs, tgt_pairs)
    result = point_to_plane_icp(source_points, target_points, init=init,
                                voxel_size=voxel_size, max_iteration=max_iteration,
                                max_correspondence_distance=max_correspondence_distance)
    initial_rmse = float(np.sqrt(np.mean(np.sum((src_pairs @ init[:3, :3].T + init[:3, 3] - tgt_pairs) ** 2, axis=1))))
    final_pairs = src_pairs @ result.transformation[:3, :3].T + result.transformation[:3, 3]
    pair_rmse = float(np.sqrt(np.mean(np.sum((final_pairs - tgt_pairs) ** 2, axis=1))))
    if pair_rmse > max(initial_rmse * 1.5, voxel_size * 2.0):
        result.accepted = False
        result.message = 'ICP 后人工对应点残差变大，请检查同名点选择'
    result.levels.insert(0, {'manual_initial_rmse': initial_rmse,
                             'manual_final_rmse': pair_rmse})
    return result


def point_to_plane_icp(source_points, target_points, *, init=None, voxel_size=0.05,
                       max_correspondence_distance=None, max_iteration=30,
                       pyramid_scales=(2.0, 1.0)):
    """Estimate residual ``Delta T`` between clouds in one global frame."""
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError('点到面 ICP 至少需要两组各 3 个有效点')

    scales = [float(voxel_size) * float(level) for level in pyramid_scales] if voxel_size else [0.05]
    if not scales:
        raise ValueError('ICP 至少需要一个有效金字塔层级')
    scales = [max(x, 1e-3) for x in scales]
    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)
    reports = []
    final = None

    base_source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    base_target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    for scale in scales:
        source = base_source.voxel_down_sample(scale)
        target = base_target.voxel_down_sample(scale)
        radius = max(scale * 2.5, 1e-3)
        for cloud in (source, target):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
            cloud.normalize_normals()
        distance = (float(max_correspondence_distance)
                    if max_correspondence_distance is not None
                    else max(scale * 2.5, 0.01))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(max_iteration))
        final = o3d.pipelines.registration.registration_icp(
            source, target, distance, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)
        T = np.asarray(final.transformation)
        reports.append({'voxel': scale, 'fitness': float(final.fitness),
                        'rmse': float(final.inlier_rmse),
                        'correspondences': len(final.correspondence_set)})

    metric_distance = (float(max_correspondence_distance)
                       if max_correspondence_distance is not None
                       else max(scales[-1] * 2.5, 0.01))
    checked = registration_metrics(src, tgt, T,
                                   max_correspondence_distance=metric_distance)
    fitness = checked['fitness']
    rmse = checked['rmse']
    correspondence_count = checked['correspondence_count']
    # Small facade patches can legitimately contain fewer than 20 downsampled
    # points; use a relative floor while retaining a correspondence ratio.
    min_correspondences = max(3, int(np.ceil(min(len(src), len(tgt)) * 0.02)))
    accepted = (fitness >= 0.05 and correspondence_count >= min_correspondences
                and np.isfinite(rmse))
    message = '' if accepted else '有效对应不足或配准质量未达到门限'
    reports.append(checked)
    return ICPResult(T, fitness, rmse, correspondence_count, reports, accepted, message)