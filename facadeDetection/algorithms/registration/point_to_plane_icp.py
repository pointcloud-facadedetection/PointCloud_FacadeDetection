from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RegistrationConfig:
    """Parameters for residual ICP on already-globalized PLY clouds."""
    voxel_size: float = 0.05
    max_correspondence_distance: float = 0.25
    max_iteration: int = 40
    pyramid_scales: tuple[float, ...] = (4.0, 2.0, 1.0)


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


def symmetric_registration_metrics(source_points, target_points, transformation,
                                   max_correspondence_distance):
    """Evaluate overlap in both directions; one-sided scores hide facade drift."""
    forward = registration_metrics(source_points, target_points, transformation,
                                    max_correspondence_distance)
    t = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    inverse = np.linalg.inv(t)
    backward = registration_metrics(target_points, source_points, inverse,
                                    max_correspondence_distance)
    finite = np.isfinite([forward['rmse'], backward['rmse']]).all()
    return {
        'fitness': float((forward['fitness'] + backward['fitness']) / 2.0),
        'forward_fitness': forward['fitness'], 'backward_fitness': backward['fitness'],
        'rmse': float(np.sqrt((forward['rmse'] ** 2 + backward['rmse'] ** 2) / 2.0))
                if finite else float('inf'),
        'forward_rmse': forward['rmse'], 'backward_rmse': backward['rmse'],
        'p95_error': float(max(forward['p95_error'], backward['p95_error'])),
        'correspondence_count': int(min(forward['correspondence_count'],
                                        backward['correspondence_count'])),
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


def _cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)))


def _prepare_feature_cloud(points, voxel_size):
    """Build the downsampled cloud and FPFH once for a station pair."""
    voxel = max(float(voxel_size), 1e-3)
    down = _cloud(points).voxel_down_sample(voxel)
    if len(down.points) < 10:
        raise ValueError(f'FPFH 特征点不足: {len(down.points)} < 10')
    radius_normal = voxel * 2.5
    radius_feature = voxel * 5.0
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=50))
    down.normalize_normals()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
    return down, feature


def fpfh_global_registration(source_points, target_points, *, voxel_size=0.10,
                             max_correspondence_distance=None,
                             ransac_iterations=100000, confidence=0.999,
                             logger=None):
    """Global registration using FPFH + mutual RANSAC correspondence matching."""
    started = time.perf_counter()
    source, source_feature = _prepare_feature_cloud(source_points, voxel_size)
    target, target_feature = _prepare_feature_cloud(target_points, voxel_size)
    distance = float(max_correspondence_distance or voxel_size * 1.5)
    if logger:
        logger('features', source_points=len(source.points), target_points=len(target.points),
               voxel=voxel_size, elapsed_ms=round((time.perf_counter()-started)*1000, 2))
    checker = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        int(ransac_iterations), float(confidence))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, checker, criteria)
    report = {'fitness': float(result.fitness), 'rmse': float(result.inlier_rmse),
              'correspondences': len(result.correspondence_set),
              'iterations': int(ransac_iterations),
              'elapsed_ms': round((time.perf_counter()-started)*1000, 2)}
    if logger:
        logger('ransac', **report)
    return np.asarray(result.transformation), report


def _transform_summary(transformation):
    t = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    angle = np.arccos(np.clip((np.trace(t[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return {'translation': [round(float(x), 6) for x in t[:3, 3]],
            'translation_norm': round(float(np.linalg.norm(t[:3, 3])), 6),
            'rotation_deg': round(float(np.degrees(angle)), 6)}


def auto_register(source_points, target_points, *, voxel_size=0.05,
                  global_voxel_size=0.10, max_correspondence_distance=0.25,
                  max_iteration=40, logger=None):
    """FPFH/RANSAC candidates followed by fine ICP and symmetric acceptance."""
    diagnostics = []
    try:
        fpfh_t, fpfh_report = fpfh_global_registration(
            source_points, target_points, voxel_size=global_voxel_size,
            max_correspondence_distance=global_voxel_size * 1.5,
            logger=logger)
    except Exception as exc:
        if logger:
            logger('rejected', candidate='fpfh', reason=str(exc))
        raise
    candidates = [('gps', np.eye(4, dtype=np.float64)), ('fpfh', fpfh_t)]
    for name, initial in candidates:
        started = time.perf_counter()
        if logger:
            logger('candidate', name=name, **_transform_summary(initial))
        result = point_to_plane_icp(
            source_points, target_points, init=initial, voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            max_iteration=max_iteration, pyramid_scales=(4.0, 2.0, 1.0))
        quality = symmetric_registration_metrics(
            source_points, target_points, result.transformation,
            max_correspondence_distance)
        item = {'name': name, 'transformation': result.transformation.tolist(),
                **quality, **_transform_summary(result.transformation),
                'levels': result.levels,
                'elapsed_ms': round((time.perf_counter()-started)*1000, 2)}
        item['accepted'] = bool(result.accepted)
        diagnostics.append(item)
        if logger:
            logger('quality', name=name, fitness=item['fitness'],
                   forward_fitness=item['forward_fitness'],
                   backward_fitness=item['backward_fitness'], rmse=item['rmse'],
                   p95_error=item['p95_error'], accepted=result.accepted,
                   elapsed_ms=item['elapsed_ms'])
    valid = [x for x in diagnostics if np.isfinite(x['rmse']) and
             x['fitness'] >= 0.05 and x['forward_fitness'] >= 0.03 and
             x['backward_fitness'] >= 0.03 and x['p95_error'] <= max_correspondence_distance]
    if not valid:
        if logger:
            logger('rejected', candidate='all', reason='bidirectional_quality_threshold')
        raise ValueError('FPFH/GPS 候选均未通过双向配准质量验收')
    chosen = min(valid, key=lambda x: (x['rmse'], -x['fitness'], x['p95_error']))
    if logger:
        logger('selected', name=chosen['name'], rmse=chosen['rmse'],
               fitness=chosen['fitness'], **_transform_summary(chosen['transformation']))
    return ICPResult(np.asarray(chosen['transformation'], dtype=np.float64), chosen['fitness'],
                     chosen['rmse'], chosen['correspondence_count'], diagnostics,
                     True, ''), {'fpfh': fpfh_report, 'candidates': diagnostics,
                                 'selected': chosen['name']}


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


def _xy_pca_frame(points):
    """Return a stable 2-D principal frame for a facade proxy cloud."""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 3)[:, :2]
    # The same unordered point set can still use its centroid exactly after a
    # rigid transform; unlike a component-wise median, it does not introduce
    # an orientation-dependent bias.
    center = np.mean(xy, axis=0)
    centered = xy - center
    if len(centered) < 2:
        return center, np.eye(2)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    frame = vt.T
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return center, frame


def estimate_xy_initial_transform(source_points, target_points):
    """Estimate a GPS-residual rigid initial guess from XY geometry.

    FLS PLY files are already in the global frame.  This is deliberately only
    an *initial guess*: it does not apply either station's JSON transform.
    PCA has a 180 degree ambiguity, so all four axis sign combinations are
    evaluated using a cheap sampled nearest-neighbour score.  Z is translated
    by the robust median difference and is never rotated independently.
    """
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError('粗配准至少需要两组各 3 个有效点')
    sc, sf = _xy_pca_frame(src)
    tc, tf = _xy_pca_frame(tgt)
    # Work on a bounded sample so large facade clouds remain interactive.
    s = src[::max(1, len(src) // 4000)]
    t = tgt[::max(1, len(tgt) // 8000)]
    best = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            # Reflection in the PCA coordinates is converted back to a proper
            # 3-D rotation by selecting only determinant +1 candidates.
            q = np.diag([sx, sy])
            r2 = tf @ q @ sf.T
            if np.linalg.det(r2) < 0:
                continue
            r = np.eye(3)
            r[:2, :2] = r2
            trans = np.zeros(3, dtype=np.float64)
            trans[:2] = tc - r2 @ sc
            trans[2] = np.median(t[:, 2]) - np.median(s[:, 2])
            moved = s @ r.T + trans
            tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(t)))
            distances = []
            for point in moved:
                count, _, d2 = tree.search_knn_vector_3d(point, 1)
                if count:
                    distances.append(d2[0])
            score = float(np.median(distances)) if distances else float('inf')
            if best is None or score < best[0]:
                best = (score, r, trans)
    if best is None:
        raise ValueError('无法从 XY 几何建立粗配准初值')
    out = np.eye(4, dtype=np.float64)
    out[:3, :3], out[:3, 3] = best[1], best[2]
    return out


def manual_seeded_icp(source_points, target_points, source_correspondences,
                       target_correspondences, *, voxel_size=0.05,
                       max_iteration=30, max_correspondence_distance=0.12,
                       pyramid_scales=(4.0, 2.0, 1.0)):
    """Global-coordinate ICP seeded by optional operator correspondences."""
    src_pairs = np.asarray(source_correspondences, dtype=np.float64).reshape(-1, 3)
    tgt_pairs = np.asarray(target_correspondences, dtype=np.float64).reshape(-1, 3)
    init = rigid_transform_from_correspondences(src_pairs, tgt_pairs)
    result = point_to_plane_icp(source_points, target_points, init=init,
                                voxel_size=voxel_size, max_iteration=max_iteration,
                                max_correspondence_distance=max_correspondence_distance,
                                pyramid_scales=pyramid_scales)
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
        level_started = time.perf_counter()
        source = base_source.voxel_down_sample(scale)
        target = base_target.voxel_down_sample(scale)
        radius = max(scale * 2.5, 1e-3)
        for cloud in (source, target):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
            cloud.normalize_normals()
        # A fixed large radius at the fine level admits facade-to-facade false
        # matches.  Keep the caller's coarse bound, but tighten each pyramid
        # level as the solution approaches the GPS residual.
        distance = (min(float(max_correspondence_distance), max(scale * 2.5, 0.015))
                    if max_correspondence_distance is not None
                    else max(scale * 2.5, 0.015))
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
            if scale > scales[-1] * 1.01 else
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=int(max_iteration))
        final = o3d.pipelines.registration.registration_icp(
            source, target, distance, T, estimation, criteria)
        T = np.asarray(final.transformation)
        reports.append({'voxel': scale, 'fitness': float(final.fitness),
                        'rmse': float(final.inlier_rmse),
                        'correspondences': len(final.correspondence_set),
                        'estimator': ('point_to_point' if scale > scales[-1] * 1.01
                                      else 'point_to_plane'),
                        'elapsed_ms': round((time.perf_counter()-level_started)*1000, 2)})

    metric_distance = (float(max_correspondence_distance)
                       if max_correspondence_distance is not None
                       else max(scales[-1] * 2.5, 0.01))
    checked = registration_metrics(src, tgt, T,
                                   max_correspondence_distance=metric_distance)
    fitness = checked['fitness']
    rmse = checked['rmse']
    correspondence_count = checked['correspondence_count']
    min_correspondences = max(3, min(50, int(np.ceil(min(len(src), len(tgt)) * 0.005))))
    accepted = (fitness >= 0.01 and correspondence_count >= min_correspondences
                and np.isfinite(rmse))
    message = '' if accepted else '有效对应不足或配准质量未达到门限'
    reports.append(checked)
    return ICPResult(T, fitness, rmse, correspondence_count, reports, accepted, message)