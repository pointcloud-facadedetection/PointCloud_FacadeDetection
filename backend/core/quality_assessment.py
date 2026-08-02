import numpy as np

from .geometry_utils import (
    fit_plane_svd, classify_plane, plane_axes,
    signed_plane_distance,
    vertical_deviation_from_normal
)


def _fit_plane_svd(points):
    """最小二乘 SVD 拟合平面（纯 NumPy，供并行调用）"""
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal) + 1e-12
    normal = normal / norm
    d = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], normal[2], d], dtype=float)


def sliding_window_flatness_2d(points, center, u_axis, v_axis, reference_plane,
                               window_size=2.0, step_size=0.05, min_points=30,
                               max_windows=6000):
    """
    2m x 2m 滑动窗口平整度检测 —— 区间基准平面 + 局部平面偏离方案。

    核心逻辑：
    1. 用当前20m区间内所有点拟合"区间基准平面"
    2. 在区间内以 2m×2m、5cm步长滑动窗口
    3. 每个窗口内拟合"局部平面"
    4. 计算局部平面中心到区间基准平面的距离作为该窗口的偏离值
    5. 取所有窗口中的最大偏离值作为该区间的平整度指标
    """
    if len(points) < min_points:
        return None

    # Step 1: 拟合区间基准平面
    interval_plane = _fit_plane_svd(points)
    interval_normal = interval_plane[:3]
    interval_normal = interval_normal / (np.linalg.norm(interval_normal) + 1e-12)
    interval_d = interval_plane[3]

    # 法向方向对齐
    ref_normal = reference_plane[:3]
    ref_normal = ref_normal / (np.linalg.norm(ref_normal) + 1e-12)
    if np.dot(interval_normal, ref_normal) < 0:
        interval_normal = -interval_normal
        interval_d = -interval_d
    interval_plane = np.array([interval_normal[0], interval_normal[1], interval_normal[2], interval_d])

    # Step 2: 建立局部2D坐标系
    local_u = np.dot(points - center, u_axis)
    local_v = np.dot(points - center, v_axis)

    u_min, u_max = float(np.min(local_u)), float(np.max(local_u))
    v_min, v_max = float(np.min(local_v)), float(np.max(local_v))

    u_starts = np.arange(u_min, u_max - window_size + 1e-9, step_size)
    v_starts = np.arange(v_min, v_max - window_size + 1e-9, step_size)

    if len(u_starts) == 0:
        u_starts = np.array([(u_min + u_max - window_size) * 0.5])
    if len(v_starts) == 0:
        v_starts = np.array([(v_min + v_max - window_size) * 0.5])

    # 窗口数量限制
    total_windows = len(u_starts) * len(v_starts)
    if total_windows > max_windows:
        scale = np.sqrt(total_windows / max_windows)
        u_starts = u_starts[::max(1, int(np.ceil(scale)))]
        v_starts = v_starts[::max(1, int(np.ceil(scale)))]

    # Step 3: 滑动窗口遍历
    worst_gap = 0.0
    worst_info = None

    for vs in v_starts:
        v_low = float(vs)
        v_high = v_low + window_size
        v_mask = (local_v >= v_low) & (local_v <= v_high)

        for us in u_starts:
            u_low = float(us)
            u_high = u_low + window_size
            mask = v_mask & (local_u >= u_low) & (local_u <= u_high)
            count = int(np.sum(mask))

            if count < min_points:
                continue

            window_points = points[mask]
            window_center = np.mean(window_points, axis=0)

            # 拟合局部平面
            local_plane = _fit_plane_svd(window_points)
            local_normal = local_plane[:3]
            local_normal = local_normal / (np.linalg.norm(local_normal) + 1e-12)

            # 法向方向对齐
            if np.dot(local_normal, interval_normal) < 0:
                local_normal = -local_normal

            # 核心指标：局部平面中心到区间基准平面的距离
            gap = float(np.abs(np.dot(interval_normal, window_center) + interval_d))

            if gap > worst_gap:
                worst_gap = gap
                worst_info = {
                    'gap': float(gap),
                    'center_u': float(u_low + window_size * 0.5),
                    'center_v': float(v_low + window_size * 0.5),
                    'point_count': count,
                    'u_range': [float(u_low), float(u_high)],
                    'v_range': [float(v_low), float(v_high)],
                    'window_center': [float(x) for x in window_center],
                }

    if worst_info is None:
        return None

    return worst_gap, worst_info


def local_flatness_metrics(points, reference_normal=None, measure_height=2.0):
    """计算局部平整度指标"""
    if len(points) < 3:
        return None

    plane_model = fit_plane_svd(points)
    normal = plane_model[:3]

    if reference_normal is not None and np.dot(normal, reference_normal) < 0:
        plane_model *= -1.0
        normal = plane_model[:3]

    signed = points @ normal + plane_model[3]
    distances = np.abs(signed)

    facade_type, _, _, horizontality = classify_plane(normal)
    verticality_deg, vertical_dev_m, vertical_dev_mm = vertical_deviation_from_normal(normal, measure_height)

    return {
        'plane_model': plane_model,
        'normal': normal,
        'verticality': verticality_deg,
        'horizontality': float(horizontality),
        'vertical_deviation_2m': vertical_dev_m,
        'vertical_deviation_2m_mm': vertical_dev_mm,
        'flatness_mean': float(np.mean(distances)),
        'flatness_max': float(np.max(distances)),
        'flatness_gap': float(np.max(signed) - np.min(signed)),
        'flatness_std': float(np.std(distances)),
        'flatness_rmse': float(np.sqrt(np.mean(distances ** 2))),
        'type': facade_type
    }


def compute_facade_quality(facade_info, pcd, grid_size=20.0,
                           flatness_limit=0.004,
                           verticality_limit_mm=4.0,
                           ruler_size=2.0,
                           ruler_step=0.05):
    """
    在下采样立面点云上计算质量。
    垂直度仅保留夹角；平整度采用每 20m 区间内 2m x 2m 模拟靠尺滑动窗口的严格最大间隙。
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return _empty_quality_result(grid_size, flatness_limit, verticality_limit_mm)

    facade_plane = np.asarray(facade_info.get('plane_model', []), dtype=float)
    if facade_plane.size != 4:
        facade_plane = fit_plane_svd(points)
    normal = facade_plane[:3]
    norm = np.linalg.norm(normal) + 1e-12
    facade_plane = facade_plane / norm
    normal = facade_plane[:3]

    center = np.asarray(facade_info.get('center', np.mean(points, axis=0)), dtype=float)
    facade_type, type_label, verticality_deg, horizontality = classify_plane(normal)
    u_axis, v_axis = plane_axes(normal, facade_type)

    local_u = np.dot(points - center, u_axis)
    local_v = np.dot(points - center, v_axis)
    u_min, u_max = float(np.min(local_u)), float(np.max(local_u))
    v_min, v_max = float(np.min(local_v)), float(np.max(local_v))
    v_bins = max(1, int(np.ceil((v_max - v_min) / grid_size)))

    grids = []
    for j in range(v_bins):
        v_low = v_min + j * grid_size
        v_high = v_min + (j + 1) * grid_size
        if j == v_bins - 1:
            v_high = v_max + 1e-9

        mask = (local_v >= v_low) & (local_v < v_high)
        grid_point_count = int(np.sum(mask))
        if grid_point_count < 30:
            continue

        grid_points = points[mask]
        grid_center = np.mean(grid_points, axis=0)
        window_result = sliding_window_flatness_2d(
            grid_points,
            grid_center,
            u_axis,
            v_axis,
            facade_plane,
            window_size=ruler_size,
            step_size=ruler_step,
            min_points=30
        )

        if window_result is None:
            signed = signed_plane_distance(grid_points, facade_plane)
            gap = float(np.max(signed) - np.min(signed))
            window_info = None
        else:
            gap, window_info = window_result

        compliance = gap <= flatness_limit
        grid_data = {
            'grid_id': int(len(grids)),
            'center': [float(x) for x in grid_center],
            'verticality': float(verticality_deg),
            'horizontality': float(horizontality),
            'gap': float(gap),
            'point_count': grid_point_count,
            'compliance': bool(compliance),
            'v_range': [float(v_low), float(v_high)],
            'u_range': [float(u_min), float(u_max)],
        }
        if window_info:
            grid_data['worst_window'] = window_info

        grids.append(grid_data)

    overall_gap = max((g['gap'] for g in grids), default=0.0)

    return {
        'overall': {
            'type': facade_type,
            'type_label': type_label,
            'normal': [float(x) for x in normal],
            'plane_model': [float(x) for x in facade_plane],
            'verticality': float(verticality_deg),
            'horizontality': float(horizontality),
            'gap': float(overall_gap),
            'point_count': int(len(points)),
        },
        'grids': grids,
        'grid_size': grid_size,
        'ruler_size': ruler_size,
        'ruler_step': ruler_step,
        'flatness_limit': flatness_limit,
        'verticality_limit_mm': verticality_limit_mm,
        'quality_source': 'downsampled',
        'flatness_metric': 'local plane center deviation from interval baseline plane in 2m x 2m sliding windows'
    }


def _empty_quality_result(grid_size, flatness_limit, verticality_limit_mm):
    return {
        'overall': {
            'type': 'unknown',
            'type_label': '平面',
            'verticality': 0.0,
            'horizontality': 0.0,
            'gap': 0.0,
            'point_count': 0
        },
        'grids': [],
        'grid_size': grid_size,
        'ruler_size': 2.0,
        'flatness_limit': flatness_limit,
        'verticality_limit_mm': verticality_limit_mm,
        'quality_source': 'downsampled'
    }