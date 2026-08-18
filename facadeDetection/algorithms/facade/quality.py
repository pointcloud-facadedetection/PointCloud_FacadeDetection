from __future__ import annotations

import numpy as np

from algorithms.geometry import classify_plane, plane_axes


def _empty_quality_result(interval_size_m: float, flatness_limit: float,
                          verticality_limit_mm: float) -> dict:
    """返回空结果模板"""
    verticality_limit_deg = float(np.degrees(np.arctan2(
        float(verticality_limit_mm) / 1000.0, 2.0
    )))
    return {
        'interval_size_m': float(interval_size_m),
        'interval_origin_m': 0.0,
        'window_size_m': 0.05,
        'step_size_m': 0.05,
        'projection_origin': [0.0, 0.0, 0.0],
        'projection_u_axis': [1.0, 0.0, 0.0],
        'projection_v_axis': [0.0, 1.0, 0.0],
        'projection_u_min_m': 0.0,
        'projection_v_min_m': 0.0,
        'thresholds': {
            'flatness_limit_mm': float(flatness_limit) * 1000.0,
            'verticality_limit_deg': verticality_limit_deg,
            'verticality_limit_mm': float(verticality_limit_mm),
        },
        'overall': {
            'type': 'unknown',
            'type_label': '平面',
            'plane_model': [0.0, 0.0, 1.0, 0.0],
            'normal': [0.0, 0.0, 1.0],
            'point_count': 0,
            'flatness_max_gap_mm': 0.0,
            'flatness_pass_rate': 1.0,
            'verticality_max_angle_deg': 0.0,
            'verticality_max_deviation_mm': 0.0,
            'verticality_pass_rate': 1.0,
            'global_z_min_m': 0.0,
            'global_z_max_m': 0.0,
            'global_z_span_m': 0.0,
        },
        'intervals': [],
        'windows': {
            'window_id': np.asarray([], dtype=np.int32),
            'cell_u': np.asarray([], dtype=np.int32),
            'cell_v': np.asarray([], dtype=np.int32),
            'interval_id': np.asarray([], dtype=np.int32),
            'point_count': np.asarray([], dtype=np.int32),
            'center_xyz': np.empty((0, 3), dtype=np.float32),
            'center_uv': np.empty((0, 2), dtype=np.float32),
            'flatness_gap_mm': np.asarray([], dtype=np.float32),
            'flatness_signed_gap_mm': np.asarray([], dtype=np.float32),
            'verticality_angle_deg': np.asarray([], dtype=np.float32),
            'verticality_deviation_mm_2m': np.asarray([], dtype=np.float32),
            'flatness_pass': np.asarray([], dtype=bool),
            'verticality_pass': np.asarray([], dtype=bool),
        },
        'quality_source': 'raw_facade_subset',
        'flatness_limit': float(flatness_limit),
        'verticality_limit_mm': float(verticality_limit_mm),
        'verticality_limit_deg': verticality_limit_deg,
    }


def _build_windows_by_sort(points: np.ndarray, center: np.ndarray,
                           u_axis: np.ndarray, v_axis: np.ndarray,
                           step_size: float, min_points: int) -> tuple[list[dict], float, float]:
    """排序分组构建窗口（高性能核心，无空窗口遍历）。

    复杂度: O(N log N) 排序 + O(N) 分组。
    """
    uv = np.column_stack(((points - center) @ u_axis,
                         (points - center) @ v_axis)).astype(np.float64)
    u_min, v_min = uv.min(axis=0)

    u_idx = np.floor((uv[:, 0] - u_min) / step_size).astype(np.int32)
    v_idx = np.floor((uv[:, 1] - v_min) / step_size).astype(np.int32)

    max_v = int(v_idx.max()) + 2
    cell_key = u_idx * max_v + v_idx

    sort_order = np.argsort(cell_key, kind='mergesort')
    sorted_key = cell_key[sort_order]

    key_changes = np.concatenate([[True], np.diff(sorted_key) != 0])
    group_starts = np.flatnonzero(key_changes)
    group_ends = np.concatenate([group_starts[1:], [len(sorted_key)]])

    group_sizes = group_ends - group_starts
    valid_mask = group_sizes >= min_points

    if not np.any(valid_mask):
        return [], float(u_min), float(v_min)

    valid_starts = group_starts[valid_mask]
    valid_ends = group_ends[valid_mask]

    sample_positions = sort_order[valid_starts]
    u_centers = u_min + (u_idx[sample_positions].astype(np.float64) + 0.5) * step_size
    v_centers = v_min + (v_idx[sample_positions].astype(np.float64) + 0.5) * step_size

    windows = []
    for i, (s, e) in enumerate(zip(valid_starts, valid_ends)):
        windows.append({
            'center_u': float(u_centers[i]),
            'center_v': float(v_centers[i]),
            'point_ids': sort_order[s:e].astype(np.int32),
        })

    return windows, float(u_min), float(v_min)


def compute_facade_quality(facade_info: dict, pcd,
                           grid_size: float = 20.0,
                           flatness_limit: float = 0.004,
                           verticality_limit_mm: float = 4.0,
                           ruler_size: float = 0.05,
                           ruler_step: float = 0.05,
                           verticality_limit_deg: float | None = None,
                           min_points: int = 3,
                           measure_height: float = 2.0,
                           interval_origin_m: float = 0.0) -> dict:
    """质量域纯统计：5cm×5cm 滑动窗口遍历和统计，不重新拟合平面。

    【核心原则】检测阶段确定的 facade.plane_model 是唯一基准面，
    质量阶段禁止修改平面参数。
    """
    points = np.asarray(pcd.points, dtype=np.float64).reshape(-1, 3)
    interval_size = max(float(grid_size), 1e-6)
    interval_origin = float(interval_origin_m)
    window_size = max(float(ruler_size or 0.05), 1e-6)
    step_size = max(float(ruler_step or window_size), 1e-6)

    if len(points) == 0:
        return _empty_quality_result(interval_size, flatness_limit, verticality_limit_mm)

    # Step 1: 直接使用传入的基准面
    plane = np.asarray(facade_info.get('plane_model') or [], dtype=np.float64)
    if plane.size != 4 or not np.all(np.isfinite(plane)):
        return _empty_quality_result(interval_size, flatness_limit, verticality_limit_mm)

    norm = np.linalg.norm(plane[:3])
    if abs(norm - 1.0) > 1e-6:
        plane[:3] = plane[:3] / norm
        plane[3] = float(plane[3]) / norm

    facade_normal = plane[:3].copy()
    center = np.asarray(facade_info.get('center', points.mean(axis=0)), dtype=np.float64)
    facade_type, type_label, _facade_verticality, horizontality = classify_plane(facade_normal)
    u_axis, v_axis = plane_axes(facade_normal, facade_type)

    # Step 2: 预计算有符号距离
    signed_distances = points @ facade_normal + plane[3]

    # Step 3: 国标垂直度全局计算
    nz = float(abs(facade_normal[2]))
    verticality_angle_global = float(np.degrees(np.arcsin(np.clip(nz, 0.0, 1.0))))
    measure_h = max(float(measure_height), 1e-6)
    verticality_deviation_mm_global = float(
        np.tan(np.radians(verticality_angle_global)) * measure_h * 1000.0
    )

    if verticality_limit_deg is None:
        verticality_limit_deg = float(np.degrees(np.arctan2(
            float(verticality_limit_mm) / 1000.0, measure_h
        )))

    # Step 4: 排序分组构建窗口
    windows, u_min, v_min = _build_windows_by_sort(
        points, center, u_axis, v_axis, step_size, min_points
    )

    # 距离分层代理点并不保证每个 5cm 单元有 3 个原始点；若严格按
    # min_points 丢弃，稀疏高层会得到零窗口。先保留稀疏单元，再由窗口
    # 统计结果标记质量，避免“有质量域但无法导出”的假失败。
    if not windows and min_points > 1:
        windows, u_min, v_min = _build_windows_by_sort(
            points, center, u_axis, v_axis, step_size, 1
        )

    if not windows:
        result = _empty_quality_result(interval_size, flatness_limit, verticality_limit_mm)
        result['overall']['plane_model'] = plane.astype(float).tolist()
        result['overall']['normal'] = facade_normal.astype(float).tolist()
        result['overall']['type'] = facade_type
        result['overall']['type_label'] = type_label
        result['z_min_m'] = float(np.min(points[:, 2]))
        result['z_max_m'] = float(np.max(points[:, 2]))
        result['interval_count'] = max(1, int(np.ceil(
            (np.floor(result['z_max_m'] / interval_size) -
             np.floor(result['z_min_m'] / interval_size)) + 1
        )))
        result['interval_origin_m'] = interval_origin
        result['first_interval_id'] = int(np.floor(result['z_min_m'] / interval_size))
        result['last_interval_id'] = int(np.floor(result['z_max_m'] / interval_size))
        result['overall']['global_z_min_m'] = result['z_min_m']
        result['overall']['global_z_max_m'] = result['z_max_m']
        result['overall']['global_z_span_m'] = result['z_max_m'] - result['z_min_m']
        result['projection_origin'] = center.astype(float).tolist()
        result['projection_u_axis'] = u_axis.astype(float).tolist()
        result['projection_v_axis'] = v_axis.astype(float).tolist()
        result['projection_u_min_m'] = float(u_min)
        result['projection_v_min_m'] = float(v_min)
        return result

    # Step 5: 逐窗口计算（纯统计，无拟合）
    flat_limit_mm = float(flatness_limit) * 1000.0
    n_windows = len(windows)

    # 预分配数组
    interval_ids_arr = np.empty(n_windows, dtype=np.int32)
    point_counts_arr = np.empty(n_windows, dtype=np.int32)
    center_xyz_arr = np.empty((n_windows, 3), dtype=np.float32)
    center_uv_arr = np.empty((n_windows, 2), dtype=np.float32)
    flatness_gap_mm_arr = np.empty(n_windows, dtype=np.float32)
    flatness_signed_gap_mm_arr = np.empty(n_windows, dtype=np.float32)
    verticality_angle_deg_arr = np.empty(n_windows, dtype=np.float32)
    verticality_deviation_mm_2m_arr = np.empty(n_windows, dtype=np.float32)
    flatness_pass_arr = np.empty(n_windows, dtype=bool)
    verticality_pass_arr = np.empty(n_windows, dtype=bool)
    center_z_arr = np.empty(n_windows, dtype=np.float32)

    for i, w in enumerate(windows):
        point_ids = w['point_ids']
        cell_points = points[point_ids]
        cell_signed_dists = signed_distances[point_ids]

        # 平整度：窗口内最差有符号间隙
        worst_idx = int(np.argmax(np.abs(cell_signed_dists)))
        signed_gap_m = float(cell_signed_dists[worst_idx])
        flatness_signed_gap_mm_val = signed_gap_m * 1000.0
        flatness_gap_mm_val = abs(flatness_signed_gap_mm_val)

        # 垂直度：所有窗口共享立面整体属性
        verticality_deviation_deg = verticality_angle_global
        verticality_deviation_mm_2m_val = verticality_deviation_mm_global

        point_counts_arr[i] = len(point_ids)
        center_xyz_arr[i] = np.mean(cell_points, axis=0)
        center_z_arr[i] = center_xyz_arr[i, 2]
        center_uv_arr[i] = [w['center_u'], w['center_v']]
        flatness_gap_mm_arr[i] = flatness_gap_mm_val
        flatness_signed_gap_mm_arr[i] = flatness_signed_gap_mm_val
        verticality_angle_deg_arr[i] = verticality_deviation_deg
        verticality_deviation_mm_2m_arr[i] = verticality_deviation_mm_2m_val
        flatness_pass_arr[i] = flatness_gap_mm_val <= flat_limit_mm
        verticality_pass_arr[i] = verticality_deviation_deg <= float(verticality_limit_deg)

    # 区间必须依据世界坐标 Z，而不是任意平面局部 v 轴；v 轴通常是水平方向。
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    first_interval = int(np.floor((z_min - interval_origin) / interval_size))
    last_interval = int(np.floor((z_max - interval_origin) / interval_size))
    interval_ids_arr[:] = np.floor(
        (center_z_arr.astype(np.float64) - interval_origin) / interval_size
    ).astype(np.int32)
    # 严格 clamp 到 [first_interval, last_interval]，避免浮点/原点偏移导致越界
    interval_ids_arr = np.clip(interval_ids_arr, first_interval, last_interval)
    # 统计越界/空窗口情况，便于排查映射错位
    orphan_count = int(np.sum((interval_ids_arr < first_interval) | (interval_ids_arr > last_interval)))
    interval_count = max(1, last_interval - first_interval + 1)

    # Step 6: 区间统计
    unique_intervals = np.arange(first_interval, last_interval + 1, dtype=np.int32)
    interval_rows = []

    for iid in unique_intervals.tolist():
        wm = interval_ids_arr == int(iid)
        interval_signed = flatness_signed_gap_mm_arr[wm]
        z0 = interval_origin + int(iid) * interval_size
        z1 = z0 + interval_size

        if not np.any(wm):
            interval_rows.append({
                'interval_id': int(iid), 'z_min_m': float(z0), 'z_max_m': float(z1),
                'label': f'{z0:.2f}–{z1:.2f}m', 'point_count': 0,
                'window_count': 0, 'coverage_status': 'no_measurement_points',
                'flatness_max_gap_mm': 0.0, 'flatness_max_recessed_mm': 0.0,
                'flatness_max_protruding_mm': 0.0, 'flatness_pass_count': 0,
                'flatness_pass_rate': 0.0, 'verticality_max_angle_deg': 0.0,
                'verticality_max_deviation_mm_2m': 0.0, 'verticality_pass_count': 0,
                'verticality_pass_rate': 0.0,
            })
            continue

        interval_rows.append({
            'interval_id': int(iid),
            'z_min_m': float(z0),
            'z_max_m': float(z1),
            'label': f'{z0:.2f}–{z1:.2f}m',
            'coverage_status': 'measured',
            'point_count': int(np.sum(point_counts_arr[wm])),
            'window_count': int(np.sum(wm)),
            'flatness_max_gap_mm': float(np.max(flatness_gap_mm_arr[wm])),
            'flatness_max_recessed_mm': float(np.min(interval_signed)),
            'flatness_max_protruding_mm': float(np.max(interval_signed)),
            'flatness_pass_count': int(np.sum(flatness_pass_arr[wm])),
            'flatness_pass_rate': float(np.mean(flatness_pass_arr[wm])),
            'verticality_max_angle_deg': float(np.max(verticality_angle_deg_arr[wm])),
            'verticality_max_deviation_mm_2m': float(np.max(verticality_deviation_mm_2m_arr[wm])),
            'verticality_pass_count': int(np.sum(verticality_pass_arr[wm])),
            'verticality_pass_rate': float(np.mean(verticality_pass_arr[wm])),
        })

    # Step 7: 组装结果
    result = {
        'interval_size_m': interval_size,
        'window_size_m': window_size,
        'step_size_m': step_size,
        'projection_origin': center.astype(float).tolist(),
        'projection_u_axis': u_axis.astype(float).tolist(),
        'projection_v_axis': v_axis.astype(float).tolist(),
        'projection_u_min_m': float(u_min),
        'projection_v_min_m': float(v_min),
        'thresholds': {
            'flatness_limit_mm': flat_limit_mm,
            'verticality_limit_deg': float(verticality_limit_deg),
            'verticality_limit_mm': float(verticality_limit_mm),
        },
        'overall': {
            'type': facade_type,
            'type_label': type_label,
            'normal': facade_normal.astype(float).tolist(),
            'plane_model': plane.astype(float).tolist(),
            'horizontality': float(horizontality),
            'point_count': int(len(points)),
            'flatness_max_gap_mm': float(np.max(flatness_gap_mm_arr)),
            'flatness_max_recessed_mm': float(np.min(flatness_signed_gap_mm_arr)),
            'flatness_max_protruding_mm': float(np.max(flatness_signed_gap_mm_arr)),
            'flatness_pass_rate': float(np.mean(flatness_pass_arr)),
            'verticality_max_angle_deg': float(np.max(verticality_angle_deg_arr)),
            'verticality_max_deviation_mm': float(np.max(verticality_deviation_mm_2m_arr)),
            'verticality_pass_rate': float(np.mean(verticality_pass_arr)),
            'global_z_min_m': z_min,
            'global_z_max_m': z_max,
            'global_z_span_m': z_max - z_min,
        },
        'intervals': interval_rows,
        'z_min_m': z_min,
        'z_max_m': z_max,
        'interval_count': interval_count,
        'interval_origin_m': interval_origin,
        'first_interval_id': first_interval,
        'last_interval_id': last_interval,
        'windows': {
            'window_id': np.arange(n_windows, dtype=np.int32),
            'cell_u': np.asarray([int(np.floor((w['center_u'] - u_min) / step_size)) for w in windows], dtype=np.int32),
            'cell_v': np.asarray([int(np.floor((w['center_v'] - v_min) / step_size)) for w in windows], dtype=np.int32),
            'interval_id': interval_ids_arr,
            'point_count': point_counts_arr,
            'center_xyz': center_xyz_arr,
            'center_uv': center_uv_arr,
            'center_z': center_z_arr,
            'flatness_gap_mm': flatness_gap_mm_arr,
            'flatness_signed_gap_mm': flatness_signed_gap_mm_arr,
            'verticality_angle_deg': verticality_angle_deg_arr,
            'verticality_deviation_mm_2m': verticality_deviation_mm_2m_arr,
            'flatness_pass': flatness_pass_arr,
            'verticality_pass': verticality_pass_arr,
        },
        'quality_source': 'raw_facade_subset',
        'flatness_limit': float(flatness_limit),
        'verticality_limit_mm': float(verticality_limit_mm),
        'verticality_limit_deg': float(verticality_limit_deg),
    }
    return result