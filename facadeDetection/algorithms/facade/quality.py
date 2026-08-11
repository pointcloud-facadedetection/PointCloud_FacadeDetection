import numpy as np

from algorithms.geometry import (
        fit_plane_svd, classify_plane, plane_axes,
        signed_plane_distance,
        vertical_deviation_from_normal
    )
from config.settings import Config

def compute_window_quality_distribution(window_results):
    """统计窗口质量分布"""
    dist = {Config.QUALITY_PASS: 0, Config.QUALITY_WARN: 0, Config.QUALITY_FAIL: 0}
    for wr in window_results:
        dist[wr['quality_level']] = dist.get(wr['quality_level'], 0) + 1
    return {
        'pass': dist[Config.QUALITY_PASS],
        'warn': dist[Config.QUALITY_WARN],
        'fail': dist[Config.QUALITY_FAIL],
    }


def generate_quality_colors(point_quality_labels, total_points):
    """
    生成质量颜色数组
    - 合格(QUALITY_PASS): 绿色
    - 警告(QUALITY_WARN): 黄色
    - 不合格(QUALITY_FAIL): 红色
    - 未评估(-1): 灰色（不参与评估的点）
    """
    colors = np.zeros((total_points, 3), dtype=float)
    for i in range(total_points):
        label = point_quality_labels[i]
        if label == -1:
            colors[i] = np.array([0.5, 0.5, 0.5])  # 灰色 - 未评估
        else:
            colors[i] = Config.QUALITY_COLORS.get(label, np.array([0.5, 0.5, 0.5]))
    return colors


def generate_sparse_defect_heatmap(signed_gap, flatness_limit, color_max=None,
                                   cmap='turbo', candidate_mask=None):
    """只为超出平整度限值的点生成连续色带。

    返回局部点索引掩码、缺陷值和 RGB 颜色；合格点不进入颜色计算，
    从而可以在上层保留点云原色。gap 单位为米。
    """
    values = np.asarray(signed_gap, dtype=np.float32).reshape(-1)
    limit = max(float(flatness_limit), 0.0)
    abs_values = np.abs(values)
    # candidate_mask 由窗口判定产生。传入时，热力图不再自行按“全局点距”
    # 重新判定，而是复用窗口的 warn/fail 覆盖范围。
    if candidate_mask is None:
        mask = np.isfinite(abs_values) & (abs_values > limit)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        if len(candidate_mask) != len(values):
            raise ValueError('candidate_mask length must match signed_gap')
        mask = np.isfinite(abs_values) & candidate_mask
    defect_values = abs_values[mask]
    if defect_values.size == 0:
        return {'mask': mask, 'values': defect_values,
                'colors': np.empty((0, 3), dtype=np.float32),
                'vmin': limit, 'vmax': limit}
    vmax = float(color_max) if color_max is not None else float(np.percentile(defect_values, 98.0))
    vmax = max(vmax, limit * 1.05)
    denom = max(vmax - limit, 1e-12)
    t = np.clip((defect_values - limit) / denom, 0.0, 1.0)
    # 与渲染服务保持一致的轻量 turbo 近似，避免质量算法依赖 matplotlib。
    colors = np.column_stack((
        np.clip(1.5 * t, 0.0, 1.0),
        np.clip(1.5 - np.abs(2.0 * t - 1.0) * 1.5, 0.0, 1.0),
        np.clip(1.2 * (1.0 - t), 0.0, 1.0),
    )).astype(np.float32)
    return {'mask': mask, 'values': defect_values, 'colors': colors,
            'vmin': limit, 'vmax': vmax}


def sliding_window_flatness_2d(points, center, u_axis, v_axis, reference_plane,
                               window_size=2.0, step_size=0.05, min_points=30,
                               max_windows=None,
                               flatness_limit=0.004):
    """
    2m x 2m 滑动窗口平整度检测 —— 区间基准平面 + 局部平面偏离方案
    """
    if len(points) < min_points:
        return None

    # Step 1: 拟合区间基准平面
    interval_plane = fit_plane_svd(points)
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

    if max_windows is not None and max_windows > 0:
        total_windows = len(u_starts) * len(v_starts)
        if total_windows > max_windows:
            scale = np.sqrt(total_windows / max_windows)
            u_starts = u_starts[::max(1, int(np.ceil(scale)))]
            v_starts = v_starts[::max(1, int(np.ceil(scale)))]

    # Step 3: 滑动窗口遍历 —— 计算所有窗口
    worst_gap = 0.0
    worst_info = None

    # 统计窗口合格率
    total_valid_windows = 0
    compliant_windows = 0
    # 记录每个窗口的质量等级和覆盖的点索引
    window_results = []  # 每个元素: {u_range, v_range, quality_level, gap, point_indices}

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

            total_valid_windows += 1

            window_points = points[mask]
            window_center = np.mean(window_points, axis=0)

            # 拟合局部平面
            local_plane = fit_plane_svd(window_points)
            local_normal = local_plane[:3]
            local_normal = local_normal / (np.linalg.norm(local_normal) + 1e-12)

            # 法向方向对齐
            if np.dot(local_normal, interval_normal) < 0:
                local_normal = -local_normal

            # 核心指标：局部平面中心到区间基准平面的距离（原判定逻辑保持不变）
            gap = float(np.abs(np.dot(interval_normal, window_center) + interval_d))

            # 【新增】判定质量等级
            if gap <= flatness_limit:
                quality_level = Config.QUALITY_PASS
                compliant_windows += 1
            elif gap <= flatness_limit * 2.0:
                quality_level = Config.QUALITY_WARN
            else:
                quality_level = Config.QUALITY_FAIL

            # 记录窗口结果。只有异常窗口才计算局部逐点偏差，避免对大量
            # 合格窗口重复执行点到平面的向量计算。
            window_data = {
                'u_range': [float(u_low), float(u_high)],
                'v_range': [float(v_low), float(v_high)],
                'quality_level': quality_level,
                'gap': gap,
                'point_indices': np.where(mask)[0].tolist(),  # 该窗口覆盖的局部点索引
                'window_center': [float(x) for x in window_center],
                'point_count': count,
            }
            if quality_level != Config.QUALITY_PASS:
                window_data['point_abs_gaps'] = np.abs(
                    window_points @ interval_normal + interval_d
                ).astype(np.float32).tolist()
            window_results.append(window_data)

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

    # 计算合格率
    compliance_rate = float(compliant_windows / total_valid_windows) if total_valid_windows > 0 else 1.0

    return worst_gap, worst_info, compliance_rate, total_valid_windows, compliant_windows, window_results


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
        'type': facade_type
    }


def compute_facade_quality(facade_info, pcd, grid_size=20.0,
                           flatness_limit=0.004,
                           verticality_limit_mm=4.0,
                           ruler_size=2.0,
                           ruler_step=0.05,
                           max_windows=None):
    """
    在下采样立面点云上计算质量。
    垂直度仅保留夹角；平整度采用每 20m 区间内 2m x 2m 模拟靠尺滑动窗口的严格最大间隙。
    生成每个点的检测颜色映射，支持定位不合格/警告区域
    """
    # grid_size is the legacy name; it is an aggregation interval, not the
    # 2m inspection window.  The latter remains ruler_size/ruler_step.
    interval_size_m = float(grid_size)
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
    v_bins = max(1, int(np.ceil((v_max - v_min) / interval_size_m)))

    grids = []
    # 【新增】点级质量标签，初始为-1（未评估）
    point_quality_labels = np.full(len(points), -1, dtype=int)
    # 仅由窗口质量决定热力图覆盖范围；数值仍是窗口点到区间基准面的偏差。
    window_heatmap_mask = np.zeros(len(points), dtype=bool)
    window_heatmap_values = np.full(len(points), np.nan, dtype=np.float32)

    for j in range(v_bins):
        v_low = v_min + j * interval_size_m
        v_high = v_min + (j + 1) * interval_size_m
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
            min_points=30,
            max_windows=max_windows,
            flatness_limit=flatness_limit
        )

        if window_result is None:
            signed = signed_plane_distance(grid_points, facade_plane)
            gap = float(np.max(signed) - np.min(signed))
            window_info = None
            compliance_rate = 1.0
            total_windows = 0
            compliant_windows = 0
            window_results = []
        else:
            gap, window_info, compliance_rate, total_windows, compliant_windows, window_results = window_result

        # 【新增】将窗口质量等级映射到点
        # 获取该grid在全局点云中的索引
        grid_global_indices = np.where(mask)[0]

        # 对每个窗口，将其覆盖的点标记为对应质量等级
        # 如果点被多个窗口覆盖，取最差等级（FAIL > WARN > PASS）
        for wr in window_results:
            local_indices = wr['point_indices']  # 局部索引（在grid_points中的索引）
            quality_level = wr['quality_level']
            point_abs_gaps = np.asarray(wr.get('point_abs_gaps', []), dtype=np.float32)
            for local_idx in local_indices:
                global_idx = grid_global_indices[local_idx]
                if point_quality_labels[global_idx] == -1:
                    point_quality_labels[global_idx] = quality_level
                else:
                    # 取更差的等级
                    point_quality_labels[global_idx] = max(point_quality_labels[global_idx], quality_level)
                if quality_level != Config.QUALITY_PASS and local_idx < len(point_abs_gaps):
                    window_heatmap_mask[global_idx] = True
                    value = float(point_abs_gaps[local_idx])
                    if not np.isfinite(window_heatmap_values[global_idx]):
                        window_heatmap_values[global_idx] = value
                    else:
                        window_heatmap_values[global_idx] = max(window_heatmap_values[global_idx], value)

        # 合格率 >= 95% 视为整体合格
        compliance = compliance_rate >= 0.95

        grid_data = {
            'grid_id': int(len(grids)),
            'center': [float(x) for x in grid_center],
            'verticality': float(verticality_deg),
            'horizontality': float(horizontality),
            'gap': float(gap),
            'point_count': grid_point_count,
            'compliance': bool(compliance),
            'compliance_rate': float(compliance_rate),
            'total_windows': int(total_windows),
            'compliant_windows': int(compliant_windows),
            'v_range': [float(v_low), float(v_high)],
            'u_range': [float(u_min), float(u_max)],
            # 该grid内的窗口质量分布
            'window_quality_distribution': compute_window_quality_distribution(window_results),
        }
        if window_info:
            grid_data['worst_window'] = window_info

        grids.append(grid_data)

    overall_gap = max((g['gap'] for g in grids), default=0.0)

    # 整体合格率：各区间合格率的加权平均（按点数加权）
    overall_compliance_rate = 1.0
    if grids:
        total_points = sum(g['point_count'] for g in grids)
        if total_points > 0:
            overall_compliance_rate = sum(
                g['compliance_rate'] * g['point_count'] for g in grids
            ) / total_points

    # 热力图复用窗口判定结果，不再使用全局参考面重新筛选点。
    signed_gap = signed_plane_distance(points, facade_plane).astype(np.float32)
    heatmap = generate_sparse_defect_heatmap(
        window_heatmap_values, flatness_limit,
        candidate_mask=window_heatmap_mask,
    )
    # 保留旧字段，兼容现有调用方；新调用方应优先使用稀疏字段。
    quality_colors = generate_quality_colors(point_quality_labels, len(points))

    # 【新增】统计各质量等级的点数
    quality_stats = {
        'pass_count': int(np.sum(point_quality_labels == Config.QUALITY_PASS)),
        'warn_count': int(np.sum(point_quality_labels == Config.QUALITY_WARN)),
        'fail_count': int(np.sum(point_quality_labels == Config.QUALITY_FAIL)),
        'unassessed_count': int(np.sum(point_quality_labels == -1)),
    }

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
            'compliance_rate': float(overall_compliance_rate),
        },
        'grids': grids,
        'grid_size': interval_size_m,
        'interval_size_m': interval_size_m,
        'window_size_m': float(ruler_size),
        'step_size_m': float(ruler_step),
        'ruler_size': ruler_size,
        'ruler_step': ruler_step,
        'flatness_limit': flatness_limit,
        'verticality_limit_mm': verticality_limit_mm,
        'quality_source': 'downsampled',
        'flatness_metric': 'local plane center deviation from interval baseline plane in 2m x 2m sliding windows (unlimited windows)',
        # 点级质量信息
        'point_quality_labels': point_quality_labels.tolist(),
        # Do not paint compliant/unassessed points; renderers should use the
        # sparse defect fields below and preserve the original RGB.
        'quality_colors': [],
        'signed_gap': signed_gap.tolist(),
        'abs_gap': np.abs(signed_gap).tolist(),
        'defect_local_indices': np.flatnonzero(heatmap['mask']).tolist(),
        'defect_values': heatmap['values'].tolist(),
        'defect_colors': heatmap['colors'].tolist(),
        'heatmap_vmin': heatmap['vmin'],
        'heatmap_vmax': heatmap['vmax'],
        'heatmap_cmap': 'turbo',
        'quality_stats': quality_stats,
    }


def _empty_quality_result(grid_size, flatness_limit, verticality_limit_mm):
    return {
        'overall': {
            'type': 'unknown',
            'type_label': '平面',
            'verticality': 0.0,
            'horizontality': 0.0,
            'gap': 0.0,
            'point_count': 0,
            'compliance_rate': 1.0,
        },
        'grids': [],
        'grid_size': grid_size,
        'interval_size_m': grid_size,
        'window_size_m': 2.0,
        'step_size_m': 0.05,
        'ruler_size': 2.0,
        'flatness_limit': flatness_limit,
        'verticality_limit_mm': verticality_limit_mm,
        'quality_source': 'downsampled',
        'point_quality_labels': [],
        'quality_colors': [],
        'quality_stats': {
            'pass_count': 0,
            'warn_count': 0,
            'fail_count': 0,
            'unassessed_count': 0,
        },
    }