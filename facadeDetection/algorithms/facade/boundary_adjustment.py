from __future__ import annotations

import numpy as np


def incremental_proxy_ids(proxy_points, geometry, old_proxy_ids, new_bounds,
                          surface_band_m=0.01):
    """只补入新扩区域的点，并移除被新边界排除的旧立面点。"""
    points = np.asarray(proxy_points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(new_bounds, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bounds)) or bounds[2] <= bounds[0] \
            or bounds[3] <= bounds[1]:
        raise ValueError('立面边界无效')
    origin = np.asarray(geometry['origin'], dtype=np.float64).reshape(3)
    u_axis = np.asarray(geometry['u_axis'], dtype=np.float64).reshape(3)
    v_axis = np.asarray(geometry['v_axis'], dtype=np.float64).reshape(3)
    plane = np.asarray(geometry['plane_model'], dtype=np.float64).reshape(4)
    normal = plane[:3] / np.linalg.norm(plane[:3])

    rel = points - origin
    u = rel @ u_axis
    v = rel @ v_axis
    in_new = ((u >= bounds[0]) & (u <= bounds[2]) &
              (v >= bounds[1]) & (v <= bounds[3]))
    old_bounds = np.asarray(
        geometry['uv_bounds'], dtype=np.float64).reshape(4)
    in_old = ((u >= old_bounds[0]) & (u <= old_bounds[2]) &
              (v >= old_bounds[1]) & (v <= old_bounds[3]))
    on_surface = np.abs(points @ normal + plane[3]) <= max(
        float(surface_band_m), 1e-6)

    old_ids = np.asarray(list(old_proxy_ids), dtype=np.int64).reshape(-1)
    old_ids = old_ids[(old_ids >= 0) & (old_ids < len(points))]
    retained = old_ids[in_new[old_ids]]
    added = np.flatnonzero(in_new & ~in_old & on_surface)
    result = np.union1d(retained, added)
    if len(result) < 3:
        raise ValueError('调整后的立面有效点不足')
    return result


def resize_uv_bounds(bounds, edge, u, v, min_size=0.1):
    """移动一个 UV 边，保持最小面尺寸且允许超出原边界。"""
    u0, v0, u1, v1 = np.asarray(bounds, dtype=np.float64).reshape(4)
    minimum = max(float(min_size), 1e-6)
    if edge == 'left':
        u0 = min(float(u), u1 - minimum)
    elif edge == 'right':
        u1 = max(float(u), u0 + minimum)
    elif edge == 'bottom':
        v0 = min(float(v), v1 - minimum)
    elif edge == 'top':
        v1 = max(float(v), v0 + minimum)
    else:
        raise ValueError(f'未知立面边: {edge}')
    return np.array([u0, v0, u1, v1], dtype=np.float64)
