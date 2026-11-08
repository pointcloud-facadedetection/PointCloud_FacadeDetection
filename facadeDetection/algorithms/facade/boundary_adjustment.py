from __future__ import annotations

import numpy as np


def incremental_proxy_ids(proxy_points, geometry, old_proxy_ids, new_bounds,
                          surface_band_m=0.01, chunk_size=200_000):
    """只补入新扩区域的点，并移除被新边界排除的旧立面点。"""
    points = np.asarray(proxy_points).reshape(-1, 3)
    bounds = np.asarray(new_bounds, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bounds)) or bounds[2] <= bounds[0] \
            or bounds[3] <= bounds[1]:
        raise ValueError('立面边界无效')
    dtype = np.float32 if points.dtype == np.float32 else np.float64
    origin = np.asarray(geometry['origin'], dtype=dtype).reshape(3)
    u_axis = np.asarray(geometry['u_axis'], dtype=dtype).reshape(3)
    v_axis = np.asarray(geometry['v_axis'], dtype=dtype).reshape(3)
    plane = np.asarray(geometry['plane_model'], dtype=dtype).reshape(4)
    normal = plane[:3] / np.linalg.norm(plane[:3])
    old_bounds = np.asarray(geometry['uv_bounds'], dtype=dtype).reshape(4)
    bounds = bounds.astype(dtype)
    band = max(float(surface_band_m), 1e-6)

    old_ids = np.asarray(list(old_proxy_ids), dtype=np.int64).reshape(-1)
    old_ids = old_ids[(old_ids >= 0) & (old_ids < len(points))]

    def in_bounds(block, limits):
        rel = block - origin
        u = rel @ u_axis
        v = rel @ v_axis
        return ((u >= limits[0]) & (u <= limits[2])
                & (v >= limits[1]) & (v <= limits[3]))

    retained = old_ids[in_bounds(points[old_ids], bounds)] if len(old_ids) \
        else old_ids
    added_chunks = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        block = points[start:stop]
        in_new = in_bounds(block, bounds)
        if not np.any(in_new):
            continue
        in_old = in_bounds(block, old_bounds)
        candidates = in_new & ~in_old
        if not np.any(candidates):
            continue
        on_surface = np.abs(block @ normal + plane[3]) <= band
        added_chunks.append(np.flatnonzero(candidates & on_surface) + start)

    added = (np.concatenate(added_chunks) if added_chunks
             else np.empty(0, dtype=np.int64))
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
