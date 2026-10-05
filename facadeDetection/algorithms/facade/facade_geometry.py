"""统一立面几何描述。

该模块只负责把质量检测使用的 UV 矩形映射到世界坐标，不参与任何
检测、拟合或质量判定。所有调用方因此得到完全一致的几何契约。
"""
from __future__ import annotations

import numpy as np


def extract_facade_geometry(plane_model, origin, u_axis, v_axis, uv_bounds):
    """返回立面四角点和四条边中点（均为世界 XYZ 坐标）。

    ``uv_bounds`` 顺序固定为 ``(u_min, v_min, u_max, v_max)``；角点按
    左下、右下、右上、左上的 UV 顺序返回，边中点按相邻角点顺序返回。

    该函数是唯一对外暴露的几何提取入口。靠尺法与模拟墙面（全局平面）
    两条检测分支必须都通过它生成几何，以保证：

      1. 函数签名、返回结构完全一致；
      2. 世界坐标系、平面法向、UV 边界完全一致；
      3. ``corners_3d[i]`` 与 ``edge_midpoints_3d[i]`` 的顺序可预测。
    """
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    u = np.asarray(u_axis, dtype=np.float64).reshape(3)
    v = np.asarray(v_axis, dtype=np.float64).reshape(3)
    bounds = np.asarray(uv_bounds, dtype=np.float64).reshape(4)
    if not all(np.all(np.isfinite(x)) for x in (plane, origin, u, v, bounds)):
        raise ValueError('facade geometry inputs must be finite')
    if np.linalg.norm(plane[:3]) < 1e-12:
        raise ValueError('facade geometry plane normal is degenerate')
    un, vn = np.linalg.norm(u), np.linalg.norm(v)
    if un < 1e-12 or vn < 1e-12:
        raise ValueError('facade geometry axes are degenerate')
    u, v = u / un, v / vn
    u0, v0, u1, v1 = bounds
    if u1 <= u0 or v1 <= v0:
        raise ValueError('facade geometry bounds are degenerate')

    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    corners = origin + uv[:, 0, None] * u + uv[:, 1, None] * v
    mids = (corners + np.roll(corners, -1, axis=0)) * 0.5
    normal = plane[:3] / np.linalg.norm(plane[:3])
    return {
        'coordinate_system': 'world_xyz',
        'plane_model': plane.tolist(),
        'origin': origin.tolist(),
        'u_axis': u.tolist(),
        'v_axis': v.tolist(),
        'uv_bounds': bounds.tolist(),
        'corners_3d': corners.tolist(),
        'edge_midpoints_3d': mids.tolist(),
        'normal': normal.tolist(),
    }