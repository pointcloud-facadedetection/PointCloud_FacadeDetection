"""统一立面几何描述。

该模块只负责把质量检测使用的 UV 矩形映射到世界坐标，不参与任何
检测、拟合或质量判定。所有调用方因此得到完全一致的几何契约。

规范化约束（v2.1）：
    1. 角点按 UV 顺序 [BL, BR, TR, TL] 一一对应 uv_bounds (u0,v0)…
       (u1,v1)，不做任何为"凑手性"而设的翻转；
    2. 唯一规范化：v 轴投影到平面上需与世界 +Z 同向。若反向，
       同时取反 v_axis 和 v0/v1（保持物理位置不变）；
    3. 手性 u×v·normal 仅作为字段返回，供渲染方判断面朝向 /
       剔除策略，不影响顶点世界坐标。
"""
from __future__ import annotations

import numpy as np


def extract_facade_geometry(plane_model, origin, u_axis, v_axis, uv_bounds):
    """返回立面四角点和四条边中点（均为世界 XYZ 坐标）。

    ``uv_bounds`` 顺序固定为 ``(u_min, v_min, u_max, v_max)``。角点按
    左下→右下→右上→左上的 UV 顺序返回，边中点按相邻角点顺序返回。

    保证：
        - corners_3d[0]=左下, [1]=右下, [2]=右上, [3]=左上
        - edge_midpoints_3d[0]=底, [1]=右, [2]=顶, [3]=左
        - u_axis / v_axis 原样规范化，但 v_axis 可能与输入反号
        - 所有输出均为世界坐标；返回结构向后兼容，新增 ``chirality``
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
    normal = plane[:3] / np.linalg.norm(plane[:3])

    u0, v0, u1, v1 = bounds
    if u1 <= u0 or v1 <= v0:
        raise ValueError('facade geometry bounds are degenerate')

    # ──规范化：重力对齐 ───────────────────────────────────
    world_up = np.array([0.0, 0.0, 1.0])
    up_in_plane = world_up - float(world_up @ normal) * normal
    up_norm = float(np.linalg.norm(up_in_plane))
    if up_norm > 1e-6:
        up_in_plane = up_in_plane / up_norm
    if float(np.dot(np.cross(u, v), normal)) < 0.0:
        u = -u
        u0, u1 = -u1, -u0

    # ── 固定 UV 角点顺序：BL → BR → TR → TL ─────────────────────
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    corners = origin + uv[:, 0, None] * u + uv[:, 1, None] * v
    mids = (corners + np.roll(corners, -1, axis=0)) * 0.5

    # ── 手性仅作信息输出，不改变顶点 ─────────────────────────────
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    chirality = float(np.dot(np.cross(e0, e1), normal))

    return {
        'coordinate_system': 'world_xyz',
        'plane_model': plane.tolist(),
        'origin': origin.tolist(),
        'u_axis': u.tolist(),
        'v_axis': v.tolist(),
        'uv_bounds': [float(u0), float(v0), float(u1), float(v1)],
        'corners_3d': corners.tolist(),
        'edge_midpoints_3d': mids.tolist(),
        'normal': normal.tolist(),
    }