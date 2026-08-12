# -*- coding: utf-8 -*-
"""
立面检测后处理：共面合并 / 完形回收 / 洞口提取 / 墙体分类。

从 detect_facades 主流程独立出来，通过注入的 rebuild/facade_ok 回调与
主流程解耦（避免循环依赖）。所有几何判定沿用检测阶段的距离自适应量
（tol_full/cos_full/spacing_all）。
"""
from collections import deque

import numpy as np

from algorithms.geometry import plane_axes


def _uv_cell_arr(p, plane, ftype, grid):
    # 共享轴、不做中心化的UV格坐标：不同点集的格子可以直接比较
    nrm = np.asarray(plane[:3], dtype=float)
    u_ax, v_ax = plane_axes(nrm, ftype)
    return (np.floor(p @ u_ax / grid).astype(np.int64),
            np.floor(p @ v_ax / grid).astype(np.int64))


def _dilate(cells, rad):
    out = set()
    for (a, b) in cells:
        for da in range(-rad, rad + 1):
            for db in range(-rad, rad + 1):
                out.add((a + da, b + db))
    return out


def merge_facades(fa, *, points, spacing_all, tol_full, rebuild, facade_ok,
                  uv_grid_size, merge_gap_m, merge_angle_deg, merge_d_thresh):
    """共面（角度+互距在容差内）且UV相邻的检出面合并，同一面墙不再分两块。

    邻接判定用"窗洞尺度"粗网格：带窗墙面被窗列切成的墙垛条带间隔1.5~3m，
    细网格膨胀够不过窗洞；粗网格半径2格即可跨过窗洞把同一面墙的条带连回来
    （共面判定本身很严，误并风险低）。
    """
    merged_any = True
    while merged_any:
        merged_any = False
        fa.sort(key=lambda f: -f['point_count'])
        for i in range(len(fa)):
            if merged_any:
                break
            for j in range(i + 1, len(fa)):
                fi, fj = fa[i], fa[j]
                if fi['type'] != fj['type']:
                    continue
                pmi = np.asarray(fi['plane_model'], dtype=float)
                pmj = np.asarray(fj['plane_model'], dtype=float)
                if abs(float(pmi[:3] @ pmj[:3])) < np.cos(np.deg2rad(merge_angle_deg)):
                    continue
                ci = np.asarray(fi['center'])
                cj = np.asarray(fj['center'])
                off = max(abs(float(cj @ pmi[:3] + pmi[3])),
                          abs(float(ci @ pmj[:3] + pmj[3])))
                idx_i = np.asarray(fi['inlier_indices'], dtype=int)
                idx_j = np.asarray(fj['inlier_indices'], dtype=int)
                tol_pair = float(np.median(np.concatenate(
                    [tol_full[idx_i], tol_full[idx_j]])))
                if off > max(merge_d_thresh, tol_pair * 1.5):
                    continue
                grid_adj = max(merge_gap_m / 2.0, 0.5)
                cu, cv = _uv_cell_arr(points[idx_i], pmi, fi['type'], grid_adj)
                dil = _dilate(set(zip(cu.tolist(), cv.tolist())), 2)
                cu2, cv2 = _uv_cell_arr(points[idx_j], pmi, fi['type'], grid_adj)
                if not any((a, b) in dil for a, b in zip(cu2.tolist(), cv2.tolist())):
                    continue
                info = rebuild(fi['id'], np.concatenate([idx_i, idx_j]))
                if not facade_ok(info):
                    continue
                fa[i] = info
                del fa[j]
                merged_any = True
                break
    return fa


def complete_facades(fa, *, points, normals, spacing_all, tol_full, cos_full,
                     rebuild, facade_ok, uv_grid_size, max_bridge_cells):
    """完形回收：未归入但落在某检出面容差带内、法向一致且UV邻接的点吸收进该面"""
    total_points = len(points)
    assigned = np.zeros(total_points, dtype=bool)
    for f in fa:
        assigned[np.asarray(f['inlier_indices'], dtype=int)] = True
    for k in sorted(range(len(fa)), key=lambda t: -fa[t]['point_count']):
        f = fa[k]
        un = np.where(~assigned)[0]
        if not len(un):
            break
        pm = np.asarray(f['plane_model'], dtype=float)
        dist = np.abs(points[un] @ pm[:3] + pm[3])
        ok = (dist <= tol_full[un] * 1.5) & \
             (np.abs(normals[un] @ pm[:3]) >= cos_full[un])
        cand = un[ok]
        if not len(cand):
            continue
        idx_f = np.asarray(f['inlier_indices'], dtype=int)
        grid = max(uv_grid_size, 1.5 * float(np.median(spacing_all[idx_f])))
        cu, cv = _uv_cell_arr(points[idx_f], pm, f['type'], grid)
        dil = _dilate(set(zip(cu.tolist(), cv.tolist())), max_bridge_cells)
        cu2, cv2 = _uv_cell_arr(points[cand], pm, f['type'], grid)
        keep = np.fromiter(((a, b) in dil for a, b in zip(
            cu2.tolist(), cv2.tolist())), dtype=bool, count=len(cand))
        add = cand[keep]
        # 回收量太小就不值得重拟合
        if len(add) < max(20, len(idx_f) // 50):
            continue
        info = rebuild(f['id'], np.concatenate([idx_f, add]))
        if facade_ok(info):
            fa[k] = info
            assigned[add] = True
    return fa


def extract_openings(fa, *, points, spacing_all, uv_grid_size, min_opening_area):
    """
    竖直立面洞口提取：UV占用图上从边界洪泛标记外部，剩余的封闭空洞
    即窗/门洞口。洞口 u/v 为共享轴绝对UV坐标（含偏移），配合
    plane_axes(normal,type) 与平面方程可直接重建3D位置。
    同时输出净面积 area_net（实墙）与毛面积 area_gross（含洞口）。
    """
    for f in fa:
        if f['type'] != 'vertical_facade':
            continue
        idx = np.asarray(f['inlier_indices'], dtype=int)
        grid = max(uv_grid_size, 1.5 * float(np.median(spacing_all[idx])))
        pm = np.asarray(f['plane_model'], dtype=float)
        cu, cv = _uv_cell_arr(points[idx], pm, f['type'], grid)
        u0, v0 = int(cu.min()), int(cv.min())
        W = int(cu.max()) - u0 + 1
        H = int(cv.max()) - v0 + 1
        if W * H > 4_000_000 or W * H <= 4:
            continue
        occ = np.zeros((W, H), dtype=bool)
        occ[cu - u0, cv - v0] = True

        outside = np.zeros((W, H), dtype=bool)
        dq = deque()
        for x in range(W):
            for y in (0, H - 1):
                if not occ[x, y] and not outside[x, y]:
                    outside[x, y] = True
                    dq.append((x, y))
        for y in range(H):
            for x in (0, W - 1):
                if not occ[x, y] and not outside[x, y]:
                    outside[x, y] = True
                    dq.append((x, y))
        while dq:
            x, y = dq.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = x + dx, y + dy
                if 0 <= a < W and 0 <= b < H and not occ[a, b] and not outside[a, b]:
                    outside[a, b] = True
                    dq.append((a, b))
        hole = ~occ & ~outside

        openings = []
        seen = np.zeros((W, H), dtype=bool)
        hx, hy = np.where(hole)
        for x, y in zip(hx.tolist(), hy.tolist()):
            if seen[x, y]:
                continue
            comp = [(x, y)]
            seen[x, y] = True
            qi = 0
            while qi < len(comp):
                a, b = comp[qi]
                qi += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p_, q_ = a + dx, b + dy
                    if 0 <= p_ < W and 0 <= q_ < H and hole[p_, q_] and not seen[p_, q_]:
                        seen[p_, q_] = True
                        comp.append((p_, q_))
            area_h = len(comp) * grid * grid
            # 过滤采样空隙；大于半面墙的"洞"多半是凹形边界误判，同样跳过
            if area_h < min_opening_area or area_h > 0.5 * max(f['area'], 1.0):
                continue
            xs = [c[0] for c in comp]
            ys = [c[1] for c in comp]
            openings.append({
                'u_min': round((u0 + min(xs)) * grid, 2),
                'u_max': round((u0 + max(xs) + 1) * grid, 2),
                'v_min': round((v0 + min(ys)) * grid, 2),
                'v_max': round((v0 + max(ys) + 1) * grid, 2),
                'area': round(area_h, 2),
            })
        openings.sort(key=lambda o: -o['area'])
        f['openings'] = openings[:64]
        f['n_openings'] = len(openings)
        f['area_net'] = f['area']
        f['area_gross'] = round(f['area'] + sum(o['area'] for o in openings), 1)
    return fa


def classify_and_group(fa):
    """
    墙体分类与面组归并：
    - fill_ratio 净面积/包围盒面积：实墙≥0.55；opening_share 洞口占比
    - wall_kind: solid实墙 / fenestrated带窗 / fragment碎片
    - face_group: 法向平行+u区间重叠≥30%+深度差≤0.45m(窗框凹退尺度)的竖直面
      归同一建筑面——带窗面的多层碎片归属同组，不再各自为政
    """
    verts = [i for i, f in enumerate(fa) if f['type'] == 'vertical_facade']
    for i in verts:
        f = fa[i]
        bb = f.get('bbox_2d') or {}
        span_u = max(float(bb.get('u_max', 0)) - float(bb.get('u_min', 0)), 1e-6)
        span_v = max(float(bb.get('v_max', 0)) - float(bb.get('v_min', 0)), 1e-6)
        fill = float(f['area'] / (span_u * span_v))
        gross = float(f.get('area_gross', f['area']) or f['area'])
        open_share = float(1.0 - f['area'] / max(gross, 1e-6))
        f['fill_ratio'] = round(fill, 3)
        f['opening_share'] = round(open_share, 3)
        if fill < 0.25:
            f['wall_kind'] = 'fragment'
        elif open_share >= 0.12 or fill < 0.55:
            f['wall_kind'] = 'fenestrated'
        else:
            f['wall_kind'] = 'solid'

    parent = list(range(len(fa)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(verts)):
        for b in range(a + 1, len(verts)):
            i, j = verts[a], verts[b]
            fi, fj = fa[i], fa[j]
            pmi = np.asarray(fi['plane_model'], dtype=float)
            pmj = np.asarray(fj['plane_model'], dtype=float)
            if abs(float(pmi[:3] @ pmj[:3])) < np.cos(np.deg2rad(8.0)):
                continue
            ci = np.asarray(fi['center'])
            cj = np.asarray(fj['center'])
            off = max(abs(float(cj @ pmi[:3] + pmi[3])),
                      abs(float(ci @ pmj[:3] + pmj[3])))
            if off > 0.45:
                continue
            bi, bj = fi.get('bbox_2d') or {}, fj.get('bbox_2d') or {}
            if not bi or not bj:
                continue
            u_ax = np.asarray(bi['u_axis'], dtype=float)
            du = float((cj - ci) @ u_ax)
            i0, i1 = float(bi['u_min']), float(bi['u_max'])
            j0, j1 = float(bj['u_min']) + du, float(bj['u_max']) + du
            overlap = min(i1, j1) - max(i0, j0)
            if overlap < 0.3 * min(i1 - i0, j1 - j0):
                continue
            ra, rb = _find(i), _find(j)
            if ra != rb:
                parent[rb] = ra

    gid_map = {}
    for i in verts:
        root = _find(i)
        if root not in gid_map:
            gid_map[root] = len(gid_map)
        fa[i]['face_group'] = gid_map[root]
    return fa


def postprocess_facades(facades, *, points, normals, spacing_all, tol_full, cos_full,
                        rebuild, facade_ok, uv_grid_size,
                        merge_pass=True, completion_pass=True,
                        merge_gap_m=3.0, min_opening_area=0.5,
                        max_bridge_cells=6,
                        merge_angle_deg=8.0, merge_d_thresh=0.08):
    """检测后处理入口：合并 -> 完形 -> 洞口 -> 分类"""
    if not facades:
        return facades
    if merge_pass:
        facades = merge_facades(
            facades, points=points, spacing_all=spacing_all, tol_full=tol_full,
            rebuild=rebuild, facade_ok=facade_ok, uv_grid_size=uv_grid_size,
            merge_gap_m=merge_gap_m, merge_angle_deg=merge_angle_deg,
            merge_d_thresh=merge_d_thresh)
    if completion_pass:
        facades = complete_facades(
            facades, points=points, normals=normals, spacing_all=spacing_all,
            tol_full=tol_full, cos_full=cos_full, rebuild=rebuild,
            facade_ok=facade_ok, uv_grid_size=uv_grid_size,
            max_bridge_cells=max_bridge_cells)
    facades = extract_openings(
        facades, points=points, spacing_all=spacing_all,
        uv_grid_size=uv_grid_size, min_opening_area=min_opening_area)
    facades = classify_and_group(facades)
    return facades
