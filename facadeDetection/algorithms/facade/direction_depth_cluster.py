"""方向桶 × 深度层种子生成 v3：全主朝向建桶 + 排序间隙分层 + ρ-Hough 校验
+ 残余方向（bin_idx=-2）补充通道。

1. 【硬性约束】深度分层由「排序 + 间隙切割」：
   ρ 升序排序后，相邻间隙 > ``FACADE_DEPTH_LAYER_HARD_M``(5cm) 即切层。
   性质保证：
     - 任意两个不同层的 ρ 代表值之差 > 5cm（切点即 >5cm 的间隙）；
     - 同一层内部 ρ 极差可以略大于 5cm（缓坡），但相邻点链上无 >5cm 跳变，
       即同层在深度方向连续——满足「相邻空间域候选深度差 >5cm 必分割」。
2. 【双归属落地】``FACADE_LAYER_BOUNDARY_EPS_M`` 真正生效：切点两侧 ε 带内
   的点同时归属相邻两层，避免硬切把同一面劈成两半（层合并阶段再去重）。
3. 【Hough 定位修正】主方向域内 ρ-Hough 仅做层中心校验/精化（原"断裂层
   补种"经两次运行证明几何上不可触发：真实层峰必落在某层中位数 2.5cm 内，
   断裂层残余峰又落在相邻层内部——已删除）。Hough 改派到补充通道：残余
   方向域的点云碎、稀，ρ 投票按支持度直接发现层中心，是 Hough 的结构性
   优势场景。
4. 【全主朝向建桶】`_quantize_directions` 使用 `_estimate_principal_directions`
   返回的【全部】主朝向（各 +{0,90,180,270}°，5° 内去重），不再只用第 1 个。
   非正交第二朝向的墙面此前整面落入 bin_idx=-2 而永不种子。
5. 【补充通道】对 bin_idx=-2 的弧外残余竖直点做二次方向发现 → ρ-Hough
   层中心 → ±2.5cm 成窗 → 3D 连通域成种，与主链完全相同的门控格式。
6. 【密度门控】接入 density_estimator：单元/连通域级 'skip' 直接跳过；
   'sparse' 放宽 min_pts。
"""
from __future__ import annotations

import time
from typing import List, Dict, Any
import numpy as np

from algorithms.geometry import (
    connected_components_3d_grid,
    fit_plane_weighted,
)
from algorithms.facade.hough_facade import rho_hough_peaks
from config.settings import Config
from utils.logging_utils import trace

try:
    from algorithms.facade.density_estimator import (
        build_density_field, adaptive_min_count)
    _HAS_DENSITY = True
except Exception:  # 密度模块缺失时优雅降级为旧行为
    _HAS_DENSITY = False


# ---------------------------------------------------------------------------
# A. 方向桶化
# ---------------------------------------------------------------------------

def _estimate_principal_directions(normals, vertical_nz=0.30,
                                   min_support_ratio=0.01,
                                   min_peak_spacing_deg=45.0,
                                   max_peaks=4):
    """从 ROI 法向估计建筑主朝向 azimuth（贪心 Top-K 峰），返回弧度。

    多建筑场景下（如并排双塔），Top-2 只能覆盖一个建筑的两个主立面。
    贪心选取最多 max_peaks 个不重叠峰（最小间距 min_peak_spacing_deg），
    确保所有建筑主朝向都被方向桶覆盖。
    """
    valid = np.abs(normals[:, 2]) <= float(vertical_nz)
    if not np.any(valid):
        return []
    az = np.arctan2(normals[valid, 1], normals[valid, 0])
    az = np.mod(az, 2.0 * np.pi)
    bin_rad = np.deg2rad(5.0)
    nbins = max(36, int(np.ceil(2.0 * np.pi / bin_rad)))
    counts, edges = np.histogram(az, bins=nbins, range=(0.0, 2.0 * np.pi))
    smooth = (np.convolve(counts.astype(float), np.array([0.25, 0.5, 0.25]),
                          mode="same") if len(counts) >= 3
              else counts.astype(float))
    min_support = max(20, int(len(az) * float(min_support_ratio)))
    peaks = []
    for i in range(len(smooth)):
        prev = smooth[i - 1] if i > 0 else 0.0
        nxt = smooth[(i + 1) % len(smooth)]
        if smooth[i] > prev and smooth[i] > nxt and smooth[i] >= min_support:
            peaks.append((float((edges[i] + edges[i + 1]) * 0.5),
                          float(smooth[i])))
    peaks.sort(key=lambda x: -x[1])

    # 贪心选取不重叠峰
    selected = []
    min_rad = np.deg2rad(float(min_peak_spacing_deg))
    for az_peak, support in peaks:
        isolated = True
        for s in selected:
            diff = float(abs(np.mod(az_peak - s + np.pi, 2.0 * np.pi) - np.pi))
            if diff < min_rad:
                isolated = False
                break
        if isolated:
            selected.append(az_peak)
        if len(selected) >= int(max_peaks):
            break
    return selected


def _circular_distance(a, b):
    return np.abs(np.mod(a - b + np.pi, 2.0 * np.pi) - np.pi)


def _quantize_directions(normals, principal_azimuths, arc_half_deg=28.0,
                         vertical_nz=0.30, fallback_n_buckets=8):
    """方向桶化 v2：用【全部】主朝向建桶。

    每个主朝向展开 {0,90,180,270}° 四个桶心，5° 以内去重；竖直点按最近
    桶心分配，弧外点 bin_idx=-2（交由补充通道二次发现）。修复旧版只用
    principal_azimuths[0] 导致非正交第二朝向整面墙永不种子的问题。
    """
    n = len(normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    n_unit = normals / np.where(norms > 1e-8, norms, 1.0)
    nz = n_unit[:, 2]
    is_vertical = np.abs(nz) <= float(vertical_nz)
    is_horizontal = ~is_vertical
    az = np.mod(np.arctan2(n_unit[:, 1], n_unit[:, 0]), 2.0 * np.pi)

    arc_half = np.deg2rad(float(arc_half_deg))
    buckets_meta: List[Dict[str, Any]] = []
    bucket_normals_list = []

    if len(principal_azimuths) >= 1:
        used: List[float] = []
        for az0 in principal_azimuths:
            for off in (0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2):
                c = float(np.mod(az0 + off, 2.0 * np.pi))
                if any(_circular_distance(np.array([c]), np.array([uc]))[0]
                       < np.deg2rad(5.0) for uc in used):
                    continue
                used.append(c)
                bn = np.array([np.cos(c), np.sin(c), 0.0])
                buckets_meta.append({"index": len(buckets_meta), "az_center": c,
                                     "half_width": arc_half, "normal": bn,
                                     "is_principal": True})
                bucket_normals_list.append(bn)
    else:
        for i in range(int(fallback_n_buckets)):
            c = 2.0 * np.pi * i / fallback_n_buckets
            bn = np.array([np.cos(c), np.sin(c), 0.0])
            buckets_meta.append({"index": i, "az_center": c,
                                 "half_width": arc_half, "normal": bn,
                                 "is_principal": False})
            bucket_normals_list.append(bn)

    buckets_meta.append({"index": -1, "az_center": None, "half_width": None,
                         "normal": np.array([0.0, 0.0, 1.0]),
                         "is_principal": False, "is_horizontal": True})
    bucket_normals_list.append(np.array([0.0, 0.0, 1.0]))
    bucket_normals = np.stack(bucket_normals_list, axis=0)

    bin_idx = np.full(n, -2, dtype=np.int32)
    bin_idx[is_horizontal] = -1
    vert_ids = np.flatnonzero(is_vertical)
    if len(vert_ids):
        vert_az = az[vert_ids]
        best_bucket = np.full(len(vert_ids), -2, dtype=np.int32)
        best_dist = np.full(len(vert_ids), np.inf)
        for bmeta in buckets_meta:
            if bmeta.get("is_horizontal"):
                continue
            d = _circular_distance(vert_az, bmeta["az_center"])
            mask = d < bmeta["half_width"]
            update = mask & (d < best_dist)
            best_dist[update] = d[update]
            best_bucket[update] = bmeta["index"]
        bin_idx[vert_ids] = best_bucket
    return bin_idx, bucket_normals, buckets_meta


# ---------------------------------------------------------------------------
# B. 深度分层 v2：排序 + 间隙切割（硬约束 5cm）+ 边界双归属
# ---------------------------------------------------------------------------

def _bimodal_split_positions(values: np.ndarray, hard_m: float,
                             min_frac: float = 0.08) -> List[float]:
    """在一段 ρ 值内检测 5cm 尺度的双峰结构，返回切割阈值列表。

    动机：相邻间隙切割无法处理「两个立面噪声带互相重叠」的情形
    （排序链上不存在 >5cm 的相邻间隙，但直方图呈双峰、峰间距 >5cm）。
    按 hard_m 宽度分箱统计直方图，找显著谷值（两侧峰均高于谷值 2 倍
    且各自占比 ≥ min_frac，峰中位数间距 > hard_m），在谷底切开。
    """
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) < 100:
        return []
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo <= hard_m:
        return []
    # 分箱必须显著细于 5cm：与硬间距同宽的分箱会把双峰平滑成单调平台，
    # 谷底不可见。取 hard_m 的 40%（默认 2cm），谷检测窗按 hard_m 尺度。
    bs = float(hard_m) * 0.4
    nbins = max(4, int(np.ceil((hi - lo) / bs)))
    counts, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
    # 3-tap 轻平滑仅用于定位局部极小；峰谷强度比较用原始计数，
    # 避免宽窗平滑把 8cm 间距的双峰抹成单调平台。
    smooth = np.convolve(counts.astype(float), [0.25, 0.5, 0.25],
                         mode='same')

    cuts = []
    n_min = max(20, int(len(vals) * min_frac))
    for i in range(1, len(smooth) - 1):
        if not (smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]):
            continue
        left = vals <= edges[i + 1]
        right = vals > edges[i + 1]
        if left.sum() < n_min or right.sum() < n_min:
            continue
        left_peak = float(np.max(counts[:i + 1]))
        right_peak = float(np.max(counts[i + 1:]))
        valley = float(max(counts[i], 1.0))
        if left_peak < valley * 2.0 or right_peak < valley * 2.0:
            continue
        med_l = float(np.median(vals[left]))
        med_r = float(np.median(vals[right]))
        if med_r - med_l > float(hard_m):      # 峰间距 > 5cm 才允许切
            cuts.append(float(edges[i + 1]))
    return cuts


def split_depth_layers(rho: np.ndarray,
                       hard_gap_m: float,
                       boundary_eps_m: float = 0.0) -> List[np.ndarray]:
    """ρ 排序后按相邻间隙 > hard_gap_m 硬切层 + 层内 5cm 双峰切割。

    返回局部索引列表（指向传入 rho 的顺序）。保证：
      - 任意两层的 ρ 代表值（中位数）之差 > hard_gap_m；
      - 层内既无 >5cm 相邻间隙，也无峰间距 >5cm 的显著双峰结构。
    boundary_eps_m > 0 时，切点两侧 eps 带内的点双归属相邻层。
    """
    if len(rho) == 0:
        return []
    order = np.argsort(rho)
    sorted_rho = rho[order]
    gaps = np.diff(sorted_rho)
    cut_pos = np.flatnonzero(gaps > float(hard_gap_m)) + 1  # 切点（排序序）

    layers = []
    start = 0
    for cut in np.r_[cut_pos, len(order)]:
        layers.append(order[start:cut])
        start = cut

    # 层内双峰二次切割：噪声带重叠导致的伪单层在此拆开
    refined: List[np.ndarray] = []
    for layer in layers:
        cuts = _bimodal_split_positions(rho[layer], hard_gap_m)
        if not cuts:
            refined.append(layer)
            continue
        vals = rho[layer]
        bounds = [-np.inf] + cuts + [np.inf]
        for bi in range(len(bounds) - 1):
            sel = (vals > bounds[bi]) & (vals <= bounds[bi + 1])
            if np.count_nonzero(sel) >= 20:
                refined.append(layer[sel])

    layers = refined

    if boundary_eps_m > 0.0 and len(layers) > 1:
        eps = float(boundary_eps_m)
        extended = []
        for li, layer in enumerate(layers):
            layer = list(layer)
            r_lo, r_hi = float(rho[layer[0]]), float(rho[layer[-1]])
            # 与上一层的切点：把上一层尾部 eps 带内的点并入本层
            if li > 0:
                prev = layers[li - 1]
                for idx in prev[::-1]:
                    if r_lo - float(rho[idx]) <= eps:
                        layer.insert(0, idx)
                    else:
                        break
            # 与下一层的切点：把下一层头部 eps 带内的点并入本层
            if li < len(layers) - 1:
                nxt = layers[li + 1]
                for idx in nxt:
                    if float(rho[idx]) - r_hi <= eps:
                        layer.append(idx)
                    else:
                        break
            extended.append(np.asarray(layer, dtype=np.int64))
        layers = extended

    return layers


# ---------------------------------------------------------------------------
# C. Hough 层中心校验（主方向域：仅校验/精化，不再补种）
# ---------------------------------------------------------------------------

def _hough_layer_centers(pts: np.ndarray, direction: np.ndarray,
                         hard_gap_m: float, min_support: int,
                         voxel_size: float) -> List[Dict[str, Any]]:
    """在方向域内做 ρ-Hough 投票，返回层中心列表。"""
    bin_size = min(max(float(voxel_size), 0.02), hard_gap_m * 0.5)
    peaks = rho_hough_peaks(
        pts, direction,
        bin_size=bin_size,
        min_support=max(20, int(min_support)),
        min_peak_distance=float(hard_gap_m),      # 峰间距 = 5cm 硬约束
        prominence_ratio=float(getattr(Config, 'HOUGH_RHO_MIN_PROMINENCE', 0.08)),
        max_peaks=int(getattr(Config, 'HOUGH_MAX_RHO_PEAKS_PER_DIRECTION', 24)))
    return [{'rho': float(p.value), 'support': int(p.support)} for p in peaks]


def _match_hough_to_layers(hough_centers, layers, rho, hard_gap_m):
    """把 Hough 层中心分配给间隙切割层；返回精化后的层中心。

    仅做校验/精化：匹配半径收紧到 hard_gap/2，保证精化后层间距仍 > 5cm。
    （"未匹配峰补种"已删除——经两次运行 n_hough_rescued=0 证明该通道几何
    上不可触发，见模块 docstring 第 3 条。）
    """
    layer_rhos = [float(np.median(rho[layer])) for layer in layers]
    for hc in hough_centers:
        best, best_d = -1, float('inf')
        for li, lr in enumerate(layer_rhos):
            d = abs(hc['rho'] - lr)
            if d < best_d:
                best, best_d = li, d
        # 匹配半径收紧到 hard_gap/2，保证精化后层间距仍 > 5cm 不可破坏
        if best >= 0 and best_d <= hard_gap_m * 0.5:
            # Hough 峰精化层中心（仅当 Hough 支持度可观）
            if hc['support'] >= len(layers[best]) * 0.3:
                layer_rhos[best] = hc['rho']
    return layer_rhos


# ---------------------------------------------------------------------------
# C2. 补充通道：残余方向（bin_idx=-2）方向发现
# ---------------------------------------------------------------------------

def _discover_supplemental_directions(normals, mask, min_support,
                                      max_dirs=4, bin_deg=5.0,
                                      min_spacing_deg=20.0):
    """在弧外残余竖直点上发现补充方向。

    无向方位角 [0,π) 直方图 → 贪心不重叠峰（最小间距 min_spacing_deg）
    → 峰内成员法向中位数精化（z 置零），绝不使用直方图 bin 中心作为
    物理法向（bin 中心误差会在长墙上造成米级 ρ 漂移）。返回
    [(direction, support), ...]，按支持度降序。
    """
    idx = np.flatnonzero(mask)
    if len(idx) < max(20, int(min_support)):
        return []
    nrms = np.asarray(normals, dtype=np.float64)[idx]
    az = np.mod(np.arctan2(nrms[:, 1], nrms[:, 0]), np.pi)
    nbins = max(18, int(np.ceil(np.pi / np.deg2rad(float(bin_deg)))))
    counts, edges = np.histogram(az, bins=nbins, range=(0.0, np.pi))
    smooth = (np.convolve(counts.astype(float), [0.25, 0.5, 0.25], mode='same')
              if len(counts) >= 3 else counts.astype(float))
    peaks = []
    for i in range(len(smooth)):
        prev = smooth[i - 1] if i > 0 else 0.0
        nxt = smooth[(i + 1) % len(smooth)]
        if smooth[i] > prev and smooth[i] > nxt and smooth[i] >= min_support:
            peaks.append((float((edges[i] + edges[i + 1]) * 0.5),
                          float(smooth[i])))
    peaks.sort(key=lambda x: -x[1])
    selected = []
    min_rad = float(np.deg2rad(min_spacing_deg))
    for az_peak, _support in peaks:
        if any(abs(az_peak - s) < min_rad for s in selected):
            continue
        selected.append(az_peak)
        if len(selected) >= int(max_dirs):
            break

    az_deg = np.degrees(az)
    half = float(bin_deg)
    directions = []
    for az_peak in selected:
        members = idx[np.abs(az_deg - np.degrees(az_peak)) <= half]
        if not len(members):
            continue
        d = np.median(np.asarray(normals, dtype=np.float64)[members], axis=0)
        d[2] = 0.0
        length = float(np.linalg.norm(d))
        if length <= 1e-12:
            continue
        directions.append((d / length, int(len(members))))
    return directions


def _supplemental_layer_seeds(pts, nrms, global_ids, direction, layer_hard,
                              grid_size, eff_seed_min, density_field,
                              voxel_size):
    """残余方向域成种：ρ-Hough 发现层中心 → ±2.5cm 成窗 → 3D 连通域。

    残余域点云碎、稀，排序间隙切割会先切碎再补；ρ 投票按支持度直接给出
    层中心，正是 Hough 的结构性优势场景（Hough 无峰时回退间隙切割）。
    """
    rho = pts @ direction
    seeds = []
    min_win = max(20, eff_seed_min // 2)
    centers = _hough_layer_centers(
        pts, direction, layer_hard,
        min_support=min_win,
        voxel_size=float(voxel_size))
    if centers:
        candidates = [(float(c['rho']), int(c['support'])) for c in centers]
    else:
        layers = split_depth_layers(rho, layer_hard, 0.0)
        candidates = [(float(np.median(rho[layer])), int(len(layer)))
                      for layer in layers if len(layer) >= min_win]

    for rho_c, _support in candidates:
        # Hough 仅作层中心，收集窗严禁超过 ±2.5cm（同主链唯一层中心门控），
        # 否则会把相邻退台面吸入同一候选。
        sel = np.abs(rho - rho_c) <= layer_hard * 0.5
        sel_local = np.flatnonzero(sel)
        if len(sel_local) < min_win:
            continue
        gids = global_ids[sel_local]
        if density_field is not None:
            lv = density_field.gate_indices(gids)
            if lv == 'skip':
                continue
            comp_min = adaptive_min_count(eff_seed_min, lv)
        else:
            comp_min = eff_seed_min
        sub_pts = pts[sel_local]
        sub_nrms = nrms[sel_local]
        comps = connected_components_3d_grid(
            sub_pts, grid_size=grid_size, min_points=comp_min)
        if not comps:
            continue
        for comp_mask in comps:
            comp_local = np.flatnonzero(comp_mask)
            if len(comp_local) < comp_min:
                continue
            c_pts = sub_pts[comp_local]
            c_nrms = sub_nrms[comp_local]
            c_gids = gids[comp_local]
            try:
                model = fit_plane_weighted(c_pts, irls_iters=2)
            except Exception:
                continue
            n_vec = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
            if np.median(np.abs(c_nrms @ n_vec)) < np.cos(np.deg2rad(20.0)):
                continue
            seeds.append({
                "direction": n_vec.copy(),
                "rho": float(rho_c),
                "spatial_ids": c_gids,
                "support": int(len(c_gids)),
                "plane_model": model.copy(),
                "layer": -2,
                "dir_bin": -3,               # 补充通道标记（<0，不参与桶隔离）
                "density_level": 'dense',
                "supplemental": True,
            })
    return seeds


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_bucket_seeds(points, normals, mask, ranges, tol, base, min_count,
                       voxel_size, elevations=None, scan_origin=None,
                       density_field=None):
    """方向桶 × 深度层种子生成（v3）。

    新增参数
    --------
    elevations : (N,) 高度角，供密度估计分层
    density_field : 可复用外部已构建的 DensityField；None 时内部构建
    """
    started = time.perf_counter()
    valid = np.flatnonzero(mask)
    if len(valid) < min_count:
        return []

    pts = points[valid]
    nrms = normals[valid]
    rng_valid = np.asarray(ranges, dtype=float)[valid] if ranges is not None \
        else np.zeros(len(valid))

    # ---- 0. 密度场 ----
    if density_field is None and _HAS_DENSITY:
        try:
            density_field = build_density_field(
                pts, rng_valid,
                elevations=(np.asarray(elevations)[valid]
                            if elevations is not None else None),
                voxel_size=float(voxel_size))
        except Exception:
            density_field = None

    # ---- A. 主朝向与方向桶 ----
    principal = _estimate_principal_directions(nrms, vertical_nz=0.30)
    trace("facade.bucket.principal",
          azimuths=[float(np.degrees(a)) for a in principal],
          n_principal=len(principal))

    bin_idx_local, bucket_normals, buckets_meta = _quantize_directions(
        nrms,
        principal_azimuths=principal,
        arc_half_deg=float(getattr(Config, "FACADE_DIR_ARC_HALF_DEG", 28.0)),
        vertical_nz=float(getattr(Config, "VERTICAL_NZ_THR", 0.30)),
        fallback_n_buckets=int(getattr(Config, "FACADE_FALLBACK_BUCKETS", 8)),
    )

    # bin 分布观测（长期保留）：-2=弧外残余，-1=水平
    _uniq, _cnts = np.unique(bin_idx_local, return_counts=True)
    trace("facade.bucket.bin_histogram",
          bins={int(b): int(c) for b, c in zip(_uniq, _cnts)})

    layer_hard = float(getattr(Config, "FACADE_DEPTH_LAYER_HARD_M", 0.05))
    boundary_eps = min(float(getattr(Config, "FACADE_LAYER_BOUNDARY_EPS_M", 0.012)),
                       layer_hard * 0.25)
    grid_size = max(float(voxel_size) * 2.0, 0.10)
    irls_iters = int(getattr(Config, "FACADE_IRLS_ITERS", 2))
    seed_min_pts = max(50, min_count // 10)
    enable_hough_fusion = bool(getattr(
        Config, "FACADE_BUCKET_HOUGH_FUSION", True))

    all_seeds: List[Dict[str, Any]] = []
    n_dir_buckets = 0
    n_skipped_sparse = 0

    for bmeta in buckets_meta:
        if bmeta.get("is_horizontal"):
            continue
        bid = bmeta["index"]
        dir_mask_local = bin_idx_local == bid
        if not np.any(dir_mask_local):
            continue
        n_dir_buckets += 1

        bucket_pts = pts[dir_mask_local]
        bucket_nrms = nrms[dir_mask_local]
        bucket_global_ids = np.flatnonzero(dir_mask_local)

        # 桶级密度门控：整桶 skip 直接放弃
        bucket_level = 'dense'
        if density_field is not None:
            bucket_level = density_field.gate_indices(bucket_global_ids)
            if bucket_level == 'skip':
                n_skipped_sparse += 1
                continue
        eff_seed_min = (adaptive_min_count(seed_min_pts, bucket_level)
                        if _HAS_DENSITY else seed_min_pts)

        n_bucket = bucket_normals[bid] / (np.linalg.norm(bucket_normals[bid]) + 1e-12)
        rho = bucket_pts @ n_bucket
        if len(rho) == 0:
            continue

        # ---- B. 深度分层（排序+间隙切割，硬约束 5cm）----
        layers_local = split_depth_layers(rho, layer_hard, boundary_eps)

        # ---- C. Hough 层中心校验/精化（不做补种）----
        layer_rhos = [float(np.median(rho[layer])) for layer in layers_local]
        if enable_hough_fusion and len(rho) >= eff_seed_min * 2:
            hough_centers = _hough_layer_centers(
                bucket_pts, n_bucket, layer_hard,
                min_support=max(20, eff_seed_min // 2),
                voxel_size=float(voxel_size))
            if hough_centers and layers_local:
                layer_rhos = _match_hough_to_layers(
                    hough_centers, layers_local, rho, layer_hard)

        # ---- D. 逐层连通域 + 轻量拟合 ----
        for li, layer in enumerate(layers_local):
            layer_ids_local = bucket_global_ids[layer]
            layer_pts = bucket_pts[layer]
            layer_nrms = bucket_nrms[layer]
            if len(layer_ids_local) < eff_seed_min:
                continue

            if density_field is not None:
                lv = density_field.gate_indices(layer_ids_local)
                if lv == 'skip':
                    n_skipped_sparse += 1
                    continue
                comp_min = adaptive_min_count(eff_seed_min, lv)
            else:
                comp_min = eff_seed_min

            comps = connected_components_3d_grid(
                layer_pts, grid_size=grid_size, min_points=comp_min)
            if not comps:
                continue

            for comp_mask in comps:
                comp_local = np.flatnonzero(comp_mask)
                if len(comp_local) < comp_min:
                    continue
                comp_pts = layer_pts[comp_local]
                comp_nrms = layer_nrms[comp_local]
                comp_global_ids = layer_ids_local[comp_local]

                try:
                    model = fit_plane_weighted(comp_pts, irls_iters=irls_iters)
                except Exception:
                    continue
                n_vec = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
                cos_ang = np.abs(comp_nrms @ n_vec)
                if np.median(cos_ang) < np.cos(np.deg2rad(15.0)):
                    continue

                all_seeds.append({
                    "direction": n_vec.copy(),
                    "rho": float(layer_rhos[li]),     # Hough 精化后的层中心
                    "spatial_ids": valid[comp_global_ids],
                    "support": int(len(comp_global_ids)),
                    "plane_model": model.copy(),
                    "layer": int(li),
                    "dir_bin": bid,
                    "density_level": bucket_level,
                })

    # ---- F. 补充通道：弧外残余方向（bin_idx=-2）----
    n_supplemental = 0
    if bool(getattr(Config, 'FACADE_SUPPLEMENTAL_ENABLE', True)):
        sup_min_total = int(min_count * float(getattr(
            Config, 'FACADE_SUPPLEMENTAL_MIN_TOTAL_RATIO', 0.5)))
        sup_mask = bin_idx_local == -2
        n_sup_points = int(np.count_nonzero(sup_mask))
        if n_sup_points >= max(50, sup_min_total):
            sup_ids = np.flatnonzero(sup_mask)
            min_sup = max(50, seed_min_pts // 2)
            sup_dirs = _discover_supplemental_directions(
                nrms, sup_mask, min_support=min_sup,
                max_dirs=int(getattr(Config, 'FACADE_SUPPLEMENTAL_MAX_DIRS', 4)))
            sup_meta = []
            arc_agree = np.cos(np.deg2rad(float(getattr(
                Config, 'FACADE_DIR_ARC_HALF_DEG', 28.0))))
            for direction, support in sup_dirs:
                agree = np.abs(nrms[sup_ids] @ direction)
                m = agree >= arc_agree
                if int(np.count_nonzero(m)) < max(50, min_sup):
                    continue
                sel_ids = sup_ids[m]
                new_seeds = _supplemental_layer_seeds(
                    pts[sel_ids], nrms[sel_ids], valid[sel_ids], direction,
                    layer_hard, grid_size, seed_min_pts, density_field,
                    voxel_size)
                all_seeds.extend(new_seeds)
                n_supplemental += len(new_seeds)
                sup_meta.append({
                    'azimuth_deg': round(float(np.degrees(
                        np.arctan2(direction[1], direction[0]))) % 360.0, 1),
                    'support': int(support),
                    'n_seeds': int(len(new_seeds)),
                })
            if sup_meta:
                trace('facade.bucket.supplemental',
                      points=int(n_sup_points), directions=sup_meta)

    trace("facade.bucket.done",
          n_dir_buckets=n_dir_buckets,
          n_seeds=len(all_seeds),
          n_supplemental=n_supplemental,
          n_skipped_sparse=n_skipped_sparse,
          seconds=f"{time.perf_counter()-started:.2f}")
    return all_seeds