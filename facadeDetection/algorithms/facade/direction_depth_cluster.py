"""方向桶 × 深度层种子生成 v2：深度间隙切割 + Hough 验证融合 + 密度门控。

1. 【硬约束】深度分层由「排序 + 间隙切割」：
   ρ 升序排序后，相邻间隙 > ``FACADE_DEPTH_LAYER_HARD_M``(5cm) 即切层。
   性质保证：
     - 任意两个不同层的 ρ 代表值之差 > 5cm（切点即 >5cm 的间隙）；
     - 同一层内部 ρ 极差可以略大于 5cm（缓坡），但相邻点链上无 >5cm 跳变，
       即同层在深度方向连续——满足「相邻空间域候选深度差 >5cm 必分割」。
2. 【双归属落地】``FACADE_LAYER_BOUNDARY_EPS_M`` 真正生效：切点两侧 ε 带内
   的点同时归属相邻两层，避免硬切把同一面劈成两半（层合并阶段再去重）。
3. 【Hough 融合】每个方向桶内先跑 ``rho_hough_peaks`` 得到投票层中心，
   与间隙切割层互相校验：
     - 间隙切割有、Hough 也有的层 → 用 Hough 峰值精化 rho_center；
     - Hough 有、间隙切割没有的层（稀疏区连通域断裂所致）→ 以 Hough 层
       中心 ±bin 直接补种（sparse 补种通道，用更宽的距离窗收集点）；
     - 间隙切割有、Hough 无 → 保留（切割更保守，不误杀）。
4. 【密度门控】接入 density_estimator：单元/连通域级 'skip' 直接跳过；
   'sparse' 放宽 min_pts 并允许 Hough 补种。
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
# A. 方向桶化（沿用 v1，未改动）
# ---------------------------------------------------------------------------

def _estimate_principal_directions(normals, vertical_nz=0.30,
                                   min_support_ratio=0.01):
    """从 ROI 法向估计建筑主朝向 azimuth（Top-2 峰），返回弧度。"""
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
    if len(peaks) >= 2:
        az0, az1 = peaks[0][0], peaks[1][0]
        diff = float(abs(np.mod(az1 - az0 + np.pi, 2.0 * np.pi) - np.pi))
        if diff >= np.deg2rad(60.0):
            return [az0, az1]
    return [peaks[0][0]] if peaks else []


def _circular_distance(a, b):
    return np.abs(np.mod(a - b + np.pi, 2.0 * np.pi) - np.pi)


def _quantize_directions(normals, principal_azimuths, arc_half_deg=28.0,
                         vertical_nz=0.30, fallback_n_buckets=8):
    """方向桶化。返回 (bin_idx, bucket_normals, buckets_meta)。与 v1 相同。"""
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
        az0 = principal_azimuths[0]
        used = []
        for off in (0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2):
            c = float(np.mod(az0 + off, 2.0 * np.pi))
            if any(_circular_distance(np.array([c]), uc)[0] < np.deg2rad(5.0)
                   for uc in used):
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
# C. Hough 验证与补种
# ---------------------------------------------------------------------------

def _hough_layer_centers(pts: np.ndarray, direction: np.ndarray,
                         hard_gap_m: float, min_support: int,
                         voxel_size: float) -> List[Dict[str, Any]]:
    """在方向桶内做 ρ-Hough 投票，返回层中心列表。"""
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
    """把 Hough 层中心分配给间隙切割层；返回 (精化后的层中心, 未匹配 Hough 峰)。"""
    layer_rhos = [float(np.median(rho[layer])) for layer in layers]
    unmatched = []
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
        else:
            unmatched.append(hc)
    return layer_rhos, unmatched


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_bucket_seeds(points, normals, mask, ranges, tol, base, min_count,
                       voxel_size, elevations=None, scan_origin=None,
                       density_field=None):
    """方向桶 × 深度层种子生成（v2）。

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
    n_hough_rescued = 0

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

        # ---- C. Hough 层中心验证与补种 ----
        hough_centers = []
        unmatched_hough = []
        layer_rhos = [float(np.median(rho[layer])) for layer in layers_local]
        if enable_hough_fusion and len(rho) >= eff_seed_min * 2:
            hough_centers = _hough_layer_centers(
                bucket_pts, n_bucket, layer_hard,
                min_support=max(20, eff_seed_min // 2),
                voxel_size=float(voxel_size))
            if hough_centers and layers_local:
                layer_rhos, unmatched_hough = _match_hough_to_layers(
                    hough_centers, layers_local, rho, layer_hard)
            elif hough_centers:
                unmatched_hough = hough_centers

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

        # ---- E. Hough 补种：连通域断裂丢失的稀疏层 ----
        for hc in unmatched_hough:
            if hc['support'] < max(20, eff_seed_min // 2):
                continue
            # Hough 仅作补种中心，最终候选必须通过唯一层中心门控。
            # 禁止使用 ±5cm/±10cm 窗口，否则会把相邻退台面吸入同一候选。
            sel = np.abs(rho - hc['rho']) <= layer_hard * 0.5
            sel_ids = bucket_global_ids[sel]
            if len(sel_ids) < max(20, eff_seed_min // 2):
                continue
            if density_field is not None and \
                    density_field.gate_indices(sel_ids) == 'skip':
                continue
            sub_pts = bucket_pts[np.isin(bucket_global_ids, sel_ids)]
            sub_nrms = bucket_nrms[np.isin(bucket_global_ids, sel_ids)]
            try:
                model = fit_plane_weighted(sub_pts, irls_iters=irls_iters)
            except Exception:
                continue
            n_vec = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
            if np.median(np.abs(sub_nrms @ n_vec)) < np.cos(np.deg2rad(20.0)):
                continue
            all_seeds.append({
                "direction": n_vec.copy(),
                "rho": float(hc['rho']),
                "spatial_ids": valid[sel_ids],
                "support": int(hc['support']),
                "plane_model": model.copy(),
                "layer": -1,                 # Hough 补种层号独立
                "dir_bin": bid,
                "density_level": bucket_level,
                "hough_rescued": True,
            })
            n_hough_rescued += 1

    trace("facade.bucket.done",
          n_dir_buckets=n_dir_buckets,
          n_seeds=len(all_seeds),
          n_hough_rescued=n_hough_rescued,
          n_skipped_sparse=n_skipped_sparse,
          seconds=f"{time.perf_counter()-started:.2f}")
    return all_seeds