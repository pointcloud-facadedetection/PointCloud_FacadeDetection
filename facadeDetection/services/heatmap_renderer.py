"""Facade heatmap triplet renderer: overlay, isolated heatmap with grid, photo overlay."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import (
    heatmap_spec,
    defect_colormap,
    bipolar_colormap,
    cold_defect_colormap,
    apply_cold_excess_color,
)


# 画布分辨率上限：让 splat 在照片 warp 后保持合理物理大小。
_HEATMAP_MAX_SIZE = 3200


def _resolve_domain_uv_range(quality: dict):
    """从 quality 中提取立面有效 UV 域，作为 rasterize_facade 的画布边界。

    优先 quality_comparison.domain（FacadeQualityService 写入的权威值），
    缺失时回退 quality_domain。返回 (u_min, v_min, u_max, v_max) 或 None。

    为什么需要这个函数：`rasterize_facade` 若用 base_points 的 min/max 计算
    边界框，2800 万扫描点里的地面/天空/邻近建筑会把 v 跨度从真实的 ~100m
    膨胀到 ~1200m，导致 max_size 降采样后立面分辨率被稀释到 0.3m/像素，
    warp 到照片后 splat 变成巨块。显式传入质量评估定义的有效域可根治。
    """
    if not isinstance(quality, dict):
        return None

    domain = None
    comparison = quality.get('quality_comparison') or {}
    if isinstance(comparison, dict):
        domain = comparison.get('domain')
    if not isinstance(domain, dict):
        domain = quality.get('quality_domain') or {}
    if not isinstance(domain, dict):
        return None

    keys = ('u_min_m', 'v_min_m', 'u_max_m', 'v_max_m')
    if not all(k in domain for k in keys):
        return None
    try:
        u0 = float(domain['u_min_m'])
        v0 = float(domain['v_min_m'])
        u1 = float(domain['u_max_m'])
        v1 = float(domain['v_max_m'])
    except (TypeError, ValueError):
        return None

    if not (np.isfinite([u0, v0, u1, v1]).all() and u1 > u0 and v1 > v0):
        return None

    # 向外扩 5% 边距，确保边缘缺陷不被裁掉
    du = (u1 - u0) * 0.05
    dv = (v1 - v0) * 0.05
    return (u0 - du, v0 - dv, u1 + du, v1 + dv)


class FacadeHeatmapTripletRenderer:
    """
    每组 3 张子图：
      - overlay:      点云底图 + 缺陷热力叠加
      - heatmap_grid: 独立热力图 + 1.0 m 细分网格（无点云）
      - photo:        2D 照片对齐叠加（预留接口，无照片时返回 None）
    """

    GRID_STEP_M = 1.0
    _OVERLAY_BASE_COLOR = np.array([230, 232, 235], dtype=np.uint8)
    _OVERLAY_FILL_COLOR = (238, 240, 243)
    _HEATMAP_BG_COLOR = 248

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(
        self,
        mode: str,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        windows: list[dict],
        plane_model: list[float],
        quality: dict,
        pixel_size: float = 0.01,
        photo_path: Optional[str] = None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Render a single heatmap triplet for the given mode."""
        spec = heatmap_spec(mode)
        method = spec.get("method", "ruler")
        metric = spec.get("metric", "flatness")

        # ---- 按 method + metric 路由到不同的样本准备策略 ----
        physical_cell_m = None
        if method == "ruler" and metric == "flatness":
            prepare_result = self._prepare_ruler_flatness_grid(
                windows, spec, quality)
            if prepare_result is None:
                prepare_result = self._prepare_subgrid(
                    windows, spec, quality, metric='flatness')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        elif method == "global_plane" and metric == "flatness":
            prepare_result = self._prepare_global_plane_point_samples(
                points, windows, spec, quality, metric='flatness')
            if prepare_result is None:
                raise ValueError(f'global {metric} requires explicit defect samples')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        elif method == "global_plane" and metric == "verticality":
            prepare_result = self._prepare_global_plane_point_samples(
                points, windows, spec, quality, metric='verticality')
            if prepare_result is None:
                raise ValueError(f'global {metric} requires explicit defect samples')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.15
        elif method == "ruler" and metric == "verticality":
            prepare_result = self._prepare_ruler_verticality_points(
                windows, spec, quality)
            if prepare_result is None:
                prepare_result = self._prepare_subgrid(
                    windows, spec, quality, metric='verticality')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        else:
            defect_points, values, defect_colors = self._prepare_subgrid(
                windows, spec, quality, metric=metric)
            physical_cell_m = 0.10 if metric == "verticality" else None
        if len(defect_points) == 0:
            raise ValueError(f"No valid defect windows for mode {mode}")

        values_m = values / 1000.0
        profile = quality.get("profile_snapshot", {}) or {}
        limit_mm = float(profile.get(
            spec["limit_key"],
            quality.get("thresholds", {}).get(
                spec["limit_key"],
                quality.get("parameters", {}).get(spec["limit_key"], 4.0),
            ),
        ))
        limit_m = limit_mm / 1000.0

        pts_arr = np.asarray(points, dtype=float).reshape(-1, 3)
        merged_base = (pts_arr if len(pts_arr)
                       else np.asarray(defect_points, dtype=float).reshape(-1, 3))

        base_colors = np.full((len(merged_base), 3),
                              [0.80, 0.83, 0.86], dtype=float)

        excess = np.maximum(np.abs(values) - limit_mm, 0.0)
        vmax_m = (
            float(limit_m + np.max(excess) / 1000.0)
            if len(excess) and np.any(np.isfinite(excess))
            else limit_m * 1.2
        )

        # ★ 从质量评估的有效域取画布范围，避免离群点稀释分辨率
        uv_range = _resolve_domain_uv_range(quality)

        raster = rasterize_facade(
            defect_points,
            np.full((len(defect_points), 3), 0.7),
            plane_model,
            values_m,
            limit_m,
            pixel_size=pixel_size,
            defect_colors=defect_colors,
            vmin=limit_m,
            vmax=vmax_m,
            max_size=_HEATMAP_MAX_SIZE,
            projection_origin=quality.get("projection_origin"),
            projection_u_axis=quality.get("projection_u_axis"),
            projection_v_axis=quality.get("projection_v_axis"),
            base_points=merged_base,
            base_colors=base_colors,
            physical_cell_m=physical_cell_m,
            min_splat_px=4,
            uv_range=uv_range,          # ★ 显式画布边界
        )

        overlay = self._build_overlay(raster)
        heatmap_grid = self._build_isolated_heatmap_with_grid(
            raster, self.GRID_STEP_M)

        # 与热力图色带一致：
        #   ruler_*        → cold_defect_colormap（单极冷色）
        #   global_plane_* → bipolar_colormap（双极）
        is_bipolar = (method == "global_plane"
                      and metric in ("flatness", "verticality"))
        is_cold = (method == "ruler"
                   and metric in ("flatness", "verticality"))
        overlay = self._embed_legend(overlay, raster, is_bipolar=is_bipolar,
                                     is_cold=is_cold, metric=metric)
        heatmap_grid = self._embed_legend(heatmap_grid, raster,
                                          is_bipolar=is_bipolar,
                                          is_cold=is_cold, metric=metric)

        if photo_path and Path(photo_path).is_file():
            photo = self._build_photo_overlay(raster, photo_path)
        else:
            overlay_rgba = raster["overlay_rgba"].copy()
            overlay_rgba[:, :, 3] = np.where(
                overlay_rgba[:, :, 3] > 0,
                np.maximum(overlay_rgba[:, :, 3], 210),
                0,
            ).astype(np.uint8)
            cropped_rgb = self._crop_to_facade(
                overlay_rgba, raster, fill=(0, 0, 0, 0))
            photo = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
            photo = self._embed_legend(photo, raster, is_bipolar=is_bipolar,
                                       is_cold=is_cold, metric=metric)

        return {
            "overlay": self._to_bgr(overlay),
            "heatmap_grid": self._to_bgr(heatmap_grid),
            "photo": self._to_bgr(photo) if photo is not None else None,
        }

    @staticmethod
    def _to_bgr(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if image is None:
            return None
        img = np.asarray(image, dtype=np.uint8)
        if img.ndim != 3:
            return img
        if img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 252.0)
            blended = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
            return blended
        return img

    # ------------------------------------------------------------------
    # Sample preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_windows(
        windows: list[dict], spec: dict, quality: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回退路径：只取失败窗口的中心点（散点）。"""
        centers_list, values_list = [], []
        for w in windows or []:
            if bool(w.get(spec["pass_key"], True)):
                continue
            cx = w.get("center_xyz")
            if cx is None or len(cx) != 3:
                continue
            try:
                if not all(np.isfinite(float(x)) for x in cx):
                    continue
            except (TypeError, ValueError):
                continue
            val = w.get(spec["value_key"], np.nan)
            try:
                val = float(val)
                if not np.isfinite(val):
                    continue
            except (TypeError, ValueError):
                continue
            centers_list.append([float(x) for x in cx])
            values_list.append(val)

        centers = np.asarray(centers_list, dtype=float).reshape(-1, 3)
        values = np.asarray(values_list, dtype=float)

        profile = quality.get("profile_snapshot", {}) or {}
        limit_mm = float(profile.get(
            spec["limit_key"],
            quality.get("thresholds", {}).get(
                spec["limit_key"],
                quality.get("parameters", {}).get(spec["limit_key"], 4.0),
            ),
        ))
        excess = np.maximum(np.abs(values) - limit_mm, 0.0)
        finite_excess = excess[np.isfinite(excess)]
        if len(finite_excess) > 0 and np.any(finite_excess > 0):
            p98 = float(np.percentile(finite_excess, 98))
            scale_mm = max(p98, limit_mm * 0.15)
        else:
            scale_mm = limit_mm * 0.15
        t = np.clip(excess / scale_mm, 0.0, 1.0)
        return centers, values, defect_colormap(t)

    @staticmethod
    def _prepare_ruler_verticality_points(windows, spec, quality):
        """靠尺垂直度专用：优先使用 strip worker 输出的 3D 靠点坐标。"""
        centers, values = [], []
        for w in windows or []:
            xyzs = w.get('verticality_defect_point_xyzs', [])
            vals = w.get('verticality_defect_values_mm', [])
            if not xyzs or not vals:
                continue
            for xyz, val in zip(xyzs, vals):
                if xyz is None or len(xyz) != 3:
                    continue
                try:
                    if not all(np.isfinite(float(x)) for x in xyz):
                        continue
                except (TypeError, ValueError):
                    continue
                centers.append([float(x) for x in xyz])
                values.append(abs(float(val)))

        if not centers:
            return None

        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        values = np.asarray(values, dtype=float)
        profile = quality.get("profile_snapshot", {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            (quality.get("thresholds") or {}).get(
                spec["limit_key"],
                (quality.get("parameters") or {}).get(spec["limit_key"], 4.0),
            ),
        ))
        excess = np.maximum(values - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(
            float(np.percentile(finite, 98)) if finite.size else limit * 0.15,
            limit * 0.15, 1e-6,
        )
        return centers, values, cold_defect_colormap(
            np.clip(excess / scale, 0.0, 1.0))

    @staticmethod
    def _prepare_subgrid(windows, spec, quality, metric='flatness'):
        """把窗口沿靠尺方向展开为物理块（与统计同源）。"""
        params = quality.get("parameters", {}) or {}
        origin = np.asarray(quality.get("projection_origin"), dtype=float)
        u_axis = np.asarray(quality.get("projection_u_axis"), dtype=float)
        v_axis = np.asarray(quality.get("projection_v_axis"), dtype=float)

        if (origin.shape != (3,) or not np.all(np.isfinite(origin))
                or u_axis.shape != (3,) or not np.all(np.isfinite(u_axis))
                or v_axis.shape != (3,) or not np.all(np.isfinite(v_axis))):
            return FacadeHeatmapTripletRenderer._prepare_windows(
                windows, spec, quality)

        u_axis = u_axis / max(np.linalg.norm(u_axis), 1e-12)
        v_axis = v_axis / max(np.linalg.norm(v_axis), 1e-12)

        profile = quality.get("profile_snapshot", {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            (quality.get("thresholds") or {}).get(
                spec["limit_key"],
                params.get(spec["limit_key"], 4.0),
            ),
        ))

        method = spec.get("method", "ruler")
        if method == "global_plane":
            length = float(params.get("window_length_m", 2.0))
            width = float(params.get("window_width_m", 0.055))
        else:
            length = float(params.get("ruler_length_m",
                                      params.get("window_length_m", 2.0)))
            width = float(params.get("ruler_width_m",
                                     params.get("window_width_m", 0.055)))
        if length <= 0 or width <= 0:
            length, width = 2.0, 0.055

        global_plane = (method == "global_plane")
        vertical = (spec.get("metric") == "verticality")

        centers, values = [], []
        for window in windows or []:
            try:
                value = float(window.get(spec["value_key"], np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue

            pass_value = window.get(spec["pass_key"])
            explicit_fail = False if pass_value is None else (not bool(pass_value))
            value_exceeds = abs(value) > limit
            if not (explicit_fail or value_exceeds):
                continue

            if metric == 'verticality':
                xyzs = window.get('verticality_defect_point_xyzs', [])
                vals = window.get('verticality_defect_values_mm', [])
                if xyzs and vals:
                    for xyz, val in zip(xyzs, vals):
                        if xyz is None or len(xyz) != 3:
                            continue
                        try:
                            if not all(np.isfinite(float(x)) for x in xyz):
                                continue
                        except (TypeError, ValueError):
                            continue
                        centers.append([float(x) for x in xyz])
                        values.append(abs(float(val)))
                    continue

            center = np.asarray(window.get("center_xyz"), dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                continue

            if global_plane or vertical:
                angle = 90.0
            else:
                try:
                    angle = float(window.get("direction_deg", 0.0))
                except (TypeError, ValueError):
                    angle = 0.0
            rad = np.deg2rad(angle)
            along = np.cos(rad) * u_axis + np.sin(rad) * v_axis
            across = -np.sin(rad) * u_axis + np.cos(rad) * v_axis

            step_long = 0.05
            step_wide = 0.025
            n_long = max(1, int(np.ceil(length / step_long)))
            n_wide = max(1, int(np.ceil(width / step_wide)))
            for i in range(n_long):
                a = (i + 0.5) * length / n_long - length / 2
                for j in range(n_wide):
                    b = (j + 0.5) * width / n_wide - width / 2
                    centers.append(center + a * along + b * across)
                    values.append(value)

        if not centers:
            return FacadeHeatmapTripletRenderer._prepare_windows(
                windows, spec, quality)

        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        values = np.asarray(values, dtype=float)
        excess = np.maximum(np.abs(values) - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(
            float(np.percentile(finite, 98)) if finite.size else limit * 0.15,
            limit * 0.15, 1e-6,
        )
        # ★ 靠尺法（ruler）统一使用单极冷色调，与色条 is_cold 一致
        if method == 'ruler':
            colors = cold_defect_colormap(np.clip(excess / scale, 0.0, 1.0))
        else:
            colors = defect_colormap(np.clip(excess / scale, 0.0, 1.0))
        return centers, values, colors

    @staticmethod
    def _prepare_ruler_flatness_grid(windows, spec, quality):
        """靠尺平整度专用：解析 ruler_defect_grid 得到 40 个网格单元中心点。

        ★ 与色条 is_cold 保持一致，使用单极冷色色带。
        """
        origin = np.asarray(quality.get("projection_origin"), dtype=float)
        u_axis = np.asarray(quality.get("projection_u_axis"), dtype=float)
        v_axis = np.asarray(quality.get("projection_v_axis"), dtype=float)

        if (origin.shape != (3,) or not np.all(np.isfinite(origin))
                or u_axis.shape != (3,) or not np.all(np.isfinite(u_axis))
                or v_axis.shape != (3,) or not np.all(np.isfinite(v_axis))):
            return None

        u_axis = u_axis / max(np.linalg.norm(u_axis), 1e-12)
        v_axis = v_axis / max(np.linalg.norm(v_axis), 1e-12)

        profile = quality.get("profile_snapshot", {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            (quality.get("thresholds") or {}).get(
                spec["limit_key"],
                (quality.get("parameters") or {}).get(spec["limit_key"], 4.0),
            ),
        ))

        centers, values = [], []
        for w in windows or []:
            grid_info = w.get('ruler_defect_grid')
            if not grid_info or not grid_info.get('grids'):
                continue
            u_center = float(w.get('u_center', np.nan))
            direction_deg = float(w.get('direction_deg', 0.0))
            if not np.isfinite(u_center):
                continue
            rad = np.deg2rad(direction_deg)
            along = np.cos(rad) * u_axis + np.sin(rad) * v_axis
            center = np.asarray(w.get('center_xyz'), dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                continue
            for g in grid_info['grids']:
                if abs(g.get('max_defect_mm', 0)) <= limit:
                    continue
                local_center = (g['u_start_m'] + g['u_end_m']) / 2.0
                u_offset = local_center - u_center
                grid_center = center + u_offset * along
                centers.append(grid_center)
                values.append(abs(g['max_defect_mm']))

        if not centers:
            return None

        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        values = np.asarray(values, dtype=float)
        excess = np.maximum(values - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(
            float(np.percentile(finite, 98)) if finite.size else limit * 0.15,
            limit * 0.15, 1e-6,
        )
        # ★ 单极冷色，与 is_cold 图例一致
        return centers, values, cold_defect_colormap(
            np.clip(excess / scale, 0.0, 1.0))

    @staticmethod
    def _prepare_global_plane_point_samples(points, windows, spec, quality,
                                             metric='flatness'):
        """模拟墙面专用：每个超阈值点作为一个 splat 中心。

        语义严格的四类候选（全部为 raw_global_row 语义，可直接索引
        ``processed_raw_points``）：

          1) quality['defect_samples'][metric]['raw_ids']
          2) quality['defect_samples'][metric]['source_rows']
             + quality['__global_indices']
          3) windows[].defect_point_rows + quality['__global_indices']
          4) windows[].defect_point_indices
        """
        source = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(source) == 0:
            return None
        n_source = int(len(source))

        raw_ids_src = quality.get('__global_indices')
        raw_ids = (np.asarray(raw_ids_src, dtype=np.int64).reshape(-1)
                   if raw_ids_src is not None
                   else np.empty(0, dtype=np.int64))
        n_raw = int(len(raw_ids))

        profile = quality.get('profile_snapshot', {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            (quality.get("thresholds") or {}).get(
                spec["limit_key"],
                (quality.get("parameters") or {}).get(spec["limit_key"], 4.0),
            ),
        ))

        candidates = []      # [(rows, values, name)]
        total_defect = 0

        def _push(rows, values, name):
            nonlocal total_defect
            rows = np.asarray(rows, dtype=np.int64).reshape(-1)
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            n = min(len(rows), len(values))
            if n == 0:
                return
            rows = rows[:n]
            values = values[:n]
            valid = ((rows >= 0) & (rows < n_source) & np.isfinite(values))
            if not np.any(valid):
                return
            rows = rows[valid]
            values = values[valid]
            if len(rows) > 1:
                order = np.argsort(-np.abs(values), kind='stable')
                _, first = np.unique(rows[order], return_index=True)
                keep = order[first]
                rows, values = rows[keep], values[keep]
            candidates.append((rows, values, name))
            total_defect = max(total_defect, n)

        # ============ 源 1、2：方法级 defect_samples（必须显式注入）============
        sample = (quality.get('defect_samples') or {}).get(metric)
        if isinstance(sample, dict):
            vals = np.asarray(sample.get('values_mm', []),
                              dtype=np.float64).reshape(-1)
            # 源 1：raw_ids 直接是 processed_raw_points 行号
            _push(sample.get('raw_ids', []), vals,
                  'defect_samples.raw_ids')
            # 源 2：source_rows 是 filtered_pts 行号 → 经 __global_indices 映射
            src_rows = np.asarray(sample.get('source_rows', []),
                                  dtype=np.int64).reshape(-1)
            index_space = sample.get('index_space', 'raw_global_rows')
            # 若上游声明为 source_row 但无 __global_indices，尝试从 quality 顶层加载
            if index_space == 'source_row' and n_raw == 0 and src_rows.size:
                from services.dal.results_repo import ResultsRepo
                ResultsRepo.ensure_global_indices(quality)
                raw_ids_src = quality.get('__global_indices')
                raw_ids = (np.asarray(raw_ids_src, dtype=np.int64).reshape(-1)
                           if raw_ids_src is not None
                           else np.empty(0, dtype=np.int64))
                n_raw = int(len(raw_ids))
            if n_raw > 0 and src_rows.size:
                m = (src_rows >= 0) & (src_rows < n_raw)
                if np.any(m):
                    _push(raw_ids[src_rows[m]], vals[m],
                          'defect_samples.source_rows_via_global')

        # ============ 源 3、4：windows 拼接 ============
        if metric == 'verticality':
            rows_key = 'verticality_defect_point_rows'
            indices_key = 'verticality_defect_point_indices'
            vals_key = 'verticality_defect_values_mm'
        else:
            rows_key = 'defect_point_rows'
            indices_key = 'defect_point_indices'
            vals_key = 'defect_values_mm'

        rows_chunks, indices_chunks, values_chunks = [], [], []
        for window in windows or []:
            vals = np.asarray(window.get(vals_key, []),
                              dtype=np.float64).reshape(-1)
            if vals.size == 0:
                continue
            n = int(vals.size)
            r = np.asarray(window.get(rows_key, []),
                           dtype=np.int64).reshape(-1)
            i = np.asarray(window.get(indices_key, []),
                           dtype=np.int64).reshape(-1)
            rows_chunks.append(r[:n] if r.size else np.full(n, -1, np.int64))
            indices_chunks.append(i[:n] if i.size else np.full(n, -1, np.int64))
            values_chunks.append(vals)

        if values_chunks:
            all_rows = np.concatenate(rows_chunks)
            all_indices = np.concatenate(indices_chunks)
            all_values = np.concatenate(values_chunks)

            # 源 3：defect_point_rows（filtered_pts 行号）经二次映射
            if n_raw > 0:
                m = (all_rows >= 0) & (all_rows < n_raw)
                if np.any(m):
                    _push(raw_ids[all_rows[m]], all_values[m],
                          'windows.rows_via_global')
            # 源 4：defect_point_indices 就是 raw 全局行号，直接索引
            _push(all_indices, all_values, 'windows.indices_direct')

        # ============ 源 5：回退到窗口中心点（防御性，保证不空白）============
        if not candidates:
            centers, center_values = [], []
            for window in windows or []:
                val = window.get(spec.get("value_key", "flatness_gap_mm"), np.nan)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(val) or abs(val) <= limit:
                    continue
                cx = window.get("center_xyz")
                if cx is None or len(cx) != 3:
                    continue
                # 通过坐标反查 source 中最接近的行号（O(n) 但仅在回退时执行）
                centers.append(cx)
                center_values.append(val)
            if centers:
                centers_arr = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
                # 找每个中心在 source 中的最近邻行号
                rows_fallback = []
                for c in centers_arr:
                    dists = np.sum((source - c) ** 2, axis=1)
                    rows_fallback.append(int(np.argmin(dists)))
                _push(rows_fallback, center_values, 'fallback.window_centers_nearest')

        # ============ 选择：raw_global_row 语义候选，全部落空即失败 ============
        if not candidates:
            print(
                f'[PCFD] _prepare_global_plane_point_samples: 无 raw_global_row '
                f'候选 (defect_total={total_defect}, n_source={n_source}, '
                f'n_raw={n_raw}, metric={metric})',
                flush=True,
            )
            return None

        # 若某候选覆盖 >= 90% 缺陷点，直接采纳（避免被稀疏候选挤掉）
        best = None
        for rows, values, name in candidates:
            if total_defect > 0 and len(rows) >= 0.9 * total_defect:
                best = (rows, values, name)
                break
        if best is None:
            best = max(candidates, key=lambda c: len(c[0]))

        rows, sel_values, strategy = best
        centers = source[rows]
        print(
            f'[PCFD] defect.samples: metric={metric} source={strategy} '
            f'index_space=raw_global_row matched={len(rows)}/{total_defect}',
            flush=True,
        )
        return centers, sel_values, bipolar_colormap(sel_values, limit)

    @staticmethod
    def _prepare_point_samples(points, windows, spec, quality):
        """备选：把窗口值贴回原始点云（依赖 covered_source_ids）。"""
        source = np.asarray(points, dtype=float).reshape(-1, 3)
        from services.dal.results_repo import ResultsRepo
        ResultsRepo.ensure_global_indices(quality)
        raw_ids = np.asarray(quality.get('__global_indices', []), dtype=np.int64)
        if len(raw_ids) != len(source):
            raw_ids = np.arange(len(source), dtype=np.int64)
        id_to_row = {int(value): i for i, value in enumerate(raw_ids.tolist())}
        point_values = np.full(len(source), -np.inf, dtype=float)
        profile = quality.get('profile_snapshot', {}) or {}
        limit = float(profile.get(
            spec['limit_key'],
            (quality.get('thresholds') or {}).get(
                spec['limit_key'],
                (quality.get('parameters') or {}).get(spec['limit_key'], 4.0))))
        for window in windows or []:
            if bool(window.get(spec['pass_key'], True)):
                continue
            try:
                value = float(window.get(spec['value_key'], np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value) or abs(value) <= limit:
                continue
            ids = np.asarray(window.get('covered_source_ids', []),
                             dtype=np.int64).reshape(-1)
            for source_id in ids.tolist():
                row = id_to_row.get(int(source_id))
                if row is not None and abs(value) > point_values[row]:
                    point_values[row] = value

        selected = np.isfinite(point_values)
        if not np.any(selected):
            return FacadeHeatmapTripletRenderer._prepare_windows(
                windows, spec, quality)
        selected_values = point_values[selected]
        excess = np.maximum(np.abs(selected_values) - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(
            float(np.percentile(finite, 98)) if finite.size else limit * .15,
            limit * .15, 1e-6,
        )
        return source[selected], selected_values, defect_colormap(
            np.clip(excess / scale, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Overlay（点云底图 + 热力）
    # ------------------------------------------------------------------
    def _build_overlay(self, raster: dict) -> np.ndarray:
        overlay_rgba = np.asarray(raster["overlay_rgba"])
        h, w = overlay_rgba.shape[:2]
        overlay_rgba[:, :, 3] = np.where(
            overlay_rgba[:, :, 3] > 0,
            np.maximum(overlay_rgba[:, :, 3], 210),
            0,
        ).astype(np.uint8)

        base_rgb = raster.get("base_rgb")
        facade_mask = np.asarray(raster.get("facade_mask"), dtype=bool)

        if base_rgb is None:
            base_rgb = np.full((h, w, 3), self._OVERLAY_BASE_COLOR.tolist(),
                               dtype=np.uint8)
        else:
            base_rgb = np.asarray(base_rgb, dtype=np.uint8).copy()
            if facade_mask.shape == base_rgb.shape[:2]:
                base_rgb[~facade_mask] = self._OVERLAY_BASE_COLOR
            else:
                empty = np.all(base_rgb < 4, axis=2)
                base_rgb[empty] = self._OVERLAY_BASE_COLOR

        base_rgb = base_rgb.astype(np.float32)
        overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
        visible = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

        composite_rgb = (
            base_rgb * (1.0 - visible[:, :, :1])
            + overlay_rgb * visible[:, :, :1]
        )
        composite_bgr = cv2.cvtColor(
            np.clip(composite_rgb, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        return self._crop_to_facade(composite_bgr, raster,
                                    fill=self._OVERLAY_FILL_COLOR)

    # ------------------------------------------------------------------
    # 独立热力图 + 网格（不含点云）
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=1,
                          dash_len=6, gap_len=6):
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        dist = max(np.hypot(dx, dy), 1e-6)
        ux, uy = dx / dist, dy / dist
        segment = dash_len + gap_len
        n_segments = int(np.ceil(dist / segment))
        for i in range(n_segments):
            s0 = i * segment
            s1 = min(s0 + dash_len, dist)
            if s1 <= s0:
                continue
            cv2.line(img,
                     (int(round(x1 + ux * s0)), int(round(y1 + uy * s0))),
                     (int(round(x1 + ux * s1)), int(round(y1 + uy * s1))),
                     color, thickness, cv2.LINE_AA)

    def _build_isolated_heatmap_with_grid(
        self, raster: dict, grid_step_m: float = 1.0
    ) -> np.ndarray:
        overlay_rgba = raster["overlay_rgba"].copy()
        h, w = overlay_rgba.shape[:2]
        pixel_size = raster["pixel_size"]
        grid_px = max(int(grid_step_m / pixel_size), 1)

        overlay_rgba[:, :, 3] = np.where(
            overlay_rgba[:, :, 3] > 0,
            np.maximum(overlay_rgba[:, :, 3], 210),
            0,
        ).astype(np.uint8)

        line_color = (150, 160, 175, 170)
        border_color = (95, 105, 120, 220)

        for x in range(grid_px, w, grid_px):
            self._draw_dashed_line(overlay_rgba, (x, 0), (x, h - 1),
                                   line_color, 1, dash_len=6, gap_len=6)
        for y in range(grid_px, h, grid_px):
            self._draw_dashed_line(overlay_rgba, (0, y), (w - 1, y),
                                   line_color, 1, dash_len=6, gap_len=6)

        cv2.rectangle(overlay_rgba, (0, 0), (w - 1, h - 1), border_color, 1)

        background = np.full((h, w, 3), self._HEATMAP_BG_COLOR, dtype=np.uint8)
        alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        foreground = overlay_rgba[:, :, :3].astype(np.float32)
        composite = (
            background.astype(np.float32) * (1.0 - alpha)
            + foreground * alpha
        ).astype(np.uint8)
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        return self._crop_to_facade(bgr, raster,
                                    fill=(self._HEATMAP_BG_COLOR,) * 3)

    # ------------------------------------------------------------------
    # Crop & photo
    # ------------------------------------------------------------------
    @staticmethod
    def _crop_to_facade(image: np.ndarray, raster: dict,
                        fill=(245, 245, 245)) -> np.ndarray:
        mask = np.asarray(raster.get("facade_mask"), dtype=bool)
        if mask.shape != image.shape[:2] or not np.any(mask):
            return image

        min_ratio = 0.02
        h, w = mask.shape
        col_cov = mask.sum(axis=0) / float(h)
        row_cov = mask.sum(axis=1) / float(w)
        cols = np.where(col_cov >= min_ratio)[0]
        rows = np.where(row_cov >= min_ratio)[0]
        if len(cols) == 0 or len(rows) == 0:
            cols = np.where(mask.any(axis=0))[0]
            rows = np.where(mask.any(axis=1))[0]
            if len(cols) == 0 or len(rows) == 0:
                return image

        x0, x1 = int(cols[0]), int(cols[-1])
        y0, y1 = int(rows[0]), int(rows[-1])

        if image.ndim == 3 and image.shape[2] == 4:
            cropped = image[y0:y1 + 1, x0:x1 + 1, :]
        else:
            cropped = image[y0:y1 + 1, x0:x1 + 1]
        return cropped.copy()

    def _build_photo_overlay(
        self, raster: dict, photo_path: str
    ) -> Optional[np.ndarray]:
        photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        if photo is None:
            return None
        h, w = raster["overlay_rgba"].shape[:2]
        return cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    @staticmethod
    def _embed_legend(image: np.ndarray, raster: dict, is_bipolar: bool = False,
                      is_cold: bool = False, metric: str = 'flatness') -> np.ndarray:
        src = np.asarray(image)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            return image
        if src.shape[2] == 4:
            src = src[:, :, :3]
        bgr = src
        h, w = bgr.shape[:2]

        bar_w = int(np.clip(round(h * 0.030), 22, 34))
        gap = int(np.clip(round(h * 0.022), 18, 30))
        label_w = int(np.clip(round(h * 0.055), 34, 62))
        pad = 6
        gutter = pad + label_w + bar_w + gap

        canvas = np.full((h, w + gutter, 3), 248, dtype=np.uint8)
        canvas[:, gutter:, :] = bgr

        bar_x = pad + label_w
        top = max(int(h * 0.10), 24)
        bottom = min(h - max(int(h * 0.10), 24), h - 20)
        if bottom - top < 40:
            top, bottom = 12, max(h - 12, 13)
        count = max(bottom - top, 1)

        limit_m = float(raster.get('vmin', 0.0))
        max_m = float(raster.get('vmax', limit_m))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(np.clip(h / 1400.0, 0.30, 0.42))
        thick = 1

        if is_bipolar:
            mid_y = (top + bottom) // 2
            scale_mm = max(max_m - limit_m, limit_m * 0.15, 1e-6) * 1000.0
            signed_mm = np.linspace(max_m * 1000.0, -max_m * 1000.0, count)
            colours = (bipolar_colormap(
                signed_mm, limit_m * 1000.0, scale_mm=scale_mm
            ) * 255.0).astype(np.uint8)
            canvas[top:bottom, bar_x:bar_x + bar_w, :] = colours[:, ::-1][:, None, :]
            cv2.line(canvas, (bar_x - 1, mid_y), (bar_x + bar_w, mid_y),
                     (90, 90, 90), 1)

            if metric == 'verticality':
                pos_label, neg_label = '外倾', '内陷'
            else:
                pos_label, neg_label = '凹陷', '凸起'
            for value_m, y, label in (
                (max_m, top + 4, pos_label),
                (0.0, mid_y, ''),
                (-max_m, bottom - 2, neg_label),
            ):
                if value_m != 0.0 or label == '':
                    text = f'{abs(value_m) * 1000:.1f}'
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    cv2.putText(canvas, text,
                                (bar_x - 4 - tw, min(max(y, th), h - 2)),
                                font, scale, (60, 62, 68), thick, cv2.LINE_AA)
                if label:
                    cv2.putText(canvas, label,
                                (bar_x - 4 - label_w + 4,
                                 min(max(y, th), h - 2)),
                                font, scale * 0.9, (60, 62, 68), thick,
                                cv2.LINE_AA)
            cv2.putText(canvas, 'mm', (bar_x, max(top - 6, 12)),
                        font, scale, (90, 92, 98), thick, cv2.LINE_AA)

        elif is_cold:
            # 色带自顶向下为 t=1 → t=0，使顶部对应最大超限（深蓝），
            #   底部对应阈值（浅灰），与下方标签方向一致。
            t = np.linspace(1.0, 0.0, count)
            colours = (np.clip(cold_defect_colormap(t), 0.0, 1.0)
                       * 255).astype(np.uint8)
            canvas[top:bottom, bar_x:bar_x + bar_w, :] = colours[:, ::-1][:, None, :]

            mid_m = (limit_m + max_m) * 0.5
            for value_m, y in ((max_m, top + 4),
                               (mid_m, (top + bottom) // 2),
                               (limit_m, bottom - 2)):
                text = f'{value_m * 1000:.1f}'
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                cv2.putText(canvas, text,
                            (bar_x - 4 - tw, min(max(y, th), h - 2)),
                            font, scale, (60, 62, 68), thick, cv2.LINE_AA)
            cv2.putText(canvas, 'mm', (bar_x, max(top - 6, 12)),
                        font, scale, (90, 92, 98), thick, cv2.LINE_AA)

        else:
            t = np.linspace(1.0, 0.0, count)
            colours = (np.clip(defect_colormap(t), 0.0, 1.0)
                       * 255).astype(np.uint8)
            canvas[top:bottom, bar_x:bar_x + bar_w, :] = colours[:, ::-1][:, None, :]

            mid_m = (limit_m + max_m) * 0.5
            for value_m, y in ((max_m, top + 4),
                               (mid_m, (top + bottom) // 2),
                               (limit_m, bottom - 2)):
                text = f'{value_m * 1000:.1f}'
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                cv2.putText(canvas, text,
                            (bar_x - 4 - tw, min(max(y, th), h - 2)),
                            font, scale, (60, 62, 68), thick, cv2.LINE_AA)
            cv2.putText(canvas, 'mm', (bar_x, max(top - 6, 12)),
                        font, scale, (90, 92, 98), thick, cv2.LINE_AA)

        cv2.rectangle(canvas, (bar_x - 1, top - 1),
                      (bar_x + bar_w, bottom), (110, 115, 125), 1)
        return canvas

    # ------------------------------------------------------------------
    # 透明热力图（照片叠加用）
    # ------------------------------------------------------------------
    def render_transparent_heatmap(self, mode, points, colors, windows,
                                   plane_model, quality, pixel_size=0.01,
                                   return_metadata=False):
        """生成仅含热力、无点云底图的透明 PNG（RGBA）。"""
        spec = heatmap_spec(mode)
        method = spec.get("method", "ruler")
        metric = spec.get("metric", "flatness")

        physical_cell_m = None
        if method == "ruler" and metric == "flatness":
            prepare_result = self._prepare_ruler_flatness_grid(
                windows, spec, quality)
            if prepare_result is None:
                prepare_result = self._prepare_subgrid(
                    windows, spec, quality, metric='flatness')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        elif method == "global_plane" and metric == "flatness":
            prepare_result = self._prepare_global_plane_point_samples(
                points, windows, spec, quality, metric='flatness')
            if prepare_result is None:
                return None
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        elif method == "global_plane" and metric == "verticality":
            prepare_result = self._prepare_global_plane_point_samples(
                points, windows, spec, quality, metric='verticality')
            if prepare_result is None:
                return None
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.15
        elif method == "ruler" and metric == "verticality":
            prepare_result = self._prepare_ruler_verticality_points(
                windows, spec, quality)
            if prepare_result is None:
                prepare_result = self._prepare_subgrid(
                    windows, spec, quality, metric='verticality')
            defect_points, values, defect_colors = prepare_result
            physical_cell_m = 0.10
        else:
            defect_points, values, defect_colors = self._prepare_subgrid(
                windows, spec, quality, metric=metric)
            physical_cell_m = 0.10 if metric == "verticality" else None
        if len(defect_points) == 0:
            return None

        values_m = values / 1000.0
        profile = quality.get("profile_snapshot", {}) or {}
        limit_mm = float(profile.get(
            spec["limit_key"],
            quality.get("thresholds", {}).get(
                spec["limit_key"],
                quality.get("parameters", {}).get(spec["limit_key"], 4.0),
            ),
        ))
        limit_m = limit_mm / 1000.0

        pts_arr = np.asarray(points, dtype=float).reshape(-1, 3)
        merged_base = (pts_arr if len(pts_arr)
                       else np.asarray(defect_points, dtype=float).reshape(-1, 3))

        base_colors = np.full((len(merged_base), 3),
                              [0.80, 0.83, 0.86], dtype=float)

        excess = np.maximum(np.abs(values) - limit_mm, 0.0)
        vmax_m = (
            float(limit_m + np.max(excess) / 1000.0)
            if len(excess) and np.any(np.isfinite(excess))
            else limit_m * 1.2
        )

        # ★ 与 render 一致：显式画布范围，避免离群点稀释分辨率
        uv_range = _resolve_domain_uv_range(quality)

        raster = rasterize_facade(
            defect_points,
            np.full((len(defect_points), 3), 0.7),
            plane_model,
            values_m,
            limit_m,
            pixel_size=pixel_size,
            defect_colors=defect_colors,
            vmin=limit_m,
            vmax=vmax_m,
            max_size=_HEATMAP_MAX_SIZE,
            projection_origin=quality.get("projection_origin"),
            projection_u_axis=quality.get("projection_u_axis"),
            projection_v_axis=quality.get("projection_v_axis"),
            base_points=merged_base,
            base_colors=base_colors,
            physical_cell_m=physical_cell_m,
            min_splat_px=4,
            uv_range=uv_range,          # ★ 显式画布边界
        )

        overlay_rgba = raster["overlay_rgba"].copy()
        if (overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4):
            raise ValueError("透明热力图 raster 必须是 RGBA 四通道")
        overlay_bgra = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA)
        cropped = self._crop_to_facade(overlay_bgra, raster,
                                        fill=(0, 0, 0, 0))
        if cropped.ndim != 3 or cropped.shape[2] != 4:
            raise ValueError("透明热力图裁剪后丢失 alpha 通道")

        if not return_metadata:
            return cropped

        mask = np.asarray(raster.get("facade_mask"), dtype=bool)
        h, w = mask.shape
        col_cov = mask.sum(axis=0) / float(h)
        row_cov = mask.sum(axis=1) / float(w)
        min_ratio = 0.02
        cols = np.where(col_cov >= min_ratio)[0]
        rows = np.where(row_cov >= min_ratio)[0]
        if len(cols) == 0 or len(rows) == 0:
            cols = np.where(mask.any(axis=0))[0]
            rows = np.where(mask.any(axis=1))[0]
        x0, x1 = int(cols[0]), int(cols[-1])
        y0, y1 = int(rows[0]), int(rows[-1])

        bounds = np.asarray(raster['bounds'], dtype=float)
        size = float(raster['pixel_size'])

        u_min_prime = bounds[0] + x0 * size
        u_max_prime = bounds[0] + x1 * size
        v_max_prime = bounds[3] - y0 * size
        v_min_prime = bounds[3] - y1 * size

        return {
            'image': cropped,
            'uv_bounds': (float(u_min_prime), float(v_min_prime),
                          float(u_max_prime), float(v_max_prime)),
            'pixel_size': size,
        }

    @staticmethod
    def fit_report_image(
        image: np.ndarray, max_width: int = 360, max_height: int = 640
    ) -> np.ndarray:
        src = np.asarray(image, dtype=np.uint8)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            raise ValueError("报告图像为空")
        h, w = src.shape[:2]
        scale = min(float(max_width) / w, float(max_height) / h, 1.0)
        if scale >= 1.0:
            return src
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)