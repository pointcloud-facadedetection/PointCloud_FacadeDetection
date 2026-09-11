"""Facade heatmap triplet renderer: overlay, isolated heatmap with grid, photo overlay."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import heatmap_spec, normalize_heatmap_mode, defect_colormap


class FacadeHeatmapTripletRenderer:
    """
    每组 3 张子图：
      - overlay:      点云底图 + 缺陷热力叠加
      - heatmap_grid: 独立热力图 + 1.0 m 细分网格（无点云）
      - photo:        2D 照片对齐叠加（预留接口，无照片时返回 None）
    """

    GRID_STEP_M = 1.0
    # 浅蓝灰底色，用于热力叠加底图回退
    _OVERLAY_BASE_COLOR = np.array([230, 232, 235], dtype=np.uint8)
    _OVERLAY_FILL_COLOR = (238, 240, 243)
    # 独立热力图背景：近纯白
    _HEATMAP_BG_COLOR = 252

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

        defect_points, values, defect_colors = self._prepare_subgrid(
            windows, spec, quality)
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

        # 点云底图颜色：提浅到近浅灰白，热力色块在浅底上更醒目
        base_colors = np.full((len(points), 3), [0.88, 0.90, 0.92], dtype=float)

        excess = np.maximum(np.abs(values) - limit_mm, 0.0)
        vmax_m = (
            float(limit_m + np.max(excess) / 1000.0)
            if len(excess) and np.any(np.isfinite(excess))
            else limit_m * 1.2
        )

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
            max_size=2400,
            projection_origin=quality.get("projection_origin"),
            projection_u_axis=quality.get("projection_u_axis"),
            projection_v_axis=quality.get("projection_v_axis"),
            base_points=points,
            base_colors=base_colors,
        )

        # 两张图共用 raster["overlay_rgba"] 的原始 RGB 前景
        overlay = self._build_overlay(raster)
        heatmap_grid = self._build_isolated_heatmap_with_grid(raster, self.GRID_STEP_M)

        overlay = self._embed_legend(overlay, raster)
        heatmap_grid = self._embed_legend(heatmap_grid, raster)

        photo = (
            self._build_photo_overlay(raster, photo_path)
            if photo_path and Path(photo_path).is_file()
            else None
        )

        return {"overlay": overlay, "heatmap_grid": heatmap_grid, "photo": photo}

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
    def _prepare_subgrid(windows, spec, quality):
        """把失败窗口沿靠尺方向展开为 5cm 物理块（与统计同源）

        过滤规则（与统计口径一致）：
          - pass_key 显式为 False                      → 保留
          - pass_key 缺失/为 True 但 |value| > limit   → 保留（兜底）
          - 其余                                       → 丢弃
        """
        params = quality.get("parameters", {}) or {}
        origin = np.asarray(quality.get("projection_origin"), dtype=float)
        u_axis = np.asarray(quality.get("projection_u_axis"), dtype=float)
        v_axis = np.asarray(quality.get("projection_v_axis"), dtype=float)

        # 投影轴/原点缺失或非有限 → 回退窗口中心散点，绝不返回空图
        if (origin.shape != (3,) or not np.all(np.isfinite(origin))
                or u_axis.shape != (3,) or not np.all(np.isfinite(u_axis))
                or v_axis.shape != (3,) or not np.all(np.isfinite(v_axis))):
            return FacadeHeatmapTripletRenderer._prepare_windows(
                windows, spec, quality)

        u_axis = u_axis / max(np.linalg.norm(u_axis), 1e-12)
        v_axis = v_axis / max(np.linalg.norm(v_axis), 1e-12)

        # 限值字段与统计数据保持完全一致
        profile = quality.get("profile_snapshot", {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            (quality.get("thresholds") or {}).get(
                spec["limit_key"],
                params.get(spec["limit_key"], 4.0),
            ),
        ))

        # 靠尺物理尺寸
        length = float(params.get("ruler_length_m",
                                  params.get("window_length_m", 2.0)))
        width = float(params.get("ruler_width_m",
                                 params.get("window_width_m", 0.055)))
        # 全局平面法无靠尺时退化为 0.5m×0.5m 的块
        if length <= 0 or width <= 0:
            length, width = 0.5, 0.5

        # 全局平面与垂直度都严格使用 I 字：长轴沿立面竖向 v_axis。
        # 靠尺平整度才使用算法输出的 direction_deg（米字方向）。
        global_plane = spec.get("method") == "global_plane"
        vertical = spec.get("metric") == "verticality"
        centers, values = [], []

        for window in windows or []:
            # ---- 值过滤：只保留真正超限的窗口 ----
            try:
                value = float(window.get(spec["value_key"], np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue

            pass_value = window.get(spec["pass_key"])
            explicit_fail = (pass_value is False)
            value_exceeds = abs(value) > limit
            if not (explicit_fail or value_exceeds):
                continue

            # ---- 中心点校验 ----
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

            n_long = max(1, int(np.ceil(length / 0.05)))
            n_wide = 1
            for i in range(n_long):
                for j in range(n_wide):
                    a = (i + 0.5) * length / n_long - length / 2
                    b = (j + 0.5) * width / n_wide - width / 2
                    centers.append(center + a * along + b * across)
                    values.append(value)

        if not centers:
            # 没有可用失败窗口 → 回退中心散点
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
        return centers, values, defect_colormap(np.clip(excess / scale, 0.0, 1.0))

    @staticmethod
    def _prepare_point_samples(points, windows, spec, quality):
        """保留为可选备选：把窗口值贴回原始点云（依赖 covered_source_ids）。
        """
        source = np.asarray(points, dtype=float).reshape(-1, 3)
        raw_ids = np.asarray(quality.get('__global_indices', []), dtype=np.int64)
        if len(raw_ids) != len(source):
            raw_ids = np.arange(len(source), dtype=np.int64)
        id_to_row = {int(value): i for i, value in enumerate(raw_ids.tolist())}
        point_values = np.full(len(source), -np.inf, dtype=float)
        profile = quality.get('profile_snapshot', {}) or {}
        limit = float(profile.get(spec['limit_key'],
            (quality.get('thresholds') or {}).get(spec['limit_key'],
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
        """点云底图 + 缺陷热力叠加（BGR）。 """
        overlay_rgba = np.asarray(raster["overlay_rgba"])
        h, w = overlay_rgba.shape[:2]

        base_rgb = raster.get("base_rgb")
        if base_rgb is None:
            base_rgb = np.full((h, w, 3), self._OVERLAY_BASE_COLOR.tolist(),
                               dtype=np.uint8)
        base_rgb = np.asarray(base_rgb, dtype=np.float32)

        overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
        visible = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

        composite_rgb = (
            base_rgb * (1.0 - visible[:, :, :1])
            + overlay_rgb * visible[:, :, :1]
        )
        composite_bgr = cv2.cvtColor(
            np.clip(composite_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        return self._crop_to_facade(composite_bgr, raster,
                                    fill=self._OVERLAY_FILL_COLOR)

    # ------------------------------------------------------------------
    # 独立热力图 + 网格（不含点云）
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness=1,
                          dash_len=6, gap_len=6):
        """绘制虚线：实线段 + 空隙。"""
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
        """独立热力图 + 1m 浅灰虚线网格（BGR）。
        """
        overlay_rgba = raster["overlay_rgba"].copy()
        h, w = overlay_rgba.shape[:2]
        pixel_size = raster["pixel_size"]
        grid_px = max(int(grid_step_m / pixel_size), 1)

        # ---- 融合离散 splat → 连续色域 ----
        alpha_u8 = overlay_rgba[:, :, 3]
        if np.any(alpha_u8 > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            alpha_closed = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
            rgb = overlay_rgba[:, :, :3]
            rgb_dil = cv2.dilate(rgb, kernel)
            need_fill = (alpha_u8 == 0) & (alpha_closed > 0)
            rgb[need_fill] = rgb_dil[need_fill]
            overlay_rgba[:, :, 3] = alpha_closed

        # 热力是前景层：避免低权重 splat 被点云底色视觉淹没。
        # 只提升已有缺陷像素的 alpha，不给无缺陷区域染色。
        overlay_rgba[:, :, 3] = np.where(
            overlay_rgba[:, :, 3] > 0,
            np.maximum(overlay_rgba[:, :, 3], 210),
            0,
        ).astype(np.uint8)

        # ---- 网格线（内部虚线，不与边框重叠） ----
        line_color = (150, 160, 175, 170)      # 浅蓝灰
        border_color = (95, 105, 120, 220)     # 边框略深

        for x in range(grid_px, w, grid_px):
            self._draw_dashed_line(overlay_rgba, (x, 0), (x, h - 1),
                                   line_color, 1, dash_len=6, gap_len=6)
        for y in range(grid_px, h, grid_px):
            self._draw_dashed_line(overlay_rgba, (0, y), (w - 1, y),
                                   line_color, 1, dash_len=6, gap_len=6)

        cv2.rectangle(overlay_rgba, (0, 0), (w - 1, h - 1), border_color, 1)

        # ---- 近纯白底 + 热力合成（无点云底图） ----
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
        """裁剪到立面有效区，并统一为 BGRA（4 通道）。"""
        mask = np.asarray(raster.get("facade_mask"), dtype=bool)
        if mask.shape != image.shape[:2] or not np.any(mask):
            return image
        ys, xs = np.where(mask)
        if image.ndim == 3 and image.shape[2] == 4:
            result = image.copy()
        else:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[~mask, 3] = 0
        return result[int(ys.min()):int(ys.max()) + 1,
                      int(xs.min()):int(xs.max()) + 1]

    def _build_photo_overlay(
        self, raster: dict, photo_path: str
    ) -> Optional[np.ndarray]:
        """预留接口：2D 照片 + 热力对齐叠加。"""
        photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        if photo is None:
            return None
        h, w = raster["overlay_rgba"].shape[:2]
        return cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    @staticmethod
    def _embed_legend(image: np.ndarray, raster: dict) -> np.ndarray:
        """在图左侧嵌入紧凑色标（4 通道输出）。"""
        src = np.asarray(image)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            return image
        has_alpha = src.shape[2] == 4
        bgr = src[:, :, :3]
        h, w = bgr.shape[:2]

        gutter = max(24, min(38, int(round(h * 0.045))))
        canvas = np.full((h, w + gutter, 4 if has_alpha else 3), 248,
                         dtype=np.uint8)
        canvas[:, gutter:, :3] = bgr
        if has_alpha:
            canvas[:, gutter:, 3] = src[:, :, 3]
            canvas[:, :gutter, 3] = 255

        bar_x = max(6, gutter // 2 - 5)
        bar_w = max(6, min(9, gutter // 3))
        top, bottom = max(12, int(h * 0.18)), min(h - 12, int(h * 0.78))
        count = max(bottom - top, 1)
        t = np.linspace(0.0, 1.0, count)
        colours = (np.clip(defect_colormap(t), 0.0, 1.0) * 255).astype(np.uint8)
        canvas[top:bottom, bar_x:bar_x + bar_w, :3] = colours[:, ::-1][:, None, :]
        cv2.rectangle(canvas, (bar_x, top), (bar_x + bar_w - 1, bottom - 1),
                      (70, 70, 70, 255) if has_alpha else (70, 70, 70), 1)

        limit_m = float(raster.get('vmin', 0.0))
        max_m = float(raster.get('vmax', limit_m))
        cv2.putText(canvas, f'{limit_m * 1000:.1f}', (2, top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .26,
                    (45, 45, 45, 255) if has_alpha else (45, 45, 45), 1)
        cv2.putText(canvas, f'{max_m * 1000:.1f}', (2, bottom + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, .26,
                    (45, 45, 45, 255) if has_alpha else (45, 45, 45), 1)
        return canvas

    # ------------------------------------------------------------------
    # Utility: scale image for report
    # ------------------------------------------------------------------
    @staticmethod
    def fit_report_image(
        image: np.ndarray, max_width: int = 360, max_height: int = 640
    ) -> np.ndarray:
        """按最长边等比缩放到报告单元格。"""
        src = np.asarray(image, dtype=np.uint8)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            raise ValueError("报告图像为空")
        h, w = src.shape[:2]
        scale = min(float(max_width) / w, float(max_height) / h, 1.0)
        if scale >= 1.0:
            return src
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)