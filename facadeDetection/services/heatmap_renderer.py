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
      - overlay:    点云底图 + 缺陷热力叠加
      - heatmap_grid: 独立热力图 + 0.5 m 细分网格
      - photo:      2D 照片对齐叠加（预留接口，无照片时返回 None）
    """

    GRID_STEP_M = 0.5
    OVERLAY_ALPHA_BOOST = 1.35

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
        pixel_size: float = 0.05,
        photo_path: Optional[str] = None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Render a single heatmap triplet for the given mode."""
        spec = heatmap_spec(mode)
        defect_points, values, defect_colors = self._prepare_point_samples(
            points, windows, spec, quality)
        if len(defect_points) == 0:
            raise ValueError(f"No valid defect windows for mode {mode}")

        # Convert mm -> m for rasterize_facade
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

        # Base colors for point cloud
        base_colors = (
            np.asarray(colors, dtype=float).reshape(-1, 3)
            if colors is not None
            else np.full((len(points), 3), [0.65, 0.70, 0.78], dtype=float)
        )
        if len(base_colors) != len(points):
            base_colors = np.full((len(points), 3), [0.65, 0.70, 0.78], dtype=float)
        
        excess = np.maximum(np.abs(values) - limit_mm, 0.0)
        # Window centres carry the measurement values, while the complete
        # facade cloud supplies the projected extent and readable point-cloud
        # background.  Omitting this distinction collapses the background to
        # a few centre pixels and produces the black sparse image seen in the
        # report preview.
        raster = rasterize_facade(
            defect_points,
            np.full((len(defect_points), 3), 0.7),
            plane_model,
            values_m,
            limit_m,
            pixel_size=pixel_size,
            defect_colors=defect_colors,
            vmin=limit_m,
            vmax=float(limit_m + np.max(excess) / 1000.0) if len(excess) and np.any(np.isfinite(excess)) else limit_m * 1.2,
            max_size=2400,
            projection_origin=quality.get("projection_origin"),
            projection_u_axis=quality.get("projection_u_axis"),
            projection_v_axis=quality.get("projection_v_axis"),
            base_points=points,
            base_colors=base_colors,
        )

        overlay = self._build_overlay(raster)
        heatmap_grid = self._build_isolated_heatmap_with_grid(raster, self.GRID_STEP_M)
        # The legend is part of every exported canvas, rather than a detached
        # sidecar image.  This keeps each PDF image self-describing and makes
        # the scale identical for overlay and isolated views.
        overlay = self._embed_legend(overlay, raster)
        heatmap_grid = self._embed_legend(heatmap_grid, raster)
        photo = (
            self._build_photo_overlay(raster, photo_path)
            if photo_path and Path(photo_path).is_file()
            else None
        )

        return {"overlay": overlay, "heatmap_grid": heatmap_grid, "photo": photo}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_windows(
        windows: list[dict], spec: dict, quality: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract centers, values, and defect colors from failed windows."""
        centers_list, values_list = [], []
        for w in windows:
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
        defect_colors = defect_colormap(t)

        return centers, values, defect_colors

    @staticmethod
    def _prepare_subgrid(windows, spec, quality):
        """Expand failed windows to 5 cm physical cells before rasterisation."""
        params = quality.get("parameters", {}) or {}
        length = float(params.get("ruler_length_m", params.get("window_length_m", 2.0)))
        width = float(params.get("ruler_width_m", params.get("window_width_m", .055)))
        origin = np.asarray(quality.get("projection_origin"), dtype=float)
        u_axis = np.asarray(quality.get("projection_u_axis"), dtype=float)
        v_axis = np.asarray(quality.get("projection_v_axis"), dtype=float)
        if origin.shape != (3,) or u_axis.shape != (3,) or v_axis.shape != (3,):
            return FacadeHeatmapTripletRenderer._prepare_windows(windows, spec, quality)
        u_axis /= max(np.linalg.norm(u_axis), 1e-12)
        v_axis /= max(np.linalg.norm(v_axis), 1e-12)
        centers, values = [], []
        # A global-plane flatness window is still a flatness footprint.  Only
        # the verticality metric uses the I-ruler orientation; otherwise the
        # global flatness heatmap is rotated by 90 degrees.
        vertical = spec.get("metric") == "verticality"
        for window in windows:
            if bool(window.get(spec["pass_key"], True)):
                continue
            center = np.asarray(window.get("center_xyz"), dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                continue
            try:
                value = float(window.get(spec["value_key"], np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            try:
                angle = 90.0 if vertical else float(window.get("direction_deg", 0.0))
            except (TypeError, ValueError):
                angle = 0.0
            rad = np.deg2rad(angle)
            along = np.cos(rad) * u_axis + np.sin(rad) * v_axis
            across = -np.sin(rad) * u_axis + np.cos(rad) * v_axis
            # Physical display cells are 5 cm along the ruler and one full
            # 5.5 cm strip across it.  ``ceil(width / .05)`` would split the
            # 55 mm ruler into two cells and double its painted area.
            n_long = max(1, int(np.ceil(length / .05)))
            n_wide = 1
            for i in range(n_long):
                for j in range(n_wide):
                    a = (i + .5) * length / n_long - length / 2
                    b = (j + .5) * width / n_wide - width / 2
                    centers.append(center + a * along + b * across)
                    values.append(value)
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        values = np.asarray(values, dtype=float)
        profile = quality.get("profile_snapshot", {}) or {}
        limit = float(profile.get(
            spec["limit_key"],
            quality.get("thresholds", {}).get(
                spec["limit_key"], params.get(spec["limit_key"], 4.0)
            ),
        ))
        excess = np.maximum(np.abs(values) - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(float(np.percentile(finite, 98)) if finite.size else limit * .15, limit * .15, 1e-6)
        return centers, values, defect_colormap(np.clip(excess / scale, 0.0, 1.0))

    @staticmethod
    def _prepare_point_samples(points, windows, spec, quality):
        """Assign each failed window value to its original source points.

        The old exporter expanded one window into many artificial 5 cm cells.
        This method keeps the physical source points and only uses a window's
        covered source IDs to transfer its measured value.  Repeated coverage
        is resolved by the largest absolute defect, preserving the worst case.
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
            ids = np.asarray(window.get('covered_source_ids', []), dtype=np.int64).reshape(-1)
            for source_id in ids.tolist():
                row = id_to_row.get(int(source_id))
                if row is not None and abs(value) > point_values[row]:
                    point_values[row] = value

        selected = np.isfinite(point_values)
        if not np.any(selected):
            # Older artifacts may not retain source IDs.  A centre fallback is
            # preferable to silently producing an empty report.
            return FacadeHeatmapTripletRenderer._prepare_windows(windows, spec, quality)
        selected_values = point_values[selected]
        excess = np.maximum(np.abs(selected_values) - limit, 0.0)
        finite = excess[np.isfinite(excess)]
        scale = max(float(np.percentile(finite, 98)) if finite.size else limit * .15,
                    limit * .15, 1e-6)
        return source[selected], selected_values, defect_colormap(
            np.clip(excess / scale, 0.0, 1.0))

    # ---------- overlay ----------
    def _build_overlay(self, raster: dict) -> np.ndarray:
        """Point-cloud base + defect heatmap composite (BGR)."""
        overlay_rgba = raster["overlay_rgba"].copy()
        alpha = overlay_rgba[:, :, 3].astype(np.float32) / 255.0

        if np.any(alpha > 0):
            rgb = overlay_rgba[:, :, :3].astype(np.float32)
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0.8)
            alpha_blur = np.clip(alpha_blur, 1e-6, 1.0)

            premul = rgb * alpha[:, :, None]
            premul_blur = cv2.GaussianBlur(premul, (3, 3), 0.5)
            rgb_smooth = premul_blur / alpha_blur[:, :, None]
            overlay_rgba[:, :, :3] = np.clip(rgb_smooth, 0, 255).astype(np.uint8)
            overlay_rgba[:, :, 3] = np.clip(alpha_blur * 255, 0, 255).astype(np.uint8)

        base_rgb = raster.get("base_rgb")
        if base_rgb is None:
            h, w = overlay_rgba.shape[:2]
            base_rgb = np.full((h, w, 3), [200, 205, 215], dtype=np.uint8)
        else:
            base_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        visible = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

        base_darkened = np.clip(base_rgb.astype(np.float32) * 1.10, 0, 255)
        boosted_overlay = np.clip(
            overlay_rgba[:, :, :3].astype(np.float32) * self.OVERLAY_ALPHA_BOOST, 0, 255
        )

        composite = (
            base_darkened * (1.0 - visible[:, :, :1])
            + boosted_overlay * visible[:, :, :1]
        ).astype(np.uint8)
        return self._crop_to_facade(composite, raster, fill=(238, 240, 242))

    # ---------- isolated heatmap + grid ----------
    def _build_isolated_heatmap_with_grid(
        self, raster: dict, grid_step_m: float
    ) -> np.ndarray:
        """Standalone heatmap with 0.5 m grid lines (BGR)."""
        overlay_rgba = raster["overlay_rgba"].copy()
        h, w = overlay_rgba.shape[:2]
        pixel_size = raster["pixel_size"]
        grid_px = max(int(grid_step_m / pixel_size), 1)

        # Compute adaptive line colour based on average luminance
        avg_luma = np.mean(overlay_rgba[:, :, :3])
        if avg_luma > 180:
            line_color = (0, 0, 0, 100)  # black translucent
        else:
            line_color = (255, 255, 255, 120)  # white translucent

        # Draw grid lines
        for x in range(0, w, grid_px):
            cv2.line(overlay_rgba, (x, 0), (x, h - 1), line_color, 1)
        for y in range(0, h, grid_px):
            cv2.line(overlay_rgba, (0, y), (w - 1, y), line_color, 1)

        # Bold border
        cv2.rectangle(overlay_rgba, (0, 0), (w - 1, h - 1), line_color, 2)

        # Composite on a neutral facade background before dropping alpha.
        # Converting transparent RGBA directly to BGR turns every uncovered
        # pixel black and makes the heatmap look like a broken grid.
        background = np.full((h, w, 3), 246, dtype=np.uint8)
        alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        foreground = overlay_rgba[:, :, :3].astype(np.float32)
        composite = (
            background.astype(np.float32) * (1.0 - alpha)
            + foreground * alpha
        ).astype(np.uint8)
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        return self._crop_to_facade(bgr, raster, fill=(250, 250, 250))

    @staticmethod
    def _crop_to_facade(image: np.ndarray, raster: dict, fill=(245, 245, 245)) -> np.ndarray:
        mask = np.asarray(raster.get("facade_mask"), dtype=bool)
        if mask.shape != image.shape[:2] or not np.any(mask):
            return image
        ys, xs = np.where(mask)
        # No-point pixels are deleted rather than painted as a background.
        # Keep an alpha channel so irregular facade boundaries and internal
        # holes do not enter PDF statistics or appear as a rectangular canvas.
        if image.ndim == 3 and image.shape[2] == 4:
            result = image.copy()
        else:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        result[~mask, 3] = 0
        return result[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]

    # ---------- photo overlay (placeholder) ----------
    def _build_photo_overlay(
        self, raster: dict, photo_path: str
    ) -> Optional[np.ndarray]:
        """
        预留接口：2D 照片 + 热力对齐叠加。
        当前版本若无对齐矩阵，直接返回照片原图（或缩放到热力图尺寸）。
        """
        photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        if photo is None:
            return None
        h, w = raster["overlay_rgba"].shape[:2]
        photo_resized = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
        return photo_resized

    @staticmethod
    def _embed_legend(image: np.ndarray, raster: dict) -> np.ndarray:
        """Prepend a compact vertical colour bar to one heatmap canvas.

        The image itself remains cropped to the occupied facade mask; only the
        legend gutter is added.  The colour scale is expressed in millimetres
        and uses the same ``defect_colormap`` as point rendering.
        """
        src = np.asarray(image)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            return image
        has_alpha = src.shape[2] == 4
        bgr = src[:, :, :3]
        h, w = bgr.shape[:2]
        gutter = max(58, min(82, int(round(h * 0.12))))
        canvas = np.full((h, w + gutter, 4 if has_alpha else 3), 248,
                         dtype=np.uint8)
        canvas[:, gutter:, :3] = bgr
        if has_alpha:
            canvas[:, gutter:, 3] = src[:, :, 3]
            canvas[:, :gutter, 3] = 255

        bar_x = max(12, gutter // 2 - 9)
        bar_w = max(10, min(18, gutter // 3))
        top, bottom = max(8, int(h * .08)), min(h - 8, int(h * .82))
        count = max(bottom - top, 1)
        # The upper label is the threshold and the lower label is the maximum;
        # keep the colour ramp in the same low-to-high direction.
        t = np.linspace(0.0, 1.0, count)
        colours = (np.clip(defect_colormap(t), 0.0, 1.0) * 255).astype(np.uint8)
        # defect_colormap returns RGB, while exported arrays are BGR/BGRA.
        canvas[top:bottom, bar_x:bar_x + bar_w, :3] = colours[:, ::-1][:, None, :]
        cv2.rectangle(canvas, (bar_x, top), (bar_x + bar_w - 1, bottom - 1),
                      (70, 70, 70, 255) if has_alpha else (70, 70, 70), 1)

        limit_m = float(raster.get('vmin', 0.0))
        max_m = float(raster.get('vmax', limit_m))
        cv2.putText(canvas, f'{limit_m * 1000:.1f}', (2, top - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, .32,
                    (45, 45, 45, 255) if has_alpha else (45, 45, 45), 1)
        cv2.putText(canvas, f'{max_m * 1000:.1f}', (2, bottom + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, .32,
                    (45, 45, 45, 255) if has_alpha else (45, 45, 45), 1)
        return canvas

    # ------------------------------------------------------------------
    # Utility: scale image for report
    # ------------------------------------------------------------------
    @staticmethod
    def fit_report_image(
        image: np.ndarray, max_width: int = 360, max_height: int = 640
    ) -> np.ndarray:
        """Resize image to fit report cell while keeping aspect ratio."""
        src = np.asarray(image, dtype=np.uint8)
        if src.ndim != 3 or src.shape[0] == 0 or src.shape[1] == 0:
            raise ValueError("报告图像为空")
        h, w = src.shape[:2]
        scale = min(float(max_width) / w, float(max_height) / h, 1.0)
        if scale >= 1.0:
            return src
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)