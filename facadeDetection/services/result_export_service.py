from __future__ import annotations
import json
import traceback
from pathlib import Path
import numpy as np
import cv2
import uuid
from algorithms.facade.projection import rasterize_facade


class ResultExportService:
    """Export service: only generates heatmap PNG on demand.
    No JSON/NPZ export to avoid large files and blocking UI.
    """

    def export_heatmap(self, results_dir, facade_no, points, colors, quality,
                       pixel_size=0.01):
        """
        按需生成缺陷热力图 PNG。不生成 JSON/NPZ。
        在 dialog 弹出后，点击"显示检测效果"时调用。
        """
        root = None
        try:
            if not isinstance(results_dir, (str, Path)) or str(results_dir) == '':
                print(f'[PCFD] export_heatmap: results_dir invalid, skip', flush=True)
                return None
            if not isinstance(quality, dict):
                print(f'[PCFD] export_heatmap: quality not dict, skip', flush=True)
                return None

            overall = quality.get('overall', {})
            plane_model = overall.get('plane_model')
            if plane_model is None or len(plane_model) != 4:
                print(f'[PCFD] export_heatmap: plane_model missing, skip', flush=True)
                return None

            root = Path(results_dir) / f'facade_{int(facade_no):03d}'
            root.mkdir(parents=True, exist_ok=True)

            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                print(f'[PCFD] export_heatmap: points shape invalid {pts.shape}, skip', flush=True)
                return None

            windows = quality.get('windows') or []
            if len(windows) == 0:
                print(f'[PCFD] export_heatmap: no windows, skip', flush=True)
                return None

            # Check valid windows
            n_valid = 0
            for w in windows:
                fgm = w.get('flatness_gap_mm', np.nan)
                try:
                    if np.isfinite(float(fgm)):
                        n_valid += 1
                except (TypeError, ValueError):
                    pass

            if n_valid == 0:
                print(f'[PCFD] export_heatmap: no valid windows, skip', flush=True)
                return None

            heatmap_path = self._export_window_heatmap(
                root, facade_no, pts, windows, plane_model, pixel_size, quality)
            legend_path = self._create_heatmap_legend(
                root,
                quality.get('parameters', {}).get(
                    'flatness_limit_mm',
                    quality.get('thresholds', {}).get('flatness_limit_mm', 4.0)),
                float(overall.get('flatness_max_gap_mm', 4.0)))

            print(f'[PCFD] export_heatmap: done facade={facade_no} '
                  f'heatmap={heatmap_path.name if heatmap_path else None}', flush=True)

            return {
                'root': str(root),
                'heatmap': str(heatmap_path) if heatmap_path else None,
                'legend': str(legend_path) if legend_path else None,
            }

        except Exception as e:
            err_msg = (
                f"=== export_heatmap 异常 ==="
                f"立面编号: {facade_no}"
                f"输出目录: {root}"
                f"异常类型: {type(e).__name__}"
                f"异常信息: {e}"
                f"堆栈:{traceback.format_exc()}"
            )
            print(err_msg, flush=True)
            if root is not None:
                try:
                    (root / 'export_error.log').write_text(err_msg, encoding='utf-8')
                except Exception:
                    pass
            return None

    def _export_window_heatmap(self, root, facade_no, pts_local, windows, plane_model, pixel_size, quality):
        """把窗口结果投影成 PNG。"""
        if isinstance(windows, list):
            centers_list = []
            for r in windows:
                cx = r.get('center_xyz')
                if cx is not None and len(cx) == 3:
                    try:
                        if all(np.isfinite(float(x)) for x in cx):
                            centers_list.append([float(x) for x in cx])
                            continue
                    except (TypeError, ValueError):
                        pass
                # fallback: reconstruct from uv
                uv = r.get('center_uv', (np.nan, np.nan))
                try:
                    if len(uv) >= 2 and np.isfinite(float(uv[0])) and np.isfinite(float(uv[1])):
                        origin = np.asarray(quality.get('projection_origin', [0,0,0]))
                        u_axis = np.asarray(quality.get('projection_u_axis', [1,0,0]))
                        v_axis = np.asarray(quality.get('projection_v_axis', [0,1,0]))
                        center_xyz = origin + float(uv[0]) * u_axis + float(uv[1]) * v_axis
                        centers_list.append(center_xyz.tolist())
                        continue
                except (TypeError, ValueError):
                    pass
                centers_list.append([np.nan, np.nan, np.nan])

            centers = np.asarray(centers_list, dtype=float).reshape(-1, 3)
            flat = np.asarray([r.get('flatness_gap_mm', np.nan) for r in windows], dtype=float)

            signed_flat = np.full_like(flat, np.nan)
            for i, r in enumerate(windows):
                gap = r.get('flatness_gap_mm', np.nan)
                try:
                    gap = float(gap)
                    if not np.isfinite(gap):
                        signed_flat[i] = np.nan
                        continue
                except (TypeError, ValueError):
                    signed_flat[i] = np.nan
                    continue
                dep_id = r.get('depression_source_id', -1)
                if isinstance(dep_id, (int, np.integer)) and dep_id >= 0:
                    signed_flat[i] = -gap
                else:
                    signed_flat[i] = gap

        else:
            centers = np.asarray(windows.get('center_xyz', []), dtype=float).reshape(-1, 3)
            flat = np.asarray(windows.get('flatness_gap_mm', []), dtype=float).reshape(-1)
            signed_flat = np.asarray(windows.get('flatness_signed_gap_mm', -flat), dtype=float).reshape(-1)

        if len(centers) == 0 or len(flat) == 0:
            raise ValueError('质量结果没有有效窗口，无法导出热力图')

        valid_mask = np.all(np.isfinite(centers), axis=1) & np.isfinite(flat)
        if not np.any(valid_mask):
            raise ValueError('质量结果没有有效窗口中心点，无法导出热力图')

        centers = centers[valid_mask]
        flat = flat[valid_mask]
        signed_flat = signed_flat[valid_mask]

        limit_m = float(quality.get('parameters', {}).get(
            'flatness_limit_mm', quality.get('thresholds', {}).get('flatness_limit_mm', 4.0))) / 1000.0
        values_m = signed_flat / 1000.0

        abs_values_m = np.abs(values_m)
        t = np.clip((abs_values_m - limit_m) / max(limit_m, 1e-9), 0.0, 1.0)

        colors = np.zeros((len(values_m), 3), dtype=float)
        is_recessed = values_m < 0

        protrude_mask = ~is_recessed
        colors[protrude_mask, 0] = np.clip(0.5 + 0.5 * t[protrude_mask], 0, 1)
        colors[protrude_mask, 1] = np.clip(0.8 - 0.8 * t[protrude_mask], 0, 1)
        colors[protrude_mask, 2] = np.clip(0.2 - 0.2 * t[protrude_mask], 0, 1)

        colors[is_recessed, 0] = np.clip(0.2 - 0.2 * t[is_recessed], 0, 1)
        colors[is_recessed, 1] = np.clip(0.5 + 0.3 * t[is_recessed], 0, 1)
        colors[is_recessed, 2] = np.clip(0.8 + 0.2 * t[is_recessed], 0, 1)

        base = np.full((len(centers), 3), 0.75, dtype=float)

        raster = rasterize_facade(
            centers, base, plane_model, values_m, limit_m,
            pixel_size=pixel_size, defect_values=values_m,
            defect_colors=colors, vmin=limit_m
        )

        overlay = raster['overlay_rgba'].copy()
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0

        if np.any(alpha > 0):
            rgb = overlay[:, :, :3].astype(np.float32)
            premul = rgb * alpha[:, :, None]
            premul_blur = cv2.GaussianBlur(premul, (3, 3), 1.0)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 1.0)
            alpha_blur = np.clip(alpha_blur, 1e-6, 1.0)
            rgb_smooth = premul_blur / alpha_blur[:, :, None]
            overlay[:, :, :3] = np.clip(rgb_smooth, 0, 255).astype(np.uint8)
            overlay[:, :, 3] = np.clip(alpha_blur * 255, 0, 255).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        base_rgb = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2BGR)

        visible = overlay[:, :, 3:4].astype(np.float32) / 255.0
        composite = (
            base_rgb.astype(np.float32) * (1.0 - visible[:, :, :1]) +
            overlay[:, :, :3].astype(np.float32) * visible[:, :, :1]
        ).astype(np.uint8)

        heatmap_path = Path(root) / f'facade_{int(facade_no):03d}_defect_heatmap.png'
        overlay_path = Path(root) / 'defect_overlay.png'

        if not cv2.imwrite(str(heatmap_path), overlay_bgr):
            raise RuntimeError('热力图 PNG 写入失败')
        if not cv2.imwrite(str(overlay_path), composite):
            raise RuntimeError('合成图 PNG 写入失败')

        return heatmap_path

    def _create_heatmap_legend(self, root, limit_mm, max_mm):
        """创建热力图颜色图例 PNG。"""
        h, w = 80, 500
        legend = np.ones((h, w, 3), dtype=np.uint8) * 245

        bar_h = 28
        bar_y = 16
        bar_x_start = 60
        bar_width = w - 120
        n_segments = bar_width

        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)
            if t < 0.5:
                tt = t * 2
                r = int(np.clip((0.2 - 0.2 * tt) * 255, 0, 255))
                g = int(np.clip((0.5 + 0.3 * tt) * 255, 0, 255))
                b = int(np.clip((0.8 + 0.2 * tt) * 255, 0, 255))
            else:
                tt = (t - 0.5) * 2
                r = int(np.clip((0.5 + 0.5 * tt) * 255, 0, 255))
                g = int(np.clip((0.8 - 0.8 * tt) * 255, 0, 255))
                b = int(np.clip((0.2 - 0.2 * tt) * 255, 0, 255))

            legend[bar_y:bar_y + bar_h, bar_x_start + i] = [b, g, r]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (60, 60, 60)
        thickness = 1

        cv2.putText(legend, f"凹陷", (10, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"-{max_mm:.1f}mm", (10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        mid_x = w // 2 - 30
        cv2.putText(legend, f"合格", (mid_x, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"<{limit_mm:.1f}mm", (mid_x - 10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        cv2.putText(legend, f"凸起", (w - 70, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"+{max_mm:.1f}mm", (w - 80, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        legend_path = Path(root) / 'heatmap_legend.png'
        cv2.imwrite(str(legend_path), legend)
        return legend_path