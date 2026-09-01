from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
import cv2
from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import (
    heatmap_error_colors,
    heatmap_limit_and_scale_mm,
    heatmap_spec,
    normalize_heatmap_mode,
)


class ResultExportService:
    """
    导出服务：按需生成热力图 PNG 文件。
    """

    def export_heatmap(self, results_dir, facade_no, points, colors, quality,
                       pixel_size=0.01):
        """Generate defect heatmap PNG. """
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

            heatmap_mode = normalize_heatmap_mode(quality.get('heatmap_mode'))

            n_valid = 0
            spec = heatmap_spec(heatmap_mode)
            for w in windows:
                fgm = w.get(spec['value_key'], np.nan)
                try:
                    if np.isfinite(float(fgm)):
                        n_valid += 1
                except (TypeError, ValueError):
                    pass

            if n_valid == 0:
                print(f'[PCFD] export_heatmap: no valid windows, skip', flush=True)
                return None

            # 确保栅格图像尺寸在限定范围内。  
            pixel_size = max(float(pixel_size), 0.01)
            heatmap_path, preview_path = self._export_window_heatmap(
                root, facade_no, pts, colors, windows, plane_model, pixel_size, quality)
            limit_mm, scale_mm = heatmap_limit_and_scale_mm(quality, spec)
            legend_path = self._create_heatmap_legend(
                root,
                limit_mm,
                float(scale_mm),
                heatmap_mode)

            print(f'[PCFD] export_heatmap: done facade={facade_no} '
                  f'heatmap={heatmap_path.name if heatmap_path else None}', flush=True)

            overlay_path = root / f'facade_{int(facade_no):03d}_{heatmap_mode}_overlay.png'
            return {
                'root': str(root),
                'mode': heatmap_mode,
                'title': heatmap_spec(heatmap_mode)['title'],
                'heatmap': str(heatmap_path) if heatmap_path else None,
                'overlay': str(overlay_path),
                'preview': str(preview_path) if preview_path else str(overlay_path),
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

    def _filter_base_points(self, pts_local, colors, plane_model, quality):
        """
        过滤 base_points，排除视口外的背景点和远离立面主平面的异常点
        """
        base_points = np.asarray(pts_local, dtype=float).reshape(-1, 3)
        if len(base_points) == 0:
            return base_points, colors
        
        # 获取立面投影参数
        projection = quality.get('projection') or {}
        projection_origin = quality.get('projection_origin')
        projection_u_axis = quality.get('projection_u_axis')
        projection_v_axis = quality.get('projection_v_axis')
        
        # 如果有投影参数，过滤掉投影范围外的点（视口背景）
        if (projection_origin is not None and 
            projection_u_axis is not None and 
            projection_v_axis is not None):
            
            origin = np.asarray(projection_origin, dtype=float)
            u_axis = np.asarray(projection_u_axis, dtype=float)
            v_axis = np.asarray(projection_v_axis, dtype=float)
            
            # 计算每个点在UV平面上的投影坐标
            rel = base_points - origin
            u = np.dot(rel, u_axis)
            v = np.dot(rel, v_axis)
            
            # 计算立面在UV平面的bbox
            u_min, u_max = np.percentile(u, [1, 99])
            v_min, v_max = np.percentile(v, [1, 99])
            
            # 添加小margin，过滤掉远离立面主体的点
            u_margin = (u_max - u_min) * 0.05
            v_margin = (v_max - v_min) * 0.05
            
            in_bounds = ((u >= u_min - u_margin) & (u <= u_max + u_margin) &
                        (v >= v_min - v_margin) & (v <= v_max + v_margin))
            
            base_points = base_points[in_bounds]
            if colors is not None and len(colors) == len(in_bounds):
                colors = np.asarray(colors)[in_bounds]
        
        # 额外过滤：基于到平面距离的异常值剔除
        a, b, c, d = plane_model
        norm = np.sqrt(a*a + b*b + c*c)
        distances = np.abs(np.dot(base_points, [a, b, c]) + d) / norm
        
        # 剔除距离过大的异常点（通常是背景或噪点）
        dist_threshold = np.percentile(distances, 99.5) * 1.5
        valid_dist = distances <= max(dist_threshold, 0.5)
        
        base_points = base_points[valid_dist]
        if colors is not None:
            if len(colors) == len(valid_dist):
                colors = np.asarray(colors)[valid_dist]
            else:
                colors = None
        
        return base_points, colors

    def _window_centers_and_values(self, windows, spec):
        centers_list = []
        values_list = []
        for r in windows:
            cx = r.get('center_xyz')
            if cx is None or len(cx) != 3:
                continue
            try:
                center = [float(x) for x in cx]
                if not all(np.isfinite(x) for x in center):
                    continue
                val = float(r.get(spec['value_key'], np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            centers_list.append(center)
            values_list.append(val)
        return (np.asarray(centers_list, dtype=float).reshape(-1, 3),
                np.asarray(values_list, dtype=float))

    def _values_on_facade_points(self, base_points, centers, values, quality, plane_model):
        """Assign each facade point the max-abs window value of its UV cell."""
        origin = quality.get('projection_origin')
        u_axis = quality.get('projection_u_axis')
        v_axis = quality.get('projection_v_axis')
        if origin is None or u_axis is None or v_axis is None:
            from algorithms.geometry import plane_axes
            normal = np.asarray(plane_model[:3], dtype=float)
            normal /= np.linalg.norm(normal) + 1e-12
            u_axis, v_axis = plane_axes(normal, 'vertical_facade')
            origin = np.mean(base_points, axis=0)
        origin = np.asarray(origin, dtype=float).reshape(3)
        u_axis = np.asarray(u_axis, dtype=float).reshape(3)
        v_axis = np.asarray(v_axis, dtype=float).reshape(3)
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)

        step = max(float((quality.get('parameters') or {}).get('scan_step_m')
                         or quality.get('step_size_m') or 0.05), 1e-6)
        proj = quality.get('projection') or {}
        cu = (centers - origin) @ u_axis
        cv = (centers - origin) @ v_axis
        bu = (base_points - origin) @ u_axis
        bv = (base_points - origin) @ v_axis
        u_min = float(proj.get('u_min_m', min(float(cu.min()), float(bu.min()))))
        v_min = float(proj.get('v_min_m', min(float(cv.min()), float(bv.min()))))

        pack = 1_000_003
        ck = (np.floor((cu - u_min) / step).astype(np.int64) * pack
              + np.floor((cv - v_min) / step).astype(np.int64))
        bk = (np.floor((bu - u_min) / step).astype(np.int64) * pack
              + np.floor((bv - v_min) / step).astype(np.int64))

        uniq, inv = np.unique(ck, return_inverse=True)
        cell_abs = np.full(len(uniq), -np.inf, dtype=np.float64)
        np.maximum.at(cell_abs, inv, np.abs(values))
        sorter = np.argsort(uniq)
        pos = np.searchsorted(uniq[sorter], bk)
        pos = np.clip(pos, 0, len(uniq) - 1)
        idx = sorter[pos]
        match = uniq[idx] == bk
        point_values = np.full(len(base_points), np.nan, dtype=np.float64)
        point_values[match] = cell_abs[idx[match]]
        return point_values

    def _export_window_heatmap(self, root, facade_no, pts_local, colors, windows, plane_model, pixel_size, quality):
        """Export a grey→yellow→red facade heatmap covering all measured windows."""
        mode = normalize_heatmap_mode(quality.get('heatmap_mode'))
        spec = heatmap_spec(mode)
        centers, values = self._window_centers_and_values(windows, spec)
        if len(centers) == 0 or len(values) == 0:
            raise ValueError('质量结果没有有效窗口，无法导出热力图')

        limit_mm, scale_mm = heatmap_limit_and_scale_mm(quality, spec)
        base_points, _ = self._filter_base_points(pts_local, colors, plane_model, quality)
        if len(base_points) == 0:
            raise ValueError('过滤后立面点云为空，无法导出叠加图')

        point_values = self._values_on_facade_points(
            base_points, centers, values, quality, plane_model)
        finite = np.isfinite(point_values)
        if np.any(finite):
            heat_pts = base_points[finite]
            heat_vals = point_values[finite]
        else:
            heat_pts = centers
            heat_vals = np.abs(values)

        defect_colors = heatmap_error_colors(heat_vals, limit_mm, scale_mm)
        base_colors = np.full((len(base_points), 3), 0.84, dtype=float)

        raster = rasterize_facade(
            heat_pts, np.full((len(heat_pts), 3), 0.7), plane_model,
            heat_vals, 0.0,
            pixel_size=pixel_size,
            defect_colors=defect_colors,
            vmin=0.0,
            vmax=scale_mm,
            base_points=base_points,
            base_colors=base_colors,
            projection_origin=quality.get('projection_origin'),
            projection_u_axis=quality.get('projection_u_axis'),
            projection_v_axis=quality.get('projection_v_axis'))

        overlay = raster['overlay_rgba']
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
        base_bgr = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2BGR)
        visible = overlay[:, :, 3:4].astype(np.float32) / 255.0
        composite = (
            base_bgr.astype(np.float32) * (1.0 - visible) +
            overlay_bgr.astype(np.float32) * visible
        ).astype(np.uint8)

        heatmap_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_heatmap.png'
        overlay_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_overlay.png'
        legend = self._render_heatmap_legend(
            limit_mm,
            scale_mm,
            mode=mode,
        )
        preview = self._compose_heatmap_preview(composite, legend)
        if not cv2.imwrite(str(heatmap_path), preview):
            raise RuntimeError('热力图 PNG 写入失败')
        if not cv2.imwrite(str(overlay_path), preview):
            raise RuntimeError('合成图 PNG 写入失败')

        preview_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_preview.png'
        if not cv2.imwrite(str(preview_path), preview):
            preview_path = overlay_path
        return heatmap_path, preview_path

    def _compose_heatmap_preview(self, composite, legend):
        """Place a vertical legend over the lower-right corner of the heatmap."""
        h, w = composite.shape[:2]
        max_w, max_h = 720, 1400
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 0.999:
            small = cv2.resize(
                composite,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA)
        else:
            small = composite
        base_canvas_w = max(480, small.shape[1])
        image_x = (base_canvas_w - small.shape[1]) // 2
        legend_x = image_x + small.shape[1] + 16
        canvas_w = max(
            base_canvas_w,
            legend_x + legend.shape[1] + 12,
        )
        canvas_h = max(small.shape[0], legend.shape[0] + 24)
        image_panel = np.full(
            (canvas_h, canvas_w, 3),
            245,
            dtype=np.uint8,
        )
        image_y = (canvas_h - small.shape[0]) // 2
        image_panel[
            image_y:image_y + small.shape[0],
            image_x:image_x + small.shape[1],
        ] = small
        legend_y = canvas_h - legend.shape[0] - 12
        image_panel[
            legend_y:legend_y + legend.shape[0],
            legend_x:legend_x + legend.shape[1],
        ] = legend
        return image_panel

    def _render_heatmap_legend(
        self,
        limit_mm,
        max_mm,
        width=210,
        mode='flatness',
    ):
        """Render a vertical legend without changing the colour mapping."""
        h, w = 330, max(190, min(int(width), 240))
        legend = np.ones((h, w, 3), dtype=np.uint8) * 245
        legend[[0, -1], :] = (210, 210, 210)
        legend[:, [0, -1]] = (210, 210, 210)
        bar_h = 210
        bar_w = 28
        bar_y = 52
        bar_x_start = 24
        values = np.linspace(float(max_mm), 0.0, bar_h)
        bar = heatmap_error_colors(values, limit_mm, max_mm)
        legend[bar_y:bar_y + bar_h, bar_x_start:bar_x_start + bar_w] = (
            np.clip(bar[:, ::-1] * 255, 0, 255).astype(np.uint8)[:, None, :]
        )
        tick_y = bar_y + int(round(
            bar_h * (1.0 - float(limit_mm) / max(float(max_mm), 1e-6))))
        tick_y = min(max(tick_y, bar_y), bar_y + bar_h - 1)
        legend[tick_y:tick_y + 2, bar_x_start - 4:bar_x_start + bar_w + 5] = (
            65,
            65,
            65,
        )
        return self._put_legend_texts(
            legend,
            mode=mode,
            text_x=bar_x_start + 42,
            bar_bottom=bar_y + bar_h,
            limit_y=tick_y,
            limit_mm=float(limit_mm),
            max_mm=float(max_mm),
        )

    def _put_legend_texts(
        self,
        legend,
        *,
        mode,
        text_x,
        bar_bottom,
        limit_y,
        limit_mm,
        max_mm,
    ):
        rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        try:
            from PIL import Image, ImageDraw, ImageFont
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            title_font = None
            font_path = None
            for candidate in (
                r'C:\Windows\Fonts\msyh.ttc',
                r'C:\Windows\Fonts\msyh.ttf',
                r'C:\Windows\Fonts\simhei.ttf',
                r'C:\Windows\Fonts\simsun.ttc',
            ):
                try:
                    title_font = ImageFont.truetype(candidate, 17)
                    font_path = candidate
                    break
                except OSError:
                    continue
            if title_font is None:
                raise RuntimeError('no CJK font')
            label_font = ImageFont.truetype(font_path, 14)
            small_font = ImageFont.truetype(font_path, 13)
            metric = '垂直度偏差' if mode == 'verticality' else '平整度偏差'
            draw.text(
                (12, 12),
                metric,
                fill=(45, 55, 72),
                font=title_font,
            )
            draw.text(
                (text_x, 45),
                f'显示上限 {max_mm:.1f} mm',
                fill=(75, 85, 99),
                font=small_font,
                anchor='lm',
            )
            draw.text(
                (text_x, limit_y),
                f'标准限值 {limit_mm:.1f} mm',
                fill=(55, 65, 81),
                font=label_font,
                anchor='lm',
            )
            draw.text(
                (text_x, bar_bottom),
                '0 mm',
                fill=(75, 85, 99),
                font=small_font,
                anchor='lm',
            )
            draw.text(
                (12, 280),
                f'合格：0–{limit_mm:.1f} mm',
                fill=(78, 86, 98),
                font=small_font,
            )
            draw.text(
                (12, 304),
                f'超限：>{limit_mm:.1f} mm',
                fill=(184, 50, 42),
                font=small_font,
            )
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except Exception:
            font = cv2.FONT_HERSHEY_SIMPLEX
            metric = 'Verticality deviation' if mode == 'verticality' else 'Flatness deviation'
            cv2.putText(
                legend,
                f'{metric} (mm)',
                (12, 27),
                font,
                0.48,
                (55, 55, 55),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                legend,
                f'max {max_mm:.1f}',
                (text_x, 57),
                font,
                0.42,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                legend,
                f'limit {limit_mm:.1f}',
                (text_x, limit_y + 5),
                font,
                0.42,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                legend,
                '0',
                (text_x, bar_bottom + 5),
                font,
                0.42,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                legend,
                f'PASS 0-{limit_mm:.1f} mm',
                (12, 292),
                font,
                0.43,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                legend,
                f'EXCEEDS >{limit_mm:.1f} mm',
                (12, 316),
                font,
                0.43,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            return legend

    def _create_heatmap_legend(self, root, limit_mm, max_mm, mode='flatness'):
        legend = self._render_heatmap_legend(
            limit_mm,
            max_mm,
            mode=mode,
        )
        legend_path = Path(root) / f'{Path(root).name}_{normalize_heatmap_mode(mode)}_legend.png'
        cv2.imwrite(str(legend_path), legend)
        return legend_path