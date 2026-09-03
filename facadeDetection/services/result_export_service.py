from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
import cv2
from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import heatmap_spec, normalize_heatmap_mode


class ResultExportService:
    """
    导出服务：按需生成热力图 PNG 文件。
    """

    def export_heatmap(self, results_dir, facade_no, points, colors, quality,
                       pixel_size=0.05):
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

            # 只有同时具有实际测量值且质量检测结果为失败的窗口才可绘制。
            n_valid = 0
            spec = heatmap_spec(heatmap_mode)
            for w in windows:
                fgm = w.get(spec['value_key'], np.nan)
                try:
                    if (not bool(w.get(spec['pass_key'], True)) and
                            np.isfinite(float(fgm))):
                        n_valid += 1
                except (TypeError, ValueError):
                    pass

            if n_valid == 0:
                print(f'[PCFD] export_heatmap: no valid windows, skip', flush=True)
                return None

            # 确保栅格图像尺寸在限定范围内。  
            pixel_size = max(float(pixel_size), 0.01)
            heatmap_path = self._export_window_heatmap(
                root, facade_no, pts, colors, windows, plane_model, pixel_size, quality)
            max_value = overall.get(spec['value_key'])
            if max_value is None:
                limit_for_legend = float(quality.get('parameters', {}).get(
                    spec['limit_key'], quality.get('thresholds', {}).get(
                        spec['limit_key'], 4.0)))
                max_value = max((float(w.get(spec['value_key'])) for w in windows
                                 if np.isfinite(float(w.get(spec['value_key'], np.nan)))),
                                default=limit_for_legend)
            legend_path = self._create_heatmap_legend(
                root,
                quality.get('parameters', {}).get(
                    heatmap_spec(heatmap_mode)['limit_key'],
                    quality.get('thresholds', {}).get(heatmap_spec(heatmap_mode)['limit_key'], 4.0)),
                float(max_value),
                heatmap_mode)

            print(f'[PCFD] export_heatmap: done facade={facade_no} '
                  f'heatmap={heatmap_path.name if heatmap_path else None}', flush=True)

            return {
                'root': str(root),
                'mode': heatmap_mode,
                'title': heatmap_spec(heatmap_mode)['title'],
                'heatmap': str(heatmap_path) if heatmap_path else None,
                'overlay': str(root / f'facade_{int(facade_no):03d}_{heatmap_mode}_overlay.png'),
                'report': str(root / f'facade_{int(facade_no):03d}_{heatmap_mode}_report.png'),
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

    def _export_window_heatmap(self, root, facade_no, pts_local, colors, windows, plane_model, pixel_size, quality):
        """将窗口结果导出为热力图 PNG 文件，并采用统一的缺陷配色方案。"""
        mode = normalize_heatmap_mode(quality.get('heatmap_mode'))
        spec = heatmap_spec(mode)
        
        # 提取中心和缺陷值
        centers_list = []
        values_list = []
        
        for r in windows:
            pass_key = spec['pass_key']
            if bool(r.get(pass_key, True)):
                continue
            cx = r.get('center_xyz')
            # 只有当窗口的几何信息和尺寸均有效时，该窗口才可绘制。
            if cx is not None and len(cx) == 3:
                try:
                    if all(np.isfinite(float(x)) for x in cx):
                        center = [float(x) for x in cx]
                    else:
                        continue
                except (TypeError, ValueError):
                    continue
            else:
                continue

            val = r.get(spec['value_key'], np.nan)
            
            try:
                val = float(val)
                if not np.isfinite(val):
                    continue
            except (TypeError, ValueError):
                continue
            centers_list.append(center)
            values_list.append(val)

        centers = np.asarray(centers_list, dtype=float).reshape(-1, 3)
        values = np.asarray(values_list, dtype=float)

        if len(centers) == 0 or len(values) == 0 or len(centers) != len(values):
            raise ValueError('质量结果没有有效窗口，无法导出热力图')

        # Get limit
        limit_key = spec['limit_key']
        limit_mm = float(quality.get('parameters', {}).get(
            limit_key, quality.get('thresholds', {}).get(limit_key, 4.0)))
        
        # 统一单位 - 与视口保持一致
        values_mm = values  # 保持 mm 单位用于颜色计算
        limit_m = limit_mm / 1000.0  # 阈值转为 m
        
        # 计算超标量
        excess_mm = np.maximum(np.abs(values_mm) - limit_mm, 0.0)
        
        # ============================================================
        # 优化1: 颜色区分度 - 使用自适应分位数归一化，避免异常值压缩动态范围
        finite_excess = excess_mm[np.isfinite(excess_mm)]
        if len(finite_excess) > 0 and np.any(finite_excess > 0):
            # 98%分位数作为归一化上限，避免单个极端值拉低整体对比度
            p98_excess = float(np.percentile(finite_excess, 98))
            scale_mm = max(p98_excess, limit_mm * 0.15)  # 至少保留15%的limit作为动态范围
        else:
            scale_mm = limit_mm * 0.15
        
        t = np.clip(excess_mm / scale_mm, 0.0, 1.0)


        # 增强颜色区分度 - 5节点色标（青→绿→黄→橙→红）
        defect_colors = np.zeros((len(values_mm), 3), dtype=float)
        
        # 节点定义: t=0.0青(0,0.7,1.0) -> t=0.25绿(0.2,0.9,0.2) -> t=0.5黄(1,1,0) -> t=0.75橙(1,0.5,0) -> t=1.0红(1,0,0)
        # 分段线性插值
        
        # 区间1: 0.0 ~ 0.25 (青 -> 绿)
        mask1 = t <= 0.25
        tt1 = t[mask1] / 0.25
        defect_colors[mask1, 0] = 0.0 + 0.2 * tt1       # R: 0 -> 0.2
        defect_colors[mask1, 1] = 0.7 + 0.2 * tt1       # G: 0.7 -> 0.9
        defect_colors[mask1, 2] = 1.0 - 0.8 * tt1       # B: 1.0 -> 0.2
        
        # 区间2: 0.25 ~ 0.5 (绿 -> 黄)
        mask2 = (t > 0.25) & (t <= 0.5)
        tt2 = (t[mask2] - 0.25) / 0.25
        defect_colors[mask2, 0] = 0.2 + 0.8 * tt2       # R: 0.2 -> 1.0
        defect_colors[mask2, 1] = 0.9 + 0.1 * tt2       # G: 0.9 -> 1.0
        defect_colors[mask2, 2] = 0.2 - 0.2 * tt2       # B: 0.2 -> 0.0
        
        # 区间3: 0.5 ~ 0.75 (黄 -> 橙)
        mask3 = (t > 0.5) & (t <= 0.75)
        tt3 = (t[mask3] - 0.5) / 0.25
        defect_colors[mask3, 0] = 1.0                   # R: 1.0
        defect_colors[mask3, 1] = 1.0 - 0.5 * tt3       # G: 1.0 -> 0.5
        defect_colors[mask3, 2] = 0.0                   # B: 0.0
        
        # 区间4: 0.75 ~ 1.0 (橙 -> 红)
        mask4 = t > 0.75
        tt4 = (t[mask4] - 0.75) / 0.25
        defect_colors[mask4, 0] = 1.0                   # R: 1.0
        defect_colors[mask4, 1] = 0.5 - 0.5 * tt4       # G: 0.5 -> 0.0
        defect_colors[mask4, 2] = 0.0                   # B: 0.0

        base_points, _ = self._filter_base_points(pts_local, colors, plane_model, quality)
        
        if len(base_points) == 0:
            raise ValueError('过滤后立面点云为空，无法导出叠加图')

        base_colors = np.full((len(base_points), 3), [0.65, 0.70, 0.78], dtype=float)

        # 转换为米单位传入 rasterize_facade
        values_m = values_mm / 1000.0
        
        projection = quality.get('projection') or {}
        projection_origin = quality.get('projection_origin')
        projection_u_axis = quality.get('projection_u_axis')
        projection_v_axis = quality.get('projection_v_axis')

        # 传入与视口一致的参数
        global_vmax_m = (limit_mm + scale_mm) / 1000.0
        
        raster = rasterize_facade(
            centers, np.full((len(centers), 3), 0.7), plane_model, 
            values_m, limit_m,
            pixel_size=pixel_size, 
            defect_colors=defect_colors, 
            vmin=limit_m,
            vmax=global_vmax_m,  # 传入全局vmax确保内部映射一致
            max_size=2400,
            base_points=base_points, 
            base_colors=base_colors,
            projection_origin=projection_origin,
            projection_u_axis=projection_u_axis,
            projection_v_axis=projection_v_axis)

        overlay = raster['overlay_rgba'].copy()
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0

        if np.any(alpha > 0):
            rgb = overlay[:, :, :3].astype(np.float32)

            # 优化模糊处理 - 保持缺陷边缘清晰
            # A small dilation keeps sparse defect windows legible in print.
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=1)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0.8)
            alpha_blur = np.clip(alpha_blur, 1e-6, 1.0)
            
            # RGB使用更小的模糊核，保持边缘清晰
            premul = rgb * alpha[:, :, None]
            premul_blur = cv2.GaussianBlur(premul, (3, 3), 0.5)
            
            rgb_smooth = premul_blur / alpha_blur[:, :, None]
            overlay[:, :, :3] = np.clip(rgb_smooth, 0, 255).astype(np.uint8)
            overlay[:, :, 3] = np.clip(alpha_blur * 255, 0, 255).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        base_rgb = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2BGR)

        visible = overlay[:, :, 3:4].astype(np.float32) / 255.0

        # Keep the neutral point-cloud layer subdued while preserving detail.
        defect_boost = 1.35
        base_darkened = np.clip(base_rgb.astype(np.float32) * 1.10, 0, 255)
        
        boosted_overlay = overlay[:, :, :3].astype(np.float32) * defect_boost
        boosted_overlay = np.clip(boosted_overlay, 0, 255)
        
        composite = (
            base_darkened * (1.0 - visible[:, :, :1]) +
            boosted_overlay * visible[:, :, :1]
        ).astype(np.uint8)

        heatmap_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_heatmap.png'
        overlay_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_overlay.png'

        if not cv2.imwrite(str(heatmap_path), overlay_bgr):
            raise RuntimeError('热力图 PNG 写入失败')
        if not cv2.imwrite(str(overlay_path), composite):
            raise RuntimeError('合成图 PNG 写入失败')


        # 优化2: 缩小报告图尺寸，适配A4半栏并排显示
        report_path = Path(root) / f'facade_{int(facade_no):03d}_{mode}_report.png'
        # 单图目标宽度360px，双图并排适配A4可用宽度（约182mm ≈ 690px @ 96dpi）
        report_image = self._fit_report_image(composite, max_width=360, max_height=640)
        if not cv2.imwrite(str(report_path), report_image):
            raise RuntimeError('PDF 报告热力图写入失败')

        return heatmap_path

    @staticmethod
    def _fit_report_image(image, max_width=360, max_height=640):
        """Create a bounded, letterboxed image for the fixed PDF image box.

        """
        source = np.asarray(image, dtype=np.uint8)
        if source.ndim != 3 or source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError('报告图像为空')
        h, w = source.shape[:2]
        scale = min(float(max_width) / w, float(max_height) / h, 1.0)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(source, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((max_height, max_width, 3), [245, 247, 250], dtype=np.uint8)
        x, y = (max_width - nw) // 2, (max_height - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas

    def _create_heatmap_legend(self, root, limit_mm, max_mm, mode='flatness'):
        """Create unified heatmap color legend PNG."""
        h, w = 80, 500
        legend = np.ones((h, w, 3), dtype=np.uint8) * 245

        bar_h = 28
        bar_y = 16
        bar_x_start = 60
        bar_width = w - 120
        n_segments = bar_width

        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)


            # 图例配色与热力图一致 - 5节点色标（青→绿→黄→橙→红）
            if t <= 0.25:
                tt = t / 0.25
                r = int(np.clip((0.0 + 0.2 * tt) * 255, 0, 255))
                g = int(np.clip((0.7 + 0.2 * tt) * 255, 0, 255))
                b = int(np.clip((1.0 - 0.8 * tt) * 255, 0, 255))
            elif t <= 0.5:
                tt = (t - 0.25) / 0.25
                r = int(np.clip((0.2 + 0.8 * tt) * 255, 0, 255))
                g = int(np.clip((0.9 + 0.1 * tt) * 255, 0, 255))
                b = int(np.clip((0.2 - 0.2 * tt) * 255, 0, 255))
            elif t <= 0.75:
                tt = (t - 0.5) / 0.25
                r = 255
                g = int(np.clip((1.0 - 0.5 * tt) * 255, 0, 255))
                b = 0
            else:
                tt = (t - 0.75) / 0.25
                r = 255
                g = int(np.clip((0.5 - 0.5 * tt) * 255, 0, 255))
                b = 0

            legend[bar_y:bar_y + bar_h, bar_x_start + i] = [b, g, r]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (60, 60, 60)
        thickness = 1

        cv2.putText(legend, "合格", (10, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"<{limit_mm:.1f}mm", (10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        mid_x = w // 2 - 30
        cv2.putText(legend, "警告", (mid_x, bar_y + bar_h + 20), font, font_scale, color, thickness)

        cv2.putText(legend, "严重", (w - 70, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"{max_mm:.1f}mm", (w - 80, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        legend_path = Path(root) / f'{Path(root).name}_{normalize_heatmap_mode(mode)}_legend.png'
        cv2.imwrite(str(legend_path), legend)
        return legend_path