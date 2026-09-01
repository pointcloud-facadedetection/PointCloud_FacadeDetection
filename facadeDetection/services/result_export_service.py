from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
import cv2
from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import (
    defect_excess_colors,
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
            limit_mm, scale_mm = heatmap_limit_and_scale_mm(quality, spec)
            legend_path = self._create_heatmap_legend(
                root,
                limit_mm,
                float(limit_mm + scale_mm),
                heatmap_mode)

            print(f'[PCFD] export_heatmap: done facade={facade_no} '
                  f'heatmap={heatmap_path.name if heatmap_path else None}', flush=True)

            return {
                'root': str(root),
                'mode': heatmap_mode,
                'title': heatmap_spec(heatmap_mode)['title'],
                'heatmap': str(heatmap_path) if heatmap_path else None,
                'overlay': str(root / f'facade_{int(facade_no):03d}_{heatmap_mode}_overlay.png'),
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

        limit_mm, scale_mm = heatmap_limit_and_scale_mm(quality, spec)
        limit_m = limit_mm / 1000.0
        excess_mm = np.maximum(np.abs(values) - limit_mm, 0.0)
        defect_colors = defect_excess_colors(excess_mm, scale_mm)

        base_points, _ = self._filter_base_points(pts_local, colors, plane_model, quality)
        
        if len(base_points) == 0:
            raise ValueError('过滤后立面点云为空，无法导出叠加图')

        # 纯色深色背景 - 与3D视口一致
        # 深色背景: RGB(0.12, 0.13, 0.15) - 深蓝灰色
        base_colors = np.full((len(base_points), 3), 0.13, dtype=float)
        # 给背景添加轻微噪点纹理，避免过于平坦
        noise = np.random.RandomState(42).uniform(-0.02, 0.02, base_colors.shape)
        base_colors = np.clip(base_colors + noise, 0.08, 0.18)

        # 转换为米单位传入 rasterize_facade
        values_m = values / 1000.0
        
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
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 1.0)
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

        # 缺陷层使用更高权重，确保颜色鲜明
        defect_boost = 1.15  # 缺陷层增强系数
        
        # 对base进行轻微暗化，进一步突出缺陷
        base_darkened = base_rgb.astype(np.float32) * 0.85
        
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

        return heatmap_path

    def _create_heatmap_legend(self, root, limit_mm, max_mm, mode='flatness'):
        """Create unified heatmap color legend PNG."""
        h, w = 80, 500
        legend = np.ones((h, w, 3), dtype=np.uint8) * 245

        bar_h = 28
        bar_y = 16
        bar_x_start = 60
        bar_width = w - 120
        n_segments = bar_width
        t = np.linspace(0.0, 1.0, n_segments)
        bar = defect_excess_colors(t * max(float(max_mm) - float(limit_mm), 1.0),
                                   max(float(max_mm) - float(limit_mm), 1.0))
        legend[bar_y:bar_y + bar_h, bar_x_start:bar_x_start + n_segments] = (
            np.clip(bar[:, ::-1] * 255, 0, 255).astype(np.uint8)[None, :, :]
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (60, 60, 60)
        thickness = 1

        cv2.putText(legend, "刚超限", (10, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"{limit_mm:.1f}mm", (10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        mid_x = w // 2 - 30
        cv2.putText(legend, "警告", (mid_x, bar_y + bar_h + 20), font, font_scale, color, thickness)

        cv2.putText(legend, "严重", (w - 70, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"{max_mm:.1f}mm", (w - 80, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        legend_path = Path(root) / f'{Path(root).name}_{normalize_heatmap_mode(mode)}_legend.png'
        cv2.imwrite(str(legend_path), legend)
        return legend_path