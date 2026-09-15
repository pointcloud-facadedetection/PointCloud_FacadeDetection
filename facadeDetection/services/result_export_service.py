from __future__ import annotations
import traceback
from pathlib import Path
import numpy as np
import cv2
from algorithms.facade.projection import rasterize_facade
from services.heatmap_spec import heatmap_spec, normalize_heatmap_mode, defect_colormap, excess_uniform
from services.heatmap_renderer import FacadeHeatmapTripletRenderer


class ResultExportService:
    """
    导出服务：按需生成热力图 PNG 文件与 PDF 报告。
    重构后输出 4 组 triplet（2 套算法 × 2 指标），每组 3 张子图。
    """

    # 仅导出报告实际使用的四类热力图；point/area 双轨已废弃。
    _DISPLAY_MODES = [
        'ruler_flatness_area',
        'ruler_verticality_area',
        'global_plane_flatness_area',
        'global_plane_verticality_area',
    ]

    def __init__(self):
        self._renderer = FacadeHeatmapTripletRenderer()

    @staticmethod
    def _safe_imwrite(path: Path, img: np.ndarray) -> bool:
        """cv2.imwrite 在 Windows 上对非 ASCII / Unicode 路径会静默失败，
        使用 imencode + write_bytes 绕过 OpenCV 的窄字符文件路径限制。
        """
        if img is None or img.size == 0:
            return False
        suffix = path.suffix.lower()
        ext = '.png' if suffix == '.png' else '.jpg'
        success, encoded = cv2.imencode(ext, img)
        if not success:
            return False
        path.write_bytes(encoded.tobytes())
        return True

    def export_all_heatmaps(self, results_dir, facade_no, points, colors, quality):
        """
        导出全部 4 组展示模式的热力图 triplet PNG。
        返回字典，键为模式名，值为包含 overlay/heatmap_grid/photo 路径的字典。

        数据源契约（唯一）：
            quality['quality_comparison']['methods'][method][metric]['windows']
        上游未构造 quality_comparison 时直接返回空，不在此层做兼容。
        """
        root = None
        try:
            if not isinstance(results_dir, (str, Path)) or str(results_dir) == '':
                print('[PCFD] export_all_heatmaps: results_dir invalid, skip', flush=True)
                return {}

            if not isinstance(quality, dict):
                print('[PCFD] export_all_heatmaps: quality not dict, skip', flush=True)
                return {}

            root = Path(results_dir) / f'facade_{int(facade_no):03d}'
            root.mkdir(parents=True, exist_ok=True)

            overall = quality.get('overall', {})
            plane_model = overall.get('plane_model')
            if plane_model is None or len(plane_model) != 4:
                plane_model = quality.get('plane_model')
            if plane_model is None or len(plane_model) != 4:
                print('[PCFD] export_all_heatmaps: plane_model missing, skip', flush=True)
                return {}

            # ============================================================
            # 唯一数据源：quality_comparison.methods
            # ============================================================
            comparison = quality.get('quality_comparison', {}) or {}
            methods_data = comparison.get('methods', {}) or {}

            if not methods_data:
                print(
                    '[PCFD] export_all_heatmaps: quality_comparison.methods missing, '
                    'skip all heatmap export. '
                    'Ensure upstream builds quality_comparison before export.',
                    flush=True,
                )
                return {}

            print(f'[PCFD] export_all_heatmaps: methods={list(methods_data.keys())}',
                  flush=True)

            exported = {}
            for mode in self._DISPLAY_MODES:
                spec = heatmap_spec(mode)
                method = spec.get('method', 'ruler')
                metric = spec.get('metric', 'flatness')

                method_dict = methods_data.get(method, {})
                if not method_dict:
                    print(f'[PCFD] export_all_heatmaps: method {method} missing, '
                          f'skip {mode}', flush=True)
                    continue

                metric_data = method_dict.get(metric, {})
                windows = metric_data.get('windows', []) or []
                if not windows:
                    print(f'[PCFD] export_all_heatmaps: skip {mode}, no windows',
                          flush=True)
                    continue

                # ---- 构造 method/metric 隔离的渲染快照 ----
                method_parameters = method_dict.get('parameters')
                if not isinstance(method_parameters, dict):
                    method_parameters = {}
                metric_rates = metric_data.get('rates') or {}
                method_thresholds = {
                    key: quality.get('thresholds', {}).get(key)
                    for key in ('flatness_limit_mm', 'verticality_limit_mm')
                    if quality.get('thresholds', {}).get(key) is not None
                }
                from services.dal.results_repo import ResultsRepo
                ResultsRepo.ensure_global_indices(quality)
                temp_quality = {
                    'windows': windows,
                    'heatmap_mode': mode,
                    'overall': method_dict.get('overall', {}) if isinstance(
                        method_dict.get('overall', {}), dict) else {},
                    'thresholds': method_thresholds,
                    'parameters': method_parameters,
                    'rates': metric_rates,
                    'profile_snapshot': quality.get('profile_snapshot', {}),
                    'defect_samples': method_dict.get('defect_samples', {}),
                    '__global_indices': quality.get('__global_indices', []),
                    '__defect_index_space': quality.get(
                        '__defect_index_space', 'raw_global_rows'),
                    # method 优先，缺失才回退顶层（方法级隔离，与数据源回退无关）
                    'projection_origin': method_dict.get(
                        'projection_origin', quality.get('projection_origin')),
                    'projection_u_axis': method_dict.get(
                        'projection_u_axis', quality.get('projection_u_axis')),
                    'projection_v_axis': method_dict.get(
                        'projection_v_axis', quality.get('projection_v_axis')),
                }

                try:
                    triplet = self._renderer.render(
                        mode=mode,
                        points=points,
                        colors=colors,
                        windows=windows,
                        plane_model=plane_model,
                        quality=temp_quality,
                        pixel_size=0.01,
                        photo_path=None,   # 预留接口
                    )
                except Exception as e:
                    print(f'[PCFD] export_all_heatmaps: render failed for {mode}: '
                          f'{e}', flush=True)
                    continue

                # ---- 写入 triplet（overlay / heatmap_grid / photo） ----
                prefix = f'facade_{int(facade_no):03d}_{mode}'
                paths = {}
                for key, img in triplet.items():
                    if img is None:
                        paths[key] = None
                        continue
                    suffix = {
                        'overlay': '_overlay.png',
                        'heatmap_grid': '_heatmap_grid.png',
                        'photo': '_photo_overlay.png',
                    }.get(key, f'_{key}.png')
                    path = root / (prefix + suffix)
                    if self._safe_imwrite(path, img):
                        paths[key] = str(path)
                    else:
                        paths[key] = None

                # report 缩放图
                report_path = root / f'{prefix}_report.png'
                try:
                    report_img = self._renderer.fit_report_image(triplet['overlay'])
                    self._safe_imwrite(report_path, report_img)
                    paths['report'] = str(report_path)
                except Exception:
                    paths['report'] = None

                # ---- 透明热力图（仅热力层，用于照片叠加） ----
                transparent_path = root / f'{prefix}_transparent.png'
                try:
                    transparent = self._renderer.render_transparent_heatmap(
                        mode=mode,
                        points=points,
                        colors=colors,
                        windows=windows,
                        plane_model=plane_model,
                        quality=temp_quality,
                        pixel_size=0.01,
                    )
                    if transparent is not None:
                        self._safe_imwrite(transparent_path, transparent)
                        paths['transparent'] = str(transparent_path)

                        # 若 triplet 未提供 photo，用透明热力作为 2D 现场热力映射图占位
                        if not paths.get('photo'):
                            photo_placeholder = root / f'{prefix}_photo_overlay.png'
                            if self._safe_imwrite(photo_placeholder, transparent):
                                paths['photo'] = str(photo_placeholder)
                    else:
                        paths['transparent'] = None
                except Exception as e:
                    print(f'[PCFD] export_all_heatmaps: transparent render '
                          f'failed for {mode}: {e}', flush=True)
                    paths['transparent'] = None

                exported[mode] = {
                    'title': spec['title'],
                    **paths,
                }

            return exported

        except Exception as e:
            err_msg = (
                f"=== export_all_heatmaps 异常 ===\n"
                f"立面编号: {facade_no}\n"
                f"输出目录: {root}\n"
                f"异常类型: {type(e).__name__}\n"
                f"异常信息: {e}\n"
                f"堆栈:\n{traceback.format_exc()}"
            )
            print(err_msg, flush=True)
            if root is not None:
                try:
                    (root / 'export_all_error.log').write_text(err_msg, encoding='utf-8')
                except Exception:
                    pass
            return {}

    def export_heatmap(self, results_dir, facade_no, points, colors, quality,
                       pixel_size=0.01):
        """
        兼容旧接口：导出单张热力图（默认导出 overlay）。
        新实现复用 triplet 渲染器但仅返回 overlay 路径。
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

            windows = quality.get('windows') or []
            if len(windows) == 0:
                print(f'[PCFD] export_heatmap: no windows, skip', flush=True)
                return None

            heatmap_mode = normalize_heatmap_mode(quality.get('heatmap_mode'))
            spec = heatmap_spec(heatmap_mode)

            triplet = self._renderer.render(
                mode=heatmap_mode,
                points=points,
                colors=colors,
                windows=windows,
                plane_model=plane_model,
                quality=quality,
                pixel_size=pixel_size,
                photo_path=None,
            )

            prefix = f'facade_{int(facade_no):03d}_{heatmap_mode}'
            overlay_path = root / f'{prefix}_overlay.png'
            self._safe_imwrite(overlay_path, triplet['overlay'])

            # report 缩放图
            report_path = root / f'{prefix}_report.png'
            report_img = self._renderer.fit_report_image(triplet['overlay'])
            self._safe_imwrite(report_path, report_img)

            # heatmap_grid
            grid_path = root / f'{prefix}_heatmap_grid.png'
            self._safe_imwrite(grid_path, triplet['heatmap_grid'])

            print(f'[PCFD] export_heatmap: done facade={facade_no} '
                  f'overlay={overlay_path.name}', flush=True)

            return {
                'root': str(root),
                'mode': heatmap_mode,
                'title': spec['title'],
                'heatmap': str(grid_path),
                'overlay': str(overlay_path),
                'report': str(report_path),
                'legend': None,
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

    _UNIFIED_LEGEND_NAME = 'legend_unified.png'

    def _create_unified_legend(self, root, limit_mm, max_mm):
        """生成全局唯一标准图例，所有热力图模式共用。"""
        legend_path = Path(root) / self._UNIFIED_LEGEND_NAME
        if legend_path.exists():
            return legend_path

        h, w = 80, 500
        legend = np.ones((h, w, 3), dtype=np.uint8) * 245

        bar_h = 28
        bar_y = 16
        bar_x_start = 60
        bar_width = w - 120
        n_segments = bar_width

        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)
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

        self._safe_imwrite(legend_path, legend)
        return legend_path

    def _create_heatmap_legend(self, root, limit_mm, max_mm, mode='flatness'):
        """兼容旧接口：直接返回全局统一图例。"""
        return self._create_unified_legend(root, limit_mm, max_mm)