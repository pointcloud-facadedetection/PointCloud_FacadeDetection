from __future__ import annotations
import json
import traceback
from pathlib import Path
import numpy as np
import cv2
import uuid
from algorithms.facade.projection import rasterize_facade


class ResultExportService:
    @staticmethod
    def _array(value, dtype=float):
        """Accept both legacy lists and the memory-safe ndarray payloads."""
        return np.asarray([] if value is None else value, dtype=dtype)

    def start_run(self, results_dir, project_name='', interval_size_m=20.0):
        """创建一次新的检测运行目录，并返回目录路径与初始清单。"""
        run = Path(results_dir) / f'run_{uuid.uuid4().hex[:12]}'
        run.mkdir(parents=True, exist_ok=True)
        return run, {
            'project_name': project_name,
            'interval_size_m': interval_size_m,
            'facades': []
        }

    @staticmethod
    def save_manifest(run_dir, manifest):
        """以原子写方式保存清单 JSON，防止写入中断导致文件损坏。"""
        path = Path(run_dir) / 'manifest.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8')
        tmp.replace(path)
        return path

    def export_facade(self, results_dir, facade_id, points, colors, quality,
                      pixel_size=0.01):
        """
        导出单个立面的投影图、缺陷热力图及叠加合成图。
        修正版：使用有符号距离区分凹陷/凸起，轻量高斯平滑替代过度膨胀。
        """
        root = None
        try:
            # ── 0. 参数防御性校验 ──
            if not isinstance(results_dir, (str, Path)) or str(results_dir) == '':
                raise ValueError(f'results_dir 无效: {results_dir}')
            if not isinstance(quality, dict):
                raise ValueError('quality 必须为字典类型')
            overall = quality.get('overall', {})
            plane_model = overall.get('plane_model')
            if plane_model is None or len(plane_model) != 4:
                raise ValueError(
                    'quality["overall"]["plane_model"] 缺失或格式错误')

            root = Path(results_dir) / f'facade_{int(facade_id):03d}'
            root.mkdir(parents=True, exist_ok=True)

            # ── 1. 提取立面局部点云 ──
            indices = self._array(quality.get('__global_indices'), dtype=np.int64).reshape(-1)
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f'points 必须为 N×3，实际形状: {pts.shape}')
            colors_arr = None
            if colors is not None:
                candidate = np.asarray(colors, dtype=float)
                if candidate.ndim == 2 and candidate.shape == (len(pts), 3):
                    colors_arr = candidate
                else:
                    print(f'export_facade: 忽略非法 colors 形状 {candidate.shape}', flush=True)

            local_raw = quality.get('__points_index_space') == 'facade_local_raw'
            if local_raw:
                if len(indices) not in (0, len(pts)):
                    raise ValueError('facade_local_raw 的 raw 全局索引必须与局部点集等长')
                pts_local, rgb = pts, colors_arr
            elif len(indices):
                if np.any(indices < 0) or np.any(indices >= len(pts)):
                    raise ValueError(
                        f'__global_indices 越界: range=[0,{len(pts)-1}], '
                        f'min={indices.min()}, max={indices.max()}')
                pts_local = pts[indices]
                rgb = colors_arr[indices] if colors_arr is not None else None
            else:
                pts_local = pts
                rgb = colors_arr

            if len(pts_local) == 0:
                raise ValueError(f'立面 {facade_id} 没有有效点，无法导出')

            windows = quality.get('windows') or {}
            if isinstance(windows, dict):
                summary = {
                    'facade_id': int(facade_id),
                    'point_count': int(len(pts_local)),
                    'interval_size_m': float(quality['interval_size_m']),
                    'thresholds': quality.get('thresholds', {}),
                    'overall': quality.get('overall', {}),
                    'intervals': quality.get('intervals', [])
                }
                json_path = root / f'facade_{int(facade_id):03d}_quality.json'
                json_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.tolist()),
                    encoding='utf-8')
                arrays = {k: np.asarray(v) for k, v in windows.items()}
                npz_path = root / f'facade_{int(facade_id):03d}_windows.npz'
                np.savez_compressed(npz_path, **arrays)
                heatmap_path = self._export_window_heatmap(
                    root, facade_id, pts_local, windows, plane_model, pixel_size, quality)
                legend_path = self._create_heatmap_legend(
                    root, quality.get('thresholds', {}).get('flatness_limit_mm', 4.0),
                    float(overall.get('flatness_max_gap_mm', 4.0)))
                return {
                    'root': str(root),
                    'json': str(json_path),
                    'windows': str(npz_path),
                    'heatmap': str(heatmap_path),
                    'legend': str(legend_path)
                }

            raise ValueError('质量结果必须包含 windows 数组契约')

        except Exception as e:
            err_msg = (
                f"=== export_facade 异常 ===\n"
                f"立面 ID: {facade_id}\n"
                f"输出目录: {root}\n"
                f"异常类型: {type(e).__name__}\n"
                f"异常信息: {e}\n"
                f"堆栈:\n{traceback.format_exc()}"
            )
            print(err_msg, flush=True)
            if root is not None:
                try:
                    (root / 'export_error.log').write_text(err_msg, encoding='utf-8')
                except Exception:
                    pass
            raise

    def _export_window_heatmap(self, root, facade_id, pts_local, windows, plane_model, pixel_size, quality):
        """
        把窗口结果投影成 PNG；使用有符号距离区分凹陷/凸起，轻量高斯平滑。
        """
        centers = np.asarray(windows.get('center_xyz', []), dtype=float).reshape(-1, 3)
        flat = np.asarray(windows.get('flatness_gap_mm', []), dtype=float).reshape(-1)
        signed_flat = np.asarray(windows.get('flatness_signed_gap_mm', []), dtype=float).reshape(-1)

        if len(centers) != len(flat) or not len(centers):
            raise ValueError('质量结果没有有效窗口，无法导出热力图')

        limit_m = float(quality.get('thresholds', {}).get('flatness_limit_mm', 4.0)) / 1000.0
        values_m = signed_flat / 1000.0  # 使用有符号值（单位：m）

        # ── 颜色映射：区分凹陷/凸起 ──
        abs_values_m = np.abs(values_m)
        t = np.clip((abs_values_m - limit_m) / max(limit_m, 1e-9), 0.0, 1.0)

        colors = np.zeros((len(values_m), 3), dtype=float)
        is_recessed = values_m < 0

        # 凸起（正值）：黄色(0.5,0.8,0.2) → 红色(1.0,0.0,0.0)
        protrude_mask = ~is_recessed
        colors[protrude_mask, 0] = np.clip(0.5 + 0.5 * t[protrude_mask], 0, 1)   # R
        colors[protrude_mask, 1] = np.clip(0.8 - 0.8 * t[protrude_mask], 0, 1)   # G
        colors[protrude_mask, 2] = np.clip(0.2 - 0.2 * t[protrude_mask], 0, 1)   # B

        # 凹陷（负值）：蓝色(0.2,0.5,0.8) → 青色(0.0,0.8,1.0)
        colors[is_recessed, 0] = np.clip(0.2 - 0.2 * t[is_recessed], 0, 1)       # R
        colors[is_recessed, 1] = np.clip(0.5 + 0.3 * t[is_recessed], 0, 1)       # G
        colors[is_recessed, 2] = np.clip(0.8 + 0.2 * t[is_recessed], 0, 1)       # B

        base = np.full((len(centers), 3), 0.75, dtype=float)

        # 使用 rasterize_facade 进行栅格化，传入有符号值
        raster = rasterize_facade(
            centers, base, plane_model, values_m, limit_m,
            pixel_size=pixel_size, defect_values=values_m,
            defect_colors=colors, vmin=limit_m
        )

        # ── 轻量高斯平滑（替代过度膨胀）──
        overlay = raster['overlay_rgba'].copy()
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0

        if np.any(alpha > 0):
            rgb = overlay[:, :, :3].astype(np.float32)
            # 预乘 alpha 避免透明区域颜色泄漏
            premul = rgb * alpha[:, :, None]
            # 轻量平滑：3x3 核，sigma=1.0，保持边界清晰
            premul_blur = cv2.GaussianBlur(premul, (3, 3), 1.0)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 1.0)
            alpha_blur = np.clip(alpha_blur, 1e-6, 1.0)
            rgb_smooth = premul_blur / alpha_blur[:, :, None]
            overlay[:, :, :3] = np.clip(rgb_smooth, 0, 255).astype(np.uint8)
            overlay[:, :, 3] = np.clip(alpha_blur * 255, 0, 255).astype(np.uint8)

        # ── 合成底图 + 热力图 ──
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)
        base_rgb = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2BGR)

        visible = overlay[:, :, 3:4].astype(np.float32) / 255.0
        composite = (
            base_rgb.astype(np.float32) * (1.0 - visible[:, :, :1]) +
            overlay[:, :, :3].astype(np.float32) * visible[:, :, :1]
        ).astype(np.uint8)

        # ── 保存 ──
        heatmap_path = Path(root) / f'facade_{int(facade_id):03d}_defect_heatmap.png'
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

        # 渐变条
        bar_h = 28
        bar_y = 16
        bar_x_start = 60
        bar_width = w - 120
        n_segments = bar_width

        for i in range(n_segments):
            t = i / max(n_segments - 1, 1)
            # 左半：凹陷（蓝→青）
            if t < 0.5:
                tt = t * 2
                r = int(np.clip((0.2 - 0.2 * tt) * 255, 0, 255))
                g = int(np.clip((0.5 + 0.3 * tt) * 255, 0, 255))
                b = int(np.clip((0.8 + 0.2 * tt) * 255, 0, 255))
            # 右半：凸起（黄→红）
            else:
                tt = (t - 0.5) * 2
                r = int(np.clip((0.5 + 0.5 * tt) * 255, 0, 255))
                g = int(np.clip((0.8 - 0.8 * tt) * 255, 0, 255))
                b = int(np.clip((0.2 - 0.2 * tt) * 255, 0, 255))

            legend[bar_y:bar_y + bar_h, bar_x_start + i] = [b, g, r]  # BGR

        # 标注文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (60, 60, 60)
        thickness = 1

        # 左侧：凹陷标注
        cv2.putText(legend, f"凹陷", (10, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"-{max_mm:.1f}mm", (10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        # 中间：合格阈值
        mid_x = w // 2 - 30
        cv2.putText(legend, f"合格", (mid_x, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"<{limit_mm:.1f}mm", (mid_x - 10, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        # 右侧：凸起标注
        cv2.putText(legend, f"凸起", (w - 70, bar_y + bar_h + 20), font, font_scale, color, thickness)
        cv2.putText(legend, f"+{max_mm:.1f}mm", (w - 80, bar_y + bar_h + 38), font, 0.35, (100, 100, 100), 1)

        # 保存
        legend_path = Path(root) / 'heatmap_legend.png'
        cv2.imwrite(str(legend_path), legend)
        return legend_path