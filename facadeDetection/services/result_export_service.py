from __future__ import annotations
import json
import traceback
from pathlib import Path
import numpy as np
import cv2
import uuid
from algorithms.facade.projection import rasterize_facade


class ResultExportService:
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
            indices = np.asarray(
                quality.get('__global_indices') or [], dtype=int)
            pts = np.asarray(points, dtype=float)

            if len(indices) == 0:
                # 若未提供全局索引，则视输入 points 已为局部点云
                pts_local = pts
                rgb = np.asarray(
                    colors, dtype=float) if colors is not None else None
            else:
                pts_local = pts[indices]
                rgb = np.asarray(colors, dtype=float)[
                    indices] if colors is not None else None

            if len(pts_local) == 0:
                raise ValueError(
                    f'立面 {facade_id} 没有有效点，无法导出')

            # ── 2. 获取有符号距离（局部索引空间）──
            signed = np.asarray(
                quality.get('signed_gap') or [], dtype=float)
            if len(signed) != len(pts_local):
                raise ValueError(
                    f'signed_gap 长度 ({len(signed)}) 与局部点云 '
                    f'({len(pts_local)}) 不匹配，必须使用立面局部索引空间')

            # ── 3. 获取稀疏缺陷信息 ──
            defect_local = np.asarray(
                quality.get('defect_local_indices') or [], dtype=int)
            defect_values = np.asarray(
                quality.get('defect_values') or [], dtype=float)
            defect_colors = np.asarray(
                quality.get('defect_colors') or [], dtype=float)

            if len(defect_local) != len(defect_values) or \
               len(defect_values) != len(defect_colors):
                raise ValueError(
                    '缺陷索引、数值与颜色长度必须相等')
            if len(defect_local) and (
                    np.any(defect_local < 0) or
                    np.any(defect_local >= len(pts_local))):
                raise ValueError(
                    'defect_local_indices 超出立面局部索引范围')

            full_values = np.full(len(pts_local), np.nan, dtype=float)
            full_colors = np.zeros((len(pts_local), 3), dtype=float)
            if len(defect_local):
                full_values[defect_local] = defect_values
                full_colors[defect_local] = defect_colors

            # ── 4. 栅格化投影 ──
            flatness_limit = quality.get('flatness_limit') or 0.004
            raster = rasterize_facade(
                pts_local,
                rgb,
                plane_model,
                signed,
                flatness_limit,
                pixel_size,
                defect_values=full_values,
                defect_colors=full_colors,
                vmin=quality.get('heatmap_vmin'),
                vmax=quality.get('heatmap_vmax'))

            # ── 5. 诊断底图：压暗灰度 + 空洞填充 ──
            base_rgb = raster['base_rgb'].astype(np.float32)
            # 保留原始彩色立面作为独立文件
            original_base = cv2.cvtColor(
                raster['base_rgb'], cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(
                base_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            base_display = np.dstack(
                [gray, gray, gray]).astype(np.float32) * 0.28
            base_display = np.clip(
                base_display + 4.0, 0, 255).astype(np.uint8)

            # 填充无点区域（count==0）为暗灰，避免纯黑背景
            count = raster['count']
            empty_mask = (count == 0)
            if np.any(empty_mask):
                base_display[empty_mask] = [30, 30, 34]

            base = cv2.cvtColor(base_display, cv2.COLOR_RGB2BGR)

            # ── 6. 自适应形态学放大 ──
            target_diameter_m = 0.20
            kernel_px = max(
                5,
                int(np.ceil(
                    target_diameter_m / max(float(pixel_size), 1e-4))))
            if kernel_px % 2 == 0:
                kernel_px += 1
            kernel_px = min(kernel_px, 41)  # 上限 41，防止过度模糊

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))

            overlay_rgba = np.asarray(
                raster['overlay_rgba'], dtype=np.uint8).copy()

            # 先对 Alpha 做 CLOSE 填补空洞
            mask = (overlay_rgba[:, :, 3] > 0).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 再对 RGBA 整体做 DILATE，使颜色与 Alpha 同步扩散
            dilate_iters = max(1, min(4, int(12 // kernel_px) + 1))
            overlay_rgba = cv2.dilate(
                overlay_rgba, kernel, iterations=dilate_iters)

            # 将 CLOSE 后的 mask 也合并到 alpha（确保空洞被填补）
            overlay_rgba[:, :, 3] = np.maximum(overlay_rgba[:, :, 3], mask)

            # 填充 dilate 后新增像素的 RGB（从最近原始缺陷点复制颜色）
            old_alpha = (raster['overlay_rgba'][:, :, 3] > 0).astype(np.uint8)
            new_pixels = (overlay_rgba[:, :, 3] > 0) & (old_alpha == 0)
            if np.any(new_pixels):
                nearest_mask = old_alpha
                _, labels = cv2.distanceTransformWithLabels(
                    (1 - nearest_mask) * 255, cv2.DIST_L2, 3,
                    labelType=cv2.DIST_LABEL_PIXEL)
                source = np.argwhere(nearest_mask > 0)
                if len(source):
                    label = labels[new_pixels] - 1
                    valid = (label >= 0) & (label < len(source))
                    yy, xx = np.where(new_pixels)
                    overlay_rgba[yy[valid], xx[valid], :3] = raster['overlay_rgba'][
                        source[label[valid], 0], source[label[valid], 1], :3]

            # ── 7. 预乘 Alpha 高斯模糊：光晕连续性 ──
            blur_sigma = max(2.0, kernel_px / 2.0)
            blur_k = max(5, int(np.ceil(blur_sigma * 6)))
            if blur_k % 2 == 0:
                blur_k += 1
            blur_k = min(blur_k, 81)

            rgb_f = overlay_rgba[:, :, :3].astype(np.float32)
            alpha_f = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

            # 预乘：避免透明区域颜色泄漏
            premul = rgb_f * alpha_f
            premul_blur = cv2.GaussianBlur(
                premul, (blur_k, blur_k), blur_sigma)
            alpha_blur = cv2.GaussianBlur(
                alpha_f, (blur_k, blur_k), blur_sigma)

            # 数值稳定性：限制最小 alpha，防止除零产生噪点
            alpha_blur = np.clip(alpha_blur, 1e-6, 1.0)
            rgb_blur = premul_blur / alpha_blur

            overlay_rgba[:, :, :3] = np.clip(
                rgb_blur, 0, 255).astype(np.uint8)
            overlay_rgba[:, :, 3] = np.clip(
                alpha_blur[:, :, 0] * 255, 0, 255).astype(np.uint8)

            # ── 8. 颜色增强：极端对比度与饱和度 ──
            heat_rgb = overlay_rgba[:, :, :3].astype(np.float32) / 255.0
            # 对比度 2.0×
            heat_rgb = np.clip((heat_rgb - 0.5) * 2.0 + 0.5, 0, 1)
            # 饱和度 1.8×
            mean = np.mean(heat_rgb, axis=2, keepdims=True)
            heat_rgb = np.clip(
                mean + (heat_rgb - mean) * 1.8, 0, 1)
            overlay_rgba[:, :, :3] = (heat_rgb * 255).astype(np.uint8)

            # ── 9. 合成：以压暗底图为基底 ──
            composite = cv2.cvtColor(
                base_display, cv2.COLOR_RGB2RGBA)

            visible = overlay_rgba[:, :, 3] > 0
            alpha = (overlay_rgba[:, :, 3:4].astype(
                np.float32) / 255.0) * 0.96
            alpha_flat = alpha[:, :, 0][visible, None]
            composite[visible, :3] = (
                composite[visible, :3] * (1.0 - alpha_flat) +
                overlay_rgba[visible, :3] * alpha_flat
            ).astype(np.uint8)
            composite[visible, 3] = 255

            # ── 10. 写入文件并验证 ──
            ok1 = cv2.imwrite(
                str(root / 'defect_heatmap_rgba.png'),
                cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA))
            ok2 = cv2.imwrite(
                str(root / 'defect_overlay.png'),
                cv2.cvtColor(composite, cv2.COLOR_RGBA2BGRA))

            if not all([ok1, ok2]):
                failed = [n for n, o in zip(
                    ['defect_heatmap_rgba', 'defect_overlay'],
                    [ok1, ok2]) if not o]
                raise RuntimeError(f'图像写入失败: {failed}')

            # 保存栅格元数据（点数、缺陷掩码、UV 坐标）
            np.savez_compressed(
                root / 'heatmap_grid.npz',
                count=raster['count'],
                defect_mask=raster['defect_mask'],
                uv=raster['uv'])

            # 构建质量元数据
            meta = {
                'facade_id': int(facade_id),
                'pixel_size': float(raster['pixel_size']),
                'vmin': float(raster['vmin']),
                'vmax': float(raster['vmax']),
                'cmap': 'diverging_blue_white_red',
                'polarity': 'signed: negative=recessed, positive=protruding',
                'flatness_limit': float(flatness_limit),
                'plane_model': [float(x) for x in plane_model],
            }
            meta['interval_size_m'] = quality.get(
                'interval_size_m', quality.get('grid_size', 20.0))
            meta['window_size_m'] = quality.get(
                'window_size_m', quality.get('ruler_size', 2.0))
            meta['step_size_m'] = quality.get(
                'step_size_m', quality.get('ruler_step', 0.05))

            (root / 'quality.json').write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding='utf-8')

            return {
                'directory': str(root),
                **{k: str(root / f) for k, f in {
                    'heatmap': 'defect_heatmap_rgba.png',
                    'overlay': 'defect_overlay.png',
                    'grid': 'heatmap_grid.npz',
                    'json': 'quality.json'
                }.items()}
            }

        except Exception as e:
            # 全链路异常捕获：确保错误被记录，不会静默失败
            err_msg = (
                f"=== export_facade 异常 ===\n"
                f"立面 ID: {facade_id}\n"
                f"输出目录: {root}\n"
                f"异常类型: {type(e).__name__}\n"
                f"异常信息: {e}\n"
                f"堆栈:\n{traceback.format_exc()}"
            )
            print(err_msg, flush=True)
            # 若目录已创建，将错误日志写入磁盘以便排查
            if root is not None:
                try:
                    (root / 'export_error.log').write_text(
                        err_msg, encoding='utf-8')
                except Exception:
                    pass
            raise  # 继续向上抛，让调用方感知失败