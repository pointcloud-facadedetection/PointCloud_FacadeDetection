from __future__ import annotations

import time
from typing import Optional, Callable, Tuple, Dict

import numpy as np
from config.settings import Config
from utils.array_utils import as_array
from utils.logging_utils import trace


class ViewportRenderService:
    """
    封装视口的渲染交互逻辑，提供统一的 API 给上层服务使用。
    """

    def __init__(self, viewport, db):
        self.viewport = viewport
        self.db = db
        self._pick_mode = False
        self._picked_points = []
        # Facade coloring scheme and selection state
        self._facade_colors = dict(Config.FACADE_TYPE_COLORS)
        self._highlight_color = tuple(Config.HIGHLIGHT_COLOR)
        self._selected_facade_id: Optional[int] = None
        self._facades_cache: Dict[str, list[dict]] = {}

    # 通知渲染器：显示点云，可选颜色
    def show_point_cloud(self, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        # 给定视口应提供用于添加点数据的 API。
        if hasattr(self.viewport, 'add_point_cloud'):
            self.viewport.add_point_cloud(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud_numpy'):
            self.viewport.add_cloud_numpy(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud'):
            self.viewport.add_cloud(name, points, colors)
        else:
            raise RuntimeError('Viewport does not support adding point cloud data')

    def show_image(self, name: str, image: np.ndarray):
        if hasattr(self.viewport, 'show_image'):
            self.viewport.show_image(name, image)
        else:
            raise RuntimeError('Viewport does not support image display')

    def get_widget(self):
        if hasattr(self.viewport, 'get_widget'):
            return self.viewport.get_widget()
        if hasattr(self.viewport, 'native') and hasattr(self.viewport.native, 'widget'):
            return self.viewport.native.widget
        raise RuntimeError('Viewport has no Qt widget to embed')

    def enter_pick_mode(self, callback: Optional[Callable] = None,
                        cloud_name: Optional[str] = None,
                        pick_radius: int = 8):
        if not hasattr(self.viewport, 'enter_pick_mode'):
            raise RuntimeError('Viewport does not support pick mode')
        self.viewport.enter_pick_mode(
            cloud_name=cloud_name,
            pick_radius=pick_radius,
            callback=callback
        )
        self._pick_mode = True

    def exit_pick_mode(self):
        if hasattr(self.viewport, 'exit_pick_mode'):
            self.viewport.exit_pick_mode()
        self._pick_mode = False

    def is_pick_mode(self) -> bool:
        return self._pick_mode

    def add_pick_marker(self, point):
        """记录拾取点并刷新视口标记。"""
        self._picked_points.append(point)
        if hasattr(self.viewport, 'update_pick_markers'):
            self.viewport.update_pick_markers(src_points=self._picked_points)

    def clear_pick_markers(self):
        """清除所有拾取标记并重置记录。"""
        if hasattr(self.viewport, 'clear_pick_markers'):
            self.viewport.clear_pick_markers()
        self._picked_points.clear()

    # ---- Facade highlighting ----
    def highlight_facades(self, cloud_name: str, facades: list[dict], base_color=(0.75, 0.75, 0.75)):
        """
        立面着色策略（统一颜色规则 + 选中高亮）：
        - 非立面点使用基础色 base_color。
        - 每个立面实例按 ID 使用不同颜色；类型颜色作为无配置时的回退。
        - 所有水平面统一使用 Config.FACADE_TYPE_COLORS['horizontal']。
        - 其它（如倾斜面）统一使用 Config.FACADE_TYPE_COLORS['inclined']（若存在）。
        - 当存在选中的立面 self._selected_facade_id 时，该立面的点使用 Config.HIGHLIGHT_COLOR。
        """
        try:
            if not hasattr(self.viewport, 'get_cloud_data'):
                return
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return

            n = len(pos)
            colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))

            try:
                self._facades_cache[cloud_name] = facades or []
            except Exception:
                pass

            instance_colors = getattr(Config, 'FACADE_INSTANCE_COLORS', [])
            for order, f in enumerate(facades or []):
                ftype = str(f.get('type') or f.get('type_label') or '').lower()
                col = None
                if 'vertical' in ftype:
                    col = self._facade_colors.get('vertical_facade')
                elif 'horizontal' in ftype:
                    col = self._facade_colors.get('horizontal')
                else:
                    col = self._facade_colors.get('inclined') or self._facade_colors.get('vertical_facade')

                if instance_colors:
                    col = Config.facade_instance_color(f.get('id', order), order)

                # Apply highlight if selected
                try:
                    fid = int(f.get('id', -1))
                except Exception:
                    fid = -1
                if self._selected_facade_id is not None and fid == self._selected_facade_id:
                    col = self._highlight_color

                if col is None:
                    continue
                col = np.asarray(col, dtype=np.float32)

                idx = np.asarray(f.get('inlier_indices') or [], dtype=int)
                m = (idx >= 0) & (idx < n)
                idx = idx[m]
                if len(idx):
                    colors[idx] = col

            self._update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"highlight_facades failed: {e}", flush=True)

    def set_facade_colors(self,
                          vertical: Tuple[float, float, float] | None = None,
                          horizontal: Tuple[float, float, float] | None = None,
                          inclined: Tuple[float, float, float] | None = None,
                          highlight: Tuple[float, float, float] | None = None) -> None:
        """更新立面配色方案."""
        if vertical is not None:
            self._facade_colors['vertical_facade'] = tuple(vertical)
        if horizontal is not None:
            self._facade_colors['horizontal'] = tuple(horizontal)
        if inclined is not None:
            self._facade_colors['inclined'] = tuple(inclined)
        if highlight is not None:
            self._highlight_color = tuple(highlight)

    def select_facade(self, cloud_name: str, facade_id: int) -> None:
        """根据 ID 突出显示单个立面，同时保持其他立面的文字颜色统一。"""
        try:
            self._selected_facade_id = int(facade_id)
        except Exception:
            self._selected_facade_id = None
        facades = self._facades_cache.get(cloud_name) or []
        if facades:
            self.highlight_facades(cloud_name, facades)

    def clear_selected_facade(self, cloud_name: str | None = None) -> None:
        self._selected_facade_id = None
        if cloud_name is None:
            return
        facades = self._facades_cache.get(cloud_name) or []
        if facades:
            self.highlight_facades(cloud_name, facades)

    def set_global_point_color(self, color: Tuple[float, float, float]) -> None:
        """在视口内将整个点云场景统一着色."""
        try:
            if not hasattr(self.viewport, 'get_cloud_names'):
                return
            names = self.viewport.get_cloud_names()
            if not names:
                return
            for name in names:
                data = self.viewport.get_cloud_data(name)
                if data is None:
                    continue
                pos = data.get('pos')
                if pos is None or len(pos) == 0:
                    continue
                n = len(pos)
                col = np.tile(np.asarray(color, dtype=np.float32).reshape(1, 3), (n, 1))
                self._update_cloud_color(name, col)
        except Exception as e:
            print(f"set_global_point_color failed: {e}", flush=True)

    # ---- 热力图渲染 ----
    def colorize_by_rgb(self, cloud_name: str, indices: np.ndarray, colors_rgb: np.ndarray,
                        base_color=(0.18, 0.20, 0.24)) -> None:
        """按算法已生成的 RGB 颜色着色，避免再次做标量归一化/色带计算。"""
        try:
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None or data.get('pos') is None or len(data['pos']) == 0:
                return
            n = len(data['pos'])
            idx = np.asarray(indices, dtype=int).reshape(-1)
            rgb = np.asarray(colors_rgb, dtype=np.float32).reshape(-1, 3)
            if len(idx) != len(rgb):
                return
            valid = (idx >= 0) & (idx < n)
            if not np.any(valid):
                return
            original = data.get('color')
            # Keep RGB only when it is sufficiently dark; bright source RGB
            # makes defect colors unreadable in the 3D view.
            if original is not None and np.asarray(original).shape == (n, 3):
                source = np.asarray(original, dtype=np.float32)
                colors = np.clip(source * 0.35, 0.0, 0.42)
            else:
                colors = np.tile(np.asarray(base_color, dtype=np.float32), (n, 1))
            colors[idx[valid]] = np.clip(rgb[valid] * 1.15, 0.0, 1.0)
            self._update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"colorize_by_rgb failed: {e}", flush=True)

    def colorize_by_scalar(self, cloud_name: str, indices: np.ndarray, values: np.ndarray,
                            vmin: float | None = None, vmax: float | None = None,
                            base_color=(0.75, 0.75, 0.75), cmap: str = 'turbo') -> None:
        """
        根据给定的标量值对指定点进行热力着色，其余点使用 base_color。
        - indices: 全局点索引（0..N-1）的一维数组
        - values: 与 indices 对齐的浮点数组
        """
        try:
            if not hasattr(self.viewport, 'get_cloud_data'):
                return
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return
            n = len(pos)
            idx = np.asarray(indices, dtype=int).reshape(-1)
            val = np.asarray(values, dtype=float).reshape(-1)
            if len(idx) == 0 or len(idx) != len(val):
                return
            m = (idx >= 0) & (idx < n)
            idx = idx[m]
            val = val[m]
            if len(idx) == 0:
                return
            if vmin is None:
                vmin = float(np.min(val))
            if vmax is None:
                vmax = float(np.max(val))
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-6
            # normalize to 0..1
            t = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
            colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))
            colors[idx] = self._colormap(t, cmap)
            self._update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"colorize_by_scalar failed: {e}", flush=True)

    def _update_cloud_color(self, cloud_name: str, colors: np.ndarray) -> None:
        queue = getattr(self.viewport, 'queue_update_cloud_color', None)
        if callable(queue):
            queue(cloud_name, colors)
        else:
            self.viewport.update_cloud_color(cloud_name, colors)

    @staticmethod
    def _colormap(t: np.ndarray, cmap: str = 'turbo') -> np.ndarray:
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if cmap in ('diverging', 'diverging_blue_white_red', 'signed'):
            # Signed deviation: -1=recessed (blue), 0=reference (white),
            # +1=protruding (red).  This is deliberately symmetric.
            t = np.clip(t, 0.0, 1.0)
            blue = np.array([0.05, 0.35, 0.95], dtype=np.float32)
            white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            red = np.array([0.95, 0.08, 0.04], dtype=np.float32)
            arr = np.empty((len(t), 3), dtype=np.float32)
            left = t <= 0.5
            q = (t[left] * 2.0)[:, None]
            arr[left] = blue + (white - blue) * q
            q = ((t[~left] - 0.5) * 2.0)[:, None]
            arr[~left] = white + (red - white) * q
            return np.clip(arr, 0.0, 1.0)
        if cmap == 'turbo':
            # Lightweight Turbo approximation
            r = 0.135 - 0.157*t + 2.776*t**2 - 2.443*t**3
            g = 0.091 - 1.33*t + 3.51*t**2 - 1.84*t**3
            b = 0.106 + 1.097*t - 2.295*t**2 + 1.98*t**3
            arr = np.stack([r, g, b], axis=1)
        else:
            # simple blue->cyan->yellow->red
            arr = np.empty((len(t), 3), dtype=np.float32)
            arr[:, 0] = np.clip(2*t - 0.5, 0.0, 1.0)
            arr[:, 1] = np.clip(2*t, 0.0, 1.0)
            arr[:, 2] = np.clip(1.5 - 2*t, 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    # ---- ROI utilities (selection -> facade plane -> building BBOX) ----
    def compute_building_bbox_from_selection(
        self,
        cloud_name: str,
        indices: list[int],
        screen_rect: tuple | None = None,
        thickness: float | None = None,
        plane: np.ndarray | None = None,
        tol: float | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        核心设计：
        - 不拟合平面，直接用选中点3D坐标
        - 沿相机front方向外扩厚度（向量形式，非单轴）
        - 使用正确的world_per_pixel
        """
        t_start = time.monotonic()

        try:
            # ---------- 0. 数据获取 ----------
            data = self.viewport.get_cloud_data(cloud_name)
            if not data or data.get('pos') is None:
                print("[ROI-BBox] 失败: 无点云数据", flush=True)
                return None, None

            all_pos = np.asarray(data['pos'], dtype=np.float64).reshape(-1, 3)
            idx = np.unique(np.asarray(indices, dtype=int).reshape(-1))
            idx = idx[(idx >= 0) & (idx < len(all_pos))]
            if len(idx) < 3:
                print(f"[ROI-BBox] 失败: 选中点过少 ({len(idx)})", flush=True)
                return None, None

            seed = all_pos[idx]

            # ---------- 1. 直接计算选中点AABB ----------
            min_raw = np.min(seed, axis=0)
            max_raw = np.max(seed, axis=0)
            center_raw = (min_raw + max_raw) / 2.0
            extent_raw = max_raw - min_raw
            print(f"[ROI-BBox] 选中点范围: [{min_raw[0]:.2f},{min_raw[1]:.2f},{min_raw[2]:.2f}] ~ "
                  f"[{max_raw[0]:.2f},{max_raw[1]:.2f},{max_raw[2]:.2f}], "
                  f"跨度=[{extent_raw[0]:.2f},{extent_raw[1]:.2f},{extent_raw[2]:.2f}]", flush=True)

            # ---------- 2. 获取相机参数 ----------
            camera = getattr(self.viewport, '_camera', None)
            if camera is None:
                print("[ROI-BBox] 警告: 无camera，使用纯几何AABB", flush=True)
                # 无相机时，直接返回选中点AABB + 各向同性外扩
                margin = np.max(extent_raw) * 0.05 + 0.5
                return min_raw - margin, max_raw + margin

            front, up, right = camera.get_camera_basis()
            if front is None:
                print("[ROI-BBox] 警告: 无法获取相机基向量，使用纯几何AABB", flush=True)
                margin = np.max(extent_raw) * 0.05 + 0.5
                return min_raw - margin, max_raw + margin

            # ---------- 3. 沿front方向外扩厚度 ----------
            # 计算选中点沿front方向的深度分布
            # 使用选中点自身中心作为参考（比lookat更稳定）
            depths = (seed - center_raw) @ front
            
            # 过滤前后各2.5%的极端点
            d_p05 = float(np.percentile(depths, 2.5))
            d_p95 = float(np.percentile(depths, 97.5))
            d_center = (d_p05 + d_p95) / 2.0
            # d_min = float(np.min(depths))
            # d_max = float(np.max(depths))
            depth_span = d_p95 - d_p05

            # 厚度策略：基于选中点自身的深度跨度，而非整个场景
            # 目标：覆盖选中点前后一定范围，给立面检测留余量
            if thickness is None:
                # 基础厚度：选中点深度跨度的50%，最小0.5米，最大5米
                base_thick = max(depth_span * 0.5, 0.5)
                base_thick = min(base_thick, 5.0)  # 限制最大5米
                thickness = base_thick

            # half_thick: 沿front方向从中心外扩的距离
            # 确保包含所有选中点（d_min到d_max）+ 额外余量
            half_thick = max(thickness / 2.0, abs(d_p05 - d_center), abs(d_p95 - d_center))

            print(f"[ROI-BBox] 深度分布: min={d_p05:.3f}, max={d_p95:.3f}, "
                  f"span={depth_span:.3f}, half_thick={half_thick:.3f}", flush=True)

            # 计算过滤后中心在front方向上的位置
            center_depth_offset = (d_p05 + d_p95) / 2.0
            effective_center = center_raw + center_depth_offset * front

            # 重新计算相对于effective_center的深度
            depths_centered = (seed - effective_center) @ front
            d_min_eff = float(np.min(depths_centered))
            d_max_eff = float(np.max(depths_centered))

            # 目标范围: [-half_thick, +half_thick]
            expand_front = max(0, half_thick - d_max_eff)
            expand_back = max(0, -half_thick - d_min_eff)


            # 应用到min/max（注意：front方向的分量可能使min变小、max变大）
            min_bound = min_raw.copy()
            max_bound = max_raw.copy()

            # 沿front方向，每个轴独立外扩
            if expand_front > 0:
                for i in range(3):
                    if front[i] > 1e-6:
                        max_bound[i] += expand_front * front[i]
                    elif front[i] < -1e-6:
                        min_bound[i] += expand_front * front[i]

            # 后向扩展（沿front负方向）
            if expand_back > 0:
                for i in range(3):
                    if front[i] > 1e-6:
                        min_bound[i] -= expand_back * front[i]
                    elif front[i] < -1e-6:
                        max_bound[i] -= expand_back * front[i]

            # ---------- 5. 非front方向轻微外扩（避免边界截断）----------
            margin = np.max(extent_raw) * 0.01 + 0.05  # 1% + 5cm
            
            # 找到垂直于front的平面内的两个轴
            for i in range(3):
                if abs(right[i]) > 1e-6:
                    if right[i] > 0:
                        min_bound[i] -= margin * right[i]
                        max_bound[i] += margin * right[i]
                    else:
                        min_bound[i] += margin * right[i]
                        max_bound[i] -= margin * right[i]
            
            # 在up方向外扩
            for i in range(3):
                if abs(up[i]) > 1e-6:
                    if up[i] > 0:
                        min_bound[i] -= margin * up[i]
                        max_bound[i] += margin * up[i]
                    else:
                        min_bound[i] += margin * up[i]
                        max_bound[i] -= margin * up[i]

            # 确保min < max
            for i in range(3):
                if min_bound[i] > max_bound[i]:
                    min_bound[i], max_bound[i] = max_bound[i], min_bound[i]

            elapsed = time.monotonic() - t_start
            print(
                f"[ROI-BBox] 成功: Bbox=[{min_bound[0]:.2f},{min_bound[1]:.2f},{min_bound[2]:.2f}] ~ "
                f"[{max_bound[0]:.2f},{max_bound[1]:.2f},{max_bound[2]:.2f}], "
                f"耗时={elapsed:.3f}s",
                flush=True,
            )

            return min_bound, max_bound

        except Exception as e:
            elapsed = time.monotonic() - t_start
            print(f"[ROI-BBox] 异常 ({elapsed:.3f}s): {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None, None

    def _estimate_scene_scale(self, cloud_name: str | None = None) -> float:
        """估计场景尺度（点云最大范围）。"""
        try:
            if cloud_name:
                data = self.viewport.get_cloud_data(cloud_name)
                if data and data.get('pos') is not None and len(data['pos']) > 0:
                    pos = np.asarray(data['pos'])
                    extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                    return max(float(np.max(extent)), 1.0)
            # 如果没有指定 cloud_name，尝试所有点云
            max_extent = 0.0
            for name in self.viewport.get_cloud_names():
                data = self.viewport.get_cloud_data(name)
                if data and data.get('pos') is not None and len(data['pos']) > 0:
                    pos = np.asarray(data['pos'])
                    extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                    max_extent = max(max_extent, float(np.max(extent)))
            return max(max_extent, 1.0)
        except Exception:
            return 100.0  # 默认 100 米

    def visualize_building_bbox(self, min_bound, max_bound, color=(1.0, 0.2, 0.2)) -> None:
        """Render a 3D bbox for the building ROI in the viewport."""
        try:
            if hasattr(self.viewport, 'show_roi_bbox'):
                self.viewport.show_roi_bbox(min_bound, max_bound, color=color)
        except Exception:
            pass

    # ---- 从原 FacadeService 迁移的 UI 渲染方法 ----

    def render_flatness_heatmap(self, cloud_name: str, facades: list[dict],
                                vmin: float | None = None,
                                vmax: float | None = None,
                                quality_results=None,
                                index_service=None) -> None:
        """渲染平整度热力图。"""
        try:
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            positions = data.get('pos')
            n_total = len(positions) if positions is not None else 0
            if n_total == 0:
                return

            if quality_results is not None:
                results = quality_results if isinstance(quality_results, dict) else {}
                if not any(k in results for k in ('defect_local_indices', 'defect_colors')):
                    results = {r.get('facade_id', r.get('id')): r for r in quality_results}

                all_indices, all_colors = [], []
                for f in facades or []:
                    r = results.get(f.get('id')) if isinstance(results, dict) else None
                    if r is None and isinstance(quality_results, dict) and len(facades) == 1:
                        r = quality_results
                    if not r:
                        continue

                    global_idx = as_array(r.get('__global_indices'), dtype=np.int32)
                    local_idx = as_array(r.get('defect_local_indices'), dtype=np.int32)
                    colors = as_array(r.get('defect_colors'), dtype=np.float32).reshape(-1, 3)

                    valid_local = ((local_idx >= 0) &
                                   (local_idx < len(global_idx)) &
                                   (np.arange(len(local_idx)) < len(colors)))
                    if np.any(valid_local):
                        gi = global_idx[local_idx[valid_local]]
                        if index_service is not None:
                            proxy_ids = index_service.map_raw_to_proxy(cloud_name, gi)
                            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
                            if len(displayed) == n_total:
                                lookup = {int(v): i for i, v in enumerate(displayed)}
                                gi = np.asarray([lookup.get(int(p), -1) for p in proxy_ids], dtype=np.int64)
                            else:
                                gi = proxy_ids

                        valid_global = (gi >= 0) & (gi < n_total)
                        if np.any(valid_global):
                            all_indices.append(gi[valid_global])
                            all_colors.append(colors[np.flatnonzero(valid_local)[valid_global]])

                if all_indices:
                    idx_cat = np.concatenate(all_indices)
                    col_cat = np.concatenate(all_colors)
                    self.colorize_by_rgb(cloud_name, idx_cat, col_cat)
                return

            # Fallback: 基于平面距离着色
            all_indices, all_values = [], []
            for f in facades or []:
                idx = np.asarray(f.get('inlier_indices', []) or [], dtype=int)
                idx = idx[(idx >= 0) & (idx < n_total)]
                if len(idx) == 0:
                    continue
                model = np.asarray(f.get('plane_model') or [], dtype=float)
                if model.shape[0] != 4:
                    continue
                pts = np.asarray(data['pos'])[idx]
                n = model[:3]
                n = n / (np.linalg.norm(n) + 1e-12)
                d = float(model[3])
                dist = np.abs(pts @ n + d)
                all_indices.append(idx)
                all_values.append(dist.astype(float))

            if not all_indices:
                return
            idx_cat = np.concatenate(all_indices, axis=0)
            val_cat = np.concatenate(all_values, axis=0)
            self.colorize_by_scalar(cloud_name, idx_cat, val_cat,
                                    vmin=vmin, vmax=vmax, cmap='turbo')

        except Exception as e:
            print(f'ViewportRenderService: 渲染质量热力贴图失败: {e}', flush=True)

    def apply_quality_colors(self, cloud_name: str, quality_result: dict,
                             base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
                             index_service=None) -> None:
        """将质量结果应用到点云颜色。"""
        try:
            if not isinstance(quality_result, dict):
                raise TypeError(f'quality_result must be dict, got {type(quality_result).__name__}')
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return
            n = len(pos)

            colors = np.tile(np.asarray((0.06, 0.08, 0.11), dtype=np.float32).reshape(1, 3), (n, 1))
            mode = quality_result.get('heatmap_mode', 'flatness')
            windows = quality_result.get('windows')

            if not isinstance(windows, dict):
                raise ValueError('质量结果缺少 windows 数组')

            if index_service is None:
                return
            dataset = index_service._get_dataset(cloud_name)
            if dataset is None:
                return

            centers = np.asarray(windows.get('center_xyz', []), dtype=np.float32).reshape(-1, 3)
            values_key = {
                'verticality': 'verticality_angle_deg',
                'recessed': 'flatness_signed_gap_mm',
                'protruding': 'flatness_signed_gap_mm',
            }.get(mode, 'flatness_gap_mm')
            values = np.asarray(windows.get(values_key, []), dtype=np.float32).reshape(-1)

            if len(centers) != len(values):
                return
            valid = np.isfinite(centers).all(axis=1) & np.isfinite(values)
            if not np.any(valid):
                return

            domain_raw = np.asarray(quality_result.get('__global_indices', []), dtype=np.int64)
            if len(domain_raw) == 0:
                return

            domain_proxy = index_service.map_raw_to_proxy(cloud_name, domain_raw)

            from algorithms.geometry import plane_axes, classify_plane
            plane = np.asarray((quality_result.get('overall') or {}).get('plane_model') or [], dtype=float)
            if plane.size != 4:
                return
            plane = plane / (np.linalg.norm(plane[:3]) + 1e-12)
            facade_type, _type_label, _v, _h = classify_plane(plane[:3])
            u_axis = np.asarray(quality_result.get('projection_u_axis', []), dtype=float)
            v_axis = np.asarray(quality_result.get('projection_v_axis', []), dtype=float)
            if u_axis.size != 3 or v_axis.size != 3:
                u_axis, v_axis = plane_axes(plane[:3], facade_type)

            valid_centers = valid
            centers = centers[valid_centers]
            values = values[valid_centers]
            origin = np.asarray(quality_result.get('projection_origin',
                                                    np.mean(centers, axis=0)),
                               dtype=np.float64).reshape(3)

            valid_proxy = (domain_proxy >= 0)
            if not np.any(valid_proxy):
                return
            domain_proxy = domain_proxy[valid_proxy]
            domain_points = dataset.index.proxy_points[domain_proxy]
            du = (domain_points - origin) @ u_axis
            dv = (domain_points - origin) @ v_axis
            cu = (centers - origin) @ u_axis
            cv = (centers - origin) @ v_axis
            step = max(float((quality_result.get('step_size_m') or
                              quality_result.get('window_size_m') or 0.05)), 1e-6)

            u_min = float(quality_result.get('projection_u_min_m', du.min()))
            v_min = float(quality_result.get('projection_v_min_m', dv.min()))
            dkey = np.column_stack((np.floor((du - u_min) / step),
                                    np.floor((dv - v_min) / step))).astype(np.int64)
            saved_u = np.asarray(windows.get('cell_u', []), dtype=np.int64).reshape(-1)
            saved_v = np.asarray(windows.get('cell_v', []), dtype=np.int64).reshape(-1)
            ckey = np.column_stack((saved_u, saved_v)) if len(saved_u) == len(values) and len(saved_v) == len(values) else np.column_stack((np.floor((cu - u_min) / step),
                                    np.floor((cv - v_min) / step))).astype(np.int64)

            cell_values = {tuple(k): float(v) for k, v in zip(ckey.tolist(), values.tolist())}
            mask = np.asarray([tuple(k) in cell_values for k in dkey], dtype=bool)
            proxy_ids = domain_proxy[mask]
            vals = np.asarray([cell_values[tuple(k)] for k in dkey if tuple(k) in cell_values], dtype=np.float32)

            best = {int(pid): float(value) for pid, value in zip(proxy_ids.tolist(), vals.tolist())}
            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
            lookup = {int(v): i for i, v in enumerate(displayed)} if len(displayed) == n else None
            rows = np.asarray([lookup.get(pid, pid) if lookup else pid for pid in best], dtype=np.int64)
            values_arr = np.asarray([best[int(pid)] for pid in best], dtype=np.float32)

            valid_rows = (rows >= 0) & (rows < n)
            if not np.any(valid_rows):
                return

            finite = values_arr[valid_rows]
            limit = float((quality_result.get('thresholds') or {}).get('flatness_limit_mm', 1.0))
            if mode == 'verticality':
                limit = float((quality_result.get('thresholds') or {}).get('verticality_limit_deg', limit))

            if mode in ('flatness', 'verticality'):
                scale = max(limit * 2.0, 1e-6)
                t = np.clip(np.abs(finite) / scale, 0, 1)
                colors[rows[valid_rows]] = np.stack([t, 0.55 + 0.35 * (1 - t), 0.08 * (1 - t)], axis=1)
            elif mode == 'recessed':
                negative = np.abs(finite[finite < 0])
                scale = max(float(np.percentile(negative, 98)) if len(negative) else limit, 1e-6)
                t = np.clip(np.abs(finite) / scale, 0, 1)
                colors[rows[valid_rows]] = np.stack([
                    np.clip(0.2 - 0.2 * t, 0, 1),
                    np.clip(0.5 + 0.3 * t, 0, 1),
                    np.clip(0.8 + 0.2 * t, 0, 1)
                ], axis=1)
            elif mode == 'protruding':
                positive = np.abs(finite[finite > 0])
                scale = max(float(np.percentile(positive, 98)) if len(positive) else limit, 1e-6)
                t = np.clip(np.abs(finite) / scale, 0, 1)
                colors[rows[valid_rows]] = np.stack([
                    np.clip(0.5 + 0.5 * t, 0, 1),
                    np.clip(0.8 - 0.8 * t, 0, 1),
                    np.clip(0.2 - 0.2 * t, 0, 1)
                ], axis=1)

            trace('quality.heatmap', mode=mode,
                   windows=len(values), raw=len(domain_raw),
                   proxy=len(domain_proxy), voxels=len(best),
                   displayed=int(np.sum(valid_rows)),
                   step=f'{step:.4f}',
                   interval_origin=quality_result.get('interval_origin_m', 0.0))

            self._update_cloud_color(cloud_name, colors)

        except Exception as e:
            print(f'立面质量着色失败: {e}', flush=True)

    def restore_highlight(self, cloud_name: str, facades: list[dict]) -> None:
        try:
            self.highlight_facades(cloud_name, facades)
        except Exception:
            pass
