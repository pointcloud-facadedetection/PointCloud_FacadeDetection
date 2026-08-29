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
        # Do not rely on the renderer to preserve business metadata.  The
        # FileService normally binds this immediately; this fallback also
        # covers alternative viewport adapters and makes the contract visible.
        try:
            data = self.viewport.get_cloud_data(name)
            if data is not None:
                data.setdefault('domain', 'proxy')
                data.setdefault('index_space', 'proxy_global')
                data.setdefault('proxy_ids', np.arange(len(points), dtype=np.int32))
        except Exception:
            pass

    def clear_station_scene(self):
        # The whole visible scene is disposable.  Preserving the registration
        # result here is incorrect because this method is also used by a
        # station switch/denoise refresh and would leave result geometry
        # overlaid with the proxy station.
        for name in list(self.viewport.get_cloud_names()
                         if hasattr(self.viewport, 'get_cloud_names') else []):
            self.viewport.remove_cloud(name)

    def clear_scene_display(self):
        """清除站点/合并/注册结果显示，但不触碰业务索引数据。"""
        self.clear_station_scene()
        try:
            if hasattr(self.viewport, 'clear_roi_visuals'):
                self.viewport.clear_roi_visuals()
            if hasattr(self.viewport, 'clear_pick_markers'):
                self.viewport.clear_pick_markers()
        except Exception:
            pass

    def show_station_proxy(self, station_id, name, points, colors=None, dataset_id=None):
        """显示站点代理点云；视口元数据明确标记为 proxy 域。"""
        cloud_name = f'pcfd.proxy.station.{station_id}'
        self.show_point_cloud(cloud_name, points, colors)
        data = self.viewport.get_cloud_data(cloud_name)
        if data is not None:
            data.update({'domain': 'proxy', 'index_space': 'proxy_global',
                         'is_processing_cloud': True,
                         'station_id': station_id,
                         'display_name': name,
                         'proxy_ids': np.arange(len(points), dtype=np.int32)})
            if dataset_id is not None:
                data['dataset_id'] = dataset_id
        return cloud_name

    def show_result_cloud(self, name, points, colors=None):
        """显示结果快照，不注册到 PointCloudService 的处理数据域。"""
        cloud_name = str(name)
        self.show_point_cloud(cloud_name, points, colors)
        data = self.viewport.get_cloud_data(cloud_name)
        if data is not None:
            data.update({'domain': 'result', 'index_space': 'result',
                         'is_processing_cloud': False})
            data.pop('dataset_id', None)
            data.pop('proxy_ids', None)
        return cloud_name

    def clear_runtime(self):
        """Drop renderer-side references that belong to a project session."""
        self._pick_mode = False
        self._picked_points.clear()
        self._selected_facade_id = None
        self._facades_cache.clear()
        if hasattr(self.viewport, 'exit_pick_mode'):
            try:
                self.viewport.exit_pick_mode()
            except Exception:
                pass
        if hasattr(self.viewport, 'exit_roi_selection'):
            try:
                self.viewport.exit_roi_selection()
            except Exception:
                pass

    def close_project(self):
        self.clear_runtime()

    def show_station_cloud(self, station_id, name, points, colors=None):
        # Compatibility alias: callers must now provide proxy points.
        return self.show_station_proxy(station_id, name, points, colors)

    def _proxy_rows_for_display(self, cloud_name, proxy_ids):
        """Convert dataset-global proxy IDs to current viewport row IDs."""
        data = self.viewport.get_cloud_data(cloud_name)
        if data is None:
            return np.empty(0, dtype=np.int64)
        ids = np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64).reshape(-1)
        n = len(data.get('pos', []))
        if len(displayed) == n:
            lookup = {int(value): row for row, value in enumerate(displayed.tolist())}
            return np.asarray([lookup.get(int(value), -1) for value in ids], dtype=np.int64)
        # Freshly loaded clouds use identity proxy rows.
        return ids

    def facade_color_for(self, facade: dict, order: int = 0):
        """Return the discrete color shared by viewport and result panel."""
        palette = getattr(Config, 'FACADE_INSTANCE_COLORS', []) or []
        if palette:
            try:
                return tuple(palette[int(facade.get('id', order)) % len(palette)])
            except Exception:
                return tuple(palette[order % len(palette)])
        ftype = str(facade.get('type') or facade.get('type_label') or '').lower()
        if 'horizontal' in ftype:
            color = self._facade_colors.get('horizontal')
        elif 'inclined' in ftype:
            color = self._facade_colors.get('inclined')
        else:
            color = self._facade_colors.get('vertical_facade')
        return tuple(color or (0.2, 0.65, 0.95))

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

            for order, f in enumerate(facades or []):
                col = self.facade_color_for(f, order)

                if col is None:
                    continue
                col = np.asarray(col, dtype=np.float32)

                # Detection service guarantees proxy_global after normalization.
                # Never fall back to algorithm-local indices here.
                proxy_ids = f.get('proxy_indices', [])
                if not proxy_ids:
                    proxy_ids = f.get('inlier_indices', [])
                idx = self._proxy_rows_for_display(cloud_name, proxy_ids)
                m = (idx >= 0) & (idx < n)
                idx = idx[m]
                if len(idx):
                    colors[idx] = col

            trace('facade.color', cloud=cloud_name,
                  facades=len(facades or []), displayed_points=n,
                  valid_proxy_indices=int(sum(
                      len(self._proxy_rows_for_display(cloud_name,
                          f.get('proxy_indices') or f.get('inlier_indices', [])))
                       for f in (facades or []))),
                  colored=int(np.sum(np.any(colors != np.asarray(base_color), axis=1))))
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
        """记录选中立面；立面颜色层保持确定性，不重复重绘。"""
        try:
            self._selected_facade_id = int(facade_id)
        except Exception:
            self._selected_facade_id = None

    def clear_selected_facade(self, cloud_name: str | None = None) -> None:
        self._selected_facade_id = None

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
            # Do not blend source RGB here.  A previous heatmap/highlight may
            # otherwise be used as the next base and produces the observed
            # white gradient.  Rendering always starts from a deterministic
            # neutral base.
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
        elif cmap == 'unified_defect':
            # UNIFIED DEFECT: gray -> yellow -> orange -> red
            arr = np.empty((len(t), 3), dtype=np.float32)
            # Gray (0.5,0.5,0.5) at t=0 -> Yellow (1,1,0) at t=0.33
            mask1 = t <= 0.33
            tt1 = t[mask1] / 0.33
            arr[mask1, 0] = 0.5 + 0.5 * tt1
            arr[mask1, 1] = 0.5 + 0.5 * tt1
            arr[mask1, 2] = 0.5 - 0.5 * tt1
            # Yellow -> Orange
            mask2 = (t > 0.33) & (t <= 0.66)
            tt2 = (t[mask2] - 0.33) / 0.33
            arr[mask2, 0] = 1.0
            arr[mask2, 1] = 1.0 - 0.5 * tt2
            arr[mask2, 2] = 0.0
            # Orange -> Red
            mask3 = t > 0.66
            tt3 = (t[mask3] - 0.66) / 0.34
            arr[mask3, 0] = 1.0
            arr[mask3, 1] = 0.5 - 0.5 * tt3
            arr[mask3, 2] = 0.0
            return np.clip(arr, 0.0, 1.0)
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
        - 各向同性外扩（所有轴等比例），避免方向性偏移
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
            n_seed = len(seed)

            # ---------- 1. 计算选中点AABB ----------
            min_raw = np.min(seed, axis=0)
            max_raw = np.max(seed, axis=0)
            center_raw = (min_raw + max_raw) / 2.0
            extent_raw = max_raw - min_raw
            max_span = float(np.max(extent_raw))
            
            print(
                f"[ROI-BBox] 选中 {n_seed} 点, 范围: "
                f"[{min_raw[0]:.2f},{min_raw[1]:.2f},{min_raw[2]:.2f}] ~ "
                f"[{max_raw[0]:.2f},{max_raw[1]:.2f},{max_raw[2]:.2f}], "
                f"跨度=[{extent_raw[0]:.2f},{extent_raw[1]:.2f},{extent_raw[2]:.2f}]",
                flush=True,
            )

            # ---------- 2. 各向同性外扩 ----------
            # 外扩比例：最大跨度的 5%
            expand_ratio = 0.05
            # 绝对最小外扩：1米（保证立面检测有足够空间）
            expand_min = 1.0
            # 绝对最大外扩：最大跨度的 10%（防止过度膨胀）
            expand_max = max_span * 0.10
            
            expand = min(max(max_span * expand_ratio, expand_min), expand_max)
            
            # 各向同性外扩：所有轴等比例扩展
            min_bound = min_raw - expand
            max_bound = max_raw + expand

            # ---------- 3. 可选：相机方向感知的外扩（仅用于厚度方向）----------
            camera = getattr(self.viewport, '_camera', None)
            if camera is not None:
                try:
                    front, _, _ = camera.get_camera_basis()
                    if front is not None:
                        # 计算选中点沿front方向的深度跨度
                        depths = (seed - center_raw) @ front
                        d_min = float(np.min(depths))
                        d_max = float(np.max(depths))
                        depth_span = d_max - d_min
                        
                        # 如果深度跨度很小（用户正对墙面），额外增加front方向厚度
                        # 这确保即使选中点都在同一深度层，也有前后余量
                        if depth_span < max_span * 0.1:  # 深度跨度小于最大跨度的10%
                            extra_thick = max(max_span * 0.05, 0.3)  # 额外5%或0.3米
                            
                            # 向量形式：沿front正负方向各外扩extra_thick/2
                            half_extra = extra_thick / 2.0
                            for i in range(3):
                                if abs(front[i]) > 1e-6:
                                    delta = half_extra * abs(front[i])
                                    min_bound[i] -= delta
                                    max_bound[i] += delta

                except Exception:
                    pass

            # 确保min < max
            for i in range(3):
                if min_bound[i] > max_bound[i]:
                    min_bound[i], max_bound[i] = max_bound[i], min_bound[i]

            elapsed = time.monotonic() - t_start
            final_extent = max_bound - min_bound
            print(
                f"[ROI-BBox] 成功: Bbox=[{min_bound[0]:.2f},{min_bound[1]:.2f},{min_bound[2]:.2f}] ~ "
                f"[{max_bound[0]:.2f},{max_bound[1]:.2f},{max_bound[2]:.2f}], 耗时={elapsed:.3f}s",
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
                idx = self._proxy_rows_for_display(
                    cloud_name, f.get('proxy_indices', f.get('inlier_indices', [])))
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
        """将质量结果应用到点云颜色 - 统一缺陷值热力图。"""
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

            # 从当前颜色层开始叠加热力：项目加载后若已有立面离散色，未命中
            # 质量窗口或无法映射的点保持其所属立面默认色；新增/噪点保持默认色。
            existing = data.get('colors')
            if existing is None:
                existing = data.get('color')
            if existing is not None and len(existing) == n:
                colors = np.asarray(existing, dtype=np.float32).reshape(n, 3).copy()
            else:
                colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))
            mode = quality_result.get('heatmap_mode', 'flatness')
            windows = quality_result.get('windows')

            if not isinstance(windows, list) or len(windows) == 0:
                return

            if index_service is None:
                return
            dataset = index_service._get_dataset(cloud_name)
            if dataset is None:
                return

            # Extract centers and values
            centers = np.asarray([r.get('center_xyz', [np.nan] * 3) for r in windows], dtype=np.float32).reshape(-1, 3)
            
            values_key = {
                'flatness_raw': 'flatness_raw_max_gap_mm',
                'verticality': 'verticality_deviation_mm',
            }.get(mode, 'flatness_gap_mm')
            values = np.asarray([r.get(values_key, np.nan) for r in windows], dtype=np.float32).reshape(-1)

            pass_key = 'verticality_pass' if mode == 'verticality' else 'flatness_pass'
            failed = np.asarray([not bool(r.get(pass_key, True)) for r in windows], dtype=bool)

            valid = np.isfinite(centers).all(axis=1) & np.isfinite(values) & failed
            if not np.any(valid):
                return

            centers = centers[valid]
            values = values[valid]

            # Get limit for scaling
            limit = float((quality_result.get('thresholds') or {}).get(
                'verticality_limit_mm' if mode == 'verticality' else 'flatness_limit_mm', 4.0))

            # Domain mapping
            domain_raw = np.asarray(quality_result.get('__global_indices', []), dtype=np.int64)
            if len(domain_raw) == 0:
                return

            domain_proxy = index_service.map_raw_to_proxy(cloud_name, domain_raw)

            plane = np.asarray((quality_result.get('overall') or {}).get('plane_model') or [], dtype=float)
            if plane.size != 4:
                return
            plane = plane / (np.linalg.norm(plane[:3]) + 1e-12)
            
            u_axis = np.asarray(quality_result.get('projection_u_axis', []), dtype=np.float64)
            v_axis = np.asarray(quality_result.get('projection_v_axis', []), dtype=np.float64)
            if u_axis.size != 3 or v_axis.size != 3:
                from algorithms.geometry import classify_plane, plane_axes
                facade_type, _, _, _ = classify_plane(plane[:3])
                u_axis, v_axis = plane_axes(plane[:3], facade_type)

            origin = np.asarray(quality_result.get('projection_origin',
                                                    np.mean(centers, axis=0)),
                               dtype=np.float64).reshape(3)

            # Map domain points to grid
            valid_proxy = (domain_proxy >= 0)
            if not np.any(valid_proxy):
                return
            domain_proxy = domain_proxy[valid_proxy]
            domain_points = dataset.index.proxy_points[domain_proxy]
            du = (domain_points - origin) @ u_axis
            dv = (domain_points - origin) @ v_axis
            cu = (centers - origin) @ u_axis
            cv = (centers - origin) @ v_axis
            
            step = max(float((quality_result.get('parameters') or {}).get('scan_step_m') or
                              quality_result.get('step_size_m') or 0.05), 1e-6)

            u_min = float(quality_result.get('projection', {}).get('u_min_m', du.min()))
            v_min = float(quality_result.get('projection', {}).get('v_min_m', dv.min()))

            # Cell keys for domain points and window centers
            dkey = np.column_stack((np.floor((du - u_min) / step),
                                    np.floor((dv - v_min) / step))).astype(np.int64)
            ckey = np.column_stack((np.floor((cu - u_min) / step),
                                    np.floor((cv - v_min) / step))).astype(np.int64)

            # Build cell value map: take max absolute defect per cell
            cell_values = {}
            for key, value in zip(ckey.tolist(), values.tolist()):
                key_t = tuple(int(x) for x in key)
                if key_t not in cell_values or abs(float(value)) > abs(cell_values[key_t]):
                    cell_values[key_t] = float(value)

            # Map domain points to cells
            mask = np.asarray([tuple(k) in cell_values for k in dkey], dtype=bool)
            proxy_ids = domain_proxy[mask]
            vals = np.asarray([cell_values[tuple(k)] for k in dkey if tuple(k) in cell_values], dtype=np.float32)

            # Map to display rows
            best = {int(pid): float(value) for pid, value in zip(proxy_ids.tolist(), vals.tolist())}
            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
            lookup = {int(v): i for i, v in enumerate(displayed)} if len(displayed) == n else None
            rows = np.asarray([lookup.get(pid, pid) if lookup else pid for pid in best], dtype=np.int64)
            values_arr = np.asarray([best[int(pid)] for pid in best], dtype=np.float32)

            valid_rows = (rows >= 0) & (rows < n)
            if not np.any(valid_rows):
                return

            finite = values_arr[valid_rows]
            
            # Scale only the excess over the applicable limit.
            excess = np.maximum(np.abs(finite) - limit, 0.0)
            scale = max(float(np.nanpercentile(excess, 97)) if len(excess) else 0.0, 1e-6)
            t = np.clip(excess / scale, 0, 1)
            
            heat_colors = np.zeros((len(t), 3), dtype=np.float32)
            
            # Gray (0.75, 0.75, 0.75) -> Yellow (1, 1, 0) -> Orange (1, 0.5, 0) -> Red (1, 0, 0)
            mask1 = t <= 0.33
            tt1 = t[mask1] / 0.33
            heat_colors[mask1, 0] = 0.75 + 0.25 * tt1
            heat_colors[mask1, 1] = 0.75 + 0.25 * tt1
            heat_colors[mask1, 2] = 0.75 - 0.75 * tt1
            
            mask2 = (t > 0.33) & (t <= 0.66)
            tt2 = (t[mask2] - 0.33) / 0.33
            heat_colors[mask2, 0] = 1.0
            heat_colors[mask2, 1] = 1.0 - 0.5 * tt2
            heat_colors[mask2, 2] = 0.0
            
            mask3 = t > 0.66
            tt3 = (t[mask3] - 0.66) / 0.34
            heat_colors[mask3, 0] = 1.0
            heat_colors[mask3, 1] = 0.5 - 0.5 * tt3
            heat_colors[mask3, 2] = 0.0

            colors[rows[valid_rows]] = np.clip(heat_colors, 0, 1)

            trace('quality.heatmap', mode=mode,
                   windows=len(values), raw=len(domain_raw),
                   proxy=len(domain_proxy), voxels=len(best),
                   displayed=int(np.sum(valid_rows)),
                   step=f'{step:.4f}')

            self._update_cloud_color(cloud_name, colors)

        except Exception as e:
            print(f'立面质量着色失败: {e}', flush=True)

    def compatible_quality_reports(self, cloud_name: str, facades: list[dict], index_service=None) -> list[tuple[dict, dict]]:
        """返回可安全用于当前代理点云的质量报告，旧 revision/缺索引结果不回放。"""
        valid: list[tuple[dict, dict]] = []
        try:
            dataset = index_service._get_dataset(cloud_name) if index_service is not None else None
            revision = getattr(dataset, 'revision', None)
            for facade in facades or []:
                report = facade.get('quality_report')
                if facade.get('quality_status') != 'complete' or not isinstance(report, dict):
                    continue
                if report.get('__global_indices') is None or not isinstance(report.get('windows'), list):
                    continue
                result_revision = facade.get('dataset_revision') or report.get('dataset_revision')
                if revision is not None and result_revision is not None and str(result_revision) != str(revision):
                    continue
                valid.append((facade, report))
        except Exception as exc:
            print(f'[PCFD] quality.compat_check_failed error={exc!r}', flush=True)
        return valid

    def render_quality_reports(self, cloud_name: str, facades: list[dict], index_service=None) -> bool:
        """先恢复立面离散色，再按当前数据集兼容的质量结果叠加热力。"""
        self.highlight_facades(cloud_name, facades or [])
        reports = self.compatible_quality_reports(cloud_name, facades or [], index_service=index_service)
        if not reports:
            return False
        for _facade, report in reports:
            self.apply_quality_colors(cloud_name, report, index_service=index_service)
        return True

    def restore_highlight(self, cloud_name: str, facades: list[dict]) -> None:
        try:
            self.highlight_facades(cloud_name, facades)
        except Exception:
            pass