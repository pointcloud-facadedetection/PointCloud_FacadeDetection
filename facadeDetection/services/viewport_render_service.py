from __future__ import annotations

import time
import hashlib
from typing import Optional, Callable, Tuple, Dict

import numpy as np
from config.settings import Config
from utils.array_utils import as_array
from utils.logging_utils import trace
from services.heatmap_spec import heatmap_spec, normalize_heatmap_mode, defect_colormap
from algorithms.facade.projection import rasterize_facade


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
        self._facade_color_signatures: Dict[str, str] = {}

    # 通知渲染器：显示点云，可选颜色
    def show_point_cloud(self, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        # Keep the initial PLY RGB contract explicit at the viewport boundary.
        # Invalid colors must not silently be bound as a malformed Open3D
        # attribute; no-color input remains a supported gray fallback.
        if colors is not None:
            candidate = np.asarray(colors, dtype=np.float32)
            if candidate.shape != (len(points), 3) or not np.all(np.isfinite(candidate)):
                trace('viewport.colors.invalid', cloud=name,
                      point_count=len(points), color_shape=tuple(candidate.shape))
                colors = None
            else:
                colors = np.ascontiguousarray(np.clip(candidate, 0.0, 1.0))
        # 给定视口应提供用于添加点数据的 API。
        if hasattr(self.viewport, 'add_point_cloud'):
            self.viewport.add_point_cloud(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud_numpy'):
            self.viewport.add_cloud_numpy(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud'):
            self.viewport.add_cloud(name, points, colors)
        else:
            raise RuntimeError('Viewport does not support adding point cloud data')
        try:
            data = self.viewport.get_cloud_data(name)
            if data is not None:
                data.setdefault('domain', 'proxy')
                data.setdefault('index_space', 'proxy_global')
                data.setdefault('proxy_ids', np.arange(len(points), dtype=np.int32))
        except Exception:
            pass

    def clear_station_scene(self):
        for name in list(self.viewport.get_cloud_names()
                         if hasattr(self.viewport, 'get_cloud_names') else []):
            self.viewport.remove_cloud(name)

    def clear_scene_display(self):
        """清除站点/合并/注册结果显示，但不触碰业务索引数据。"""
        invalidate = getattr(self.viewport, 'invalidate_render_queue', None)
        if callable(invalidate):
            invalidate()
        self.clear_station_scene()
        self._facade_color_signatures.clear()
        try:
            if hasattr(self.viewport, 'clear_roi_visuals'):
                self.viewport.clear_roi_visuals()
            if hasattr(self.viewport, 'clear_pick_markers'):
                self.viewport.clear_pick_markers()
        except Exception:
            pass

    def show_station_proxy(self, station_id, name, points, colors=None, dataset_id=None):
        """显示站点代理点云；视口元数据明确标记为 proxy 域。
        若同一 dataset 已显示且点数未变，跳过重复 add_point_cloud 以消除闪烁。"""
        cloud_name = f'pcfd.proxy.station.{station_id}'
        # 早期短路：检查当前视口是否已显示同 dataset 且点数一致的云
        try:
            existing_data = self.viewport.get_cloud_data(cloud_name)
            if (existing_data is not None and
                    dataset_id is not None and
                    existing_data.get('dataset_id') == dataset_id and
                    len(existing_data.get('pos', [])) == len(points)):
                # 仅更新元数据，不重新 add_point_cloud
                existing_data.update({'domain': 'proxy', 'index_space': 'proxy_global',
                                      'is_processing_cloud': True,
                                      'station_id': station_id,
                                      'display_name': name})
                return cloud_name
        except Exception:
            pass
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
        self._facade_color_signatures.clear()
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
            lookup = data.get('display_proxy_lookup')
            if lookup is None:
                lookup = {int(value): row for row, value in enumerate(displayed.tolist())}
                data['display_proxy_lookup'] = lookup
            return np.asarray([lookup.get(int(value), -1) for value in ids], dtype=np.int64)
        # Freshly loaded clouds use identity proxy rows.
        return ids

    def facade_color_for(self, facade: dict, order: int = 0):
        """Return the discrete color shared by viewport and result panel."""
        # Historical projects persist the exact facade color.  Prefer it over
        # the current palette so reopening a project does not remap colors.
        saved = facade.get('color')
        if saved is not None:
            try:
                value = tuple(float(channel) for channel in saved)
                if len(value) == 3:
                    return value
            except (TypeError, ValueError):
                pass
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

    def enter_registration_pick_mode(self, source_cloud, target_cloud,
                                     callback, pick_radius=10):
        """Start registration picking through the public viewport contract."""
        method = getattr(self.viewport, 'enter_registration_pick_mode', None)
        if not callable(method):
            raise RuntimeError('Viewport does not support registration picking')
        method(source_cloud, target_cloud, callback, pick_radius=pick_radius)
        self._pick_mode = True

    def registration_pick_points(self):
        method = getattr(self.viewport, 'registration_pick_points', None)
        return method() if callable(method) else ([], [])

    def get_cloud_data(self, cloud_name):
        """读取视口中某朵点云的元数据字典，供配准快照等只读场景使用。"""
        method = getattr(self.viewport, 'get_cloud_data', None)
        return method(cloud_name) if callable(method) else None

    def get_cloud_names(self):
        """返回视口当前全部点云名称，供活动点云解析等只读场景使用。"""
        method = getattr(self.viewport, 'get_cloud_names', None)
        return method() if callable(method) else []

    def clear_viewport(self):
        """清空视口场景（项目销毁时调用），委托给 viewport.clear。"""
        method = getattr(self.viewport, 'clear', None)
        if callable(method):
            method()

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
        # TODO(内存/渲染性能): highlight_facades：整云 np.tile 颜色矩阵及代理索引映射。
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
            dataset_id = str(data.get('dataset_id') or '')
            revision = str(data.get('dataset_revision') or '')
            digest = hashlib.sha1()
            digest.update(f'{cloud_name}|{dataset_id}|{revision}|{n}'.encode())
            for facade in facades or []:
                digest.update(str(facade.get('id', '')).encode())
                indices = facade.get('proxy_indices') or facade.get('inlier_indices') or []
                digest.update(np.asarray(indices, dtype=np.int64).tobytes())
            signature = digest.hexdigest()
            if self._facade_color_signatures.get(cloud_name) == signature:
                return
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
            self._facade_color_signatures[cloud_name] = signature
        except Exception as e:
            print(f"highlight_facades failed: {e}", flush=True)

    def _facade_base_colors(self, cloud_name, facades, base_color=(0.75, 0.75, 0.75)):
        data = self.viewport.get_cloud_data(cloud_name)
        n = len(data.get('pos', [])) if data is not None else 0
        colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))
        for order, facade in enumerate(facades or []):
            idx = self._proxy_rows_for_display(
                cloud_name, facade.get('proxy_indices') or facade.get('inlier_indices', []))
            idx = idx[(idx >= 0) & (idx < n)]
            if len(idx):
                colors[idx] = np.asarray(self.facade_color_for(facade, order), dtype=np.float32)
        return colors

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
            # 此处勿混合源RGB。
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
        由屏幕二维选框生成三维 ROI AABB。

        实现流程（严格按屏幕投影筛选）：
        1. 用与框选完全相同的相机基准，把点云分块投影到屏幕坐标系；
        2. 保留投影落在屏幕矩形内的**全部**点（矩形本身再略微外扩几像素）；
        3. 直接对这批点的世界坐标取 min/max，得到严格轴对齐包围盒 AABB；
        4. 对 AABB 各轴做适度外扩，避免框选边缘的有效立面点被截断。

        这里刻意不再采用「四角反投影 + 深度分位」的构造方式：那种做法的
        AABB 由 8 个反投影出的虚拟角点决定，深度分位会被框内前景/背景杂点
        拉偏，而且一旦相机基准读取失败就会整体旋转、偏移。改由真实点集直接
        统计后，盒子必然包住框内的所有点，且与相机朝向无关。
        """
        t_start = time.monotonic()

        try:
            # ---------- 0. 数据获取 ----------
            data = self.viewport.get_cloud_data(cloud_name)
            if not data or data.get('pos') is None:
                print("[ROI-BBox] 失败: 无点云数据", flush=True)
                return None, None

            pos = np.asarray(data['pos'], dtype=np.float64)

            camera = getattr(self.viewport, '_camera', None)
            if camera is None:
                print("[ROI-BBox] 失败: 无相机对象", flush=True)
                return None, None

            # ---------- 1~3. 投影筛选 + 严格 AABB ----------
            min_bound = max_bound = None
            hit_count = 0
            if screen_rect is not None:
                min_bound, max_bound, hit_count = self._aabb_from_screen_projection(
                    pos, camera, screen_rect)

            if min_bound is None:
                # 兼容非框选调用 / 投影不可用：退回给定索引点集
                idx = np.unique(np.asarray(indices, dtype=int).reshape(-1))
                idx = idx[(idx >= 0) & (idx < len(pos))]
                if len(idx) < 3:
                    print("[ROI-BBox] 失败: 框内无有效点，且索引点不足以构成包围盒", flush=True)
                    return None, None
                pts = pos[idx]
                min_bound = np.min(pts, axis=0)
                max_bound = np.max(pts, axis=0)
                hit_count = int(len(idx))
                print(f"[ROI-BBox] 提示: 屏幕投影无命中，回退索引点集 n={hit_count}", flush=True)
            elif hit_count == 0:
                print("[ROI-BBox] 失败: 屏幕选框内没有点", flush=True)
                return None, None

            # ---------- 4. 适度外扩 ----------
            min_raw = np.asarray(min_bound, dtype=np.float64).copy()
            max_raw = np.asarray(max_bound, dtype=np.float64).copy()
            min_bound, max_bound = self._expand_bbox(min_raw, max_raw)

            elapsed = time.monotonic() - t_start
            print(
                f"[ROI-BBox] 选中点范围: [{min_raw[0]:.2f},{min_raw[1]:.2f},{min_raw[2]:.2f}] ~ "
                f"[{max_raw[0]:.2f},{max_raw[1]:.2f},{max_raw[2]:.2f}], "
                f"跨度=[{max_raw[0]-min_raw[0]:.2f},{max_raw[1]-min_raw[1]:.2f},{max_raw[2]-min_raw[2]:.2f}]",
                flush=True,
            )
            print(
                f"[ROI-BBox] 成功: n={hit_count}, Bbox=[{min_bound[0]:.2f},{min_bound[1]:.2f},{min_bound[2]:.2f}] ~ "
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

    def _aabb_from_screen_projection(self, pos, camera, screen_rect, pad_px=2.0):
        """分块投影点云，返回屏幕矩形内点集的严格世界坐标 AABB。

        Args:
            pos: (N, 3) 点云世界坐标
            camera: CameraController，提供 project_points()
            screen_rect: (QPoint p1, QPoint p2) 选框对角点（逻辑像素）
            pad_px: 选框像素外扩量，把恰好压在边线上的点也纳入

        Returns:
            (min_bound, max_bound, hit_count)；无命中或投影不可用时
            min_bound/max_bound 为 None、hit_count 为 0。
        """
        p1, p2 = screen_rect
        x1, x2 = sorted([float(p1.x()), float(p2.x())])
        y1, y2 = sorted([float(p1.y()), float(p2.y())])
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            return None, None, 0
        x1 -= pad_px
        x2 += pad_px
        y1 -= pad_px
        y2 += pad_px

        n = int(len(pos))
        chunk = 1_000_000
        min_bound = None
        max_bound = None
        hit_count = 0
        base = 0

        while base < n:
            tail = min(n, base + chunk)
            chunk_pts = pos[base:tail]
            base = tail

            proj = camera.project_points(chunk_pts)
            if proj is None:
                # 相机基准不可用：不要用估测参数硬算，宁可由上层退到索引路径
                print("[ROI-BBox] 警告: 相机基准不可用，投影筛选中断", flush=True)
                return None, None, 0
            screen, valid = proj
            if screen is None or len(screen) == 0:
                continue

            m = (
                valid
                & (screen[:, 0] >= x1)
                & (screen[:, 0] <= x2)
                & (screen[:, 1] >= y1)
                & (screen[:, 1] <= y2)
            )
            if not np.any(m):
                continue

            sel = chunk_pts[m]
            cmin = np.min(sel, axis=0)
            cmax = np.max(sel, axis=0)
            min_bound = cmin if min_bound is None else np.minimum(min_bound, cmin)
            max_bound = cmax if max_bound is None else np.maximum(max_bound, cmax)
            hit_count += int(np.count_nonzero(m))

        return min_bound, max_bound, hit_count

    def _expand_bbox(self, min_bound, max_bound, rel=0.02, floor=0.5):
        """对 AABB 做适度外扩，避免框选边缘的有效立面点被截断。

        各轴外扩量取「跨度 * rel」与 floor（米）中的较大者：
        跨度正常的轴按比例留边，跨度接近 0 的轴（如一次性薄立面）用 floor
        保底，防止几何退化成零厚度平面。
        """
        min_bound = np.asarray(min_bound, dtype=np.float64).copy()
        max_bound = np.asarray(max_bound, dtype=np.float64).copy()
        span = np.maximum(max_bound - min_bound, 0.0)
        pad = np.maximum(span * float(rel), float(floor))
        return min_bound - pad, max_bound + pad

    def _compute_bbox_from_indices(self, pos, indices, t_start):
        """兼容路径：直接对给定索引点求严格 AABB 后适度外扩。"""
        try:
            pts = np.asarray(pos, dtype=np.float64)[indices]
            min_raw = np.min(pts, axis=0)
            max_raw = np.max(pts, axis=0)
            min_bound, max_bound = self._expand_bbox(min_raw, max_raw)
            elapsed = time.monotonic() - t_start
            print(
                f"[ROI-BBox] 成功(索引路径): n={len(indices)}, "
                f"Bbox=[{min_bound[0]:.2f},{min_bound[1]:.2f},{min_bound[2]:.2f}] ~ "
                f"[{max_bound[0]:.2f},{max_bound[1]:.2f},{max_bound[2]:.2f}], 耗时={elapsed:.3f}s",
                flush=True,
            )
            return min_bound, max_bound
        except Exception as e:
            print(f"[ROI-BBox] 索引路径异常: {e}", flush=True)
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
        """渲染平整度热力图（已废弃，请使用 render_quality_reports）。"""
        self.render_quality_reports(cloud_name, facades, index_service=index_service,
                                    heatmap_mode='flatness')

    def apply_quality_colors(self, cloud_name: str, quality_result: dict,
                             base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
                             index_service=None, _colors=None) -> None:
        """将质量结果应用到点云颜色 - 统一缺陷值热力图（与导出图一致）。

        根据 heatmap_mode 的 method 字段，从 quality_comparison.methods
        中动态选取对应的窗口集（靠尺法/全局平面法 × 平整度/垂直度）。
        使用与 result_export_service 一致的 5 节点色标和值域映射。
        """
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

            colors = (_colors if _colors is not None else self._facade_base_colors(
                cloud_name, self._facades_cache.get(cloud_name, []), base_color))
            mode = normalize_heatmap_mode(quality_result.get('heatmap_mode'))
            spec = heatmap_spec(mode)

            method = spec.get('method', 'ruler')
            metric = spec.get('metric', 'flatness')

            comparison = quality_result.get('quality_comparison', {})
            methods_data = comparison.get('methods', {})
            method_data = methods_data.get(method, {})
            metric_data = method_data.get(metric, {})
            windows = metric_data.get('windows', [])

            if not isinstance(windows, list) or len(windows) == 0:
                return

            if index_service is None:
                return
            dataset = index_service._get_dataset(cloud_name)
            if dataset is None:
                return

            centers = np.asarray([r.get('center_xyz', [np.nan] * 3) for r in windows], dtype=np.float32).reshape(-1, 3)
            values_key = spec['value_key']
            values = np.asarray([r.get(values_key, np.nan) for r in windows], dtype=np.float32).reshape(-1)
            pass_key = spec['pass_key']
            failed = np.asarray([not bool(r.get(pass_key, True)) for r in windows], dtype=bool)

            valid = np.isfinite(centers).all(axis=1) & np.isfinite(values) & failed
            if not np.any(valid):
                return

            centers = centers[valid]
            values = values[valid]

            limit = float((quality_result.get('thresholds') or {}).get(
                spec['limit_key'], 4.0))

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

            valid_proxy = ((domain_proxy >= 0) &
                           (domain_proxy < len(dataset.index.proxy_points)))
            if not np.any(valid_proxy):
                return
            domain_proxy = domain_proxy[valid_proxy]
            domain_points = dataset.index.proxy_points[domain_proxy]

            # 复用 projection.py 的标准投影映射：像素级光栅化，非离散窗口块染色。
            # 与离线 result_export_service 使用同一 rasterize_facade 实现，
            # 值域、色标、投影轴完全一致，保证三维视口与导出 PNG 内外一致。
            excess = np.maximum(np.abs(values) - limit, 0.0)
            scale = max(float(np.percentile(excess[excess > 0], 98)) if np.any(excess > 0) else 0.0,
                        limit * 0.15, 1e-6)
            t = np.clip(excess / scale, 0.0, 1.0)
            heat_colors = defect_colormap(t)  # 统一色标（青→绿→黄→橙→红）

            values_m = (values / 1000.0).astype(np.float64)
            limit_m = float(limit) / 1000.0
            global_vmax_m = float((limit + scale) / 1000.0)

            raster = rasterize_facade(
                centers.astype(np.float64),
                np.full((len(centers), 3), 0.7),
                plane,
                values_m,
                limit_m,
                pixel_size=0.01,
                defect_colors=heat_colors,
                vmin=limit_m,
                vmax=global_vmax_m,
                max_size=2400,
                projection_origin=origin,
                projection_u_axis=u_axis,
                projection_v_axis=v_axis)
            overlay = raster['overlay_rgba']
            lo = raster['bounds'][:2]
            size = raster['pixel_size']
            h_pix, w_pix = overlay.shape[:2]

            # 将三维点投影到 UV 并查询对应像素颜色
            rel_domain = domain_points - origin
            dom_u = rel_domain @ u_axis
            dom_v = rel_domain @ v_axis
            px = np.clip(((dom_u - lo[0]) / size).astype(int), 0, w_pix - 1)
            py = np.clip((h_pix - 1 - (dom_v - lo[1]) / size).astype(int), 0, h_pix - 1)
            pix_val = overlay[py, px]
            opaque = pix_val[:, 3] > 0

            # Map proxy IDs to display rows
            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
            lookup = {int(v): i for i, v in enumerate(displayed)} if len(displayed) == n else None
            cols = pix_val[:, :3].astype(np.float32) / 255.0
            for pid, col, vis in zip(domain_proxy.tolist(), cols, opaque):
                if not vis:
                    continue
                row = lookup.get(int(pid), int(pid)) if lookup else int(pid)
                if 0 <= row < n:
                    colors[row] = np.clip(col, 0, 1)

            trace('quality.heatmap', mode=mode,
                   windows=len(values), raw=len(domain_raw),
                   proxy=len(domain_proxy), voxels=int(np.count_nonzero(opaque)),
                   displayed=int(np.count_nonzero(opaque)),
                   step=f'{scale:.4f}')

            if _colors is None:
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

    def render_quality_reports(self, cloud_name: str, facades: list[dict], index_service=None,
                              heatmap_mode: str = 'flatness') -> bool:
        """恢复分色并按指定模式叠加全部兼容质量报告。"""
        heatmap_mode = normalize_heatmap_mode(heatmap_mode)
        self.highlight_facades(cloud_name, facades or [])
        reports = self.compatible_quality_reports(cloud_name, facades or [], index_service=index_service)
        if not reports:
            return False
        data = self.viewport.get_cloud_data(cloud_name)
        colors = self._facade_base_colors(cloud_name, facades or [])
        for _facade, report in reports:
            display_report = dict(report)
            display_report['heatmap_mode'] = heatmap_mode
            self.apply_quality_colors(cloud_name, display_report,
                                       index_service=index_service, _colors=colors)
        self._update_cloud_color(cloud_name, colors)
        return True

    def restore_highlight(self, cloud_name: str, facades: list[dict]) -> None:
        try:
            self.highlight_facades(cloud_name, facades)
        except Exception:
            pass