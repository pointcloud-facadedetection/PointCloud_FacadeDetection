import numpy as np
import time
from PySide6.QtCore import QPoint, Qt
from config.settings import Config


class ViewportInteractor:
    # 首轮只需找屏幕邻域候选；随后会在原始索引附近精细搜索。
    # 25 万点可显著降低交互延迟，同时保持建筑点云的拾取精度。
    PICK_SAMPLE_POINTS = 250_000
    PICK_REFINE_STEPS = 64

    def __init__(self, adapter, camera, scene):
        self.adapter = adapter
        self.camera = camera
        self.scene = scene

        self.pick_enabled = False
        self.pick_radius = 8
        self.pick_cloud = None
        self.point_pick_callback = None
        self.selection_enabled = False
        self.selection_cloud = None
        self.selection_callback = None
        self._selection_start = None
        self._selection_current = None
        self._completed_selection_rect = None   # 持久化已释放的 ROI 框
        self._pan_start = None
        self._click_start = None
        self._rotate_start = None
        # Speed/scales from config
        self._right_pan_scale = float(getattr(Config, 'PAN_BASE_SPEED', 0.06))
        self._left_rotate_scale = float(getattr(Config, 'ROTATE_SPEED', 1.0))
        self._wheel_scale = float(getattr(Config, 'ZOOM_WHEEL_SCALE', 1.6))  # 鼠标滚轮缩放倍率
        self.last_picked_point = None
        self._last_pick_key = None
        self._last_pick_ts = 0.0
        self._cached_pan_scale = None
        self._pick_projection_cache = {}

    # ------------------------------------------------------------------
    # 选择模式状态管理
    # ------------------------------------------------------------------
    def set_pick_enabled(self, enabled: bool, radius: int | None = None, cloud_name: str | None = None):
        self.pick_enabled = bool(enabled)
        self._invalidate_pick_projection_cache()
        if radius is not None:
            try:
                self.pick_radius = int(radius)
            except Exception:
                pass
        if cloud_name is not None:
            self.pick_cloud = cloud_name

    def _invalidate_pick_projection_cache(self):
        self._pick_projection_cache.clear()

    def set_selection_enabled(self, enabled, cloud_name=None):
        """
        启用/禁用选择模式。

        Args:
            enabled: 是否启用选择模式
            cloud_name: 目标点云名称
        """
        try:
            self.selection_enabled = bool(enabled)
            if enabled:
                if cloud_name is not None:
                    self.selection_cloud = cloud_name
                # 启用框选时禁止点选，重置交互状态
                self.pick_enabled = False
                self._selection_start = None
                self._selection_current = None
                # 不清除 _completed_selection_rect，允许在退出后保留最后一次红框
                self._pan_start = None
                self._rotate_start = None
                self._click_start = None
            else:
                # 退出框选时仅重置进行中的交互，不清除完成的红框（由上层决定何时 clear）
                self._selection_start = None
                self._selection_current = None
                self._pan_start = None
                self._rotate_start = None
                self._click_start = None
        except Exception:
            # 兜底，避免异常导致模式切换失败
            self.selection_enabled = bool(enabled)
            if enabled and cloud_name is not None:
                self.selection_cloud = cloud_name
    def clear_selection_rect(self):
        self._selection_start = None
        self._selection_current = None
        self._completed_selection_rect = None

    def get_selection_rect(self):
        if not self.selection_enabled and self._completed_selection_rect is None:
            return None
        if self.selection_enabled and self._selection_start is not None and self._selection_current is not None:
            return self._selection_start, self._selection_current
        if self._completed_selection_rect is not None:
            return self._completed_selection_rect
        return None

    # ------------------------------------------------------------------
    # 鼠标事件：选择模式下全部拦截，零漫游
    # ------------------------------------------------------------------
    def handle_mouse_press(self, event):
        pos = self._event_pos(event)

        # selection_enabled 模式由 Overlay 接管；当 input_locked 时直接吞事件
        if getattr(self, 'input_locked', False):
            return True

        if event.button() == Qt.LeftButton:
            self._rotate_start = QPoint(pos)
            if self.pick_enabled:
                self._click_start = QPoint(pos)
            return True

        if event.button() == Qt.RightButton:
            self._pan_start = QPoint(pos)
            ctr = self.adapter.get_view_control()
            self._cached_pan_scale = self._compute_pan_scale(ctr) if ctr is not None else self._right_pan_scale
            return True

        return False

    def handle_mouse_move(self, event):
        pos = self._event_pos(event)

        # ROI 硬锁时禁止相机交互
        if getattr(self, 'input_locked', False):
            return True

        # 右键拖拽平移 —— 映射到相机局部坐标：camera_local_translate(forward, right, up)
        if self._pan_start is not None and (event.buttons() & Qt.RightButton):
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()
            self._pan_start = QPoint(pos)
            if dx or dy:
                self._invalidate_pick_projection_cache()
            ctr = self.adapter.get_view_control()
            if ctr is not None:
                try:
                    # 优先使用 ViewControl.translate(像素位移)，方向随鼠标
                    if hasattr(ctr, 'translate'):
                        ctr.translate(int(-dx), int(dy))
                    else:
                        # 像素->世界：手工平移 lookat（右手坐标系）
                        scale = self._cached_pan_scale if self._cached_pan_scale is not None else self._compute_pan_scale(ctr)
                        try:
                            import numpy as _np
                            look = _np.asarray(ctr.get_lookat(), dtype=float)
                            front = _np.asarray(ctr.get_front(), dtype=float)
                            up = _np.asarray(ctr.get_up(), dtype=float)
                            front = front / ( _np.linalg.norm(front) + 1e-12 )
                            up = up - front * float(_np.dot(front, up))
                            up = up / ( _np.linalg.norm(up) + 1e-12 )
                            right = _np.cross(front, up)
                            right = right / ( _np.linalg.norm(right) + 1e-12 )
                            move = (-dx * scale) * right + (dy * scale) * up
                            ctr.set_lookat(look + move)
                            ctr.set_front(front)
                            ctr.set_up(up)
                        except Exception:
                            pass
                except Exception:
                    pass
            return True

        # 左键拖拽旋转
        if self._rotate_start is not None and (event.buttons() & Qt.LeftButton):
            dx = pos.x() - self._rotate_start.x()
            dy = pos.y() - self._rotate_start.y()
            self._rotate_start = QPoint(pos)
            if dx or dy:
                self._invalidate_pick_projection_cache()
            ctr = self.adapter.get_view_control()
            if ctr is not None:
                try:
                    ctr.rotate(int(dx * self._left_rotate_scale), int(dy * self._left_rotate_scale))
                except Exception:
                    # Fallback：使用 translate 做少量平移模拟
                    try:
                        if hasattr(ctr, 'translate'):
                            ctr.translate(int(-dx * 0.5), int(dy * 0.5))
                        else:
                            scale = self._right_pan_scale * 0.5
                            import numpy as _np
                            look = _np.asarray(ctr.get_lookat(), dtype=float)
                            front = _np.asarray(ctr.get_front(), dtype=float)
                            up = _np.asarray(ctr.get_up(), dtype=float)
                            front = front / ( _np.linalg.norm(front) + 1e-12 )
                            up = up - front * float(_np.dot(front, up))
                            up = up / ( _np.linalg.norm(up) + 1e-12 )
                            right = _np.cross(front, up)
                            right = right / ( _np.linalg.norm(right) + 1e-12 )
                            move = (-dx * scale) * right + (dy * scale) * up
                            ctr.set_lookat(look + move)
                            ctr.set_front(front)
                            ctr.set_up(up)
                    except Exception:
                        pass
            return True

        return False

    def handle_mouse_release(self, event):
        pos = self._event_pos(event)

        # ROI 硬锁时吞掉释放事件
        if getattr(self, 'input_locked', False):
            return True

        # 右键释放：结束平移
        if event.button() == Qt.RightButton:
            self._pan_start = None
            self._cached_pan_scale = None
            return True

        # 左键释放：拾取
        if event.button() == Qt.LeftButton:
            self._rotate_start = None
            if self.pick_enabled:
                start = self._click_start
                self._click_start = None
                if start is not None:
                    moved = abs(pos.x() - start.x()) + abs(pos.y() - start.y())
                    if moved <= 4:
                        picked = self.pick_nearest_point(pos, cloud_name=self.pick_cloud)
                        if picked:
                            key = (picked.get("cloud_name"), picked.get("index"))
                            now = time.monotonic()
                            if self._last_pick_key == key and (now - self._last_pick_ts) < 0.15:
                                return True
                            self._last_pick_key = key
                            self._last_pick_ts = now
                            self.last_picked_point = np.asarray(picked["point"], dtype=float)
                            if self.point_pick_callback:
                                try:
                                    self.point_pick_callback(picked)
                                except Exception:
                                    pass
                            return True
                return False

        return False

    def select_indices_in_rect(self, cloud_name, start, end):
        """
        根据屏幕矩形选框，投影点云并返回框内点的全局索引。

        Args:
            cloud_name: 点云名称
            start, end: 选框对角点（QPoint，逻辑像素坐标）

        Returns:
            np.ndarray: 框内点的全局索引数组
        """
        name = cloud_name or self.scene.active_name
        data = self.scene.get_cloud_data(name)
        if data is None:
            return np.array([], dtype=np.int64)

        pos = data.get("pos")
        if pos is None or len(pos) == 0:
            return np.array([], dtype=np.int64)

        x1, x2 = sorted([int(start.x()), int(end.x())])
        y1, y2 = sorted([int(start.y()), int(end.y())])

        if (x2 - x1) < 1 or (y2 - y1) < 1:
            return np.array([], dtype=np.int64)

        n = int(len(pos))
        chunk = 1_000_000
        hits = []
        base = 0
        
        while base < n:
            tail = min(n, base + chunk)
            pts = pos[base:tail]
            proj = self.camera.project_points(pts)
            if proj is None:
                base = tail
                continue
            screen, valid = proj
            if screen is None or len(screen) == 0:
                base = tail
                continue
            
            # 注意：screen坐标是逻辑像素，与Qt事件坐标一致
            m = (
                valid
                & (screen[:, 0] >= x1)
                & (screen[:, 0] <= x2)
                & (screen[:, 1] >= y1)
                & (screen[:, 1] <= y2)
            )
            if np.any(m):
                local = np.nonzero(m)[0].astype(np.int64)
                if len(local):
                    hits.extend((base + local).tolist())
            base = tail

        if not hits:
            return np.array([], dtype=np.int64)

        idx = np.unique(np.asarray(hits, dtype=np.int64))

        try:
            if getattr(Config, 'DEBUG_SELECTION', False):
                print(
                    f"[DEBUG] Selection: rect=({x1},{y1})-({x2},{y2}), "
                    f"cloud={name}, total={n}, selected={len(idx)}",
                    flush=True,
                )
        except Exception:
            pass
            
        return idx

    def pick_nearest_point(self, pos, cloud_name=None):
        started = time.perf_counter()
        best = None
        best_dist = float(self.pick_radius)
        best_any = None
        best_any_dist = float("inf")
        if cloud_name:
            items = [(cloud_name, self.scene.get_cloud_data(cloud_name))]
        else:
            items = self.scene.point_data.items()

        for name, data in items:
            if data is None:
                continue
            points = data["pos"]
            if len(points) == 0:
                continue

            count = len(points)
            step = max(1, (count + self.PICK_SAMPLE_POINTS - 1) // self.PICK_SAMPLE_POINTS)
            cache_key = (name, id(points), count, step)
            cached = self._pick_projection_cache.get(cache_key)
            cache_hit = cached is not None
            if cached is None:
                sampled = points[::step]
                projected = self.camera.project_points(sampled)
                if projected is None:
                    continue
                screen, valid = projected
                self._pick_projection_cache = {
                    cache_key: (screen, valid)
                }
            else:
                screen, valid = cached

            dist = np.hypot(screen[:, 0] - pos.x(), screen[:, 1] - pos.y())
            dist[~valid] = np.inf
            idx = int(np.argmin(dist))
            if np.isfinite(dist[idx]):
                point_index = min(idx * step, count - 1)
                d = float(dist[idx])

                # 粗采样只用于定位候选区；对原始索引附近的小窗口再投影，
                # 避免降低采样量后红点偏离鼠标点击位置。
                radius = self.PICK_REFINE_STEPS * step
                lo = max(0, point_index - radius)
                hi = min(count, point_index + radius + 1)
                refined = self.camera.project_points(points[lo:hi])
                if refined is not None:
                    refined_screen, refined_valid = refined
                    refined_dist = np.hypot(
                        refined_screen[:, 0] - pos.x(),
                        refined_screen[:, 1] - pos.y(),
                    )
                    refined_dist[~refined_valid] = np.inf
                    refined_idx = int(np.argmin(refined_dist))
                    if np.isfinite(refined_dist[refined_idx]):
                        point_index = lo + refined_idx
                        d = float(refined_dist[refined_idx])

                if d < best_any_dist:
                    point_index_any = point_index
                    best_any_dist = d
                    best_any = {
                        "cloud_name": name,
                        "index": point_index_any,
                        "point": points[point_index_any].astype(float).tolist(),
                    }
                if d < best_dist:
                    best_dist = d
                    best = {
                        "cloud_name": name,
                        "index": point_index,
                        "point": points[point_index].astype(float).tolist(),
                    }
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(
                    f'[Pick] search points={count} sampled={len(screen)} '
                    f'cache={cache_hit} elapsed={elapsed_ms:.1f}ms',
                    flush=True,
                )
        return best or best_any

    def handle_wheel(self, event):
        if self.selection_enabled:
            return True
        try:
            delta = 0
            if hasattr(event, "angleDelta"):
                ad = event.angleDelta()
                delta = ad.y() if hasattr(ad, "y") else 0
            if delta == 0 and hasattr(event, "pixelDelta"):
                pd = event.pixelDelta()
                delta = pd.y() if hasattr(pd, "y") else 0
            if delta == 0:
                return False

            steps = round(delta / 120.0)
            factor = float(self._wheel_scale ** steps)
            self._invalidate_pick_projection_cache()

            ctr = self.adapter.get_view_control()
            if ctr is None:
                return False

            try:
                if hasattr(ctr, "scale"):
                    ctr.scale(factor)
                else:
                    raise AttributeError()
            except Exception:
                try:
                    z = ctr.get_zoom()
                    new_z = max(min(z / factor, 2.5), 0.02)
                    ctr.set_zoom(new_z)
                except Exception:
                    try:
                        sign = -1.0 if steps > 0 else 1.0
                        ctr.camera_local_translate(0.0, 0.0, sign * 0.02)
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _viewport_metrics(self):
        w = h = 1
        dpr = 1.0
        vw = getattr(self.camera, "viewport_widget", None)
        try:
            if vw is not None:
                w = max(1, int(vw.width()))
                h = max(1, int(vw.height()))
                if hasattr(vw, "devicePixelRatioF"):
                    dpr = float(vw.devicePixelRatioF())
                elif hasattr(vw, "devicePixelRatio"):
                    dpr = float(vw.devicePixelRatio())
        except Exception:
            pass
        return w, h, dpr

    def _compute_pan_scale(self, ctr):
        _, _, dpr = self._viewport_metrics()
        try:
            z = float(ctr.get_zoom())
        except Exception:
            z = 0.6

        zoom_factor = 0.6 / max(z, 0.02)
        scene_scale = self._estimate_scene_scale()
        scene_factor = max(0.2, min(scene_scale / 10.0, 3.0))
        scale = self._right_pan_scale * zoom_factor * scene_factor / max(1.0, dpr)
        return float(scale)

    def _estimate_scene_scale(self):
        if not self.scene.point_data:
            return 1.0
        max_extent = 0.0
        for data in self.scene.point_data.values():
            pos = data["pos"]
            if len(pos) > 0:
                extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                max_extent = max(max_extent, float(np.max(extent)))
        return max(max_extent, 0.1)