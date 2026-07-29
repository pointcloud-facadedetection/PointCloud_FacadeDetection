import numpy as np
import time
from PySide6.QtCore import QPoint, Qt

from .lod import sample_step


class ViewportInteractor:
    SELECT_SAMPLE_POINTS = 2_000_000
    PICK_SAMPLE_POINTS = 1_000_000

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
        self._pan_start = None
        self._click_start = None
        self._rotate_start = None
        self._right_pan_scale = 0.05       # 基础灵敏度
        self._left_rotate_scale = 1.0
        self._wheel_scale = 1.1
        self._panning = False
        self.last_picked_point = None
        self._last_pick_key = None
        self._last_pick_ts = 0.0
        self._cached_pan_scale = None      # 缓存 pan scale，避免 move 时重复计算

    def set_selection_enabled(self, enabled, cloud_name=None):
        self.selection_enabled = bool(enabled)
        self.selection_cloud = cloud_name
        self._selection_start = None
        self._pan_start = None
        self._click_start = None
        self._rotate_start = None

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        self.pick_enabled = bool(enabled)
        self.pick_radius = int(radius)
        self.pick_cloud = cloud_name
        self._click_start = None

    def handle_mouse_press(self, event):
        pos = self._event_pos(event)

        # 左键矩形选择
        if self.selection_enabled and event.button() == Qt.LeftButton:
            self._selection_start = QPoint(pos)
            return True

        # 左键旋转（非选择模式）
        if event.button() == Qt.LeftButton and not self.selection_enabled:
            self._rotate_start = QPoint(pos)
            if self.pick_enabled:
                self._click_start = QPoint(pos)
            return True

        # 右键平移
        if event.button() == Qt.RightButton:
            self._pan_start = QPoint(pos)
            self._panning = True
            ctr = self.adapter.get_view_control()
            self._cached_pan_scale = self._compute_pan_scale(ctr) if ctr is not None else self._right_pan_scale
            return True

        return False

    def handle_mouse_move(self, event):
        pos = self._event_pos(event)

        # 右键拖拽平移：严格检查右键是否持续按下，杜绝状态标志导致的误触发
        if self._pan_start is not None and (event.buttons() & Qt.RightButton):
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()
            self._pan_start = QPoint(pos)

            ctr = self.adapter.get_view_control()
            if ctr is not None:
                try:
                    scale = self._cached_pan_scale if self._cached_pan_scale is not None else self._compute_pan_scale(ctr)
                    # 推动模式：鼠标往哪拖，场景往哪走。
                    # camera_local_translate 中 x+ 为相机右移(场景左移)，y+ 为相机上移(场景下移)。
                    # 因此 dx 取反、dy 保持正值，实现鼠标与场景同向移动。
                    ctr.camera_local_translate(-dx * scale, dy * scale, 0.0)
                except Exception:
                    pass
            return True

        # 左键拖拽旋转：严格检查左键是否持续按下
        if self._rotate_start is not None and (event.buttons() & Qt.LeftButton) and not self.selection_enabled:
            dx = pos.x() - self._rotate_start.x()
            dy = pos.y() - self._rotate_start.y()
            self._rotate_start = QPoint(pos)

            ctr = self.adapter.get_view_control()
            if ctr is not None:
                try:
                    ctr.rotate(int(dx * self._left_rotate_scale), int(dy * self._left_rotate_scale))
                except Exception:
                    try:
                        ctr.camera_local_translate(
                            -dx * self._right_pan_scale * 0.5,
                            dy * self._right_pan_scale * 0.5,
                            0.0,
                        )
                    except Exception:
                        pass
            return True

        return False

    def handle_mouse_release(self, event):
        pos = self._event_pos(event)

        # 左键释放：矩形选择
        if self.selection_enabled and self._selection_start is not None and event.button() == Qt.LeftButton:
            start = self._selection_start
            self._selection_start = None
            self._click_start = None
            self._rotate_start = None
            indices = self.select_indices_in_rect(self.selection_cloud, start, pos)
            if self.selection_callback:
                self.selection_callback(indices)
            return True

        # 右键释放：结束平移
        if event.button() == Qt.RightButton:
            self._pan_start = None
            self._panning = False
            self._cached_pan_scale = None
            return True

        # 左键释放：拾取（在拾取模式下且基本无拖拽）
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
        name = cloud_name or self.scene.active_name
        data = self.scene.get_cloud_data(name)
        if data is None:
            return np.array([], dtype=np.int64)

        pos = data["pos"]
        if len(pos) == 0:
            return np.array([], dtype=np.int64)

        step = sample_step(len(pos), self.SELECT_SAMPLE_POINTS)
        sampled = pos[::step]
        projected = self.camera.project_points(sampled)
        if projected is None:
            return np.array([], dtype=np.int64)

        screen, valid = projected
        x1, x2 = sorted([start.x(), end.x()])
        y1, y2 = sorted([start.y(), end.y()])
        mask = (
            valid
            & (screen[:, 0] >= x1)
            & (screen[:, 0] <= x2)
            & (screen[:, 1] >= y1)
            & (screen[:, 1] <= y2)
        )
        return np.nonzero(mask)[0].astype(np.int64) * step

    def pick_nearest_point(self, pos, cloud_name=None):
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

            step = sample_step(len(points), self.PICK_SAMPLE_POINTS)
            sampled = points[::step]
            projected = self.camera.project_points(sampled)
            if projected is None:
                continue

            screen, valid = projected
            dist = np.hypot(screen[:, 0] - pos.x(), screen[:, 1] - pos.y())
            dist[~valid] = np.inf
            idx = int(np.argmin(dist))
            if np.isfinite(dist[idx]):
                d = float(dist[idx])
                if d < best_any_dist:
                    point_index_any = idx * step
                    best_any_dist = d
                    best_any = {
                        "cloud_name": name,
                        "index": point_index_any,
                        "point": points[point_index_any].astype(float).tolist(),
                    }
                if d < best_dist:
                    point_index = idx * step
                    best_dist = d
                    best = {
                        "cloud_name": name,
                        "index": point_index,
                        "point": points[point_index].astype(float).tolist(),
                    }
        return best or best_any

    def handle_wheel(self, event):
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

            steps = delta / 120.0
            factor = float(self._wheel_scale ** steps)

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
                        sign = -1.0 if delta > 0 else 1.0
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
        """根据相机 zoom 和场景尺度动态计算平移灵敏度。
        zoom 越小（拉远/视野越大），scale 越大，确保平移视觉速度与旋转体验持平。
        """
        _, _, dpr = self._viewport_metrics()
        try:
            z = float(ctr.get_zoom())
        except Exception:
            z = 0.6

        # zoom 因子：拉远时放大平移量（以 zoom=0.6 为基准）
        zoom_factor = 0.6 / max(z, 0.02)

        # 场景尺度因子：大场景需要更大的绝对平移量，但限制上限避免失控
        scene_scale = self._estimate_scene_scale()
        scene_factor = max(0.2, min(scene_scale / 10.0, 3.0))

        # 综合计算；DPI 过高时适当降低灵敏度
        scale = self._right_pan_scale * zoom_factor * scene_factor / max(1.0, dpr)
        return float(scale)

    def _estimate_scene_scale(self):
        """根据当前所有点云的空间范围估算场景尺度，用于自适应平移灵敏度。"""
        if not self.scene.point_data:
            return 1.0
        max_extent = 0.0
        for data in self.scene.point_data.values():
            pos = data["pos"]
            if len(pos) > 0:
                extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                max_extent = max(max_extent, float(np.max(extent)))
        return max(max_extent, 0.1)