import numpy as np
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
        self.pick_radius = 14
        self.pick_cloud = None
        self.point_pick_callback = None
        self.selection_enabled = False
        self.selection_cloud = None
        self.selection_callback = None
        self._selection_start = None
        self._pan_start = None
        self._click_start = None
        self._right_pan_scale = 0.003
        self.last_picked_point = None

    def set_selection_enabled(self, enabled, cloud_name=None):
        self.selection_enabled = bool(enabled)
        self.selection_cloud = cloud_name
        self._selection_start = None
        self._pan_start = None
        self._click_start = None

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        self.pick_enabled = bool(enabled)
        self.pick_radius = int(radius)
        self.pick_cloud = cloud_name
        self._click_start = None

    def handle_mouse_press(self, event):
        pos = self._event_pos(event)

        if self.selection_enabled and event.button() == Qt.LeftButton:
            self._selection_start = QPoint(pos)
            return True

        if event.button() == Qt.RightButton:
            self._pan_start = QPoint(pos)
            return True

        if self.pick_enabled and event.button() == Qt.LeftButton:
            self._click_start = QPoint(pos)
            return True

        return False

    def handle_mouse_move(self, event):
        if self._pan_start is None or not (event.buttons() & Qt.RightButton):
            return False

        pos = self._event_pos(event)
        dx = pos.x() - self._pan_start.x()
        dy = pos.y() - self._pan_start.y()
        self._pan_start = QPoint(pos)

        ctr = self.adapter.get_view_control()
        if ctr is not None:
            try:
                ctr.camera_local_translate(
                    -dx * self._right_pan_scale,
                    dy * self._right_pan_scale,
                    0.0,
                )
            except Exception:
                pass
        return True

    def handle_mouse_release(self, event):
        pos = self._event_pos(event)

        if self.selection_enabled and self._selection_start is not None and event.button() == Qt.LeftButton:
            start = self._selection_start
            self._selection_start = None
            self._click_start = None
            indices = self.select_indices_in_rect(self.selection_cloud, start, pos)
            if self.selection_callback:
                self.selection_callback(indices)
            return True

        if event.button() == Qt.RightButton:
            self._pan_start = None
            return True

        if self.pick_enabled and event.button() == Qt.LeftButton:
            start = self._click_start
            self._click_start = None

            if start is not None:
                moved = abs(pos.x() - start.x()) + abs(pos.y() - start.y())
                if moved <= 4:
                    picked = self.pick_nearest_point(pos, cloud_name=self.pick_cloud)
                    if picked:
                        self.last_picked_point = np.asarray(picked["point"], dtype=float)
                        if self.point_pick_callback:
                            self.point_pick_callback(picked)
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
            if dist[idx] < best_dist:
                point_index = idx * step
                best_dist = float(dist[idx])
                best = {
                    "cloud_name": name,
                    "index": point_index,
                    "point": points[point_index].astype(float).tolist(),
                }
        return best

    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()
