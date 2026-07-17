import uuid

import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QImage, QWindow
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_viewport import BaseViewport

from .camera import CameraController
from .geometry_factory import make_bbox, make_normals, make_pair_lines, make_sphere
from .interaction import ViewportInteractor
from .open3d_adapter import Open3DAdapter
from .scene import PointCloudScene
from .window_embed import NativeWindowFinder


class _Open3DAppProxy:
    def __init__(self, owner):
        self.owner = owner

    def process_events(self):
        self.owner.process_events()


class _ContainerEventBridge(QObject):
    def __init__(self, interactor, parent=None):
        super().__init__(parent)
        self.interactor = interactor

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress:
            return self.interactor.handle_mouse_press(event)
        if event.type() == QEvent.MouseMove:
            return self.interactor.handle_mouse_move(event)
        if event.type() == QEvent.MouseButtonRelease:
            return self.interactor.handle_mouse_release(event)
        return super().eventFilter(watched, event)


class Open3DViewport(BaseViewport):
    """
    Open3D legacy Visualizer embedded in Qt.

    The widget remains the public API used by facade.ui.main_window, while
    rendering, LOD, camera, interaction and geometry helpers are split into
    focused modules under facade.ui.render.
    """

    def __init__(self, parent=None):
        self._root = QWidget(parent)
        self._layout = QVBoxLayout(self._root)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._adapter = Open3DAdapter()
        self._scene = PointCloudScene(self._adapter)
        self._camera = CameraController(self._adapter)
        self._interactor = ViewportInteractor(self._adapter, self._camera, self._scene)
        self._window_finder = NativeWindowFinder()

        self._container = None
        self._qwindow = None
        self._fallback_label = None
        self._event_bridge = None
        self._render_paused = False
        self._updating_geometry = False
        self._window_title = f"Open3DQtRenderer_{uuid.uuid4().hex}"
        self._pick_markers = []
        self._pick_lines = []

        self.native = self
        self.app = _Open3DAppProxy(self)

        self._init_ui()

        self._timer = QTimer(self._root)
        self._timer.timeout.connect(self.process_events)
        self._timer.start(16)

    @property
    def _clouds(self):
        return self._scene.clouds

    @property
    def _point_data(self):
        return self._scene.point_data

    @property
    def _bbox_visible(self):
        return self._scene.bbox_visible

    @property
    def _active_name(self):
        return self._scene.active_name

    @property
    def _vis(self):
        return self._adapter.vis

    @property
    def point_pick_callback(self):
        return self._interactor.point_pick_callback

    @point_pick_callback.setter
    def point_pick_callback(self, callback):
        self._interactor.point_pick_callback = callback

    @property
    def selection_callback(self):
        return self._interactor.selection_callback

    @selection_callback.setter
    def selection_callback(self, callback):
        self._interactor.selection_callback = callback

    @property
    def pick_enabled(self):
        return self._interactor.pick_enabled

    @pick_enabled.setter
    def pick_enabled(self, value):
        self._interactor.pick_enabled = bool(value)

    @property
    def selection_enabled(self):
        return self._interactor.selection_enabled

    @selection_enabled.setter
    def selection_enabled(self, value):
        self._interactor.selection_enabled = bool(value)

    def _init_ui(self):
        layout = self._layout

        try:
            self._adapter.create_window(self._window_title, width=1280, height=960, visible=True)
            handle = self._window_finder.find(
                self._window_title,
                timeout=3.0,
                process_events=self.process_events,
            )
            if handle is None:
                raise RuntimeError("Cannot locate Open3D native window handle")

            self._qwindow = QWindow.fromWinId(int(handle))
            self._container = QWidget.createWindowContainer(self._qwindow, self._root)
            self._container.setFocusPolicy(Qt.StrongFocus)
            self._container.setMouseTracking(True)
            self._event_bridge = _ContainerEventBridge(self._interactor, self._root)
            self._container.installEventFilter(self._event_bridge)
            self._qwindow.installEventFilter(self._event_bridge)
            layout.addWidget(self._container)

            self._camera.viewport_widget = self._container
            self._adapter.configure_render_options()
        except Exception as exc:
            self._fallback_label = QLabel(
                "Open3D render initialization failed.\n"
                "Please check GPU driver/OpenGL availability and native window embedding.\n\n"
                f"{type(exc).__name__}: {exc}"
            )
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "background:#0a0a1a;color:#f56565;font-size:14px;padding:24px;"
            )
            layout.addWidget(self._fallback_label)
            self._adapter.destroy()

    def process_events(self):
        if self._render_paused or self._updating_geometry:
            return
        try:
            self._adapter.poll()
        except Exception:
            pass

    def close(self):
        self.destroy()

    def destroy(self):
        self._adapter.destroy()

    def get_widget(self):
        return self._root

    def get_picked_point(self):
        return self._interactor.last_picked_point

    def add_cloud(self, name, positions, colors=None, point_size=0.3, reset_view=False):
        had_clouds = bool(self._scene.point_data)
        self._scene.add_cloud(name, positions, colors, point_size=point_size, reset_view=reset_view)
        if reset_view or not had_clouds:
            self.auto_range()

    def update_cloud_color(self, name, colors, preserve_view=True):
        if name not in self._scene.point_data:
            return
        state = self.get_camera_state() if preserve_view else None
        self._scene.update_cloud_color(name, colors)
        if preserve_view and state is not None:
            self.set_camera_state(state)

    def update_cloud_points(self, name, positions, colors=None, preserve_view=True):
        if name not in self._scene.point_data:
            return
        state = self.get_camera_state() if preserve_view else None
        self._scene.update_cloud_points(name, positions, colors)
        if preserve_view and state is not None:
            self.set_camera_state(state)

    def remove_cloud(self, name):
        self._scene.remove_cloud(name)
        if not self._scene.point_data:
            self.clear_pick_markers()

    def clear(self):
        self.clear_pick_markers()
        self._scene.clear()

    def set_point_size(self, name, size):
        self._scene.set_point_size(name, size)

    def set_all_point_size(self, size):
        for name in self._scene.get_cloud_names():
            self._scene.set_point_size(name, size)

    def toggle_bbox(self, name, min_bound, max_bound):
        bbox_name = f"{name}__bbox"
        visible = self._scene.bbox_visible.get(name, False)
        if visible:
            self._adapter.remove_geometry(bbox_name)
            self._scene.bbox_visible[name] = False
            return False

        self._adapter.add_geometry(bbox_name, make_bbox(min_bound, max_bound), reset_bounding_box=False)
        self._scene.bbox_visible[name] = True
        return True

    def toggle_normals(self, name, normals, length=0.5, max_lines=8000):
        normal_name = f"{name}__normals"
        if normal_name in self._scene.normal_names:
            self._adapter.remove_geometry(normal_name)
            self._scene.normal_names.remove(normal_name)
            return False

        data = self._scene.get_cloud_data(name)
        if data is None:
            return False

        lines = make_normals(data["pos"], normals, length=length, max_lines=max_lines)
        if lines is None:
            return False

        self._adapter.add_geometry(normal_name, lines, reset_bounding_box=False)
        self._scene.normal_names.add(normal_name)
        return True

    def auto_range(self):
        if self._scene.point_data:
            self._camera.auto_range()

    def reset_view(self):
        self.auto_range()

    def get_camera_state(self):
        return self._camera.get_state()

    def set_camera_state(self, state):
        self._camera.set_state(state)

    def get_cloud_names(self):
        return self._scene.get_cloud_names()

    def get_cloud_data(self, name):
        return self._scene.get_cloud_data(name)

    def set_selection_enabled(self, enabled, cloud_name=None):
        self._interactor.set_selection_enabled(enabled, cloud_name=cloud_name)

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        self._interactor.set_pick_enabled(enabled, radius=radius, cloud_name=cloud_name)

    def update_pick_markers(self, src_points=None, tgt_points=None):
        self.clear_pick_markers()

        src = np.asarray(src_points if src_points is not None else [], dtype=np.float64).reshape(-1, 3)
        tgt = np.asarray(tgt_points if tgt_points is not None else [], dtype=np.float64).reshape(-1, 3)
        radius = self._marker_radius()

        for i, point in enumerate(src):
            name = f"__pick_src_{i}"
            self._adapter.add_geometry(name, make_sphere(point, [1.0, 0.2, 0.2], radius), reset_bounding_box=False)
            self._pick_markers.append(name)

        for i, point in enumerate(tgt):
            name = f"__pick_tgt_{i}"
            self._adapter.add_geometry(name, make_sphere(point, [0.2, 1.0, 0.2], radius), reset_bounding_box=False)
            self._pick_markers.append(name)

        lines = make_pair_lines(src, tgt)
        if lines is not None:
            name = "__pick_pair_lines"
            self._adapter.add_geometry(name, lines, reset_bounding_box=False)
            self._pick_lines.append(name)

    def clear_pick_markers(self):
        for name in self._pick_markers + self._pick_lines:
            self._adapter.remove_geometry(name)
        self._pick_markers = []
        self._pick_lines = []

    def save_screenshot(self, path):
        image = self._adapter.capture_screen()
        if image is None:
            return
        arr = (np.asarray(image) * 255).clip(0, 255).astype(np.uint8)
        height, width, _ = arr.shape
        qimg = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888).copy()
        qimg.save(path)

    def _marker_radius(self):
        extents = []
        for data in self._scene.point_data.values():
            pos = data["pos"]
            if len(pos):
                extents.append(np.max(pos, axis=0) - np.min(pos, axis=0))
        if not extents:
            return 0.2
        max_dim = float(np.max(np.vstack(extents)))
        return max(max_dim * 0.005, 0.05)


# Backward-compatible name used by facade.ui.main_window.
Open3DPointCloudWidget = Open3DViewport