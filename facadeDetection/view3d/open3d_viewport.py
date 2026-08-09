import sys
import ctypes
import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Qt, QAbstractNativeEventFilter, QCoreApplication, QPoint
from PySide6.QtGui import QImage, QWindow
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_viewport import BaseViewport
from .camera import CameraController
from .geometry_factory import (
    make_bbox, make_normals, make_pair_lines, make_sphere,
)
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
        try:
            et = event.type()
            if et == QEvent.MouseButtonPress:
                return self.interactor.handle_mouse_press(event)
            if et == QEvent.MouseMove:
                return self.interactor.handle_mouse_move(event)
            if et == QEvent.MouseButtonRelease:
                return self.interactor.handle_mouse_release(event)
            if et == QEvent.Wheel:
                return self.interactor.handle_wheel(event)
        except Exception:
            pass
        return super().eventFilter(watched, event)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _WinNativeMouseBridge(QAbstractNativeEventFilter):
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    WM_MOUSEWHEEL = 0x020A
    MK_LBUTTON = 0x0001
    MK_RBUTTON = 0x0002

    class MSG(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("message", ctypes.c_uint),
            ("wParam", ctypes.c_size_t),
            ("lParam", ctypes.c_size_t),
            ("time", ctypes.c_uint),
            ("pt", POINT),
        ]

    def __init__(self, hwnd, interactor, dpr_provider=None):
        super().__init__()
        self._hwnd = int(hwnd)
        self._interactor = interactor
        self._dpr_provider = dpr_provider

    def _dpr(self):
        try:
            if callable(self._dpr_provider):
                dpr = float(self._dpr_provider())
                return dpr if dpr else 1.0
        except Exception:
            pass
        return 1.0

    class _SimpleMouseEvent:
        def __init__(self, x, y, button, buttons):
            self._pos = QPoint(x, y)
            self._button = button
            self._buttons = buttons

        def button(self):
            return self._button

        def buttons(self):
            return self._buttons

        def position(self):
            class _P:
                def __init__(self, p): self._p = p
                def toPoint(self): return self._p
            return _P(self._pos)

        def pos(self):
            return self._pos

    class _SimpleWheelEvent:
        def __init__(self, x, y, delta):
            self._pos = QPoint(x, y)
            self._delta = int(delta)

        def angleDelta(self):
            return QPoint(0, self._delta)

        def pixelDelta(self):
            return QPoint(0, 0)

        def position(self):
            class _P:
                def __init__(self, p): self._p = p
                def toPoint(self): return self._p
            return _P(self._pos)

        def pos(self):
            return self._pos

    def nativeEventFilter(self, eventType, message):
        try:
            et = eventType.decode() if isinstance(eventType, (bytes, bytearray)) else eventType
            if et != "windows_generic_MSG":
                return False, 0
            try:
                addr = int(message)
            except Exception:
                return False, 0

            msg = ctypes.cast(addr, ctypes.POINTER(self.MSG)).contents
            if int(msg.hwnd) != self._hwnd:
                return False, 0

            m = msg.message
            x = msg.lParam & 0xFFFF
            y = (msg.lParam >> 16) & 0xFFFF
            if x & 0x8000: x = x - 0x10000
            if y & 0x8000: y = y - 0x10000

            dpr = self._dpr()
            if dpr and dpr != 1.0:
                x = int(x / dpr)
                y = int(y / dpr)

            buttons = Qt.NoButton
            if (msg.wParam & self.MK_LBUTTON) != 0:
                buttons |= Qt.LeftButton
            if (msg.wParam & self.MK_RBUTTON) != 0:
                buttons |= Qt.RightButton

            if m == self.WM_LBUTTONDOWN:
                ev = self._SimpleMouseEvent(x, y, Qt.LeftButton, Qt.LeftButton)
                self._interactor.handle_mouse_press(ev)
                return False, 0
            if m == self.WM_LBUTTONUP:
                ev = self._SimpleMouseEvent(x, y, Qt.LeftButton, Qt.NoButton)
                self._interactor.handle_mouse_release(ev)
                return False, 0
            if m == self.WM_RBUTTONDOWN:
                ev = self._SimpleMouseEvent(x, y, Qt.RightButton, Qt.RightButton)
                self._interactor.handle_mouse_press(ev)
                return False, 0
            if m == self.WM_RBUTTONUP:
                ev = self._SimpleMouseEvent(x, y, Qt.RightButton, Qt.NoButton)
                self._interactor.handle_mouse_release(ev)
                return False, 0
            if m == self.WM_MOUSEMOVE:
                ev = self._SimpleMouseEvent(x, y, Qt.NoButton, buttons)
                self._interactor.handle_mouse_move(ev)
                return False, 0
            if m == self.WM_MOUSEWHEEL:
                delta = ctypes.c_short((msg.wParam >> 16) & 0xFFFF).value
                ev = self._SimpleWheelEvent(x, y, delta)
                self._interactor.handle_wheel(ev)
                return False, 0
        except Exception:
            pass
        return False, 0


class Open3DViewport(BaseViewport):
    def __init__(self, parent=None):
        self._root = QWidget(parent)
        self._root.setObjectName('open3dViewport')
        self._root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        self._window_title = "PointCloud FacadeDetection"
        self._pick_markers = []
        self._pick_lines = []
        self._init_success = False
        self._grid_visible = True

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

            if sys.platform == "win32":
                def _get_dpr():
                    try:
                        if hasattr(self._container, "devicePixelRatioF"):
                            return self._container.devicePixelRatioF()
                        if hasattr(self._container, "devicePixelRatio"):
                            return self._container.devicePixelRatio()
                    except Exception:
                        pass
                    return 1.0
                self._native_mouse_bridge = _WinNativeMouseBridge(
                    handle, self._interactor, dpr_provider=_get_dpr
                )
                app = QCoreApplication.instance()
                if app is not None:
                    app.installNativeEventFilter(self._native_mouse_bridge)

            self._camera.viewport_widget = self._container
            self._adapter.configure_render_options()
            self._init_success = True
            self._setup_initial_camera()

        except Exception as exc:
            import traceback
            error_msg = f"Open3D render initialization failed.\\n\\n{type(exc).__name__}: {exc}\\n\\n{traceback.format_exc()}"
            self._fallback_label = QLabel(error_msg)
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "background:#111827;color:#fca5a5;font-size:12px;padding:24px;"
            )
            layout.addWidget(self._fallback_label)
            self._adapter.destroy()
            self._init_success = False

    def _setup_initial_camera(self):
        try:
            ctr = self._adapter.get_view_control()
            if ctr is None:
                return
            front = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            front = front / np.linalg.norm(front)
            ctr.set_lookat(np.array([0.0, 0.0, 0.0], dtype=np.float64))
            ctr.set_front(front)
            ctr.set_up(np.array([0.0, 1.0, 0.0], dtype=np.float64))
            ctr.set_zoom(0.6)
        except Exception:
            pass

    def process_events(self):
        if not self._init_success or self._render_paused:
            return
        try:
            self._adapter.poll()
        except Exception:
            pass

    def close(self):
        self.destroy()

    def destroy(self):
        self._init_success = False
        if hasattr(self, '_timer') and self._timer is not None:
            self._timer.stop()
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
        else:
            self._setup_initial_camera()

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

    def enter_pick_mode(self, cloud_name=None, pick_radius=8, callback=None):
        self.set_selection_enabled(False)
        self.set_pick_enabled(True, radius=pick_radius, cloud_name=cloud_name)
        if callback is not None:
            self.point_pick_callback = callback

    def exit_pick_mode(self):
        self.set_pick_enabled(False)

    def update_pick_markers(self, src_points=None, tgt_points=None):
        self.clear_pick_markers()
        src = np.asarray(src_points if src_points is not None else [], dtype=np.float64).reshape(-1, 3)
        tgt = np.asarray(tgt_points if tgt_points is not None else [], dtype=np.float64).reshape(-1, 3)

        r_src = self._marker_radius_for_points(src) if len(src) else None
        r_tgt = self._marker_radius_for_points(tgt) if len(tgt) else None

        for i, point in enumerate(src):
            name = f"__pick_src_{i}"
            try:
                self._adapter.add_geometry(name, make_sphere(point, [1.0, 0.2, 0.2], r_src or 0.08), reset_bounding_box=False)
                self._pick_markers.append(name)
            except Exception:
                pass

        for i, point in enumerate(tgt):
            name = f"__pick_tgt_{i}"
            try:
                self._adapter.add_geometry(name, make_sphere(point, [0.2, 1.0, 0.2], r_tgt or 0.08), reset_bounding_box=False)
                self._pick_markers.append(name)
            except Exception:
                pass

        lines = make_pair_lines(src, tgt)
        if lines is not None:
            name = "__pick_pair_lines"
            try:
                self._adapter.add_geometry(name, lines, reset_bounding_box=False)
                self._pick_lines.append(name)
            except Exception:
                pass
        try:
            self._adapter.poll()
        except Exception:
            pass

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

    def _marker_radius_for_points(self, pts: np.ndarray | None, pixel_radius: float = 6.0):
        try:
            if pts is None or len(pts) == 0:
                return max(self._fallback_scene_scale() * 0.002, 0.05)
            ctr = self._adapter.get_view_control()
            if ctr is None:
                return 0.08
            params = ctr.convert_to_pinhole_camera_parameters()
            fx = float(params.intrinsic.intrinsic_matrix[0, 0])
            screen, valid = self._camera.project_points(pts)
            if screen is None:
                return 0.08
            z = screen[:, 2][valid]
            if len(z) == 0:
                return 0.08
            z_avg = float(np.median(z))
            world_per_pixel = z_avg / max(fx, 1e-6)
            r = world_per_pixel * float(pixel_radius)
            return float(max(0.02, min(r, 0.5)))
        except Exception:
            return 0.08

    def _fallback_scene_scale(self):
        extents = []
        for data in self._scene.point_data.values():
            pos = data["pos"]
            if len(pos):
                extents.append(np.max(pos, axis=0) - np.min(pos, axis=0))
        if not extents:
            return 1.0
        max_dim = float(np.max(np.vstack(extents)))
        return max_dim if max_dim > 0 else 1.0
