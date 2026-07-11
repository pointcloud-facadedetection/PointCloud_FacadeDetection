"""Minimal Open3D SceneWidget container for the PySide6 UI.

This module intentionally stops at the boundary required for the first UI
milestone: create an Open3D window, embed it in Qt, render a reference grid,
and shut it down safely. Point-cloud loading, picking and algorithm result
rendering remain reserved for the teammate who owns the later ``view3d`` work.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys
import uuid

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from .base_viewport import BaseViewport


class ViewportSignals(QObject):
    ready = Signal(bool, str)


class Open3DViewport(BaseViewport):
    """Embeds Open3D's native GLFW window without implementing interactions."""

    _application_initialized = False

    def __init__(self) -> None:
        self._host = QFrame()
        self._host.setObjectName("open3dHost")
        self._host.setStyleSheet("QFrame#open3dHost { background: #080814; border: none; }")
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel("正在创建 Open3D 原生视口…")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #6f7691; font-size: 14px;")
        self._layout.addWidget(self._placeholder)

        self.signals = ViewportSignals(self._host)
        self._timer = QTimer(self._host)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._render_frame)

        self._o3d_app = gui.Application.instance
        self._window_title = f"PointCloud FacadeDetection Open3D {os.getpid()} {uuid.uuid4().hex}"
        self._window: gui.Window | None = None
        self._scene_widget: gui.SceneWidget | None = None
        self._scene: rendering.Open3DScene | None = None
        self._foreign_window: QWindow | None = None
        self._container: QWidget | None = None
        self._native_handle: int | None = None
        self._native_size: tuple[int, int] | None = None
        self._started = False
        self._ready = False
        self._closing = False
        self._grid = self._make_grid(extent=10, step=1)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def start(self) -> bool:
        """Create and adopt the Open3D native window once."""
        if self._closing:
            return False
        if self._ready:
            return True
        if self._started:
            return False
        if os.environ.get("FACADE_DISABLE_OPEN3D") == "1":
            self._show_fallback("Open3D 初始化已由测试环境关闭")
            return False
        if sys.platform != "win32":
            self._show_fallback("当前原生嵌入适配器面向 Windows")
            return False

        self._started = True
        try:
            if not type(self)._application_initialized:
                self._o3d_app.initialize()
                type(self)._application_initialized = True

            self._window = self._o3d_app.create_window(self._window_title, 960, 640)
            self._scene_widget = gui.SceneWidget()
            self._scene = rendering.Open3DScene(self._window.renderer)
            self._scene_widget.scene = self._scene
            self._window.add_child(self._scene_widget)
            self._window.set_on_layout(self._layout_open3d_scene)
            self._configure_scene()

            self._native_handle = self._find_native_window(self._window_title)
            if not self._native_handle:
                raise RuntimeError("找不到 Open3D 的 Windows 窗口句柄")
            self._prepare_child_window(self._native_handle)
            self._foreign_window = QWindow.fromWinId(self._native_handle)
            if self._foreign_window is None:
                raise RuntimeError("Qt 无法接管 Open3D 原生窗口")

            self._container = QWidget.createWindowContainer(self._foreign_window, self._host)
            self._container.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._container.setMinimumSize(320, 240)
            self._layout.replaceWidget(self._placeholder, self._container)
            self._placeholder.hide()
            self._sync_native_window(force=True)

            self._ready = True
            self._timer.start()
            self._render_frame()
            self.signals.ready.emit(True, "Open3D 0.19 SceneWidget 已嵌入 PySide6")
            return True
        except Exception as exc:
            self._dispose_active_window()
            self._started = False
            self._show_fallback(f"Open3D 视口初始化失败：{exc}")
            return False

    def _layout_open3d_scene(self, layout_context) -> None:
        del layout_context
        if self._window is not None and self._scene_widget is not None:
            self._scene_widget.frame = self._window.content_rect

    def _configure_scene(self) -> None:
        if self._scene is None or self._scene_widget is None:
            return
        self._scene.set_background([0.031, 0.031, 0.078, 1.0])
        self._scene.show_skybox(False)
        self._scene.show_axes(False)
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 1.0
        self._scene.add_geometry("__reference_grid__", self._grid, material)
        bounds = self._grid.get_axis_aligned_bounding_box()
        self._scene_widget.setup_camera(55.0, bounds, np.asarray([0.0, 0.0, 0.0]))
        self._scene_widget.look_at([0.0, 0.0, 0.0], [12.5, -12.5, 10.0], [0.0, 0.0, 1.0])

    @staticmethod
    def _make_grid(extent: int, step: int) -> o3d.geometry.LineSet:
        points: list[list[float]] = []
        lines: list[list[int]] = []
        colors: list[list[float]] = []
        cursor = 0
        for value in range(-extent, extent + 1, step):
            points.extend(
                (
                    [float(value), float(-extent), 0.0],
                    [float(value), float(extent), 0.0],
                    [float(-extent), float(value), 0.0],
                    [float(extent), float(value), 0.0],
                )
            )
            lines.extend(([cursor, cursor + 1], [cursor + 2, cursor + 3]))
            color = [0.22, 0.25, 0.48] if value == 0 or value % 5 == 0 else [0.11, 0.12, 0.24]
            colors.extend((color, color))
            cursor += 4
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
        grid.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
        return grid

    @staticmethod
    def _find_native_window(title: str) -> int:
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        user32.FindWindowW.restype = wintypes.HWND
        handle = int(user32.FindWindowW(None, title) or 0)
        if not handle:
            return 0
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowThreadProcessId(wintypes.HWND(handle), ctypes.byref(process_id))
        class_name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(wintypes.HWND(handle), class_name, len(class_name))
        if process_id.value != os.getpid() or class_name.value != "GLFW30":
            return 0
        return handle

    @staticmethod
    def _prepare_child_window(handle: int) -> None:
        user32 = ctypes.windll.user32
        gwl_style = -16
        ws_child = 0x40000000
        top_level_bits = 0x80000000 | 0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            getter = user32.GetWindowLongPtrW
            setter = user32.SetWindowLongPtrW
            long_ptr = ctypes.c_ssize_t
        else:
            getter = user32.GetWindowLongW
            setter = user32.SetWindowLongW
            long_ptr = ctypes.c_long
        getter.argtypes = (wintypes.HWND, ctypes.c_int)
        getter.restype = long_ptr
        setter.argtypes = (wintypes.HWND, ctypes.c_int, long_ptr)
        setter.restype = long_ptr
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        hwnd = wintypes.HWND(handle)
        style = int(getter(hwnd, gwl_style))
        setter(hwnd, gwl_style, (style & ~top_level_bits) | ws_child)
        user32.SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020,
        )

    def _sync_native_window(self, force: bool = False) -> None:
        if self._native_handle is None or self._container is None or self._foreign_window is None:
            return
        width = max(1, self._container.width())
        height = max(1, self._container.height())
        size = (width, height)
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = (wintypes.HWND,)
        user32.GetParent.restype = wintypes.HWND
        user32.SetParent.argtypes = (wintypes.HWND, wintypes.HWND)
        user32.SetParent.restype = wintypes.HWND
        native = wintypes.HWND(self._native_handle)
        parent = wintypes.HWND(int(self._container.winId()))
        if force or int(user32.GetParent(native) or 0) != int(self._container.winId()):
            user32.SetParent(native, parent)
            if int(user32.GetParent(native) or 0) != int(self._container.winId()):
                raise RuntimeError("Open3D 原生窗口未能嵌入 Qt 容器")
            force = True
        if force or size != self._native_size:
            # QWindow geometry uses the same device-independent coordinates as
            # QWidget; Qt applies the monitor scale to both native children.
            self._foreign_window.setGeometry(0, 0, width, height)
            self._native_size = size

    def _render_frame(self) -> None:
        if self._closing or not self._ready:
            return
        try:
            self._sync_native_window()
            if not self._o3d_app.run_one_tick():
                self._handle_native_window_closed()
        except RuntimeError as exc:
            self._timer.stop()
            self._ready = False
            self.signals.ready.emit(False, f"Open3D 渲染循环已停止：{exc}")

    def _handle_native_window_closed(self) -> None:
        self._timer.stop()
        self._ready = False
        self._started = False
        self._window = None
        self._scene_widget = None
        self._scene = None
        self._release_qt_wrapper()
        self._native_handle = None
        self._native_size = None
        self._show_fallback("Open3D 原生窗口已关闭")

    def _show_fallback(self, message: str) -> None:
        if self._layout.indexOf(self._placeholder) < 0:
            self._layout.addWidget(self._placeholder)
        self._placeholder.setText(f"Open3D 视口暂不可用\n\n{message}\n\n主窗口与 UI 控件仍可正常检查。")
        self._placeholder.setStyleSheet("color: #ed8936; font-size: 13px;")
        self._placeholder.show()
        self.signals.ready.emit(False, message)

    def reset_view(self) -> None:
        if self._ready and self._scene_widget is not None:
            bounds = self._grid.get_axis_aligned_bounding_box()
            self._scene_widget.setup_camera(55.0, bounds, np.asarray([0.0, 0.0, 0.0]))
            self._scene_widget.look_at([0.0, 0.0, 0.0], [12.5, -12.5, 10.0], [0.0, 0.0, 1.0])

    def get_widget(self):
        return self._host

    def load_cloud(self, pcd, name: str):
        del pcd, name
        raise NotImplementedError("点云加载由后续 view3d 协作任务实现")

    def remove_cloud(self, name: str):
        del name
        raise NotImplementedError("点云移除由后续 view3d 协作任务实现")

    def highlight_points(self, indices: np.ndarray):
        del indices
        raise NotImplementedError("点选高亮由后续 view3d 协作任务实现")

    def clear_highlight(self):
        raise NotImplementedError("点选高亮由后续 view3d 协作任务实现")

    def set_camera(self, center, eye, up):
        del center, eye, up
        raise NotImplementedError("相机服务接口由后续 view3d 协作任务实现")

    def get_picked_point(self):
        return None

    def shutdown(self) -> None:
        self._closing = True
        self._timer.stop()
        self._ready = False
        self._dispose_active_window()
        self._started = False

    def close(self) -> None:
        self.shutdown()

    def _dispose_active_window(self) -> None:
        scene = self._scene
        window = self._window
        self._scene = None
        self._scene_widget = None
        self._window = None
        if scene is not None:
            try:
                scene.clear_geometry()
            except RuntimeError:
                pass
        if window is not None:
            try:
                window.close()
                self._o3d_app.run_one_tick()
            except RuntimeError:
                pass
        self._release_qt_wrapper()
        self._native_handle = None
        self._native_size = None

    def _release_qt_wrapper(self) -> None:
        if self._container is not None:
            self._layout.removeWidget(self._container)
            self._container.deleteLater()
        self._container = None
        self._foreign_window = None
