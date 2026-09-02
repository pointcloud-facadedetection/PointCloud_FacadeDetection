import ctypes
import platform
import subprocess
import time
from ctypes import wintypes

from PySide6.QtCore import QEvent, QObject, Qt


_WS_CHILD = 0x40000000
_WS_VISIBLE = 0x10000000
_WS_CLIPSIBLINGS = 0x04000000
_WS_CLIPCHILDREN = 0x02000000
_WS_POPUP = 0x80000000
_WS_CAPTION = 0x00C00000
_WS_THICKFRAME = 0x00040000
_GWL_STYLE = -16
_SWP_FRAMECHANGED = 0x0020
_SWP_NOZORDER = 0x0004
_SWP_SHOWWINDOW = 0x0040


def _user32():
    user32 = ctypes.windll.user32
    user32.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
    user32.SetParent.restype = wintypes.HWND
    user32.MoveWindow.argtypes = [
        wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL,
    ]
    user32.MoveWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = ctypes.c_longlong
        user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
        user32.SetWindowLongPtrW.restype = ctypes.c_longlong
    else:
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
    return user32


def _get_style(user32, hwnd):
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        return int(user32.GetWindowLongPtrW(hwnd, _GWL_STYLE))
    return int(user32.GetWindowLongW(hwnd, _GWL_STYLE))


def _set_style(user32, hwnd, style):
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        user32.SetWindowLongPtrW(hwnd, _GWL_STYLE, int(style))
    else:
        user32.SetWindowLongW(hwnd, _GWL_STYLE, int(style))


def resize_embedded_window(hwnd, container):
    """Keep the GLFW child window the same size as the Qt container."""
    if platform.system().lower() != 'windows' or hwnd is None or container is None:
        return
    try:
        width = max(1, int(container.width()))
        height = max(1, int(container.height()))
        _user32().MoveWindow(int(hwnd), 0, 0, width, height, True)
    except Exception:
        pass


def embed_as_child(hwnd, container):
    """Force the Open3D/GLFW HWND to be a child of the Qt container.

    QWidget.createWindowContainer() does not reliably reparent GLFW windows
    on Windows, so the native view can float above other apps or sit behind
    the main window.  SetParent + WS_CHILD keeps it inside the 3D viewport.
    """
    if platform.system().lower() != 'windows' or hwnd is None or container is None:
        return False
    try:
        container.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        parent = int(container.winId())
        child = int(hwnd)
        if parent == 0 or child == 0:
            return False
        user32 = _user32()
        user32.SetParent(child, parent)
        style = _get_style(user32, child)
        style = (
            style
            & ~_WS_POPUP
            & ~_WS_CAPTION
            & ~_WS_THICKFRAME
        ) | _WS_CHILD | _WS_VISIBLE | _WS_CLIPSIBLINGS | _WS_CLIPCHILDREN
        _set_style(user32, child, style)
        user32.SetWindowPos(
            child, 0, 0, 0,
            max(1, int(container.width())),
            max(1, int(container.height())),
            _SWP_FRAMECHANGED | _SWP_SHOWWINDOW | _SWP_NOZORDER,
        )
        resize_embedded_window(child, container)
        return True
    except Exception as exc:
        print(f'[PCFD] viewport.embed_failed reason={exc}', flush=True)
        return False


class NativeResizeWatcher(QObject):
    """Resize the embedded GLFW window when the Qt container changes size."""

    def __init__(self, hwnd, container):
        super().__init__(container)
        self._hwnd = int(hwnd)
        self._container = container

    def eventFilter(self, watched, event):
        if watched is self._container and event.type() in (QEvent.Resize, QEvent.Show):
            resize_embedded_window(self._hwnd, self._container)
        return False



class NativeWindowFinder:
    """Find native Open3D/GLFW windows so Qt can embed them."""

    def find(self, title, timeout=3.0, process_events=None):
        deadline = time.time() + timeout
        system = platform.system().lower()

        while time.time() < deadline:
            handle = None
            if system == "windows":
                handle = self._find_windows_handle(title)
            elif system == "linux":
                handle = self._find_linux_handle(title)
            elif system == "darwin":
                handle = self._find_macos_handle(title)

            if handle:
                return handle

            if process_events:
                process_events()
            time.sleep(0.05)

        return None

    def _find_windows_handle(self, title):
        user32 = ctypes.windll.user32
        handles = []

        enum_windows_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd, _lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                if title in buffer.value:
                    handles.append(hwnd)
                    return False
            return True

        user32.EnumWindows(enum_windows_proc(callback), 0)
        return handles[0] if handles else None

    def _find_linux_handle(self, title):
        commands = [
            ["xdotool", "search", "--name", title],
            ["wmctrl", "-l"],
        ]

        for cmd in commands:
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            except Exception:
                continue

            if cmd[0] == "xdotool":
                for line in out.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        return int(line)
            else:
                for line in out.splitlines():
                    if title in line:
                        return int(line.split()[0], 16)
        return None

    def _find_macos_handle(self, _title):
        # Qt embedding of Open3D/GLFW native windows is not stable on macOS.
        return None
