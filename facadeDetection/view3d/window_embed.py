import ctypes
import platform
import subprocess
import time
from ctypes import wintypes


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
