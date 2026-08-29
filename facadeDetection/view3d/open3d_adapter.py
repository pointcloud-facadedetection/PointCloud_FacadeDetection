import numpy as np
import open3d as o3d
import threading


class Open3DAdapter:
    MIN_POINT_SIZE = 0.01
    MAX_POINT_SIZE = 1.0
    MIN_POINT_PIXEL_SIZE = 0.5
    MAX_POINT_PIXEL_SIZE = 5.0

    def __init__(self):
        self.vis = None
        self.geometries = {}
        self._owner_thread_id = None
        self._destroyed = False

    def _assert_owner(self):
        """Visualizer/GLFW is single-threaded; fail early instead of racing WGL."""
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D visualization must run on the GUI thread')
        if self._destroyed:
            return False
        return self.vis is not None

    def create_window(self, title, width=1280, height=960, visible=True):
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D window must be created on the GUI thread')
        self._owner_thread_id = threading.get_ident()
        self._destroyed = False
        self.vis = o3d.visualization.Visualizer()
        ok = self.vis.create_window(
            window_name=title,
            width=width,
            height=height,
            visible=visible,
        )
        if not ok:
            raise RuntimeError("Open3D Visualizer create_window returned False")
        return self.vis

    def configure_render_options(self):
        if self.vis is None:
            return
        opt = self.vis.get_render_option()
        # Match the Corporate Clean viewport token (#111827).
        opt.background_color = np.array([17 / 255, 24 / 255, 39 / 255])
        opt.point_size = self.MIN_POINT_PIXEL_SIZE
        opt.show_coordinate_frame = True

    def add_geometry(self, name, geometry, reset_bounding_box=False):
        if not self._assert_owner():
            return
        old = self.geometries.get(name)
        if old is not None:
            try:
                self.vis.remove_geometry(old, reset_bounding_box=False)
            except Exception:
                pass
        self.geometries[name] = geometry
        self.vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)

    def remove_geometry(self, name):
        if not self._assert_owner():
            return
        geom = self.geometries.pop(name, None)
        if geom is not None:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass

    def clear(self):
        if not self._assert_owner():
            return
        for name in list(self.geometries.keys()):
            self.remove_geometry(name)

    def set_point_size(self, size):
        if self._assert_owner():
            # Map the normalized application value to Open3D's pixel range;
            # values below one pixel otherwise render indistinguishably.
            value = max(self.MIN_POINT_SIZE, min(float(size), self.MAX_POINT_SIZE))
            ratio = ((value - self.MIN_POINT_SIZE) /
                     (self.MAX_POINT_SIZE - self.MIN_POINT_SIZE))
            pixel_size = (self.MIN_POINT_PIXEL_SIZE + ratio *
                          (self.MAX_POINT_PIXEL_SIZE - self.MIN_POINT_PIXEL_SIZE))
            self.vis.get_render_option().point_size = pixel_size
            self.vis.update_renderer()

    def poll(self):
        if not self._assert_owner():
            return
        self.vis.poll_events()
        self.vis.update_renderer()

    def get_view_control(self):
        if not self._assert_owner():
            return None
        return self.vis.get_view_control()

    def capture_screen(self):
        if not self._assert_owner():
            return None
        return self.vis.capture_screen_float_buffer(do_render=True)

    def destroy(self):
        """Destroy the native window once, while GLFW is still available."""
        if self._destroyed:
            return
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D window must be destroyed on the GUI thread')
        vis = self.vis
        # Mark the adapter unavailable before entering Open3D.  Queued Qt
        # callbacks can therefore not touch a half-destroyed Visualizer.
        self._destroyed = True
        self.vis = None
        self.geometries.clear()
        try:
            if vis is not None:
                vis.destroy_window()
        except Exception:
            # 关闭阶段 Open3D 可能已经由 GLFW 清理了窗口；不能让异常
            # 阻止 Qt 主窗口和其余线程退出。
            pass
        finally:
            del vis
