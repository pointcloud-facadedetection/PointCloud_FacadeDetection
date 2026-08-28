import numpy as np
import open3d as o3d
import threading


class Open3DAdapter:
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
        opt.point_size = 2.0
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
            self.vis.get_render_option().point_size = float(size)

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

    def capture_color_depth_camera(self, point_size=5.0):
        """在同一渲染帧捕获彩色图、相机深度和针孔相机参数。"""
        if not self._assert_owner():
            return None
        render_option = self.vis.get_render_option()
        old_point_size = float(render_option.point_size)
        try:
            render_option.point_size = float(point_size)
            self.vis.poll_events()
            self.vis.update_renderer()
            color = self.vis.capture_screen_float_buffer(do_render=True)
            depth = self.vis.capture_depth_float_buffer(do_render=False)
            camera = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            return color, depth, camera
        finally:
            render_option.point_size = old_point_size
            self.vis.update_renderer()

    def destroy(self):
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D window must be destroyed on the GUI thread')
        try:
            if self.vis is not None:
                self.vis.destroy_window()
        finally:
            self.vis = None
            self.geometries.clear()
            self._destroyed = True
