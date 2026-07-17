import numpy as np
import open3d as o3d


class Open3DAdapter:
    def __init__(self):
        self.vis = None
        self.geometries = {}

    def create_window(self, title, width=1280, height=960, visible=True):
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
        opt.background_color = np.array([0.04, 0.04, 0.10])
        opt.point_size = 2.0
        opt.show_coordinate_frame = False

    def add_geometry(self, name, geometry, reset_bounding_box=False):
        if self.vis is None:
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
        if self.vis is None:
            return
        geom = self.geometries.pop(name, None)
        if geom is not None:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass

    def clear(self):
        for name in list(self.geometries.keys()):
            self.remove_geometry(name)

    def set_point_size(self, size):
        if self.vis is not None:
            self.vis.get_render_option().point_size = float(size)

    def poll(self):
        if self.vis is None:
            return
        self.vis.poll_events()
        self.vis.update_renderer()

    def reset_view_point(self):
        if self.vis is not None:
            self.vis.reset_view_point(True)

    def get_view_control(self):
        if self.vis is None:
            return None
        return self.vis.get_view_control()

    def capture_screen(self):
        if self.vis is None:
            return None
        return self.vis.capture_screen_float_buffer(do_render=True)

    def destroy(self):
        try:
            if self.vis is not None:
                self.vis.destroy_window()
        finally:
            self.vis = None
            self.geometries.clear()
