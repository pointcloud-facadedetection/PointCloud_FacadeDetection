import numpy as np

from .geometry_factory import make_point_cloud
from .lod import display_arrays, normalize_colors


class PointCloudScene:

    def __init__(self, adapter):
        self.adapter = adapter
        self.clouds = {}
        self.point_data = {}
        self.bbox_visible = {}
        self.normal_names = set()
        self.active_name = None

    def add_cloud(self, name, positions, colors=None, point_size=0.3, reset_view=False):
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        colors = np.ascontiguousarray(normalize_colors(colors, len(positions)).astype(np.float32))

        had_clouds = bool(self.point_data)
        self.point_data[name] = {
            "pos": positions,
            "color": colors,
            "size": float(point_size),
        }
        self.active_name = name
        self.refresh_cloud(name, reset_bounding_box=reset_view or not had_clouds)

    def refresh_cloud(self, name, reset_bounding_box=False):
        if name not in self.point_data:
            return
        data = self.point_data[name]
        pos, colors = display_arrays(data)
        pcd = make_point_cloud(pos, colors)
        self.clouds[name] = pcd
        self.adapter.add_geometry(name, pcd, reset_bounding_box=reset_bounding_box)
        self.adapter.set_point_size(data["size"])

    def update_cloud_color(self, name, colors):
        if name not in self.point_data:
            return
        data = self.point_data[name]
        data["color"] = np.ascontiguousarray(normalize_colors(colors, len(data["pos"])).astype(np.float32))
        self.refresh_cloud(name, reset_bounding_box=False)

    def update_cloud_points(self, name, positions, colors=None):
        if name not in self.point_data:
            return
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        data = self.point_data[name]
        data["pos"] = positions
        if colors is None:
            colors = data.get("color")
        data["color"] = np.ascontiguousarray(normalize_colors(colors, len(positions)).astype(np.float32))
        self.active_name = name
        self.refresh_cloud(name, reset_bounding_box=False)

    def remove_cloud(self, name):
        self.adapter.remove_geometry(name)
        self.adapter.remove_geometry(f"{name}__bbox")
        self.adapter.remove_geometry(f"{name}__normals")
        self.clouds.pop(name, None)
        self.point_data.pop(name, None)
        self.bbox_visible.pop(name, None)
        if self.active_name == name:
            self.active_name = next(iter(self.point_data), None)

    def clear(self):
        self.adapter.clear()
        self.clouds.clear()
        self.point_data.clear()
        self.bbox_visible.clear()
        self.normal_names.clear()
        self.active_name = None

    def set_point_size(self, name, size):
        if name in self.point_data:
            self.point_data[name]["size"] = float(size)
        self.adapter.set_point_size(size)

    def get_cloud_names(self):
        return list(self.point_data.keys())

    def get_cloud_data(self, name):
        return self.point_data.get(name)
