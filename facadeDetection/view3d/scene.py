import numpy as np
import open3d as o3d

from .geometry_factory import make_point_cloud
from .lod import display_arrays, normalize_colors


class PointCloudScene:

    MIN_POINT_SIZE = 0.01
    MAX_POINT_SIZE = 1.0

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
            "size": max(self.MIN_POINT_SIZE, min(float(point_size), self.MAX_POINT_SIZE)),
            "_render_sequence": 0,
        }
        self.active_name = name
        self.refresh_cloud(name, reset_bounding_box=reset_view or not had_clouds)

    def refresh_cloud(self, name, reset_bounding_box=False):
        if name not in self.point_data:
            return
        data = self.point_data[name]
        old = self.clouds.pop(name, None)
        if old is not None:
            self.adapter.remove_geometry(name)
            del old

        pos, colors = display_arrays(data)
        # Open3D 会将这些缓冲区复制到其几何体中。因此，我们在这里删除它们，以避免内存泄漏。
        pos = np.asarray(pos, dtype=np.float32, order='C')
        colors = np.asarray(colors, dtype=np.float32, order='C')
        try:
            pcd = make_point_cloud(pos, colors)
        finally:
            del pos, colors
        self.clouds[name] = pcd
        self.adapter.add_geometry(name, pcd, reset_bounding_box=reset_bounding_box)
        self.adapter.set_point_size(data["size"])

    def update_cloud_color(self, name, colors):
        if name not in self.point_data:
            return
        data = self.point_data[name]
        data["color"] = np.ascontiguousarray(normalize_colors(colors, len(data["pos"])).astype(np.float32))
        data.pop('display_proxy_lookup', None)
        geometry = self.clouds.get(name)
        if geometry is None:
            self.refresh_cloud(name, reset_bounding_box=False)
            return
        colors64 = np.asarray(data["color"], dtype=np.float64, order='C')
        try:
            geometry.colors = o3d.utility.Vector3dVector(colors64)
        finally:
            del colors64
        self.adapter.update_geometry(geometry)

    def update_cloud_points(self, name, positions, colors=None):
        if name not in self.point_data:
            return
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        data = self.point_data[name]
        data["pos"] = positions
        if colors is None:
            colors = data.get("color")
        data["color"] = np.ascontiguousarray(normalize_colors(colors, len(positions)).astype(np.float32))
        data.pop('display_proxy_lookup', None)
        self.active_name = name
        self.refresh_cloud(name, reset_bounding_box=False)

    def replace_cloud_snapshot(self, name, positions, colors=None, metadata=None):
        """Atomically replace displayed points and business metadata."""
        if name not in self.point_data:
            return False
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        data = self.point_data[name]
        data["pos"] = positions
        if colors is None:
            colors = data.get("color")
        data["color"] = np.ascontiguousarray(normalize_colors(colors, len(positions)).astype(np.float32))
        if metadata:
            data.update(metadata)
            data.pop('display_proxy_lookup', None)
        self.active_name = name
        self.refresh_cloud(name, reset_bounding_box=False)
        return True

    def commit_cloud_snapshot(self, name, positions, colors=None, metadata=None,
                              point_size=0.3, reset_view=False):
        """Create or atomically replace a cloud owned by the GUI thread."""
        if name not in self.point_data:
            self.add_cloud(name, positions, colors, point_size=point_size,
                           reset_view=reset_view)
            data = self.point_data.get(name)
            if data is not None and metadata:
                data.update(metadata)
                data.pop('display_proxy_lookup', None)
            return len(data.get('pos', [])) if data is not None else 0
        self.replace_cloud_snapshot(name, positions, colors, metadata)
        return len(self.point_data[name].get('pos', []))

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
            value = max(self.MIN_POINT_SIZE, min(float(size), self.MAX_POINT_SIZE))
            self.point_data[name]["size"] = value
            self.adapter.set_point_size(value)

    def get_cloud_names(self):
        return list(self.point_data.keys())

    def get_cloud_data(self, name):
        data = self.point_data.get(name)
        if data is not None and 'display_proxy_lookup' not in data:
            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
            if len(displayed) == len(data.get('pos', [])):
                data['display_proxy_lookup'] = {int(v): i for i, v in enumerate(displayed)}
        return data
