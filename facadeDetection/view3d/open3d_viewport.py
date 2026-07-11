from .base_viewport import BaseViewport


class Open3DViewport(BaseViewport):
    def __init__(self):
        # TODO: init open3d.visualization.gui.Application
        pass

    def load_cloud(self, pcd, name: str):
        pass

    def remove_cloud(self, name: str):
        pass

    def highlight_points(self, indices):
        pass

    def clear_highlight(self):
        pass

    def set_camera(self, center, eye, up):
        pass

    def get_picked_point(self):
        return None

    def get_widget(self):
        return None
