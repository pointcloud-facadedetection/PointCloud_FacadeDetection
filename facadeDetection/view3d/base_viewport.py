from abc import ABC, abstractmethod

import numpy as np


class BaseViewport(ABC):
    """Renderer-independent viewport contract."""

    @abstractmethod
    def get_widget(self):
        ...

    @abstractmethod
    def add_cloud(self, name: str, positions, colors=None, point_size=0.3):
        ...

    @abstractmethod
    def update_cloud_color(self, name: str, colors):
        ...

    @abstractmethod
    def update_cloud_points(self, name: str, positions, colors=None):
        ...

    @abstractmethod
    def remove_cloud(self, name: str):
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def get_cloud_names(self) -> list[str]:
        ...

    @abstractmethod
    def get_cloud_data(self, name: str) -> dict | None:
        ...

    @abstractmethod
    def set_point_size(self, name: str, size: float):
        ...

    @abstractmethod
    def set_all_point_size(self, size: float):
        ...

    @abstractmethod
    def toggle_bbox(self, name: str, min_bound, max_bound) -> bool:
        ...

    @abstractmethod
    def toggle_normals(self, name: str, normals, length=0.5, max_lines=8000) -> bool:
        ...

    @abstractmethod
    def set_selection_enabled(self, enabled: bool, cloud_name=None):
        ...

    @abstractmethod
    def set_pick_enabled(self, enabled: bool, radius=14, cloud_name=None):
        ...

    @abstractmethod
    def get_picked_point(self) -> np.ndarray | None:
        ...

    @abstractmethod
    def update_pick_markers(self, src_points=None, tgt_points=None):
        ...

    @abstractmethod
    def clear_pick_markers(self):
        ...

    @abstractmethod
    def save_screenshot(self, path: str):
        ...
