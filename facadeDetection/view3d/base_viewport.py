from abc import ABC, abstractmethod

import numpy as np


class BaseViewport(ABC):

    @abstractmethod
    def load_cloud(self, pcd, name: str):
        ...

    @abstractmethod
    def remove_cloud(self, name: str):
        ...

    @abstractmethod
    def highlight_points(self, indices: np.ndarray):
        ...

    @abstractmethod
    def clear_highlight(self):
        ...

    @abstractmethod
    def set_camera(self, center, eye, up):
        ...

    @abstractmethod
    def get_picked_point(self) -> np.ndarray | None:
        ...

    @abstractmethod
    def get_widget(self):
        ...
