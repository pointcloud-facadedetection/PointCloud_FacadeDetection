from __future__ import annotations

import os
from typing import Optional, Callable, List, Tuple, Dict

import numpy as np


class ViewportRenderService:
    """
    封装视口的渲染交互逻辑，提供统一的 API 给上层服务使用。
    """

    def __init__(self, viewport, db):
        self.viewport = viewport
        self.db = db
        self._pick_mode = False
        self._picked_points = []

    # Notify render: show point cloud with optional colors
    def show_point_cloud(self, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        # The given viewport is expected to provide an API to add point data.
        # Our shipped Open3DViewport contains adapter+scene; we provide a thin
        # compatibility layer here: call add_point_cloud if available, else fallback.
        if hasattr(self.viewport, 'add_point_cloud'):
            self.viewport.add_point_cloud(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud_numpy'):
            self.viewport.add_cloud_numpy(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud'):
            # Some adapters expect Nx3 points and Nx3 colors in [0,1]
            self.viewport.add_cloud(name, points, colors)
        else:
            # As an absolutely minimal fallback, try to call a generic update method
            raise RuntimeError('Viewport does not support adding point cloud data')

        if hasattr(self.viewport, 'auto_range'):
            self.viewport.auto_range()

    def show_image(self, name: str, image: np.ndarray):
        if hasattr(self.viewport, 'show_image'):
            self.viewport.show_image(name, image)
            if hasattr(self.viewport, 'auto_range'):
                self.viewport.auto_range()
        else:
            raise RuntimeError('Viewport does not support image display')

    def get_widget(self):
        if hasattr(self.viewport, 'get_widget'):
            return self.viewport.get_widget()
        # A viewport may already be a QWidget
        if hasattr(self.viewport, 'native') and hasattr(self.viewport.native, 'widget'):
            return self.viewport.native.widget
        raise RuntimeError('Viewport has no Qt widget to embed')

    def enter_pick_mode(self, callback: Optional[Callable] = None,
                        cloud_name: Optional[str] = None,
                        pick_radius: int = 8):
        if not hasattr(self.viewport, 'enter_pick_mode'):
            raise RuntimeError('Viewport does not support pick mode')
        self.viewport.enter_pick_mode(
            cloud_name=cloud_name,
            pick_radius=pick_radius,
            callback=callback
        )
        self._pick_mode = True

    def exit_pick_mode(self):
        if hasattr(self.viewport, 'exit_pick_mode'):
            self.viewport.exit_pick_mode()
        self._pick_mode = False

    def is_pick_mode(self) -> bool:
        return self._pick_mode

    def add_pick_marker(self, point):
        """记录拾取点并刷新视口标记。"""
        self._picked_points.append(point)
        if hasattr(self.viewport, 'update_pick_markers'):
            self.viewport.update_pick_markers(src_points=self._picked_points)

    def clear_pick_markers(self):
        """清除所有拾取标记并重置记录。"""
        if hasattr(self.viewport, 'clear_pick_markers'):
            self.viewport.clear_pick_markers()
        self._picked_points.clear()