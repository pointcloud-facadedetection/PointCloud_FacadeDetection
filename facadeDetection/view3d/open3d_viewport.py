"""Open3D 视口在 PySide6 主窗口中的承载控件。

当前阶段只建立 UI 接口并预留渲染区域；点云加载、拾取和渲染逻辑
由后续的 view3d 任务继续实现。
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from .base_viewport import BaseViewport


class Open3DViewport(BaseViewport):
    def __init__(self):
        self._widget = QFrame()
        self._widget.setObjectName('open3dViewport')
        self._widget.setMinimumSize(480, 320)
        self._widget.setStyleSheet(
            "QFrame#open3dViewport {"
            "background-color: #151922;"
            "border: 1px solid #3a4252;"
            "}"
        )

        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel('3D Viewport\nOpen3D 三维视口位置已预留')
        label.setObjectName('viewportPlaceholder')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet('color: #aeb7c8; font-size: 18px; border: none;')
        layout.addWidget(label)

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
        return self._widget
