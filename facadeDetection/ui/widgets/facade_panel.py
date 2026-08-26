from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QListWidget, QGroupBox, QVBoxLayout,
    QPushButton, QLabel, QWidget
)


class FacadePanel(QWidget):
    """立面结果面板的通用容器，保留主窗口现有列表控件接口。"""
    evaluate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('facadePanel')
        
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        
        # ── 检测结果区域 ──
        result_group = QGroupBox('检测结果')
        result_group.setObjectName('resultGroup')
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 14, 10, 10)
        result_layout.setSpacing(8)
        
        self.summary = QLabel('未检测')
        self.summary.setObjectName('lblFacadeSummary')
        self.summary.setStyleSheet('color:#5b626d; font-size:13px;')
        self.summary.setMinimumHeight(28)
        self.summary.setWordWrap(True)
        result_layout.addWidget(self.summary)
        
        self.evaluate_button = QPushButton('评估选中立面')
        self.evaluate_button.setMinimumHeight(34)
        self.evaluate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        result_layout.addWidget(self.evaluate_button)
        
        self.list = QListWidget()
        self.list.setObjectName('lstFacades')
        self.list.setSpacing(4)
        self.list.setUniformItemSizes(False)
        result_layout.addWidget(self.list, 1)
        
        root.addWidget(result_group, 1)