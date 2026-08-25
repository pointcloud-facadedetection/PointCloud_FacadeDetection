from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget, QGroupBox, QVBoxLayout, QPushButton, QLabel, QWidget


class FacadePanel(QWidget):
    """立面结果面板的通用容器，保留主窗口现有列表控件接口。"""
    evaluate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('facadePanel')
        box = QGroupBox('立面检测结果')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        self.summary = QLabel('未检测')
        self.summary.setObjectName('lblFacadeSummary')
        self.evaluate_button = QPushButton('评估选中立面')
        self.list = QListWidget()
        self.list.setObjectName('lstFacades')
        layout.addWidget(self.summary)
        layout.addWidget(self.evaluate_button)
        layout.addWidget(self.list, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(box)