from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class PointCloudControls(QWidget):
    """项目操作页左侧的点云显示控制区。"""
    point_size_changed = Signal(float)
    reset_view_requested = Signal()
    change_color_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('pointCloudControls')
        box = QGroupBox('点云显示')
        box.setObjectName('pointCloudDisplayGroup')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        row = QHBoxLayout()
        reset = QPushButton('重置视图')
        color = QPushButton('改变颜色')
        reset.setObjectName('btn_reset_view_panel')
        color.setObjectName('btn_change_color_panel')
        reset.clicked.connect(self.reset_view_requested)
        color.clicked.connect(self.change_color_requested)
        row.addWidget(reset)
        row.addWidget(color)
        layout.addLayout(row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel('点云尺寸'))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(30)
        self.slider.setToolTip('调整视口中的点云显示尺寸')
        self.value_label = QLabel('0.30')
        self.value_label.setMinimumWidth(38)
        self.slider.valueChanged.connect(self._on_value_changed)
        size_row.addWidget(self.slider, 1)
        size_row.addWidget(self.value_label)
        layout.addLayout(size_row)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(box)

    def _on_value_changed(self, value):
        size = value / 100.0
        self.value_label.setText(f'{size:.2f}')
        self.point_size_changed.emit(size)