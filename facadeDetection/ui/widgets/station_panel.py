from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget


class StationPanel(QWidget):
    """站点管理面板，业务动作通过信号交给 MainWindow。"""
    delete_requested = Signal()
    merge_requested = Signal()
    station_clicked = Signal(object)
    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('stationPanel')
        box = QGroupBox('站点管理')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        self.list = QListWidget()
        self.list.setObjectName('stationList')
        self.list.setWordWrap(False)
        # QListWidget emits QListWidgetItem*, while these public signals use
        # object for compatibility with service-layer payloads.  A lambda
        # performs the explicit Qt-to-Python signal adaptation; connecting the
        # signals directly raises a RuntimeError on PySide6.
        self.list.itemClicked.connect(lambda item: self.station_clicked.emit(item))
        self.list.itemChanged.connect(lambda item: self.selection_changed.emit(item))
        layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        delete = QPushButton('删除站点')
        merge = QPushButton('合并显示')
        delete.setProperty('buttonRole', 'danger')
        merge.setProperty('buttonRole', 'primary')
        delete.clicked.connect(self.delete_requested)
        merge.clicked.connect(self.merge_requested)
        buttons.addWidget(delete)
        buttons.addWidget(merge)
        layout.addLayout(buttons)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(box)

    def clear(self):
        self.list.clear()

    def add_item(self, item):
        self.list.addItem(item)