from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QStyle, QStyleOptionViewItem
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget, QSizePolicy


class StationListWidget(QListWidget):
    """Separate checkbox clicks from station-content clicks."""

    station_content_clicked = Signal(object)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None and event.button() == Qt.MouseButton.LeftButton:
            # QListWidget does not expose QListView.viewOptions() in PySide6.
            # Build the same style option explicitly so the checkbox hit test
            # remains compatible with the installed Qt binding.
            option = QStyleOptionViewItem()
            option.initFrom(self.viewport())
            option.rect = self.visualItemRect(item)
            indicator = self.style().subElementRect(
                QStyle.SubElement.SE_ItemViewItemCheckIndicator,
                option,
                self,
            )
            if not indicator.contains(event.position().toPoint()):
                self.station_content_clicked.emit(item)
        super().mousePressEvent(event)


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
        self.list = StationListWidget()
        self.list.setObjectName('stationList')
        self.list.setWordWrap(False)
        self.list.setUniformItemSizes(True)
        self.list.setMinimumHeight(150)
        self.list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # QListWidget emits QListWidgetItem*, while these public signals use
        # object for compatibility with service-layer payloads.  A lambda
        # performs the explicit Qt-to-Python signal adaptation; connecting the
        # signals directly raises a RuntimeError on PySide6.
        self.list.station_content_clicked.connect(
            lambda item: self.station_clicked.emit(item)
        )
        self.list.itemChanged.connect(lambda item: self.selection_changed.emit(item))
        layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        delete = QPushButton('删除站点')
        merge = QPushButton('合并站点显示')
        for button in (delete, merge):
            button.setMinimumHeight(34)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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