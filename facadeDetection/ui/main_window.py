from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ..services.button_service import ButtonService
    from ..view3d.open3d_viewport import Open3DViewport
except ImportError:
    from services.button_service import ButtonService
    from view3d.open3d_viewport import Open3DViewport


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PointCloud FacadeDetection')
        self.resize(1600, 900)
        self.viewport = Open3DViewport()
        self.button_service = ButtonService()
        self._setup_ui()
        self._connect_buttons()

    def _setup_ui(self):
        # 中央区域来自 view3d，后续真实 Open3D 渲染接入时无需改主窗口布局。
        self.setCentralWidget(self.viewport.get_widget())

        self.setDockNestingEnabled(False)
        self.setCorner(Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.TopDockWidgetArea)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.TopDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

        self.header_dock, self.button1 = self._create_dock(
            'Header', 'headerDock', 'button1', Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.bottom_dock, self.button2 = self._create_dock(
            'Bottom', 'bottomDock', 'button2', Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.left_dock, self.button3 = self._create_dock(
            'Left Sidebar', 'leftDock', 'button3', Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.right_dock, self.button4 = self._create_dock(
            'Right Sidebar', 'rightDock', 'button4', Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.resizeDocks(
            [self.header_dock, self.bottom_dock],
            [90, 90],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.left_dock, self.right_dock],
            [210, 210],
            Qt.Orientation.Horizontal,
        )

    def _create_dock(self, title, object_name, button_name, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(area)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        panel = QWidget()
        panel.setObjectName(f'{object_name}Panel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)

        button = QPushButton(button_name)
        button.setObjectName(button_name)
        button.setMinimumSize(110, 34)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)

        dock.setWidget(panel)
        self.addDockWidget(area, dock)

        if area in (
            Qt.DockWidgetArea.TopDockWidgetArea,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        ):
            dock.setMinimumHeight(80)
            dock.setMaximumHeight(110)
        else:
            dock.setMinimumWidth(180)
            dock.setMaximumWidth(260)

        return dock, button

    def _connect_buttons(self):
        # 按钮名与 service 方法名一一对应，便于之后替换成真实业务操作。
        self.button1.clicked.connect(self.button_service.button1)
        self.button2.clicked.connect(self.button_service.button2)
        self.button3.clicked.connect(self.button_service.button3)
        self.button4.clicked.connect(self.button_service.button4)
