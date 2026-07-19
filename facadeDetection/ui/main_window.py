from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QToolButton,
    QWidget,
)

from .widgets.flow_layout import FlowLayout
from services.button_service import ButtonService
from view3d.open3d_viewport import Open3DViewport


HEADER_ACTIONS = (
    ('上传文件', 'upload_file'),
    ('点云去噪', 'point_cloud_denoise'),
    ('立面检测', 'facade_detection'),
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PointCloud FacadeDetection')
        self.resize(1600, 900)
        self.viewport = Open3DViewport()
        self.button_service = ButtonService()
        self.header_buttons = {}
        self._header_resize_pending = False
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

        (
            self.left_dock,
            self.left_sidebar_button,
        ) = self._create_sidebar(
            'Left Sidebar', 'leftDock', 'left', Qt.DockWidgetArea.LeftDockWidgetArea
        )
        (
            self.right_dock,
            self.right_sidebar_button,
        ) = self._create_sidebar(
            'Right Sidebar', 'rightDock', 'right', Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
        self.centralWidget().installEventFilter(self)
        self.header_dock = self._create_header()
        self.bottom_dock = self._create_bottom()

        self.resizeDocks(
            [self.left_dock, self.right_dock],
            [210, 210],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.bottom_dock],
            [80],
            Qt.Orientation.Vertical,
        )

    def _create_header(self):
        dock = QDockWidget('Header', self)
        dock.setObjectName('headerDock')
        dock.setAllowedAreas(Qt.DockWidgetArea.TopDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        panel = QWidget()
        panel.setObjectName('headerPanel')
        self.header_panel = panel
        panel.installEventFilter(self)
        self.header_layout = FlowLayout(
            panel,
            margin=10,
            horizontal_spacing=8,
            vertical_spacing=8,
        )

        for label, action_name in HEADER_ACTIONS:
            button = QPushButton(label)
            button.setObjectName(action_name)
            button.setMinimumSize(120, 34)
            self.header_buttons[action_name] = button
            self.header_layout.addWidget(button)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        QTimer.singleShot(0, self._resize_header_to_contents)
        return dock

    def _create_sidebar(self, title, object_name, side, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(area)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        title_bar = QWidget()
        title_bar.setObjectName(f'{object_name}TitleBar')
        title_bar.setStyleSheet(
            f'QWidget#{object_name}TitleBar {{ background-color: #d9d9d9; }}'
        )
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 4, 6, 4)

        title_label = QLabel(title)
        title_label.setObjectName(f'{object_name}TitleLabel')
        title_layout.addWidget(title_label)

        toggle_button = QToolButton()
        toggle_button.setObjectName(f'toggle_{side}_sidebar')
        toggle_button.setText('◀' if side == 'left' else '▶')
        label = '左侧栏' if side == 'left' else '右侧栏'
        toggle_button.setToolTip(f'收起{label}')
        toggle_button.setAccessibleName(f'收起{label}')
        toggle_button.setFixedSize(28, 28)
        toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        title_layout.addWidget(
            toggle_button,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        title_layout.addStretch(1)
        dock.setTitleBarWidget(title_bar)

        panel = QWidget()
        panel.setObjectName(f'{object_name}Panel')
        dock.setWidget(panel)
        dock.setMinimumWidth(180)
        dock.setMaximumWidth(260)
        dock.setProperty('expandedWidth', 210)
        self.addDockWidget(area, dock)
        return dock, toggle_button

    def _create_sidebar_expand_button(self, side):
        button = QToolButton(self.centralWidget())
        button.setObjectName(f'expand_{side}_sidebar')
        button.setText('▶' if side == 'left' else '◀')
        label = '左侧栏' if side == 'left' else '右侧栏'
        button.setToolTip(f'展开{label}')
        button.setAccessibleName(f'展开{label}')
        button.setFixedSize(30, 46)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.hide()
        return button

    def _create_bottom(self):
        """保留空的 Bottom 区域，当前阶段不放置功能控件。"""
        dock = QDockWidget('Bottom', self)
        dock.setObjectName('bottomDock')
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        panel = QWidget()
        panel.setObjectName('bottomDockPanel')
        dock.setWidget(panel)
        dock.setMinimumHeight(70)
        dock.setMaximumHeight(100)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def _connect_buttons(self):
        # objectName 与 service 方法名一一对应，当前只验证事件链，不接业务算法。
        for action_name, button in self.header_buttons.items():
            button.clicked.connect(getattr(self.button_service, action_name))

        self.left_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                self.right_dock,
                self.right_sidebar_expand_button,
            )
        )
        self.left_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        )
        self.right_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                self.right_dock,
                self.right_sidebar_expand_button,
            )
        )

    def _collapse_sidebar(self, dock, expand_button):
        dock.setProperty('expandedWidth', max(180, min(dock.width(), 260)))
        dock.hide()
        expand_button.show()
        expand_button.raise_()
        QTimer.singleShot(0, self._position_sidebar_expand_buttons)

    def _expand_sidebar(self, dock, expand_button):
        expand_button.hide()
        dock.show()
        target_width = int(dock.property('expandedWidth') or 210)
        QTimer.singleShot(
            0,
            lambda: self.resizeDocks(
                [dock],
                [target_width],
                Qt.Orientation.Horizontal,
            ),
        )

    def _position_sidebar_expand_buttons(self):
        viewport = self.centralWidget()
        top = 8
        if self.left_sidebar_expand_button.isVisible():
            self.left_sidebar_expand_button.move(0, top)
            self.left_sidebar_expand_button.raise_()
        if self.right_sidebar_expand_button.isVisible():
            self.right_sidebar_expand_button.move(
                max(0, viewport.width() - self.right_sidebar_expand_button.width()),
                top,
            )
            self.right_sidebar_expand_button.raise_()

    def eventFilter(self, watched, event):
        if watched is self.header_panel and event.type() == QEvent.Type.Resize:
            self._schedule_header_resize()
        if watched is self.centralWidget() and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            QTimer.singleShot(0, self._position_sidebar_expand_buttons)
        return super().eventFilter(watched, event)

    def _schedule_header_resize(self):
        if self._header_resize_pending:
            return
        self._header_resize_pending = True
        QTimer.singleShot(0, self._resize_header_to_contents)

    def _resize_header_to_contents(self):
        self._header_resize_pending = False
        if not hasattr(self, 'header_dock') or self.header_panel.width() <= 0:
            return

        content_height = self.header_layout.heightForWidth(self.header_panel.width())
        title_height = max(self.header_dock.height() - self.header_panel.height(), 16)
        target_height = max(70, min(content_height + title_height, 180))

        if (
            self.header_dock.minimumHeight() == target_height
            and self.header_dock.maximumHeight() == target_height
        ):
            return

        self.header_dock.setMinimumHeight(target_height)
        self.header_dock.setMaximumHeight(target_height)
        self.resizeDocks(
            [self.header_dock],
            [target_height],
            Qt.Orientation.Vertical,
        )
