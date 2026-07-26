from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
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

PAGE_DEFINITIONS = (
    ('点云操作', 'point_cloud'),
    ('项目信息', 'project_info'),
    ('检测结果', 'inspection_results'),
    ('报告预览', 'report_preview'),
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
        # 第一个页面保留真实 Open3D 视口，其余页面先搭建可切换的空白框架。
        self.setCentralWidget(self._create_page_stack())

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
        (
            self.left_sidebar_expand_dock,
            self.left_sidebar_expand_button,
        ) = self._create_sidebar_expand_control(
            'left', Qt.DockWidgetArea.LeftDockWidgetArea
        )
        (
            self.right_sidebar_expand_dock,
            self.right_sidebar_expand_button,
        ) = self._create_sidebar_expand_control(
            'right', Qt.DockWidgetArea.RightDockWidgetArea
        )
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

    def _create_page_stack(self):
        stack = QStackedWidget()
        stack.setObjectName('pageStack')
        self.page_widgets = []

        viewport_page = self.viewport.get_widget()
        viewport_page.setObjectName('pointCloudPage')
        stack.addWidget(viewport_page)
        self.page_widgets.append(viewport_page)

        for page_title, page_key in PAGE_DEFINITIONS[1:]:
            page = QWidget()
            page.setObjectName(f'{page_key}Page')
            layout = QVBoxLayout(page)
            layout.setContentsMargins(24, 24, 24, 24)

            label = QLabel(page_title)
            label.setObjectName(f'{page_key}PageTitle')
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                'color: #666666; font-size: 20px; font-weight: 600;'
            )
            layout.addWidget(label)
            stack.addWidget(page)
            self.page_widgets.append(page)

        return stack

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

    def _create_sidebar_expand_control(self, side, area):
        dock = QDockWidget('', self)
        dock.setObjectName(f'{side}SidebarExpandDock')
        dock.setAllowedAreas(area)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setMinimumWidth(34)
        dock.setMaximumWidth(34)

        title_bar = QWidget()
        title_bar.setFixedHeight(0)
        dock.setTitleBarWidget(title_bar)

        panel = QWidget()
        panel.setObjectName(f'{side}SidebarExpandPanel')
        panel.setStyleSheet(
            f'QWidget#{side}SidebarExpandPanel {{ background-color: #d9d9d9; }}'
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(0)

        button = QToolButton()
        button.setObjectName(f'expand_{side}_sidebar')
        button.setText('▶' if side == 'left' else '◀')
        label = '左侧栏' if side == 'left' else '右侧栏'
        button.setToolTip(f'展开{label}')
        button.setAccessibleName(f'展开{label}')
        button.setFixedSize(34, 46)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)

        dock.setWidget(panel)
        self.addDockWidget(area, dock)
        dock.hide()
        return dock, button

    def _create_bottom(self):
        """在页面底部提供四个互斥页面页签。"""
        dock = QDockWidget('Bottom', self)
        dock.setObjectName('bottomDock')
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        title_bar = QWidget()
        title_bar.setFixedHeight(0)
        dock.setTitleBarWidget(title_bar)

        panel = QWidget()
        panel.setObjectName('bottomDockPanel')
        panel.setStyleSheet(
            """
            QWidget#bottomDockPanel {
                background-color: #f2f3f5;
                border-top: 1px solid #c8c8c8;
            }
            QPushButton {
                min-width: 128px;
                min-height: 38px;
                padding: 0 18px;
                color: #333333;
                background-color: #ffffff;
                border: 1px solid #b8b8b8;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #eef4fb;
                border-color: #7f9fca;
            }
            QPushButton:checked {
                color: #ffffff;
                background-color: #2f5f98;
                border-color: #2f5f98;
            }
            """
        )
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.page_button_group = QButtonGroup(self)
        self.page_button_group.setExclusive(True)
        self.page_buttons = {}
        for index, (page_title, page_key) in enumerate(PAGE_DEFINITIONS):
            button = QPushButton(page_title)
            button.setObjectName(f'page_tab_{page_key}')
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setChecked(index == 0)
            button.clicked.connect(
                lambda _checked=False, page_index=index: self.set_current_page(
                    page_index
                )
            )
            self.page_button_group.addButton(button, index)
            self.page_buttons[page_key] = button
            layout.addWidget(button)
        layout.addStretch(1)

        dock.setWidget(panel)
        dock.setMinimumHeight(60)
        dock.setMaximumHeight(72)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def set_current_page(self, page_index):
        self.centralWidget().setCurrentIndex(page_index)

    def _connect_buttons(self):
        # objectName 与 service 方法名一一对应，当前只验证事件链，不接业务算法。
        for action_name, button in self.header_buttons.items():
            button.clicked.connect(getattr(self.button_service, action_name))

        self.left_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                self.left_dock,
                self.left_sidebar_expand_dock,
            )
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                self.right_dock,
                self.right_sidebar_expand_dock,
            )
        )
        self.left_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                self.left_dock,
                self.left_sidebar_expand_dock,
            )
        )
        self.right_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                self.right_dock,
                self.right_sidebar_expand_dock,
            )
        )

    def _collapse_sidebar(self, dock, expand_dock):
        dock.setProperty('expandedWidth', max(180, min(dock.width(), 260)))
        dock.hide()
        expand_dock.show()
        QTimer.singleShot(
            0,
            lambda: self.resizeDocks(
                [expand_dock],
                [34],
                Qt.Orientation.Horizontal,
            ),
        )

    def _expand_sidebar(self, dock, expand_dock):
        expand_dock.hide()
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

    def eventFilter(self, watched, event):
        if watched is self.header_panel and event.type() == QEvent.Type.Resize:
            self._schedule_header_resize()
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
