from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFileDialog,
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


PAGE_DEFINITIONS = (
    ('项目概览', 'project_overview'),
    ('项目操作', 'project_operation'),
    ('检测复核', 'inspection_review'),
    ('报告预览/导出', 'report_export'),
)

PAGE_BUTTON_NAMES = {
    'project_overview': 'btn_overview',
    'project_operation': 'btn_operation',
    'inspection_review': 'btn_inspection_review',
    'report_export': 'btn_report_export',
}

PAGE_HEADER_ACTIONS = {
    'project_overview': (
        ('上传文件', 'btn_upload', 'upload_files'),
    ),
    'project_operation': (
        ('重置视图', 'btn_reset_view', 'reset'),
        ('改变颜色', 'btn_change_color', 'change_colors'),
        ('点云去噪', 'btn_denoise', 'denoise'),
        ('点云配准', 'btn_registration', 'registration'),
        ('立面检测', 'btn_facade_detection', 'facade_detection'),
        ('质量检测', 'btn_quality_inspection', 'compute_quality'),
        ('框选分割', 'btn_box_segmentation', 'segmentation'),
        ('计算细节', 'btn_calculate_detail', 'compute_detail'),
        ('二维-三维对齐', 'btn_align_2d_3d', 'align_2d_3d'),
    ),
    'inspection_review': (),
    'report_export': (),
}

HEADER_ACTIONS = tuple(
    action
    for _page_title, page_key in PAGE_DEFINITIONS
    for action in PAGE_HEADER_ACTIONS[page_key]
)

UPLOAD_FILE_FILTER = (
    '项目支持文件 '
    '(*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls '
    '*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '点云文件 (*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls);;'
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '所有文件 (*)'
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PointCloud FacadeDetection')
        self.resize(1600, 900)
        self.viewport = Open3DViewport()
        self.button_service = ButtonService()
        self.header_buttons = {}
        self.header_button_methods = {}
        self._last_upload_directory = str(Path.home())
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
        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
        self.centralWidget().installEventFilter(self)
        self.header_dock = self._create_header()
        self.bottom_dock = self._create_bottom()
        self._update_header_actions(0)

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

        for page_title, page_key in PAGE_DEFINITIONS:
            if page_key == 'project_operation':
                page = self.viewport.get_widget()
            else:
                page = QWidget()
                layout = QVBoxLayout(page)
                layout.setContentsMargins(24, 24, 24, 24)

                label = QLabel(page_title)
                label.setObjectName(f'{page_key}PageTitle')
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(
                    'color: #666666; font-size: 20px; font-weight: 600;'
                )
                layout.addWidget(label)

            page.setObjectName(f'{page_key}Page')
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

        for label, button_name, action_name in HEADER_ACTIONS:
            button = QPushButton(label)
            button.setObjectName(button_name)
            button.setMinimumSize(120, 34)
            self.header_buttons[button_name] = button
            self.header_button_methods[button_name] = action_name
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
        toggle_button.setObjectName(f'btn_collapse_{side}_sidebar')
        toggle_button.setText('◀' if side == 'left' else '▶')
        label = 'Left Sidebar' if side == 'left' else 'Right Sidebar'
        toggle_button.setToolTip(f'Collapse {label}')
        toggle_button.setAccessibleName(f'Collapse {label}')
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
        button = QToolButton()
        button.setParent(self)
        button.setObjectName(f'btn_expand_{side}_sidebar')
        button.setText('▶' if side == 'left' else '◀')
        label = 'Left Sidebar' if side == 'left' else 'Right Sidebar'
        button.setToolTip(f'Expand {label}')
        button.setAccessibleName(f'Expand {label}')
        button.setFixedSize(30, 46)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        button.setStyleSheet(
            """
            QToolButton {
                color: #333333;
                background-color: #d9d9d9;
                border: 1px solid #a8a8a8;
                border-radius: 3px;
                font-size: 16px;
            }
            QToolButton:hover {
                background-color: #eeeeee;
                border-color: #777777;
            }
            """
        )
        button.hide()
        return button

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
            button.setObjectName(PAGE_BUTTON_NAMES[page_key])
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
        if not 0 <= page_index < len(PAGE_DEFINITIONS):
            return
        self.centralWidget().setCurrentIndex(page_index)
        self._update_header_actions(page_index)

    def _update_header_actions(self, page_index):
        page_key = PAGE_DEFINITIONS[page_index][1]
        visible_buttons = {
            button_name
            for _label, button_name, _action_name
            in PAGE_HEADER_ACTIONS[page_key]
        }
        for button_name, button in self.header_buttons.items():
            button.setVisible(button_name in visible_buttons)

        self.header_layout.invalidate()
        self.header_panel.updateGeometry()
        self._schedule_header_resize()

    def _connect_buttons(self):
        # Button object names identify UI controls; service names identify interfaces.
        for button_name, button in self.header_buttons.items():
            action_name = self.header_button_methods[button_name]
            if action_name == 'upload_files':
                button.clicked.connect(self._open_upload_file_dialog)
            else:
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

    def _open_upload_file_dialog(self):
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            '选择点云或图像文件',
            self._last_upload_directory,
            UPLOAD_FILE_FILTER,
        )
        if not file_paths:
            return

        self._last_upload_directory = str(Path(file_paths[0]).parent)
        self.button_service.upload_files(file_paths)

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
        viewport_top_left = viewport.mapTo(self, QPoint(0, 0))
        top = viewport_top_left.y() + 8
        margin = 8

        if self.left_sidebar_expand_button.isVisible():
            self.left_sidebar_expand_button.move(
                viewport_top_left.x() + margin,
                top,
            )
            self.left_sidebar_expand_button.raise_()

        if self.right_sidebar_expand_button.isVisible():
            self.right_sidebar_expand_button.move(
                viewport_top_left.x()
                + viewport.width()
                - self.right_sidebar_expand_button.width()
                - margin,
                top,
            )
            self.right_sidebar_expand_button.raise_()

    def eventFilter(self, watched, event):
        if watched is self.header_panel and event.type() == QEvent.Type.Resize:
            self._schedule_header_resize()
        if watched is self.centralWidget() and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.Move,
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
