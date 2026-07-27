from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .widgets.flow_layout import FlowLayout
from services.file_service import FileService
from services.pointcloud_service import PointCloudService
from services.project_service import ProjectService
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
        ('上传文件', 'btn_upload'),
        ('打开项目', 'btn_open_project'),
        ('选择项目', 'btn_select_project'),
        ('新建项目', 'btn_new_project'),
    ),
    'project_operation': (
        ('重置视图', 'btn_reset_view'),
        ('改变颜色', 'btn_change_color'),
        ('点云去噪', 'btn_denoise'),
        ('点云配准', 'btn_registration'),
        ('立面检测', 'btn_facade_detection'),
        ('质量检测', 'btn_quality_inspection'),
        ('框选分割', 'btn_box_segmentation'),
        ('计算细节', 'btn_calculate_detail'),
        ('二维-三维对齐', 'btn_align_2d_3d'),
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
        self.file_service = FileService()
        self.project_service = ProjectService()
        self.pointcloud_service = PointCloudService(self.viewport)
        self.current_project = None
        self.header_buttons = {}
        self.page_title_labels = {}
        self._sidebar_collapsed = {'left': False, 'right': False}
        self._last_upload_directory = str(Path.home())
        self._header_resize_pending = False
        self._setup_ui()
        self._connect_buttons()
        self._refresh_project_list()
        self._set_current_project(None)

    def _setup_ui(self):
        # 页面按整页切换；三维视口只属于“项目操作”页面。
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
        self.set_current_page(0)

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
            if page_key == 'project_overview':
                page = self._create_project_overview_page(page_title, page_key)
            elif page_key == 'project_operation':
                page = self._create_operation_page(page_title, page_key)
            else:
                page = self._create_placeholder_page(page_title, page_key)

            page.setObjectName(f'{page_key}Page')
            stack.addWidget(page)
            self.page_widgets.append(page)

        return stack

    def _create_page_title(self, page_title, page_key):
        label = QLabel(page_title)
        label.setObjectName(f'{page_key}PageTitle')
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        label.setStyleSheet(
            'color: #2f3440; font-size: 20px; font-weight: 600; padding: 2px 0;'
        )
        self.page_title_labels[page_key] = label
        return label

    def _create_project_overview_page(self, page_title, page_key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        layout.addWidget(self._create_page_title(page_title, page_key))

        description = QLabel('历史项目')
        description.setObjectName('projectListSectionTitle')
        description.setStyleSheet(
            'color: #565d69; font-size: 14px; font-weight: 600;'
        )
        layout.addWidget(description)

        scroll_area = QScrollArea()
        scroll_area.setObjectName('projectListScrollArea')
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.project_list_container = QWidget()
        self.project_list_container.setObjectName('projectListContainer')
        self.project_list_layout = QVBoxLayout(self.project_list_container)
        self.project_list_layout.setContentsMargins(0, 0, 0, 0)
        self.project_list_layout.setSpacing(10)
        self.project_list_layout.addStretch(1)
        scroll_area.setWidget(self.project_list_container)
        layout.addWidget(scroll_area, 1)
        return page

    def _create_operation_page(self, page_title, page_key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(self._create_page_title(page_title, page_key))
        layout.addWidget(self.viewport.get_widget(), 1)
        return page

    def _create_placeholder_page(self, page_title, page_key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(self._create_page_title(page_title, page_key))

        placeholder = QLabel('该模块将在本周末开始接入')
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet('color: #8a9099; font-size: 15px;')
        layout.addWidget(placeholder, 1)
        return page

    def _create_header(self):
        dock = QDockWidget('Header', self)
        dock.setObjectName('headerDock')
        dock.setAllowedAreas(Qt.DockWidgetArea.TopDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        title_bar = QWidget()
        title_bar.setFixedHeight(0)
        dock.setTitleBarWidget(title_bar)

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

        for label, button_name in HEADER_ACTIONS:
            button = QPushButton(label)
            button.setObjectName(button_name)
            button.setMinimumSize(120, 34)
            setattr(self, button_name, button)
            self.header_buttons[button_name] = button
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
            QPushButton:disabled {
                color: #a4a8ae;
                background-color: #e6e8eb;
                border-color: #d3d6da;
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

        self.current_project_label = QLabel('当前项目：未选择')
        self.current_project_label.setObjectName('currentProjectLabel')
        self.current_project_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.current_project_label.setStyleSheet(
            'color: #5b626d; font-size: 13px; padding: 0 4px;'
        )
        layout.addWidget(self.current_project_label)

        dock.setWidget(panel)
        dock.setMinimumHeight(60)
        dock.setMaximumHeight(72)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def set_current_page(self, page_index):
        if not 0 <= page_index < len(PAGE_DEFINITIONS):
            return
        page_key = PAGE_DEFINITIONS[page_index][1]
        button = self.page_buttons.get(page_key)
        if button is not None and not button.isEnabled():
            return

        self.centralWidget().setCurrentIndex(page_index)
        if button is not None:
            button.setChecked(True)
        self._update_header_actions(page_index)
        self._update_sidebar_visibility(page_key)

    def _update_header_actions(self, page_index):
        page_key = PAGE_DEFINITIONS[page_index][1]
        visible_buttons = {
            button_name
            for _label, button_name in PAGE_HEADER_ACTIONS[page_key]
        }
        for button_name, button in self.header_buttons.items():
            button.setVisible(button_name in visible_buttons)

        self.header_dock.setVisible(bool(visible_buttons))
        self.header_layout.invalidate()
        self.header_panel.updateGeometry()
        self._schedule_header_resize()

    def _connect_buttons(self):
        overview_actions = {
            'btn_upload': self._open_upload_file_dialog,
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_reset_view': self.pointcloud_service.reset_view,
            'btn_change_color': self.pointcloud_service.change_color,
            'btn_denoise': self.pointcloud_service.denoise,
            'btn_registration': self.pointcloud_service.registration,
            'btn_facade_detection': self.pointcloud_service.facade_detection,
            'btn_quality_inspection': self.pointcloud_service.quality_inspection,
            'btn_box_segmentation': self.pointcloud_service.box_segmentation,
            'btn_calculate_detail': self.pointcloud_service.calculate_detail,
            'btn_align_2d_3d': self.pointcloud_service.align_2d_3d,
        }
        for button_name, callback in {**overview_actions, **pointcloud_actions}.items():
            self.header_buttons[button_name].clicked.connect(callback)

        self.left_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                'left',
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar(
                'right',
                self.right_dock,
                self.right_sidebar_expand_button,
            )
        )
        self.left_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                'left',
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        )
        self.right_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar(
                'right',
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
        uploaded_paths = self.file_service.upload_files(file_paths)
        project = self.project_service.register_upload(
            uploaded_paths,
            current_project=self.current_project,
        )
        self._refresh_project_list()
        self._activate_project(project)

    def _open_project_directory(self):
        directory_path = QFileDialog.getExistingDirectory(
            self,
            '打开项目文件夹',
            self._last_upload_directory,
        )
        if not directory_path:
            return

        self._last_upload_directory = directory_path
        project = self.project_service.open_project(directory_path)
        self._refresh_project_list()
        self._activate_project(project)

    def _create_project(self):
        name, accepted = QInputDialog.getText(
            self,
            '新建项目',
            '项目名称：',
        )
        if not accepted or not name.strip():
            return

        directory_path = QFileDialog.getExistingDirectory(
            self,
            '选择项目文件夹',
            self._last_upload_directory,
        )
        if not directory_path:
            return

        self._last_upload_directory = directory_path
        project = self.project_service.create_project(name, directory_path)
        self._refresh_project_list()
        self._activate_project(project)

    def _select_project(self):
        projects = self.project_service.list_projects()
        if not projects:
            QMessageBox.information(self, '选择项目', '当前没有可选择的项目。')
            return

        labels = [
            f'{project.name}  |  {project.directory_path}'
            for project in projects
        ]
        selected_label, accepted = QInputDialog.getItem(
            self,
            '选择项目',
            '项目：',
            labels,
            0,
            False,
        )
        if not accepted:
            return

        selected_index = labels.index(selected_label)
        self._activate_project(projects[selected_index])

    def _refresh_project_list(self):
        while self.project_list_layout.count() > 1:
            item = self.project_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        projects = self.project_service.list_projects()
        if not projects:
            empty_label = QLabel('暂无项目，请通过顶部按钮上传文件或新建项目')
            empty_label.setObjectName('emptyProjectLabel')
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet(
                'color: #969ca5; font-size: 15px; padding: 48px 16px;'
            )
            self.project_list_layout.insertWidget(0, empty_label)
            return

        for project in projects:
            card = QPushButton()
            card.setObjectName('projectCard')
            card.setText(
                f'{project.name}\n'
                f'路径：{project.directory_path}\n'
                f'文件数量：{len(project.file_paths)}'
            )
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            card.setMinimumHeight(88)
            card.setStyleSheet(
                """
                QPushButton#projectCard {
                    color: #303641;
                    background-color: #ffffff;
                    border: 1px solid #d5d9df;
                    border-radius: 6px;
                    padding: 12px 16px;
                    text-align: left;
                }
                QPushButton#projectCard:hover {
                    background-color: #f2f6fb;
                    border-color: #7195c4;
                }
                """
            )
            card.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._open_project_card(project_id)
            )
            self.project_list_layout.insertWidget(
                self.project_list_layout.count() - 1,
                card,
            )

    def _open_project_card(self, project_id):
        project = self.project_service.get_project(project_id)
        if project is None:
            return

        if not Path(project.directory_path).exists():
            choice = QMessageBox.question(
                self,
                '项目路径不存在',
                '该项目文件夹已经不存在，是否从项目列表中移除？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                self.project_service.remove_project(project_id)
                if (
                    self.current_project is not None
                    and self.current_project.project_id == project_id
                ):
                    self._set_current_project(None)
                self._refresh_project_list()
            return

        self._activate_project(project)

    def _activate_project(self, project):
        self._set_current_project(project)
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)

    def _set_current_project(self, project):
        self.current_project = project
        has_project = project is not None

        for page_key, button in self.page_buttons.items():
            button.setEnabled(page_key == 'project_overview' or has_project)

        project_name = project.name if has_project else ''
        for page_title, page_key in PAGE_DEFINITIONS:
            label_text = (
                f'{page_title} — {project_name}'
                if project_name
                else page_title
            )
            self.page_title_labels[page_key].setText(label_text)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(project.directory_path)
            self.setWindowTitle(
                f'PointCloud FacadeDetection - {project.name}'
            )
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.setWindowTitle('PointCloud FacadeDetection')
            self.set_current_page(0)

    def _collapse_sidebar(self, side, dock, expand_button):
        self._sidebar_collapsed[side] = True
        dock.setProperty('expandedWidth', max(180, min(dock.width(), 260)))
        dock.hide()
        expand_button.show()
        expand_button.raise_()
        QTimer.singleShot(0, self._position_sidebar_expand_buttons)

    def _expand_sidebar(self, side, dock, expand_button):
        self._sidebar_collapsed[side] = False
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

    def _update_sidebar_visibility(self, page_key):
        if page_key != 'project_operation':
            self.left_dock.hide()
            self.right_dock.hide()
            self.left_sidebar_expand_button.hide()
            self.right_sidebar_expand_button.hide()
            return

        sidebars = (
            ('left', self.left_dock, self.left_sidebar_expand_button),
            ('right', self.right_dock, self.right_sidebar_expand_button),
        )
        for side, dock, expand_button in sidebars:
            if self._sidebar_collapsed[side]:
                dock.hide()
                expand_button.show()
                expand_button.raise_()
            else:
                expand_button.hide()
                dock.show()
        QTimer.singleShot(0, self._position_sidebar_expand_buttons)

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
