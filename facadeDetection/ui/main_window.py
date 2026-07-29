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
    QSplitter,
    QStackedWidget,
    QStyle,
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
        self.page_header_layouts = {}
        self.page_title_labels = {}
        self._sidebar_collapsed = {'left': False, 'right': False}
        self._last_upload_directory = str(Path.home())
        self._header_resize_pending = set()
        self._setup_ui()
        self._connect_buttons()
        self._refresh_project_list()
        self._set_current_project(None)

    def _setup_ui(self):
        # 每个页面自身包含 Header 和内容，切换时不再只替换中间视口。
        self.setCentralWidget(self._create_page_stack())

        self.setDockNestingEnabled(False)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)

        self.bottom_dock = self._create_bottom()
        self.set_current_page(0)

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._create_page_header(page_key))

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(14)
        content_layout.addWidget(self._create_page_title(page_title, page_key))

        description = QLabel('历史项目')
        description.setObjectName('projectListSectionTitle')
        description.setStyleSheet(
            'color: #565d69; font-size: 14px; font-weight: 600;'
        )
        content_layout.addWidget(description)

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
        content_layout.addWidget(scroll_area, 1)
        layout.addWidget(content, 1)
        return page

    def _create_operation_page(self, page_title, page_key):
        page = QWidget()
        self.operation_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._create_page_header(page_key))

        self.operation_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.operation_splitter.setObjectName('operationPageSplitter')
        self.operation_splitter.setChildrenCollapsible(False)

        (
            self.left_dock,
            self.left_sidebar_button,
        ) = self._create_sidebar('Left Sidebar', 'leftDock', 'left')
        (
            self.right_dock,
            self.right_sidebar_button,
        ) = self._create_sidebar('Right Sidebar', 'rightDock', 'right')

        viewport_panel = QWidget()
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(12, 10, 12, 10)
        viewport_layout.setSpacing(8)
        viewport_layout.addWidget(self._create_page_title(page_title, page_key))
        viewport_layout.addWidget(self.viewport.get_widget(), 1)

        self.operation_splitter.addWidget(self.left_dock)
        self.operation_splitter.addWidget(viewport_panel)
        self.operation_splitter.addWidget(self.right_dock)
        self.operation_splitter.setStretchFactor(0, 0)
        self.operation_splitter.setStretchFactor(1, 1)
        self.operation_splitter.setStretchFactor(2, 0)
        self.operation_splitter.setSizes([210, 1000, 210])
        self.operation_splitter.installEventFilter(self)
        layout.addWidget(self.operation_splitter, 1)

        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
        return page

    def _create_placeholder_page(self, page_title, page_key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        return page

    def _create_page_header(self, page_key):
        panel = QWidget()
        panel.setObjectName(f'{page_key}HeaderPanel')
        panel.setMinimumHeight(54)
        panel.setMaximumHeight(180)
        panel.installEventFilter(self)
        header_layout = FlowLayout(
            panel,
            margin=10,
            horizontal_spacing=8,
            vertical_spacing=8,
        )
        self.page_header_layouts[panel] = header_layout

        for label, button_name in PAGE_HEADER_ACTIONS[page_key]:
            button = QPushButton(label)
            button.setObjectName(button_name)
            button.setMinimumSize(120, 34)
            setattr(self, button_name, button)
            self.header_buttons[button_name] = button
            header_layout.addWidget(button)

        QTimer.singleShot(0, lambda: self._resize_page_header(panel))
        return panel

    def _create_sidebar(self, title, object_name, side):
        sidebar = QFrame()
        sidebar.setObjectName(object_name)
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

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
        sidebar_layout.addWidget(title_bar)

        panel = QWidget()
        panel.setObjectName(f'{object_name}Panel')
        sidebar_layout.addWidget(panel, 1)
        sidebar.setMinimumWidth(180)
        sidebar.setMaximumWidth(260)
        sidebar.setProperty('expandedWidth', 210)
        return sidebar, toggle_button

    def _create_sidebar_expand_button(self, side):
        button = QToolButton(self.operation_page)
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
        if page_key == 'project_operation':
            QTimer.singleShot(0, self._position_sidebar_expand_buttons)

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
            project_row = QWidget()
            project_row.setObjectName('projectRow')
            project_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            project_row.setMinimumHeight(88)
            project_row_layout = QHBoxLayout(project_row)
            project_row_layout.setContentsMargins(0, 0, 6, 0)
            project_row_layout.setSpacing(0)
            project_row.setStyleSheet(
                """
                QWidget#projectRow {
                    background-color: #ffffff;
                    border: 1px solid #d5d9df;
                    border-radius: 6px;
                }
                QPushButton#projectCard {
                    color: #303641;
                    background-color: transparent;
                    border: none;
                    padding: 12px 16px;
                    text-align: left;
                }
                QToolButton#btn_delete_project {
                    background-color: transparent;
                    border: none;
                    padding: 3px;
                }
                """
            )

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
            card.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._open_project_card(project_id)
            )

            delete_button = QToolButton()
            delete_button.setObjectName('btn_delete_project')
            delete_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
            )
            delete_button.setToolTip('删除项目')
            delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
            delete_button.setAutoRaise(True)
            delete_button.setFixedSize(28, 28)
            delete_button.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._delete_project(project_id)
            )

            project_row_layout.addWidget(card, 1)
            project_row_layout.addWidget(
                delete_button,
                0,
                Qt.AlignmentFlag.AlignVCenter,
            )
            self.project_list_layout.insertWidget(
                self.project_list_layout.count() - 1,
                project_row,
            )

    def _delete_project(self, project_id):
        project = self.project_service.get_project(project_id)
        if project is None:
            return

        choice = QMessageBox.question(
            self,
            '删除项目',
            (
                f'确定从项目列表中删除“{project.name}”吗？\n'
                '此操作不会删除本地项目文件。'
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        self.project_service.remove_project(project_id)
        if (
            self.current_project is not None
            and self.current_project.project_id == project_id
        ):
            self._set_current_project(None)
        self._refresh_project_list()

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
            page_title_label = self.page_title_labels.get(page_key)
            if page_title_label is not None:
                page_title_label.setText(label_text)

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
            lambda: self._restore_sidebar_width(side, target_width),
        )

    def _restore_sidebar_width(self, side, target_width):
        sizes = self.operation_splitter.sizes()
        total_width = max(sum(sizes), self.operation_splitter.width())
        side_index = 0 if side == 'left' else 2
        other_index = 2 if side == 'left' else 0
        sizes[side_index] = target_width
        sizes[1] = max(1, total_width - target_width - sizes[other_index])
        self.operation_splitter.setSizes(sizes)

    def _position_sidebar_expand_buttons(self):
        splitter_top_left = self.operation_splitter.mapTo(
            self.operation_page,
            QPoint(0, 0),
        )
        top = splitter_top_left.y() + 8
        margin = 8

        if not self.left_sidebar_expand_button.isHidden():
            self.left_sidebar_expand_button.move(
                splitter_top_left.x() + margin,
                top,
            )
            self.left_sidebar_expand_button.raise_()

        if not self.right_sidebar_expand_button.isHidden():
            self.right_sidebar_expand_button.move(
                splitter_top_left.x()
                + self.operation_splitter.width()
                - self.right_sidebar_expand_button.width()
                - margin,
                top,
            )
            self.right_sidebar_expand_button.raise_()

    def eventFilter(self, watched, event):
        if (
            watched in self.page_header_layouts
            and event.type() == QEvent.Type.Resize
        ):
            self._schedule_page_header_resize(watched)
        if (
            watched is getattr(self, 'operation_splitter', None)
            and event.type() in (
                QEvent.Type.Resize,
                QEvent.Type.Show,
                QEvent.Type.Move,
            )
        ):
            QTimer.singleShot(0, self._position_sidebar_expand_buttons)
        return super().eventFilter(watched, event)

    def _schedule_page_header_resize(self, panel):
        if panel in self._header_resize_pending:
            return
        self._header_resize_pending.add(panel)
        QTimer.singleShot(0, lambda: self._resize_page_header(panel))

    def _resize_page_header(self, panel):
        self._header_resize_pending.discard(panel)
        if panel.width() <= 0:
            return

        header_layout = self.page_header_layouts[panel]
        content_height = header_layout.heightForWidth(panel.width())
        target_height = max(54, min(content_height, 180))

        if (
            panel.minimumHeight() == target_height
            and panel.maximumHeight() == target_height
        ):
            return

        panel.setMinimumHeight(target_height)
        panel.setMaximumHeight(target_height)
