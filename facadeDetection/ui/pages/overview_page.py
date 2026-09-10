from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.window_chrome import ElidedLabel


UPLOAD_FILE_FILTER = (
    '项目支持文件 '
    '(*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls '
    '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.dist);;'
    '点云文件 (*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls *.dist);;'
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '所有文件 (*)'
)


class OverviewPageMixin:
    def _create_project_overview_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
        )

        # 概览与其他页面共用同一工作区；内部只用分栏，不再套第二层卡片。
        overview_columns = QWidget()
        overview_columns.setObjectName('overviewColumns')
        overview_columns.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        columns_layout = QHBoxLayout(overview_columns)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(0)

        projects_panel = QFrame()
        projects_panel.setObjectName('projectActivityPanel')
        projects_panel.setProperty('uiRole', 'workspaceSection')
        projects_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        projects_panel_layout = QVBoxLayout(projects_panel)
        projects_panel_layout.setContentsMargins(0, 0, 0, 0)
        projects_panel_layout.setSpacing(0)

        section_heading = QWidget()
        section_heading.setObjectName('projectActivityHeader')
        section_heading.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        section_heading_layout = QHBoxLayout(section_heading)
        section_heading_layout.setContentsMargins(16, 0, 16, 0)
        section_heading_layout.setSpacing(8)
        section_heading.setFixedHeight(56)
        description = QLabel('项目列表')
        description.setObjectName('projectListSectionTitle')
        description.setProperty('uiRole', 'sectionTitle')
        section_heading_layout.addWidget(description)
        section_heading_layout.addStretch(1)
        projects_panel_layout.addWidget(section_heading)

        scroll_area = QScrollArea()
        scroll_area.setObjectName('projectListScrollArea')
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.project_list_container = QWidget()
        self.project_list_container.setObjectName('projectListContainer')
        self.project_list_layout = QVBoxLayout(self.project_list_container)
        self.project_list_layout.setContentsMargins(0, 0, 0, 0)
        self.project_list_layout.setSpacing(0)
        self.project_list_layout.addStretch(1)
        scroll_area.setWidget(self.project_list_container)
        projects_panel_layout.addWidget(scroll_area, 1)
        columns_layout.addWidget(projects_panel, 1)

        workspace_panel = QFrame()
        workspace_panel.setObjectName('currentWorkspacePanel')
        workspace_panel.setProperty('uiRole', 'workspaceAside')
        workspace_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        workspace_panel.setMinimumWidth(304)
        workspace_panel.setMaximumWidth(384)
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(24, 20, 24, 24)
        workspace_layout.setSpacing(8)

        workspace_accent = QFrame()
        workspace_accent.setProperty('uiRole', 'accentLine')
        workspace_accent.setFixedSize(36, 3)
        workspace_layout.addWidget(workspace_accent)
        workspace_layout.addSpacing(8)

        workspace_title = QLabel('当前工作区')
        workspace_title.setProperty('uiRole', 'sectionTitle')
        self.overview_workspace_name_label = ElidedLabel('未选择项目')
        self.overview_workspace_name_label.setObjectName(
            'overviewWorkspaceNameLabel'
        )
        self.overview_workspace_path_label = ElidedLabel(
            '选择项目后显示本地目录'
        )
        self.overview_workspace_path_label.setObjectName(
            'overviewWorkspacePathLabel'
        )
        self.overview_workspace_file_label = QLabel('0 个数据文件')
        self.overview_workspace_file_label.setObjectName(
            'overviewWorkspaceFileLabel'
        )
        workspace_layout.addWidget(workspace_title)
        workspace_layout.addSpacing(8)
        workspace_layout.addWidget(self.overview_workspace_name_label)
        workspace_layout.addWidget(self.overview_workspace_path_label)
        workspace_layout.addWidget(self.overview_workspace_file_label)
        workspace_layout.addStretch(1)
        columns_layout.addWidget(workspace_panel)

        body_layout.addWidget(overview_columns, 1)
        return page

    def _open_upload_file_dialog(self):
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            '选择点云或图像文件',
            self._last_upload_directory,
            UPLOAD_FILE_FILTER,
        )
        if not file_paths:
            return

        if self.current_project is None:
            QMessageBox.information(self, '直接上传文件', '请先新建或选择项目，再上传 PLY 点云文件。')
            return

        self._last_upload_directory = str(Path(file_paths[0]).parent)
        # 当前项目的新增上传是增量操作；不能先销毁运行时，否则已有站点
        # 会被重新读取。项目切换仍由打开/选择项目入口负责销毁。
        self._start_load('upload', self.current_project.project_id,
                         file_paths=file_paths)

    def _open_import_fls_directory(self):
        directory_path = QFileDialog.getExistingDirectory(
            self,
            '导入 FLS 目录',
            self._last_upload_directory,
        )
        if not directory_path:
            return
        self._last_upload_directory = directory_path
        project_id = getattr(self.current_project, 'project_id', None)
        if project_id:
            self._start_load('fls', project_id, directory=directory_path)

    def _open_project_directory(self):
        directory_path = QFileDialog.getExistingDirectory(
            self,
            '打开项目文件夹',
            self._last_upload_directory,
        )
        if not directory_path:
            return

        self._last_upload_directory = directory_path
        self._prepare_project_activation(None)
        # open_project 目前包含 Open3D 渲染功能，必须保持在GUI 线程上。
        project = self.project_overview_service.open_project(directory_path)
        self._refresh_project_list()
        self._activate_project(project)

    def _create_project(self):
        from ui.dialogs.project_create_dialog import ProjectCreateDialog

        dlg = ProjectCreateDialog(self)
        result_code = dlg.exec()
        if result_code != int(QDialog.DialogCode.Accepted):
            return
        self._prepare_project_activation(None)
        payload = dlg.values()
        project = self.project_overview_service.create_project(
            name=payload.get('name', ''),
            org_unit=payload.get('org_unit'),
            address=payload.get('address'),
            remarks=payload.get('remarks'),
            building_floor=payload.get('building_floor'),
            construction_unit=payload.get('construction_unit'),
            construction_unit_executor=payload.get('construction_unit_executor'),
            inspection_unit=payload.get('inspection_unit'),
            supervision_unit=payload.get('supervision_unit'),
            client_unit=payload.get('client_unit'),
            report_no=payload.get('report_no'),
            inspection_date=payload.get('inspection_date'),
            report_date=payload.get('report_date'),
            inspection_params_json=payload.get('inspection_params_json'),
            fls_directories=payload.get('fls_directories'),
            pointcloud_files=payload.get('pointcloud_files'),
            photo_files=payload.get('photo_files'),
        )
        self._refresh_project_list()
        self._activate_project(project)
        # 资源已在项目保存时完成绑定。点云不再重复走 upload：激活流程会
        # 从刚写入的 FileAsset 恢复首站，避免首次创建被误判为恢复已有站点。
        # FLS 仍需转换并增量导入，沿用现有后台处理管线。
        if payload.get('fls_directories'):
            for directory in payload['fls_directories']:
                self._start_load('fls', project.project_id, directory=directory)

    def _prompt_project_name(self, title, initial_text=''):
        """使用可容纳真实工程长名称的项目名称输入框。"""
        dialog = QInputDialog(self)
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setWindowTitle(title)
        dialog.setLabelText('项目名称：')
        dialog.setTextValue(initial_text)
        dialog.setMinimumSize(560, 190)
        dialog.resize(620, 210)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return dialog.textValue(), accepted

    def _select_project(self):
        projects = self.project_overview_service.list_projects()
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
        self._prepare_project_activation(projects[selected_index].project_id)
        self._activate_project(projects[selected_index])

    def _update_overview_workspace(self):
        """同步概览右侧的当前工作区信息。"""
        if self.current_project is None:
            self.overview_workspace_name_label.setText('未选择项目')
            self.overview_workspace_path_label.setText('选择项目后显示本地目录')
            self.overview_workspace_path_label.setToolTip('')
            self.overview_workspace_file_label.setText('0 个数据文件')
            return

        self.overview_workspace_name_label.setText(self.current_project.name)
        self.overview_workspace_name_label.setToolTip(self.current_project.name)
        self.overview_workspace_path_label.setText(
            self.current_project.directory_path
        )
        self.overview_workspace_path_label.setToolTip(
            self.current_project.directory_path
        )
        # 新持久化模型按需读取资源，不在项目卡片中缓存可能过期的文件数。
        self.overview_workspace_file_label.setText('项目目录已连接')

    def _refresh_project_list(self):
        while self.project_list_layout.count() > 1:
            item = self.project_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # 先隐藏再延迟销毁，避免空状态切换到项目列表时短暂残留。
                widget.hide()
                widget.deleteLater()

        projects = self.project_overview_service.list_projects()
        self._update_overview_workspace()
        if not projects:
            empty_state = QWidget()
            empty_state.setObjectName('projectEmptyState')
            empty_state.setStyleSheet('background:#FFFFFF;')
            empty_layout = QVBoxLayout(empty_state)
            empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title = QLabel('瑞捷建筑外立面质量检测平台')
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet('font-size:30px;font-weight:700;color:#334155;')
            hint = QLabel('请新建 / 打开一个项目开始外立面质量检测工作')
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setStyleSheet('font-size:16px;color:#94A3B8;margin-top:10px;')
            empty_layout.addWidget(title)
            empty_layout.addWidget(hint)
            self.project_list_layout.insertWidget(0, empty_state, 1)
            self.project_list_layout.setStretch(1, 0)
            return

        # 恢复列表底部弹性空间，项目行保持紧凑高度并从上向下排列。
        self.project_list_layout.setStretch(0, 1)
        for project in projects:
            project_row = QWidget()
            project_row.setObjectName('projectRow')
            project_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            project_row.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            project_row.setMinimumHeight(88)
            project_row_layout = QHBoxLayout(project_row)
            project_row_layout.setContentsMargins(18, 12, 14, 12)
            project_row_layout.setSpacing(12)

            project_marker = QLabel((project.name or 'P')[:1].upper())
            project_marker.setObjectName('projectMarkerLabel')
            project_marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
            project_marker.setFixedSize(36, 36)
            project_row_layout.addWidget(project_marker)

            project_info = QWidget()
            project_info.setObjectName('projectInfo')
            project_info_layout = QVBoxLayout(project_info)
            project_info_layout.setContentsMargins(0, 0, 0, 0)
            project_info_layout.setSpacing(4)

            project_name = ElidedLabel(
                project.name,
                maximum_hint_width=None,
            )
            project_name.setObjectName('projectNameLabel')
            project_name.setToolTip(project.name)
            project_name.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            project_path = ElidedLabel(f'目录  {project.directory_path}')
            project_path.setObjectName('projectPathLabel')
            project_path.setToolTip(project.directory_path)
            # 最新项目模型只暴露项目元数据，卡片不再猜测未加载的文件数量。
            project_meta = QLabel('本地工程')
            project_meta.setObjectName('projectMetaLabel')
            project_info_layout.addWidget(project_name)
            project_info_layout.addWidget(project_path)
            project_info_layout.addWidget(project_meta)
            project_row_layout.addWidget(project_info, 1)

            open_button = QPushButton('打开')
            open_button.setObjectName('btn_open_project_card')
            open_button.setProperty('buttonRole', 'primary')
            open_button.setToolTip('进入项目工作区')
            open_button.setAccessibleName('打开项目')
            open_button.setCursor(Qt.CursorShape.PointingHandCursor)
            open_button.setMinimumSize(72, 36)
            open_button.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._open_project_card(project_id)
            )

            edit_button = QPushButton('编辑')
            edit_button.setObjectName('btn_edit_project')
            edit_button.setToolTip('修改项目名称')
            edit_button.setAccessibleName('编辑项目')
            edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
            edit_button.setMinimumSize(72, 36)
            edit_button.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._edit_project(project_id)
            )

            delete_button = QPushButton('删除')
            delete_button.setObjectName('btn_delete_project')
            delete_button.setProperty('buttonRole', 'danger')
            delete_button.setToolTip('删除项目')
            delete_button.setAccessibleName('删除项目')
            delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
            delete_button.setMinimumSize(72, 36)
            delete_button.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._delete_project(project_id)
            )

            # 项目信息与动作分列，避免把名称、路径和数量塞进一个按钮文本。
            action_panel = QWidget()
            action_panel.setObjectName('projectActionPanel')
            action_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            action_layout = QHBoxLayout(action_panel)
            action_layout.setContentsMargins(4, 4, 4, 4)
            action_layout.setSpacing(6)
            action_layout.addWidget(open_button)
            action_layout.addWidget(edit_button)
            action_layout.addWidget(delete_button)

            project_row_layout.addWidget(
                action_panel,
                0,
                Qt.AlignmentFlag.AlignVCenter,
            )
            self.project_list_layout.insertWidget(
                self.project_list_layout.count() - 1,
                project_row,
            )

    def _edit_project(self, project_id):
        from ui.dialogs.project_create_dialog import ProjectCreateDialog

        project = self.project_overview_service.get_project(project_id)
        if project is None:
            QMessageBox.warning(self, '编辑项目', '项目不存在或已被删除。')
            return

        dlg = ProjectCreateDialog(self, project=project)
        if dlg.exec() != int(QDialog.DialogCode.Accepted):
            return

        try:
            updated_project = self.project_overview_service.update_project(
                project_id, **dlg.values()
            )
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, '编辑项目', str(error))
            return

        # 当前项目被重命名时，同步顶部项目名、概览摘要和窗口标题。
        if (
            self.current_project is not None
            and self.current_project.project_id == project_id
        ):
            self._set_current_project(updated_project)
            # Refresh the live station projection immediately after editing;
            # downstream detection/report services read this project session.
            try:
                self.station_service.refresh()
                self._refresh_station_panel()
            except Exception as error:
                QMessageBox.warning(self, '编辑项目', f'资源已保存，但站点视图刷新失败：{error}')
        self._refresh_project_list()

    def _delete_project(self, project_id):
        project = self.project_overview_service.get_project(project_id)
        if project is None:
            return

        choice = QMessageBox.question(
            self,
            '删除项目',
            (
                f'确定永久删除“{project.name}”吗？\n'
                '项目文件夹、点云、检测结果和数据库数据都会被删除，且无法恢复。'
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice != QMessageBox.StandardButton.Yes:
            return

        if (self.current_project is not None and
                self.current_project.project_id == project_id):
            self._prepare_project_activation(None)
        try:
            self.project_overview_service.remove_project(project_id)
        except Exception as error:
            QMessageBox.critical(self, '删除项目失败', str(error))
            return
        if (
            self.current_project is not None
            and self.current_project.project_id == project_id
        ):
            self._set_current_project(None)
        self._refresh_project_list()

    def _open_project_card(self, project_id):
        project = self.project_overview_service.get_project(project_id)
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
                self.project_overview_service.remove_project(project_id)
                if (
                    self.current_project is not None
                    and self.current_project.project_id == project_id
                ):
                    self._set_current_project(None)
                self._refresh_project_list()
            return

        # Ensure per-project DB is active and latest raw point cloud is loaded
        self._prepare_project_activation(project.project_id)
        self._start_load('activate', project.project_id, project=project)
