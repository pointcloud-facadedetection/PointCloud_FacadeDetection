"""Workbench page for the facade inspection desktop prototype."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.styles import PLACEHOLDER, make_button, set_active


WORKBENCH_MODES = (
    ('data', '数据'),
    ('walls', '墙体'),
    ('review', '复核'),
    ('results', '结果'),
    ('report', '报告'),
)


def _placeholder_label(object_name: str = '') -> QLabel:
    label = QLabel(PLACEHOLDER)
    label.setProperty('role', 'placeholder')
    if object_name:
        label.setObjectName(object_name)
    return label


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty('role', 'sectionTitle')
    return label


class WorkbenchPage(QWidget):
    """Five-mode project workbench with a lightweight 3D placeholder panel."""

    back_requested = Signal()
    report_requested = Signal()
    action_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('workbenchPage')
        self.nav_buttons: dict[str, QPushButton] = {}
        self.project_data: dict = {}
        self.current_mode = 'review'
        self._setup_ui()
        self.set_mode('review')

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QFrame()
        body.setObjectName('workbenchBody')
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_nav_rail())

        workspace = QWidget()
        workspace.setObjectName('workspaceContent')
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(16, 16, 16, 14)
        workspace_layout.setSpacing(12)

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName('workbenchContentStack')
        self.data_page = self._build_data_page()
        self.analysis_page = self._build_analysis_page()
        self.results_page = self._build_results_page()
        self.report_landing_page = self._build_report_landing_page()
        for page in (
            self.data_page,
            self.analysis_page,
            self.results_page,
            self.report_landing_page,
        ):
            self.content_stack.addWidget(page)
        workspace_layout.addWidget(self.content_stack, 1)
        workspace_layout.addWidget(self._build_summary_bar())
        body_layout.addWidget(workspace, 1)
        root.addWidget(body, 1)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName('workbenchHeader')
        header.setProperty('role', 'topbar')
        header.setFixedHeight(72)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(14)

        back = make_button('← 项目首页', 'back_to_home', 'topGhost', height=40)
        back.clicked.connect(self.back_requested)
        layout.addWidget(back)

        project_group = QVBoxLayout()
        project_group.setSpacing(2)
        self.project_name_label = QLabel('')
        self.project_name_label.setObjectName('workbenchProjectName')
        self.project_name_label.setStyleSheet(
            'color:#FFFFFF;font-size:17px;font-weight:700;'
        )
        self.project_meta_label = QLabel('检测范围   ·  检测标准 ')
        self.project_meta_label.setStyleSheet('color:#AEB9C6;font-size:11px;')
        project_group.addWidget(self.project_name_label)
        project_group.addWidget(self.project_meta_label)
        layout.addLayout(project_group)

        progress = QWidget()
        progress.setObjectName('workbenchProgress')
        progress_layout = QHBoxLayout(progress)
        progress_layout.setContentsMargins(8, 0, 8, 0)
        progress_layout.setSpacing(7)
        for index, text in enumerate(('导入数据', '自动计算', '人工复核', '生成报告')):
            step = QLabel(text)
            step.setStyleSheet(
                'color:#DCE3EC;background:rgba(255,255,255,0.08);'
                'padding:6px 9px;border-radius:12px;font-size:11px;'
            )
            progress_layout.addWidget(step)
            if index < 3:
                arrow = QLabel('›')
                arrow.setStyleSheet('color:#8391A4;font-size:16px;')
                progress_layout.addWidget(arrow)
        layout.addWidget(progress, 1, Qt.AlignmentFlag.AlignCenter)

        save = make_button('保存项目', 'save_project', 'topGhost', height=40)
        recalc = make_button('重新计算', 'recalculate_project', 'topGhost', height=40)
        pdf = make_button('生成PDF报告', 'generate_pdf_report', 'accent', height=40)
        self._connect_action(save, 'save_project')
        self._connect_action(recalc, 'recalculate_project')
        self._connect_action(pdf, 'generate_pdf_report')
        layout.addWidget(save)
        layout.addWidget(recalc)
        layout.addWidget(pdf)
        return header

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName('navRail')
        rail.setFixedWidth(76)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(7, 14, 7, 14)
        layout.setSpacing(8)
        icons = {'data': '▣', 'walls': '▤', 'review': '✓', 'results': '▥', 'report': '▧'}
        for mode, label in WORKBENCH_MODES:
            button = make_button(
                f'{icons[mode]}\n{label}',
                f'nav_{mode}',
                'nav',
                height=58,
            )
            button.clicked.connect(lambda _checked=False, name=mode: self.set_mode(name))
            self.nav_buttons[mode] = button
            layout.addWidget(button)
        layout.addStretch(1)
        return rail

    def _build_data_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('dataPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status = QFrame()
        status.setProperty('role', 'panel')
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(18, 12, 18, 12)
        status_layout.addWidget(_section_title('设备与数据连接'))
        status_layout.addSpacing(12)
        status_layout.addWidget(QLabel('设备状态'))
        status_layout.addWidget(_placeholder_label('deviceStatusValue'))
        status_layout.addStretch(1)
        connect = make_button('连接', 'connect_device')
        disconnect = make_button('断开', 'disconnect_device')
        self._connect_action(connect, 'connect_device')
        self._connect_action(disconnect, 'disconnect_device')
        status_layout.addWidget(connect)
        status_layout.addWidget(disconnect)
        layout.addWidget(status)

        imports = QHBoxLayout()
        imports.setSpacing(12)
        for title, object_name in (
            ('导入点云文件', 'import_point_cloud'),
            ('导入现场照片', 'import_site_photos'),
            ('导入2D立面图', 'import_facade_drawing'),
        ):
            card = QFrame()
            card.setObjectName(f'{object_name}Card')
            card.setProperty('role', 'dropzone')
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(18, 18, 18, 18)
            card_layout.setSpacing(8)
            label = _section_title(title)
            hint = QLabel(f'当前文件：{PLACEHOLDER}')
            hint.setProperty('role', 'muted')
            button = make_button('选择文件', object_name, 'primary')
            self._connect_action(button, object_name)
            card_layout.addWidget(label)
            card_layout.addWidget(hint)
            card_layout.addStretch(1)
            card_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
            imports.addWidget(card, 1)
        layout.addLayout(imports, 1)

        table_panel = QFrame()
        table_panel.setProperty('role', 'panel')
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(16, 14, 16, 16)
        table_layout.addWidget(_section_title('已导入文件'))
        table = QTableWidget(3, 5)
        table.setObjectName('importedFilesTable')
        table.setHorizontalHeaderLabels(('文件名', '类型', '大小', '状态', '操作'))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        for row in range(3):
            for column in range(4):
                item = QTableWidgetItem(PLACEHOLDER)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
            actions = QWidget()
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(6)
            view = make_button('查看', f'view_imported_file_{row + 1}', height=32)
            reload_button = make_button(
                '重新导入', f'reimport_file_{row + 1}', height=32
            )
            self._connect_action(view, f'view_imported_file_{row + 1}')
            self._connect_action(reload_button, f'reimport_file_{row + 1}')
            action_layout.addWidget(view)
            action_layout.addWidget(reload_button)
            table.setCellWidget(row, 4, actions)
        table_layout.addWidget(table)
        layout.addWidget(table_panel, 2)
        return page

    def _build_analysis_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('analysisPage')
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName('analysisSplitter')
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_wall_list())
        splitter.addWidget(self._build_visual_center())
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes((250, 760, 300))
        layout.addWidget(splitter)
        return page

    def _build_wall_list(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('wallListPanel')
        panel.setProperty('role', 'panel')
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        heading.addWidget(_section_title('全部墙体'))
        heading.addStretch(1)
        heading.addWidget(_placeholder_label('wallCountValue'))
        layout.addLayout(heading)
        for index in range(1, 7):
            button = make_button(
                f'墙体 {PLACEHOLDER}\n楼层区间 {PLACEHOLDER}',
                f'select_wall_{index}',
                'secondary',
                height=58,
            )
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setStyleSheet('text-align:left;padding-left:12px;')
            self._connect_action(button, f'select_wall_{index}')
            layout.addWidget(button)
        layout.addStretch(1)
        return panel

    def _build_visual_center(self) -> QWidget:
        center = QWidget()
        center.setObjectName('visualCenter')
        layout = QVBoxLayout(center)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(12)

        viewport_panel = QFrame()
        viewport_panel.setObjectName('viewportHost')
        viewport_panel.setProperty('role', 'panel')
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName('viewportToolbar')
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 8, 10, 8)
        toolbar_layout.setSpacing(7)
        toolbar_layout.addWidget(_section_title(f'3D 点云视图 · 墙体{PLACEHOLDER}'))
        toolbar_layout.addStretch(1)
        for text, name in (
            ('平整度', 'view_flatness'),
            ('垂直度', 'view_verticality'),
            ('⟳', 'reset_3d_view'),
            ('⤢', 'fit_3d_view'),
            ('✋', 'pan_3d_view'),
        ):
            button = make_button(text, name, 'secondary', height=32)
            self._connect_action(button, name)
            toolbar_layout.addWidget(button)
        viewport_layout.addWidget(toolbar)
        viewport_canvas = QFrame()
        viewport_canvas.setObjectName('pointCloudViewport')
        viewport_canvas.setStyleSheet(
            'QFrame#pointCloudViewport {'
            'background:#17202D;border-top:1px solid #293547;'
            '}'
        )
        canvas_layout = QVBoxLayout(viewport_canvas)
        canvas_layout.setContentsMargins(20, 20, 20, 20)
        canvas_title = QLabel('三维点云视口')
        canvas_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_title.setStyleSheet('color:#8090A4;font-size:16px;font-weight:700;')
        canvas_value = QLabel(PLACEHOLDER)
        canvas_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_value.setStyleSheet('color:#58687C;font-size:28px;')
        canvas_layout.addStretch(1)
        canvas_layout.addWidget(canvas_title)
        canvas_layout.addWidget(canvas_value)
        canvas_layout.addStretch(1)
        viewport_layout.addWidget(viewport_canvas, 1)
        layout.addWidget(viewport_panel, 3)

        facade = QFrame()
        facade.setObjectName('facade2dPanel')
        facade.setProperty('role', 'panel')
        facade_layout = QVBoxLayout(facade)
        facade_layout.setContentsMargins(14, 10, 14, 12)
        title_row = QHBoxLayout()
        title_row.addWidget(_section_title(f'2D 立面图 · {PLACEHOLDER}热力图'))
        title_row.addStretch(1)
        for text, name in (('楼层线', 'toggle_floor_lines'), ('异常框选', 'toggle_anomaly_boxes')):
            button = make_button(text, name, 'secondary', height=32)
            self._connect_action(button, name)
            title_row.addWidget(button)
        facade_layout.addLayout(title_row)
        canvas = QFrame()
        canvas.setObjectName('facade2dCanvas')
        canvas.setProperty('role', 'soft')
        canvas_layout = QVBoxLayout(canvas)
        canvas_label = QLabel(f'2D 立面图数据\n{PLACEHOLDER}')
        canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_label.setStyleSheet('color:#8A93A0;font-size:14px;')
        canvas_layout.addWidget(canvas_label)
        facade_layout.addWidget(canvas, 1)
        layout.addWidget(facade, 2)
        return center

    def _build_inspector(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName('rightInspector')
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.inspector_stack = QStackedWidget()
        self.inspector_stack.addWidget(self._build_info_inspector())
        self.inspector_stack.addWidget(self._build_review_inspector())
        layout.addWidget(self.inspector_stack)
        return panel

    def _build_info_inspector(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(_section_title('墙体检测信息'))
        for label in (
            '当前墙体', '当前楼层区间', '当前指标', '总测量点数',
            '合格点数', '不合格点数', '合格率', '复核状态',
        ):
            layout.addLayout(self._value_row(label))
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        layout.addWidget(_section_title('异常区域信息'))
        for label in ('区域编号', '楼层区间', '检测指标', '严重程度', '备注'):
            layout.addLayout(self._value_row(label))
        anomaly_actions = QHBoxLayout()
        for text, name in (
            ('✕', 'close_anomaly'),
            ('标记为有效', 'mark_anomaly_valid'),
            ('标记为忽略', 'mark_anomaly_ignored'),
        ):
            button = make_button(text, name, 'secondary', height=34)
            self._connect_action(button, name)
            anomaly_actions.addWidget(button)
        layout.addLayout(anomaly_actions)
        layout.addStretch(1)
        return page

    def _build_review_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(_section_title('人工复核'))
        groups = (
            ('墙体边界', (
                ('自动识别边界', 'auto_detect_boundary'),
                ('手动调整边界', 'manual_adjust_boundary'),
                ('确认墙体边界', 'confirm_wall_boundary'),
            )),
            ('楼层区间', (
                ('自动生成楼层线', 'auto_generate_floor_lines'),
                ('手动调整楼层线', 'manual_adjust_floor_lines'),
                ('确认楼层区间', 'confirm_floor_range'),
            )),
            ('点云与立面对齐', (
                ('自动对齐', 'auto_align'),
                ('手动微调', 'manual_align'),
                ('确认对齐', 'confirm_alignment'),
            )),
            ('异常复核', (
                ('查看异常区域', 'view_anomaly_regions'),
                ('添加备注', 'add_review_note'),
            )),
        )
        for title, actions in groups:
            group = QFrame()
            group.setProperty('role', 'soft')
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(10, 10, 10, 10)
            group_layout.setSpacing(7)
            group_layout.addWidget(QLabel(title))
            for text, name in actions:
                button = make_button(text, name, 'secondary', height=34)
                self._connect_action(button, name)
                group_layout.addWidget(button)
            layout.addWidget(group)
        save = make_button('保存复核结果', 'save_review_result', 'primary')
        done = make_button('标记当前墙体已复核', 'mark_wall_reviewed', 'accent')
        self._connect_action(save, 'save_review_result')
        self._connect_action(done, 'mark_wall_reviewed')
        layout.addWidget(save)
        layout.addWidget(done)
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName('resultsPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        filter_panel = QFrame()
        filter_panel.setProperty('role', 'panel')
        filter_layout = QHBoxLayout(filter_panel)
        filter_layout.setContentsMargins(14, 10, 14, 10)
        filter_layout.addWidget(_section_title('检测结果'))
        filter_layout.addStretch(1)
        for text, name in (
            ('全部墙体', 'filter_all_walls'),
            ('全部指标', 'filter_all_metrics'),
            ('只看未复核', 'filter_unreviewed'),
            ('合格率排序', 'sort_pass_rate'),
        ):
            button = make_button(text, name, 'secondary', height=34)
            self._connect_action(button, name)
            filter_layout.addWidget(button)
        layout.addWidget(filter_panel)

        table = QTableWidget(6, 9)
        table.setObjectName('resultsTable')
        table.setHorizontalHeaderLabels((
            '墙体', '楼层区间', '检测指标', '总测量点', '合格点数',
            '不合格点数', '合格率', '不合格率', '复核状态',
        ))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = QTableWidgetItem(PLACEHOLDER)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)
        layout.addWidget(table, 1)
        return page

    def _build_report_landing_page(self) -> QWidget:
        page = QFrame()
        page.setObjectName('reportLandingPage')
        page.setProperty('role', 'panel')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.addStretch(1)
        title = QLabel('生成检测报告')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color:#13294B;font-size:26px;font-weight:700;')
        description = QLabel(f'报告数据状态：{PLACEHOLDER}')
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setProperty('role', 'muted')
        open_report = make_button(
            '前往报告预览与导出 →',
            'open_report_preview',
            'accent',
            height=46,
        )
        open_report.setMaximumWidth(260)
        open_report.clicked.connect(self.report_requested)
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(description)
        layout.addSpacing(20)
        layout.addWidget(open_report, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
        return page

    def _build_summary_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName('summaryBar')
        bar.setFixedHeight(118)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for title, name in (
            ('整体合格率', 'summary_overall'),
            ('平整度', 'summary_flatness'),
            ('垂直度', 'summary_verticality'),
            ('报告状态', 'summary_report'),
        ):
            card = QFrame()
            card.setProperty('role', 'card')
            card.setObjectName(f'{name}Card')
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 12, 16, 12)
            label = QLabel(title)
            label.setProperty('role', 'muted')
            value = _placeholder_label(f'{name}Value')
            value.setStyleSheet('color:#13294B;font-size:21px;font-weight:700;')
            button = make_button('查看', name, 'secondary', height=30)
            self._connect_action(button, name)
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            card_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignRight)
            layout.addWidget(card, 1)
        return bar

    @staticmethod
    def _value_row(label_text: str) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setProperty('role', 'muted')
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(_placeholder_label())
        return row

    def _connect_action(self, button: QPushButton, action_name: str):
        button.clicked.connect(
            lambda _checked=False, name=action_name, text=button.text():
            self.action_requested.emit(name, text)
        )

    def set_project_data(self, data: dict):
        """Update the workbench header from the active project's form data."""
        self.project_data = dict(data or {})

        def text_value(key: str) -> str:
            value = self.project_data.get(key, '')
            return '' if value is None else str(value).strip()

        project_name = text_value('project_name')
        start_floor = text_value('start_floor')
        end_floor = text_value('end_floor')
        inspection_standard = text_value('inspection_standard')

        if start_floor and end_floor:
            inspection_range = f'{start_floor} 至 {end_floor}'
        else:
            inspection_range = start_floor or end_floor

        self.project_name_label.setText(project_name)
        self.project_meta_label.setText(
            f'检测范围 {inspection_range}  ·  检测标准 {inspection_standard}'
        )

    def set_mode(self, mode: str):
        if mode not in self.nav_buttons:
            raise ValueError(f'Unknown workbench mode: {mode}')
        self.current_mode = mode
        for name, button in self.nav_buttons.items():
            set_active(button, name == mode)

        if mode == 'data':
            self.content_stack.setCurrentWidget(self.data_page)
        elif mode in ('walls', 'review'):
            self.content_stack.setCurrentWidget(self.analysis_page)
            self.inspector_stack.setCurrentIndex(1 if mode == 'review' else 0)
        elif mode == 'results':
            self.content_stack.setCurrentWidget(self.results_page)
        else:
            self.content_stack.setCurrentWidget(self.report_landing_page)
