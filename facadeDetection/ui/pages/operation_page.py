import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.station_panel import StationPanel
from ui.widgets.pointcloud_controls import PointCloudControls
from ui.dialogs.facade_quality_dialog import FacadeQualityDialog
from services.inspection_profile import InspectionProfileService
from services.result_export_service import ResultExportService


class OperationPageMixin:
    def _create_operation_page(self, page_title, page_key):
        page = QWidget()
        self.operation_page = page
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
            page=page,
        )

        self.operation_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.operation_splitter.setObjectName('operationPageSplitter')
        self.operation_splitter.setChildrenCollapsible(False)
        # 三个区域共享一个工作台外框，细分隔线代替三张彼此孤立的卡片。
        self.operation_splitter.setHandleWidth(1)

        self.left_dock = self._create_sidebar('leftDock', 'left')
        self.right_dock = self._create_sidebar('rightDock', 'right')
        # 准备结果的右侧面板布局，避免重复创建。
        try:
            self._init_right_panel_widgets()
        except Exception:
            pass

        viewport_panel = QWidget()
        viewport_panel.setObjectName('viewportPanel')
        viewport_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(8, 0, 8, 8)
        viewport_layout.setSpacing(0)
        viewport_heading_row = QHBoxLayout()
        viewport_heading_row.setContentsMargins(16, 6, 16, 6)
        viewport_heading_row.setSpacing(12)
        viewport_title = QLabel('三维视口')
        viewport_title.setObjectName('viewportTitleLabel')
        viewport_title.setProperty('uiRole', 'sectionTitle')
        viewport_heading_row.addWidget(viewport_title)
        viewport_heading_row.addStretch(1)
        viewport_state = QLabel('等待点云加载')
        viewport_state.setObjectName('viewportStateLabel')
        viewport_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewport_heading_row.addWidget(viewport_state)
        viewport_title.setVisible(False)
        viewport_state.setVisible(False)
        viewport_layout.addLayout(viewport_heading_row)

        # 检测标准与区间不再在项目操作页暴露编辑入口：统一由【创建/编辑项目】
        # 表单下发（main_window 会同步这两个控件），此处仅作为隐藏数据载体
        # 保留，_on_standard_changed 继续据此解析 _inspection_profile。
        self.standard_combo = QComboBox()
        for profile in InspectionProfileService.all():
            self.standard_combo.addItem(
                f'{profile.standard_name} · {profile.version}', profile.standard_id)
        self.standard_combo.setVisible(False)
        self.standard_summary = QLabel()
        self.standard_summary.setObjectName('standardSummary')
        self.standard_summary.setVisible(False)
        self.interval_combo = QComboBox()
        for value in (3.0, 5.0, 10.0, 20.0):
            self.interval_combo.addItem(f'{value:g}m', value)
        self.interval_combo.setCurrentIndex(3)
        self.interval_combo.setVisible(False)
        self.standard_combo.currentIndexChanged.connect(self._on_standard_changed)
        self._on_standard_changed(0)

        # 步骤导航栏由命令栏（main_window._install_step_nav）统一装配，
        viewport_layout.addWidget(self.render_facade.widget(), 1)

        # 保存视口面板引用，供【检测复核】页迁移使用
        self.viewport_panel = viewport_panel

        self.operation_splitter.addWidget(self.left_dock)
        self.operation_splitter.addWidget(viewport_panel)
        self.operation_splitter.addWidget(self.right_dock)
        self.operation_splitter.setStretchFactor(0, 0)
        self.operation_splitter.setStretchFactor(1, 1)
        self.operation_splitter.setStretchFactor(2, 0)
        # Sidebars are intentionally compact: the viewport is the primary
        # workspace.  QSplitter keeps these as user-adjustable widths while
        # the stretch factor lets the centre absorb window resizing.
        self.operation_splitter.setSizes([220, 1040, 300])
        self.operation_splitter.setCollapsible(0, False)
        self.operation_splitter.setCollapsible(1, False)
        self.operation_splitter.setCollapsible(2, False)
        self.operation_splitter.splitterMoved.connect(
            self._remember_operation_splitter_sizes
        )
        body_layout.addWidget(self.operation_splitter, 1)
        # 左侧 panel 默认隐藏：三维视口是主工作区，站点管理按需展开。
        # 复用既有折叠链路（_sidebar_collapsed / _update_sidebar_toggle_button），
        # 使默认态与用户手动"收起"完全一致：宽度记忆、图标、tooltip 共用一套逻辑。
        self._sidebar_collapsed['left'] = True
        self.left_dock.hide()
        self._update_sidebar_toggle_button('left')
        return page

    def _on_standard_changed(self, _index):
        profile = InspectionProfileService.get(self.standard_combo.currentData())
        if profile is None:
            return
        self._inspection_profile = profile
        interval_index = self.interval_combo.findData(float(profile.interval_size_m))
        if interval_index >= 0:
            self.interval_combo.blockSignals(True)
            self.interval_combo.setCurrentIndex(interval_index)
            self.interval_combo.blockSignals(False)
        self.standard_summary.setText(
            f'平整度 ≤ {profile.flatness_limit_mm:g} mm  | '
            f'垂直度 ≤ {profile.verticality_limit_mm:g} mm  | ')

    def _init_right_panel_widgets(self):
        panel = self.right_dock.findChild(QWidget, 'rightDockPanel')
        if panel is None:
            return
        
        # Clear any existing layout
        old_layout = panel.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
        
        # Main scroll area
        scroll = QScrollArea()
        scroll.setObjectName('rightPanelScroll')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container.setObjectName('rightPanelContainer')
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        
        # ── 检测结果 ──
        title = QLabel('检测结果')
        title.setStyleSheet('font-weight:700; color:#1e293b; font-size:14px;')
        lay.addWidget(title)
        
        self.lbl_facade_summary = QLabel('未检测')
        self.lbl_facade_summary.setObjectName('lblFacadeSummary')
        self.lbl_facade_summary.setStyleSheet('color:#5b626d; font-size:12px;')
        self.lbl_facade_summary.setMinimumHeight(28)
        self.lbl_facade_summary.setWordWrap(True)
        self.lbl_facade_summary.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.lbl_facade_summary)
        
        # 原"评估立面质量"按钮已由顶部步骤④取代，此处不再重复提供入口。

        # 热力切换显示：按改造要求从右侧面板迁移到左侧站点管理面板。
        self.btn_heatmap_toggle = QPushButton('热力切换显示')
        self.btn_heatmap_toggle.setObjectName('btn_heatmap_toggle')
        self.btn_heatmap_toggle.setToolTip('在平整度热力与垂直度热力之间切换')
        self.btn_heatmap_toggle.setMinimumHeight(34)
        self.btn_heatmap_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_heatmap_toggle.clicked.connect(self._toggle_heatmap_display)
        self.btn_heatmap_toggle.setEnabled(False)
        left_panel = self.left_dock.findChild(QWidget, 'leftDockPanel')
        left_layout = left_panel.layout() if left_panel is not None else None
        if left_layout is not None:
            left_layout.addWidget(self.btn_heatmap_toggle)
        else:
            lay.addWidget(self.btn_heatmap_toggle)

        # 立面列表
        from PySide6.QtWidgets import QListWidget
        self.list_facades = QListWidget()
        self.list_facades.setObjectName('lstFacades')
        self.list_facades.setSpacing(4)
        self.list_facades.setUniformItemSizes(False)
        # 单选保持原交互；按住 Ctrl/Shift 可复用同一评估按钮批量检测。
        self.list_facades.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        
        # 保持原有单选交互；批量入口由控制层保留，UI 不强制改变列表选择模式。
        # 这样旧用户无需学习 Ctrl/Shift 多选，单立面只要标记“完整”即可评估。
        # self.list_facades.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_facades.itemClicked.connect(self._on_facade_item_clicked)
        lay.addWidget(self.list_facades, 3)
        
        # ── 质量检测参数 ──
        config = QFrame()
        config.setObjectName('qualityParameterPanel')
        config.setFrameShape(QFrame.Shape.StyledPanel)
        config.setStyleSheet("""
            #qualityParameterPanel {
                background: #fafbfc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
        """)
        config_layout = QVBoxLayout(config)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(10)
        
        param_title = QLabel('质量检测参数')
        param_title.setStyleSheet('font-weight:700; color:#1e293b; font-size:13px;')
        config_layout.addWidget(param_title)
        
        # ── 普通参数：单列紧凑布局 ──
        normal_group = QFrame()
        normal_group.setStyleSheet("""
            QFrame { background: transparent; }
            QLabel { color: #475569; font-size: 11px; padding-right: 4px; }
        """)
        normal_layout = QVBoxLayout(normal_group)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(5)
        
        self.quality_length_spin = self._quality_double(2.0, 0.0, 100.0, .001)
        self.quality_step_spin = self._quality_double(.05, 0.0, 100.0, .001)
        self.quality_width_spin = self._quality_double(.055, 0.0, 100.0, .001)
        self.quality_select_band_spin = self._quality_double(.01, 0.0, 100.0, .001)
        self.quality_hole_band_spin = self._quality_double(.02, 0.0, 100.0, .001)
        self.quality_bin_size_spin = self._quality_double(.04, 0.0, 100.0, .001)
        self.quality_top_q_spin = self._quality_double(1.0, 0.0, 100.0, .001)
        self.quality_max_hole_ratio_spin = self._quality_double(.20, 0.0, 1.0, .001)
        self.quality_min_points_spin = QSpinBox()
        self.quality_min_points_spin.setRange(3, 100000)
        self.quality_min_points_spin.setValue(30)
        
        normal_params = (
            ('靠尺长度 (m)', self.quality_length_spin),
            ('滑移步距 (m)', self.quality_step_spin),
            ('靠尺宽度 (m)', self.quality_width_spin),
            ('表面带宽 (m)', self.quality_select_band_spin),
            ('空洞带宽 (m)', self.quality_hole_band_spin),
            ('分段长度 (m)', self.quality_bin_size_spin),
            ('段代表分位数', self.quality_top_q_spin),
            ('最大空洞占比', self.quality_max_hole_ratio_spin),
            ('最小点数', self.quality_min_points_spin),
        )
        
        for label_text, widget in normal_params:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(82)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            if isinstance(widget, QDoubleSpinBox):
                widget.setFixedWidth(90)
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            elif isinstance(widget, QSpinBox):
                widget.setFixedWidth(90)
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(widget)
            row.addStretch(1)
            normal_layout.addLayout(row)
        
        config_layout.addWidget(normal_group)
        
        # ── 分隔线 ──
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet('color: #e2e8f0;')
        config_layout.addWidget(line)
        
        # ── SOR参数：独立分组，单列布局 ──
        sor_title = QLabel('SOR 离群剔除参数')
        sor_title.setStyleSheet('font-weight:700; color:#475569; font-size:12px;')
        config_layout.addWidget(sor_title)
        
        sor_group = QFrame()
        sor_group.setStyleSheet("""
            QFrame { background: transparent; }
            QLabel { color: #475569; font-size: 11px; padding-right: 4px; }
        """)
        sor_layout = QVBoxLayout(sor_group)
        sor_layout.setContentsMargins(0, 0, 0, 0)
        sor_layout.setSpacing(5)
        
        self.quality_sor_sigma_spin = self._quality_double(4.0, 0.0, 100.0, .001)
        self.quality_sor_k_spin = QSpinBox()
        self.quality_sor_k_spin.setRange(1, 128)
        self.quality_sor_k_spin.setValue(8)
        self.quality_sor_method_combo = QComboBox()
        self.quality_sor_method_combo.addItem('local', 'local')
        self.quality_sor_method_combo.addItem('grid', 'grid')
        self.quality_sor_method_combo.addItem('exact', 'exact')
        self.quality_sor_w_weight_spin = self._quality_double(50.0, 0.0, 100.0, .001)
        
        sor_params = (
            ('SOR 阈值 σ', self.quality_sor_sigma_spin),
            ('SOR 邻居数 k', self.quality_sor_k_spin),
            ('SOR 方法', self.quality_sor_method_combo),
            ('SOR 高度权重', self.quality_sor_w_weight_spin),
        )
        
        for label_text, widget in sor_params:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(82)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            if isinstance(widget, QDoubleSpinBox):
                widget.setFixedWidth(90)
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            elif isinstance(widget, QSpinBox):
                widget.setFixedWidth(90)
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            elif isinstance(widget, QComboBox):
                widget.setFixedWidth(90)
            row.addWidget(widget)
            row.addStretch(1)
            sor_layout.addLayout(row)
        
        # SOR启用开关
        check_row = QHBoxLayout()
        check_row.setContentsMargins(0, 4, 0, 0)
        self.quality_sor_check = QCheckBox('启用 SOR 离群剔除')
        self.quality_sor_check.setChecked(True)
        self.quality_sor_check.setStyleSheet('font-weight:600; color:#334155; font-size:11px;')
        check_row.addWidget(self.quality_sor_check)
        check_row.addStretch(1)
        sor_layout.addLayout(check_row)
        
        config_layout.addWidget(sor_group)
        
        # 恢复标准参数按钮
        reset = QPushButton('恢复标准参数')
        reset.setMinimumHeight(30)
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(self._reset_quality_parameters)
        config_layout.addWidget(reset)
        
        # 检测参数不再在项目操作页暴露编辑入口：统一由【创建/编辑项目】表单下发。
        # 控件保留为不可见的数据载体，供质量控制器与"恢复标准参数"读取默认值。
        config.setVisible(False)
        lay.addWidget(config, 0)
        lay.addStretch(0)
        
        scroll.setWidget(container)
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(scroll)

    def _create_sidebar_toggle_button(self, side):
        """创建 VS Code 式分栏图标，一个按钮切换一侧边栏。"""
        button = QToolButton()
        button.setObjectName(f'btn_toggle_{side}_sidebar')
        button.setProperty('uiRole', 'sidebarToggle')
        button.setCheckable(True)
        button.setFixedSize(38, 36)
        button.setIconSize(QSize(20, 20))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_sidebar_toggle_button(side, button)
        return button

    def _create_sidebar_toggle_icon(self, side, expanded):
        """绘制 VS Code 的左右侧栏轮廓，避免依赖机器上的图标主题。"""
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        outline = QRectF(2.25, 2.75, 15.5, 14.5)
        panel_fill = QColor('#BFDBFE' if expanded else '#5B7FD8')
        if side == 'left':
            panel = QRectF(3.25, 3.75, 4.5, 12.5)
            divider_x = 7.75
        else:
            panel = QRectF(12.25, 3.75, 4.5, 12.5)
            divider_x = 12.25

        painter.fillRect(panel, panel_fill)
        painter.setPen(QPen(QColor('#FFFFFF'), 1.5))
        painter.drawRoundedRect(outline, 2.0, 2.0)
        painter.drawLine(
            QPointF(divider_x, outline.top()),
            QPointF(divider_x, outline.bottom()),
        )
        painter.end()
        return QIcon(pixmap)

    def _create_sidebar(self, object_name, side):
        sidebar = QFrame()
        sidebar.setObjectName(object_name)
        sidebar.setProperty('uiRole', 'sidebar')
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        panel = QWidget()
        panel.setObjectName(f'{object_name}Panel')
        panel.setProperty('uiRole', 'sidebarBody')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet('font-size:12px;')
        sidebar_layout.addWidget(panel, 1)
        if side == 'left':
            content = QVBoxLayout(panel)
            content.setContentsMargins(12, 12, 12, 12)
            content.setSpacing(8)
            station = StationPanel()
            self.station_list = station.list
            self.station_list.setToolTip('单击站点切换视图；复选框用于多选')
            # StationListWidget emits station clicks only from the item body;
            # checkbox changes therefore never replace the current viewport.
            self.station_list.itemChanged.connect(self._on_station_item_changed)
            station.station_clicked.connect(self._on_station_clicked)
            station.delete_requested.connect(self._delete_stations)
            station.merge_requested.connect(self._merge_stations)
            content.addWidget(station, 3)
            controls = PointCloudControls()
            controls.reset_view_requested.connect(self.project_operation_service.reset_view)
            controls.change_color_requested.connect(self.project_operation_service.change_color)
            controls.point_size_changed.connect(self.viewport.set_all_point_size)
            self.pointcloud_controls = controls
            content.addWidget(controls, 1)
        sidebar.setMinimumWidth(200 if side == 'left' else 260)
        sidebar.setMaximumWidth(280 if side == 'left' else 380)
        sidebar.setProperty('expandedWidth', 230 if side == 'left' else 320)
        return sidebar

    def _refresh_station_panel(self, active_station_id=None):
        """增量式刷新站点列表：仅增删改差异项，保留滚动位置与选中状态，避免整表重绘闪烁。"""
        if not hasattr(self, 'station_list'):
            return
        stations = self.station_service.list_stations()
        new_snap = {}
        for s in stations:
            new_snap[int(s.id)] = {
                'display_name': s.display_name,
                'is_selected': bool(s.is_selected),
                'last_error': getattr(s, 'last_error', None),
                'source_path': str(s.source_path or ''),
            }
        if not hasattr(self, '_station_list_snapshot') or self.station_list.count() == 0:
            self._station_list_snapshot = {}
            self._do_full_refresh(stations, active_station_id)
            self._station_list_snapshot = dict(new_snap)
            return
        old_snap = self._station_list_snapshot
        self.station_list.blockSignals(True)
        try:
            for row in range(self.station_list.count() - 1, -1, -1):
                item = self.station_list.item(row)
                sid = int(item.data(Qt.ItemDataRole.UserRole))
                if sid not in new_snap:
                    self.station_list.takeItem(row)
            existing_rows = {}
            for row in range(self.station_list.count()):
                item = self.station_list.item(row)
                existing_rows[int(item.data(Qt.ItemDataRole.UserRole))] = (row, item)
            for index, (sid, data) in enumerate(new_snap.items()):
                label = data['display_name']
                if data['last_error']:
                    label = f"{label}  [文件失效: {data['last_error']}]"
                if sid in existing_rows:
                    row, item = existing_rows[sid]
                    if item.text() != label:
                        item.setText(label)
                    check = Qt.CheckState.Checked if data['is_selected'] else Qt.CheckState.Unchecked
                    if item.checkState() != check:
                        item.setCheckState(check)
                    if item.toolTip() != data['source_path']:
                        item.setToolTip(data['source_path'])
                    if data['last_error'] and item.foreground() != Qt.GlobalColor.red:
                        item.setForeground(Qt.GlobalColor.red)
                    elif not data['last_error'] and item.foreground() == Qt.GlobalColor.red:
                        item.setForeground(self.station_list.palette().text().color())
                    if row != index:
                        taken = self.station_list.takeItem(row)
                        self.station_list.insertItem(index, taken)
                else:
                    item = QListWidgetItem(label)
                    item.setData(Qt.ItemDataRole.UserRole, sid)
                    item.setCheckState(Qt.CheckState.Checked if data['is_selected'] else Qt.CheckState.Unchecked)
                    item.setToolTip(data['source_path'])
                    if data['last_error']:
                        item.setForeground(Qt.GlobalColor.red)
                    self.station_list.insertItem(index, item)
        finally:
            self.station_list.blockSignals(False)
        self._station_list_snapshot = dict(new_snap)
        active_station_id = (active_station_id if active_station_id is not None
                             else getattr(self.station_service, '_active_station_id', None))
        if active_station_id is not None:
            for row in range(self.station_list.count()):
                item = self.station_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == active_station_id:
                    self.station_list.setCurrentItem(item)
                    break

    def _do_full_refresh(self, stations, active_station_id=None):
        """全量重建站点列表（仅首次或恢复时使用）。"""
        self.station_list.blockSignals(True)
        self.station_list.clear()
        for station in stations:
            label = station.display_name
            if getattr(station, 'last_error', None):
                label = f'{label}  [文件失效: {station.last_error}]'
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, station.id)
            item.setCheckState(Qt.CheckState.Checked if station.is_selected else Qt.CheckState.Unchecked)
            item.setToolTip(station.source_path)
            if getattr(station, 'last_error', None):
                item.setForeground(Qt.GlobalColor.red)
            self.station_list.addItem(item)
        self.station_list.blockSignals(False)
        if active_station_id is not None:
            for row in range(self.station_list.count()):
                item = self.station_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == active_station_id:
                    self.station_list.setCurrentItem(item)
                    break

    def _on_station_item_changed(self, item):
        """复选框变更仅累积状态，不触发任何视图刷新；由 timer flush 后按需更新。"""
        try:
            sid = item.data(Qt.ItemDataRole.UserRole)
            self._pending_station_selection[sid] = (
                item.checkState() == Qt.CheckState.Checked)
            self._station_selection_timer.start()
        except Exception as exc:
            self.statusBar().showMessage(f'保存站点选择失败：{exc}', 5000)

    def _flush_station_selection(self):
        pending = self._pending_station_selection
        self._pending_station_selection = {}
        try:
            for station_id, selected in pending.items():
                self.station_service.set_selected(station_id, selected)
            # 仅当存在已选站点且当前处于合并显示逻辑下，才刷新合并视口
            # 单站点模式下复选框变更不触发视口重渲染
            selected_ids = [sid for sid, sel in pending.items() if sel]
            if len(selected_ids) >= 2:
                # 延迟刷新合并视图，避免连续勾选导致多次重渲染
                QTimer.singleShot(50, self._refresh_merge_view_if_needed)
        except Exception as exc:
            self.statusBar().showMessage(f'保存站点选择失败：{exc}', 5000)

    def _refresh_merge_view_if_needed(self):
        """若当前应显示合并视图，则执行合并；否则保持当前单站点视图不变。"""
        try:
            view_state = self.station_service.list_stations()
            selected = [s for s in view_state if s.is_selected and not s.last_error]
            if len(selected) >= 2:
                self.station_service.merge_selected()
        except Exception as exc:
            print(f'[PCFD] merge_refresh_skipped error={exc!r}', flush=True)

    def _on_station_clicked(self, item):
        station_id = item.data(Qt.ItemDataRole.UserRole)
        # 防抖：若当前有挂起的切换或点击同一站点，忽略
        if getattr(self, '_pending_station_switch_id', None) == station_id:
            return
        self._pending_station_switch_id = station_id

        station = next((x for x in self.station_service.list_stations() if x.id == station_id), None)
        if not station:
            self._pending_station_switch_id = None
            return

        # 若已是活动站点，仅同步 UI 状态，不触发重渲染
        current_active = getattr(self.station_service, '_active_station_id', None)
        if current_active is not None and int(current_active) == int(station_id):
            self._pending_station_switch_id = None
            return

        # 立即切换视口（渲染服务内部已有短路保护）
        self.station_service.show_single(station)

        # 异步恢复立面结果，避免阻塞 GUI 主线程
        def _deferred_facade_refresh():
            try:
                self._pending_station_switch_id = None
                project_id = getattr(self.current_project, 'project_id', None)
                historical = (self.project_overview_service.load_historical_facades(
                    project_id, station.id) if project_id else [])
                self.project_operation_service.set_facade_results_for_station(station.id, historical or [])
                self._show_facade_results(historical or [])
            except Exception as exc:
                self._pending_station_switch_id = None
                print(f'[PCFD] facade.color_refresh_failed error={exc!r}', flush=True)

        QTimer.singleShot(0, _deferred_facade_refresh)

    def _delete_stations(self):
        try:
            self.station_service.delete_selected()
            self._refresh_station_panel()
        except Exception as exc:
            QMessageBox.warning(self, '删除站点', str(exc))

    def _merge_stations(self):
        try:
            self._station_selection_timer.stop()
            self._flush_station_selection()
            self.station_service.merge_selected()
            self._refresh_station_panel()
        except Exception as exc:
            QMessageBox.warning(self, '合并站点', str(exc))

    def _toggle_sidebar(self, side):
        """同一个命令栏按钮负责侧栏的展开和收起。"""
        dock = self.left_dock if side == 'left' else self.right_dock
        if self._sidebar_collapsed[side]:
            self._expand_sidebar(side, dock)
        else:
            self._collapse_sidebar(side, dock)

    def _collapse_sidebar(self, side, dock):
        self._sidebar_collapsed[side] = True
        dock.setProperty('expandedWidth', max(180, min(dock.width(), 260)))
        dock.hide()
        self._update_sidebar_toggle_button(side)

    def _expand_sidebar(self, side, dock):
        self._sidebar_collapsed[side] = False
        dock.show()
        self._update_sidebar_toggle_button(side)
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

    def _remember_operation_splitter_sizes(self, _pos, _index):
        """Keep the user's manually chosen sidebar widths for later restores."""
        if not hasattr(self, 'operation_splitter'):
            return
        sizes = self.operation_splitter.sizes()
        if len(sizes) >= 3:
            if not self._sidebar_collapsed['left']:
                self.left_dock.setProperty('expandedWidth', sizes[0])
            if not self._sidebar_collapsed['right']:
                self.right_dock.setProperty('expandedWidth', sizes[2])

    def _update_sidebar_toggle_button(self, side, button=None):
        button = button or (
            self.left_sidebar_button
            if side == 'left'
            else self.right_sidebar_button
        )
        collapsed = self._sidebar_collapsed[side]
        button.setText('')
        button.setIcon(
            self._create_sidebar_toggle_icon(side, expanded=not collapsed)
        )
        label = '左侧栏' if side == 'left' else '右侧栏'
        action = '展开' if collapsed else '收起'
        button.setToolTip(f'{action}{label}')
        button.setAccessibleName(f'{action}{label}')
        button.setChecked(collapsed)

    # ------------------------------------------------------------------
    # 【框选检测】交互流程（步骤 ②：区域选取 + 立面提取合并）
    # ------------------------------------------------------------------
    def _enter_box_detection_mode(self):
        """激活 ROI 框选模式；清理旧状态后进入视口绘制模式。"""
        nav = getattr(self, 'step_nav', None)
        if nav is not None:
            nav.reset_to_running(self.STEP_BOX_DETECT)
        cloud = self.project_operation_service._active_cloud_name()
        if not cloud:
            QMessageBox.information(
                getattr(self, 'viewport_panel', self),
                '框选检测', '请先加载点云数据。')
            return
        # 清除之前的 ROI 视觉与立面高亮
        try:
            if hasattr(self.viewport, 'clear_roi_visuals'):
                self.viewport.clear_roi_visuals()
        except Exception:
            pass
        try:
            self.render_service.clear_selected_facade(cloud)
        except Exception:
            pass
        # 进入 ROI 框选；完成回调交给确认卡片流程
        self.viewport.enter_roi_selection(
            cloud_name=cloud,
            on_complete=self._on_roi_box_detection_selected,
        )

    def _on_roi_box_detection_selected(self, min_bound, max_bound, indices, p1=None, p2=None):
        """ROI 框选绘制完成：保存数据、渲染预览 AABB、弹出确认卡片。"""
        self._pending_roi_data = {
            'min_bound': min_bound,
            'max_bound': max_bound,
            'indices': indices,
            'p1': p1,
            'p2': p2,
        }
        # 先渲染预览用的 3D AABB 框，让用户直观确认框选范围
        cloud = self.project_operation_service._active_cloud_name()
        if cloud and indices is not None and len(indices) > 0:
            try:
                self.project_operation_service._render_roi_bbox(cloud, indices)
            except Exception:
                pass
        self._show_roi_confirm_card(p1, p2)

    def _show_roi_confirm_card(self, p1, p2):
        """在视口边缘弹出轻量级悬浮确认卡片，不遮挡中心、不抢焦点。

        卡片以 ``self``（MainWindow）为父窗口并设置 ``Qt.Tool`` 标志，
        确保在 Windows 原生 Open3D 子窗口之上可见、可交互。
        使用 ``WindowStaysOnTopHint`` 保持置顶，同时不调用 ``activateWindow()``
        避免抢走 Open3D 视口焦点导致渲染暂停。
        """
        from PySide6.QtCore import QPoint
        card = QFrame(
            self,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        card.setObjectName('roiConfirmCard')
        card.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 固定尺寸避免被内容撑爆
        card.setFixedSize(180, 90)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        msg = QLabel('是否检测当前框选区域？')
        msg.setStyleSheet('font-size:13px; font-weight:600; color:#0F172A;')
        layout.addWidget(msg)

        row = QHBoxLayout()
        row.setSpacing(8)
        btn_no = QPushButton('否')
        btn_no.setProperty('buttonRole', 'secondary')
        btn_no.clicked.connect(lambda: self._on_roi_confirm_no(card))
        btn_yes = QPushButton('是')
        btn_yes.setProperty('buttonRole', 'primary')
        btn_yes.clicked.connect(lambda: self._on_roi_confirm_yes(card))
        row.addWidget(btn_no)
        row.addWidget(btn_yes)
        layout.addLayout(row)

        # 弹窗固定在视口右下角边缘，避免遮挡中心点云和 3D AABB 框
        vp = getattr(self, 'viewport_panel', None)
        if vp is not None:
            global_pos = vp.mapToGlobal(QPoint(0, 0))
            # 右下角留 12px 边距
            x = global_pos.x() + max(0, vp.width() - 180 - 12)
            y = global_pos.y() + max(0, vp.height() - 90 - 12)
            # 确保不超出屏幕
            from PySide6.QtWidgets import QApplication
            screen = QApplication.primaryScreen().availableGeometry()
            x = min(max(x, screen.left()), screen.right() - 180)
            y = min(max(y, screen.top()), screen.bottom() - 90)
            card.move(x, y)
        else:
            card.move(100, 100)

        card.show()
        card.raise_()
        # 不调用 activateWindow()，避免抢走 Open3D 视口焦点导致渲染暂停
        self._roi_confirm_card = card

    def _on_roi_confirm_no(self, card):
        """选项 A【否】：清除选框，回到闲置状态，可再次框选。"""
        try:
            card.close()
            card.deleteLater()
        except Exception:
            pass
        self._roi_confirm_card = None
        self._pending_roi_data = None
        try:
            self.project_operation_service._clear_roi_visuals()
        except Exception:
            pass
        nav = getattr(self, 'step_nav', None)
        if nav is not None:
            nav.mark_pending(self.STEP_BOX_DETECT)

    def _on_roi_confirm_yes(self, card):
        """选项 B【是】：复用原有 ROI 回调链路，再触发立面提取算法。"""
        try:
            card.close()
            card.deleteLater()
        except Exception:
            pass
        self._roi_confirm_card = None
        data = getattr(self, '_pending_roi_data', None)
        self._pending_roi_data = None
        if data is None:
            return
        # 复用原有 ROI 处理：3D AABB 生成、视口框体渲染、set_detection_roi
        self.project_operation_service._on_roi_selected(
            data['min_bound'], data['max_bound'], data['indices'],
            data['p1'], data['p2'],
        )
        # 启动立面提取进度弹窗并触发算法
        self._begin_step_task(self.STEP_BOX_DETECT)
        self.project_operation_service.facade_detection()

    # ------------------------------------------------------------------
    # 立面结果面板与质量评估（零逻辑变化搬运自 main_window.py）
    # ------------------------------------------------------------------
    @property
    def _quality_reports(self):
        return self.facade_quality_controller.quality_reports

    @property
    def _quality_result_cache(self):
        return self.facade_quality_controller.quality_result_cache

    @property
    def _active_quality_worker(self):
        return self.facade_quality_controller.active_quality_worker

    @_active_quality_worker.setter
    def _active_quality_worker(self, value):
        self.facade_quality_controller.active_quality_worker = value

    def _quality_context(self):
        # 供 FacadeQualityController 实时判断质量结果是否已过期。
        return (
            getattr(self.current_project, 'project_id', None),
            self._project_generation,
        )

    def _quality_profile_provider(self):
        """质量算子的唯一参数来源：当前项目持久化快照。"""
        import json
        from services.inspection_profile import InspectionProfileService
        project = getattr(self, 'current_project', None)
        raw = getattr(project, 'inspection_params_json', None)
        if raw:
            try:
                values = json.loads(raw) if isinstance(raw, str) else dict(raw)
                standard_id = values.get('standard_id')
                profile = InspectionProfileService.get(standard_id)
                if profile is not None:
                    aliases = {
                        'ruler_length_m': 'measure_height_m',
                        'step_longitudinal_m': 'scan_step_m',
                        'step_transverse_m': 'step_size_m',
                    }
                    snapshot = {}
                    for key in (
                        'interval_size_m', 'window_size_m', 'step_size_m',
                        'measure_height_m', 'min_points', 'flatness_limit_mm',
                        'verticality_limit_mm', 'ruler_width_m', 'select_band_m',
                        'hole_band_m', 'bin_size_m', 'top_q', 'sor_enabled',
                        'sor_sigma', 'sor_k', 'sor_method', 'sor_w_weight',
                        'scan_step_m', 'max_hole_ratio'):
                        source = next((name for name, target in aliases.items() if target == key), key)
                        if source in values:
                            snapshot[key] = values[source]
                    return replace(profile, **snapshot)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return getattr(self, '_inspection_profile', None)

    def _quality_grid_size(self):
        profile = self._quality_profile_provider()
        return float(getattr(profile, 'interval_size_m', 20.0))

    def _connect_facade_quality_controller(self):
        controller = self.facade_quality_controller
        controller.status_message.connect(self._show_quality_status)
        controller.status_cleared.connect(self._clear_quality_status)
        controller.warning_requested.connect(self._show_quality_warning)
        controller.info_requested.connect(self._show_quality_info)
        controller.report_preview_refresh_requested.connect(
            self._refresh_report_preview)
        controller.heatmap_button_refresh_requested.connect(
            self._refresh_heatmap_button_state)
        controller.show_dialog_requested.connect(self._show_quality_dialog)

    def _show_quality_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def _clear_quality_status(self):
        self.statusBar().clearMessage()

    def _show_quality_warning(self, message):
        QMessageBox.warning(self, '质量评估', message)

    def _show_quality_info(self, message):
        QMessageBox.information(self, '质量评估', message)

    @staticmethod
    def _quality_double(value, minimum, maximum, step):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setDecimals(3)
        box.setValue(value)
        return box

    def _reset_quality_parameters(self):
        profile = getattr(self, '_inspection_profile', None)
        if profile is None:
            return
        self.quality_length_spin.setValue(profile.measure_height_m)
        self.quality_step_spin.setValue(profile.scan_step_m)
        self.quality_width_spin.setValue(profile.ruler_width_m)
        self.quality_select_band_spin.setValue(profile.select_band_m)
        self.quality_hole_band_spin.setValue(profile.hole_band_m)
        self.quality_bin_size_spin.setValue(profile.bin_size_m)
        self.quality_top_q_spin.setValue(profile.top_q)
        self.quality_sor_check.setChecked(profile.sor_enabled)
        self.quality_sor_sigma_spin.setValue(profile.sor_sigma)
        self.quality_sor_k_spin.setValue(profile.sor_k)
        method_index = self.quality_sor_method_combo.findData(profile.sor_method)
        if method_index >= 0:
            self.quality_sor_method_combo.setCurrentIndex(method_index)
        self.quality_sor_w_weight_spin.setValue(profile.sor_w_weight)
        self.quality_max_hole_ratio_spin.setValue(profile.max_hole_ratio)
        self.quality_min_points_spin.setValue(profile.min_points)

    def _quality_profile_snapshot(self, profile):
        if profile is None or not hasattr(self, 'quality_length_spin'):
            return profile
        return replace(
            profile,
            measure_height_m=self.quality_length_spin.value(),
            scan_step_m=self.quality_step_spin.value(),
            ruler_width_m=self.quality_width_spin.value(),
            select_band_m=self.quality_select_band_spin.value(),
            hole_band_m=self.quality_hole_band_spin.value(),
            bin_size_m=self.quality_bin_size_spin.value(),
            top_q=self.quality_top_q_spin.value(),
            sor_enabled=self.quality_sor_check.isChecked(),
            sor_sigma=self.quality_sor_sigma_spin.value(),
            sor_k=self.quality_sor_k_spin.value(),
            sor_method=str(self.quality_sor_method_combo.currentData() or 'local'),
            sor_w_weight=self.quality_sor_w_weight_spin.value(),
            max_hole_ratio=self.quality_max_hole_ratio_spin.value(),
            min_points=self.quality_min_points_spin.value())

    def _show_facade_results(self, results: list[dict]):
        """项目操作页：显示全部立面，保留【标记处理】按钮（传统模式）。"""
        results = self.facade_quality_controller.process_facade_results(results)
        count = len(results)
        self.lbl_facade_summary.setText(f'检测立面数量：{count}')
        if not results:
            self.list_facades.clear()
            self._refresh_heatmap_button_state()
            self.facade_quality_controller.set_latest_results([])
            self._refresh_report_preview()
            return
        self.list_facades.clear()
        for index, f in enumerate(results, 1):
            display_no = int(f.get('display_no') or index)
            f['display_no'] = display_no
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setSizeHint(QSize(0, 40))
            self.list_facades.addItem(item)
            row = self._create_facade_list_row(f, display_no, review_mode=False)
            self.list_facades.setItemWidget(item, row)

        self.facade_quality_controller.set_latest_results(results)
        self._refresh_heatmap_button_state()
        self._refresh_report_preview()

    def _refresh_heatmap_button_state(self):
        button = getattr(self, 'btn_heatmap_toggle', None)
        if button is None:
            return
        controller = self.facade_quality_controller
        enabled = bool(controller.compatible_quality_results())
        button.setEnabled(enabled)
        mode = controller.heatmap_mode
        current = '平整度' if mode == 'flatness' else '垂直度'
        next_mode = '垂直度' if mode == 'flatness' else '平整度'
        button.setText(f'热力切换显示（当前：{current}）')
        button.setToolTip(f'点击切换至{next_mode}热力映射')

    def _toggle_heatmap_display(self):
        self.facade_quality_controller.toggle_heatmap_display()

    def _set_facade_preview_status(self, facade, button, status):
        # review_status 是唯一的规范运行时字段。
        facade['review_status'] = status
        for row in range(self.list_facades.count()):
            item = self.list_facades.item(row)
            payload = item.data(Qt.ItemDataRole.UserRole) or {}
            if int(payload.get('id', -1)) == int(facade.get('id', -2)):
                self.list_facades.setCurrentItem(item)
                item.setData(Qt.ItemDataRole.UserRole, facade)
                break
        for current in (self.project_operation_service.last_facade_results or []):
            if int(current.get('id', -1)) == int(facade.get('id', -2)):
                current['review_status'] = status
                facade = current
                break
        item = self.list_facades.currentItem()
        if item is not None and int((item.data(Qt.ItemDataRole.UserRole) or {}).get('id', -1)) == int(facade.get('id', -2)):
            item.setData(Qt.ItemDataRole.UserRole, facade)
        button.setText('处理' if status == 'complete' else '不处理')
        if hasattr(self.project_operation_service, 'persist_facade_review_status'):
            self.project_operation_service.persist_facade_review_status(facade)

    def _toggle_facade_review_status(self, facade, button):
        """Toggle pending/incomplete -> complete, complete -> incomplete."""
        current = self._facade_review_status(facade)
        target = 'incomplete' if current == 'complete' else 'complete'
        self._set_facade_preview_status(facade, button, target)

    @staticmethod
    def _facade_review_status(facade):
        """Canonical status reader; tolerate legacy/null review_status."""
        value = (facade or {}).get('review_status')
        if value not in {'complete', 'incomplete'}:
            value = (facade or {}).get('preview_status')
        return value if value in {'complete', 'incomplete'} else 'pending'

    def _collect_complete_facades(self) -> tuple[list[dict], int]:
        """返回 (完整立面列表, 被跳过的数量)。

        供批量评估按钮与其它潜在入口复用，避免各处重复实现筛选逻辑。
        """
        latest = self.project_operation_service.last_facade_results or []
        complete = [
            f for f in latest
            if self._facade_review_status(f) == 'complete'
        ]
        return complete, len(latest) - len(complete)
    
    def _evaluate_selected_facade(self):
        """触屏友好的批量评估入口（基于“处理”标记，不依赖多选）。"""
        facades, skipped = self._collect_complete_facades()
        if not facades:
            if skipped == 0:
                QMessageBox.information(
                    self, '质量评估',
                    '当前没有可评估的立面，请先执行立面检测。')
            else:
                QMessageBox.information(
                    self, '质量评估',
                    '尚未有任何立面被标记为“处理”。\n'
                    '请先在结果列表中点击“标记处理”，再执行质量检测。')
            return
        if skipped:
            self.statusBar().showMessage(
                f'已自动跳过 {skipped} 个未标记为处理的立面', 4000)
        if len(facades) == 1:
            self.facade_quality_controller.evaluate_facade(facades[0])
        else:
            self.facade_quality_controller.evaluate_facades_batch(facades)

    def _on_facade_item_clicked(self, item):
        f = item.data(Qt.ItemDataRole.UserRole)
        if not f:
            return
        cloud = self.facade_quality_controller.active_cloud_name()
        if not cloud:
            return

        self.render_facade.select_facade(cloud, int(f.get('id', 0)))
        self.statusBar().showMessage(f"已选中立面 {int(f.get('display_no', 1))}，请使用“评估”按钮执行质量检测", 3000)
        return

    def _show_quality_dialog(self, cloud, facade, quality):
        facade_id = int(facade.get('id', 0))
        facade_no = int(facade.get('display_no', facade_id))

        print(f'[PCFD] ui.show_dialog facade_id={facade_id} facade_no={facade_no}', flush=True)

        def _export_context(display_quality):
            context = display_quality.get('__export_context') or {}
            if context.get('points') is not None and context.get('results_dir'):
                return context
            # 历史报告会刻意不保留大型点数组。从当前处理中的数据集重建导出输入。
            try:
                dataset = self.facade_service.get_dataset(cloud)
                proxy_ids = np.asarray(
                    facade.get('proxy_indices') or facade.get('inlier_indices') or [],
                    dtype=np.int64)
                if len(proxy_ids) and dataset.index.has_source_mapping():
                    raw_indices = dataset.index.proxy_to_source_ids(
                        proxy_ids, deduplicate=True)
                else:
                    raw_indices = proxy_ids
                raw_indices = raw_indices[(raw_indices >= 0) &
                                          (raw_indices < len(dataset.processed_raw_points))]
                if len(raw_indices) == 0:
                    print(f'[PCFD] export_context_failed facade_id={facade_id} '
                          'reason=no_facade_source_indices', flush=True)
                    return context
                points = np.asarray(dataset.processed_raw_points)[raw_indices]
                source_colors = dataset.index.get_source_colors()
                colors = (np.asarray(source_colors)[raw_indices]
                          if source_colors is not None and
                          len(source_colors) > int(raw_indices.max()) else None)
                if colors is None:
                    colors = np.tile(np.asarray(
                        self.render_service.facade_color_for(facade), dtype=float),
                        (len(points), 1))
                else:
                    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
                    # 确保分段立面的颜色在导出的基础图层中可见，且不受源RGB数据是否可用影响。
                    colors[:] = np.asarray(
                        self.render_service.facade_color_for(facade), dtype=float)
                project_uuid = getattr(self.current_project, 'project_id', None)
                if project_uuid:
                    results_dir = self.facade_quality_controller._quality_results_dir(
                        project_uuid, facade)
                else:
                    results_dir = None
                return {'results_dir': results_dir, 'points': points,
                        'colors': colors}
            except Exception as exc:
                print(f'[PCFD] export_context_failed facade_id={facade_id} '
                      f'error={exc!r}', flush=True)
                return context

        def _show_effect(mode='flatness'):
            """仅在三维视口渲染热力图，不自动导出 PNG/PDF（导出由用户手动触发）."""
            try:
                display_quality = dict(quality) if isinstance(quality, dict) else {}
                display_quality['heatmap_mode'] = mode
                self.render_facade.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service.index_service)
                self.statusBar().showMessage(
                    f'已切换至 {mode} 检测效果渲染', 3000)
            except Exception as e:
                print(f'[PCFD] ui.show_effect_error facade_id={facade_id} error={e}', flush=True)

        # 检测完成后自动导出全部热力图
        if isinstance(quality, dict) and not quality.get('__auto_exported'):
            try:
                context = _export_context(quality)
                if context.get('results_dir') and context.get('points') is not None:
                    exporter = ResultExportService()
                    exported = exporter.export_all_heatmaps(
                        context.get('results_dir'), facade_no,
                        context.get('points'), context.get('colors'),
                        quality)
                    quality['__auto_exported'] = True
                    # 将导出路径写回 quality，供 PDF 生成时直接读取
                    if exported:
                        quality.setdefault('__export_context', {})
                        quality['__export_context']['heatmaps'] = exported
                        quality['__export_context']['results_dir'] = context.get('results_dir')
            except Exception as e:
                print(f'[PCFD] ui.auto_export_error facade_id={facade_id} error={e}', flush=True)

        def _restore():
            try:
                results = self.project_operation_service.last_facade_results
                self.render_facade.restore_highlight(cloud, results or [])
            except Exception as e:
                print(f'[PCFD] ui.restore_error facade_id={facade_id} error={e}', flush=True)

        label = f'立面 {facade_no}'
        project_name = getattr(self.current_project, 'name', '') if self.current_project else ''

        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_not_dict facade_id={facade_id} type={type(quality)}', flush=True)
            quality = {}

        try:
            previous = self._quality_dialog
            if previous is not None and previous.isVisible():
                previous.close()

            dlg = FacadeQualityDialog(self, label, quality,
                                      project_name=project_name,
                                      on_show_colors=_show_effect,
                                      on_restore_colors=_restore)
            self._quality_dialog = dlg

            def _dialog_finished(result_code, dialog=dlg):
                if self._quality_dialog is dialog:
                    self._quality_dialog = None
                print(
                    f'[PCFD] ui.dialog_closed facade_id={facade_id} '
                    f'result={result_code}',
                    flush=True,
                )

            dlg.finished.connect(_dialog_finished)
            dlg.open()
        except Exception as e:
            print(f'[PCFD] ui.dialog_exception facade_id={facade_id} error={e}', flush=True)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, '质量评估',
                f'显示质量结果时出错：\n{e}')
