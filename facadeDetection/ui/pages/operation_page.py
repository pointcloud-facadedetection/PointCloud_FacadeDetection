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
from services.inspection_profile import InspectionProfileService


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
        viewport_layout.setContentsMargins(0, 0, 0, 0)
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
        viewport_layout.addLayout(viewport_heading_row)

        # 新增检测标准配置沿用当前扁平工作台样式，不再恢复旧页面标题卡片。
        config_bar = QFrame()
        config_bar.setObjectName('inspectionConfigBar')
        config_bar.setProperty('uiRole', 'inspectionConfig')
        config_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        config_layout = QHBoxLayout(config_bar)
        config_layout.setContentsMargins(16, 6, 16, 6)
        config_layout.setSpacing(8)
        config_layout.addWidget(QLabel('墙面标准'))
        self.standard_combo = QComboBox()
        for profile in InspectionProfileService.all():
            self.standard_combo.addItem(
                f'{profile.standard_name} · {profile.version}', profile.standard_id)
        config_layout.addWidget(self.standard_combo)
        self.standard_summary = QLabel()
        self.standard_summary.setObjectName('standardSummary')
        self.standard_summary.setProperty('uiRole', 'supportingText')
        config_layout.addWidget(self.standard_summary, 1)
        config_layout.addWidget(QLabel('区间'))
        self.interval_combo = QComboBox()
        for value in (3.0, 5.0, 10.0, 20.0):
            self.interval_combo.addItem(f'{value:g}m', value)
        self.interval_combo.setCurrentIndex(3)
        config_layout.addWidget(self.interval_combo)
        viewport_layout.addWidget(config_bar)
        self.standard_combo.currentIndexChanged.connect(self._on_standard_changed)
        self._on_standard_changed(0)
        viewport_layout.addWidget(self.render_facade.widget(), 1)

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
        
        self.btn_evaluate_selected = QPushButton('评估选中立面')
        self.btn_evaluate_selected.setToolTip('对右侧列表当前选中的立面执行质量评估')
        self.btn_evaluate_selected.clicked.connect(self._evaluate_selected_facade)
        self.btn_evaluate_selected.setMinimumHeight(34)
        self.btn_evaluate_selected.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.btn_evaluate_selected)

        self.btn_heatmap_toggle = QPushButton('热力切换显示')
        self.btn_heatmap_toggle.setObjectName('btn_heatmap_toggle')
        self.btn_heatmap_toggle.setToolTip('在平整度热力与垂直度热力之间切换')
        self.btn_heatmap_toggle.setMinimumHeight(34)
        self.btn_heatmap_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_heatmap_toggle.clicked.connect(self._toggle_heatmap_display)
        self.btn_heatmap_toggle.setEnabled(False)
        lay.addWidget(self.btn_heatmap_toggle)
        
        # 立面列表
        from PySide6.QtWidgets import QListWidget
        self.list_facades = QListWidget()
        self.list_facades.setObjectName('lstFacades')
        self.list_facades.setSpacing(4)
        self.list_facades.setUniformItemSizes(False)
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
        if not hasattr(self, 'station_list'):
            return
        self.station_list.blockSignals(True)
        self.station_list.clear()
        stations = self.station_service.list_stations()
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
        active_station_id = (active_station_id if active_station_id is not None
                             else getattr(self.station_service, '_active_station_id', None))
        if active_station_id is not None:
            for row in range(self.station_list.count()):
                item = self.station_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == active_station_id:
                    self.station_list.setCurrentItem(item)
                    break

    def _on_station_item_changed(self, item):
        try:
            self._pending_station_selection[item.data(Qt.ItemDataRole.UserRole)] = (
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
        except Exception as exc:
            self.statusBar().showMessage(f'保存站点选择失败：{exc}', 5000)

    def _on_station_clicked(self, item):
        station_id = item.data(Qt.ItemDataRole.UserRole)
        station = next((x for x in self.station_service.list_stations() if x.id == station_id), None)
        if station:
            self.station_service.show_single(station)
            # A station switch changes the processing domain.  Replay only
            # results belonging to the newly active station; never reuse the
            # previous station's facade indices or colours.
            try:
                project_id = getattr(self.current_project, 'project_id', None)
                historical = (self.project_overview_service.load_historical_facades(
                    project_id, station.id) if project_id else [])
                self.project_operation_service.last_facade_results = historical or []
                self._show_facade_results(historical or [])
            except Exception as exc:
                print(f'[PCFD] facade.color_refresh_failed error={exc!r}', flush=True)

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
