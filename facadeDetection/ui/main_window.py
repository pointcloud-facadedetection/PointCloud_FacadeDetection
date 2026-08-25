from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt, QThreadPool
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
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
    QMenu,
    QVBoxLayout,
    QWidget,
    QListWidgetItem,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)
from dataclasses import replace
import numpy as np
from qtwebview2 import QtWebView2Widget

from .widgets.flow_layout import FlowLayout
from services.inspection_review import InspectionReviewService
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.viewport_render_service import ViewportRenderService
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from services.facade.facade_service import FacadeService
from services.report_export import ReportExportService
from config.storage import Storage
from view3d.open3d_viewport import Open3DViewport
from ui.dialogs.facade_quality_dialog import FacadeQualityDialog
from services.inspection_profile import InspectionProfileService
from utils.workers import QualityWorker


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
        ('导入FLS目录', 'btn_import_fls_dir'),
        ('直接上传文件', 'btn_upload_files'),
        ('打开项目', 'btn_open_project'),
        ('选择项目', 'btn_select_project'),
        ('新建项目', 'btn_new_project'),
    ),
    'project_operation': (
        ('改变颜色', 'btn_change_color'),
        ('点云去噪', 'btn_denoise'),
        ('点云配准', 'btn_registration'),
        ('框选检测区域', 'btn_select_detection_area'),
        ('立面检测', 'btn_facade_detection'),
        ('质量检测', 'btn_quality_inspection'),
        ('框选分割', 'btn_box_segmentation'),
        ('计算细节', 'btn_calculate_detail'),
        ('二维-三维对齐', 'btn_align_2d_3d'),
    ),
    'inspection_review': (),
    'report_export': (
        ('打开 PDF', 'btn_open_report_pdf'),
        ('导出质量报告', 'btn_export_quality_report'),
    ),
}

UPLOAD_FILE_FILTER = (
    '项目支持文件 '
    '(*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls '
    '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.dist);;'
    '鐐逛簯鏂囦欢 (*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls *.dist);;'
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '所有文件 (*)'
)

REPORT_PDF_FILTER = 'PDF 文件 (*.pdf)'
REPORT_EMPTY_TITLE = '请选择PDF上传'


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PointCloud FacadeDetection')
        self.resize(1600, 900)
        self.viewport = Open3DViewport()
        # Unified render service for business modules
        self.render_service = ViewportRenderService(self.viewport, db=None)
        self.project_overview_service = ProjectOverviewService(self.viewport, self.render_service, db=None)
        # Facade service uses viewport + render service
        self.pointcloud_service = PointCloudService(self.viewport, self.render_service)
        self.facade_service = FacadeService(
            self.viewport, db=None, render_service=self.render_service,
            pointcloud_service=self.pointcloud_service)
        self.facade_service.set_pointcloud_service(self.pointcloud_service)
        self.render_service.pointcloud_service = self.pointcloud_service
        # Pass facade service into operation scheduler for ROI detection
        self.project_operation_service = ProjectOperationService(
            self.viewport,
            facade_service=self.facade_service,
            pointcloud_service=self.pointcloud_service,
            render_service=self.render_service,
        )
        self.station_service = PointCloudStationService(self.render_service)
        self.project_operation_service.set_station_service(self.station_service)
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        self.page_title_labels = {}
        self._sidebar_collapsed = {'left': False, 'right': False}
        # 默认定位到 data/projects，便于跨机迁移
        try:
            Storage.ensure_base_dirs()
            self._last_upload_directory = str(Storage.PROJECTS_ROOT)
        except Exception:
            self._last_upload_directory = str(Path.home())
        self._header_resize_pending = set()
        self._current_report_pdf_name = None
        self._report_webview_error = None
        self._quality_reports = []
        self._quality_result_cache = {}
        # Quality jobs may hold a large raw-point working set.  The global Qt
        # pool can otherwise start several facade jobs simultaneously and
        # exhaust Windows commit memory while CPUs remain underutilised.
        self._quality_pool = QThreadPool(self)
        self._quality_pool.setMaxThreadCount(1)
        self._active_quality_worker = None
        self._setup_ui()
        self._connect_buttons()
        # Hook: display facade stats in the right dock when results ready
        try:
            self.project_operation_service.on_facade_results = self._show_facade_results
        except Exception:
            pass
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
            elif page_key == 'report_export':
                page = self._create_report_export_page(page_title, page_key)
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
        ) = self._create_sidebar('leftDock', 'left')
        (
            self.right_dock,
            self.right_sidebar_button,
        ) = self._create_sidebar('rightDock', 'right')
        # Prepare right panel layout for results
        try:
            self._init_right_panel_widgets()
        except Exception:
            pass

        viewport_panel = QWidget()
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(12, 10, 12, 10)
        viewport_layout.setSpacing(8)
        viewport_layout.addWidget(self._create_page_title(page_title, page_key))
        config_bar = QFrame()
        config_bar.setObjectName('inspectionConfigBar')
        config_layout = QHBoxLayout(config_bar)
        config_layout.setContentsMargins(10, 6, 10, 6)
        config_layout.addWidget(QLabel('墙面标准'))
        self.standard_combo = QComboBox()
        for profile in InspectionProfileService.all():
            self.standard_combo.addItem(
                f'{profile.standard_name} · {profile.version}', profile.standard_id)
        config_layout.addWidget(self.standard_combo)
        self.standard_summary = QLabel()
        self.standard_summary.setObjectName('standardSummary')
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
        viewport_layout.addWidget(self.viewport.get_widget(), 1)

        self.operation_splitter.addWidget(self.left_dock)
        self.operation_splitter.addWidget(viewport_panel)
        self.operation_splitter.addWidget(self.right_dock)
        self.operation_splitter.setStretchFactor(0, 0)
        self.operation_splitter.setStretchFactor(1, 1)
        self.operation_splitter.setStretchFactor(2, 0)
        # Keep both side panels usable at the default 1600px window width;
        # the result rows contain an action control and must not be squeezed.
        self.operation_splitter.setSizes([230, 940, 300])
        self.operation_splitter.installEventFilter(self)
        layout.addWidget(self.operation_splitter, 1)

        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
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
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        title = QLabel('检测结果')
        title.setStyleSheet('font-weight:600; color:#303641;')
        lay.addWidget(title)
        # summary label
        self.lbl_facade_summary = QLabel('未检测')
        self.lbl_facade_summary.setObjectName('lblFacadeSummary')
        self.lbl_facade_summary.setStyleSheet('color:#5b626d;')
        self.lbl_facade_summary.setMinimumHeight(28)
        lay.addWidget(self.lbl_facade_summary)
        self.btn_evaluate_selected = QPushButton('评估选中立面')
        self.btn_evaluate_selected.setToolTip('对右侧列表当前选中的立面执行质量评估')
        self.btn_evaluate_selected.clicked.connect(self._evaluate_selected_facade)
        lay.addWidget(self.btn_evaluate_selected)
        # 条目点击只选中/高亮，质量计算由条目右侧独立按钮触发。
        from PySide6.QtWidgets import QListWidget
        self.list_facades = QListWidget()
        self.list_facades.setObjectName('lstFacades')
        self.list_facades.setSpacing(5)
        self.list_facades.setUniformItemSizes(False)
        lay.addWidget(self.list_facades, 1)
        self.list_facades.itemClicked.connect(self._on_facade_item_clicked)

        # Keep facade review information above a dedicated, bounded quality
        # configuration area.  Widgets are snapshotted before a worker starts.
        config = QFrame()
        config.setObjectName('qualityParameterPanel')
        config.setFrameShape(QFrame.Shape.StyledPanel)
        config_layout = QVBoxLayout(config)
        config_layout.setContentsMargins(6, 6, 6, 6)
        config_layout.addWidget(QLabel('质量检测参数'))
        form = QFormLayout()
        self.quality_length_spin = self._quality_double(2.0, .1, 20.0, .01)
        self.quality_step_spin = self._quality_double(.05, .005, 2.0, .005)
        self.quality_width_spin = self._quality_double(.055, .005, 1.0, .005)
        self.quality_select_band_spin = self._quality_double(.01, .001, .2, .001)
        self.quality_hole_band_spin = self._quality_double(.02, .001, .5, .001)
        self.quality_sor_check = QCheckBox('启用 SOR 离群剔除')
        self.quality_sor_check.setChecked(True)
        self.quality_sor_sigma_spin = self._quality_double(4.0, 1.0, 20.0, .5)
        self.quality_min_points_spin = QSpinBox()
        self.quality_min_points_spin.setRange(3, 100000)
        self.quality_min_points_spin.setValue(30)
        form.addRow('靠尺长度 (m)', self.quality_length_spin)
        form.addRow('滑移步距 (m)', self.quality_step_spin)
        form.addRow('靠尺宽度 (m)', self.quality_width_spin)
        form.addRow('表面带宽 (m)', self.quality_select_band_spin)
        form.addRow('空洞带宽 (m)', self.quality_hole_band_spin)
        form.addRow(self.quality_sor_check)
        form.addRow('SOR 阈值 σ', self.quality_sor_sigma_spin)
        form.addRow('最小点数', self.quality_min_points_spin)
        config_layout.addLayout(form)
        reset = QPushButton('恢复标准参数')
        reset.clicked.connect(self._reset_quality_parameters)
        config_layout.addWidget(reset)
        lay.addWidget(config)

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
        self.quality_sor_check.setChecked(profile.sor_enabled)
        self.quality_sor_sigma_spin.setValue(profile.sor_sigma)
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
            sor_enabled=self.quality_sor_check.isChecked(),
            sor_sigma=self.quality_sor_sigma_spin.value(),
            min_points=self.quality_min_points_spin.value())

    def _show_facade_results(self, results: list[dict]):
        self._quality_result_cache.clear()
        # Detection is a data operation; this GUI-thread callback is the single
        # authoritative point at which the proxy facade colors are committed.
        try:
            cloud = (self.pointcloud_service.resolve_processing_cloud()
                     if self.pointcloud_service is not None else None)
            if cloud and self.render_service is not None:
                self.render_service.highlight_facades(cloud, results or [])
        except Exception as exc:
            print(f'[PCFD] facade.color_refresh_failed error={exc!r}', flush=True)
        # Detection can be reloaded/merged more than once; the panel is a
        # stable review list, not an append-only event log.
        unique = {}
        for facade in results or []:
            fid = int(facade.get('id', len(unique)))
            if fid not in unique:
                unique[fid] = facade
            else:
                unique[fid].update({k: v for k, v in facade.items()
                                    if k not in ('preview_status', 'preview_status_source')})
        results = list(unique.values())
        count = len(results)
        self.lbl_facade_summary.setText(f'检测立面数量：{count}')
        if not results:
            self.list_facades.clear()
            return
        self.list_facades.clear()
        for display_no, f in enumerate(results, 1):
            # Keep algorithm id zero-based for persistence/log correlation;
            # display_no is the human-facing one-based label.
            f['display_no'] = display_no
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.list_facades.addItem(item)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(5)
            info = QLabel(f"立面 {display_no}   点数 {int(f.get('point_count') or 0):,}")
            info.setToolTip(f"业务索引 facade_id={int(f.get('id', 0))}")
            row_layout.addWidget(info, 1)
            # Use exactly the same discrete palette as the 3D renderer.
            color = self.render_service.facade_color_for(f, display_no)
            swatch = QLabel()
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet('background-color: rgb(%d,%d,%d); border:1px solid #555;' %
                                 tuple(int(max(0, min(1, x)) * 255) for x in color))
            swatch.setToolTip('该立面在视口中的显示颜色')
            row_layout.addWidget(swatch)
            # Keep the compact row, but preserve the mandatory manual review
            # gate: this control never starts quality computation directly.
            status = self._facade_review_status(f)
            action_button = QPushButton(
                '完整' if status == 'complete' else '确认完整')
            action_button.setFixedWidth(76)
            action_button.setToolTip('点击切换完整/不完整；仅完整立面允许质量计算')
            action_button.clicked.connect(
                lambda _=False, obj=f, button=action_button:
                self._toggle_facade_review_status(obj, button))
            row_layout.addWidget(action_button)
            row.setMinimumHeight(32)
            row.setMaximumHeight(38)
            self.list_facades.setItemWidget(item, row)

        # cache for dialog use
        self._latest_facade_results = results
        self.project_operation_service._last_facade_results = results

    def _set_facade_preview_status(self, facade, button, status):
        # review_status is the only canonical runtime field.  Keep the
        # QListWidget payload and operation-service object identical by ID.
        facade['review_status'] = status
        # Clicking the status control also selects its owning list row.
        for row in range(self.list_facades.count()):
            item = self.list_facades.item(row)
            payload = item.data(Qt.ItemDataRole.UserRole) or {}
            if int(payload.get('id', -1)) == int(facade.get('id', -2)):
                self.list_facades.setCurrentItem(item)
                item.setData(Qt.ItemDataRole.UserRole, facade)
                break
        for current in (getattr(self.project_operation_service, '_last_facade_results', None) or []):
            if int(current.get('id', -1)) == int(facade.get('id', -2)):
                current['review_status'] = status
                facade = current
                break
        item = self.list_facades.currentItem()
        if item is not None and int((item.data(Qt.ItemDataRole.UserRole) or {}).get('id', -1)) == int(facade.get('id', -2)):
            item.setData(Qt.ItemDataRole.UserRole, facade)
        button.setText('完整' if status == 'complete' else '不完整')
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

    def _evaluate_selected_facade(self):
        item = self.list_facades.currentItem()
        if item is None:
            QMessageBox.information(self, '质量评估', '请先在右侧结果列表中选择一个立面。')
            return
        facade = item.data(Qt.ItemDataRole.UserRole)
        if facade:
            if self._facade_review_status(facade) != 'complete':
                QMessageBox.information(self, '质量评估', '请先人工确认该立面为完整立面。')
                return
            current = next((f for f in (getattr(self.project_operation_service, '_last_facade_results', None) or [])
                            if int(f.get('id', -1)) == int(facade.get('id', -2))), facade)
            self._evaluate_facade(current)

    def _active_cloud_name(self) -> str | None:
        try:
            service = getattr(self, 'pointcloud_service', None)
            if service is not None:
                resolved = service.resolve_processing_cloud()
                if resolved:
                    return resolved
            names = self.viewport.get_cloud_names()
            return names[-1] if names else None
        except Exception:
            return None

    def _quality_cache_key(self, cloud, facade, profile, grid_size):
        facade_id = int((facade or {}).get('id', 0))
        standard_id = getattr(profile, 'standard_id', None)
        flatness_limit_mm = getattr(profile, 'flatness_limit_mm', None)
        verticality_limit_mm = getattr(profile, 'verticality_limit_mm', None)
        window_size_m = getattr(profile, 'window_size_m', None)
        step_size_m = getattr(profile, 'step_size_m', None)
        measure_height_m = getattr(profile, 'measure_height_m', None)
        domain = facade.get('measurement_indices') or facade.get('voxel_ids') or []
        runtime_profile = self._quality_profile_snapshot(profile)
        parameter_snapshot = (runtime_profile.snapshot()
                              if runtime_profile is not None else {})
        return (
            str(cloud or ''), facade_id, standard_id,
            float(grid_size), flatness_limit_mm, verticality_limit_mm,
            window_size_m, step_size_m, measure_height_m,
            tuple(sorted(parameter_snapshot.items())),
            len(domain), hash(tuple(domain[:32])),
        )

    def _show_quality_dialog(self, cloud, facade, quality):
        facade_id = int(facade.get('id', 0))
        facade_no = int(facade.get('display_no', facade_id))

        print(f'[PCFD] ui.show_dialog facade_id={facade_id} facade_no={facade_no}', flush=True)

        def _show_effect(mode='flatness'):
            try:
                display_quality = dict(quality) if isinstance(quality, dict) else {}
                display_quality['heatmap_mode'] = mode
                self.render_service.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service._index_service)
            except Exception as e:
                print(f'[PCFD] ui.show_effect_error facade_id={facade_id} error={e}', flush=True)

        def _restore():
            try:
                results = getattr(self.project_operation_service,
                                  '_last_facade_results', None)
                self.render_service.restore_highlight(cloud, results or [])
            except Exception as e:
                print(f'[PCFD] ui.restore_error facade_id={facade_id} error={e}', flush=True)

        label = f"#{facade_no} (ID:{facade_id}) {str(facade.get('type_label') or facade.get('type') or '-')}"

        # Ensure quality is a dict
        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_not_dict facade_id={facade_id} type={type(quality)}', flush=True)
            quality = {}

        try:
            dlg = FacadeQualityDialog(self, label, quality, 
                                      on_show_colors=_show_effect, 
                                      on_restore_colors=_restore)
            dlg.exec()
            print(f'[PCFD] ui.dialog_closed facade_id={facade_id}', flush=True)
        except Exception as e:
            print(f'[PCFD] ui.dialog_exception facade_id={facade_id} error={e}', flush=True)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, '质量评估', 
                f'显示质量结果时出错：\n{e}')

    def _on_facade_item_clicked(self, item):
        f = item.data(Qt.ItemDataRole.UserRole)
        if not f:
            return
        cloud = self._active_cloud_name()
        if not cloud:
            return
        # Selection remains an explicit, non-expensive interaction.
        self.render_service.select_facade(cloud, int(f.get('id', 0)))
        self.statusBar().showMessage(f"已选中立面 {int(f.get('display_no', 1))}，请使用“评估”按钮执行质量检测", 3000)
        return

    def _evaluate_facade(self, f):
        cloud = self._active_cloud_name()
        if not cloud:
            return
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            QMessageBox.warning(self, '质量评估', '请先选择项目。')
            return
        try:
            results_dir = Storage.ensure_project_dirs(project_uuid)['results']
        except Exception as exc:
            QMessageBox.critical(self, '结果目录', f'无法创建项目 results 目录：{exc}')
            return

        # FIX: Ensure facade dict has stable id and display_no
        facade_copy = dict(f)
        facade_id = int(facade_copy.get('id', 0))
        facade_no = int(facade_copy.get('display_no', facade_id))
        facade_copy['id'] = facade_id
        facade_copy['display_no'] = facade_no

        print(f'[PCFD] ui.evaluate_start facade_id={facade_id} facade_no={facade_no} '
              f'cloud={cloud}', flush=True)

        # Snapshot all UI-owned values before submitting. The worker must not
        # access widgets or the Open3D viewport while running.
        profile = self._quality_profile_snapshot(
            getattr(self, '_inspection_profile', None))
        grid_size = float(self.interval_combo.currentData())
        cache_key = self._quality_cache_key(cloud, facade_copy, profile, grid_size)
        cached_quality = self._quality_result_cache.get(cache_key)
        if cached_quality:
            self.statusBar().showMessage('已命中质量结果缓存', 3000)
            self._show_quality_dialog(cloud, facade_copy, cached_quality)
            return
        kwargs = {'profile': profile,
                  'grid_size': grid_size,
                  'results_dir': results_dir}
        self._quality_request_token = getattr(self, '_quality_request_token', 0) + 1
        token = self._quality_request_token
        self._quality_request_cache_key = cache_key
        self.statusBar().showMessage(f'正在计算立面 #{facade_no} 质量指标...')

        # FIX: Pass facade_copy with stable IDs to worker
        worker = QualityWorker(self.facade_service, cloud, facade_copy, kwargs)
        self._active_quality_worker = worker
        worker.signals.finished.connect(
            lambda facade, quality: self._on_quality_finished(token, cloud, facade, quality))
        self._quality_pool.start(worker)
        return

    def _on_quality_failed(self, token, error):
        if token != getattr(self, '_quality_request_token', -1):
            return

        self._active_quality_worker = None
        self.statusBar().showMessage('质量计算失败')
        QMessageBox.warning(self, '质量评估', f'质量计算失败：{error}')

    def _on_quality_finished(self, token, cloud, f, quality):
        """Handle quality computation completion with full state machine."""
        facade_no = int(f.get('display_no', f.get('id', 0)))

        print(f'[PCFD] ui.quality_finished token={token} '
              f'facade_no={facade_no}', flush=True)

        if token != getattr(self, '_quality_request_token', -1):
            print(f'[PCFD] ui.quality_stale token={token} ignored', flush=True)
            return

        self.statusBar().clearMessage()

        # State 1: Worker returned None (exception or unexpected failure)
        if quality is None:
            print(f'[PCFD] ui.quality_none facade_id={facade_no}', flush=True)
            QMessageBox.warning(self, '质量评估', 
                f'立面 #{facade_no} 质量计算失败：未返回结果。请检查日志。')
            return

        # Ensure quality is a dict
        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_invalid_type facade_id={facade_no} '
                  f'type={type(quality)}', flush=True)
            QMessageBox.warning(self, '质量评估',
                f'立面 #{facade_no} 质量计算返回异常类型：{type(quality)}')
            return

        # Cache the result regardless of ok status
        cache_key = getattr(self, '_quality_request_cache_key', None)
        if cache_key is not None:
            self._quality_result_cache[cache_key] = quality

        # Update reports list
        self._quality_reports = [r for r in self._quality_reports
                                 if (r.get('facade') or {}).get('display_no') != facade_no]
        self._quality_reports.append({'facade': f, 'quality': quality})

        # State 2: Algorithm returned ok=False (narrow facade, no windows, etc.)
        if not quality.get('ok', True):
            error_reason = quality.get('reason', 'unknown')
            error_message = quality.get('message', f'质量计算失败：{error_reason}')
            print(f'[PCFD] ui.quality_error facade_id={facade_no} '
                  f'reason={error_reason} message={error_message}', flush=True)

            # Show info message but still open dialog for diagnostics
            QMessageBox.information(self, '质量评估', 
                f'立面 #{facade_no} 质量评估结果：\n\n{error_message}')

            # Open dialog even for error results so user can see diagnostics
            QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))
            return

        # State 3: ok=True but no valid windows
        overall = quality.get('overall') or {}
        window_count = int(overall.get('candidate_window_count', 0) or 0)
        valid_count = int(overall.get('quality_valid_window_count', 0) or 0)

        if valid_count <= 0:
            print(f'[PCFD] ui.quality_no_valid_windows facade_id={facade_no} '
                  f'candidates={window_count}', flush=True)
            QMessageBox.information(self, '质量评估',
                f'立面 #{facade_no} 质量计算完成，但未找到有效检测窗口。\n'
                f'候选窗口数：{window_count}\n'
                f'可能原因：立面尺寸过小、点云密度不足或存在大面积空洞。')
            QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))
            return

        # State 4: Success with valid windows
        print(f'[PCFD] ui.quality_success facade_id={facade_no} '
              f'windows={window_count} valid={valid_count} '
              f'intervals={len(quality.get("intervals", []))}', flush=True)
        QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))

    def _export_quality_report(self):
        """Compatibility placeholder; PDF generation is intentionally disabled."""
        QMessageBox.information(self, '导出报告', 'PDF 报告导出功能暂未启用。')

    def _create_report_export_page(self, page_title, page_key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._create_page_header(page_key))

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(12)
        page_title_label = self._create_page_title(page_title, page_key)
        page_title_label.setText(REPORT_EMPTY_TITLE)
        content_layout.addWidget(page_title_label)

        self.report_pdf_status_label = QLabel('PDF未加载')
        self.report_pdf_status_label.setObjectName('reportPdfStatusLabel')
        self.report_pdf_status_label.setStyleSheet(
            'color: #656c77; font-size: 13px;'
        )
        content_layout.addWidget(self.report_pdf_status_label)

        self.report_webview = QtWebView2Widget(url='about:blank', debug=False,
                                              context_menus=False,
                                              background_color='#f5f7fa', parent=page)
        self.report_webview.setObjectName('reportPdfWebView')
        content_layout.addWidget(self.report_webview, 1)

        layout.addWidget(content, 1)
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

    def _create_sidebar(self, object_name, side):
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

        toggle_button = QToolButton()
        toggle_button.setObjectName(f'btn_collapse_{side}_sidebar')
        toggle_button.setText('◀' if side == 'left' else '▶')
        label = '左侧栏' if side == 'left' else '右侧栏'
        toggle_button.setToolTip(f'收起{label}')
        toggle_button.setAccessibleName(f'收起{label}')
        toggle_button.setFixedSize(28, 28)
        toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        if side == 'left':
            title_layout.addStretch(1)
        title_layout.addWidget(
            toggle_button,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        if side == 'right':
            title_layout.addStretch(1)
        sidebar_layout.addWidget(title_bar)

        panel = QWidget()
        panel.setObjectName(f'{object_name}Panel')
        sidebar_layout.addWidget(panel, 1)
        if side == 'left':
            content = QVBoxLayout(panel)
            content.setContentsMargins(8, 8, 8, 8)
            content.addWidget(QLabel('多站点 PLY 文件管理'))
            from PySide6.QtWidgets import QListWidget
            self.station_list = QListWidget()
            self.station_list.setWordWrap(False)
            self.station_list.setToolTip('单击站点切换视图；复选框用于多选')
            self.station_list.itemClicked.connect(self._on_station_clicked)
            self.station_list.itemChanged.connect(self._on_station_item_changed)
            content.addWidget(self.station_list, 1)
            buttons = QHBoxLayout()
            delete_button = QPushButton('删除站点')
            merge_button = QPushButton('合并显示')
            delete_button.clicked.connect(self._delete_stations)
            merge_button.clicked.connect(self._merge_stations)
            buttons.addWidget(delete_button)
            buttons.addWidget(merge_button)
            content.addLayout(buttons)
        sidebar.setMinimumWidth(180)
        sidebar.setMaximumWidth(260)
        sidebar.setProperty('expandedWidth', 210)
        return sidebar, toggle_button

    def _refresh_station_panel(self):
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

    def _on_station_item_changed(self, item):
        try:
            self.station_service.set_selected(
                item.data(Qt.ItemDataRole.UserRole),
                item.checkState() == Qt.CheckState.Checked)
        except Exception as exc:
            self.statusBar().showMessage(f'保存站点选择失败：{exc}', 5000)

    def _on_station_clicked(self, item):
        station_id = item.data(Qt.ItemDataRole.UserRole)
        station = next((x for x in self.station_service.list_stations() if x.id == station_id), None)
        if station:
            self.station_service.show_single(station)

    def _delete_stations(self):
        try:
            self.station_service.delete_selected()
            self._refresh_station_panel()
            self.station_service.restore_view()
        except Exception as exc:
            QMessageBox.warning(self, '删除站点', str(exc))

    def _merge_stations(self):
        try:
            self.station_service.merge_selected()
            self._refresh_station_panel()
        except Exception as exc:
            QMessageBox.warning(self, '合并站点', str(exc))

    def _create_sidebar_expand_button(self, side):
        button = QToolButton(self.operation_page)
        button.setObjectName(f'btn_expand_{side}_sidebar')
        button.setText('▶' if side == 'left' else '◀')
        label = '左侧栏' if side == 'left' else '右侧栏'
        button.setToolTip(f'展开{label}')
        button.setAccessibleName(f'展开{label}')
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
        self._update_window_title(page_key)

    def _connect_buttons(self):
        overview_actions = {
            'btn_import_fls_dir': self._open_import_fls_directory,
            'btn_upload_files': self._open_upload_file_dialog,
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_change_color': self.project_operation_service.change_color,
            'btn_denoise': self.project_operation_service.denoise,
            'btn_registration': self._run_station_registration,
            'btn_select_detection_area': (
                self.project_operation_service.select_detection_area
            ),
            'btn_facade_detection': (
                self.project_operation_service.facade_detection
            ),
            'btn_quality_inspection': (
                self.project_operation_service.quality_inspection
            ),
            'btn_box_segmentation': (
                self.project_operation_service.box_segmentation
            ),
            'btn_calculate_detail': (
                self.project_operation_service.calculate_detail
            ),
            'btn_align_2d_3d': self.project_operation_service.align_2d_3d,
        }
        report_actions = {
            'btn_open_report_pdf': self._open_report_pdf,
            'btn_export_quality_report': self._export_quality_report,
        }
        all_actions = {
            **overview_actions,
            **pointcloud_actions,
            **report_actions,
        }
        for button_name, callback in all_actions.items():
            self.header_buttons[button_name].clicked.connect(callback)

        self.left_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar('left', self.left_dock, self.left_sidebar_expand_button)
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._collapse_sidebar('right', self.right_dock, self.right_sidebar_expand_button)
        )
        self.left_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar('left', self.left_dock, self.left_sidebar_expand_button)
        )
        self.right_sidebar_expand_button.clicked.connect(
            lambda: self._expand_sidebar('right', self.right_dock, self.right_sidebar_expand_button)
        )

    def _run_station_registration(self):
        try:
            self.project_operation_service.registration()
            self._refresh_station_panel()
            self.statusBar().showMessage('点云配准完成，已显示注册合并结果。', 5000)
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', str(exc))

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
        try:
            uploaded_paths = self.project_overview_service.upload_files(
                file_paths, self.current_project.project_id)
        except Exception as exc:
            QMessageBox.warning(self, '直接上传文件', f'上传失败：{exc}')
            return
        if not uploaded_paths:
            QMessageBox.warning(self, '直接上传文件', '未成功绑定任何点云文件。')
            return
        self._refresh_project_list()
        self._activate_project(self.current_project)
    def _open_import_fls_directory(self):
        directory_path = QFileDialog.getExistingDirectory(
            self,
            '导入 FLS 目录',
            self._last_upload_directory,
        )
        if not directory_path:
            return
        self._last_upload_directory = directory_path
        self.project_overview_service.import_fls_directory(directory_path, getattr(self.current_project, 'project_id', None))
        self._refresh_project_list()
        if self.current_project is not None:
            self._activate_project(self.current_project)

    def _open_project_directory(self):
        directory_path = QFileDialog.getExistingDirectory(
            self,
            '打开项目文件夹',
            self._last_upload_directory,
        )
        if not directory_path:
            return

        self._last_upload_directory = directory_path
        project = self.project_overview_service.open_project(directory_path)
        self._refresh_project_list()
        self._activate_project(project)

    def _create_project(self):
        from ui.dialogs.project_create_dialog import ProjectCreateDialog
        dlg = ProjectCreateDialog(self)
        result_code = dlg.exec()
        try:
            from PySide6.QtWidgets import QDialog
            accepted_code = int(QDialog.DialogCode.Accepted)
        except Exception:
            accepted_code = 1
        if result_code != accepted_code:
            return
        payload = dlg.values()
        info = self.project_overview_service.create_project(
            name=payload.get('name', ''),
            org_unit=payload.get('org_unit'),
            address=payload.get('address'),
            remarks=payload.get('remarks'),
        )
        # Activate the newly created project
        project = type('PC', (object,), info)()
        # Normalize to ProjectCard-like
        project.project_id = info.get('project_uuid')
        project.name = info.get('name')
        project.directory_path = info.get('root_dir')
        self._refresh_project_list()
        self._activate_project(project)

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
        self._activate_project(projects[selected_index])

    def _open_report_pdf(self):
        pdf_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            '选择 PDF 报告',
            self._last_upload_directory,
            REPORT_PDF_FILTER,
        )
        if not pdf_path:
            return

        self._last_upload_directory = str(Path(pdf_path).parent)
        self.show_report_pdf(pdf_path)

    def show_report_pdf(self, pdf_path):
        try:
            document = self.report_export_service.prepare_pdf(pdf_path)
        except ValueError as exc:
            QMessageBox.warning(
                self,
                '打开 PDF',
                str(exc),
            )
            return

        self.report_pdf_status_label.setText(f'正在加载：{document.name}')
        self._current_report_pdf_name = document.name
        self.page_title_labels['report_export'].setText(document.name)
        self._update_window_title('report_export')
        self.report_webview.load_url(document.uri)

    def _on_report_webview_initialized(self, success, error_message):
        if success:
            if self._current_report_pdf_name is None:
                self.report_pdf_status_label.setText('PDF未加载')
            return

        self._report_webview_error = error_message
        if self._current_report_pdf_name is None:
            self.report_pdf_status_label.setText('PDF未加载')
            return

        self.report_pdf_status_label.setText('WebView2 初始化失败')
        QMessageBox.warning(
            self,
            'WebView2 初始化失败',
            error_message or '请检查 Microsoft Edge WebView2 Runtime。',
        )

    def _on_report_pdf_loaded(self):
        if self._current_report_pdf_name is not None:
            self.report_pdf_status_label.setText('PDF 已加载')

    def _refresh_project_list(self):
        while self.project_list_layout.count() > 1:
            item = self.project_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        projects = self.project_overview_service.list_projects()
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
                f'路径：{project.directory_path}'
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
        project = self.project_overview_service.get_project(project_id)
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

        self.project_overview_service.remove_project(project_id)
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
        try:
            self.project_overview_service.activate_project(project.project_id)
        except Exception:
            pass
        self._activate_project(project)

    def _activate_project(self, project):
        # Historical facade geometry is not a formal result.  Never carry it
        # across projects or restore detection-only cache entries.
        try:
            self.project_operation_service.clear_processing_state()
        except Exception:
            pass
        self._quality_result_cache.clear()
        self._quality_reports.clear()
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
        self._set_current_project(project)
        try:
            self.station_service.set_project(getattr(project, 'project_id', None))
            self._refresh_station_panel()
            self.station_service.restore_view()
        except Exception as exc:
            self.statusBar().showMessage(f'站点恢复失败：{exc}', 5000)
        try:
            # propagate active project UUID to operation scheduler for DAL persistence
            self.project_operation_service.set_active_project_uuid(getattr(project, 'project_id', None))
        except Exception:
            pass
        try:
            project_uuid = getattr(project, 'project_id', None)
            if project_uuid:
                historical = self.project_overview_service.load_historical_facades(project_uuid)
                if historical:
                    self.project_operation_service._last_facade_results = historical
                    self._show_facade_results(historical)
        except Exception as exc:
            self.statusBar().showMessage(f'项目历史数据恢复部分失败：{exc}', 5000)
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
            if page_key == 'report_export':
                label_text = self._current_report_pdf_name or REPORT_EMPTY_TITLE
            else:
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
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_window_title()

    def _update_window_title(self, page_key=None):
        if page_key is None:
            page_index = self.centralWidget().currentIndex()
            page_key = PAGE_DEFINITIONS[page_index][1]

        if page_key == 'report_export':
            suffix = self._current_report_pdf_name or ''
        elif self.current_project is not None:
            suffix = self.current_project.name
        else:
            suffix = ''

        title = 'PointCloud FacadeDetection'
        if suffix:
            title = f'{title} - {suffix}'
        self.setWindowTitle(title)

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