from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt, QThreadPool
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
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
    QVBoxLayout,
    QWidget,
)
from qtwebview2 import QtWebView2Widget

from .widgets.flow_layout import FlowLayout
from .widgets.photo_view_widget import PhotoViewWidget
from services.inspection_review import InspectionReviewService
from services.photo_match_service import PhotoMatchService, bgr_to_qimage
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.viewport_render_service import ViewportRenderService
from services.pointcloud_service import PointCloudService
from services.facade.facade_service import FacadeService
from services.report_export import ReportExportService
from config.storage import Storage
from config.settings import Config
from view3d.open3d_viewport import Open3DViewport
from ui.dialogs.facade_quality_dialog import FacadeQualityDialog
from services.inspection_profile import InspectionProfileService
from utils.workers import AutoMatchWorker, MapBackWorker, QualityWorker


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
PHOTO_FILE_FILTER = (
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)'
)


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
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.photo_match_service = PhotoMatchService()
        self._photo_match_facade = None
        self._auto_match_busy = False
        self._auto_match_token = 0
        self._map_back_busy = False
        self._map_back_token = 0
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        self.page_title_labels = {}
        self._sidebar_collapsed = {'left': True, 'right': False}
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
        self._auto_match_pool = QThreadPool(self)
        self._auto_match_pool.setMaxThreadCount(1)
        self._map_back_pool = QThreadPool(self)
        self._map_back_pool.setMaxThreadCount(1)
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
        try:
            self._init_left_panel_widgets()
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
        self.viewport_stage = QWidget()
        self.viewport_stage.setObjectName('viewportStage')
        stage_layout = QVBoxLayout(self.viewport_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)
        self.photo_strip = QWidget()
        self.photo_strip.setObjectName('photoStrip')
        strip_layout = QHBoxLayout(self.photo_strip)
        strip_layout.setContentsMargins(0, 0, 0, 0)
        strip_layout.setSpacing(8)
        self.photo_view = PhotoViewWidget(self.photo_strip, placeholder='尚未上传 2D 照片')
        self.photo_view.point_clicked.connect(self._on_photo_match_point_clicked)
        self.scan_view = PhotoViewWidget(self.photo_strip, placeholder='备用 1')
        self.scan_view.set_interactive(False)
        self.spare_view = PhotoViewWidget(self.photo_strip, placeholder='备用 2')
        self.spare_view.set_interactive(False)
        for widget in (self.photo_view, self.scan_view, self.spare_view):
            strip_layout.addWidget(widget, 1)
        self.photo_strip.setVisible(False)
        stage_layout.addWidget(self.photo_strip, 0)
        stage_layout.addWidget(self.viewport.get_widget(), 1)
        self.viewport_stage.installEventFilter(self)
        viewport_layout.addWidget(self.viewport_stage, 1)

        self.operation_splitter.addWidget(self.left_dock)
        self.operation_splitter.addWidget(viewport_panel)
        self.operation_splitter.addWidget(self.right_dock)
        self.operation_splitter.setStretchFactor(0, 0)
        self.operation_splitter.setStretchFactor(1, 1)
        self.operation_splitter.setStretchFactor(2, 0)
        self.operation_splitter.setSizes([230, 1000, 210])
        self.operation_splitter.installEventFilter(self)
        layout.addWidget(self.operation_splitter, 1)

        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
        self._collapse_sidebar(
            'left',
            self.left_dock,
            self.left_sidebar_expand_button,
        )
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
        lay.addWidget(self.lbl_facade_summary)

        from PySide6.QtWidgets import QListWidget
        self.list_facades = QListWidget()
        self.list_facades.setObjectName('lstFacades')
        self.list_facades.setIconSize(QPixmap(18, 18).size())
        self.list_facades.setSpacing(2)
        lay.addWidget(self.list_facades, 1)
        self.list_facades.itemClicked.connect(self._on_facade_item_clicked)

    def _init_left_panel_widgets(self):
        panel = self.left_dock.findChild(QWidget, 'leftDockPanel')
        if panel is None:
            return
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.photo_match_panel = QWidget()
        self.photo_match_panel.setObjectName('photoMatchPanel')
        self.photo_match_panel.setVisible(False)
        lay = QVBoxLayout(self.photo_match_panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel('2D-3D 匹配')
        title.setStyleSheet('font-weight:600; color:#303641;')
        lay.addWidget(title)

        hint = QLabel(
            '手动：上传照片后进入标注，照片与点云交替点击，至少 6 对后估算相机内外参。\n'
            '自动：先完成立面检测，在右侧列表点选目标立面，再点击自动匹配。\n'
            '映射：匹配完成后可映射回 3D。将按扫描 JSON 位姿展示点云图，并摆正照片后贴到点云图上。'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet('color:#5b626d; font-size:12px;')
        lay.addWidget(hint)

        self.btn_upload_photo = QPushButton('上传2D照片')
        self.btn_auto_photo_match = QPushButton('自动匹配')
        self.btn_annotate_matches = QPushButton('进入标注模式')
        self.btn_undo_photo_match = QPushButton('撤销点对')
        self.btn_exit_photo_annotate = QPushButton('退出标注')
        self.btn_photo_cloud_match = QPushButton('估算相机内外参数')
        self.btn_map_back_to_cloud = QPushButton('映射回3D点云')
        for button in (
            self.btn_upload_photo,
            self.btn_auto_photo_match,
            self.btn_annotate_matches,
            self.btn_undo_photo_match,
            self.btn_exit_photo_annotate,
            self.btn_photo_cloud_match,
            self.btn_map_back_to_cloud,
        ):
            button.setMinimumHeight(34)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            lay.addWidget(button)

        self.lbl_photo_match_status = QLabel('尚未上传照片')
        self.lbl_photo_match_status.setWordWrap(True)
        self.lbl_photo_match_status.setStyleSheet('color:#5b626d; font-size:12px;')
        lay.addWidget(self.lbl_photo_match_status)
        lay.addStretch(1)
        outer.addWidget(self.photo_match_panel)

        self.facade_detection_panel = QWidget()
        self.facade_detection_panel.setObjectName('facadeDetectionPanel')
        self.facade_detection_panel.setVisible(False)
        facade_lay = QVBoxLayout(self.facade_detection_panel)
        facade_lay.setContentsMargins(8, 8, 8, 8)
        facade_lay.setSpacing(8)
        facade_title = QLabel('立面检测')
        facade_title.setStyleSheet('font-weight:600; color:#303641;')
        facade_lay.addWidget(facade_title)
        facade_hint = QLabel(
            '运行顶部「立面检测」会把结果暂存到项目 results 目录。'
            '若已有缓存文件，可在此直接加载，无需重新检测。'
        )
        facade_hint.setWordWrap(True)
        facade_hint.setStyleSheet('color:#5b626d; font-size:12px;')
        facade_lay.addWidget(facade_hint)
        self.btn_load_facade_results = QPushButton('加载立面检测结果')
        self.btn_load_facade_results.setMinimumHeight(34)
        self.btn_load_facade_results.setCursor(Qt.CursorShape.PointingHandCursor)
        facade_lay.addWidget(self.btn_load_facade_results)
        self.lbl_facade_cache_status = QLabel('尚未加载缓存结果')
        self.lbl_facade_cache_status.setWordWrap(True)
        self.lbl_facade_cache_status.setStyleSheet('color:#5b626d; font-size:12px;')
        facade_lay.addWidget(self.lbl_facade_cache_status)
        facade_lay.addStretch(1)
        outer.addWidget(self.facade_detection_panel)

        self.btn_upload_photo.clicked.connect(self._open_upload_photo_dialog)
        self.btn_auto_photo_match.clicked.connect(self._run_auto_photo_match)
        self.btn_annotate_matches.clicked.connect(self._enter_photo_annotate_mode)
        self.btn_undo_photo_match.clicked.connect(self._undo_photo_match_pair)
        self.btn_exit_photo_annotate.clicked.connect(self._exit_photo_annotate_mode)
        self.btn_photo_cloud_match.clicked.connect(self._estimate_photo_camera_pose)
        self.btn_map_back_to_cloud.clicked.connect(self._map_photo_back_to_cloud)
        self.btn_load_facade_results.clicked.connect(self._load_cached_facade_results)
        self._refresh_photo_match_ui()

    def _show_facade_results(self, results: list[dict]):
        self._quality_result_cache.clear()
        count = len(results or [])
        self.lbl_facade_summary.setText(f'检测立面数量：{count}')
        if not results:
            self.list_facades.clear()
            self._photo_match_facade = None
            return

        from PySide6.QtWidgets import QListWidgetItem
        self.list_facades.clear()
        for order, f in enumerate(results):
            pid = int(f.get('id', order))
            pts = int(f.get('point_count') or 0)
            rgb = Config.facade_instance_color(pid, order)
            item = QListWidgetItem(f"立面 {pid}    点数 {pts:,}")
            item.setIcon(self._facade_color_icon(rgb))
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.list_facades.addItem(item)

        # cache for dialog use
        self._latest_facade_results = results
        if self._photo_match_facade is not None:
            selected_id = int(self._photo_match_facade.get('id', -1))
            replacement = next(
                (item for item in results if int(item.get('id', -1)) == selected_id),
                None,
            )
            self._photo_match_facade = replacement
        if hasattr(self, 'photo_match_panel'):
            self._refresh_photo_match_ui()
        if hasattr(self, 'lbl_facade_cache_status'):
            self._refresh_facade_cache_status()

    def _facade_color_icon(self, rgb, size=16):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(
            int(round(max(0.0, min(1.0, rgb[0])) * 255)),
            int(round(max(0.0, min(1.0, rgb[1])) * 255)),
            int(round(max(0.0, min(1.0, rgb[2])) * 255)),
        )
        painter.setBrush(color)
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.drawRoundedRect(1, 1, size - 3, size - 3, 3, 3)
        painter.end()
        return QIcon(pixmap)

    def _active_cloud_name(self) -> str | None:
        try:
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
        return (
            str(cloud or ''), facade_id, standard_id,
            float(grid_size), flatness_limit_mm, verticality_limit_mm,
            window_size_m, step_size_m, measure_height_m,
            len(domain), hash(tuple(domain[:32])),
        )

    def _show_quality_dialog(self, cloud, facade, quality):
        def _show_effect(mode='flatness'):
            try:
                display_quality = dict(quality)
                display_quality['heatmap_mode'] = mode
                self.render_service.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service._index_service)
            except Exception:
                pass

        def _restore():
            try:
                results = getattr(self.project_operation_service,
                                  '_last_facade_results', None)
                self.render_service.restore_highlight(cloud, results or [])
            except Exception:
                pass

        label = f"#{int(facade.get('id', 0))} ({str(facade.get('type_label') or facade.get('type') or '-')})"
        dlg = FacadeQualityDialog(self, label, quality, 
                                  on_show_colors=_show_effect, 
                                  on_restore_colors=_restore)
        dlg.exec()

    def _on_facade_item_clicked(self, item):
        f = item.data(Qt.ItemDataRole.UserRole)
        if not f:
            return
        cloud = self._active_cloud_name()
        if not cloud:
            return
        photo_match_open = (
            hasattr(self, 'photo_match_panel') and self.photo_match_panel.isVisible()
        )
        if photo_match_open:
            self._photo_match_facade = f
            try:
                self.render_service.select_facade(cloud, int(f.get('id', -1)))
            except Exception:
                pass
            self.statusBar().showMessage(
                f"已选择立面 #{int(f.get('id', 0))} 用于自动匹配", 4000
            )
            self._refresh_photo_match_ui()
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
        # Snapshot all UI-owned values before submitting. The worker must not
        # access widgets or the Open3D viewport while running.
        profile = getattr(self, '_inspection_profile', None)
        grid_size = float(self.interval_combo.currentData())
        cache_key = self._quality_cache_key(cloud, f, profile, grid_size)
        cached_quality = self._quality_result_cache.get(cache_key)
        if cached_quality:
            self.statusBar().showMessage('已命中质量结果缓存', 3000)
            self._show_quality_dialog(cloud, f, cached_quality)
            return
        kwargs = {'profile': profile,
                  'grid_size': grid_size,
                  'results_dir': results_dir}
        self._quality_request_token = getattr(self, '_quality_request_token', 0) + 1
        token = self._quality_request_token
        self._quality_request_cache_key = cache_key
        self.statusBar().showMessage('正在计算质量指标...')
        worker = QualityWorker(self.facade_service, cloud, dict(f), kwargs)
        worker.signals.finished.connect(
            lambda facade, quality: self._on_quality_finished(token, cloud, facade, quality))
        worker.signals.failed.connect(
            lambda facade, error: self._on_quality_failed(token, error))
        self._quality_pool.start(worker)
        return

    def _on_quality_failed(self, token, error):
        if token != getattr(self, '_quality_request_token', -1):
            return
        self.statusBar().showMessage('质量计算失败')
        QMessageBox.warning(self, '质量评估', f'质量计算失败：{error}')

    def _on_quality_finished(self, token, cloud, f, quality):
        if token != getattr(self, '_quality_request_token', -1):
            return
        self.statusBar().clearMessage()
        if not quality:
            QMessageBox.information(self, '质量评估', '该立面无法计算质量指标。')
            return
        cache_key = getattr(self, '_quality_request_cache_key', None)
        if cache_key is not None:
            self._quality_result_cache[cache_key] = quality
        self._quality_reports = [r for r in self._quality_reports
                                 if (r.get('facade') or {}).get('id') != f.get('id')]
        self._quality_reports.append({'facade': f, 'quality': quality,
                                      'artifacts': quality.get('artifacts', {})})
        self._show_quality_dialog(cloud, f, quality)

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
        sidebar.setMinimumWidth(180)
        sidebar.setMaximumWidth(280)
        sidebar.setProperty('expandedWidth', 230)
        return sidebar, toggle_button

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
        else:
            self._set_photo_strip_visible(False)
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
            'btn_registration': self.project_operation_service.registration,
            'btn_select_detection_area': (
                self.project_operation_service.select_detection_area
            ),
            'btn_facade_detection': self._run_facade_detection,
            'btn_quality_inspection': (
                self.project_operation_service.quality_inspection
            ),
            'btn_box_segmentation': (
                self.project_operation_service.box_segmentation
            ),
            'btn_calculate_detail': (
                self.project_operation_service.calculate_detail
            ),
            'btn_align_2d_3d': self._enter_photo_match_workspace,
        }
        for action_name, callback in list(pointcloud_actions.items()):
            if action_name != 'btn_align_2d_3d':
                pointcloud_actions[action_name] = self._with_photo_strip_hidden(
                    callback
                )
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

    def _show_facade_detection_workspace(self):
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)
        if hasattr(self, 'photo_match_panel'):
            self.photo_match_panel.setVisible(False)
        if hasattr(self, 'facade_detection_panel'):
            self.facade_detection_panel.setVisible(True)
        self._set_photo_strip_visible(False)
        if self._sidebar_collapsed.get('left'):
            self._expand_sidebar(
                'left',
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        self._refresh_facade_cache_status()

    def _run_facade_detection(self):
        self._show_facade_detection_workspace()
        self.project_operation_service.facade_detection()
        self._refresh_facade_cache_status()

    def _facade_results_directory(self) -> str:
        project_uuid = getattr(self.current_project, 'project_id', None)
        if project_uuid:
            try:
                from services.facade.facade_cache import results_dir
                return str(results_dir(project_uuid))
            except Exception:
                pass
        return self._last_upload_directory

    def _refresh_facade_cache_status(self):
        label = getattr(self, 'lbl_facade_cache_status', None)
        if label is None:
            return
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            label.setText('请先打开项目，再加载或运行立面检测。')
            return
        try:
            from services.facade.facade_cache import list_facade_snapshots
            files = list_facade_snapshots(project_uuid)
        except Exception:
            files = []
        if not files:
            label.setText('当前项目还没有缓存的立面检测结果。')
            return
        latest = files[0]
        label.setText(
            f'已找到 {len(files)} 个缓存文件。\n'
            f'最近：{latest.name}'
        )

    def _load_cached_facade_results(self):
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            QMessageBox.information(self, '加载立面检测结果', '请先打开项目。')
            return
        from services.facade.facade_cache import list_facade_snapshots
        cached = list_facade_snapshots(project_uuid)
        if not cached:
            QMessageBox.information(
                self,
                '加载立面检测结果',
                '当前项目没有缓存的立面检测结果。请先运行一次立面检测。',
            )
            return
        file_path, _selected = QFileDialog.getOpenFileName(
            self,
            '选择立面检测结果文件',
            str(cached[0].parent),
            '立面检测结果 (facade_detection*.json);;JSON 文件 (*.json);;所有文件 (*)',
        )
        if not file_path:
            return
        self._apply_facade_snapshot_file(file_path)

    def _apply_facade_snapshot_file(self, file_path):
        from services.facade.facade_cache import load_facade_snapshot_file
        snapshot = load_facade_snapshot_file(file_path)
        if not snapshot:
            QMessageBox.warning(
                self,
                '加载立面检测结果',
                '无法读取该文件，或文件中没有立面检测结果。',
            )
            return
        if not self._apply_facade_snapshot(snapshot):
            QMessageBox.warning(
                self,
                '加载立面检测结果',
                '已读取文件，但当前视口没有点云，无法显示立面。',
            )
            return
        if hasattr(self, 'lbl_facade_cache_status'):
            self.lbl_facade_cache_status.setText(
                f'已加载：{Path(file_path).name}\n立面数量：{len(snapshot.get("facades") or [])}'
            )

    def _apply_facade_snapshot(self, snapshot) -> bool:
        names = []
        try:
            names = self.viewport.get_cloud_names()
        except Exception:
            names = []
        if not names:
            return False
        stored_name = snapshot.get('cloud_name')
        cloud_name = stored_name if stored_name in names else names[-1]
        facades = snapshot.get('facades') or []
        return bool(
            self.project_operation_service.restore_facade_results(cloud_name, facades)
        )

    def _enter_photo_match_workspace(self):
        self.project_operation_service.align_2d_3d()
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)
        if hasattr(self, 'photo_match_panel'):
            self.photo_match_panel.setVisible(True)
        if hasattr(self, 'facade_detection_panel'):
            self.facade_detection_panel.setVisible(False)
        if self._sidebar_collapsed.get('left'):
            self._expand_sidebar(
                'left',
                self.left_dock,
                self.left_sidebar_expand_button,
            )
        self._set_photo_strip_visible(True)
        self._refresh_photo_match_ui()

    def _resolve_auto_match_facade(self):
        if self._photo_match_facade:
            return self._photo_match_facade
        current = self.list_facades.currentItem() if hasattr(self, 'list_facades') else None
        if current is not None:
            facade = current.data(Qt.ItemDataRole.UserRole)
            if facade:
                self._photo_match_facade = facade
                return facade
        results = getattr(self, '_latest_facade_results', None) or []
        if results:
            self._photo_match_facade = results[0]
            return results[0]
        return None

    def _current_cloud_arrays(self):
        cloud = self._active_cloud_name()
        if not cloud:
            raise ValueError('当前没有已加载的点云')
        data = self.viewport.get_cloud_data(cloud) if hasattr(self.viewport, 'get_cloud_data') else None
        if not data or data.get('pos') is None:
            raise ValueError('无法读取当前点云数据')
        import numpy as np

        points = np.asarray(data['pos'], dtype=float).copy()
        colors = data.get('color')
        colors = None if colors is None else np.asarray(colors, dtype=float).copy()
        return cloud, points, colors

    def _latest_ply_path(self):
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            return None
        try:
            from services.dal.file_repo import FileRepo

            asset = FileRepo.get_latest_raw_pointcloud(project_uuid)
            path = getattr(asset, 'path', None)
            if path and str(path).lower().endswith('.ply'):
                return str(path)
        except Exception:
            return None
        return None

    def _run_auto_photo_match(self):
        state = self.photo_match_service.state
        if not state.photo_path:
            QMessageBox.information(self, '自动匹配', '请先上传 2D 照片。')
            return
        facade = self._resolve_auto_match_facade()
        if facade is None:
            QMessageBox.information(
                self,
                '自动匹配',
                '请先完成立面检测，并在右侧列表中点选要匹配的立面。',
            )
            return
        if self._auto_match_busy:
            return
        try:
            _cloud, points, colors = self._current_cloud_arrays()
        except ValueError as exc:
            QMessageBox.warning(self, '自动匹配', str(exc))
            return
        self._exit_photo_annotate_mode(silent=True)
        self._auto_match_busy = True
        self._auto_match_token += 1
        token = self._auto_match_token
        self.statusBar().showMessage(
            f"正在自动匹配立面 #{int(facade.get('id', 0))}..."
        )
        self._refresh_photo_match_ui()
        worker = AutoMatchWorker({
            'photo_path': state.photo_path,
            'facade': dict(facade),
            'points': points,
            'image_width': float(state.image_width),
            'image_height': float(state.image_height),
            'colors': colors,
            'ply_path': self._latest_ply_path(),
        })
        worker.signals.finished.connect(
            lambda result: self._on_auto_photo_match_finished(token, result)
        )
        worker.signals.failed.connect(
            lambda error: self._on_auto_photo_match_failed(token, error)
        )
        self._auto_match_pool.start(worker)

    def _on_auto_photo_match_finished(self, token, result):
        if token != self._auto_match_token:
            return
        self._auto_match_busy = False
        try:
            pose = self.photo_match_service.apply_auto_match_result(result)
        except ValueError as exc:
            self.statusBar().showMessage('自动匹配失败')
            QMessageBox.warning(self, '自动匹配', str(exc))
            self._refresh_photo_match_ui()
            return
        self._refresh_photo_match_markers()
        self._refresh_photo_match_ui()
        auto_info = pose.get('auto_match') or {}
        backend = auto_info.get('matcher_backend') or ''
        mean_error = pose.get('reprojection_mean_px')
        inliers = pose.get('inlier_count')
        extra = f'，平均重投影误差 {float(mean_error):.2f} px' if mean_error is not None else ''
        self.statusBar().showMessage('自动匹配完成', 5000)
        QMessageBox.information(
            self,
            '自动匹配',
            f"立面 #{int((self._photo_match_facade or {}).get('id', 0))} 匹配完成。\n"
            f"匹配器：{backend or 'auto'}\n"
            f"内点：{inliers}{extra}",
        )

    def _on_auto_photo_match_failed(self, token, error):
        if token != self._auto_match_token:
            return
        self._auto_match_busy = False
        self.statusBar().showMessage('自动匹配失败')
        viz = getattr(error, 'match_visualization', None) or {}
        if viz:
            kpts = viz.get('kpts_photo') or []
            if kpts:
                self.photo_view.set_markers(
                    [(float(pt[0]), float(pt[1])) for pt in kpts]
                )
        message = str(error)
        if isinstance(error, ImportError) or 'torch' in message.lower():
            message = (
                '自动匹配依赖未安装或无法导入。\n'
                '请先按 facadeDetection/requirements.txt 安装 torch、LightGlue / GlueStick。\n\n'
                f'{error}'
            )
        QMessageBox.warning(self, '自动匹配', message)
        self._refresh_photo_match_ui()

    def _open_upload_photo_dialog(self):
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            '选择 2D 照片',
            self._last_upload_directory,
            PHOTO_FILE_FILTER,
        )
        if not file_path:
            return
        self._last_upload_directory = str(Path(file_path).parent)
        try:
            image = self.photo_match_service.load_photo(file_path)
        except ValueError as exc:
            QMessageBox.warning(self, '上传 2D 照片', str(exc))
            return
        self.photo_view.set_image(image)
        self._sync_photo_strip()
        QTimer.singleShot(0, self._sync_photo_strip)
        self._exit_photo_annotate_mode(silent=True)
        self._refresh_photo_match_ui()

    def _enter_photo_annotate_mode(self):
        state = self.photo_match_service.state
        if not state.photo_path:
            QMessageBox.information(self, '标注模式', '请先上传 2D 照片。')
            return
        state.annotating = True
        state.next_is_photo = True
        self.photo_view.set_interactive(True)
        self.viewport.enter_pick_mode(
            callback=self._on_cloud_match_point_picked,
            pick_radius=24,
        )
        self.statusBar().showMessage('标注模式：先在照片上点选，再在点云上点选对应位置', 6000)
        self._refresh_photo_match_ui()

    def _exit_photo_annotate_mode(self, silent=False):
        self.photo_match_service.state.annotating = False
        self.photo_match_service.state.next_is_photo = True
        self.photo_view.set_interactive(False)
        try:
            self.viewport.exit_pick_mode()
        except Exception:
            pass
        if not silent:
            self._refresh_photo_match_ui()

    def _undo_photo_match_pair(self):
        self.photo_match_service.undo_last()
        self._refresh_photo_match_markers()
        self._refresh_photo_match_ui()

    def _on_photo_match_point_clicked(self, pixel_x, pixel_y):
        try:
            self.photo_match_service.add_photo_point(pixel_x, pixel_y)
        except RuntimeError as exc:
            QMessageBox.information(self, '标注照片', str(exc))
            return
        self._refresh_photo_match_markers()
        self._refresh_photo_match_ui()
        self.statusBar().showMessage('已在照片标注，请在 3D 点云上点击对应点', 5000)

    def _on_cloud_match_point_picked(self, picked):
        point = picked.get('point') if isinstance(picked, dict) else picked
        try:
            self.photo_match_service.add_cloud_point(point)
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._refresh_photo_match_markers()
        self._refresh_photo_match_ui()
        count = self.photo_match_service.complete_pair_count()
        self.statusBar().showMessage(f'已在点云标注，当前完整点对 {count}', 4000)

    def _estimate_photo_camera_pose(self):
        try:
            result = self.photo_match_service.solve_pose()
        except ValueError as exc:
            QMessageBox.warning(self, '估算相机内外参数', str(exc))
            return
        mean_error = result.get('reprojection_mean_px')
        inliers = result.get('inlier_count')
        method = result.get('pnp_method', '')
        message = (
            f'估计完成。\n'
            f'PnP：{method}\n'
            f'内点：{inliers}\n'
        )
        if mean_error is not None:
            message += f'平均重投影误差：{float(mean_error):.2f} px'
        QMessageBox.information(self, '估算相机内外参数', message)
        self._refresh_photo_match_ui()

    def _map_photo_back_to_cloud(self):
        state = self.photo_match_service.state
        if not state.photo_path:
            QMessageBox.information(self, '映射回3D点云', '请先上传 2D 照片。')
            return
        if state.pose is None or self.photo_match_service.complete_pair_count() < 4:
            QMessageBox.information(
                self, '映射回3D点云', '请先完成 2D-3D 匹配（手动或自动）。',
            )
            return
        facade = self._resolve_auto_match_facade()
        if facade is None:
            QMessageBox.information(
                self, '映射回3D点云', '请先在右侧列表点选要映射的立面。',
            )
            return
        if self._map_back_busy:
            return
        try:
            _cloud, points, colors = self._current_cloud_arrays()
        except ValueError as exc:
            QMessageBox.warning(self, '映射回3D点云', str(exc))
            return
        ply_path = self._latest_ply_path()
        search_dir = str(Path(ply_path).parent) if ply_path else self._last_upload_directory
        self._map_back_busy = True
        self._map_back_token += 1
        token = self._map_back_token
        self.statusBar().showMessage('正在按扫描位姿映射照片到点云视图...')
        self._refresh_photo_match_ui()
        worker = MapBackWorker({
            'photo_path': state.photo_path,
            'points': points,
            'colors': colors,
            'correspondences': list(state.correspondences),
            'facade': dict(facade),
            'ply_path': ply_path,
            'search_dir': search_dir,
        })
        worker.signals.finished.connect(
            lambda result: self._on_map_back_finished(token, result)
        )
        worker.signals.failed.connect(
            lambda error: self._on_map_back_failed(token, error)
        )
        self._map_back_pool.start(worker)

    def _on_map_back_finished(self, token, result):
        if token != self._map_back_token:
            return
        self._map_back_busy = False
        try:
            rectified = bgr_to_qimage(result['rectified_bgr'])
            overlay = bgr_to_qimage(result['overlay_bgr'])
        except Exception as exc:
            QMessageBox.warning(self, '映射回3D点云', f'结果图像转换失败：{exc}')
            self._refresh_photo_match_ui()
            return
        self.photo_view.set_image(rectified)
        self.photo_view.set_markers([])
        self.scan_view.set_image(overlay)
        QTimer.singleShot(0, self._sync_photo_strip)
        json_name = Path(result.get('json_path') or '').name
        self.statusBar().showMessage('映射完成', 5000)
        QMessageBox.information(
            self,
            '映射回3D点云',
            f'已按 {json_name or "扫描 JSON"} 位姿渲染点云视图。\n'
            f'照片摆正方法：{result.get("rectify_method")}\n'
            f'映射内点：{result.get("inlier_count")}',
        )
        self._refresh_photo_match_ui()

    def _on_map_back_failed(self, token, error):
        if token != self._map_back_token:
            return
        self._map_back_busy = False
        self.statusBar().showMessage('映射失败')
        QMessageBox.warning(self, '映射回3D点云', str(error))
        self._refresh_photo_match_ui()

    def _refresh_photo_match_markers(self):
        self.photo_view.set_markers(self.photo_match_service.photo_points())
        cloud_points = self.photo_match_service.cloud_points()
        try:
            self.viewport.update_pick_markers(src_points=cloud_points)
        except Exception:
            pass

    def _refresh_photo_match_ui(self):
        state = self.photo_match_service.state
        complete = self.photo_match_service.complete_pair_count()
        has_photo = bool(state.photo_path)
        if not hasattr(self, 'lbl_photo_match_status'):
            return
        facade = self._resolve_auto_match_facade() if has_photo else None
        can_auto = (
            has_photo
            and facade is not None
            and not self._auto_match_busy
            and not self._map_back_busy
        )
        if hasattr(self, 'btn_auto_photo_match'):
            self.btn_auto_photo_match.setEnabled(can_auto)
            self.btn_auto_photo_match.setText(
                '自动匹配中...' if self._auto_match_busy else '自动匹配'
            )
        self.btn_annotate_matches.setEnabled(
            has_photo and not state.annotating and not self._auto_match_busy and not self._map_back_busy
        )
        self.btn_undo_photo_match.setEnabled(has_photo and bool(state.correspondences))
        self.btn_exit_photo_annotate.setEnabled(state.annotating)
        self.btn_photo_cloud_match.setEnabled(
            complete >= 6 and not self._auto_match_busy and not self._map_back_busy
        )
        can_map = (
            has_photo
            and state.pose is not None
            and complete >= 4
            and facade is not None
            and not self._auto_match_busy
            and not self._map_back_busy
            and not state.annotating
        )
        if hasattr(self, 'btn_map_back_to_cloud'):
            self.btn_map_back_to_cloud.setEnabled(can_map)
            self.btn_map_back_to_cloud.setText(
                '映射中...' if self._map_back_busy else '映射回3D点云'
            )
        facade_hint = ''
        if facade is not None:
            facade_hint = f"\n目标立面：#{int(facade.get('id', 0))}"
        if self._auto_match_busy:
            self.lbl_photo_match_status.setText('正在自动匹配，请稍候...')
            return
        if self._map_back_busy:
            self.lbl_photo_match_status.setText('正在映射回 3D 点云视图，请稍候...')
            return
        if not has_photo:
            self.lbl_photo_match_status.setText(
                '尚未上传照片。自动匹配还需先检测立面并在右侧点选目标立面。'
            )
            return
        name = Path(state.photo_path).name
        if state.annotating:
            next_hint = '请点击照片' if state.next_is_photo else '请点击 3D 点云'
            self.lbl_photo_match_status.setText(
                f'已上传：{name}{facade_hint}\n匹配点：{complete} 对（至少 6 对）\n{next_hint}'
            )
        elif state.pose:
            error = state.pose.get('reprojection_mean_px')
            extra = f'，误差 {float(error):.2f} px' if error is not None else ''
            mode = '自动' if state.match_mode == 'auto' else '手动'
            self.lbl_photo_match_status.setText(
                f'已上传：{name}{facade_hint}\n匹配点：{complete} 对\n{mode}位姿估计已完成{extra}'
            )
        else:
            extra = ''
            if facade is None:
                extra = '\n自动匹配未启用：请先完成立面检测，并在右侧列表点选目标立面'
            self.lbl_photo_match_status.setText(
                f'已上传：{name}{facade_hint}\n匹配点：{complete} 对（手动至少 6 对，或点击自动匹配）{extra}'
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
        uploaded_paths = self.project_overview_service.upload_files(file_paths, getattr(self.current_project, 'project_id', None))
        self._refresh_project_list()
        if self.current_project is not None:
            self._activate_project(self.current_project, load_cloud=False)
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
            self._activate_project(self.current_project, load_cloud=False)

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
        self._activate_project(project, load_cloud=False)

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
        self._activate_project(project)

    def _activate_project(self, project, load_cloud=True):
        if load_cloud:
            try:
                self.project_overview_service.activate_project(project.project_id)
            except Exception as exc:
                print(f'打开项目点云失败: {exc}', flush=True)
        self._set_current_project(project)
        try:
            # propagate active project UUID to operation scheduler for DAL persistence
            self.project_operation_service.set_active_project_uuid(getattr(project, 'project_id', None))
        except Exception:
            pass
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)
        self._restore_cached_facade_results(project)

    def _restore_cached_facade_results(self, project):
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            return
        try:
            from services.facade.facade_cache import load_facade_snapshot
            snapshot = load_facade_snapshot(project_uuid)
        except Exception:
            return
        if not snapshot:
            return
        self._apply_facade_snapshot(snapshot)
        self._refresh_facade_cache_status()

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

    def _with_photo_strip_hidden(self, callback):
        def wrapper():
            self._set_photo_strip_visible(False)
            return callback()
        return wrapper

    def _set_photo_strip_visible(self, visible: bool):
        strip = getattr(self, 'photo_strip', None)
        stage = getattr(self, 'viewport_stage', None)
        if strip is None:
            return
        layout = stage.layout() if stage is not None else None
        if layout is not None:
            layout.setSpacing(8 if visible else 0)
        strip.setVisible(visible)
        if visible:
            self._sync_photo_strip()
            QTimer.singleShot(0, self._sync_photo_strip)
        elif strip.height() != 0:
            strip.setFixedHeight(0)

    def _sync_photo_strip(self):
        strip = getattr(self, 'photo_strip', None)
        host = getattr(self, 'viewport_stage', None)
        if strip is None or host is None or strip.isHidden() or host.width() <= 0:
            return
        available = max(240, host.width())
        slot_width = max(160, int((available - 16) / 3))
        height = max(110, min(slot_width // 2, 220))
        strip.setFixedHeight(height)

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
        if (
            watched is getattr(self, 'viewport_stage', None)
            and event.type() in (
                QEvent.Type.Resize,
                QEvent.Type.Show,
            )
        ):
            QTimer.singleShot(0, self._sync_photo_strip)
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
