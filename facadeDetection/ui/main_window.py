from pathlib import Path
from datetime import datetime

from PySide6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    QThreadPool,
    Slot,
    QTimer,
    Qt,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from dataclasses import replace
import numpy as np
from qtwebview2 import QtWebView2Widget

from .widgets.flow_layout import FlowLayout
from .widgets.photo_view_widget import PhotoViewWidget
from .widgets.station_panel import StationPanel
from .widgets.pointcloud_controls import PointCloudControls
from .widgets.facade_panel import FacadePanel
from services.inspection_review import InspectionReviewService
from services.photo_match_service import PhotoMatchService, bgr_to_qimage
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.viewport_render_service import ViewportRenderService
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from services.facade.facade_service import FacadeService
from services.report_export import ReportExportService, ReportDataService, PdfReportRenderer
from services.dal.report_repo import ReportRepo
from services.result_export_service import ResultExportService
from config.settings import Config
from config.storage import Storage
from view3d.open3d_viewport import Open3DViewport
from ui.dialogs.facade_quality_dialog import FacadeQualityDialog
from services.inspection_profile import InspectionProfileService
from services.dal.results_repo import ResultsRepo
from utils.workers import QualityWorker, PointCloudLoadWorker, ViewExportWorker,RegistrationWorker


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
    '点云文件 (*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls *.dist);;'
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '所有文件 (*)'
)

REPORT_PDF_FILTER = 'PDF 文件 (*.pdf)'
REPORT_EMPTY_TITLE = '请选择PDF上传'
PHOTO_FILE_FILTER = (
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)'
)
SCAN_POSE_FILE_FILTER = '扫描位姿 JSON (*.json);;所有文件 (*)'
APPLICATION_TITLE = '点云外立面智能检测平台'

# 每个页面只突出一个主要操作，避免顶部十余个按钮全部使用主色。
PRIMARY_HEADER_ACTIONS = {
    'btn_upload_files',
    'btn_facade_detection',
    'btn_open_report_pdf',
}

# Corporate Clean 强调清晰的功能分组。按钮名称和业务连接保持不变，
# 这里只定义它们在顶部命令栏中的视觉归属。
PAGE_HEADER_GROUPS = {
    'project_overview': (
        (
            '文件导入',
            (
                'btn_import_fls_dir',
                'btn_upload_files',
            ),
        ),
        (
            '项目管理',
            (
                'btn_open_project',
                'btn_select_project',
                'btn_new_project',
            ),
        ),
    ),
    'project_operation': (
        ('点云处理', ('btn_denoise', 'btn_registration')),
        (
            '区域处理',
            (
                'btn_select_detection_area',
                'btn_box_segmentation',
            ),
        ),
        (
            '立面分析',
            (
                'btn_facade_detection',
                'btn_quality_inspection',
            ),
        ),
        (
            '结果关联',
            (
                'btn_calculate_detail',
                'btn_align_2d_3d',
            ),
        ),
    ),
    'inspection_review': (),
    'report_export': (
        # 质量报告导出是 ruiqi_dev 新增能力，必须随当前报告页命令栏一起创建，
        # 后续按钮连接才能安全取得对应控件。
        ('', ('btn_open_report_pdf', 'btn_export_quality_report')),
    ),
}


class ElidedLabel(QLabel):
    """在空间不足时省略中间文本，同时保留完整内容供 Tooltip 查看。"""

    def __init__(self, text='', parent=None, maximum_hint_width=480):
        super().__init__('', parent)
        self._full_text = str(text)
        self._maximum_hint_width = maximum_hint_width
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._update_elided_text()

    def setText(self, text):
        self._full_text = str(text)
        self.updateGeometry()
        self._update_elided_text()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def sizeHint(self):
        hint = super().sizeHint()
        # 使用完整文本计算理想宽度；布局空间不足时再由 resizeEvent 省略。
        full_text_width = self.fontMetrics().horizontalAdvance(self._full_text) + 32
        if self._maximum_hint_width is not None:
            full_text_width = min(full_text_width, self._maximum_hint_width)
        hint.setWidth(max(hint.width(), full_text_width))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def _update_elided_text(self):
        available_width = max(0, self.contentsRect().width())
        visible_text = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        QLabel.setText(self, visible_text)


class ApplicationTitleBar(QWidget):
    """无边框窗口的可拖动标题栏，保留系统标题栏的常用交互。"""

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            window_handle = self.window().windowHandle()
            if window_handle is not None and window_handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            toggle_window_state = getattr(
                self.window(),
                '_toggle_maximize_restore',
                None,
            )
            if toggle_window_state is not None:
                toggle_window_state()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class WindowResizeHandle(QWidget):
    """透明窗口缩放热区，替代 Windows 会露出白边的原生粗边框。"""

    def __init__(self, edges, cursor, parent=None):
        super().__init__(parent)
        self._edges = edges
        self._last_start_result = None
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setCursor(cursor)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.window().isMaximized()
        ):
            window_handle = self.window().windowHandle()
            if window_handle is not None:
                # 交给操作系统执行缩放，拖动过程比手算 geometry 更顺滑。
                self._last_start_result = window_handle.startSystemResize(
                    self._edges
                )
                if self._last_start_result:
                    event.accept()
                    return
        super().mousePressEvent(event)


class TechnicalCanvas(QWidget):
    """绘制与点云/工程场景相关的轻量空状态，不依赖额外图片资源。"""

    def __init__(self, variant='facade', message='', parent=None):
        super().__init__(parent)
        self.variant = variant
        self.message = message
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(220)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FBFCFE'))

        canvas = self.rect().adjusted(24, 24, -24, -24)
        self._draw_grid(painter, canvas)
        if self.variant == 'document':
            self._draw_document(painter, canvas)
        else:
            self._draw_facade(painter, canvas)
        painter.end()

    def _draw_grid(self, painter, rect):
        painter.setPen(QPen(QColor('#EDF2F7'), 1))
        step = 32
        for x in range(rect.left(), rect.right() + 1, step):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(rect.top(), rect.bottom() + 1, step):
            painter.drawLine(rect.left(), y, rect.right(), y)

    def _draw_facade(self, painter, rect):
        width = min(440, max(220, int(rect.width() * 0.42)))
        height = min(248, max(150, int(rect.height() * 0.48)))
        center = rect.center()
        x = center.x() - width / 2
        y = center.y() - height / 2 - (18 if self.message else 0)
        facade = QRectF(x, y, width, height)

        painter.setPen(QPen(QColor('#B9C7DA'), 1.4))
        painter.drawRect(facade)
        depth = min(28, width * 0.08)
        painter.drawLine(
            facade.topRight(),
            facade.topRight() + QPointF(depth, -16),
        )
        painter.drawLine(
            facade.bottomRight(),
            facade.bottomRight() + QPointF(depth, -16),
        )
        painter.drawLine(
            facade.topRight() + QPointF(depth, -16),
            facade.bottomRight() + QPointF(depth, -16),
        )

        columns, rows = 7, 5
        painter.setPen(QPen(QColor('#D6DFEB'), 1))
        for column in range(1, columns):
            px = facade.left() + facade.width() * column / columns
            painter.drawLine(int(px), int(facade.top()), int(px), int(facade.bottom()))
        for row in range(1, rows):
            py = facade.top() + facade.height() * row / rows
            painter.drawLine(int(facade.left()), int(py), int(facade.right()), int(py))

        # 在立面网格交点绘制确定性的点云采样，避免伪造业务检测结果。
        point_color = QColor('#3B82F6')
        point_color.setAlpha(145 if self.variant == 'heatmap' else 105)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(point_color)
        for row in range(rows + 1):
            for column in range(columns + 1):
                px = facade.left() + facade.width() * column / columns
                py = facade.top() + facade.height() * row / rows
                jitter_x = ((row * 7 + column * 3) % 5) - 2
                jitter_y = ((row * 5 + column * 11) % 5) - 2
                radius = 2 if (row + column) % 3 else 2.6
                painter.drawEllipse(
                    QRectF(
                        px + jitter_x - radius,
                        py + jitter_y - radius,
                        radius * 2,
                        radius * 2,
                    )
                )

        if self.message:
            text_font = QFont(self.font())
            text_font.setPixelSize(16)
            text_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(text_font)
            painter.setPen(QColor('#334155'))
            text_rect = QRectF(
                facade.left(),
                facade.bottom() + 22,
                facade.width(),
                28,
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                self.message,
            )

    def _draw_document(self, painter, rect):
        width = min(240, max(150, int(rect.width() * 0.22)))
        height = min(300, max(190, int(rect.height() * 0.54)))
        x = rect.center().x() - width / 2
        y = rect.center().y() - height / 2
        page = QRectF(x, y, width, height)

        painter.setBrush(QColor('#FFFFFF'))
        painter.setPen(QPen(QColor('#B9C7DA'), 1.4))
        painter.drawRoundedRect(page, 8, 8)
        painter.setPen(QPen(QColor('#D6DFEB'), 2))
        line_left = int(page.left() + 28)
        line_right = int(page.right() - 28)
        for offset, ratio in ((48, 1.0), (78, 0.72), (108, 0.88), (154, 1.0)):
            painter.drawLine(
                line_left,
                int(page.top() + offset),
                int(line_left + (line_right - line_left) * ratio),
                int(page.top() + offset),
            )
        painter.setPen(QPen(QColor('#3B82F6'), 3))
        painter.drawLine(
            line_left,
            int(page.top() + 24),
            int(page.left() + width * 0.48),
            int(page.top() + 24),
        )


class _MatchLinkOverlay(QWidget):
    """在照片、二维投影和三维视口之间绘制同号匹配连线。"""

    _COLORS = (
        QColor(255, 90, 90, 185),
        QColor(255, 180, 50, 185),
        QColor(75, 205, 255, 185),
        QColor(100, 225, 120, 185),
        QColor(220, 100, 255, 185),
    )

    def __init__(self, parent):
        super().__init__(parent)
        self._links = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.hide()

    def set_links(self, links):
        self._links = list(links or [])
        self.setGeometry(self.parentWidget().rect())
        self.setVisible(bool(self._links))
        if self._links:
            self.raise_()
        self.update()

    def paintEvent(self, event):
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, (photo, projection, cloud, _label) in enumerate(self._links):
            color = self._COLORS[index % len(self._COLORS)]
            painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
            painter.drawLine(photo, projection)
            painter.drawLine(projection, cloud)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('mainWindow')
        # 完全隐藏 Windows 非客户区，避免缩放边框在深色标题栏上方露出白条。
        # 窗口缩放由八个透明 WindowResizeHandle 继续交给系统完成。
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        self.setWindowTitle(APPLICATION_TITLE)
        self.resize(1600, 900)
        # 三维工作台在过窄尺寸下失去可用性；该下限同时保证命令栏和四页签不溢出。
        self.setMinimumSize(960, 640)
        self.viewport = Open3DViewport()
        # Unified render service for business modules
        self.render_service = ViewportRenderService(self.viewport, db=None)
        self.project_overview_service = ProjectOverviewService(self.viewport, self.render_service, db=None)
        # 使用 ruiqi_dev 最新服务编排，UI 只负责展示与交互。
        self.pointcloud_service = PointCloudService(self.viewport, self.render_service)
        self.facade_service = FacadeService(
            self.viewport,
            db=None,
            render_service=self.render_service,
            pointcloud_service=self.pointcloud_service,
        )
        self.facade_service.set_pointcloud_service(self.pointcloud_service)
        self.render_service.pointcloud_service = self.pointcloud_service
        self.project_operation_service = ProjectOperationService(
            self.viewport,
            facade_service=self.facade_service,
            pointcloud_service=self.pointcloud_service,
            render_service=self.render_service,
        )
        self.station_service = PointCloudStationService(
            self.render_service, pointcloud_service=self.pointcloud_service)
        self.project_operation_service.denoise_completed_callback = (
            self._on_match_cloud_denoised
        )
        self.project_operation_service.set_station_service(self.station_service)
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.photo_match_service = PhotoMatchService()
        self._photo_match_display_image = None
        self._photo_heatmap_image = None
        self._photo_angle_adjusted_image = None
        self._cloud_angle_adjusted_image = None
        self._latest_facade_results = []
        self._saved_match_view = None
        self._saved_match_view_path = None
        self._projection_enabled = False
        self._match_denoise_busy = False
        self._match_preprocessed_cloud = None
        self._view_export_worker = None
        self._live_projection_view = None
        self._reprojection_display_indices = None
        self._projection_marker_points = []
        self._projected_3d_labels = []
        self._heatmap_scale = None
        self._heatmap_projection_data = None
        self._show_match_markers = True
        self._projection_sliders = {}
        self._projection_value_labels = {}
        self._projection_controls_updating = False
        self._projection_refresh_timer = QTimer(self)
        self._projection_refresh_timer.setSingleShot(True)
        self._projection_refresh_timer.setInterval(100)
        self._projection_refresh_timer.timeout.connect(
            self._refresh_projection_preview
        )
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        self._sidebar_collapsed = {'left': False, 'right': False}
        # 默认定位到 data/projects，便于跨机迁移
        try:
            Storage.ensure_base_dirs()
            self._last_upload_directory = str(Storage.PROJECTS_ROOT)
        except Exception:
            self._last_upload_directory = str(Path.home())
        self._header_resize_pending = set()
        self._current_report_pdf_name = None
        self._report_navigation_index = 0
        self._report_webview_error = None
        self._quality_reports = []
        self._heatmap_mode = 'flatness'
        self._quality_result_cache = {}
        self._report_snapshot = {'project': {}, 'facades': []}
        # 质量结果窗口采用非阻塞打开方式；必须由主窗口持有引用，避免窗口被
        # Python 垃圾回收，同时避免再次进入 QDialog.exec() 的嵌套事件循环。
        self._quality_dialog = None
        # Quality jobs may hold a large raw-point working set.  The global Qt
        # pool can otherwise start several facade jobs simultaneously and
        # exhaust Windows commit memory while CPUs remain underutilised.
        self._quality_pool = QThreadPool(self)
        self._quality_pool.setMaxThreadCount(1)
        self._registration_pool = QThreadPool(self)
        self._registration_pool.setMaxThreadCount(1)
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(1)
        self._active_load_worker = None
        self._load_cancel_button = None
        self._load_in_progress = False
        self._active_quality_worker = None
        self._registration_worker = None
        self._pending_station_selection = {}
        self._station_selection_timer = QTimer(self)
        self._station_selection_timer.setSingleShot(True)
        self._station_selection_timer.setInterval(160)
        self._station_selection_timer.timeout.connect(self._flush_station_selection)
        self._project_generation = 0
        self._setup_ui()
        self._match_link_overlay = _MatchLinkOverlay(self)
        self.viewport.marker_projection_callback = (
            self._on_3d_marker_projection_changed
        )
        self._create_resize_handles()
        self._connect_buttons()
        # Hook: display facade stats in the right dock when results ready
        try:
            self.project_operation_service.on_facade_results = self._show_facade_results
        except Exception:
            pass
        self._refresh_project_list()
        self._set_current_project(None)

    def _setup_ui(self):
        # 统一应用外壳让品牌栏、页面内容和底部导航形成稳定层级。
        shell = QWidget()
        shell.setObjectName('applicationShell')
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._create_application_header())

        self.page_stack = self._create_page_stack()
        shell_layout.addWidget(self.page_stack, 1)
        self.setCentralWidget(shell)

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

    def _create_application_header(self):
        """创建可拖动的自定义标题栏，并承载窗口控制按钮。"""
        header = ApplicationTitleBar()
        header.setObjectName('applicationHeader')
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(68)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 12, 0, 12)
        layout.setSpacing(10)

        brand_mark = QLabel('P3D')
        brand_mark.setObjectName('applicationBrandMark')
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(44, 44)
        brand_mark.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        layout.addWidget(brand_mark)

        # 顶栏只显示当前页面名称，避免和窗口标题重复展示平台名称。
        self.application_page_title = QLabel(PAGE_DEFINITIONS[0][0])
        self.application_page_title.setObjectName('applicationPageTitle')
        self.application_page_title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        layout.addWidget(self.application_page_title)

        layout.addStretch(1)

        self.current_project_label = ElidedLabel(
            '当前项目：未选择',
            maximum_hint_width=None,
        )
        self.current_project_label.setObjectName('currentProjectLabel')
        self.current_project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_project_label.setProperty('uiRole', 'supportingText')
        self.current_project_label.setMinimumWidth(200)
        self.current_project_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self.current_project_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        layout.addWidget(self.current_project_label)

        window_controls = QWidget()
        window_controls.setObjectName('windowControls')
        controls_layout = QHBoxLayout(window_controls)
        controls_layout.setContentsMargins(4, 0, 0, 0)
        controls_layout.setSpacing(0)

        self.window_minimize_button = self._create_window_control_button(
            'btn_window_minimize',
            '—',
            '最小化',
        )
        self.window_maximize_button = self._create_window_control_button(
            'btn_window_maximize_restore',
            '□',
            '最大化',
        )
        self.window_close_button = self._create_window_control_button(
            'btn_window_close',
            '×',
            '关闭',
        )
        self.window_close_button.setProperty('windowAction', 'close')
        controls_layout.addWidget(self.window_minimize_button)
        controls_layout.addWidget(self.window_maximize_button)
        controls_layout.addWidget(self.window_close_button)
        layout.addWidget(window_controls)

        self.window_minimize_button.clicked.connect(self.showMinimized)
        self.window_maximize_button.clicked.connect(
            self._toggle_maximize_restore
        )
        self.window_close_button.clicked.connect(self.close)
        return header

    def _create_window_control_button(self, object_name, text, tooltip):
        """创建与深色标题栏一致的标准窗口控制按钮。"""
        button = QToolButton()
        button.setObjectName(object_name)
        button.setProperty('uiRole', 'windowControl')
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(48, 44)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _create_resize_handles(self):
        """在窗口四边和四角创建不可见的系统缩放热区。"""
        handle_definitions = (
            ('top', Qt.Edge.TopEdge, Qt.CursorShape.SizeVerCursor),
            ('bottom', Qt.Edge.BottomEdge, Qt.CursorShape.SizeVerCursor),
            ('left', Qt.Edge.LeftEdge, Qt.CursorShape.SizeHorCursor),
            ('right', Qt.Edge.RightEdge, Qt.CursorShape.SizeHorCursor),
            (
                'top_left',
                Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
                Qt.CursorShape.SizeFDiagCursor,
            ),
            (
                'top_right',
                Qt.Edge.TopEdge | Qt.Edge.RightEdge,
                Qt.CursorShape.SizeBDiagCursor,
            ),
            (
                'bottom_left',
                Qt.Edge.BottomEdge | Qt.Edge.LeftEdge,
                Qt.CursorShape.SizeBDiagCursor,
            ),
            (
                'bottom_right',
                Qt.Edge.BottomEdge | Qt.Edge.RightEdge,
                Qt.CursorShape.SizeFDiagCursor,
            ),
        )
        self._resize_handles = {}
        for name, edges, cursor in handle_definitions:
            handle = WindowResizeHandle(edges, cursor, self)
            handle.setObjectName(f'windowResizeHandle_{name}')
            self._resize_handles[name] = handle
        self._position_resize_handles()

    def _position_resize_handles(self):
        if not hasattr(self, '_resize_handles'):
            return

        width = self.width()
        height = self.height()
        # Frameless windows have no native non-client resize frame.  Keep a
        # generous hit target so the embedded Open3D child cannot make the
        # system resize gesture effectively impossible to start.
        edge_size = 9
        corner_size = 14
        geometries = {
            'top': (
                corner_size,
                0,
                max(0, width - corner_size * 2),
                edge_size,
            ),
            'bottom': (
                corner_size,
                max(0, height - edge_size),
                max(0, width - corner_size * 2),
                edge_size,
            ),
            'left': (
                0,
                corner_size,
                edge_size,
                max(0, height - corner_size * 2),
            ),
            'right': (
                max(0, width - edge_size),
                corner_size,
                edge_size,
                max(0, height - corner_size * 2),
            ),
            'top_left': (0, 0, corner_size, corner_size),
            'top_right': (
                max(0, width - corner_size),
                0,
                corner_size,
                corner_size,
            ),
            'bottom_left': (
                0,
                max(0, height - corner_size),
                corner_size,
                corner_size,
            ),
            'bottom_right': (
                max(0, width - corner_size),
                max(0, height - corner_size),
                corner_size,
                corner_size,
            ),
        }
        visible = not (self.isMaximized() or self.isFullScreen())
        for name, handle in self._resize_handles.items():
            handle.setGeometry(*geometries[name])
            handle.setVisible(visible)
            if visible:
                handle.raise_()

    def _toggle_maximize_restore(self):
        """在最大化和普通窗口间切换，并同步中间按钮的含义。"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        QTimer.singleShot(0, self._update_maximize_button)

    def _update_maximize_button(self):
        if not hasattr(self, 'window_maximize_button'):
            return
        is_maximized = self.isMaximized()
        self.window_maximize_button.setText('❐' if is_maximized else '□')
        action = '还原' if is_maximized else '最大化'
        self.window_maximize_button.setToolTip(action)
        self.window_maximize_button.setAccessibleName(action)

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
            page.setProperty('pageRole', 'workspace')
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            stack.addWidget(page)
            self.page_widgets.append(page)

        return stack

    def _create_page_shell(self, page_title, page_key, page=None):
        """为四个主页面提供无外层间隙的命令栏和工作区。"""
        page = page or QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        content = QWidget()
        content.setProperty('uiRole', 'contentArea')
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        workspace = QFrame()
        workspace.setObjectName(f'{page_key}WorkspaceSurface')
        workspace.setProperty('uiRole', 'workspaceSurface')
        workspace.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        if PAGE_HEADER_GROUPS[page_key]:
            workspace_layout.addWidget(self._create_page_header(page_key))

        workspace_body = QWidget()
        workspace_body.setObjectName(f'{page_key}WorkspaceBody')
        workspace_body.setProperty('uiRole', 'workspaceBody')
        workspace_body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        workspace_body_layout = QVBoxLayout(workspace_body)
        workspace_body_layout.setContentsMargins(0, 0, 0, 0)
        workspace_body_layout.setSpacing(0)
        workspace_layout.addWidget(workspace_body, 1)

        content_layout.addWidget(workspace, 1)
        page_layout.addWidget(content, 1)
        return page, workspace_body_layout

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
        # Prepare right panel layout for results
        self._init_right_panel_widgets()

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

        self.viewport_stage = QWidget()
        self.viewport_stage.setObjectName('viewportStage')
        stage_layout = QHBoxLayout(self.viewport_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(8)

        stage_layout.addWidget(self.viewport.get_widget(), 1)
        viewport_layout.addWidget(self.viewport_stage, 1)

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
        self.right_results_scroll = scroll

        self.photo_match_right_panel = QScrollArea()
        self.photo_match_right_panel.setObjectName('photoMatchRightPanel')
        self.photo_match_right_panel.setWidgetResizable(True)
        self.photo_match_right_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.photo_match_right_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        match_container = QWidget()
        match_layout = QVBoxLayout(match_container)
        match_layout.setContentsMargins(8, 8, 8, 8)
        match_layout.setSpacing(8)

        match_title = QLabel('二维视口')
        match_title.setProperty('uiRole', 'sectionTitle')
        match_layout.addWidget(match_title)

        facade_title = QLabel('检测立面（选择后生成并映射热力图）')
        facade_title.setProperty('uiRole', 'supportingText')
        self.photo_match_facade_list = QListWidget()
        self.photo_match_facade_list.setObjectName('photoMatchFacadeList')
        self.photo_match_facade_list.setIconSize(QSize(16, 16))
        self.photo_match_facade_list.setMinimumHeight(96)
        self.photo_match_facade_list.setMaximumHeight(160)
        self.photo_match_facade_list.itemClicked.connect(
            self._on_photo_match_facade_clicked
        )
        self.photo_match_facade_list.currentItemChanged.connect(
            lambda _current, _previous: self._refresh_photo_match_controls()
        )

        photo_title = QLabel('上传的 2D 照片')
        photo_title.setProperty('uiRole', 'supportingText')
        match_layout.addWidget(photo_title)
        self.photo_view = PhotoViewWidget(
            match_container,
            placeholder='请先上传 2D 照片',
        )
        self.photo_view.setMinimumHeight(280)
        self.photo_view.point_clicked.connect(
            self._on_photo_match_point_clicked
        )
        self.photo_view.view_changed.connect(self._update_reprojection_links)
        match_layout.addWidget(self.photo_view)

        cloud_view_title = QLabel('针孔投影二维视图')
        cloud_view_title.setProperty('uiRole', 'supportingText')
        match_layout.addWidget(cloud_view_title)
        self.cloud_match_view = PhotoViewWidget(
            match_container,
            placeholder='请点击“点云映射”',
        )
        self.cloud_match_view.setMinimumHeight(280)
        self.cloud_match_view.set_interactive(False)
        self.cloud_match_view.view_changed.connect(
            self._update_reprojection_links
        )
        match_layout.addWidget(self.cloud_match_view)

        projection_hint = QLabel('投影参数（拖动滑块实时更新）')
        projection_hint.setProperty('uiRole', 'supportingText')
        match_layout.addWidget(projection_hint)
        self._projection_specs = {
            'fov': ('视场角', 30, 150, 1.0),
            'yaw': ('水平角', -180, 180, 1.0),
            'pitch': ('俯仰角', -90, 90, 1.0),
            'roll': ('翻滚角', -180, 180, 1.0),
            'tx': ('相机 X', -500, 500, 10.0),
            'ty': ('相机 Y', -500, 500, 10.0),
            'tz': ('相机 Z', -100, 1000, 10.0),
            'point_size': ('点大小', 1, 11, 1.0),
        }
        for key, (label, minimum, maximum, scale) in self._projection_specs.items():
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            name = QLabel(label)
            name.setFixedWidth(48)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setProperty('projectionKey', key)
            slider.valueChanged.connect(self._on_projection_slider_changed)
            value_label = QLabel('—')
            value_label.setFixedWidth(48)
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(name)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            match_layout.addLayout(row)
            self._projection_sliders[key] = slider
            self._projection_value_labels[key] = value_label

        self.btn_auto_fit_projection = QPushButton('重新自动取景')
        self.btn_auto_fit_projection.clicked.connect(self._auto_fit_projection)
        match_layout.addWidget(self.btn_auto_fit_projection)
        match_layout.addStretch(1)
        match_layout.addWidget(facade_title)
        match_layout.addWidget(self.photo_match_facade_list)
        grid_title = QLabel('选中立面平整度网格图')
        grid_title.setProperty('uiRole', 'supportingText')
        match_layout.addWidget(grid_title)
        self.facade_heatmap_grid_view = PhotoViewWidget(
            match_container,
            placeholder='请先选择立面并点击“热力图映射”',
        )
        self.facade_heatmap_grid_view.setMinimumHeight(300)
        self.facade_heatmap_grid_view.set_interactive(False)
        match_layout.addWidget(self.facade_heatmap_grid_view)

        self.photo_match_right_panel.setWidget(match_container)
        self.photo_match_right_panel.hide()
        panel_layout.addWidget(self.photo_match_right_panel)

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
        self._quality_result_cache.clear()
        try:
            cloud = (self.pointcloud_service.resolve_processing_cloud()
                     if self.pointcloud_service is not None else None)
            if cloud and self.render_service is not None:
                self.render_service.highlight_facades(cloud, results or [])
        except Exception as exc:
            print(f'[PCFD] facade.color_refresh_failed error={exc!r}', flush=True)
        
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
        self._latest_facade_results = results
        self.project_operation_service._last_facade_results = results
        if hasattr(self, 'facade_heatmap_grid_view'):
            self.facade_heatmap_grid_view.clear_image()
        self._heatmap_projection_data = None
        self._populate_photo_match_facades(results)
        if not results:
            self.list_facades.clear()
            self._refresh_heatmap_button_state()
            self._latest_facade_results = []
            self.project_operation_service._last_facade_results = []
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
            
            row = QWidget()
            row.setStyleSheet("""
                QWidget { background: transparent; }
                QLabel { font-size: 12px; color: #334155; }
            """)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 6, 8, 6)
            row_layout.setSpacing(8)
            
            info = QLabel(f"立面{display_no}　点数 {int(f.get('point_count') or 0):,}")
            info.setToolTip(f"业务索引 facade_id={int(f.get('id', 0))}")
            info.setStyleSheet('font-size: 12px; color: #334155;')
            row_layout.addWidget(info, 1)
            
            color = self.render_service.facade_color_for(f, display_no)
            swatch = QFrame()
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(
                'background-color: rgb(%d,%d,%d); border: 1px solid #94a3b8; border-radius: 3px;' %
                tuple(int(max(0, min(1, x)) * 255) for x in color)
            )
            swatch.setToolTip('该立面在视口中的显示颜色')
            row_layout.addWidget(swatch)
            
            status = self._facade_review_status(f)
            action_button = QPushButton(
                '完整' if status == 'complete' else '确认完整')
            action_button.setFixedWidth(72)
            action_button.setMinimumHeight(26)
            action_button.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    padding: 2px 8px;
                    border-radius: 4px;
                    border: 1px solid #cbd5e1;
                    background: #ffffff;
                    color: #475569;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    border-color: #94a3b8;
                }
            """)
            action_button.setToolTip('点击切换完整/不完整；仅完整立面允许质量计算')
            action_button.clicked.connect(
                lambda _=False, obj=f, button=action_button:
                self._toggle_facade_review_status(obj, button))
            row_layout.addWidget(action_button)
            
            row.setMaximumHeight(48)
            self.list_facades.setItemWidget(item, row)

        self._latest_facade_results = results
        self.project_operation_service._last_facade_results = results
        self._refresh_heatmap_button_state()
        self._refresh_report_preview()

    def _compatible_quality_results(self):
        results = getattr(self.project_operation_service, '_last_facade_results', None) or []
    def _populate_photo_match_facades(self, results):
        """把检测结果同步到 2D-3D 工作区右侧的可选立面列表。"""
        if not hasattr(self, 'photo_match_facade_list'):
            return
        selected_id = None
        current = self.photo_match_facade_list.currentItem()
        if current is not None:
            selected_id = int(
                (current.data(Qt.ItemDataRole.UserRole) or {}).get('id', -1)
            )
        self.photo_match_facade_list.clear()
        selected_item = None
        for display_no, facade in enumerate(results or [], 1):
            facade['display_no'] = int(facade.get('display_no', display_no))
            item = QListWidgetItem(
                f"立面 {facade['display_no']}　"
                f"点数 {int(facade.get('point_count') or len(facade.get('inlier_indices') or [])):,}"
            )
            color = self.render_service.facade_color_for(
                facade,
                display_no - 1,
            )
            swatch = QPixmap(16, 16)
            swatch.fill(
                QColor.fromRgbF(
                    *tuple(float(np.clip(channel, 0.0, 1.0)) for channel in color)
                )
            )
            item.setIcon(QIcon(swatch))
            item.setData(Qt.ItemDataRole.UserRole, facade)
            item.setToolTip('选择该立面后可将其平整度热力图映射到照片')
            self.photo_match_facade_list.addItem(item)
            if selected_id is not None and int(facade.get('id', -2)) == selected_id:
                selected_item = item
        if selected_item is not None:
            self.photo_match_facade_list.setCurrentItem(selected_item)

    def _clear_facade_angle_adjustment(self, restore=False):
        had_adjusted_view = (
            self._photo_angle_adjusted_image is not None
            or self._cloud_angle_adjusted_image is not None
        )
        self._photo_angle_adjusted_image = None
        self._cloud_angle_adjusted_image = None
        if not restore or not had_adjusted_view:
            return
        state = self.photo_match_service.state
        source = (
            self._photo_rectified_image
            if state.annotation_space == 'rectified'
            and self._photo_rectified_image is not None
            else self._photo_raw_image
        )
        if source is not None and not source.isNull():
            self._photo_heatmap_image = None
            self._photo_match_display_image = source
            self.photo_view.set_image(source)
        if self._live_projection_view is not None:
            cloud_bgr = self._live_projection_view.get('view_bgr')
            if cloud_bgr is not None:
                self.cloud_match_view.set_image(bgr_to_qimage(cloud_bgr))

    def _focus_view_on_facade(self, facade):
        """正对并缩放到选中的立面。"""
        if not facade or not hasattr(self.viewport, 'focus_on_plane'):
            return False
        try:
            center = np.asarray(facade.get('center'), dtype=np.float64).reshape(3)
            normal_value = facade.get('normal')
            if normal_value is None:
                plane = np.asarray(
                    facade.get('plane_model'), dtype=np.float64
                ).reshape(-1)
                normal_value = plane[:3] if len(plane) >= 3 else None
            normal = np.asarray(normal_value, dtype=np.float64).reshape(3)
            if not np.isfinite(center).all() or not np.isfinite(normal).all():
                return False

            bbox = facade.get('bbox_2d') or {}
            u_span = abs(float(bbox.get('u_max', 0.0))
                         - float(bbox.get('u_min', 0.0)))
            v_span = abs(float(bbox.get('v_max', 0.0))
                         - float(bbox.get('v_min', 0.0)))
            area_extent = np.sqrt(max(float(facade.get('area', 0.0)), 0.0))
            extent = max(u_span, v_span, area_extent, 1.0)
            return bool(
                self.viewport.focus_on_plane(
                    center,
                    normal,
                    extent=extent,
                    zoom=0.55,
                )
            )
        except (TypeError, ValueError):
            return False

    def _on_photo_match_facade_clicked(self, item):
        facade = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        cloud = self._active_cloud_name()
        self._clear_facade_angle_adjustment(restore=True)
        self._heatmap_projection_data = None
        if facade and cloud:
            if hasattr(self, 'facade_heatmap_grid_view'):
                self.facade_heatmap_grid_view.clear_image()
            self.render_service.select_facade(
                cloud,
                int(facade.get('id', 0)),
                facade=facade,
            )
            self._focus_view_on_facade(facade)
        self._refresh_photo_match_controls()

    def _refresh_heatmap_button_state(self):
        button = getattr(self, 'btn_heatmap_toggle', None)
        if button is None:
            return
        enabled = bool(self._compatible_quality_results())
        button.setEnabled(enabled)
        mode = getattr(self, '_heatmap_mode', 'flatness')
        current = '平整度' if mode == 'flatness' else '垂直度'
        next_mode = '垂直度' if mode == 'flatness' else '平整度'
        button.setText(f'热力切换显示（当前：{current}）')
        button.setToolTip(f'点击切换至{next_mode}热力映射')

    def _toggle_heatmap_display(self):
        if not self._compatible_quality_results():
            self._refresh_heatmap_button_state()
            return
        self._heatmap_mode = ('verticality' if self._heatmap_mode == 'flatness'
                              else 'flatness')
        cloud = self._active_cloud_name()
        self.render_service.render_quality_reports(
            cloud, getattr(self.project_operation_service, '_last_facade_results', None) or [],
            index_service=self.facade_service._index_service,
            heatmap_mode=self._heatmap_mode)
        self._refresh_heatmap_button_state()

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

    def _toggle_photo_match_workspace(self):
        opening = not self.photo_match_panel.isVisible()
        if not opening:
            return
        if opening:
            self.station_panel.hide()
            self.pointcloud_controls.hide()
            self.photo_match_panel.show()
            self.right_results_scroll.hide()
            self.photo_match_right_panel.show()
            if self._sidebar_collapsed.get('left'):
                self._expand_sidebar('left', self.left_dock)
            else:
                self.left_dock.show()
            if self._sidebar_collapsed.get('right'):
                self._expand_sidebar('right', self.right_dock)
            else:
                self.right_dock.show()
            self._populate_photo_match_facades(
                getattr(
                    self.project_operation_service,
                    '_last_facade_results',
                    None,
                ) or []
            )
            self.statusBar().showMessage(
                '已进入 2D-3D 匹配工作区',
                4000,
            )
            self._auto_preprocess_match_cloud()
        else:
            self._leave_photo_match_workspace()
        self._refresh_photo_match_controls()

    def _activate_default_operation_workspace(self, module_name):
        """显示点云处理和立面分析功能共用的左右面板。"""
        self._leave_photo_match_workspace()
        if self._sidebar_collapsed.get('left'):
            self._expand_sidebar('left', self.left_dock)
        else:
            self.left_dock.show()
        if self._sidebar_collapsed.get('right'):
            self._expand_sidebar('right', self.right_dock)
        else:
            self.right_dock.show()
        self.statusBar().showMessage(f'已切换到{module_name}模块', 3000)

    def _run_default_operation(self, module_name, callback):
        self._activate_default_operation_workspace(module_name)
        callback()

    def _run_facade_detection(self):
        """切换到立面检测工作区后执行检测或加载缓存。"""
        self._activate_default_operation_workspace('立面检测')
        self.project_operation_service.facade_detection()

    def _leave_photo_match_workspace(self):
        if self.photo_match_service.state.annotating:
            try:
                self.photo_match_service.exit_annotation(force=True)
            except Exception:
                pass
        self.render_service.exit_pick_mode()
        self.photo_view.set_interactive(False)
        self.photo_match_panel.hide()
        self.photo_match_right_panel.hide()
        self.right_results_scroll.show()
        self.station_panel.show()
        self.pointcloud_controls.show()

    def _upload_match_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '上传 2D 照片',
            self._last_upload_directory,
            PHOTO_FILE_FILTER,
        )
        if not file_path:
            return
        try:
            image = self.photo_match_service.load_photo(file_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '上传 2D 照片', str(exc))
            return
        self._last_upload_directory = str(Path(file_path).parent)
        self._photo_match_display_image = image
        self._photo_heatmap_image = None
        self._clear_facade_angle_adjustment()
        # 新上传的照片使摆正结果失效，自动匹配需要重新回退到用原图。
        self._photo_raw_image = image
        self._photo_rectified_image = None
        self.photo_view.set_image(image)
        self.photo_view.set_markers([])
        self.photo_view.set_remap_markers([])
        self._show_match_markers = True
        self._refresh_manual_match_markers()
        self._refresh_photo_match_controls()

    def _rectify_match_photo(self):
        try:
            result = self.photo_match_service.rectify_perspective()
            image = bgr_to_qimage(result['rectified_bgr'])
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, '摆正照片', str(exc))
            return
        self._photo_match_display_image = image
        self._photo_rectified_image = image
        self.photo_view.set_image(image)
        self.photo_view.set_markers([])
        self.photo_view.set_remap_markers([])
        self._refresh_photo_match_controls()
        QMessageBox.information(
            self,
            '摆正照片',
            f"照片摆正完成：{result.get('method', 'unknown')}",
        )

    def _grayscale_match_photo(self):
        state = self.photo_match_service.state
        if not state.photo_path:
            QMessageBox.information(self, '黑白化照片', '请先上传 2D 照片。')
            return
        source = (
            self._photo_rectified_image
            if state.rectified and self._photo_rectified_image is not None
            else self._photo_raw_image
        )
        if source is None or source.isNull():
            QMessageBox.warning(self, '黑白化照片', '当前照片不可用。')
            return
        grayscale = source.convertToFormat(
            QImage.Format.Format_Grayscale8
        ).convertToFormat(QImage.Format.Format_RGB888)
        if state.rectified and self._photo_rectified_image is not None:
            self._photo_rectified_image = grayscale
        else:
            self._photo_raw_image = grayscale
        self._photo_match_display_image = grayscale
        self.photo_view.set_image(grayscale)
        self._refresh_manual_match_markers()
        self.statusBar().showMessage('照片已转换为黑白图像', 5000)

    def _upload_scan_pose(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '上传扫描仪位姿',
            self._last_upload_directory,
            SCAN_POSE_FILE_FILTER,
        )
        if not file_path:
            return
        try:
            meta = self.photo_match_service.load_scan_pose(file_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '上传扫描仪位姿', str(exc))
            return
        self._last_upload_directory = str(Path(file_path).parent)
        self._refresh_photo_match_controls()
        origin = meta.get('origin') or []
        origin_text = ', '.join(f'{float(value):.3f}' for value in origin)
        self.statusBar().showMessage(
            f'扫描仪位姿已加载：({origin_text})',
            5000,
        )

    def _set_projection_controls(self, params):
        self._projection_controls_updating = True
        try:
            for key, slider in self._projection_sliders.items():
                if key not in params:
                    continue
                scale = self._projection_specs[key][3]
                raw = int(round(float(params[key]) * scale))
                slider.setValue(max(slider.minimum(), min(slider.maximum(), raw)))
                value = slider.value() / scale
                suffix = '°' if key in ('fov', 'yaw', 'pitch', 'roll') else ''
                self._projection_value_labels[key].setText(
                    f'{value:.1f}{suffix}' if scale > 1 else f'{value:.0f}{suffix}'
                )
        finally:
            self._projection_controls_updating = False

    def _projection_params_from_controls(self):
        params = dict(self.photo_match_service.state.projection_params or {})
        for key, slider in self._projection_sliders.items():
            scale = self._projection_specs[key][3]
            value = slider.value() / scale
            params[key] = int(value) if key == 'point_size' else float(value)
        params.setdefault('near', 0.3)
        params.setdefault('far', 300.0)
        return params

    def _projection_image_size(self, max_side=1024):
        """按右侧投影显示框宽高比生成渲染尺寸，避免上下留黑。"""
        width = max(1, int(self.cloud_match_view.width()))
        height = max(1, int(self.cloud_match_view.height()))
        if width <= 10 or height <= 10:
            return int(max_side), int(max_side)
        scale = float(max_side) / max(width, height)
        return (
            max(320, int(round(width * scale))),
            max(320, int(round(height * scale))),
        )

    def _on_projection_slider_changed(self, _value):
        sender = self.sender()
        key = sender.property('projectionKey') if sender is not None else None
        if key in self._projection_sliders:
            scale = self._projection_specs[key][3]
            value = self._projection_sliders[key].value() / scale
            suffix = '°' if key in ('fov', 'yaw', 'pitch', 'roll') else ''
            self._projection_value_labels[key].setText(
                f'{value:.1f}{suffix}' if scale > 1 else f'{value:.0f}{suffix}'
            )
        if self._projection_controls_updating:
            return
        self.photo_match_service.state.projection_params = (
            self._projection_params_from_controls()
        )
        self._saved_match_view = None
        self._saved_match_view_path = None
        self._live_projection_view = None
        if self._projection_enabled:
            self._projection_refresh_timer.start()
        self._refresh_photo_match_controls()

    def _auto_fit_projection(self):
        cloud_name = self._active_cloud_name()
        data = self.viewport.get_cloud_data(cloud_name) if cloud_name else None
        points = None if not data else data.get('pos')
        if points is None or len(points) == 0:
            QMessageBox.information(self, '针孔投影', '请先加载点云。')
            return
        try:
            params = self.photo_match_service.initialize_projection(
                points,
                image_size=self._projection_image_size(),
            )
            camera = self.photo_match_service.build_projection_viewport_camera(
                params
            )
            applied = self.viewport.apply_scan_pose_view(
                camera['eye'],
                camera['lookat'],
                camera['up'],
                zoom=float(getattr(Config, 'INITIAL_VIEW_ZOOM', 0.45)),
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            QMessageBox.warning(self, '针孔投影', str(exc))
            return
        if not applied:
            QMessageBox.warning(self, '针孔投影', '视口相机设置失败。')
            return
        self._set_projection_controls(params)
        self._saved_match_view = None
        self._saved_match_view_path = None
        if self._projection_enabled:
            self._refresh_projection_preview()

    def _refresh_projection_preview(self):
        cloud_name = self._active_cloud_name()
        data = self.viewport.get_cloud_data(cloud_name) if cloud_name else None
        if not data or data.get('pos') is None:
            return
        try:
            points = data['pos']
            colors = data.get('color')
            # 滑块拖动时使用均匀代理，保存时仍使用完整显示点云。
            step = max(1, (len(points) + 349_999) // 350_000)
            preview_points = points[::step]
            preview_colors = colors[::step] if colors is not None else None
            rendered = self.photo_match_service.render_projection_view(
                preview_points,
                preview_colors,
                self._projection_params_from_controls(),
                crop_subject=False,
                image_size=self._projection_image_size(),
            )
            image = bgr_to_qimage(rendered['view_bgr'])
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.statusBar().showMessage(f'针孔投影失败：{exc}', 5000)
            return
        self._live_projection_view = rendered
        self.cloud_match_view.set_image(image)
        self.photo_match_service.clear_projection_view_points()
        self._refresh_manual_match_markers()
        self._refresh_photo_match_controls()

    def _apply_scan_pose_view(self, *, title='调整点云视角', announce=True):
        cloud_name = self._active_cloud_name()
        if not cloud_name:
            QMessageBox.information(self, title, '请先加载点云。')
            return False
        data = self.viewport.get_cloud_data(cloud_name)
        points = None if not data else data.get('pos')
        if points is None or len(points) == 0:
            QMessageBox.warning(self, title, '当前点云为空。')
            return False
        points = np.asarray(points, dtype=np.float64)
        finite_points = points[np.isfinite(points).all(axis=1)]
        if len(finite_points) == 0:
            QMessageBox.warning(self, title, '当前点云不包含有效坐标。')
            return False
        try:
            projection_params = self.photo_match_service.initialize_projection(
                finite_points,
                image_size=self._projection_image_size(),
            )
            camera = self.photo_match_service.build_projection_viewport_camera(
                projection_params
            )
            applied = self.viewport.apply_scan_pose_view(
                camera['eye'],
                camera['lookat'],
                camera['up'],
                zoom=float(getattr(Config, 'INITIAL_VIEW_ZOOM', 0.45)),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return False
        if not applied:
            QMessageBox.warning(self, title, '视口相机设置失败。')
            return False
        self._saved_match_view = None
        self._saved_match_view_path = None
        self._live_projection_view = None
        self._projection_enabled = False
        self._set_projection_controls(projection_params)
        self.cloud_match_view.clear_image()
        self._refresh_photo_match_controls()
        if announce:
            self.statusBar().showMessage(
                '已调整三维点云视角；可点击“点云映射”生成投影', 5000
            )
        return True

    def _map_cloud_to_view(self):
        if not self._apply_scan_pose_view(title='点云映射', announce=False):
            return
        self._save_cloud_match_view(title='点云映射')

    def _auto_preprocess_match_cloud(self):
        cloud_name = self._active_cloud_name()
        if not cloud_name or self._match_denoise_busy:
            return
        if self._match_preprocessed_cloud == cloud_name:
            return
        self._remove_match_cloud_outliers(silent=True)

    def _remove_match_cloud_outliers(self, silent=False):
        if self._match_denoise_busy:
            return
        if not self._active_cloud_name():
            if not silent:
                QMessageBox.information(self, '点云消除离群点', '请先加载点云。')
            return
        self._match_denoise_busy = True
        if hasattr(self, 'btn_remove_match_outliers'):
            self.btn_remove_match_outliers.setEnabled(False)
            self.btn_remove_match_outliers.setText('正在消除离群点…')
        self._saved_match_view = None
        self._saved_match_view_path = None
        self._live_projection_view = None
        self.statusBar().showMessage('正在消除离群点并下采样…')
        self.project_operation_service.denoise(
            trim_far=True,
            far_bin_width=5.0,
            far_min_fraction=0.005,
            absolute_max_range=200.0,
            display_voxel_size=0.10,
        )
        if getattr(self.project_operation_service, '_denoise_thread', None) is None:
            self._on_match_cloud_denoised(None)

    def _on_match_cloud_denoised(self, stats):
        self._match_denoise_busy = False
        if hasattr(self, 'btn_remove_match_outliers'):
            self.btn_remove_match_outliers.setText('点云消除离群点')
        self._saved_match_view = None
        self._saved_match_view_path = None
        if stats:
            self._match_preprocessed_cloud = stats.get('name') or self._active_cloud_name()
        if (
            self._projection_enabled
            and self.photo_match_service.state.projection_params
        ):
            self._refresh_projection_preview()
        self._refresh_photo_match_controls()
        if stats:
            self.statusBar().showMessage(
                f"离群点处理完成：{stats.get('points_before', 0)} → "
                f"{stats.get('points_after', 0)} 个代理点",
                8000,
            )
            if stats.get('loaded_from_cache'):
                message = (
                    f"已加载离群点与下采样缓存："
                    f"{stats.get('points_after', 0)} 个代理点"
                )
            else:
                message = (
                    f"离群点消除与下采样完成并已缓存："
                    f"{stats.get('points_before', 0)} → "
                    f"{stats.get('points_after', 0)} 个代理点"
                )
            self.statusBar().showMessage(message, 8000)

    def _enter_manual_match_mode(self):
        if not self._active_cloud_name():
            QMessageBox.information(self, '进入标注模式', '请先加载点云。')
            return
        try:
            self.photo_match_service.enter_annotation()
        except ValueError as exc:
            QMessageBox.information(self, '进入标注模式', str(exc))
            return
        self.render_service.clear_pick_markers()
        self._show_match_markers = True
        self._refresh_manual_match_markers()
        self._sync_manual_match_phase()
        self._refresh_photo_match_controls()

    def _sync_manual_match_phase(self):
        state = self.photo_match_service.state
        if not state.annotating:
            self.render_service.exit_pick_mode()
            self.photo_view.set_interactive(False)
            return
        if state.next_is_photo:
            self.render_service.exit_pick_mode()
            self.photo_view.set_interactive(True)
            self.statusBar().showMessage('请在照片上点击一个匹配点')
        else:
            self.photo_view.set_interactive(False)
            self.render_service.enter_pick_mode(
                callback=self._on_manual_cloud_point_picked,
                cloud_name=self._active_cloud_name(),
                pick_radius=32,
            )
            self.statusBar().showMessage('请在三维点云上点击对应点')

    def _on_photo_match_point_clicked(self, pixel_x, pixel_y):
        try:
            self.photo_match_service.add_photo_point(pixel_x, pixel_y)
        except RuntimeError as exc:
            QMessageBox.information(self, '照片标点', str(exc))
            return
        print(
            f'[2D-3D] 照片点: x={pixel_x:.3f}, y={pixel_y:.3f}',
            flush=True,
        )
        self._refresh_manual_match_markers()
        self._sync_manual_match_phase()
        self._refresh_photo_match_controls()

    def _on_manual_cloud_point_picked(self, picked):
        point = picked.get('point') if isinstance(picked, dict) else picked
        try:
            self.photo_match_service.add_cloud_point(point)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        xyz = np.asarray(point, dtype=np.float64).reshape(3)
        count = self.photo_match_service.complete_pair_count()
        print(
            f'[2D-3D] 点对 #{count} 三维点: '
            f'X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f}',
            flush=True,
        )
        self._refresh_manual_match_markers()
        self._sync_manual_match_phase()
        self._refresh_photo_match_controls()

    def _undo_manual_match_point(self):
        self.photo_match_service.undo_last()
        self._refresh_manual_match_markers()
        self._sync_manual_match_phase()
        self._refresh_photo_match_controls()

    def _exit_manual_match_mode(self):
        try:
            count = self.photo_match_service.exit_annotation()
        except ValueError as exc:
            QMessageBox.information(self, '退出标注', str(exc))
            return
        self._sync_manual_match_phase()
        self._refresh_photo_match_controls()
        self.statusBar().showMessage(
            f'已退出标注，共完成 {count} 对匹配点',
            5000,
        )

    def _estimate_manual_match_matrix(self):
        try:
            result = self.photo_match_service.solve_pose()
        except ValueError as exc:
            QMessageBox.warning(self, '粗略估计匹配矩阵', str(exc))
            return
        matrix = np.asarray(
            result.get('match_matrix') or result.get('projection_matrix'),
            dtype=np.float64,
        ).reshape(3, 4)
        matrix_text = np.array2string(
            matrix,
            precision=5,
            suppress_small=True,
        )
        print(f'[2D-3D] 匹配矩阵 P=K[R|t]:\n{matrix_text}', flush=True)
        self._refresh_photo_match_controls()
        QMessageBox.information(
            self,
            '粗略估计匹配矩阵',
            (
                '匹配矩阵估计完成。\n'
                f"有效内点：{result.get('inlier_count', 0)}/"
                f"{result.get('point_count', 0)}\n"
                f"平均重投影误差："
                f"{float(result.get('reprojection_mean_px', 0.0)):.2f} px\n\n"
                f'P =\n{matrix_text}'
            ),
        )

    def _save_cloud_match_view(self, *, title='投影到二维视图'):
        if not self._active_cloud_name():
            QMessageBox.information(self, title, '请先加载点云。')
            return

        self._clear_facade_angle_adjustment(restore=True)
        self.render_service.exit_pick_mode()
        self.render_service.clear_pick_markers()
        try:
            cloud_name = self._active_cloud_name()
            data = self.viewport.get_cloud_data(cloud_name)
            if not data or data.get('pos') is None:
                raise ValueError('当前点云为空')
            state = self.photo_match_service.state
            if state.scan_pose_path:
                if not state.projection_params:
                    params = self.photo_match_service.initialize_projection(
                        data['pos']
                    )
                    self._set_projection_controls(params)
                captured = self.photo_match_service.render_projection_view(
                    data['pos'],
                    data.get('color'),
                    self._projection_params_from_controls(),
                    crop_subject=True,
                    image_size=self._projection_image_size(),
                )
            else:
                # 未上传扫描仪位姿时直接复用当前三维视口相机，因此打开
                # 二维-三维对齐模块后即可投影，不要求先执行视角调整。
                captured = self.viewport.capture_match_view(cloud_name)
            view_image = bgr_to_qimage(captured['view_bgr'])
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return

        self._projection_enabled = True
        self._live_projection_view = captured
        self._saved_match_view = None
        self._saved_match_view_path = None
        self.cloud_match_view.set_image(view_image)
        self.photo_match_service.clear_projection_view_points()
        self._refresh_manual_match_markers()
        self._refresh_photo_match_controls()
        self.statusBar().showMessage(
            '点云已映射到右侧二维视图', 5000
        )

    def _export_current_cloud_view_images(self):
        cloud_name = self._active_cloud_name()
        if not cloud_name:
            QMessageBox.information(self, '保存点云视图', '请先加载点云。')
            return
        if self._live_projection_view is None or not self._projection_enabled:
            QMessageBox.information(
                self,
                '保存点云视图',
                '请先点击“点云映射”。',
            )
            return
        if self._view_export_worker is not None:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            '选择五张点云视图图片的保存目录',
            self._last_upload_directory,
        )
        if not output_dir:
            return

        data = self.viewport.get_cloud_data(cloud_name)
        dataset_id = data.get('dataset_id') if data else None
        dataset = (
            self.pointcloud_service.get_dataset(dataset_id)
            if dataset_id
            else None
        )
        points = colors = None
        if dataset is not None:
            points = dataset.index.get_source_points()
            colors = dataset.index.get_source_colors()
            if points is None:
                points = dataset.raw.points
                colors = dataset.raw.colors
        if points is None or len(points) == 0:
            QMessageBox.warning(
                self,
                '保存点云视图',
                '无法取得原始点云数据。',
            )
            return

        try:
            camera_view = self.viewport.capture_match_view(cloud_name=cloud_name)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, '保存点云视图', str(exc))
            return
        height, width = camera_view['depth_image'].shape
        pinhole_height, pinhole_width = (
            self._live_projection_view['depth_image'].shape
        )
        stem = (
            f'{Path(cloud_name).stem}_'
            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        worker = ViewExportWorker(
            points=points,
            colors=colors,
            camera_matrix=camera_view['camera_matrix'],
            extrinsic=camera_view['extrinsic'],
            image_size=(width, height),
            pinhole_camera_matrix=self._live_projection_view['camera_matrix'],
            pinhole_extrinsic=self._live_projection_view['extrinsic'],
            pinhole_image_size=(pinhole_width, pinhole_height),
            output_dir=output_dir,
            stem=stem,
        )
        worker.signals.finished.connect(self._on_cloud_view_images_exported)
        worker.signals.failed.connect(self._on_cloud_view_images_export_failed)
        self._view_export_worker = worker
        self.btn_export_cloud_view_images.setEnabled(False)
        self.btn_export_cloud_view_images.setText('正在导出原始点云视图…')
        self.statusBar().showMessage('正在使用原始点云生成五张视图图片…')
        QThreadPool.globalInstance().start(worker)

    def _on_cloud_view_images_exported(self, paths):
        self._view_export_worker = None
        self.btn_export_cloud_view_images.setText('保存点云当前视图图片')
        self._refresh_photo_match_controls()
        folder = str(Path(next(iter(paths.values()))).parent) if paths else ''
        QMessageBox.information(
            self,
            '保存点云视图',
            f'已保存 raw、smooth、depth、depth-edge、针孔投影共 5 张图片。\n{folder}',
        )

    def _on_cloud_view_images_export_failed(self, message):
        self._view_export_worker = None
        self.btn_export_cloud_view_images.setText('保存点云当前视图图片')
        self._refresh_photo_match_controls()
        QMessageBox.warning(self, '保存点云视图', message)

    def _auto_match_current_view(self):
        if not self._active_cloud_name():
            QMessageBox.information(self, '自动匹配', '请先加载点云。')
            return
        # 若照片已摆正，自动匹配必须使用摆正后的照片（而不是原图）与保存的
        # 点云视角图片做特征匹配；未摆正时才回退到原图。这里显式按摆正状态
        # 选择图源，不依赖照片显示框当前恰好展示的是哪一张。
        state = self.photo_match_service.state
        use_rectified = bool(state.rectified) and self._photo_rectified_image is not None \
            and not self._photo_rectified_image.isNull()
        image = self._photo_rectified_image if use_rectified else self._photo_raw_image
        if image is None or image.isNull():
            QMessageBox.information(self, '自动匹配', '请先上传 2D 照片。')
            return
        if self._live_projection_view is None or not self._projection_enabled:
            QMessageBox.information(
                self,
                '自动匹配',
                '请先点击“点云映射”。',
            )
            return

        photo_source_label = '摆正后的照片' if use_rectified else '原始照片'
        self.render_service.exit_pick_mode()
        self.render_service.clear_pick_markers()
        self.photo_view.set_markers([])
        self.photo_view.set_remap_markers([])
        self.cloud_match_view.set_markers([])
        self._clear_facade_angle_adjustment(restore=True)
        self._show_match_markers = True
        self.btn_auto_match_view.setEnabled(False)
        self.photo_match_status.setText(
            f'正在使用 SuperPoint + LightGlue 匹配'
            f'{photo_source_label}与右侧当前二维投影…'
        )
        QApplication.processEvents()
        try:
            rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
            height, width = rgb_image.height(), rgb_image.width()
            rows = np.frombuffer(
                rgb_image.constBits(),
                dtype=np.uint8,
                count=rgb_image.sizeInBytes(),
            ).reshape(height, rgb_image.bytesPerLine())
            rgb = rows[:, :width * 3].reshape(height, width, 3).copy()
            result = self.photo_match_service.auto_match_current_view(
                rgb[:, :, ::-1],
                self._live_projection_view,
            )
        except (KeyError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            self._refresh_photo_match_controls()
            QMessageBox.warning(self, '自动匹配', str(exc))
            return

        self._refresh_manual_match_markers()
        self._refresh_photo_match_controls()
        matrix_data = result.get('match_matrix') or result.get(
            'projection_matrix'
        )
        if not result.get('pose_estimated') or matrix_data is None:
            count = int(result.get('depth_match_count', 0))
            needed = max(0, 6 - count)
            QMessageBox.information(
                self,
                '自动匹配点已保存',
                (
                    f'自动识别并保存了 {count} 对有效 2D-3D 匹配点。\n'
                    f'请进入手动标注模式，再补充至少 {needed} 对；'
                    '退出标注后可统一估算匹配矩阵。'
                ),
            )
            return
        matrix = np.asarray(
            matrix_data,
            dtype=np.float64,
        ).reshape(3, 4)
        matrix_text = np.array2string(matrix, precision=5, suppress_small=True)
        print(f'[2D-3D] 自动匹配矩阵 P=K[R|t]:\n{matrix_text}', flush=True)
        QMessageBox.information(
            self,
            '自动匹配完成',
            (
                f"算法：{result.get('feature_algorithm', 'SuperPoint + LightGlue')}"
                f"（{result.get('inference_device', 'unknown')}）\n"
                f"2D 特征匹配：{result.get('feature_match_count', 0)}\n"
                f"几何一致匹配：{result.get('geometric_match_count', 0)}\n"
                f"有效 2D-3D 内点：{result.get('inlier_count', 0)}/"
                f"{result.get('depth_match_count', 0)}\n"
                f"平均重投影误差："
                f"{float(result.get('reprojection_mean_px', 0.0)):.2f} px\n"
                f"照片来源：{photo_source_label}\n"
                '点云视图：右侧当前二维投影'
            ),
        )

    def _generate_selected_facade_heatmap(self, *, title='生成热力图', announce=True):
        selected = self.photo_match_facade_list.currentItem()
        if selected is None:
            QMessageBox.information(
                self,
                title,
                '请先从右侧立面列表中选择一个立面。',
            )
            return False
        cloud_name = self._active_cloud_name()
        if not cloud_name:
            QMessageBox.information(self, title, '请先加载点云。')
            return False
        facade = selected.data(Qt.ItemDataRole.UserRole)
        if not isinstance(facade, dict):
            QMessageBox.warning(
                self,
                title,
                '所选立面缓存数据无效，请重新选择立面。',
            )
            return False
        try:
            stats = self.render_service.render_selected_facade_flatness(
                cloud_name,
                facade,
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return False
        self._heatmap_scale = {
            'facade_id': facade.get('id'),
            'neutral_mm': float(stats.get('neutral_mm', 5.0)),
            'limit_mm': float(stats.get('limit_mm', 10.0)),
            'vmin_mm': float(stats.get('vmin_mm', -stats.get('limit_mm', 10.0))),
            'vmax_mm': float(stats.get('vmax_mm', stats.get('limit_mm', 10.0))),
        }
        self._heatmap_projection_data = {
            key: stats.get(key)
            for key in (
                'facade_id',
                'points_3d',
                'values_mm',
                'plane_model',
                'grid_layout',
            )
        }
        grid_bgr = stats.get('grid_bgr')
        if grid_bgr is not None:
            self.facade_heatmap_grid_view.set_image(
                bgr_to_qimage(grid_bgr)
            )
        self._show_match_markers = False
        self._refresh_manual_match_markers()
        self._refresh_photo_match_controls()
        if announce:
            self.statusBar().showMessage(
                f"立面 {int(facade.get('display_no', 1))} 平整度热力图已生成："
                '灰色=平整（±5 mm），蓝色=凹陷，红色=凸起；'
                f"范围 ±{float(stats.get('limit_mm', 0.0)):.1f} mm。"
                '匹配点对已隐藏。',
                8000,
            )
        return True

    def _map_heatmap_to_photo_and_cloud(self):
        selected = self.photo_match_facade_list.currentItem()
        facade = (
            selected.data(Qt.ItemDataRole.UserRole)
            if selected is not None
            else None
        )
        cached_id = (
            (self._heatmap_projection_data or {}).get('facade_id')
        )
        if (
            not facade
            or int(cached_id if cached_id is not None else -1)
            != int(facade.get('id', -2))
        ):
            if not self._generate_selected_facade_heatmap(
                title='热力图映射',
                announce=False,
            ):
                return
        self._map_heatmap_back_to_photo(title='热力图映射')

    def _adjust_photo_to_facade_angle(self):
        title = '调整照片角度'
        state = self.photo_match_service.state
        selected = self.photo_match_facade_list.currentItem()
        if selected is None:
            QMessageBox.information(
                self, title, '请先从右侧立面列表中选择一个立面。'
            )
            return
        if state.pose is None:
            QMessageBox.information(self, title, '请先完成自动匹配。')
            return
        if self._live_projection_view is None or not self._projection_enabled:
            QMessageBox.information(self, title, '请先点击“点云映射”。')
            return
        facade = selected.data(Qt.ItemDataRole.UserRole)
        cached_id = (
            (self._heatmap_projection_data or {}).get('facade_id')
        )
        if (
            int(cached_id if cached_id is not None else -1)
            != int((facade or {}).get('id', -2))
            and not self._generate_selected_facade_heatmap(
                title=title,
                announce=False,
            )
        ):
            return

        source = (
            self._photo_rectified_image
            if state.annotation_space == 'rectified'
            and self._photo_rectified_image is not None
            else self._photo_raw_image
        )
        if source is None or source.isNull():
            QMessageBox.information(self, title, '请先上传 2D 照片。')
            return

        cloud_name = self._active_cloud_name()
        cloud_data = (
            self.viewport.get_cloud_data(cloud_name) if cloud_name else None
        )
        dataset_id = cloud_data.get('dataset_id') if cloud_data else None
        dataset = (
            self.pointcloud_service.get_dataset(dataset_id)
            if dataset_id
            else None
        )
        if dataset is None or dataset.proxy_points is None:
            QMessageBox.warning(
                self, title, '当前点云处理数据不可用，请重新加载点云。'
            )
            return

        facade_revision = (facade or {}).get('dataset_revision')
        if facade_revision and str(facade_revision) != str(dataset.revision):
            QMessageBox.warning(
                self,
                title,
                '所选立面属于旧点云处理版本，请重新执行立面检测。',
            )
            return

        self.btn_adjust_photo_angle.setEnabled(False)
        self.btn_adjust_photo_angle.setText('正在调整照片角度…')
        QApplication.processEvents()
        try:
            scale = self._heatmap_scale or {}
            same_facade = scale.get('facade_id') == facade.get('id')
            result = self.photo_match_service.align_facade_view_heatmaps(
                self._qimage_to_bgr(source),
                dataset.proxy_points,
                facade,
                cloud_projection=self._live_projection_view,
                heatmap_data=self._heatmap_projection_data,
                neutral_mm=scale.get('neutral_mm') if same_facade else None,
                limit_mm=scale.get('limit_mm') if same_facade else None,
                vmin_mm=scale.get('vmin_mm') if same_facade else None,
                vmax_mm=scale.get('vmax_mm') if same_facade else None,
            )
            photo_image = bgr_to_qimage(result['image_bgr'])
            cloud_image = bgr_to_qimage(result['cloud_image_bgr'])
        except (
            ImportError,
            KeyError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            QMessageBox.warning(self, title, str(exc))
            return
        finally:
            self.btn_adjust_photo_angle.setText('调整照片角度')
            self._refresh_photo_match_controls()

        self._photo_angle_adjusted_image = photo_image
        self._cloud_angle_adjusted_image = cloud_image
        self._photo_heatmap_image = photo_image
        self._photo_match_display_image = photo_image
        self.photo_view.set_image(photo_image)
        self.cloud_match_view.set_image(cloud_image)
        self._show_match_markers = False
        self._refresh_manual_match_markers()
        width, height = result.get('output_size', (0, 0))
        self.statusBar().showMessage(
            f"立面 {int(facade.get('display_no', 1))} 已调整为正视角度；"
            f"热力图已重新显示（{int(width)}×{int(height)}）",
            8000,
        )

    @staticmethod
    def _qimage_to_bgr(image):
        rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
        height, width = rgb_image.height(), rgb_image.width()
        rows = np.frombuffer(
            rgb_image.constBits(),
            dtype=np.uint8,
            count=rgb_image.sizeInBytes(),
        ).reshape(height, rgb_image.bytesPerLine())
        rgb = rows[:, :width * 3].reshape(height, width, 3).copy()
        return rgb[:, :, ::-1]

    def _map_heatmap_back_to_photo(self, *, title='热力图映射回照片'):
        state = self.photo_match_service.state
        selected = self.photo_match_facade_list.currentItem()
        if selected is None:
            QMessageBox.information(
                self,
                title,
                '请先从右侧立面列表中选择一个立面。',
            )
            return
        if state.pose is None:
            QMessageBox.information(
                self,
                title,
                '请先完成二维-三维匹配并估算匹配矩阵。',
            )
            return
        if self._live_projection_view is None:
            QMessageBox.information(
                self,
                title,
                '请先点击“点云映射”，以生成点云二维映射图。',
            )
            return

        source = (
            self._photo_rectified_image
            if state.annotation_space == 'rectified'
            and self._photo_rectified_image is not None
            else self._photo_raw_image
        )
        if source is None or source.isNull():
            QMessageBox.information(
                self,
                title,
                '请先上传 2D 照片。',
            )
            return

        cloud_name = self._active_cloud_name()
        cloud_data = self.viewport.get_cloud_data(cloud_name) if cloud_name else None
        dataset_id = cloud_data.get('dataset_id') if cloud_data else None
        dataset = (
            self.pointcloud_service.get_dataset(dataset_id)
            if dataset_id
            else None
        )
        if dataset is None or dataset.proxy_points is None:
            QMessageBox.warning(
                self,
                title,
                '当前点云处理数据不可用，请重新加载点云。',
            )
            return

        facade = selected.data(Qt.ItemDataRole.UserRole)
        facade_revision = (facade or {}).get('dataset_revision')
        if (
            facade_revision
            and str(facade_revision) != str(dataset.revision)
        ):
            QMessageBox.warning(
                self,
                title,
                '所选立面属于旧点云处理版本，请重新执行立面检测。',
            )
            return
        mapping_button = getattr(self, 'btn_heatmap_map', None)
        if mapping_button is None:
            mapping_button = getattr(self, 'btn_map_heatmap_to_photo', None)
        if mapping_button is not None:
            mapping_button.setEnabled(False)
            mapping_button.setText('正在映射热力图…')
        QApplication.processEvents()
        try:
            scale = self._heatmap_scale or {}
            same_facade = scale.get('facade_id') == facade.get('id')
            result = self.photo_match_service.map_facade_heatmap_to_photo(
                self._qimage_to_bgr(source),
                dataset.proxy_points,
                facade,
                cloud_projection=self._live_projection_view,
                heatmap_data=self._heatmap_projection_data,
                neutral_mm=scale.get('neutral_mm') if same_facade else None,
                limit_mm=scale.get('limit_mm') if same_facade else None,
                vmin_mm=scale.get('vmin_mm') if same_facade else None,
                vmax_mm=scale.get('vmax_mm') if same_facade else None,
            )
            image = bgr_to_qimage(result['image_bgr'])
            cloud_image = bgr_to_qimage(result['cloud_image_bgr'])
        except (ImportError, KeyError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            QMessageBox.warning(self, title, str(exc))
            return
        finally:
            if hasattr(self, 'btn_map_heatmap_to_photo'):
                self.btn_map_heatmap_to_photo.setText('热力图映射回照片')
            if hasattr(self, 'btn_heatmap_map'):
                self.btn_heatmap_map.setText('热力图映射')
            self._refresh_photo_match_controls()

        self._photo_heatmap_image = image
        self._photo_match_display_image = image
        self.photo_view.set_image(image)
        self.cloud_match_view.set_image(cloud_image)
        self._show_match_markers = False
        self._refresh_manual_match_markers()
        self.statusBar().showMessage(
            f"立面 {int(facade.get('display_no', 1))} 热力图已映射到照片"
            f"（{int(result.get('point_count', 0)):,} 点，"
            f"色标 ±{float(result.get('deviation_limit_mm', 0.0)):.1f} mm）",
            8000,
        )

    def _remap_manual_match_points(self):
        try:
            result = (
                self.photo_match_service.remap_cloud_annotations_to_photo()
            )
        except ValueError as exc:
            QMessageBox.warning(self, '重映射', str(exc))
            return
        valid_indices = [
            index
            for index, point in enumerate(
                self.photo_match_service.state.remapped_photo_points
            )
            if point is not None
        ]
        if len(valid_indices) > 10:
            selected = np.random.default_rng().choice(
                valid_indices,
                size=10,
                replace=False,
            )
            valid_indices = sorted(int(value) for value in selected)
        self._reprojection_display_indices = valid_indices
        self._refresh_manual_match_markers()
        mean_error = result.get('mean_reprojection_error_px')
        message = (
            f"已将 {result.get('in_image_count', 0)}/"
            f"{result.get('point_count', 0)} 个三维点重映射到照片。"
        )
        if mean_error is not None:
            message += f'\n平均偏差：{float(mean_error):.2f} px'
        if len(valid_indices) == 10 and result.get('in_image_count', 0) > 10:
            message += '\n匹配点超过 10 对，已随机显示其中 10 对。'
        QMessageBox.information(self, '重映射', message)

    def _on_3d_marker_projection_changed(self, labels):
        self._projected_3d_labels = list(labels or [])
        self._update_reprojection_links()

    def _widget_point_in_window(self, widget, point):
        if point is None:
            return None
        local = widget.pixel_to_widget(point[0], point[1])
        if local is None:
            return None
        global_point = widget.mapToGlobal(
            QPoint(int(round(local.x())), int(round(local.y())))
        )
        mapped = self.mapFromGlobal(global_point)
        return QPointF(float(mapped.x()), float(mapped.y()))

    def _update_reprojection_links(self):
        overlay = getattr(self, '_match_link_overlay', None)
        indices = self._reprojection_display_indices
        if overlay is None or not indices:
            if overlay is not None:
                overlay.set_links([])
            return
        pairs = [
            pair
            for pair in self.photo_match_service.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        projection_by_label = dict(self._projection_marker_points)
        cloud_by_label = {
            int(label): (float(x), float(y))
            for x, y, label in self._projected_3d_labels
        }
        container = getattr(self.viewport, '_container', None)
        links = []
        for index in indices:
            if index >= len(pairs):
                continue
            label = index + 1
            photo = self._widget_point_in_window(
                self.photo_view,
                pairs[index].photo,
            )
            projection = self._widget_point_in_window(
                self.cloud_match_view,
                projection_by_label.get(label),
            )
            cloud_screen = cloud_by_label.get(label)
            cloud = None
            if cloud_screen is not None and container is not None:
                global_point = container.mapToGlobal(
                    QPoint(
                        int(round(cloud_screen[0])),
                        int(round(cloud_screen[1])),
                    )
                )
                mapped = self.mapFromGlobal(global_point)
                cloud = QPointF(float(mapped.x()), float(mapped.y()))
            if photo is not None and projection is not None and cloud is not None:
                links.append((photo, projection, cloud, label))
        overlay.set_links(links)

    def _match_overlay_points(self, pairs=None, labels=None):
        """完整点对在照片与点云映射图上的同号标注。"""
        captured = getattr(self, '_live_projection_view', None)
        camera_matrix = None if captured is None else captured.get('camera_matrix')
        extrinsic = None if captured is None else captured.get('extrinsic')
        source = (
            pairs
            if pairs is not None
            else self.photo_match_service.state.correspondences
        )
        photo_points = []
        photo_labels = []
        cloud_points = []
        cloud_labels = []
        projection_points = []
        sequence = 0
        for index, pair in enumerate(source):
            if pair.photo is None:
                continue
            if labels is None:
                sequence += 1
                label = sequence
            else:
                label = int(labels[index])
            photo_points.append(pair.photo)
            photo_labels.append(label)
            if pair.cloud is None:
                continue
            view = pair.view
            if view is None and camera_matrix is not None and extrinsic is not None:
                view = self.photo_match_service._project_world_point(
                    pair.cloud, camera_matrix, extrinsic
                )
            if view is not None:
                cloud_points.append(view)
                cloud_labels.append(label)
                projection_points.append((label, view))
        return {
            'photo_points': photo_points,
            'photo_labels': photo_labels,
            'cloud_points': cloud_points,
            'cloud_labels': cloud_labels,
            'projection_points': projection_points,
        }

    def _refresh_manual_match_markers(self):
        if not getattr(self, '_show_match_markers', True):
            self.photo_view.set_markers([])
            self.photo_view.set_remap_markers([])
            self.render_service.clear_pick_markers()
            if hasattr(self, 'cloud_match_view'):
                self.cloud_match_view.set_markers([])
            return
        state = self.photo_match_service.state
        pairs = [
            pair
            for pair in state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        if not state.remapped_photo_points:
            self._reprojection_display_indices = None
        indices = self._reprojection_display_indices
        if indices is None:
            overlay = self._match_overlay_points()
            self.photo_view.set_markers(
                overlay['photo_points'], overlay['photo_labels']
            )
            self.photo_view.set_remap_markers([])
            self.cloud_match_view.set_markers(
                overlay['cloud_points'], overlay['cloud_labels']
            )
            self._projection_marker_points = overlay['projection_points']
            self.viewport.update_pick_markers(
                src_points=self.photo_match_service.cloud_points()
            )
            self._update_reprojection_links()
            return

        indices = [index for index in indices if index < len(pairs)]
        selected = [pairs[index] for index in indices]
        labels = [index + 1 for index in indices]
        cloud_points = [pair.cloud for pair in selected]
        remapped = [
            state.remapped_photo_points[index]
            if index < len(state.remapped_photo_points)
            else None
            for index in indices
        ]
        overlay = self._match_overlay_points(selected, labels)
        self._projection_marker_points = overlay['projection_points']
        self.photo_view.set_markers(overlay['photo_points'], overlay['photo_labels'])
        self.photo_view.set_remap_markers(remapped, labels)
        self.cloud_match_view.set_markers(
            overlay['cloud_points'], overlay['cloud_labels']
        )
        self.viewport.update_pick_markers(
            src_points=cloud_points,
            labels=labels,
        )
        self._update_reprojection_links()

    def _refresh_photo_match_controls(self):
        state = self.photo_match_service.state
        complete = self.photo_match_service.complete_pair_count()
        has_photo = bool(state.photo_path)
        has_cloud = bool(self._active_cloud_name())

        self.btn_rectify_photo.setEnabled(has_photo and not state.annotating)
        self.btn_grayscale_photo.setEnabled(has_photo and not state.annotating)
        can_map_cloud = (
            bool(state.scan_pose_path) and has_cloud and not state.annotating
        )
        self.btn_apply_scan_pose.setEnabled(can_map_cloud)
        if hasattr(self, 'btn_map_cloud'):
            self.btn_map_cloud.setEnabled(
                can_map_cloud and not self._match_denoise_busy
            )
        if hasattr(self, 'btn_remove_match_outliers'):
            self.btn_remove_match_outliers.setEnabled(
                has_cloud
                and not state.annotating
                and not self._match_denoise_busy
            )
        if hasattr(self, 'btn_auto_fit_projection'):
            self.btn_auto_fit_projection.setEnabled(
                bool(state.scan_pose_path) and has_cloud and not state.annotating
            )
        for slider in self._projection_sliders.values():
            slider.setEnabled(
                bool(state.scan_pose_path) and has_cloud and not state.annotating
            )
        self.btn_save_cloud_match_view.setEnabled(
            has_cloud
            and not state.annotating
        )
        if hasattr(self, 'btn_export_cloud_view_images'):
            self.btn_export_cloud_view_images.setEnabled(
                has_cloud
                and self._projection_enabled
                and self._live_projection_view is not None
                and self._view_export_worker is None
                and not state.annotating
            )
        self.btn_auto_match_view.setEnabled(
            has_photo
            and has_cloud
            and self._projection_enabled
            and self._live_projection_view is not None
            and not state.annotating
        )
        self.btn_start_manual_match.setEnabled(
            has_photo and has_cloud and not state.annotating
        )
        self.btn_undo_match_point.setEnabled(
            state.annotating and bool(state.correspondences)
        )
        self.btn_exit_manual_match.setEnabled(
            self.photo_match_service.can_exit_annotation()
        )
        self.btn_estimate_match_matrix.setEnabled(
            self.photo_match_service.can_estimate_match_matrix()
        )
        self.btn_remap_match_points.setEnabled(
            state.pose is not None and not state.annotating
        )
        selected_facade = (
            hasattr(self, 'photo_match_facade_list')
            and self.photo_match_facade_list.currentItem() is not None
        )
        has_generated_heatmap = (
            hasattr(self, 'facade_heatmap_grid_view')
            and self.facade_heatmap_grid_view.has_image()
        )
        if hasattr(self, 'btn_generate_facade_heatmap'):
            self.btn_generate_facade_heatmap.setEnabled(
                has_cloud
                and selected_facade
                and not state.annotating
            )
        if hasattr(self, 'btn_map_heatmap_to_photo'):
            self.btn_map_heatmap_to_photo.setEnabled(
                has_photo
                and has_cloud
                and has_generated_heatmap
                and selected_facade
                and state.pose is not None
                and not state.annotating
            )
        if hasattr(self, 'btn_heatmap_map'):
            self.btn_heatmap_map.setEnabled(
                has_photo
                and has_cloud
                and selected_facade
                and state.pose is not None
                and not state.annotating
            )
        if hasattr(self, 'btn_adjust_photo_angle'):
            self.btn_adjust_photo_angle.setEnabled(
                has_photo
                and has_cloud
                and selected_facade
                and state.pose is not None
                and self._projection_enabled
                and self._live_projection_view is not None
                and not state.annotating
            )
        if not has_photo:
            status = '尚未上传照片'
        elif state.annotating:
            next_step = (
                '请点击照片'
                if state.next_is_photo
                else '请点击三维点云'
            )
            status = f'已完成 {complete} 对；{next_step}'
            if complete < 6:
                status += f'（还需 {6 - complete} 对）'
        elif state.pose is not None:
            error = float(state.pose.get('reprojection_mean_px', 0.0))
            mode = '自动' if state.match_mode == 'auto' else '手动'
            status = f'{mode}矩阵已估计；{complete} 对，误差 {error:.2f} px'
        else:
            status = f'已上传照片；当前 {complete} 对匹配点'
        if self._projection_enabled and self._live_projection_view is not None:
            status += '\n二维投影：右侧当前视图'
        if state.scan_pose_path:
            status += f'\n位姿：{Path(state.scan_pose_path).name}'
        self.photo_match_status.setText(status)

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

        def _export_context(display_quality):
            context = display_quality.get('__export_context') or {}
            if context.get('points') is not None and context.get('results_dir'):
                return context
            # Historical reports intentionally do not persist large point arrays.
            # Rebuild the export input from the active processing dataset.
            try:
                dataset = self.facade_service._index_service._get_dataset(cloud)
                # Export the segmented facade domain as the front-facing base.
                # Quality indices are measurement support only and must not
                # define the image extent.
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
                    # Ensure the segmented facade colour is visible in the
                    # exported base, independent of source RGB availability.
                    colors[:] = np.asarray(
                        self.render_service.facade_color_for(facade), dtype=float)
                project_uuid = getattr(self.current_project, 'project_id', None)
                results_dir = (Storage.ensure_project_dirs(project_uuid)['results']
                               if project_uuid else None)
                return {'results_dir': results_dir, 'points': points,
                        'colors': colors}
            except Exception as exc:
                print(f'[PCFD] export_context_failed facade_id={facade_id} '
                      f'error={exc!r}', flush=True)
                return context

        def _show_effect(mode='flatness'):
            try:
                display_quality = dict(quality) if isinstance(quality, dict) else {}
                display_quality['heatmap_mode'] = mode
                self.render_service.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service._index_service)
                context = _export_context(display_quality)
                exported = ResultExportService().export_heatmap(
                    context.get('results_dir'), facade_no,
                    context.get('points'), context.get('colors'), display_quality)
                if exported and exported.get('heatmap'):
                    # Persist only small, portable artifact metadata. Runtime
                    # point arrays stay in __export_context and are not stored.
                    artifact = {key: exported.get(key) for key in
                                ('mode', 'title', 'heatmap', 'overlay', 'legend')}
                    quality_report = facade.get('quality_report')
                    if isinstance(quality_report, dict):
                        artifacts = quality_report.setdefault('heatmap_artifacts', {})
                        artifacts[exported.get('mode', mode)] = artifact
                        self._refresh_report_preview()
                    preview = exported.get('overlay') or exported.get('heatmap')
                    self.statusBar().showMessage(
                        f'热力图已保存：{exported["heatmap"]}', 6000)
                    return preview
                self.statusBar().showMessage('热力图显示成功，但导出失败，请检查日志。', 5000)
                return None
            except Exception as e:
                print(f'[PCFD] ui.show_effect_error facade_id={facade_id} error={e}', flush=True)
                return None

        def _restore():
            try:
                results = getattr(self.project_operation_service,
                                  '_last_facade_results', None)
                self.render_service.restore_highlight(cloud, results or [])
            except Exception as e:
                print(f'[PCFD] ui.restore_error facade_id={facade_id} error={e}', flush=True)

        label = f'立面 {facade_no}'
        project_name = getattr(self.current_project, 'name', '') if self.current_project else ''

        # Ensure quality is a dict
        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_not_dict facade_id={facade_id} type={type(quality)}', flush=True)
            quality = {}

        try:
            # 不使用 exec()：质量计算完成后这里本来就在 GUI 线程中，exec() 会
            # 再启动一个嵌套事件循环。主窗口又包含 Open3D 原生子窗口，在
            # Windows 上拖动该模态窗口时可能形成 Qt/原生窗口消息循环互等，
            # 最终表现为整个进程锁死。open() 保留模态输入限制，但不阻塞主
            # GUI 事件循环，因此窗口拖动、重绘和 Open3D 消息均可正常处理。
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
        self.render_service.select_facade(
            cloud,
            int(f.get('id', 0)),
            facade=f,
        )
        self._focus_view_on_facade(f)
        return

    def _evaluate_facade(self, f):
        cloud = self._active_cloud_name()
        if not cloud:
            return
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            QMessageBox.warning(self, '质量评估', '请先选择项目。')
            return
        # Quality PNG export is written to the active project's results folder.
        # The folder is created here, while actual export remains user-triggered
        # from the result dialog.
        results_dir = Storage.ensure_project_dirs(project_uuid)['results']

        # FIX: Ensure facade dict has stable id and display_no
        facade_copy = dict(f)
        facade_id = int(facade_copy.get('id', 0))
        facade_no = int(facade_copy.get('display_no', facade_id))
        facade_copy['id'] = facade_id
        facade_copy['display_no'] = facade_no

        # A completed historical result is already a valid report. Reopen it
        # directly; never spend CPU or memory recomputing it on every project
        # activation. Revision mismatches are deliberately handled by the
        # normal worker path, which will reject stale geometry safely.
        historical_quality = facade_copy.get('quality_report')
        if facade_copy.get('quality_status') == 'complete' and isinstance(historical_quality, dict):
            self._show_quality_dialog(cloud, facade_copy, historical_quality)
            return

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
        self._refresh_report_preview()
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
            self._refresh_report_preview()
            return

        # Ensure quality is a dict
        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_invalid_type facade_id={facade_no} '
                  f'type={type(quality)}', flush=True)
            QMessageBox.warning(self, '质量评估',
                f'立面 #{facade_no} 质量计算返回异常类型：{type(quality)}')
            self._refresh_report_preview()
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
            self._refresh_report_preview()
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
            self._refresh_report_preview()
            return

        # State 4: Success with valid windows
        print(f'[PCFD] ui.quality_success facade_id={facade_no} '
              f'windows={window_count} valid={valid_count} '
              f'intervals={len(quality.get("intervals", []))}', flush=True)
        # Persist only after the algorithm has produced a valid report. The
        # repository transaction is the commit point for the completed state.
        try:
            project_uuid = getattr(self.current_project, 'project_id', None)
            if not project_uuid:
                raise RuntimeError('当前项目已失效，无法保存质量结果')
            # results_dir belongs to the evaluation request, but the finished
            # callback must not read a local variable from _evaluate_facade.
            # Resolve it here so asynchronous completion has an independent,
            # valid persistence context.
            dataset = self.facade_service._index_service._get_dataset(cloud)
            # Quality reports are self-contained; no results-domain NPZ is
            # needed for replay. Keep export context separate from persistence.
            artifact_path = None
            ResultsRepo.commit_quality_success(
                project_uuid, int(f.get('id', 0)), quality,
                display_no=facade_no,
                facade_data=f,
                dataset_revision=getattr(dataset, 'revision', None),
                quality_artifact_path=artifact_path,
                color=self.render_service.facade_color_for(f, facade_no),
            )
            f['quality_status'] = 'complete'
            f['quality_report'] = quality
            # The worker receives a copy; update the canonical facade list so
            # the heatmap toggle becomes available immediately after success.
            for current in (getattr(self.project_operation_service,
                                    '_last_facade_results', None) or []):
                if int(current.get('id', -1)) == int(f.get('id', -2)):
                    current.update({'quality_status': 'complete',
                                    'quality_report': quality,
                                    'dataset_revision': getattr(dataset, 'revision', None)})
                    break
            self._refresh_heatmap_button_state()
        except Exception as exc:
            print(f'[PCFD] quality.persist_failed facade_id={facade_no} error={exc!r}', flush=True)
            QMessageBox.warning(self, '质量评估', f'算法已完成，但结果保存失败：{exc}')
            self._refresh_report_preview()
            return
        self._refresh_report_preview()
        QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))

    def _export_quality_report(self):
        if self.current_project is None:
            QMessageBox.information(self, '导出报告', '请先创建或选择项目。')
            return
        default_dir = Path(self.current_project.directory_path) / 'reports'
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, '导出质量报告',
            str(default_dir / f'{self.current_project.name}_质量检测报告.pdf'), REPORT_PDF_FILTER)
        if not path:
            return
        try:
            PdfReportRenderer.write_pdf(self._report_html, path)
            ReportRepo.register_pdf(
                getattr(self.current_project, 'project_id', None),
                path,
                '建筑外立面质量检测报告',
            )
            self._set_report_pdf_status(f'已导出：{Path(path).name}', 'success')
            self._current_report_pdf_name = Path(path).name
            self.statusBar().showMessage(f'报告已导出：{path}', 6000)
        except Exception as exc:
            QMessageBox.warning(self, '导出报告', f'报告导出失败：{exc}')

    def _project_db_id(self):
        """Resolve the numeric project id required by the legacy report registry."""
        from db.connection import index_session
        from models import Project
        from sqlalchemy import select
        with index_session() as session:
            row = session.execute(select(Project).where(Project.uuid == self.current_project.project_id)).scalar_one_or_none()
            if row is None:
                raise ValueError('当前项目不存在')
            return row.id

    def _refresh_report_preview(self):
        if not hasattr(self, 'report_preview_browser'):
            return
        facades = getattr(self.project_operation_service, '_last_facade_results', None) or []
        self._report_snapshot = ReportDataService.build(
            self.current_project, facades,
            getattr(self.current_project, 'directory_path', None))
        self._report_html = PdfReportRenderer.html(self._report_snapshot)
        self.report_document_title_label.setText('建筑外立面质量检测报告')
        self.report_preview_browser.setHtml(self._report_html)
        self.report_preview_browser.setVisible(True)
        self.report_webview.setVisible(False)
        self.report_preview_state_stack.setCurrentIndex(1)
        self._set_report_pdf_status('在线报告预览已更新', 'success')

    def _create_report_export_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.report_navigation_stack = QStackedWidget()
        self.report_navigation_stack.setObjectName('reportNavigationStack')

        document_header = QWidget()
        document_header.setObjectName('reportDocumentHeader')
        document_header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        document_header.setFixedHeight(48)
        document_header_layout = QHBoxLayout(document_header)
        document_header_layout.setContentsMargins(16, 0, 16, 0)
        document_header_layout.setSpacing(16)
        self.report_document_title_label = ElidedLabel(REPORT_EMPTY_TITLE)
        self.report_document_title_label.setObjectName('reportDocumentTitleLabel')
        self.report_document_title_label.setProperty('uiRole', 'sectionTitle')
        document_header_layout.addWidget(self.report_document_title_label, 1)

        self.report_pdf_status_label = QLabel('PDF未加载')
        self.report_pdf_status_label.setObjectName('reportPdfStatusLabel')
        self.report_pdf_status_label.setProperty('uiRole', 'supportingText')
        self.report_pdf_status_label.setProperty('statusState', 'neutral')
        document_header_layout.addWidget(self.report_pdf_status_label)
        body_layout.addWidget(document_header)

        report_preview_page = QWidget()
        report_preview_page.setObjectName('reportPreviewPage')
        report_preview_page.setProperty('uiRole', 'contentArea')
        report_preview_page.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        report_preview_layout = QVBoxLayout(report_preview_page)
        report_preview_layout.setContentsMargins(0, 0, 0, 0)
        report_preview_layout.setSpacing(0)

        self.report_preview_state_stack = QStackedWidget()
        self.report_preview_state_stack.setObjectName('reportPreviewStateStack')

        report_empty_state = TechnicalCanvas('document')
        report_empty_state.setObjectName('reportEmptyState')
        self.report_preview_state_stack.addWidget(report_empty_state)

        report_document_page = QWidget()
        report_document_page.setObjectName('reportDocumentPage')
        report_document_layout = QVBoxLayout(report_document_page)
        report_document_layout.setContentsMargins(0, 0, 0, 0)
        report_document_layout.setSpacing(0)
        # QtWebView2Widget 只嵌入系统 WebView2 内容区，不创建浏览器地址栏。
        self.report_preview_browser = QTextBrowser(report_document_page)
        self.report_preview_browser.setOpenExternalLinks(False)
        self.report_preview_browser.setStyleSheet('QTextBrowser { background: #f8fafc; border: 0; }')
        self.report_preview_browser.setHtml('<h2>报告预览</h2><p>创建或选择项目后将自动生成报告预览。</p>')
        report_document_layout.addWidget(self.report_preview_browser, 1)
        self.report_webview = QtWebView2Widget(
            url='about:blank',
            debug=False,
            context_menus=False,
            background_color='#f8fafc',
            parent=report_document_page,
        )
        self.report_webview.setObjectName('reportPdfWebView')
        self.report_webview.bridge.initialization_done.connect(
            self._on_report_webview_initialized
        )
        self.report_webview.bridge.domContentLoaded.connect(
            self._on_report_pdf_loaded
        )
        self.report_webview.setVisible(False)
        self.report_preview_state_stack.addWidget(report_document_page)
        report_preview_layout.addWidget(self.report_preview_state_stack, 1)

        heatmap_page = QWidget()
        heatmap_page.setObjectName('reportHeatmapPage')
        heatmap_page.setProperty('uiRole', 'contentArea')
        heatmap_page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        heatmap_layout = QVBoxLayout(heatmap_page)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)

        # 保留稳定的控件接口，后续算法视口可以直接替换此占位控件。
        heatmap_placeholder = TechnicalCanvas('heatmap')
        heatmap_placeholder.setObjectName('heatmapPlaceholder')
        heatmap_layout.addWidget(heatmap_placeholder, 1)

        self.report_navigation_stack.addWidget(report_preview_page)
        self.report_navigation_stack.addWidget(heatmap_page)
        body_layout.addWidget(self.report_navigation_stack, 1)
        return page

    def _create_report_navigation(self):
        """创建报告预览和热力图之间的页面内导航。"""
        panel = QWidget()
        panel.setObjectName('reportNavigation')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        navigation_layout = QHBoxLayout(panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(8)

        self.report_navigation_group = QButtonGroup(self)
        self.report_navigation_group.setExclusive(True)
        navigation_items = (
            ('报告预览', 'btn_report_preview_navigation'),
            ('热力图', 'btn_heatmap_navigation'),
        )
        for index, (label, object_name) in enumerate(navigation_items):
            button = QPushButton(label)
            button.setObjectName(object_name)
            button.setProperty('uiRole', 'navigationItem')
            button.setProperty('navigationLevel', 'internal')
            button.setCheckable(True)
            button.setChecked(index == self._report_navigation_index)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, target_index=index:
                self._set_report_navigation(target_index)
            )
            self.report_navigation_group.addButton(button, index)
            navigation_layout.addWidget(button)

        return panel

    def _create_placeholder_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        review_canvas = TechnicalCanvas('review')
        review_canvas.setObjectName('reviewTechnicalCanvas')
        body_layout.addWidget(review_canvas, 1)
        return page

    def _create_page_header(self, page_key):
        panel = QWidget()
        panel.setObjectName(f'{page_key}HeaderPanel')
        panel.setProperty('uiRole', 'pageHeader')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setMinimumHeight(56)
        if page_key == 'report_export':
            # 报告页命令较少，使用固定横向布局把内部切换放到右侧空白区。
            panel.setMaximumHeight(56)
            header_layout = QHBoxLayout(panel)
            header_layout.setContentsMargins(6, 6, 14, 6)
            header_layout.setSpacing(8)
        else:
            panel.setMaximumHeight(200)
            panel.installEventFilter(self)
            header_layout = FlowLayout(
                panel,
                margin=6,
                horizontal_spacing=8,
                vertical_spacing=8,
            )
            self.page_header_layouts[panel] = header_layout

        actions_by_name = {
            button_name: label
            for label, button_name in PAGE_HEADER_ACTIONS[page_key]
        }
        groups = PAGE_HEADER_GROUPS[page_key]

        if page_key == 'project_operation':
            # 回放中老师指定：两个侧栏开关常驻在“视图”分组之前。
            sidebar_group = QFrame()
            sidebar_group.setProperty('uiRole', 'commandGroup')
            sidebar_group.setAttribute(
                Qt.WidgetAttribute.WA_StyledBackground,
                True,
            )
            sidebar_group_layout = QHBoxLayout(sidebar_group)
            sidebar_group_layout.setContentsMargins(8, 4, 12, 4)
            sidebar_group_layout.setSpacing(6)
            self.left_sidebar_button = self._create_sidebar_toggle_button(
                'left'
            )
            self.right_sidebar_button = self._create_sidebar_toggle_button(
                'right'
            )
            sidebar_group_layout.addWidget(self.left_sidebar_button)
            sidebar_group_layout.addWidget(self.right_sidebar_button)
            header_layout.addWidget(sidebar_group)

        for group_index, (group_name, button_names) in enumerate(groups):
            group = QFrame()
            group.setProperty('uiRole', 'commandGroup')
            group.setProperty('groupLast', group_index == len(groups) - 1)
            group.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(8, 4, 16, 4)
            group_layout.setSpacing(8)

            if group_name:
                group_label = QLabel(group_name)
                group_label.setProperty('uiRole', 'commandGroupLabel')
                group_layout.addWidget(group_label)

            for button_name in button_names:
                button = QPushButton(actions_by_name[button_name])
                button.setObjectName(button_name)
                button.setProperty('uiRole', 'headerAction')
                if button_name in PRIMARY_HEADER_ACTIONS:
                    button.setProperty('buttonRole', 'primary')
                button.setMinimumSize(100, 36)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                setattr(self, button_name, button)
                self.header_buttons[button_name] = button
                group_layout.addWidget(button)

            header_layout.addWidget(group)

        if page_key == 'report_export':
            # 报告内部切换与“打开 PDF”同处一个命令栏，释放原独立标题行。
            header_layout.addStretch(1)
            header_layout.addWidget(self._create_report_navigation())
        else:
            QTimer.singleShot(0, lambda: self._resize_page_header(panel))
        return panel

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
            self.station_panel = station
            content.addWidget(station, 3)
            controls = PointCloudControls()
            controls.reset_view_requested.connect(self.project_operation_service.reset_view)
            controls.change_color_requested.connect(self.project_operation_service.change_color)
            controls.point_size_changed.connect(self.viewport.set_all_point_size)
            self.pointcloud_controls = controls
            content.addWidget(controls, 1)
            self.photo_match_panel = self._create_photo_match_panel()
            self.photo_match_panel.hide()
            content.addWidget(self.photo_match_panel, 1)
        sidebar.setMinimumWidth(200 if side == 'left' else 260)
        sidebar.setMaximumWidth(280 if side == 'left' else 380)
        sidebar.setProperty('expandedWidth', 230 if side == 'left' else 320)
        return sidebar

    def _create_photo_match_panel(self):
        panel = QWidget()
        panel.setObjectName('photoMatchPanel')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        title = QLabel('2D照片-3D点云匹配')
        title.setProperty('uiRole', 'sectionTitle')
        layout.addWidget(title)

        self.btn_upload_photo = QPushButton('上传 2D 照片')
        # 摆正 / 黑白化功能保留，暂不在面板上显示。
        self.btn_rectify_photo = QPushButton('摆正照片')
        self.btn_grayscale_photo = QPushButton('黑白化照片')
        self.btn_rectify_photo.hide()
        self.btn_grayscale_photo.hide()
        self.btn_upload_scan_pose = QPushButton('上传扫描仪位姿')
        self.btn_map_cloud = QPushButton('点云映射')
        self.btn_map_cloud.setToolTip(
            '按扫描仪位姿调整三维视角，并将点云投影到右侧二维视图'
        )
        # 调整视角 / 投影到二维功能保留，由“点云映射”一并执行。
        self.btn_apply_scan_pose = QPushButton('调整点云视角')
        self.btn_apply_scan_pose.hide()
        # 消除离群点 / 下采样功能保留，进入对齐工作区时自动执行。
        self.btn_remove_match_outliers = QPushButton('点云消除离群点')
        self.btn_remove_match_outliers.hide()
        self.btn_save_cloud_match_view = QPushButton('投影到二维视图')
        self.btn_save_cloud_match_view.hide()
        # 导出当前视图图片功能保留，暂不在面板上显示。
        self.btn_export_cloud_view_images = QPushButton('保存点云当前视图图片')
        self.btn_export_cloud_view_images.hide()
        self.btn_auto_match_view = QPushButton('自动匹配')
        # 手动匹配功能保留，暂不在面板上显示。
        self.btn_start_manual_match = QPushButton('进入标注模式')
        self.btn_undo_match_point = QPushButton('撤销上一个标注点')
        self.btn_exit_manual_match = QPushButton('退出标注')
        self.btn_estimate_match_matrix = QPushButton('粗略估计匹配矩阵')
        self.btn_remap_match_points = QPushButton('重映射 3D 标注点')
        self.btn_start_manual_match.hide()
        self.btn_undo_match_point.hide()
        self.btn_exit_manual_match.hide()
        self.btn_estimate_match_matrix.hide()
        self.btn_remap_match_points.hide()
        self.btn_heatmap_map = QPushButton('热力图映射')
        self.btn_heatmap_map.setToolTip(
            '为右侧选中的立面生成平整度热力图，并映射回照片'
        )
        self.btn_adjust_photo_angle = QPushButton('调整照片角度')
        self.btn_adjust_photo_angle.setToolTip(
            '按选中立面的法向量，将照片和点云映射图调整为正视角度'
        )
        # 生成 / 映射回照片功能保留，由“热力图映射”一并执行。
        self.btn_generate_facade_heatmap = QPushButton('生成热力图')
        self.btn_generate_facade_heatmap.hide()
        self.btn_map_heatmap_to_photo = QPushButton('热力图映射回照片')
        self.btn_map_heatmap_to_photo.hide()

        for button in (
            self.btn_upload_photo,
            self.btn_upload_scan_pose,
            self.btn_map_cloud,
            self.btn_auto_match_view,
            self.btn_start_manual_match,
            self.btn_undo_match_point,
            self.btn_exit_manual_match,
            self.btn_estimate_match_matrix,
            self.btn_remap_match_points,
            self.btn_heatmap_map,
            self.btn_adjust_photo_angle,
        ):
            button.setMinimumHeight(32)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)

        self.photo_match_status = QLabel('尚未上传照片')
        self.photo_match_status.setWordWrap(True)
        self.photo_match_status.setProperty('uiRole', 'supportingText')
        layout.addWidget(self.photo_match_status)
        layout.addStretch(1)

        self.btn_upload_photo.clicked.connect(self._upload_match_photo)
        self.btn_rectify_photo.clicked.connect(self._rectify_match_photo)
        self.btn_grayscale_photo.clicked.connect(self._grayscale_match_photo)
        self.btn_upload_scan_pose.clicked.connect(self._upload_scan_pose)
        self.btn_map_cloud.clicked.connect(self._map_cloud_to_view)
        self.btn_apply_scan_pose.clicked.connect(self._apply_scan_pose_view)
        self.btn_remove_match_outliers.clicked.connect(
            self._remove_match_cloud_outliers
        )
        self.btn_save_cloud_match_view.clicked.connect(
            self._save_cloud_match_view
        )
        self.btn_export_cloud_view_images.clicked.connect(
            self._export_current_cloud_view_images
        )
        self.btn_auto_match_view.clicked.connect(self._auto_match_current_view)
        self.btn_start_manual_match.clicked.connect(
            self._enter_manual_match_mode
        )
        self.btn_undo_match_point.clicked.connect(self._undo_manual_match_point)
        self.btn_exit_manual_match.clicked.connect(
            self._exit_manual_match_mode
        )
        self.btn_estimate_match_matrix.clicked.connect(
            self._estimate_manual_match_matrix
        )
        self.btn_remap_match_points.clicked.connect(
            self._remap_manual_match_points
        )
        self.btn_heatmap_map.clicked.connect(
            self._map_heatmap_to_photo_and_cloud
        )
        self.btn_adjust_photo_angle.clicked.connect(
            self._adjust_photo_to_facade_angle
        )
        self.btn_generate_facade_heatmap.clicked.connect(
            self._generate_selected_facade_heatmap
        )
        self.btn_map_heatmap_to_photo.clicked.connect(
            self._map_heatmap_back_to_photo
        )
        self._refresh_photo_match_controls()
        return panel

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
                self.project_operation_service._last_facade_results = historical or []
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
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(24, 8, 24, 8)
        layout.setSpacing(0)

        navigation_panel = QWidget()
        navigation_panel.setObjectName('bottomNavigation')
        navigation_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        navigation_panel.setMinimumWidth(720)
        navigation_panel.setMaximumWidth(1040)
        navigation_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        navigation_layout = QHBoxLayout(navigation_panel)
        navigation_layout.setContentsMargins(4, 4, 4, 4)
        navigation_layout.setSpacing(4)

        self.page_button_group = QButtonGroup(self)
        self.page_button_group.setExclusive(True)
        self.page_buttons = {}
        for index, (page_title, page_key) in enumerate(PAGE_DEFINITIONS):
            button = QPushButton(page_title)
            button.setObjectName(PAGE_BUTTON_NAMES[page_key])
            button.setProperty('uiRole', 'navigationItem')
            button.setProperty('navigationLevel', 'main')
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
            navigation_layout.addWidget(button, 1)
        layout.addStretch(1)
        layout.addWidget(navigation_panel, 4)
        layout.addStretch(1)

        dock.setWidget(panel)
        dock.setMinimumHeight(68)
        dock.setMaximumHeight(68)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def set_current_page(self, page_index):
        if not 0 <= page_index < len(PAGE_DEFINITIONS):
            return
        page_title, page_key = PAGE_DEFINITIONS[page_index]
        button = self.page_buttons.get(page_key)
        if button is not None and not button.isEnabled():
            return

        self.page_stack.setCurrentIndex(page_index)
        self.application_page_title.setText(page_title)
        if button is not None:
            button.setChecked(True)
        self._update_window_title(page_key)

    def _enter_project_operation(self):
        """点云加载完成后进入项目操作页，三维视口只在这一页。"""
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)

    def _bind_active_project_to_operations(self):
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            return
        current = getattr(self.project_operation_service, '_project_uuid', None)
        if current == project_uuid:
            return
        try:
            self.project_operation_service.set_active_project_uuid(project_uuid)
        except Exception:
            pass

    def _display_imported_station(self, before_ids):
        self.station_service.refresh()
        new_station = next(
            (row for row in reversed(self.station_service.list_stations())
             if row.id not in before_ids),
            None,
        )
        if new_station is not None:
            self.station_service.show_single(new_station)
            self._refresh_station_panel(new_station.id)
        else:
            self._refresh_station_panel()
        self._bind_active_project_to_operations()
        self._enter_project_operation()
        self._notify_project_files_loaded()
        return new_station

    def _notify_project_files_loaded(self):
        loaded = []
        try:
            rows = self.station_service.list_stations()
            active_id = getattr(self.station_service, '_active_station_id', None)
            shown = [row for row in rows if row.id == active_id] or rows[:1]
            for row in shown:
                source = Path(row.source_path)
                loaded.append(str(source))
                dist = source.with_suffix('.dist')
                if dist.exists():
                    loaded.append(str(dist))
        except Exception:
            pass
        print('加载项目文件完成', flush=True)
        for path in loaded:
            print(f'  {path}', flush=True)
        self.statusBar().showMessage('加载项目文件完成', 5000)

    def _set_report_navigation(self, page_index):
        """切换报告页内部内容，不改变底部四个主页面。"""
        if not 0 <= page_index < self.report_navigation_stack.count():
            return

        self._report_navigation_index = page_index
        self.report_navigation_stack.setCurrentIndex(page_index)
        button = self.report_navigation_group.button(page_index)
        if button is not None:
            button.setChecked(True)

        title = (
            '热力图'
            if page_index == 1
            else self._current_report_pdf_name or REPORT_EMPTY_TITLE
        )
        self.report_document_title_label.setText(title)
        current_page_key = PAGE_DEFINITIONS[
            self.page_stack.currentIndex()
        ][1]
        if current_page_key == 'report_export':
            self._update_window_title('report_export')

    def _connect_buttons(self):
        overview_actions = {
            'btn_import_fls_dir': self._open_import_fls_directory,
            'btn_upload_files': self._open_upload_file_dialog,
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_denoise': lambda _checked=False: self._run_default_operation(
                '点云去噪', self.project_operation_service.denoise
            ),
            'btn_registration': (
                lambda _checked=False: self._run_default_operation(
                    '点云配准', self._run_station_registration
                )
            ),
            'btn_select_detection_area': (
                lambda _checked=False: self._run_default_operation(
                    '框选检测区域',
                    self.project_operation_service.select_detection_area,
                )
            ),
            'btn_facade_detection': self._run_facade_detection,
            'btn_quality_inspection': (
                lambda _checked=False: self._run_default_operation(
                    '质量检测',
                    self.project_operation_service.quality_inspection,
                )
            ),
            'btn_box_segmentation': (
                lambda _checked=False: self._run_default_operation(
                    '框选分割',
                    self.project_operation_service.box_segmentation,
                )
            ),
            'btn_calculate_detail': (
                lambda _checked=False: self._run_default_operation(
                    '计算细节',
                    self.project_operation_service.calculate_detail,
                )
            ),
            'btn_align_2d_3d': self._toggle_photo_match_workspace,
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
            lambda: self._toggle_sidebar('left')
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._toggle_sidebar('right')
        )

    def _run_station_registration(self):
        if getattr(self, '_registration_worker', None) is not None:
            QMessageBox.information(self, '点云配准', '已有配准任务正在执行，请稍候。')
            return
        try:
            self.station_service.refresh()
            rows = [x for x in self.station_service.list_stations() if x.is_selected]
            if len(rows) != 2:
                raise ValueError('点云配准需要恰好选择两个 PLY 站点')
            # PLY 已在 FLS 导出阶段应用 transformToGlobal。默认直接执行
            # GPS 全局坐标上的残差 ICP；人工选点仅作为失败后的兜底工具。
            self._registration_rows = tuple(x.id for x in rows)
            self._start_auto_registration(self._registration_rows)
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', str(exc))

    def _start_auto_registration(self, station_ids):
        """Run residual ICP on already-globalized PLY clouds off the GUI thread."""
        generation = self._project_generation
        project_uuid = getattr(self.current_project, 'project_id', None)

        def run_registration():
            payload = self.station_service.register_selected(update_viewport=False)
            payload['_project_generation'] = generation
            payload['_project_uuid'] = project_uuid
            payload['_station_ids_snapshot'] = tuple(station_ids)
            return payload

        worker = RegistrationWorker(run_registration)
        self._registration_worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.statusBar().showMessage('正在执行 GPS 全局坐标残差 ICP，请稍候...')
        self._set_registration_buttons_enabled(False)
        self._registration_pool.start(worker)

    def _enter_manual_registration_fallback(self):
        """Prepare the optional correspondence workflow after auto ICP fails."""
        rows = [x for x in self.station_service.list_stations()
                if x.is_selected]
        if len(rows) != 2:
            return
        try:
            rows, cloud_names = self.station_service.prepare_registration_view(rows)
            source_cloud, target_cloud = cloud_names[1], cloud_names[0]
            self._registration_rows = tuple(row.id for row in rows)
            self._registration_prompted_pairs = 0

            def on_pick(_picked, source_next):
                src, tgt = self.render_service.registration_pick_points()
                pairs = min(len(src), len(tgt))
                self.statusBar().showMessage(
                    f'配准选点：已完成 {pairs} 对，请继续点击'
                    f'{"源站点" if source_next else "目标站点"}同名点')
                if pairs >= 3 and pairs != self._registration_prompted_pairs:
                    self._registration_prompted_pairs = pairs
                    answer = QMessageBox.question(
                        self, '点云配准',
                        f'已选择 {pairs} 对对应点，是否执行人工初值 ICP？',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if answer == QMessageBox.StandardButton.Yes:
                        self._start_manual_registration(src, tgt)

            self.render_service.enter_registration_pick_mode(
                source_cloud, target_cloud, on_pick, pick_radius=10)
            self._registration_proxy_snapshot = (
                np.asarray(self.viewport.get_cloud_data(source_cloud)['pos'],
                           dtype=np.float64).copy(),
                np.asarray(self.viewport.get_cloud_data(target_cloud)['pos'],
                           dtype=np.float64).copy())
            self.statusBar().showMessage(
                '自动配准未达到质量门限。请按源站点、目标站点交替选择至少 3 对同名点')
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', f'进入人工选点失败：{exc}')

    def _start_manual_registration(self, source_points, target_points):
        self.viewport.exit_pick_mode()
        generation = self._project_generation
        project_uuid = getattr(self.current_project, 'project_id', None)
        station_ids = tuple(self._registration_rows)
        proxy_snapshot = tuple(
            np.asarray(points, dtype=np.float64).copy()
            for points in self._registration_proxy_snapshot)

        def run_registration():
            payload = self.station_service.register_selected(
                update_viewport=False,
                manual_points=(np.asarray(source_points), np.asarray(target_points)),
                proxy_clouds=proxy_snapshot)
            payload['_project_generation'] = generation
            payload['_project_uuid'] = project_uuid
            payload['_station_ids_snapshot'] = station_ids
            return payload

        worker = RegistrationWorker(run_registration)
        self._registration_worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.statusBar().showMessage('正在执行人工初值代理域 ICP，请稍候...')
        self._set_registration_buttons_enabled(False)
        self._registration_pool.start(worker)

    def _set_registration_buttons_enabled(self, enabled):
        button = self.header_buttons.get('btn_registration')
        if button is not None:
            button.setEnabled(enabled)

    @Slot(object)
    def _on_registration_finished(self, payload):
        generation = payload.pop('_project_generation', None)
        project_uuid = payload.pop('_project_uuid', None)
        station_snapshot = tuple(payload.pop('_station_ids_snapshot', ()))
        current_uuid = getattr(self.current_project, 'project_id', None)
        if (getattr(self, '_closing', False) or generation != self._project_generation or
                project_uuid != current_uuid or
                station_snapshot != tuple(x.id for x in self.station_service.list_stations()
                                           if x.is_selected)):
            self._discard_registration_result(payload)
            self._registration_worker = None
            self._set_registration_buttons_enabled(True)
            return
        try:
            self.station_service.commit_registration(payload)
            self._refresh_station_panel()
            self.statusBar().showMessage('点云配准完成，已显示注册合并结果。', 5000)
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', f'配准结果提交失败：{exc}')
        finally:
            self._registration_worker = None
            self._set_registration_buttons_enabled(True)

    @Slot(str)
    def _on_registration_failed(self, message):
        self._registration_worker = None
        self._set_registration_buttons_enabled(True)
        if getattr(self, '_registration_rows', None):
            self._enter_manual_registration_fallback()
            self.statusBar().showMessage(f'自动配准失败：{message}', 5000)
            return
        self.statusBar().showMessage('点云配准失败', 5000)
        QMessageBox.warning(self, '点云配准', message)

    @staticmethod
    def _discard_registration_result(payload):
        path = payload.get('result_path') if isinstance(payload, dict) else None
        if not path:
            return
        result = Path(path)
        if result.name.startswith('registration_') and result.exists():
            try:
                result.unlink()
            except OSError:
                pass

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
        # open_project currently includes Open3D rendering and must remain on
        # the GUI thread. Do not send this legacy combined pipeline to a worker.
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
        )
        self._refresh_project_list()
        self._activate_project(project)

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

    def _set_report_pdf_status(self, text, state='neutral'):
        """Update PDF status text and its Corporate Clean semantic color."""
        self.report_pdf_status_label.setText(text)
        self.report_pdf_status_label.setProperty('statusState', state)
        # Dynamic QSS properties need a repolish before their selector updates.
        style = self.report_pdf_status_label.style()
        style.unpolish(self.report_pdf_status_label)
        style.polish(self.report_pdf_status_label)
        self.report_pdf_status_label.update()

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

        self._set_report_pdf_status(
            f'正在加载：{document.name}',
            'loading',
        )
        self.report_preview_state_stack.setCurrentIndex(0)
        self.report_preview_browser.setVisible(False)
        self.report_webview.setVisible(True)
        self._current_report_pdf_name = document.name
        self._set_report_navigation(0)
        self.report_webview.load_url(document.uri)

    def _on_report_webview_initialized(self, success, error_message):
        if success:
            if self._current_report_pdf_name is None:
                self._set_report_pdf_status('PDF未加载')
            return

        self._report_webview_error = error_message
        self.report_preview_state_stack.setCurrentIndex(0)
        if self._current_report_pdf_name is None:
            self._set_report_pdf_status('PDF未加载')
            return

        self._set_report_pdf_status('WebView2 初始化失败', 'error')
        QMessageBox.warning(
            self,
            'WebView2 初始化失败',
            error_message or '请检查 Microsoft Edge WebView2 Runtime。',
        )

    def _on_report_pdf_loaded(self):
        if self._current_report_pdf_name is not None:
            self._set_report_pdf_status('PDF 已加载', 'success')
            self.report_preview_state_stack.setCurrentIndex(1)

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

    def _start_load(self, operation, project_id, *, file_paths=None,
                    directory=None, project=None):
        """Run the combined legacy pipeline on GUI thread.

        FileService currently registers Open3D geometry as part of loading.
        Running it in QRunnable raises Open3D's GUI-thread guard. Keep the
        worker API for the future split pipeline, but never dispatch this
        combined operation to a worker.
        """
        if getattr(self, '_closing', False):
            return
        # This pipeline intentionally commits Open3D only on the GUI thread.
        # Reject re-entry rather than allowing two imports to multiply the
        # resident source/proxy arrays and exhaust Windows commit memory.
        if getattr(self, '_load_in_progress', False):
            QMessageBox.information(self, '点云加载', '已有加载任务正在执行，请稍候。')
            return
        self._load_in_progress = True
        done_message = None
        try:
            self.statusBar().showMessage('正在加载点云，请稍候...')
            if operation == 'activate':
                self._activate_project(project)
                if self.station_service.list_stations():
                    done_message = '加载项目文件完成'
            elif operation == 'upload':
                before_ids = {row.id for row in self.station_service.list_stations()}
                uploaded = self.project_overview_service.upload_files(file_paths, project_id)
                if uploaded:
                    # 只同步站点投影；已有运行时 dataset 保持不变。
                    # 三维视口在项目操作页，加载完成后必须切过去。
                    self._display_imported_station(before_ids)
                    done_message = '加载项目文件完成'
                else:
                    QMessageBox.warning(self, '直接上传文件', '未成功绑定任何点云文件。')
            elif operation == 'fls':
                before_ids = {row.id for row in self.station_service.list_stations()}
                payload = self.project_overview_service.import_fls_directory(directory, project_id)
                if payload.get('success'):
                    # FLS 导入已经持久化并同步资产；显示一个新站点后进入操作页。
                    self._display_imported_station(before_ids)
                    done_message = '加载项目文件完成'
                else:
                    QMessageBox.warning(self, 'FLS 导入', payload.get('message', '导入失败'))
            self._refresh_project_list()
        except Exception as exc:
            self._on_load_failed(self._project_generation, str(exc))
        finally:
            self._load_in_progress = False
            if done_message:
                self.statusBar().showMessage(done_message, 5000)
            else:
                self.statusBar().clearMessage()

    def _on_load_failed(self, generation, error):
        if generation != self._project_generation:
            return
        self._active_load_worker = None
        self.statusBar().showMessage('点云加载失败', 5000)
        QMessageBox.warning(self, '点云加载', error)

    def _on_load_finished(self, generation, operation, project_id, project, result):
        if generation != self._project_generation:
            return
        self._active_load_worker = None
        self.statusBar().clearMessage()
        if operation == 'activate' and project is not None:
            self._refresh_project_list()
            self._activate_project(project)
        elif operation == 'upload':
            uploaded = result.get('uploaded') or []
            if uploaded:
                self._refresh_project_list()
                self._display_imported_station(set())
                self.statusBar().showMessage('加载项目文件完成', 5000)
            else:
                QMessageBox.warning(self, '直接上传文件', '未成功绑定任何点云文件。')
        elif operation == 'fls':
            payload = result.get('result') or {}
            if payload.get('success'):
                self._refresh_project_list()
                self._display_imported_station(set())
                self.statusBar().showMessage('加载项目文件完成', 5000)
            else:
                QMessageBox.warning(self, 'FLS 导入', payload.get('message', '导入失败'))

    def _activate_project(self, project):
        # Historical facade geometry is not a formal result.  Never carry it
        # across projects or restore detection-only cache entries.
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            raise ValueError('项目标识为空，无法恢复项目')
        # All entry points (open directory, project picker, upload and FLS)
        # use the same strict activation transaction.  Do this before changing
        # current_project so a failed restore cannot leave a false active UI.
        self.project_overview_service.activate_project(project_uuid)
        try:
            self.pointcloud_service.set_project(project_uuid)
            self.station_service.set_project(project_uuid)
            self._refresh_station_panel()
            if not self.station_service.list_stations():
                self.render_service.clear_scene_display()
                self._set_current_project(project)
                self._bind_active_project_to_operations()
                self.statusBar().showMessage('项目已打开，但未发现可用 PLY 站点。', 5000)
                return
            self.station_service.restore_view()
        except Exception as exc:
            self.render_service.clear_scene_display()
            raise RuntimeError(f'站点恢复失败：{exc}') from exc
        self._set_current_project(project)
        self._refresh_report_preview()
        try:
            # propagate active project UUID to operation scheduler for DAL persistence
            self.project_operation_service.set_active_project_uuid(project_uuid)
        except Exception:
            pass
        try:
            if project_uuid:
                active_station_id = getattr(self.station_service, '_active_station_id', None)
                historical = self.project_overview_service.load_historical_facades(
                    project_uuid, active_station_id)
                # Restore historical facades through the same state path as a
                # fresh detection. This synchronizes the list, renderer cache,
                # heatmap availability and report snapshot, including the
                # empty-history case (which must clear the previous project).
                self.project_operation_service._last_facade_results = historical or []
                self._show_facade_results(historical or [])
                self._refresh_report_preview()
        except Exception as exc:
            self.statusBar().showMessage(f'项目历史数据恢复部分失败：{exc}', 5000)
        self._enter_project_operation()
        self._notify_project_files_loaded()

    def _prepare_project_activation(self, project_id):
        """Dispose the old session before a restore/import loads new arrays."""
        current_id = getattr(self.current_project, 'project_id', None)
        if current_id == project_id and project_id is not None:
            # Re-importing the current project still needs a clean registry.
            self._dispose_project_runtime()
        else:
            self._dispose_project_runtime()
        self._project_generation += 1

    def _dispose_project_runtime(self):
        """Single GUI-thread disposal gate for project switches and close."""
        self._load_in_progress = False
        try:
            self.project_operation_service.invalidate_async_jobs()
        except Exception:
            pass
        self._active_quality_worker = None
        self._quality_result_cache.clear()
        self._quality_reports.clear()
        try:
            self.project_operation_service.clear_processing_state()
        except Exception:
            pass
        try:
            self.render_service.clear_runtime()
        except Exception:
            pass
        try:
            self.pointcloud_service.close_project()
        except Exception:
            pass
        try:
            self.viewport.clear()
        except Exception:
            pass
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()
        if hasattr(self, 'station_list'):
            self.station_list.blockSignals(True)
            self.station_list.clear()
            self.station_list.blockSignals(False)

    def closeEvent(self, event):
        """严格按顺序销毁，先停 timer，再断信号，再销毁 viewport，最后退出 app。"""
        if getattr(self, '_closing', False):
            event.accept()
            return
        self._closing = True
        # Step 1: 停止 viewport timer，防止 poll_events 在销毁中调用
        timer = getattr(self.viewport, '_timer', None)
        if timer is not None:
            timer.stop()
        # Step 2: 断开 viewport 的渲染队列信号
        if hasattr(self.viewport, '_render_queue'):
            rq = self.viewport._render_queue
            rq.color.disconnect()
            rq.points.disconnect()
        # Step 3: 关闭质量结果对话框
        quality_dialog = getattr(self, '_quality_dialog', None)
        if quality_dialog is not None:
            quality_dialog.close()
            self._quality_dialog = None
        # Step 4: 清理线程池（不再接受新任务）
        for pool_name in ('_quality_pool', '_load_pool', '_registration_pool'):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                try:
                    pool.clear()
                except Exception:
                    pass

        # Step 5: 释放项目运行时资源
        self._dispose_project_runtime()

        # Step 6: 销毁 viewport（关键：在 Qt 窗口销毁前完成）
        try:
            if hasattr(self, 'viewport') and self.viewport is not None:
                self.viewport.destroy()
        except Exception:
            pass

        # Step 7: 等待线程池完成（短超时）
        for pool_name in ('_quality_pool', '_load_pool', '_registration_pool'):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                try:
                    pool.waitForDone(100)
                except Exception:
                    pass

        # Step 8: 调用父类关闭事件
        try:
            super().closeEvent(event)
        except Exception:
            event.accept()

        # Step 9: 确保 Qt 事件循环退出（最关键）
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                    app.quit()
        except Exception:
            pass

    def _set_current_project(self, project):
        self.current_project = project
        has_project = project is not None

        for page_key, button in self.page_buttons.items():
            button.setEnabled(page_key == 'project_overview' or has_project)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(
                f'{project.name}\n{project.directory_path}'
            )
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_workspace()
        self._refresh_report_preview()
        self._update_window_title()

    def _update_window_title(self, page_key=None):
        if page_key is None:
            page_index = self.page_stack.currentIndex()
            page_key = PAGE_DEFINITIONS[page_index][1]

        if page_key == 'report_export':
            suffix = (
                '热力图'
                if self._report_navigation_index == 1
                else self._current_report_pdf_name or ''
            )
        elif self.current_project is not None:
            suffix = self.current_project.name
        else:
            suffix = ''

        title = APPLICATION_TITLE
        if suffix:
            title = f'{title} - {suffix}'
        self.setWindowTitle(title)

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

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._update_maximize_button()
            QTimer.singleShot(0, self._position_resize_handles)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_resize_handles()
        overlay = getattr(self, '_match_link_overlay', None)
        if overlay is not None:
            overlay.setGeometry(self.rect())
            self._update_reprojection_links()

    def eventFilter(self, watched, event):
        if (
            watched in self.page_header_layouts
            and event.type() == QEvent.Type.Resize
        ):
            self._schedule_page_header_resize(watched)
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
        target_height = max(56, min(content_height, 200))

        if (
            panel.minimumHeight() == target_height
            and panel.maximumHeight() == target_height
        ):
            return

        panel.setMinimumHeight(target_height)
        panel.setMaximumHeight(target_height)
