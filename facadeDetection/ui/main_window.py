from dataclasses import replace
from pathlib import Path

import numpy as np

from PySide6.QtCore import QEvent, QSize, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QPixmap

from .main_window_config import PAGE_DEFINITIONS
from .widgets.flow_layout import FlowLayout
from .widgets.window_chrome import (
    ApplicationTitleBar,
    ElidedLabel,
    WindowResizeHandle,
)
from .widgets.technical_canvas import TechnicalCanvas
from .widgets.step_nav_bar import StepNavBar
from .pages.overview_page import OverviewPageMixin
from .pages.operation_page import OperationPageMixin
from .pages.inspection_review_page import InspectionReviewPageMixin
from .pages.report_page import ReportPageMixin
from .controllers.facade_quality import FacadeQualityController
from .controllers.registration import RegistrationController
from .controllers.project_lifecycle import ProjectLifecycleController
from .controllers.model_export import ModelExportController
from .controllers.task_progress import (
    TASK_DENOISE,
    TASK_DETECTION,
    TASK_LOAD,
    TASK_MODEL_EXPORT,
    TASK_QUALITY,
    TASK_QUALITY_BATCH,
    TASK_REGION,
    TASK_REPORT,
    TASK_UPLOAD,
    TaskProgressController,
)
from .dialogs.facade_quality_dialog import FacadeQualityDialog
from services.inspection_review import InspectionReviewService
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.viewport_render_service import ViewportRenderService
from services.viewport_render_facade import ViewportRenderFacade
from runtime.lifecycle import RuntimeLifecycle
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from services.facade.facade_service import FacadeService
from services.report_export import ReportExportService
from services.result_export_service import ResultExportService
from config.storage import Storage
from view3d.open3d_viewport import Open3DViewport


PAGE_BUTTON_NAMES = {
    'project_overview': 'btn_overview',
    'project_operation': 'btn_operation',
    'inspection_review': 'btn_inspection_review',
    'report_export': 'btn_report_export',
}

PAGE_HEADER_ACTIONS = {
    'project_overview': (
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
        # 【打开 PDF】已按业务要求整体重构为【导出模型】：
        # 选择项目站点 → 源点云 voxel=0.2 下采样 → 导出 PLY。
        # 对象名沿用 btn_open_report_pdf，避免影响页面装配顺序与测试断言；
        # 控件文案、工具提示与连接的回调均已替换为导出模型语义。
        ('导出模型', 'btn_open_report_pdf'),
        ('导出质量报告', 'btn_export_quality_report'),
    ),
}

APPLICATION_TITLE = '点云外立面智能检测平台'

# 每个页面只突出一个主要操作，避免顶部十余个按钮全部使用主色。
PRIMARY_HEADER_ACTIONS = {
    'btn_facade_detection',
    'btn_open_report_pdf',
}

# Corporate Clean 强调清晰的功能分组。按钮名称和业务连接保持不变，
# 这里只定义它们在顶部命令栏中的视觉归属。
PAGE_HEADER_GROUPS = {
    'project_overview': (
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


class MainWindow(OverviewPageMixin, OperationPageMixin,
                 InspectionReviewPageMixin, ReportPageMixin,
                 QMainWindow):
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
        self.render_facade = ViewportRenderFacade(self.render_service)
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
        self.project_operation_service.set_station_service(self.station_service)
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        # 左侧"站点/点云显示"面板默认收拢：外立面检测的主视线是三维视口，
        # 左栏只在需要切换站点或调点云显示参数时展开。右栏（检测结果）
        # 是常态可用的复核面板，因此保持展开。
        self._sidebar_collapsed = {'left': True, 'right': False}
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
        # 质量域状态（_quality_reports/_quality_result_cache/_active_quality_worker
        # 等）由 FacadeQualityController 持有，经下方同名 property 委托访问。
        self._report_snapshot = {'project': {}, 'facades': []}
        # 质量结果窗口采用非阻塞打开方式；必须由主窗口持有引用，避免窗口被
        # Python 垃圾回收，同时避免再次进入 QDialog.exec() 的嵌套事件循环。
        self._quality_dialog = None
        self._runtime = RuntimeLifecycle(self, max_thread_count=1)
        self._quality_pool = self._runtime.pool('quality')
        self._registration_pool = self._runtime.pool('registration')
        self._load_pool = self._runtime.pool('load')
        self._load_cancel_button = None
        # 加载/项目代际状态全局唯一的模态进度控制器：所有重要耗时操作共享
        # 同一套"模态弹窗 + 每 5 秒刷新一次进度 + 终态强制收尾"的进度体验。
        self.task_progress = TaskProgressController(self)
        # 【导出模型】编排控制器：站点选择、后台 voxel=0.2 下采样导出 PLY、
        # 进度弹窗与结果提示都收拢在控制器内；报告页只保留一次点击的入口。
        self.model_export_controller = ModelExportController(
            station_service=self.station_service,
            pool=self._load_pool,
            project_provider=lambda: getattr(self, 'current_project', None),
            progress=self.task_progress,
            parent=self,
        )
        self.model_export_controller.status_message.connect(
            lambda message, timeout: self.statusBar().showMessage(
                message, timeout))
        self.model_export_controller.warning_requested.connect(
            lambda message: QMessageBox.warning(self, '导出模型', message))
        self.model_export_controller.info_requested.connect(
            lambda message: QMessageBox.information(self, '导出模型', message))
        self._pending_station_selection = {}
        self._station_selection_timer = QTimer(self)
        self._station_selection_timer.setSingleShot(True)
        self._station_selection_timer.setInterval(160)
        self._station_selection_timer.timeout.connect(self._flush_station_selection)
        # 配准编排移入独立 controller；按钮使能、弹窗、视口选点等 UI 反馈
        # 全部经信号接回本窗口，worker 状态由 controller 自行持有。
        self.registration_controller = RegistrationController(
            self.station_service,
            self.render_service,
            self._registration_pool,
            self._registration_context,
            parent=self,
        )
        self._connect_registration_controller()
        # 质量编排移入独立 controller；对话框、立面列表、热力图按钮等 UI 反馈
        # 全部经信号接回本窗口。参数快照与采样间距依赖 UI 控件读数，
        # 以可调用对象注入，由 controller 在计算发起时实时取数。
        self.facade_quality_controller = FacadeQualityController(
            facade_service=self.facade_service,
            render_facade=self.render_facade,
            render_service=self.render_service,
            pointcloud_service=self.pointcloud_service,
            project_operation_service=self.project_operation_service,
            station_service=self.station_service,
            pool=self._quality_pool,
            context_provider=self._quality_context,
            profile_provider=self._quality_profile_provider,
            grid_size_provider=self._quality_grid_size,
            progress=self.task_progress,
            parent=self,
        )
        self._connect_facade_quality_controller()
        # 项目加载/激活/销毁编排移入独立 controller；项目列表、页签、标题、
        # 状态栏等 UI 反馈全部经信号接回本窗口。销毁链路需要直接复位质量域
        # 状态，故显式注入 facade_quality_controller。
        self.lifecycle_controller = ProjectLifecycleController(
            project_overview_service=self.project_overview_service,
            pointcloud_service=self.pointcloud_service,
            station_service=self.station_service,
            project_operation_service=self.project_operation_service,
            render_service=self.render_service,
            facade_quality_controller=self.facade_quality_controller,
            context_provider=self._lifecycle_context,
            load_pool=self._load_pool,
            parent=self,
        )
        self._connect_lifecycle_controller()
        # "关闭即中止"绑定：加载/导入/激活与模型导出两条链路的 worker
        # 都挂到统一的取消处理器上，关闭进度窗即取消对应后台进程。
        self.task_progress.set_cancel_handler(
            TASK_LOAD, self.lifecycle_controller.cancel_active_load)
        self.task_progress.set_cancel_handler(
            TASK_MODEL_EXPORT, self.model_export_controller.cancel_active_export)
        # 其余任务的"右上角关闭即中止"落点：一次耗时常同时挂着后台 worker、
        # 调度器排队任务等多处，故用 add_cancel_handler 逐个登记，取消时全跑一遍。
        self._register_task_cancel_handlers()
        # service 层的信息弹窗与取色交互上移到本窗口（时机与文案不变）。
        self.project_operation_service.info_requested.connect(
            self._show_operation_info)
        self.project_operation_service.color_pick_requested.connect(
            self._pick_scene_color)
        self._setup_ui()
        self._create_resize_handles()
        self._connect_buttons()
        # 步骤 √ 只有在业务真正完成后才回填，故绑定放在控件与命令连接就绪之后。
        self._bind_step_completion_signals()
        # Hook: 当结果准备就绪时，在右侧停靠栏中显示立面统计数据
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

        brand_mark = QLabel()
        brand_mark.setObjectName('applicationBrandMark')
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(44, 44)
        logo_path = Path(__file__).resolve().parents[1] / 'utils' / 'logo.png'
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            brand_mark.setPixmap(pixmap.scaled(
                36, 36, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            brand_mark.setStyleSheet('background:transparent;border:none;')
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
            elif page_key == 'inspection_review':
                page = self._create_inspection_review_page(page_title, page_key)
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

    def _create_placeholder_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        review_canvas = TechnicalCanvas('review')
        review_canvas.setObjectName('reviewTechnicalCanvas')
        body_layout.addWidget(review_canvas, 1)
        return page

    # ------------------------------------------------------------------
    # 项目操作页：三步业务流程编排
    # 步骤栏只做两件事——把点击转发给既有命令、在业务真正结束后回填 √。
    # 算法、数据模型与原有命令按钮全部保持原状。
    # 改造后：区域选取 + 立面提取合并为【框选检测】
    # ------------------------------------------------------------------
    STEP_DATA = 0          # ① 数据处理
    STEP_BOX_DETECT = 1    # ② 框选检测（合并原区域选取+立面提取）
    STEP_QUALITY = 2       # ③ 质量评估

    #: 步骤序号 -> 承接该业务的既有命令按钮对象名
    STEP_BUTTONS = {
        STEP_DATA: 'btn_denoise',
    }

    STEP_TIPS = (
        '对已加载站点执行点云处理',
        '框选检测区域并自动执行立面提取',
        '对已标记为“处理”的立面执行质量评估',
    )

    def _install_step_nav(self, header_layout):
        """在项目操作页命令栏位置装配顶部四步导航栏。"""
        self.step_nav = StepNavBar()
        for index, tip in enumerate(self.STEP_TIPS):
            self.step_nav.chips[index].set_tooltip(tip)
        self.step_nav.step_clicked.connect(self._on_step_nav_clicked)
        # stretch=1：四步导航栏吞掉侧栏开关组之外的全部宽度，配合
        # StepNavBar 内部四个卡片相同的 stretch，实现"均分铺满整个顶层栏"。
        header_layout.addWidget(self.step_nav, 1)
        for button in (getattr(self, 'left_sidebar_button', None),
                       getattr(self, 'right_sidebar_button', None)):
            if button is not None:
                button.setSizePolicy(QSizePolicy.Policy.Fixed,
                                     QSizePolicy.Policy.Fixed)

    def _on_step_nav_clicked(self, index):
        """步骤点击只触发既有命令按钮，行为与改造前完全一致。"""
        if index == self.STEP_BOX_DETECT:
            # ② 框选检测：进入 ROI 框选 + 确认卡片流程
            self._enter_box_detection_mode()
            return
        if index == self.STEP_QUALITY:
            # ③ 复用原“评估选中立面”（按“处理”标记批量评估）。
            nav = getattr(self, 'step_nav', None)
            if nav is not None:
                nav.reset_to_running(index)
            try:
                self._evaluate_selected_facade()
            except Exception as exc:
                self.statusBar().showMessage(f'质量评估失败：{exc}', 8000)
            return
        button_name = self.STEP_BUTTONS.get(index)
        button = self.header_buttons.get(button_name) if button_name else None
        if button is None:
            return
        # 重做该步：自身进入执行态（清掉上一次的 ✗），下游结果一并失效，
        # 之后再靠业务完成/失败信号回填 √ 或 ✗，绝不"点击即打勾"。
        self.step_nav.reset_to_running(index)
        self._begin_step_task(index)
        button.click()

    #: 步骤序号 -> 该步耗时计算对应的进度任务键
    STEP_TASK_KEYS = {
        STEP_DATA: TASK_DENOISE,
        STEP_BOX_DETECT: TASK_DETECTION,
    }

    STEP_TASK_TITLES = {
        TASK_DENOISE: '点云去噪',
        TASK_DETECTION: '立面提取',
        TASK_QUALITY: '质量评估',
        TASK_QUALITY_BATCH: '质量评估',
        TASK_MODEL_EXPORT: '导出模型',
    }

    def _register_task_cancel_handlers(self):
        """把每个任务的"全部相关进程"中止落点登记到统一进度控制器。

        回调只做置取消标志这类轻量操作（worker.cancel / 调度器 cancel_all），
        绝不在 GUI 线程 join 或阻塞等待，避免关窗时反而卡住界面。
        """
        # 去噪/立面提取/质量评估都跑在项目级调度器与质量 worker 上，
        # 任一任务被中止都要把这两条链路一并停掉，避免"窗口关了线程还在跑"。
        for key in (TASK_DENOISE, TASK_DETECTION, TASK_QUALITY,
                    TASK_QUALITY_BATCH):
            self.task_progress.add_cancel_handler(
                key, self._cancel_project_computation)
        region = getattr(self.project_operation_service,
                         'cancel_region_selection', None)
        if callable(region):
            self.task_progress.add_cancel_handler(TASK_REGION, region)
        report = getattr(self.report_export_service, 'cancel_export', None)
        if callable(report):
            self.task_progress.add_cancel_handler(TASK_REPORT, report)

    def _cancel_project_computation(self):
        """中止当前项目全部后台计算：质量 worker + 调度器排队任务。"""
        worker = getattr(self.facade_quality_controller,
                         'active_quality_worker', None)
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
            self.facade_quality_controller.active_quality_worker = None
        try:
            self.project_operation_service.invalidate_async_jobs()
        except Exception:
            pass

    def _begin_step_task(self, index):
        """为某个耗时步骤立起模态进度窗；无对应任务的步骤（②区域选取）跳过。"""
        task_key = self.STEP_TASK_KEYS.get(index)
        if task_key is not None:
            self._begin_task_progress(task_key)

    def _begin_task_progress(self, task_key):
        """立起模态进度窗：可取消、5 秒节流、文案极简。

        起始为不定量模式（算法未回报百分比前不假装知道进度），一旦后台
        回报真实百分比，update_progress 会自动切回定量显示。
        """
        title = self.STEP_TASK_TITLES.get(task_key, '任务进行中')
        self.task_progress.begin(
            task_key, title, '任务处理中',
            determinate=False, cancellable=True)

    def _on_step_task_finished(self, index, task_key, success):
        """耗时步骤的统一收尾：先给进度窗定终态，再回填步骤徽标。

        顺序很关键——先 finish 让进度条落到 100%（成功）/保留当前值（失败），
        再改徽标，避免用户看到"进度条还停在 60% 但步骤已打勾"的割裂感。
        """
        self.task_progress.finish(task_key, success)
        if success:
            self._mark_step_done(index)
        else:
            self._mark_step_failed(index)

    def _mark_step_done(self, index):
        """仅在对应业务动作真正执行完毕后调用，避免“点击即打勾”。"""
        nav = getattr(self, 'step_nav', None)
        if nav is not None:
            nav.mark_done(index)

    def _mark_step_failed(self, index):
        """业务失败时把步骤标成 ✗；重跑该步会自动清除失败标记。"""
        nav = getattr(self, 'step_nav', None)
        if nav is not None:
            nav.mark_failed(index)
        # 失败即代表该步结果无效，下游步骤必须重新执行。
        if nav is not None:
            nav.reset_after(index)

    def _bind_step_completion_signals(self):
        """把 √ 状态、进度窗收尾与**真实进度信号**挂到服务层。

        改造后每条业务链路都是"进度信号 → 节流写进度窗"、
        "完成/失败信号 → 先收进度窗、再改徽标"。
        """
        service = self.project_operation_service

        # ---- (1) 进度信号接线 ----
        for task_key, names in (
            (TASK_DENOISE,
             ('denoise_progress', 'denoise_progress_changed')),
            (TASK_DETECTION,
             ('detection_progress', 'facade_detection_progress')),
            (TASK_REGION,
             ('region_progress', 'selection_progress')),
        ):
            for name in names:
                signal = getattr(service, name, None)
                if signal is None:
                    continue
                # 默认参数 _k=task_key 规避 Python 闭包晚绑定
                signal.connect(
                    lambda percent, text='', _k=task_key:
                    self.task_progress.report(
                        _k, percent, text or '任务处理中'))
                break

        # ---- (2) 去噪 / 框选检测（立面提取）：完成 → 立改徽标 √ ----
        for signal_name, index, task_key in (
            ('denoise_finished', self.STEP_DATA, TASK_DENOISE),
            ('detection_finished', self.STEP_BOX_DETECT, TASK_DETECTION),
        ):
            signal = getattr(service, signal_name, None)
            if signal is not None:
                signal.connect(
                    lambda _payload=None, _i=index, _k=task_key:
                    self._on_step_task_finished(_i, _k, True))

        # ---- (3) 去噪 / 框选检测（立面提取）：失败 → 保留当前进度 + 徽标 ✗ ----
        for signal_name, index, task_key in (
            ('denoise_failed', self.STEP_DATA, TASK_DENOISE),
            ('detection_failed', self.STEP_BOX_DETECT, TASK_DETECTION),
        ):
            signal = getattr(service, signal_name, None)
            if signal is not None:
                signal.connect(
                    lambda _payload=None, _i=index, _k=task_key:
                    self._on_step_task_finished(_i, _k, False))

        # ---- (4) 区域选取（已并入框选检测）：完成 / 失败 ----
        for signal_name, index, task_key, ok in (
            ('region_finished', self.STEP_BOX_DETECT, TASK_REGION, True),
            ('region_failed', self.STEP_BOX_DETECT, TASK_REGION, False),
        ):
            signal = getattr(service, signal_name, None)
            if signal is not None:
                signal.connect(
                    lambda _payload=None, _i=index, _k=task_key, _ok=ok:
                    self._on_step_task_finished(_i, _k, _ok))

        # ---- (5) 质量评估：批量 + 单个 + 失败 ----
        controller = self.facade_quality_controller
        batch_finished = getattr(controller, 'batch_finished', None)
        if batch_finished is not None:
            batch_finished.connect(
                lambda _payload=None: self._mark_step_done(self.STEP_QUALITY))
        quality_completed = getattr(controller, 'quality_completed', None)
        if quality_completed is not None:
            quality_completed.connect(
                lambda *_args: self._mark_step_done(self.STEP_QUALITY))
        quality_failed = getattr(controller, 'quality_failed', None)
        if quality_failed is not None:
            quality_failed.connect(
                lambda *_args: self._mark_step_failed(self.STEP_QUALITY))

    def _on_quality_batch_progress(self, current, total):
        """把批量评估的 (已完成, 总数) 换算成真实百分比喂给进度窗。"""
        total = max(int(total or 0), 1)
        current = max(0, min(int(current or 0), total))
        percent = int(round(100.0 * current / total))
        self.task_progress.report(TASK_QUALITY_BATCH, percent, '任务处理中')

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
        elif page_key == 'project_operation':
            # 项目操作页改为等宽横向布局：四步导航栏必须"均分铺满整个顶层栏"，
            panel.setMinimumHeight(64)
            panel.setMaximumHeight(64)
            header_layout = QHBoxLayout(panel)
            header_layout.setContentsMargins(12, 6, 12, 6)
            header_layout.setSpacing(10)
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
            if page_key == 'project_operation':
                # 八个原子命令已由顶部四步流程承载；分组壳仍创建（按钮续存并
                # 保持信号连接），只是不再占位，避免命令栏出现整片空白。
                group.setVisible(False)
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
                if page_key == 'project_operation':
                    # 项目操作页的八个原子命令已由顶部四步业务流程取代。
                    # 按钮仍照常创建并连信号（_connect_buttons 依赖对象名属性），
                    # 只是不再显示，作为隐藏的命令载体延长其生命周期。
                    button.setVisible(False)
                setattr(self, button_name, button)
                self.header_buttons[button_name] = button
                group_layout.addWidget(button)

            header_layout.addWidget(group)

        if page_key == 'project_operation':
            # 顶部导航栏：数据处理 → 区域选取 → 立面提取 → 质量评估
            self._install_step_nav(header_layout)

        if page_key == 'report_export':
            # 报告内部切换与“打开 PDF”同处一个命令栏，释放原独立标题行。
            header_layout.addStretch(1)
            header_layout.addWidget(self._create_report_navigation())
        else:
            QTimer.singleShot(0, lambda: self._resize_page_header(panel))
        return panel

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

        # ===== 视口迁移：检测复核页 ↔ 项目操作页 =====
        if page_key == 'inspection_review':
            self._migrate_workspace_to_review()
        elif page_key == 'project_operation':
            self._migrate_workspace_to_operation()

        self.page_stack.setCurrentIndex(page_index)
        self.application_page_title.setText(page_title)
        if button is not None:
            button.setChecked(True)
        # 项目操作页和检测复核页都需要渲染
        viewport = getattr(self, 'viewport', None)
        if viewport is not None and hasattr(viewport, 'set_render_enabled'):
            viewport.set_render_enabled(page_key in ('project_operation', 'inspection_review'))
        self._update_window_title(page_key)

    def _connect_buttons(self):
        overview_actions = {
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_denoise': self.project_operation_service.denoise,
            'btn_registration': self.registration_controller.run_station_registration,
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
            lambda: self._toggle_sidebar('left')
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._toggle_sidebar('right')
        )

    def _registration_context(self):
        # 供 RegistrationController 实时判断配准结果是否已过期。
        return (
            self._project_generation,
            getattr(self.current_project, 'project_id', None),
            getattr(self, '_closing', False),
        )

    def _connect_registration_controller(self):
        controller = self.registration_controller
        controller.busy_changed.connect(self._set_registration_buttons_enabled)
        controller.status_message.connect(self._show_registration_status)
        controller.warning_requested.connect(self._show_registration_warning)
        controller.info_requested.connect(self._show_registration_info)
        controller.station_panel_refresh_requested.connect(
            self._refresh_station_panel)
        controller.manual_pick_requested.connect(
            self._enter_registration_pick_mode)
        controller.manual_icp_prompt_requested.connect(
            self._prompt_manual_registration_icp)

    def _set_registration_buttons_enabled(self, enabled):
        button = self.header_buttons.get('btn_registration')
        if button is not None:
            button.setEnabled(enabled)

    def _show_registration_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def _show_registration_warning(self, message):
        QMessageBox.warning(self, '点云配准', message)

    def _show_registration_info(self, message):
        QMessageBox.information(self, '点云配准', message)

    def _enter_registration_pick_mode(self, source_cloud, target_cloud):
        self.render_service.enter_registration_pick_mode(
            source_cloud, target_cloud, self._on_registration_pick,
            pick_radius=10)

    def _on_registration_pick(self, _picked, source_next):
        self.registration_controller.handle_pick_updated(source_next)

    def _prompt_manual_registration_icp(self, pairs):
        answer = QMessageBox.question(
            self, '点云配准',
            f'已选择 {pairs} 对对应点，是否执行人工初值 ICP？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.registration_controller.start_manual_registration()

    # 质量域状态由 FacadeQualityController 持有。以下 property 保持项目生命周期
    # 代码（project_lifecycle）对同名属性的读写语义不变。
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

    # 项目代际计数由 ProjectLifecycleController 持有。注册/质量 controller 的
    # context_provider 经此 property 读取，语义与原实例属性一致。
    @property
    def _project_generation(self):
        return self.lifecycle_controller.project_generation

    @_project_generation.setter
    def _project_generation(self, value):
        self.lifecycle_controller.project_generation = value

    def _lifecycle_context(self):
        # 供 ProjectLifecycleController 实时读取关闭标志与当前项目标识。
        return (
            getattr(self, '_closing', False),
            getattr(self.current_project, 'project_id', None),
        )

    def _connect_lifecycle_controller(self):
        controller = self.lifecycle_controller
        controller.info_requested.connect(self._show_lifecycle_info)
        controller.warning_requested.connect(self._show_lifecycle_warning)
        controller.status_message.connect(self._show_lifecycle_status)
        controller.status_cleared.connect(self._clear_lifecycle_status)
        controller.load_started.connect(self._show_loading_dialog)
        controller.load_progress.connect(self._update_loading_dialog)
        controller.load_finished.connect(self._hide_loading_dialog)
        controller.station_panel_refresh_requested.connect(
            self._refresh_station_panel)
        controller.project_list_refresh_requested.connect(
            self._refresh_project_list)
        controller.facade_list_reset_requested.connect(self._reset_facade_list)
        controller.station_list_reset_requested.connect(self._reset_station_list)
        controller.current_project_change_requested.connect(
            self._set_current_project)
        controller.report_preview_refresh_requested.connect(
            self._refresh_report_preview)
        controller.facade_results_refresh_requested.connect(
            self._show_facade_results)
        controller.page_change_requested.connect(self.set_current_page)

    def _show_lifecycle_info(self, title, message):
        QMessageBox.information(self, title, message)

    def _show_lifecycle_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def _show_lifecycle_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def _clear_lifecycle_status(self):
        self.statusBar().clearMessage()

    # 点云加载/导入/激活：统一走 TaskProgressController 的 TASK_LOAD 任务
    def _show_loading_dialog(self):
        self.task_progress.begin(
            TASK_LOAD, '点云加载', '任务处理中',
            determinate=False, cancellable=True)
        # 兼容旧属性：历史代码/测试通过 _loading_dialog 探测控件。
        self._loading_dialog = self.task_progress.dialog(TASK_LOAD)

    def _update_loading_dialog(self, percent, text):
        # 真实进度直写节流器：后台回报多少就是多少，终态时强制刷到 100%。
        self.task_progress.report(TASK_LOAD, percent, '任务处理中')

    def _hide_loading_dialog(self, success=True, message='处理完成'):
        # load_finished 现在携带终态：成功刷到 100% 并短暂停留后关闭
        self.task_progress.finish(TASK_LOAD, bool(success), message)


    def _reset_facade_list(self):
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()

    def _reset_station_list(self):
        if hasattr(self, 'station_list'):
            self.station_list.blockSignals(True)
            self.station_list.clear()
            self.station_list.blockSignals(False)

    def _start_load(self, operation, project_id, *, file_paths=None,
                    directory=None, directories=None, project=None):
        self.lifecycle_controller.start_load(
            operation, project_id,
            file_paths=file_paths, directory=directory, directories=directories,
            project=project)

    def _prepare_project_activation(self, project_id):
        self.lifecycle_controller.prepare_project_activation(project_id)

    def _activate_project(self, project):
        self.lifecycle_controller.activate_project(project)

    def _show_operation_info(self, title, message):
        QMessageBox.information(self, title, message)

    def _pick_scene_color(self):
        # 从 service 上移的取色弹窗；取消或选色无效时不做任何事。
        dlg = QColorDialog()
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        if not dlg.exec():
            return
        qcol = dlg.selectedColor()
        if not qcol.isValid():
            return
        color = (qcol.redF(), qcol.greenF(), qcol.blueF())
        self.project_operation_service.apply_global_color(color)

    def _set_current_project(self, project):
        self.current_project = project
        has_project = project is not None

        for page_key, button in self.page_buttons.items():
            button.setEnabled(page_key == 'project_overview' or has_project)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(
                f'{project.name}\n{project.directory_path}')
            self._apply_project_inspection_params(project)
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_workspace()
        self._refresh_report_preview()
        self._update_window_title()

    def _apply_project_inspection_params(self, project):
        """当项目被激活时，将创建项目时录入的检测参数恢复到 UI 控件。"""
        import json
        from services.inspection_profile import InspectionProfileService

        raw = getattr(project, 'inspection_params_json', None)
        if not raw:
            return
        try:
            p = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return

        # 标准
        std_id = p.get('standard_id')
        if std_id:
            idx = self.standard_combo.findData(std_id)
            if idx >= 0:
                self.standard_combo.blockSignals(True)
                self.standard_combo.setCurrentIndex(idx)
                self.standard_combo.blockSignals(False)
                self._on_standard_changed(idx)

        # 区间
        interval = p.get('interval_size_m')
        if interval is not None:
            idx = self.interval_combo.findData(float(interval))
            if idx >= 0:
                self.interval_combo.blockSignals(True)
                self.interval_combo.setCurrentIndex(idx)
                self.interval_combo.blockSignals(False)

        # 质量参数控件
        def _set_spin(spin, value):
            if value is not None and hasattr(spin, 'setValue'):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

        _set_spin(self.quality_length_spin, p.get('ruler_length_m'))
        # 操作页“滑移步距”对应创建弹窗的纵向采样步长；横向步长
        # 对应 InspectionProfile.step_size_m，不应覆盖滑移步距。
        _set_spin(self.quality_step_spin, p.get('step_longitudinal_m'))
        _set_spin(self.quality_width_spin, p.get('ruler_width_m'))
        _set_spin(self.quality_select_band_spin, p.get('select_band_m'))
        _set_spin(self.quality_hole_band_spin, p.get('hole_band_m'))
        _set_spin(self.quality_bin_size_spin, p.get('bin_size_m'))
        _set_spin(self.quality_top_q_spin, p.get('top_q'))
        _set_spin(self.quality_sor_sigma_spin, p.get('sor_sigma'))
        _set_spin(self.quality_sor_k_spin, p.get('sor_k'))
        _set_spin(self.quality_sor_w_weight_spin, p.get('sor_w_weight'))
        _set_spin(self.quality_max_hole_ratio_spin, p.get('max_hole_ratio'))
        _set_spin(self.quality_min_points_spin, p.get('min_points'))

        # SOR 启用
        sor_enabled = p.get('sor_enabled')
        if sor_enabled is not None and hasattr(self, 'quality_sor_check'):
            self.quality_sor_check.blockSignals(True)
            self.quality_sor_check.setChecked(bool(sor_enabled))
            self.quality_sor_check.blockSignals(False)

        # SOR 方法
        sor_method = p.get('sor_method')
        if sor_method and hasattr(self, 'quality_sor_method_combo'):
            idx = self.quality_sor_method_combo.findData(str(sor_method))
            if idx >= 0:
                self.quality_sor_method_combo.blockSignals(True)
                self.quality_sor_method_combo.setCurrentIndex(idx)
                self.quality_sor_method_combo.blockSignals(False)

        # 同步 _inspection_profile 到项目参数快照
        profile = InspectionProfileService.get(std_id) if std_id else None
        if profile is not None:
            aliases = {
                'ruler_length_m': 'measure_height_m',
                'step_longitudinal_m': 'scan_step_m',
                'step_transverse_m': 'step_size_m',
            }
            snapshot = {}
            for source, target in aliases.items():
                if p.get(source) is not None:
                    snapshot[target] = p[source]
            for key in (
                    'interval_size_m', 'ruler_width_m', 'select_band_m',
                    'hole_band_m', 'flatness_limit_mm',
                    'verticality_limit_mm', 'sor_enabled', 'sor_sigma',
                    'sor_k', 'sor_method', 'sor_w_weight'):
                if p.get(key) is not None and hasattr(profile, key):
                    snapshot[key] = p[key]
            self._inspection_profile = replace(profile, **snapshot)

    def closeEvent(self, event):
        # TODO(生命周期/稳定性): closeEvent：核查关闭期间线程池短超时、原生窗口销毁和 Qt 退出顺序，避免后台任务继续访问已销毁视口导致未响应。
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
        # Step 4: 由 runtime 统一清理线程池（不再接受新任务）
        runtime = getattr(self, '_runtime', None)
        if runtime is not None:
            runtime.clear()

        # Step 5: 释放项目运行时资源
        self.lifecycle_controller.dispose_project_runtime()

        # Step 6: 销毁 viewport（关键：在 Qt 窗口销毁前完成）
        try:
            if hasattr(self, 'viewport') and self.viewport is not None:
                self.viewport.destroy()
        except Exception:
            pass

        # Step 7: 等待 runtime 线程池完成（短超时）
        if runtime is not None:
            runtime.stop(100)

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

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._update_maximize_button()
            QTimer.singleShot(0, self._position_resize_handles)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_resize_handles()

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

        # 页面可能在构造期注册过 header panel，但重建页面/切换项目后
        # 残留在 _header_resize_pending 里的旧 widget 已被销毁，
        # 此时查表会 KeyError。用 get() 兜底即可安全跳过。
        header_layout = self.page_header_layouts.get(panel)
        if header_layout is None:
            self._header_resize_pending.discard(panel)
            return
        content_height = header_layout.heightForWidth(panel.width())
        target_height = max(56, min(content_height, 200))

        if (
            panel.minimumHeight() == target_height
            and panel.maximumHeight() == target_height
        ):
            return

        panel.setMinimumHeight(target_height)
        panel.setMaximumHeight(target_height)

    # ------------------------------------------------------------------
    # 视口迁移：检测复核页 ↔ 项目操作页
    # ------------------------------------------------------------------
    def _migrate_workspace_to_review(self):
        """将 viewport_panel + right_dock 迁到检测复核页。"""
        if getattr(self, '_workspace_in_review', False):
            return
        self._operation_splitter_sizes = self.operation_splitter.sizes()
        self.viewport_panel.setParent(None)
        self.right_dock.setParent(None)
        self.review_splitter.replaceWidget(0, self.viewport_panel)
        self.review_splitter.replaceWidget(1, self.right_dock)
        self.review_splitter.setSizes([1040, 300])
        self._show_operation_placeholder()
        results = self.project_operation_service.last_facade_results or []
        self._show_facade_results_review(results)
        self._workspace_in_review = True

    def _migrate_workspace_to_operation(self):
        """将 viewport_panel + right_dock 迁回项目操作页。"""
        if not getattr(self, '_workspace_in_review', False):
            return
        self.viewport_panel.setParent(None)
        self.right_dock.setParent(None)
        self.operation_splitter.insertWidget(1, self.viewport_panel)
        self.operation_splitter.insertWidget(2, self.right_dock)
        self.operation_splitter.setSizes(self._operation_splitter_sizes)
        self._hide_operation_placeholder()
        results = self.project_operation_service.last_facade_results or []
        self._show_facade_results(results)
        self._workspace_in_review = False

    def _show_operation_placeholder(self):
        """视口迁走后，操作页工作区显示引导提示。"""
        if not hasattr(self, '_operation_placeholder'):
            ph = QWidget()
            ph.setObjectName('operationPlaceholder')
            lay = QVBoxLayout(ph)
            lay.setAlignment(Qt.AlignCenter)
            lbl = QLabel('🔧 3D检测工作台已移至【检测复核】页面')
            lbl.setStyleSheet('font-size:16px; color:#64748B; font-weight:600;')
            lay.addWidget(lbl)
            btn = QPushButton('前往检测复核页')
            btn.setProperty('buttonRole', 'primary')
            btn.clicked.connect(lambda: self.set_current_page(2))
            lay.addWidget(btn, alignment=Qt.AlignCenter)
            self._operation_placeholder = ph
        self.operation_splitter.insertWidget(1, self._operation_placeholder)
        self.operation_splitter.setSizes([220, 1040, 0])

    def _hide_operation_placeholder(self):
        if hasattr(self, '_operation_placeholder'):
            self._operation_placeholder.setParent(None)

    # ------------------------------------------------------------------
    # 立面列表双版本
    # ------------------------------------------------------------------
    def _show_facade_results_review(self, results: list[dict]):
        """检测复核页：只显示 complete 立面，隐藏点数，【图片匹配】按钮。"""
        results = self.facade_quality_controller.process_facade_results(results)
        complete_results = [
            f for f in results
            if self._facade_review_status(f) == 'complete'
        ]
        count = len(complete_results)
        self.lbl_facade_summary.setText(f'已检测复核立面：{count}' if count else '未检测')
        if not complete_results:
            self.list_facades.clear()
            return
        self.list_facades.clear()
        for index, f in enumerate(complete_results, 1):
            display_no = int(f.get('display_no') or index)
            f['display_no'] = display_no
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setSizeHint(QSize(0, 40))
            self.list_facades.addItem(item)
            row = self._create_facade_list_row(f, display_no, review_mode=True)
            self.list_facades.setItemWidget(item, row)

    def _create_facade_list_row(self, f: dict, display_no: int, review_mode: bool):
        """创建立面列表行 widget。review_mode=True 时为检测复核模式。"""
        row = QWidget()
        row.setStyleSheet("""
            QWidget { background: transparent; }
            QLabel { font-size: 12px; color: #334155; }
        """)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(8)
        if review_mode:
            info = QLabel(f"立面{display_no}")
        else:
            info = QLabel(f"立面{display_no}")
        info.setStyleSheet('font-size: 12px; color: #334155;')
        row_layout.addWidget(info, 1)
        color = self.render_facade.facade_color(f, display_no)
        swatch = QFrame()
        swatch.setFixedSize(18, 18)
        swatch.setStyleSheet(
            'background-color: rgb(%d,%d,%d); border: 1px solid #94a3b8; border-radius: 3px;' %
            tuple(int(max(0, min(1, x)) * 255) for x in color)
        )
        row_layout.addWidget(swatch)
        if review_mode:
            action_button = QPushButton('图片匹配')
            action_button.setFixedWidth(84)
            action_button.setMinimumHeight(32)
            action_button.clicked.connect(
                lambda _=False, obj=f: self._on_facade_image_match(obj))
        else:
            status = self._facade_review_status(f)
            action_button = QPushButton('处理' if status == 'complete' else '标记处理')
            action_button.setFixedWidth(84)
            action_button.setMinimumHeight(32)
            action_button.setToolTip('点击确认立面状态；仅标记立面允许质量计算')
            action_button.clicked.connect(
                lambda _=False, obj=f, button=action_button:
                self._toggle_facade_review_status(obj, button))
        action_button.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 2px 8px; border-radius: 4px;
                border: 1px solid #cbd5e1; background: #ffffff; color: #475569;
            }
            QPushButton:hover {
                background: #f1f5f9; border-color: #94a3b8;
            }
        """)
        row_layout.addWidget(action_button)
        row.setMaximumHeight(48)
        return row

    def _on_facade_image_match(self, facade: dict):
        """【图片匹配】按钮回调：调用预留桩接口。"""
        from services.two_d_matching_service import TwoDMatchingService
        TwoDMatchingService.match(facade)