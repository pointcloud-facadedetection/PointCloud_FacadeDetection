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

from .main_window_config import PAGE_DEFINITIONS
from .widgets.flow_layout import FlowLayout
from .widgets.window_chrome import (
    ApplicationTitleBar,
    ElidedLabel,
    WindowResizeHandle,
)
from .widgets.technical_canvas import TechnicalCanvas
from .pages.overview_page import OverviewPageMixin
from .pages.operation_page import OperationPageMixin
from .pages.report_page import ReportPageMixin
from .controllers.facade_quality import FacadeQualityController
from .controllers.registration import RegistrationController
from .controllers.project_lifecycle import ProjectLifecycleController
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


class MainWindow(OverviewPageMixin, OperationPageMixin, ReportPageMixin,
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
        # 加载/项目代际状态（_active_load_worker、_load_in_progress、
        # _project_generation）由 ProjectLifecycleController 持有，
        # _project_generation 经下方同名 property 委托访问。
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
        # service 层的信息弹窗与取色交互上移到本窗口（时机与文案不变）。
        self.project_operation_service.info_requested.connect(
            self._show_operation_info)
        self.project_operation_service.color_pick_requested.connect(
            self._pick_scene_color)
        self._setup_ui()
        self._create_resize_handles()
        self._connect_buttons()
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
        # 仅项目操作页持有三维视口，其余页面暂停其 GLFW 轮询与帧提交
        viewport = getattr(self, 'viewport', None)
        if viewport is not None and hasattr(viewport, 'set_render_enabled'):
            viewport.set_render_enabled(page_key == 'project_operation')
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
        return self._quality_profile_snapshot(
            getattr(self, '_inspection_profile', None))

    def _quality_grid_size(self):
        return float(self.interval_combo.currentData())

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

            color = self.render_facade.facade_color(f, display_no)
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
            current = next((f for f in (self.project_operation_service.last_facade_results or [])
                            if int(f.get('id', -1)) == int(facade.get('id', -2))), facade)
            self.facade_quality_controller.evaluate_facade(current)

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
                self.render_facade.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service.index_service)
                context = _export_context(display_quality)
                exported = ResultExportService().export_heatmap(
                    context.get('results_dir'), facade_no,
                    context.get('points'), context.get('colors'), display_quality)
                if exported and exported.get('heatmap'):
                    # 仅持久化小型、可移植的工件元数据。
                    # 运行时点数组保存在 __export_context 中，不会被存储。
                    artifact = {key: exported.get(key) for key in
                                ('mode', 'title', 'heatmap', 'overlay', 'report', 'legend')}
                    quality_report = facade.get('quality_report')
                    if isinstance(quality_report, dict):
                        artifacts = quality_report.setdefault('heatmap_artifacts', {})
                        artifacts[exported.get('mode', mode)] = artifact
                        self._refresh_report_preview()
                    self.statusBar().showMessage(
                        f'热力图已保存：{exported["heatmap"]}', 6000)
                else:
                    self.statusBar().showMessage('热力图显示成功，但导出失败，请检查日志。', 5000)
            except Exception as e:
                print(f'[PCFD] ui.show_effect_error facade_id={facade_id} error={e}', flush=True)

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
                    directory=None, project=None):
        self.lifecycle_controller.start_load(
            operation, project_id,
            file_paths=file_paths, directory=directory, project=project)

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
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_workspace()
        self._refresh_report_preview()
        self._update_window_title()

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
