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
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qtwebview2 import QtWebView2Widget

from .widgets.flow_layout import FlowLayout
from services.inspection_review import InspectionReviewService
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.report_export import ReportExportService
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
    ),
}

UPLOAD_FILE_FILTER = (
    '项目支持文件 '
    '(*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls '
    '*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '点云文件 (*.ply *.pcd *.xyz *.pts *.las *.laz *.e57 *.fls);;'
    '图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;'
    '所有文件 (*)'
)

REPORT_PDF_FILTER = 'PDF 文件 (*.pdf)'
REPORT_EMPTY_TITLE = '请选择PDF上传'
APPLICATION_TITLE = '点云外立面智能检测平台'

# 每个页面只突出一个主要操作，避免顶部十余个按钮全部使用主色。
PRIMARY_HEADER_ACTIONS = {
    'btn_upload',
    'btn_facade_detection',
    'btn_open_report_pdf',
}

# Corporate Clean 强调清晰的功能分组。按钮名称和业务连接保持不变，
# 这里只定义它们在顶部命令栏中的视觉归属。
PAGE_HEADER_GROUPS = {
    'project_overview': (
        (
            '',
            (
                'btn_upload',
                'btn_open_project',
                'btn_select_project',
                'btn_new_project',
            ),
        ),
    ),
    'project_operation': (
        ('视图', ('btn_reset_view', 'btn_change_color')),
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
        ('', ('btn_open_report_pdf',)),
    ),
}


class ElidedLabel(QLabel):
    """在空间不足时省略中间文本，同时保留完整内容供 Tooltip 查看。"""

    def __init__(self, text='', parent=None):
        super().__init__('', parent)
        self._full_text = text
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._update_elided_text()

    def setText(self, text):
        self._full_text = text
        self._update_elided_text()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(min(hint.width(), 480))
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('mainWindow')
        self.setWindowTitle(APPLICATION_TITLE)
        self.resize(1600, 900)
        # 三维工作台在过窄尺寸下失去可用性；该下限同时保证命令栏和四页签不溢出。
        self.setMinimumSize(960, 640)
        self.viewport = Open3DViewport()
        self.project_overview_service = ProjectOverviewService()
        self.project_operation_service = ProjectOperationService(self.viewport)
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        self.page_title_labels = {}
        self._sidebar_collapsed = {'left': False, 'right': False}
        self._last_upload_directory = str(Path.home())
        self._header_resize_pending = set()
        self._current_report_pdf_name = None
        self._report_navigation_index = 0
        self._report_webview_error = None
        self._setup_ui()
        self._connect_buttons()
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
        """创建始终可见的应用栏，页面名称与业务操作留给内容区。"""
        header = QWidget()
        header.setObjectName('applicationHeader')
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(64)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(8)

        brand_mark = QLabel('P3D')
        brand_mark.setObjectName('applicationBrandMark')
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(40, 40)
        layout.addWidget(brand_mark)

        brand_title = QLabel('点云外立面智能检测平台')
        brand_title.setObjectName('applicationBrandTitle')
        layout.addWidget(brand_title)

        layout.addStretch(1)

        self.current_project_label = ElidedLabel('当前项目：未选择')
        self.current_project_label.setObjectName('currentProjectLabel')
        self.current_project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_project_label.setProperty('uiRole', 'supportingText')
        self.current_project_label.setMinimumWidth(200)
        self.current_project_label.setMaximumWidth(360)
        self.current_project_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(self.current_project_label)
        return header

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

    def _create_page_title(self, page_title, page_key):
        label = QLabel(page_title)
        label.setObjectName(f'{page_key}PageTitle')
        label.setProperty('uiRole', 'pageTitle')
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.page_title_labels[page_key] = label
        return label

    def _create_page_shell(self, page_title, page_key, page=None):
        """为四个主页面提供相同的标题、命令栏和工作区层级。"""
        page = page or QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        content = QWidget()
        content.setProperty('uiRole', 'contentArea')
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(16)

        heading_row = QWidget()
        heading_row.setProperty('uiRole', 'pageHeadingRow')
        heading_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 固定标题行高度，避免报告页内部导航把工作区整体向下挤动。
        heading_row.setFixedHeight(40)
        heading_layout = QHBoxLayout(heading_row)
        heading_layout.setContentsMargins(0, 0, 0, 0)
        heading_layout.setSpacing(16)
        heading_layout.addWidget(self._create_page_title(page_title, page_key))
        heading_layout.addStretch(1)

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

        content_layout.addWidget(heading_row)
        content_layout.addWidget(workspace, 1)
        page_layout.addWidget(content, 1)
        return page, workspace_body_layout, heading_layout

    def _create_project_overview_page(self, page_title, page_key):
        page, body_layout, _heading_layout = self._create_page_shell(
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
        workspace_panel.setMinimumWidth(288)
        workspace_panel.setMaximumWidth(352)
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(24, 20, 24, 24)
        workspace_layout.setSpacing(8)

        workspace_title = QLabel('当前工作区')
        workspace_title.setProperty('uiRole', 'sectionTitle')
        self.overview_workspace_name_label = QLabel('未选择项目')
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
        workspace_layout.addSpacing(16)

        workspace_divider = QFrame()
        workspace_divider.setObjectName('workspaceSummaryDivider')
        workspace_divider.setFrameShape(QFrame.Shape.HLine)
        workspace_layout.addWidget(workspace_divider)

        project_count_title = QLabel('项目总数')
        project_count_title.setProperty('uiRole', 'metricTitle')
        self.metric_project_count_label = QLabel('0')
        self.metric_project_count_label.setProperty('uiRole', 'summaryMetricValue')
        file_count_title = QLabel('数据文件')
        file_count_title.setProperty('uiRole', 'metricTitle')
        self.metric_file_count_label = QLabel('0')
        self.metric_file_count_label.setProperty('uiRole', 'summaryMetricValue')

        summary_metrics = QHBoxLayout()
        summary_metrics.setContentsMargins(0, 4, 0, 0)
        summary_metrics.setSpacing(24)
        for title_label, value_label in (
            (project_count_title, self.metric_project_count_label),
            (file_count_title, self.metric_file_count_label),
        ):
            metric_column = QVBoxLayout()
            metric_column.setContentsMargins(0, 0, 0, 0)
            metric_column.setSpacing(3)
            metric_column.addWidget(title_label)
            metric_column.addWidget(value_label)
            summary_metrics.addLayout(metric_column, 1)
        workspace_layout.addLayout(summary_metrics)
        workspace_layout.addStretch(1)
        columns_layout.addWidget(workspace_panel)

        body_layout.addWidget(overview_columns, 1)
        return page

    def _create_operation_page(self, page_title, page_key):
        page = QWidget()
        self.operation_page = page
        page, body_layout, _heading_layout = self._create_page_shell(
            page_title,
            page_key,
            page=page,
        )

        self.operation_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.operation_splitter.setObjectName('operationPageSplitter')
        self.operation_splitter.setChildrenCollapsible(False)
        # 三个区域共享一个工作台外框，细分隔线代替三张彼此孤立的卡片。
        self.operation_splitter.setHandleWidth(1)

        (
            self.left_dock,
            self.left_sidebar_button,
        ) = self._create_sidebar('leftDock', 'left')
        (
            self.right_dock,
            self.right_sidebar_button,
        ) = self._create_sidebar('rightDock', 'right')

        viewport_panel = QWidget()
        viewport_panel.setObjectName('viewportPanel')
        viewport_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        viewport_layout = QVBoxLayout(viewport_panel)
        viewport_layout.setContentsMargins(16, 16, 16, 16)
        viewport_layout.setSpacing(8)
        viewport_heading_row = QHBoxLayout()
        viewport_heading_row.setContentsMargins(0, 0, 0, 0)
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
        viewport_layout.addWidget(self.viewport.get_widget(), 1)

        self.operation_splitter.addWidget(self.left_dock)
        self.operation_splitter.addWidget(viewport_panel)
        self.operation_splitter.addWidget(self.right_dock)
        self.operation_splitter.setStretchFactor(0, 0)
        self.operation_splitter.setStretchFactor(1, 1)
        self.operation_splitter.setStretchFactor(2, 0)
        self.operation_splitter.setSizes([210, 1000, 210])
        self.operation_splitter.installEventFilter(self)
        body_layout.addWidget(self.operation_splitter, 1)

        self.left_sidebar_expand_button = self._create_sidebar_expand_button('left')
        self.right_sidebar_expand_button = self._create_sidebar_expand_button('right')
        return page

    def _create_report_export_page(self, page_title, page_key):
        page, body_layout, heading_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(16)

        self.report_navigation_stack = QStackedWidget()
        self.report_navigation_stack.setObjectName('reportNavigationStack')
        heading_layout.addWidget(self._create_report_navigation())

        document_header = QWidget()
        document_header.setObjectName('reportDocumentHeader')
        document_header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        document_header_layout = QHBoxLayout(document_header)
        document_header_layout.setContentsMargins(0, 0, 0, 0)
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

        report_empty_state = QWidget()
        report_empty_state.setObjectName('reportEmptyState')
        report_empty_state.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        report_empty_layout = QVBoxLayout(report_empty_state)
        report_empty_layout.setContentsMargins(0, 0, 0, 0)
        self.report_preview_state_stack.addWidget(report_empty_state)

        report_document_page = QWidget()
        report_document_page.setObjectName('reportDocumentPage')
        report_document_layout = QVBoxLayout(report_document_page)
        report_document_layout.setContentsMargins(0, 0, 0, 0)
        report_document_layout.setSpacing(0)
        # QtWebView2Widget 只嵌入系统 WebView2 内容区，不创建浏览器地址栏。
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
        report_document_layout.addWidget(self.report_webview, 1)
        self.report_preview_state_stack.addWidget(report_document_page)
        report_preview_layout.addWidget(self.report_preview_state_stack, 1)

        heatmap_page = QWidget()
        heatmap_page.setObjectName('reportHeatmapPage')
        heatmap_page.setProperty('uiRole', 'contentArea')
        heatmap_page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        heatmap_layout = QVBoxLayout(heatmap_page)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)

        # 保留稳定的控件接口，后续算法视口可以直接替换此占位控件。
        heatmap_placeholder = QLabel('')
        heatmap_placeholder.setObjectName('heatmapPlaceholder')
        heatmap_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        page, _body_layout, _heading_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        return page

    def _create_page_header(self, page_key):
        panel = QWidget()
        panel.setObjectName(f'{page_key}HeaderPanel')
        panel.setProperty('uiRole', 'pageHeader')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setMinimumHeight(56)
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

        QTimer.singleShot(0, lambda: self._resize_page_header(panel))
        return panel

    def _create_sidebar(self, object_name, side):
        sidebar = QFrame()
        sidebar.setObjectName(object_name)
        sidebar.setProperty('uiRole', 'sidebar')
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title_bar = QWidget()
        title_bar.setObjectName(f'{object_name}TitleBar')
        title_bar.setProperty('uiRole', 'sidebarTitleBar')
        title_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        panel.setProperty('uiRole', 'sidebarBody')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar_layout.addWidget(panel, 1)
        sidebar.setMinimumWidth(180)
        sidebar.setMaximumWidth(260)
        sidebar.setProperty('expandedWidth', 210)
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
        # 保持普通 Qt 子控件；原生化会连带提升 PageStack 并造成切页层级错乱。
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
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        navigation_panel = QWidget()
        navigation_panel.setObjectName('bottomNavigation')
        navigation_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        navigation_layout = QHBoxLayout(navigation_panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(0)

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
        layout.addWidget(navigation_panel, 1)

        dock.setWidget(panel)
        dock.setMinimumHeight(64)
        dock.setMaximumHeight(64)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    def set_current_page(self, page_index):
        if not 0 <= page_index < len(PAGE_DEFINITIONS):
            return
        page_key = PAGE_DEFINITIONS[page_index][1]
        button = self.page_buttons.get(page_key)
        if button is not None and not button.isEnabled():
            return

        self.page_stack.setCurrentIndex(page_index)
        if button is not None:
            button.setChecked(True)
        if page_key == 'project_operation':
            QTimer.singleShot(0, self._position_sidebar_expand_buttons)
        self._update_window_title(page_key)

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
            'btn_upload': self._open_upload_file_dialog,
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_reset_view': self.project_operation_service.reset_view,
            'btn_change_color': self.project_operation_service.change_color,
            'btn_denoise': self.project_operation_service.denoise,
            'btn_registration': self.project_operation_service.registration,
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
        uploaded_paths = self.project_overview_service.upload_files(file_paths)
        project = self.project_overview_service.register_upload(
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
        project = self.project_overview_service.open_project(directory_path)
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
        project = self.project_overview_service.create_project(
            name,
            directory_path,
        )
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

    def _update_overview_metrics(self, projects=None):
        """同步概览统计卡，避免在 UI 层复制项目业务状态。"""
        if not hasattr(self, 'metric_project_count_label'):
            return
        if projects is None:
            projects = self.project_overview_service.list_projects()

        project_list = list(projects)
        file_count = sum(len(project.file_paths) for project in project_list)

        self.metric_project_count_label.setText(str(len(project_list)))
        self.metric_file_count_label.setText(str(file_count))
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
        self.overview_workspace_file_label.setText(
            f'{len(self.current_project.file_paths)} 个数据文件'
        )

    def _refresh_project_list(self):
        while self.project_list_layout.count() > 1:
            item = self.project_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        projects = self.project_overview_service.list_projects()
        self._update_overview_metrics(projects)
        if not projects:
            empty_label = QLabel('暂无项目')
            empty_label.setObjectName('emptyProjectLabel')
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.project_list_layout.insertWidget(0, empty_label)
            return

        for project in projects:
            project_row = QWidget()
            project_row.setObjectName('projectRow')
            project_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            project_row.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            project_row.setMinimumHeight(88)
            project_row_layout = QHBoxLayout(project_row)
            project_row_layout.setContentsMargins(18, 12, 14, 12)
            project_row_layout.setSpacing(12)

            project_info = QWidget()
            project_info.setObjectName('projectInfo')
            project_info_layout = QVBoxLayout(project_info)
            project_info_layout.setContentsMargins(0, 0, 0, 0)
            project_info_layout.setSpacing(4)

            project_name = QLabel(project.name)
            project_name.setObjectName('projectNameLabel')
            project_name.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            project_path = ElidedLabel(f'目录  {project.directory_path}')
            project_path.setObjectName('projectPathLabel')
            project_path.setToolTip(project.directory_path)
            project_meta = QLabel(f'数据文件  {len(project.file_paths)}')
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
            open_button.setFixedSize(72, 32)
            open_button.clicked.connect(
                lambda _checked=False, project_id=project.project_id:
                self._open_project_card(project_id)
            )

            edit_button = QPushButton('编辑')
            edit_button.setObjectName('btn_edit_project')
            edit_button.setToolTip('修改项目名称')
            edit_button.setAccessibleName('编辑项目')
            edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
            edit_button.setFixedSize(72, 32)
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
            delete_button.setFixedSize(72, 32)
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
        project = self.project_overview_service.get_project(project_id)
        if project is None:
            QMessageBox.warning(self, '编辑项目', '项目不存在或已被删除。')
            return

        new_name, accepted = QInputDialog.getText(
            self,
            '编辑项目',
            '项目名称：',
            text=project.name,
        )
        if not accepted:
            return

        try:
            updated_project = self.project_overview_service.rename_project(
                project_id,
                new_name,
            )
        except ValueError as error:
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

        for page_title, page_key in PAGE_DEFINITIONS:
            page_title_label = self.page_title_labels.get(page_key)
            if page_title_label is not None:
                page_title_label.setText(page_title)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(project.directory_path)
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_metrics()
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
        target_height = max(56, min(content_height, 200))

        if (
            panel.minimumHeight() == target_height
            and panel.maximumHeight() == target_height
        ):
            return

        panel.setMinimumHeight(target_height)
        panel.setMaximumHeight(target_height)
