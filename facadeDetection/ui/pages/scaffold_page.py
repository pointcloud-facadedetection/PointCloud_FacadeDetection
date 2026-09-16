"""主窗口外壳 mixin：标题栏/页面导航/项目栏/页面栈/页头/窗口缩放。

零逻辑变化搬运自 main_window.py（MieMie 37152ed 拆分层级的延续）。
"""
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.main_window_config import (
    APPLICATION_TITLE,
    PAGE_BUTTON_NAMES,
    PAGE_DEFINITIONS,
    PAGE_HEADER_ACTIONS,
    PAGE_HEADER_GROUPS,
    PRIMARY_HEADER_ACTIONS,
)
from ui.controllers.task_progress import TASK_LOAD, TASK_PAGE_SWITCH
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.technical_canvas import TechnicalCanvas
from ui.widgets.window_chrome import (
    ApplicationTitleBar,
    ElidedLabel,
    WindowResizeHandle,
    make_window_control_icon,
)


class ScaffoldPageMixin:
    """窗口外壳与页面骨架（方法经 mixin 挂到 MainWindow）。"""

    def _setup_ui(self):
        # 统一应用外壳：标题栏(含页面导航) + 项目栏 + 页面内容。
        shell = QWidget()
        shell.setObjectName('applicationShell')
        shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._create_application_header())
        shell_layout.addWidget(self._create_project_context_bar())

        self.page_stack = self._create_page_stack()
        shell_layout.addWidget(self.page_stack, 1)
        self.setCentralWidget(shell)

        self.set_current_page(0)

    def _create_application_header(self):
        """可拖动自定义标题栏：左品牌+页名，正中页面导航，右侧窗口按钮。"""
        header = ApplicationTitleBar()
        header.setObjectName('applicationHeader')
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(40)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(10)

        # 左侧与右侧使用相同的拉伸权重，保证页面导航落在标题栏正中央。
        left_section = QWidget()
        left_layout = QHBoxLayout(left_section)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        brand_mark = QLabel()
        brand_mark.setObjectName('applicationBrandMark')
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(30, 30)
        logo_path = Path(__file__).resolve().parents[2] / 'utils' / 'logo.png'
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            brand_mark.setPixmap(pixmap.scaled(
                24, 24, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            brand_mark.setStyleSheet('background:transparent;border:none;')
        brand_mark.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        left_layout.addWidget(brand_mark)

        # 顶栏只显示当前页面名称，避免和窗口标题重复展示平台名称。
        self.application_page_title = QLabel(PAGE_DEFINITIONS[0][0])
        self.application_page_title.setObjectName('applicationPageTitle')
        self.application_page_title.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        left_layout.addWidget(self.application_page_title)
        left_layout.addStretch(1)
        layout.addWidget(left_section, 1)

        # 正中：页面导航（原底部 dock 的四枚互斥页签）。
        layout.addWidget(self._create_page_navigation(), 0)

        window_controls = QWidget()
        window_controls.setObjectName('windowControls')
        controls_layout = QHBoxLayout(window_controls)
        controls_layout.setContentsMargins(4, 0, 0, 0)
        controls_layout.setSpacing(0)

        self.window_minimize_button = self._create_window_control_button(
            'btn_window_minimize',
            'minimize',
            '最小化',
        )
        self.window_maximize_button = self._create_window_control_button(
            'btn_window_maximize_restore',
            'maximize',
            '最大化',
        )
        self.window_close_button = self._create_window_control_button(
            'btn_window_close',
            'close',
            '关闭',
        )
        self.window_close_button.setProperty('windowAction', 'close')
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.window_minimize_button)
        controls_layout.addWidget(self.window_maximize_button)
        controls_layout.addWidget(self.window_close_button)
        layout.addWidget(window_controls, 1)

        self.window_minimize_button.clicked.connect(self.showMinimized)
        self.window_maximize_button.clicked.connect(
            self._toggle_maximize_restore
        )
        self.window_close_button.clicked.connect(self.close)
        return header

    def _create_page_navigation(self):
        """标题栏正中的四枚互斥页面页签（原底部导航 dock）。"""
        navigation_panel = QWidget()
        navigation_panel.setObjectName('titlebarNavigation')
        navigation_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
            button.setProperty('navigationLevel', 'titlebar')
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
        return navigation_panel

    def _create_project_context_bar(self):
        """紧贴标题栏下方的当前项目栏：高度与字号匹配的一行。"""
        bar = QWidget()
        bar.setObjectName('projectContextBar')
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(30)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 3, 24, 3)
        layout.setSpacing(0)

        self.current_project_label = ElidedLabel(
            '当前项目：未选择',
            maximum_hint_width=None,
        )
        self.current_project_label.setObjectName('currentProjectLabel')
        self.current_project_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.current_project_label.setProperty('uiRole', 'supportingText')
        # ElidedLabel 默认水平 Ignored + 最小宽度 0，必须显式给最小宽度
        # 与 Preferred 策略，否则布局会把它压成 0 宽、文本被裁空。
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
        layout.addStretch(1)
        return bar

    def _create_window_control_button(self, object_name, icon_kind, tooltip):
        """创建与深色标题栏一致的标准窗口控制按钮（QPainter 图标）。"""
        button = QToolButton()
        button.setObjectName(object_name)
        button.setProperty('uiRole', 'windowControl')
        button.setIcon(make_window_control_icon(icon_kind, logical_size=14))
        button.setIconSize(QSize(14, 14))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(46, 40)
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
        self.window_maximize_button.setIcon(make_window_control_icon(
            'restore' if is_maximized else 'maximize'))
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

    def set_current_page(self, page_index):
        if not 0 <= page_index < len(PAGE_DEFINITIONS):
            return
        page_title, page_key = PAGE_DEFINITIONS[page_index]
        button = self.page_buttons.get(page_key)
        if button is not None and not button.isEnabled():
            return

        # 切换期间弹忙碌条阻挡操作，防止视口迁移/首帧渲染等开销被并发点击
        # 放大。仅覆盖带 Open3D 视口的两个页面；加载忙碌条已在时由它覆盖
        # 本次切换，不再叠第二个弹窗；启动期（窗口尚未显示）不弹。
        load_active = self.task_progress.is_active(TASK_LOAD)
        show_busy = (not load_active and self.isVisible()
                     and page_key in ('project_operation', 'inspection_review'))
        if show_busy:
            self.task_progress.begin(TASK_PAGE_SWITCH, '页面切换', '正在切换页面...')

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

        if show_busy:
            self._finish_task_after_first_frame(TASK_PAGE_SWITCH)

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
