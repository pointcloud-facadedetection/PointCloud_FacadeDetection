from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class InspectionReviewPageMixin:
    """检测复核页 mixin：布局 + 视口迁移 + 复核立面列表。

    接收从【项目操作】页迁移过来的 viewport_panel（3D 视口）和
    right_dock（立面检测结果面板），形成两栏布局。
    """

    def _create_inspection_review_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(page_title, page_key)

        # 检测复核页：两栏 splitter（视口 + 右侧面板）
        self.review_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.review_splitter.setObjectName('reviewSplitter')
        self.review_splitter.setChildrenCollapsible(False)
        self.review_splitter.setHandleWidth(1)

        # 左侧占位：视口迁移到本页时被摘出隐藏，迁回操作页时重新挂回
        self.review_viewport_placeholder = QWidget()
        self.review_viewport_placeholder.setObjectName('reviewViewportPlaceholder')
        pl = QVBoxLayout(self.review_viewport_placeholder)
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel('3D 检测工作台将在此处显示')
        lbl.setStyleSheet('font-size:14px; color:#64748B;')
        pl.addWidget(lbl)

        # 右侧占位：右侧面板迁移到本页时被摘出隐藏，迁回操作页时重新挂回
        self.review_right_placeholder = QWidget()
        self.review_right_placeholder.setObjectName('reviewRightPlaceholder')
        pl2 = QVBoxLayout(self.review_right_placeholder)
        pl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl2 = QLabel('立面检测结果将在此处显示')
        lbl2.setStyleSheet('font-size:14px; color:#64748B;')
        pl2.addWidget(lbl2)

        self.review_splitter.addWidget(self.review_viewport_placeholder)
        self.review_splitter.addWidget(self.review_right_placeholder)
        self.review_splitter.setStretchFactor(0, 1)
        self.review_splitter.setStretchFactor(1, 0)
        self.review_splitter.setSizes([1040, 300])

        body_layout.addWidget(self.review_splitter, 1)
        return page

    # ------------------------------------------------------------------
    # 视口迁移：检测复核页 ↔ 项目操作页
    # ------------------------------------------------------------------
    def _migrate_workspace_to_review(self):
        """将 viewport_panel + right_dock 迁到检测复核页。

        占位 widget 常驻不销毁：迁入时把占位摘出、面板插入；迁出时反向。
        绝不用 QSplitter.replaceWidget——它会消费占位 widget，迁出后
        splitter 计数归零，二次进入时按索引替换必然越界。
        """
        if getattr(self, '_workspace_in_review', False):
            return
        self._operation_splitter_sizes = self.operation_splitter.sizes()
        self.review_viewport_placeholder.setParent(None)
        self.review_right_placeholder.setParent(None)
        # insertWidget 自动把面板从 operation_splitter 摘除并挂入本页
        self.review_splitter.insertWidget(0, self.viewport_panel)
        self.review_splitter.insertWidget(1, self.right_dock)
        self.review_splitter.setSizes([1040, 300])
        self._show_operation_placeholder()
        results = self.project_operation_service.last_facade_results or []
        self._show_facade_results_review(results)
        self._workspace_in_review = True

    def _migrate_workspace_to_operation(self):
        """将 viewport_panel + right_dock 迁回项目操作页。"""
        if not getattr(self, '_workspace_in_review', False):
            return
        self.operation_splitter.insertWidget(1, self.viewport_panel)
        self.operation_splitter.insertWidget(2, self.right_dock)
        # 占位 widget 挂回复核页，保持其默认引导界面可用
        self.review_splitter.insertWidget(0, self.review_viewport_placeholder)
        self.review_splitter.insertWidget(1, self.review_right_placeholder)
        self.review_splitter.setSizes([1040, 300])
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
        row_layout.setContentsMargins(8, 6, 14, 6)
        row_layout.setSpacing(8)
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
