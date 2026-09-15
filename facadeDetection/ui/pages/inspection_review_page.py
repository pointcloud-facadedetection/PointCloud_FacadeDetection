from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget, QLabel


class InspectionReviewPageMixin:
    """检测复核页布局 mixin。

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

        # 左侧占位：接收 viewport_panel
        self.review_viewport_placeholder = QWidget()
        self.review_viewport_placeholder.setObjectName('reviewViewportPlaceholder')
        pl = QVBoxLayout(self.review_viewport_placeholder)
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel('3D 检测工作台将在此处显示')
        lbl.setStyleSheet('font-size:14px; color:#64748B;')
        pl.addWidget(lbl)

        # 右侧占位：接收 right_dock
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