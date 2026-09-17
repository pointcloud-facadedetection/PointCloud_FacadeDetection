"""操作页步骤导航控制器。

从 main_window.py 抽取（零逻辑变化）：步骤栏只做两件事——把点击转发给
既有命令、在业务真正结束后回填 √/✗ 徽标。算法、数据模型与命令按钮
全部保持原状。

区域选取 + 立面提取已合并为【框选检测】。
"""

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QSizePolicy

from ui.widgets.step_nav_bar import StepNavBar
from ui.controllers.task_progress import (
    TASK_DENOISE,
    TASK_DETECTION,
    TASK_MODEL_EXPORT,
    TASK_QUALITY,
    TASK_QUALITY_BATCH,
    TASK_REGION,
)


class StepNavigationController(QObject):
    """三步业务流程编排：点击转发 + 徽标回填 + 进度窗联动。"""

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

    def __init__(self, window, task_progress, parent=None):
        """window 为 MainWindow（装配根），生命周期覆盖本控制器。"""
        super().__init__(parent)
        self._window = window
        self.task_progress = task_progress

    @property
    def nav(self):
        return getattr(self._window, 'step_nav', None)

    # ------------------------------------------------------------------
    # 装配与点击转发
    # ------------------------------------------------------------------
    def install(self, header_layout):
        """在项目操作页命令栏位置装配顶部步骤导航栏。"""
        window = self._window
        window.step_nav = StepNavBar()
        for index, tip in enumerate(self.STEP_TIPS):
            window.step_nav.chips[index].set_tooltip(tip)
        window.step_nav.step_clicked.connect(self.on_nav_clicked)
        # stretch=1：步骤导航栏吞掉侧栏开关组之外的全部宽度，配合
        # StepNavBar 内部卡片相同的 stretch，实现"均分铺满整个顶层栏"。
        header_layout.addWidget(window.step_nav, 1)
        for button in (getattr(window, 'left_sidebar_button', None),
                       getattr(window, 'right_sidebar_button', None)):
            if button is not None:
                button.setSizePolicy(QSizePolicy.Policy.Fixed,
                                     QSizePolicy.Policy.Fixed)

    def on_nav_clicked(self, index):
        """步骤点击只触发既有命令按钮，行为与改造前完全一致。"""
        window = self._window
        if index == self.STEP_BOX_DETECT:
            # ② 框选检测：进入 ROI 框选 + 确认卡片流程
            window._enter_box_detection_mode()
            return
        if index == self.STEP_QUALITY:
            # ③ 复用原“评估选中立面”（按“处理”标记批量评估）。
            nav = self.nav
            if nav is not None:
                nav.reset_to_running(index)
            try:
                window._evaluate_selected_facade()
            except Exception as exc:
                window.statusBar().showMessage(f'质量评估失败：{exc}', 8000)
            return
        button_name = self.STEP_BUTTONS.get(index)
        button = window.header_buttons.get(button_name) if button_name else None
        if button is None:
            return
        # 重做该步：自身进入执行态（清掉上一次的 ✗），下游结果一并失效，
        # 之后再靠业务完成/失败信号回填 √ 或 ✗，绝不"点击即打勾"。
        window.step_nav.reset_to_running(index)
        self._begin_step_task(index)
        button.click()

    # ------------------------------------------------------------------
    # 进度窗联动
    # ------------------------------------------------------------------
    def _begin_step_task(self, index):
        """为某个耗时步骤立起模态进度窗；无对应任务的步骤跳过。"""
        task_key = self.STEP_TASK_KEYS.get(index)
        if task_key is not None:
            self.begin_task_progress(task_key)

    def begin_task_progress(self, task_key):
        """立起模态进度窗：不定量忙碌条，文案极简。"""
        title = self.STEP_TASK_TITLES.get(task_key, '任务进行中')
        self.task_progress.begin(task_key, title, '任务处理中')

    def on_step_task_finished(self, index, task_key, success):
        """耗时步骤的统一收尾：先给进度窗定终态，再回填步骤徽标。"""
        self.task_progress.finish(task_key, success)
        if success:
            self.mark_step_done(index)
        else:
            self.mark_step_failed(index)

    def mark_step_done(self, index):
        """仅在对应业务动作真正执行完毕后调用，避免“点击即打勾”。"""
        nav = self.nav
        if nav is not None:
            nav.mark_done(index)

    def mark_step_failed(self, index):
        """业务失败时把步骤标成 ✗；重跑该步会自动清除失败标记。"""
        nav = self.nav
        if nav is not None:
            nav.mark_failed(index)
        # 失败即代表该步结果无效，下游步骤必须重新执行。
        if nav is not None:
            nav.reset_after(index)

    def on_quality_batch_progress(self, current, total):
        """批量评估进度：忙碌条只刷新阶段文案。"""
        total = max(int(total or 0), 1)
        current = max(0, min(int(current or 0), total))
        self.task_progress.report(
            TASK_QUALITY_BATCH, text=f'任务处理中（{current}/{total}）')

    # ------------------------------------------------------------------
    # 业务信号接线
    # ------------------------------------------------------------------
    def bind_completion_signals(self):
        """把 √ 状态、进度窗收尾与**真实进度信号**挂到服务层。

        每条业务链路都是"进度信号 → 节流写进度窗"、
        "完成/失败信号 → 先收进度窗、再改徽标"。
        """
        window = self._window
        service = window.project_operation_service

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
                    self.on_step_task_finished(_i, _k, True))

        # ---- (3) 去噪 / 框选检测（立面提取）：失败 → 保留当前进度 + 徽标 ✗ ----
        for signal_name, index, task_key in (
            ('denoise_failed', self.STEP_DATA, TASK_DENOISE),
            ('detection_failed', self.STEP_BOX_DETECT, TASK_DETECTION),
        ):
            signal = getattr(service, signal_name, None)
            if signal is not None:
                signal.connect(
                    lambda _payload=None, _i=index, _k=task_key:
                    self.on_step_task_finished(_i, _k, False))

        # ---- (4) 区域选取（已并入框选检测）：完成 / 失败 ----
        for signal_name, index, task_key, ok in (
            ('region_finished', self.STEP_BOX_DETECT, TASK_REGION, True),
            ('region_failed', self.STEP_BOX_DETECT, TASK_REGION, False),
        ):
            signal = getattr(service, signal_name, None)
            if signal is not None:
                signal.connect(
                    lambda _payload=None, _i=index, _k=task_key, _ok=ok:
                    self.on_step_task_finished(_i, _k, _ok))

        # ---- (5) 质量评估：批量 + 单个 + 失败 ----
        controller = window.facade_quality_controller
        batch_finished = getattr(controller, 'batch_finished', None)
        if batch_finished is not None:
            batch_finished.connect(
                lambda _payload=None: self.mark_step_done(self.STEP_QUALITY))
        quality_completed = getattr(controller, 'quality_completed', None)
        if quality_completed is not None:
            quality_completed.connect(
                lambda *_args: self.mark_step_done(self.STEP_QUALITY))
        quality_failed = getattr(controller, 'quality_failed', None)
        if quality_failed is not None:
            quality_failed.connect(
                lambda *_args: self.mark_step_failed(self.STEP_QUALITY))
