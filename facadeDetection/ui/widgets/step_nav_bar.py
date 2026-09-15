"""顶部四步业务流程导航栏。

替代原【项目操作】页面顶部十余个平铺命令按钮。四个步骤是一个有顺序的
业务漏斗，每一步完成状态需长期可见，因此用"徽标 + 标题卡片"表达：

    ①数据处理 → ②区域选取 → ③立面提取 → ④质量评估

状态语义：
- ``pending``：未执行，徽标显示序号；
- ``active``：当前正在执行（任务进行中，徽标保留序号并高亮）；
- ``done``：已完成，徽标显示 √；
- ``failed``：执行失败，徽标显示 ✗，直到重跑该步才恢复。

四个卡片在容器内等宽均分（``QSizePolicy.Expanding`` + 相同 stretch），
窗口缩放时四步始终均匀分布，不会挤在一侧留出空白。

组件本身不承载任何业务逻辑，只负责状态渲染与点击转发；步骤与业务动作的
绑定由 :class:`OperationPageMixin` 完成，从而保证底层算法与数据模型零改动。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget


PENDING = 'pending'
ACTIVE = 'active'
DONE = 'done'
FAILED = 'failed'
#: 徽标符号：完成 √ / 失败 ✗（未完成时显示步骤序号）
DONE_MARK = '✓'
FAILED_MARK = '✗'

#: 状态优先级。数值大者不可被小者覆盖：
#: 已完成（√）是"业绩记录"，只有显式强制失效（数据被替换/项目切换）
#: 才允许回到未完成态，因此点击步骤、重跑下游都不会抹掉已有的 √。
_STATE_PRIORITY = {
    PENDING: 0,
    FAILED: 1,
    ACTIVE: 2,
    DONE: 3,
}

#: 三步业务漏斗的展示顺序，索引即步骤号（0-based）。
# 改造后：区域选取 + 立面提取合并为【框选检测】
STEP_TITLES = ('数据处理', '框选检测', '质量评估')


def _repolish(widget: QWidget) -> None:
    """动态属性变更后强制重算 QSS，否则视觉状态不会刷新。"""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class StepBadge(QLabel):
    """圆形序号徽标：未完成显示序号，完成后显示 √。"""

    SIZE = 34

    def __init__(self, number: int, parent=None):
        super().__init__(str(number), parent)
        self._number = int(number)
        self.setObjectName('stepBadge')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._state = PENDING
        self._apply()

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._apply()

    def _apply(self) -> None:
        if self._state == DONE:
            self.setText(DONE_MARK)
        elif self._state == FAILED:
            self.setText(FAILED_MARK)
        else:
            self.setText(str(self._number))
        self.setProperty('badgeState', self._state)
        _repolish(self)


class StepChip(QFrame):
    """单个步骤卡片：徽标 + 标题，整块可点击。"""

    clicked = Signal(int)

    def __init__(self, index: int, title: str, parent=None):
        super().__init__(parent)
        self._index = int(index)
        self._state = PENDING

        self.setObjectName('stepChip')
        self.setProperty('stepState', PENDING)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 命中区域覆盖整个卡片（含内边距），避免必须精确点中文字。
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.badge = StepBadge(self._index + 1)
        self.title = QLabel(title)
        self.title.setObjectName('stepChipTitle')

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 18, 6)
        layout.setSpacing(10)
        layout.addWidget(self.badge)
        layout.addWidget(self.title)

    # ------------------------------------------------------------------
    @property
    def index(self) -> int:
        return self._index

    @property
    def state(self) -> str:
        return self._state

    @property
    def title_text(self) -> str:
        return self.title.text()

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.badge.set_state(state)
        self.setProperty('stepState', state)
        _repolish(self)

    def set_tooltip(self, text: str) -> None:
        self.setToolTip(text)

    # ------------------------------------------------------------------
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
                event.position().toPoint()):
            self.clicked.emit(self._index)
        super().mouseReleaseEvent(event)


class StepNavBar(QWidget):
    """四步导航栏容器。对外只暴露状态查询与点击信号。"""

    #: 参数为 0-based 步骤索引（0 数据处理 / 1 区域选取 / 2 立面提取 / 3 质量评估）
    step_clicked = Signal(int)
    #: (index, state) 状态变更广播，便于控制层同步其他区域
    step_state_changed = Signal(int, str)

    def __init__(self, titles=STEP_TITLES, parent=None):
        super().__init__(parent)
        self.setObjectName('stepNavBar')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._chips: list[StepChip] = []
        self._states: list[str] = []
        for index, title in enumerate(titles):
            chip = StepChip(index, title)
            chip.clicked.connect(self.step_clicked)
            chip.clicked.connect(self._remember_active)
            # 等宽均分：每个卡片都允许横向拉伸，并且 stretch 相同，
            # 于是四步在任意窗口宽度下都均匀分布，不再依赖尾部 addStretch。
            chip.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Preferred)
            layout.addWidget(chip, 1)
            self._chips.append(chip)
            self._states.append(PENDING)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def chips(self) -> list[StepChip]:
        return list(self._chips)

    def state(self, index: int) -> str:
        return self._states[index]

    def is_done(self, index: int) -> bool:
        return self._states[index] == DONE

    def is_failed(self, index: int) -> bool:
        return self._states[index] == FAILED

    def states(self) -> list[str]:
        return list(self._states)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self._states if s == DONE)

    def all_done(self) -> bool:
        return self.completed_count == len(self._states)

    # ------------------------------------------------------------------
    # 状态设置
    # ------------------------------------------------------------------
    def mark_done(self, index: int) -> None:
        """标记步骤完成（徽标变 √）。"""
        self.set_state(index, DONE)

    def mark_failed(self, index: int) -> None:
        """标记步骤失败（徽标变 ✗）。重跑该步时会被 mark_running 清除。"""
        self.set_state(index, FAILED)

    def mark_pending(self, index: int) -> None:
        self.set_state(index, PENDING)

    def mark_running(self, index: int) -> None:
        """进入执行态：清掉上一次的失败标记，但保留已有的 √。"""
        if self._states[index] == DONE:
            return
        self.set_state(index, ACTIVE)

    def set_state(self, index: int, state: str, *, force: bool = False) -> None:
        """设置步骤状态。

        默认遵守 :data:`_STATE_PRIORITY`：**已完成的 √ 不会被低优先级状态
        覆盖**。于是"点击步骤跳转""重跑下游""失败重试"等操作都不会抹掉
        用户已经取得的完成记录；只有显式 ``force=True``（数据被替换、
        项目切换等语义变化）或 :meth:`invalidate` 才能清掉 √。
        """
        current = self._states[index]
        if current == state:
            return
        if not force and _STATE_PRIORITY.get(state, 0) < _STATE_PRIORITY.get(current, 0):
            return
        self._states[index] = state
        self._chips[index].set_state(state)
        self.step_state_changed.emit(index, state)

    def invalidate(self, index: int) -> None:
        """强制让某一步回到未完成态（仅在上游结果语义真的失效时调用）。"""
        self.set_state(index, PENDING, force=True)

    def reset(self, *, keep_until: int = -1) -> None:
        """整体复位。``keep_until`` 之前的步骤保持已完成（用于重做后续步骤）。"""
        for index in range(len(self._chips)):
            self.set_state(index, DONE if index <= keep_until else PENDING,
                           force=True)

    def reset_after(self, index: int) -> None:
        """使 ``index`` 之后的所有步骤失效（上游数据被替换时必须重新执行）。"""
        for later in range(index + 1, len(self._chips)):
            self.set_state(later, PENDING, force=True)

    def reset_to_running(self, index: int) -> None:
        """重做某一步：自身回到执行态，下游全部失效。"""
        self.mark_running(index)
        self.reset_after(index)

    # ------------------------------------------------------------------
    def _remember_active(self, index: int) -> None:
        """点击即视为"正在这一步"；纯展示状态，不写业务数据。"""
        for other, chip in enumerate(self._chips):
            if other != index and chip.state == ACTIVE:
                chip.set_state(DONE if self._states[other] == DONE else PENDING)
        if self._states[index] != DONE:
            self.set_state(index, ACTIVE)
