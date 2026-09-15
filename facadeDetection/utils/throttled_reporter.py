"""耗时任务的文案节流合并器。

设计要点（对应"进度数据每 5 秒合并一次、但 UI 保持流畅"的需求）：

1. 节流点放在 GUI 侧，而不是信号链路。后台 worker 照常高频 emit，
   跨线程信号经 QueuedConnection 投递到 GUI 线程槽后，只调用
   ``report()``——一次赋值，代价可忽略。
2. ``QTimer`` 每 ``interval_ms`` 触发一次，仅当缓存被写过（dirty）
   才发出一次 ``flushed``，于是重绘频率恒定在 5 秒一次，界面不会
   被高频文本更新拖慢，事件循环始终空闲。
3. 终态必须刷：``flush_now()`` 绕过节流，避免终态文案停留在上一阶段。

进度条统一为不定量忙碌条，因此本模块只节流阶段文案，不再承载百分比。

本模块不依赖任何业务代码，可独立测试。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal


DEFAULT_INTERVAL_MS = 5000


class ThrottledReporter(QObject):
    """把高频文案回报合并为固定间隔的低频 UI 刷新。"""

    #: (text)
    flushed = Signal(str)

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self._interval_ms = max(1, int(interval_ms))
        self._text: str = ''
        self._dirty: bool = False
        self._auto_start: bool = True

        self._timer = QTimer(self)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self._on_timeout)

    # ------------------------------------------------------------------
    # 只读属性（测试与调试用）
    # ------------------------------------------------------------------
    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    @property
    def pending(self) -> bool:
        """缓存中是否有尚未刷新的文案。"""
        return self._dirty

    @property
    def snapshot(self) -> str:
        """最近一次写入缓存的文案（不触发刷新）。"""
        return self._text

    def is_active(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------
    # 写入：仅更新缓存，绝不触碰 UI
    # ------------------------------------------------------------------
    def report(self, text: str = '') -> None:
        """记录最新文案。可在任意线程高频调用，代价为一次赋值。"""
        if text:
            self._text = str(text)
        self._dirty = True
        if self._auto_start and not self._timer.isActive():
            self._timer.start()

    # ------------------------------------------------------------------
    # 定时器
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def set_auto_start(self, enabled: bool) -> None:
        """关闭后需显式调用 ``start()``，供测试与批处理场景使用。"""
        self._auto_start = bool(enabled)

    def _on_timeout(self) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # 刷新
    # ------------------------------------------------------------------
    def flush(self) -> bool:
        """把缓存推给 UI。无新数据时不做任何事，返回是否真的刷新过。"""
        if not self._dirty:
            return False
        self._dirty = False
        self.flushed.emit(self._text)
        return True

    def flush_now(self, text: str = '') -> bool:
        """终态刷新：绕过节流。完成 / 失败路径都应调用。"""
        if text:
            self.report(text)
        self._dirty = True
        refreshed = self.flush()
        self.stop()
        return refreshed

    def reset(self) -> None:
        """复用前清空状态（例如同一弹窗承载连续任务）。"""
        self.stop()
        self._text = ''
        self._dirty = False
