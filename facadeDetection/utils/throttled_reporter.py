"""耗时任务的进度节流合并器。

设计要点（对应"进度数据每 5 秒合并一次、但 UI 保持流畅"的需求）：

1. 节流点放在 GUI 侧，而不是信号链路。后台 worker 照常高频 emit，
   跨线程信号经 QueuedConnection 投递到 GUI 线程槽后，只调用
   ``report()``——一次元组赋值，代价可忽略。
2. ``QTimer`` 每 ``interval_ms`` 触发一次，仅当缓存被写过（dirty）
   才发出一次 ``flushed``，于是重绘频率恒定在 5 秒一次，界面不会
   被高频 setValue 拖慢，事件循环始终空闲。
3. 终态必须刷：``flush_now()`` 绕过节流，避免进度条停在 99% 的观感
   问题（完成 / 失败 / 取消路径都要调用）。

本模块不依赖任何业务代码，可独立测试。
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal


DEFAULT_INTERVAL_MS = 5000


class ThrottledReporter(QObject):
    """把高频进度回报合并为固定间隔的低频 UI 刷新。"""

    #: (percent, text)
    flushed = Signal(int, str)

    def __init__(self, interval_ms: int = DEFAULT_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self._interval_ms = max(1, int(interval_ms))
        self._percent: int = 0
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
        """缓存中是否有尚未刷新的进度。"""
        return self._dirty

    @property
    def snapshot(self) -> tuple[int, str]:
        """最近一次写入缓存的进度值（不触发刷新）。"""
        return self._percent, self._text

    def is_active(self) -> bool:
        return self._timer.isActive()

    # ------------------------------------------------------------------
    # 写入：仅更新缓存，绝不触碰 UI
    # ------------------------------------------------------------------
    def report(self, percent: Optional[int], text: str = '') -> None:
        """记录最新进度。可在任意线程高频调用，代价为一次赋值。"""
        if percent is not None:
            try:
                self._percent = max(0, min(100, int(percent)))
            except (TypeError, ValueError):
                pass
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
        self.flushed.emit(self._percent, self._text)
        return True

    def flush_now(self, percent: Optional[int] = None, text: str = '') -> bool:
        """终态刷新：绕过节流。完成 / 失败 / 取消路径都应调用。"""
        if percent is not None or text:
            self.report(percent, text)
        self._dirty = True
        refreshed = self.flush()
        self.stop()
        return refreshed

    def reset(self) -> None:
        """复用前清空状态（例如同一弹窗承载连续任务）。"""
        self.stop()
        self._percent = 0
        self._text = ''
        self._dirty = False
