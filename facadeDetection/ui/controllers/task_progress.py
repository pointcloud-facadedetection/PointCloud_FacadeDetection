"""统一模态进度控制器。"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QObject

from ui.dialogs.task_progress_dialog import TaskProgressDialog

#: 任务键常量：集中定义避免各处散落字符串字面量。
TASK_UPLOAD = 'upload'
TASK_LOAD = 'load'
TASK_DENOISE = 'denoise'
TASK_REGION = 'region'
TASK_DETECTION = 'detection'
TASK_QUALITY = 'quality'
TASK_QUALITY_BATCH = 'quality_batch'
TASK_MODEL_EXPORT = 'model_export'
TASK_PAGE_SWITCH = 'page_switch'
TASK_REPORT = 'report'


class TaskProgressController(QObject):
    """按任务键管理模态进度弹窗（纯忙碌条）的生命周期。"""

    def __init__(self, parent=None, *, interval_ms: Optional[int] = None):
        super().__init__(parent)
        self._parent = parent
        self._interval_ms = interval_ms
        self._dialogs: Dict[str, TaskProgressDialog] = {}

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def dialog(self, task_key: str) -> Optional[TaskProgressDialog]:
        return self._dialogs.get(task_key)

    def is_active(self, task_key: str) -> bool:
        dialog = self._dialogs.get(task_key)
        return bool(dialog is not None and dialog.isVisible()
                    and not dialog.is_finished)

    def active_keys(self) -> List[str]:
        return [key for key in self._dialogs if self.is_active(key)]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def begin(self, task_key: str, title: str, message: str = '',
              **_ignored) -> TaskProgressDialog:
        """创建/复用一个任务弹窗并立即以模态方式显示。

        ``**_ignored`` 兼容历史调用方传入的 determinate/cancellable，
        忙碌条模式下这两个概念已移除。
        """
        dialog = self._dialogs.get(task_key)
        if dialog is None:
            kwargs = {}
            if self._interval_ms is not None:
                kwargs['interval_ms'] = self._interval_ms
            dialog = TaskProgressDialog(self._parent, **kwargs)
            self._dialogs[task_key] = dialog

        dialog.start(title, message or None)
        # 非阻塞模态：show() 而非 exec()，事件循环继续跑，后台信号照常投递。
        dialog.show()
        dialog.raise_()
        return dialog

    def report(self, task_key: str, percent=None, text: str = '') -> None:
        """节流写入阶段文案；百分比参数被忽略（忙碌条无数值）。

        不定量护栏：弹窗不存在或已终态时不报错，避免后台线程在 GUI 关闭后仍然写进度。
        """
        dialog = self._dialogs.get(task_key)
        if dialog is None or dialog.is_finished:
            return
        dialog.report_progress(text=text)

    def finish(self, task_key: str, success: bool = True,
               message: str = '') -> None:
        """终态收尾并关闭弹窗（幂等）。"""
        dialog = self._dialogs.get(task_key)
        if dialog is None:
            return
        dialog.finish(success, message)
        if dialog.is_finished and dialog.isVisible():
            dialog.hide()

    def close_all(self) -> None:
        """窗口关闭/项目切换时的统一兜底：所有弹窗落终态并收掉。"""
        for task_key in list(self._dialogs):
            try:
                self.finish(task_key, success=False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 便捷包装（供 GUI 槽直接连接）
    # ------------------------------------------------------------------
    def begin_slot(self, task_key: str, title: str, message: str = ''):
        def _slot(*_args, **_kwargs):
            self.begin(task_key, title, message)
        return _slot

    def connect_progress(self, task_key: str, signal) -> None:
        """把 worker 的 ``progress(int, str)`` 信号接到节流写入（忽略百分比）。"""
        signal.connect(lambda percent, text='': self.report(
            task_key, percent, text))

    def connect_progress_slot(self, task_key: str):
        """返回可直接 connect 到 ``Signal(int, str)`` 的槽函数。"""
        def _slot(percent, text=''):
            self.report(task_key, percent, text)
        return _slot
