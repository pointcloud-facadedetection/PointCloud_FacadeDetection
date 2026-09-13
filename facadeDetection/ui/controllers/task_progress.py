"""统一模态进度控制器。"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

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
TASK_REPORT = 'report'


class TaskProgressController(QObject):
    """按任务键管理模态进度弹窗的生命周期。"""

    #: 某任务的取消请求已下发（含未注册回调的空跑情形），供 UI 侧收尾徽标。
    task_cancelled = Signal(str)

    def __init__(self, parent=None, *, interval_ms: Optional[int] = None):
        super().__init__(parent)
        self._parent = parent
        self._interval_ms = interval_ms
        self._dialogs: Dict[str, TaskProgressDialog] = {}
        self._cancel_handlers: Dict[str, List[Callable[[], None]]] = {}
        self._wired_dialogs: set = set()

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
    # 中止回调注册
    # ------------------------------------------------------------------
    def set_cancel_handler(self, task_key: str, handler) -> None:
        """注册"关闭即中止"回调；``handler=None`` 表示清空该任务的全部回调。

        同 key 重复注册会**替换**既有回调，保持与本方法历史语义一致
        （调用方通常在启动任务前一次性注册）。若一次任务需要多个中止落点，
        请改用 :meth:`add_cancel_handler` 逐个追加。
        """
        if handler is None:
            self._cancel_handlers.pop(task_key, None)
            return
        self._cancel_handlers[task_key] = [handler]

    def add_cancel_handler(self, task_key: str, handler) -> None:
        """追加一个中止落点；同一 key 可登记多个（去重后按注册顺序调用）。"""
        if handler is None:
            return
        handlers = self._cancel_handlers.setdefault(task_key, [])
        if handler not in handlers:
            handlers.append(handler)

    def clear_cancel_handlers(self, task_key: str) -> None:
        self._cancel_handlers.pop(task_key, None)

    def cancel_task(self, task_key: str) -> None:
        """下发取消：先落地弹窗终态，再通知全部业务方取消后台进程。

        顺序不可颠倒——即使某个 handler 抛异常，弹窗也不会悬在屏幕上。
        """
        handler_list = list(self._cancel_handlers.get(task_key) or [])
        self.finish(task_key, success=False, message='已取消')
        for handler in handler_list:
            try:
                handler()
            except Exception as exc:
                print(f'[PCFD] task_progress.cancel_handler_failed '
                      f'task={task_key} error={exc!r}', flush=True)
        self.task_cancelled.emit(task_key)

    #: 兼容旧调用名：语义与 :meth:`cancel_task` 完全一致。
    def _on_cancel_requested(self, task_key: str) -> None:
        self.cancel_task(task_key)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def begin(self, task_key: str, title: str, message: str = '',
              *, determinate: bool = True,
              cancellable: bool = False) -> TaskProgressDialog:
        """创建/复用一个任务弹窗并立即以模态方式显示。"""
        dialog = self._dialogs.get(task_key)
        if dialog is None:
            kwargs = {'cancellable': cancellable}
            if self._interval_ms is not None:
                kwargs['interval_ms'] = self._interval_ms
            dialog = TaskProgressDialog(self._parent, **kwargs)
            self._dialogs[task_key] = dialog

        # 取消/关闭信号每个弹窗实例只绑定一次；弹窗在同一 key 下复用，
        # 重复 connect 会导致一次点击触发多次中止回调。
        if id(dialog) not in self._wired_dialogs:
            dialog.cancel_requested.connect(
                lambda key=task_key: self.cancel_task(key))
            self._wired_dialogs.add(id(dialog))
            # 弹窗销毁后回收登记，避免 id 复用造成误判。
            dialog.destroyed.connect(
                lambda *_args, key=id(dialog): self._wired_dialogs.discard(key))

        dialog.set_cancellable(cancellable)
        dialog.start(title, message or None, determinate=determinate)

        if cancellable and not self._cancel_handlers.get(task_key):
            # 开发期护栏：可取消却没有取消落点，等于"点了没反应"。
            print(f'[PCFD] task_progress.no_cancel_handler task={task_key}',
                  flush=True)
        # 非阻塞模态：show() 而非 exec()，事件循环继续跑，后台信号照常投递。
        dialog.show()
        dialog.raise_()
        return dialog

    def report(self, task_key: str, percent: Optional[int],
               text: str = '') -> None:
        """节流写入进度；``percent=None`` 时只更新阶段文案。

        不定量护栏：弹窗不存在或已终态时不报错，避免后台线程在 GUI 关闭后仍然写进度。
        """
        dialog = self._dialogs.get(task_key)
        if dialog is None or dialog.is_finished:
            return
        # 不定量阶段：丢弃百分比，仅更新文案，保持滚动动画。
        if dialog._bar.maximum() == 0:
            dialog.report_progress(-1, text)
            return
        dialog.report_progress(percent, text)

    def progress_callback(self, task_key: str):
        """返回可直接交给 worker/service 的 ``progress_cb`` 闭包。

        后台线程调用它只会写缓存（一次元组赋值），真正的 UI 刷新仍由
        GUI 线程的定时器按 5 秒节拍触发。
        """
        def _callback(percent, text=''):
            self.report(task_key, percent, text)
        return _callback

    def finish(self, task_key: str, success: bool = True,
               message: str = '') -> None:
        """终态收尾：强制刷进度条并关闭弹窗（幂等）。

        成功时进度条落到 100%，失败/取消时保留当前值；三条路径都会关闭
        弹窗，且终态关闭不会再触发取消回调。
        """
        dialog = self._dialogs.get(task_key)
        if dialog is None:
            return
        dialog.finish(success, message)
        if dialog.is_finished and dialog.isVisible():
            dialog.hide()

    def cancel(self, task_key: str) -> None:
        """外部（取消按钮以外的路径）强制收尾。"""
        self.finish(task_key, success=False, message='已取消')

    def close_all(self) -> None:
        """窗口关闭/项目切换时的统一兜底：所有弹窗落终态并收掉。"""
        for task_key in list(self._dialogs):
            try:
                self.finish(task_key, success=False, message='已取消')
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 便捷包装（供 GUI 槽直接连接）
    # ------------------------------------------------------------------
    def begin_slot(self, task_key: str, title: str, message: str = '',
                   *, determinate: bool = True,
                   cancellable: bool = False):
        def _slot(*_args, **_kwargs):
            self.begin(task_key, title, message,
                       determinate=determinate, cancellable=cancellable)
        return _slot

    def connect_progress(self, task_key: str, signal) -> None:
        """把 worker 的 ``progress(int, str)`` 信号接到节流写入。"""
        signal.connect(lambda percent, text='': self.report(
            task_key, percent, text))

    def connect_progress_slot(self, task_key: str):
        """返回可直接 connect 到 ``Signal(int, str)`` 的槽函数。

        与 :meth:`connect_progress` 的区别是"先拿到槽再连接"，便于调用方
        自行决定连接时机与断开方式。
        """
        def _slot(percent, text=''):
            self.report(task_key, percent, text)
        return _slot