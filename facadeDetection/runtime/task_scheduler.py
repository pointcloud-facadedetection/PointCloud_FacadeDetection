from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal

from .task_context import TaskContext, TaskCancelledError


class _Signals(QObject):
    succeeded = Signal(object, object)
    failed = Signal(object, str)
    cancelled = Signal(object)


class _Task(QRunnable):
    def __init__(self, context: TaskContext, operation: Callable, signals: _Signals):
        super().__init__()
        self.context = context
        self.operation = operation
        self.signals = signals
        self.setAutoDelete(False)

    def run(self):
        try:
            self.context.check_cancelled()
            result = self.operation(self.context)
            self.context.check_cancelled()
            self.signals.succeeded.emit(self.context, result)
        except TaskCancelledError:
            self.signals.cancelled.emit(self.context)
        except Exception as exc:
            self.signals.failed.emit(self.context, f"{type(exc).__name__}: {exc}")


@dataclass
class _Running:
    context: TaskContext
    cancel_event: Event
    task: _Task


class RuntimeTaskScheduler(QObject):
    """Small Qt-native scheduler with latest-request-wins semantics per kind."""

    def __init__(self, parent=None, max_thread_count: int = 1):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, int(max_thread_count)))
        self._lock = Lock()
        self._tokens: dict[str, int] = {}
        self._running: dict[str, _Running] = {}

    def submit(self, kind: str, project_uuid: str | None,
               project_generation: int, operation: Callable,
               dataset_id: str | None = None,
               dataset_revision: str | None = None,
               on_success: Callable | None = None,
               on_error: Callable | None = None,
               on_cancelled: Callable | None = None):
        kind = str(kind)
        with self._lock:
            old = self._running.get(kind)
            if old is not None:
                old.cancel_event.set()
            token = self._tokens.get(kind, 0) + 1
            self._tokens[kind] = token
            cancel_event = Event()
            context = TaskContext(kind, project_uuid, int(project_generation),
                                  dataset_id, dataset_revision, token, cancel_event)
            signals = _Signals(self)
            task = _Task(context, operation, signals)
            self._running[kind] = _Running(context, cancel_event, task)
        signals.succeeded.connect(self._on_succeeded, Qt.QueuedConnection)
        signals.failed.connect(self._on_failed, Qt.QueuedConnection)
        signals.cancelled.connect(self._on_cancelled, Qt.QueuedConnection)
        if on_success is not None:
            signals.succeeded.connect(on_success, Qt.QueuedConnection)
        if on_error is not None:
            signals.failed.connect(on_error, Qt.QueuedConnection)
        if on_cancelled is not None:
            signals.cancelled.connect(on_cancelled, Qt.QueuedConnection)
        self.pool.start(task)
        return context

    def cancel(self, kind: str) -> None:
        with self._lock:
            running = self._running.get(str(kind))
            if running is not None:
                running.cancel_event.set()

    def cancel_all(self) -> None:
        with self._lock:
            for running in self._running.values():
                running.cancel_event.set()

    def is_current(self, context: TaskContext) -> bool:
        with self._lock:
            # 内部完成插槽在客户端插槽之前被连接，并且会首先移除正在运行的条目。
            # 因此，在队列完成回调期间，令牌状态是权威的标识。
            return (self._tokens.get(context.kind) == context.operation_token
                    and not (context.cancel_event and context.cancel_event.is_set()))

    def _finish(self, context):
        with self._lock:
            running = self._running.get(context.kind)
            if running is not None and running.context.operation_token == context.operation_token:
                self._running.pop(context.kind, None)

    def _on_succeeded(self, context, result):
        self._finish(context)

    def _on_failed(self, context, message):
        self._finish(context)

    def _on_cancelled(self, context):
        self._finish(context)
