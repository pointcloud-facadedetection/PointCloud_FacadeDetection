"""通用 Qt 后台工作线程。"""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, QRunnable, Signal


class QualitySignals(QObject):
    finished = Signal(object, object)  # facade, result
    failed = Signal(object, object)    # facade, exception


class QualityWorker(QRunnable):
    """Run CPU-only quality calculation outside the Qt GUI/Open3D thread."""

    def __init__(self, service, cloud_name, facade, kwargs):
        super().__init__()
        self.service = service
        self.cloud_name = cloud_name
        self.facade = facade
        self.kwargs = kwargs
        self.signals = QualitySignals()
        self.setAutoDelete(True)

    def run(self):
        started = time.perf_counter()
        facade_id = self.facade.get('id')
        print(f"[PCFD] quality.worker_start facade={facade_id} cloud={self.cloud_name}", flush=True)
        try:
            result = self.service.compute_quality(self.cloud_name, self.facade, **self.kwargs)
            print(f"[PCFD] quality.worker_finish facade={facade_id} seconds={time.perf_counter()-started:.2f}", flush=True)
            self.signals.finished.emit(self.facade, result)
        except Exception as exc:
            print(f"[PCFD] quality.worker_error facade={facade_id} seconds={time.perf_counter()-started:.2f} error={exc!r}", flush=True)
            self.signals.failed.emit(self.facade, exc)
