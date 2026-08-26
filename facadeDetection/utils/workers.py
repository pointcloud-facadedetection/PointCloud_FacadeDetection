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


class AutoMatchSignals(QObject):
    finished = Signal(object)
    failed = Signal(object)


class AutoMatchWorker(QRunnable):
    """在后台线程跑自动 2D-3D 匹配，避免卡住界面。"""

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs
        self.signals = AutoMatchSignals()
        self.setAutoDelete(True)

    def run(self):
        started = time.perf_counter()
        facade_id = (self.kwargs.get('facade') or {}).get('id')
        print(f"[PCFD] auto_match.worker_start facade={facade_id}", flush=True)
        try:
            from algorithms.auto_photo_pointcloud_matching.matcher import (
                match_photo_to_facade,
            )

            result = match_photo_to_facade(**self.kwargs)
            print(
                f"[PCFD] auto_match.worker_finish facade={facade_id} "
                f"seconds={time.perf_counter()-started:.2f}",
                flush=True,
            )
            self.signals.finished.emit(result)
        except Exception as exc:
            print(
                f"[PCFD] auto_match.worker_error facade={facade_id} "
                f"seconds={time.perf_counter()-started:.2f} error={exc!r}",
                flush=True,
            )
            self.signals.failed.emit(exc)


class MapBackSignals(QObject):
    finished = Signal(object)
    failed = Signal(object)


class MapBackWorker(QRunnable):
    """后台把摆正后的 2D 照片映射到扫描位姿点云视图。"""

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs
        self.signals = MapBackSignals()
        self.setAutoDelete(True)

    def run(self):
        started = time.perf_counter()
        print('[PCFD] map_back.worker_start', flush=True)
        try:
            from algorithms.auto_photo_pointcloud_matching.scan_view_mapper import (
                map_photo_onto_scan_view,
            )

            result = map_photo_onto_scan_view(**self.kwargs)
            print(
                f'[PCFD] map_back.worker_finish seconds={time.perf_counter()-started:.2f}',
                flush=True,
            )
            self.signals.finished.emit(result)
        except Exception as exc:
            print(
                f'[PCFD] map_back.worker_error seconds={time.perf_counter()-started:.2f} error={exc!r}',
                flush=True,
            )
            self.signals.failed.emit(exc)
