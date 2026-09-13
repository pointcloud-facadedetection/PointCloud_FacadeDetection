"""Unified serial import queue for FLS/E57/PLY/Photo resources.

All import operations (conversion, registration, viewport commit) are serialized
through a single QThreadPool with maxThreadCount=1 to prevent state races.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal

from models.enums import FileKind
from services.dal.file_repo import FileRepo
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.e57_cache_service import ensure_e57_cache
from services import proxy_cache
from utils.logging_utils import log_event, trace
from config.storage import Storage


class _ImportSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)


@dataclass
class ImportTaskPayload:
    kind: str  # 'ply' | 'e57' | 'fls' | 'photo'
    paths: list[str]
    project_uuid: str
    mode: str  # 'create' | 'edit'
    # Callbacks (invoked via queued connection back to GUI thread)
    on_progress: Optional[Callable[[int, str], None]] = None
    on_finished: Optional[Callable[[object], None]] = None
    on_failed: Optional[Callable[[str], None]] = None


class _ImportWorker(QRunnable):
    def __init__(self, payload: ImportTaskPayload, execute_fn: Callable,
                 signals: _ImportSignals):
        super().__init__()
        self.payload = payload
        self.execute_fn = execute_fn
        self.signals = signals
        self._cancelled = False
        self.setAutoDelete(False)

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.signals.progress.emit(5, '正在准备导入任务...')
            result = self.execute_fn(self.payload, self)
            if not self._cancelled:
                self.signals.progress.emit(100, '导入完成')
                self.signals.finished.emit(result)
        except Exception as exc:
            if not self._cancelled:
                self.signals.failed.emit(f'{type(exc).__name__}: {exc}')

    def is_cancelled(self) -> bool:
        return self._cancelled

    def check_cancelled(self):
        if self._cancelled:
            raise RuntimeError('导入任务已取消')


class ImportTaskQueue(QObject):
    """Serial import queue: one task at a time, latest-wins per project."""

    def __init__(self, file_service, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.file_service = file_service
        self._current_worker: Optional[_ImportWorker] = None
        self._current_payload: Optional[ImportTaskPayload] = None

    def submit(self, payload: ImportTaskPayload):
        """Enqueue an import task; cancel any running task for the same project."""
        self.cancel(payload.project_uuid)
        signals = _ImportSignals(self)
        worker = _ImportWorker(payload, self._execute, signals)
        self._current_worker = worker
        self._current_payload = payload

        # Wire callbacks through queued connections so they run on GUI thread
        if payload.on_progress is not None:
            signals.progress.connect(payload.on_progress, Qt.QueuedConnection)
        if payload.on_finished is not None:
            signals.finished.connect(payload.on_finished, Qt.QueuedConnection)
        if payload.on_failed is not None:
            signals.failed.connect(payload.on_failed, Qt.QueuedConnection)

        self.pool.start(worker)

    def cancel(self, project_uuid: str):
        if (self._current_payload is not None
                and self._current_payload.project_uuid == project_uuid
                and self._current_worker is not None):
            self._current_worker.cancel()

    def _execute(self, payload: ImportTaskPayload, worker: _ImportWorker) -> dict:
        """Core import logic executed in background thread."""
        project_uuid = payload.project_uuid
        mode = payload.mode
        results = {
            'project_uuid': project_uuid,
            'mode': mode,
            'imported': [],
            'failed': [],
            'prepared': [],
        }

        # ------------------------------------------------------------------
        # Phase 1: pre-sync existing assets so we can compute delta
        # ------------------------------------------------------------------
        PointCloudStationRepo.sync_assets(project_uuid)

        # ------------------------------------------------------------------
        # Phase 2: classify inputs
        # ------------------------------------------------------------------
        fls_dirs = []
        e57_paths = []
        ply_paths = []
        photo_paths = []

        for p in payload.paths:
            suffix = Path(p).suffix.lower()
            if suffix == '.e57':
                e57_paths.append(p)
            elif suffix == '.ply':
                ply_paths.append(p)
            elif Path(p).is_dir():
                fls_dirs.append(p)
            else:
                photo_paths.append(p)

        total_steps = len(fls_dirs) + len(e57_paths) + len(ply_paths) + len(photo_paths)
        step = 0

        # ------------------------------------------------------------------
        # Phase 3: FLS directories (convert → PLY → import)
        # ------------------------------------------------------------------
        for d in fls_dirs:
            worker.check_cancelled()
            step += 1
            worker.signals.progress.emit(
                int(10 + 80 * step / max(total_steps, 1)),
                f'正在导入 FLS 目录: {Path(d).name}')
            try:
                res = self.file_service.import_fls_directory(
                    d, project_uuid,
                    progress_cb=lambda done, total, name: worker.signals.progress.emit(
                        int(10 + 80 * (step - 1 + done / max(total_steps, 1)) / max(total_steps, 1)),
                        f'正在转换站点 {name} ({done}/{total})'))
                if res.get('success'):
                    results['imported'].extend(res.get('ply_paths', []))
                    results['prepared'].extend(res.get('metadata', []))
                else:
                    results['failed'].append((d, res.get('message', 'FLS导入失败')))
            except Exception as exc:
                results['failed'].append((d, str(exc)))
                trace('import_queue.fls_failed', path=d, error=exc)

        # ------------------------------------------------------------------
        # Phase 4: E57 files (convert to cache PLY → import)
        # ------------------------------------------------------------------
        for p in e57_paths:
            worker.check_cancelled()
            step += 1
            worker.signals.progress.emit(
                int(10 + 80 * step / max(total_steps, 1)),
                f'正在导入 E57: {Path(p).name}')
            try:
                asset = FileRepo.import_file(
                    project_uuid, p, FileKind.raw_pointcloud,
                    copy_into_project=False)
                if asset is not None:
                    cache_path = ensure_e57_cache(project_uuid, asset)
                    # Now treat the cache PLY as a regular PLY for rendering
                    prep = self.file_service.prepare_upload(
                        project_uuid, str(cache_path),
                        copy_into_project=False)
                    results['prepared'].append(prep)
                    results['imported'].append(str(cache_path))
                else:
                    results['failed'].append((p, 'E57资产注册失败'))
            except Exception as exc:
                results['failed'].append((p, str(exc)))
                trace('import_queue.e57_failed', path=p, error=exc)

        # ------------------------------------------------------------------
        # Phase 5: Native PLY files (direct import)
        # ------------------------------------------------------------------
        for p in ply_paths:
            worker.check_cancelled()
            step += 1
            worker.signals.progress.emit(
                int(10 + 80 * step / max(total_steps, 1)),
                f'正在导入 PLY: {Path(p).name}')
            try:
                prep = self.file_service.prepare_upload(
                    project_uuid, p, copy_into_project=False)
                results['prepared'].append(prep)
                results['imported'].append(p)
            except Exception as exc:
                results['failed'].append((p, str(exc)))
                trace('import_queue.ply_failed', path=p, error=exc)

        # ------------------------------------------------------------------
        # Phase 6: Photos (metadata only)
        # ------------------------------------------------------------------
        for p in photo_paths:
            worker.check_cancelled()
            step += 1
            try:
                FileRepo.import_file(
                    project_uuid, p, FileKind.raw_image,
                    copy_into_project=False)
                results['imported'].append(p)
            except Exception as exc:
                results['failed'].append((p, str(exc)))

        # ------------------------------------------------------------------
        # Phase 7: sync station projections
        # ------------------------------------------------------------------
        worker.check_cancelled()
        worker.signals.progress.emit(95, '正在同步站点信息...')
        PointCloudStationRepo.sync_assets(project_uuid)

        log_event(project_uuid, 'import_queue.completed',
                  imported=len(results['imported']),
                  failed=len(results['failed']))
        return results