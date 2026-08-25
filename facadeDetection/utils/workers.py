"""Quality worker for facade quality computation.

FIX: Added stable facade_no passing and comprehensive error handling.
FIX: Removed facade_id duplication, unified to facade_no.
"""
from __future__ import annotations

import traceback
from PySide6.QtCore import QObject, Signal, QRunnable


class QualityWorkerSignals(QObject):
    """Signals for quality worker.

    finished: (facade_dict, quality_result) - always emitted
    failed: (facade_dict, error_message) - emitted on exception
    """
    finished = Signal(object, object)
    failed = Signal(object, str)


class QualityWorker(QRunnable):
    """Worker for facade quality computation in QThreadPool."""

    def __init__(self, facade_service, cloud_name, facade, kwargs):
        super().__init__()
        self.facade_service = facade_service
        self.cloud_name = cloud_name
        # FIX: Ensure facade dict has stable facade_no; remove facade_id duplication
        self.facade = dict(facade)
        self.facade_no = int(self.facade.get('display_no', self.facade.get('id', 0)))
        self.kwargs = kwargs
        self.signals = QualityWorkerSignals()
        # Keep the QObject-owned signals alive until the GUI has consumed the
        # queued result. MainWindow releases this worker explicitly.
        self.setAutoDelete(False)

    def run(self):
        facade_no = self.facade_no

        print(f'[PCFD] quality.worker_start facade_no={facade_no} cloud={self.cloud_name}', flush=True)

        try:
            # Call the service
            result = self.facade_service.compute_quality(
                self.cloud_name, self.facade, **self.kwargs)

            # FIX: Ensure result is always a dict with facade info
            if result is None:
                result = {
                    'ok': False,
                    'reason': 'service_returned_none',
                    'message': '质量计算服务返回空结果',
                    'facade_no': facade_no,
                }
            elif not isinstance(result, dict):
                result = {
                    'ok': False,
                    'reason': 'invalid_result_type',
                    'message': f'质量计算返回异常类型: {type(result)}',
                    'facade_no': facade_no,
                }
            else:
                # Ensure facade info is in result for diagnostics
                result['facade_no'] = facade_no

            print(f'[PCFD] quality.worker_done facade_no={facade_no} ok={result.get("ok", False)}', flush=True)

            # FIX: Always emit finished, even for error results
            # The UI will handle ok=False appropriately
            self.signals.finished.emit(self.facade, result)

        except Exception as e:
            error_msg = f'{type(e).__name__}: {e}'
            print(f'[PCFD] quality.worker_error facade_no={facade_no} error={error_msg}', flush=True)
            traceback.print_exc()

            # FIX: Emit failed signal with error result dict
            error_result = {
                'ok': False,
                'reason': 'worker_exception',
                'message': f'质量计算异常: {error_msg}',
                'error': traceback.format_exc(),
                'facade_no': facade_no,
            }
            self.signals.finished.emit(self.facade, error_result)