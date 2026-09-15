"""Quality worker for facade quality computation.

FIX: Added stable facade_no passing and comprehensive error handling.
FIX: Removed facade_id duplication, unified to facade_no.
"""
from __future__ import annotations

import traceback
import gc
from PySide6.QtCore import QObject, Signal, QRunnable

# 注意：此处**不能**在模块级导入 services.*。services 包初始化会反向导入
# utils.workers（services/__init__ -> project_overview -> utils.workers），
# 模块级导入会形成循环，报 "partially initialized module" 导入错误。
# 因此 ModelExportCancelled 改为在真正需要它的 run() 内惰性导入。


class QualityCancelled(Exception):
    """内部信号：进度回调检测到中止请求时抛出，用于提前结束质量计算。"""


class LoadWorkerSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)


class PointCloudLoadWorker(QRunnable):
    """Run the legacy file pipeline off the GUI thread.

    The callable must not update widgets.  Its result is committed by the
    queued GUI callback only when the project generation is still current.
    """
    def __init__(self, operation, *, memory_budget_bytes=0):
        super().__init__()
        self.operation = operation
        self.memory_budget_bytes = int(memory_budget_bytes or 0)
        self.signals = LoadWorkerSignals()
        self._cancelled = False
        self.setAutoDelete(False)

    def cancel(self):
        self._cancelled = True

    def check_cancelled(self):
        if self._cancelled:
            raise RuntimeError('点云加载任务已取消')

    def is_cancelled(self):
        return self._cancelled

    def report_progress(self, percent, text=''):
        """供 operation 内部各阶段调用，转成跨线程进度信号。"""
        if self._cancelled:
            raise RuntimeError('点云加载任务已取消')
        try:
            self.signals.progress.emit(int(percent), str(text or ''))
        except Exception:
            pass

    def run(self):
        try:
            if self._cancelled:
                # 被中止时不发 finished/failed：GUI 侧 dispose 已把
                # 进度窗落到"已中止"，再发终态会造成二次收尾。
                return
            result = self.operation(self)
            if self._cancelled:
                return
            # 终态由 load_finished → task_progress.finish() 统一关闭，
            # 避免与 finish 路径重复收尾。
            self.signals.finished.emit(result)
        except Exception as exc:
            if self._cancelled:
                return
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            # Release temporary numpy/Open3D objects as soon as the worker ends.
            gc.collect()


class QualityWorkerSignals(QObject):
    """Signals for quality worker.

    finished: (facade_dict, quality_result) - always emitted
    failed: (facade_dict, error_message) - emitted on exception
    progress: (percent, text) - throttled by the GUI-side dialog
    """
    finished = Signal(object, object)
    failed = Signal(object, str)
    progress = Signal(int, str)


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
        # 中止标志：GUI 线程置位，工作线程在进度检查点读取（与模型导出 worker
        # 一致，bool 赋值在 CPython 下原子可见，无需额外同步对象）。
        self._cancelled = False
        # Keep the QObject-owned signals alive until the GUI has consumed the
        # queued result. MainWindow releases this worker explicitly.
        self.setAutoDelete(False)

    def cancel(self):
        """请求中止质量评估；算法在下一个进度检查点退出。"""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _report_progress(self, percent, text=''):
        """跨线程进度回传：只 emit，绝不触碰控件。

        质量评估的窗口/条带总数是已知量，因此这里传出的就是真实百分比。
        该回调同时充当中止检查点——算法每隔若干窗口上报一次，
        借此抛出 QualityCancelled 让计算尽早收束，避免任务被中止后
        后台仍在空转。文案保持极简，阶段细节由状态栏承载。
        """
        if self._cancelled:
            raise QualityCancelled('质量评估已取消')
        try:
            self.signals.progress.emit(int(percent), '任务处理中')
        except Exception:
            pass

    def run(self):
        facade_no = self.facade_no

        print(f'[PCFD] quality.worker_start facade_no={facade_no} cloud={self.cloud_name}', flush=True)

        try:
            # 算法层通过关键字回调上报进度；未声明该形参的实现会走 TypeError
            # 兜底分支，保持对旧签名的兼容。
            kwargs = dict(self.kwargs)
            kwargs.setdefault('progress_cb', self._report_progress)
            try:
                result = self.facade_service.compute_quality(
                    self.cloud_name, self.facade, **kwargs)
            except TypeError as exc:
                if 'progress_cb' not in str(exc):
                    raise
                kwargs.pop('progress_cb', None)
                result = self.facade_service.compute_quality(
                    self.cloud_name, self.facade, **kwargs)

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

            if self._cancelled:
                # 中止后不再投递结果：进度窗已由取消处理器收尾，继续 emit 会让
                # 被取消的任务当作成功结果写入质量缓存与报告。
                return
            # FIX: Always emit finished, even for error results
            # The UI will handle ok=False appropriately
            self.signals.finished.emit(self.facade, result)

        except QualityCancelled:
            # 中止不是失败：静默返回，终态由取消处理器统一给出。
            print(f'[PCFD] quality.worker_cancelled facade_no={facade_no}', flush=True)
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


class ModelExportWorkerSignals(QObject):
    """Signals for the station source-cloud PLY export worker."""
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)


class ModelExportWorker(QRunnable):
    """在后台线程执行站点源点云 voxel 下采样并导出 PLY。

    ``service.export_station_ply`` 本身不依赖 Qt，进度经 ``progress_cb``
    回调原样转成信号发给 GUI 线程；GUI 侧再做 5 秒节流刷新。
    """

    def __init__(self, service, station_id, target_path, *,
                 voxel_size=0.2, project_uuid=None):
        super().__init__()
        self.service = service
        self.station_id = station_id
        self.target_path = target_path
        self.voxel_size = voxel_size
        self.project_uuid = project_uuid
        self.signals = ModelExportWorkerSignals()
        # 中止标志：GUI 线程置位，工作线程在阶段边界读取。
        # 用 bool 而非 threading.Event 是刻意的——只需一次可见性，
        # CPython 的 GIL 已保证赋值原子，且避免引入额外同步对象。
        self._cancelled = False
        self.setAutoDelete(False)

    def cancel(self):
        """请求中止；线程可能在下次检查点才真正退出。"""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _report_progress(self, percent, text=''):
        try:
            self.signals.progress.emit(int(percent), str(text or ''))
        except Exception:
            pass

    def run(self):
        # 惰性导入，规避与 services 包的循环导入（详见模块头注释）。
        from services.model_export_service import ModelExportCancelled
        try:
            result = self.service.export_station_ply(
                self.station_id, self.target_path,
                voxel_size=self.voxel_size,
                project_uuid=self.project_uuid,
                progress_cb=self._report_progress,
                cancel_check=self.is_cancelled)
            if self._cancelled:
                # 检查点之间的极窄竞态：已在写盘后又被取消。
                # 不发 finished，避免 GUI 弹"导出成功"。
                self.signals.failed.emit('导出已取消')
                return
            self.signals.finished.emit(result)
        except ModelExportCancelled:
            # 取消不是失败：用 failed 通道回到 GUI，由控制器区分文案。
            self.signals.failed.emit('导出已取消')
        except Exception as exc:
            traceback.print_exc()
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')


class RegistrationWorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class RegistrationWorker(QRunnable):
    """Run the expensive raw-point ICP pipeline outside the Qt GUI thread."""

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = RegistrationWorkerSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            self.signals.finished.emit(self.operation())
        except Exception as exc:
            traceback.print_exc()
            self.signals.failed.emit(f'{type(exc).__name__}: {exc}')