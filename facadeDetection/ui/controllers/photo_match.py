"""照片-点云匹配编排控制器。

把【检测复核】页"图片匹配"按钮背后的全部编排——后台 2D-3D 配准、
主线程结果/异常回调、进度窗收尾、状态栏/弹窗 UI 反馈——收拢到本控制器，
与 RegistrationController / FacadeQualityController / ModelExportController
保持同一套结构。UI 侧只保留一次按钮点击的入口。
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from .task_progress import TASK_PHOTO_MATCH, TaskProgressController


class PhotoMatchController(QObject):
    """图片匹配后台任务与主线程的通信桥与编排。

    UI 只通过 ``start(facade)`` 触发；结果、异常、状态栏文案、警告
    弹窗、报告预览刷新分别以信号形式向主窗口回报，避免控制器直接
    持有任何 widget。
    """

    #: 状态栏文案（message, timeout_ms）
    status_message = Signal(str, int)
    #: 需要 QMessageBox.warning 的文案
    warning_requested = Signal(str)
    #: 匹配成功后请主窗口刷新报告预览
    report_preview_refresh_requested = Signal()

    #: 后台线程 → 主线程：AutoConnection 会跨线程自动排队。
    _result_ready = Signal(dict)
    _error_ready = Signal(str)

    def __init__(
        self,
        *,
        station_service,
        pointcloud_service,
        facade_service,
        task_progress: TaskProgressController,
        project_provider: Callable[[], object],
        pool=None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._station_service = station_service
        self._pointcloud_service = pointcloud_service
        self._facade_service = facade_service
        self._task_progress = task_progress
        self._project_provider = project_provider
        # 优先复用 RuntimeLifecycle 的 load pool；缺省时退回全局池。
        # 复用 load pool 的好处：closeEvent 的 runtime.stop(100) 会等待
        # 该池清空，避免关窗期间后台任务继续访问已销毁视口。
        self._pool = pool

        self._result_ready.connect(self._on_result)
        self._error_ready.connect(self._on_error)

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def start(self, facade: dict) -> None:
        """由检测复核页【图片匹配】按钮触发。"""
        from services.two_d_matching_service import TwoDMatchingService

        if not facade.get('quality_report'):
            self.warning_requested.emit(
                '该立面尚未完成质量评估，无法生成热力叠加图。')
            return

        project = self._project_provider()
        if project is None:
            self.warning_requested.emit('请先选择项目。')
            return

        self._task_progress.begin(
            TASK_PHOTO_MATCH, '图片匹配', '任务处理中...')

        facade_snapshot = dict(facade)
        controller = self

        class _MatchTask(QRunnable):
            def run(self):
                try:
                    result = TwoDMatchingService.match(
                        facade_snapshot,
                        project=project,
                        station_service=controller._station_service,
                        pointcloud_service=controller._pointcloud_service,
                        facade_service=controller._facade_service,
                    )
                    controller._result_ready.emit(result)
                except Exception as exc:
                    controller._error_ready.emit(str(exc))

        task = _MatchTask()
        if self._pool is not None:
            self._pool.start(task)
        else:
            QThreadPool.globalInstance().start(task)

    # ------------------------------------------------------------------
    # 主线程回调
    # ------------------------------------------------------------------
    def _on_result(self, result: dict) -> None:
        """处理匹配成功/失败结果。终态必须反映真实结果。"""
        ok = bool(result.get('success'))
        try:
            if ok:
                self.report_preview_refresh_requested.emit()
                self.status_message.emit(
                    '图片匹配完成，已更新 2D 现场热力映射图。', 5000)
            else:
                reason = result.get('reason', '未知错误')
                self.status_message.emit(f'图片匹配失败：{reason}', 8000)
                self.warning_requested.emit(reason)
        finally:
            self._task_progress.finish(TASK_PHOTO_MATCH, ok)

    def _on_error(self, message: str) -> None:
        """处理后台异常。"""
        self.status_message.emit(f'图片匹配异常：{message}', 8000)
        self._task_progress.finish(TASK_PHOTO_MATCH, False)