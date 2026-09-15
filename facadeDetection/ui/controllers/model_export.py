"""【导出模型】编排控制器。

负责把"选站点 → 后台下采样导出 PLY → 模态进度弹窗 → 结果提示"这条链路
串起来，让报告页（UI 层）只保留一次点击的入口。

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from services.model_export_service import DEFAULT_VOXEL_SIZE, ModelExportService
from ui.controllers.task_progress import TASK_MODEL_EXPORT, TaskProgressController
from utils.workers import ModelExportWorker


class StationExportDialog(QDialog):
    """站点选择对话框：只选择"要导出哪个站点的点云"。"""

    def __init__(self, stations, parent=None, project_name: str = ''):
        super().__init__(parent)
        self.setObjectName('stationExportDialog')
        self.setWindowTitle('导出模型')
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        hint = QLabel('请选择要导出的站点点云')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        if project_name:
            project_label = QLabel(f'当前项目：{project_name}')
            project_label.setObjectName('stationExportProjectLabel')
            layout.addWidget(project_label)

        form = QFormLayout()
        form.setSpacing(8)
        self.station_combo = QComboBox()
        self.station_combo.setObjectName('stationExportCombo')
        for station in stations:
            label = str(getattr(station, 'display_name', '') or '未命名站点')
            if getattr(station, 'last_error', None):
                label = f'{label}（资产失效）'
            self.station_combo.addItem(label, int(station.id))
        form.addRow('导出站点', self.station_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('导出')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_station_id(self) -> Optional[int]:
        value = self.station_combo.currentData()
        return None if value is None else int(value)


class ModelExportController(QObject):
    """把站点源点云导出 PLY 的编排收拢到一处。"""

    status_message = Signal(str, int)
    warning_requested = Signal(str)
    info_requested = Signal(str)
    export_finished = Signal(object)

    def __init__(self, station_service, pool, project_provider,
                 *, progress: Optional[TaskProgressController] = None,
                 parent=None):
        super().__init__(parent)
        self.station_service = station_service
        self.pool = pool
        self._project_provider = project_provider
        self.progress = progress or TaskProgressController(parent)
        self.service = ModelExportService()
        self._workers = set()
        #: 正在执行的导出 worker（同一时刻只允许一个）。
        self._active_worker = None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def exportable_stations(self):
        """返回当前项目可导出的站点（排除资产失效项）。"""
        try:
            stations = list(self.station_service.list_stations())
        except Exception:
            return []
        return [s for s in stations if not getattr(s, 'last_error', None)]

    def _current_project(self):
        return self._project_provider()

    # ------------------------------------------------------------------
    # 站点选择
    # ------------------------------------------------------------------
    def prompt_station(self, parent=None, stations=None):
        """弹出站点选择对话框；取消返回 None。"""
        stations = list(stations if stations is not None
                        else self.exportable_stations())
        if not stations:
            return None
        if len(stations) == 1:
            # 只有一个站点时免去一次点击，直接进入路径选择。
            return stations[0]
        project = self._current_project()
        dialog = StationExportDialog(
            stations, parent,
            project_name=str(getattr(project, 'name', '') or ''))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        station_id = dialog.selected_station_id()
        return next((s for s in stations if int(s.id) == station_id), None)

    # ------------------------------------------------------------------
    # 导出主流程
    # ------------------------------------------------------------------
    def export_station(self, station, target_path):
        """在后台线程池执行 voxel=0.2 下采样并导出 PLY。"""
        if station is None:
            return
        project = self._current_project()
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            self.warning_requested.emit('请先创建或选择项目。')
            return

        self.service.set_project(project_uuid)
        station_name = str(getattr(station, 'display_name', '') or '站点')
        self.progress.begin(
            TASK_MODEL_EXPORT,
            '正在导出模型',
            f'准备导出「{station_name}」的点云模型',
        )

        # 参数严格对齐 ModelExportWorker.__init__：
        # (service, station_id, target_path, *, voxel_size, project_uuid)。
        # 线程池不属于 worker 的构造参数，改由 _start() 决定在哪执行。
        worker = ModelExportWorker(
            self.service,
            int(station.id),
            str(target_path),
            voxel_size=DEFAULT_VOXEL_SIZE,
            project_uuid=project_uuid,
        )
        self._workers.add(worker)
        self._active_worker = worker
        worker.signals.progress.connect(
            self.progress.connect_progress_slot(TASK_MODEL_EXPORT)
            if hasattr(self.progress, 'connect_progress_slot')
            else (lambda percent, text='': self.progress.report(
                TASK_MODEL_EXPORT, percent, text))
        )
        worker.signals.finished.connect(
            lambda payload, w=worker: self._on_finished(w, payload))
        worker.signals.failed.connect(
            lambda message, w=worker: self._on_failed(w, message))
        # worker 内部若自建线程池，则直接 start；否则交外部池执行。
        self._start(worker)

    def _start(self, worker):
        if getattr(worker, 'pool', None) is not None:
            worker.pool.start(worker)
            return
        if self.pool is not None:
            self.pool.start(worker)
            return
        worker.run()

    def _on_finished(self, worker, payload):
        self._workers.discard(worker)
        if worker is self._active_worker:
            self._active_worker = None
        path = (payload or {}).get('path')
        exported = int((payload or {}).get('exported_points') or 0)
        message = (f'已导出 {exported:,} 点')
        self.progress.finish(TASK_MODEL_EXPORT, True, message)
        self.status_message.emit(f'模型已导出：{Path(str(path)).name}', 8000)
        self.info_requested.emit(
            f'{message}\n\n输出文件：\n{path}')
        self.export_finished.emit(payload)

    def _on_failed(self, worker, message):
        self._workers.discard(worker)
        if worker is self._active_worker:
            self._active_worker = None
        # 用户主动中止走 worker 的 failed 通道回报；此时进度窗已被
        # TaskProgressController 收成"已中止"终态，不能再弹失败框，
        # 否则等于把用户自己的操作报成错误。
        if '取消' in str(message):
            self.status_message.emit('模型导出已取消', 5000)
            return
        self.progress.finish(TASK_MODEL_EXPORT, False, str(message))
        self.warning_requested.emit(f'模型导出失败：{message}')
