from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, Signal, Slot

from utils.workers import RegistrationWorker


class RegistrationController(QObject):
    """站点配准编排：持有配准 worker 与选点状态，UI 反馈一律经 Signal 发回。"""

    busy_changed = Signal(bool)              # 配准按钮使能
    status_message = Signal(str, int)        # 状态栏文本与超时（毫秒，0 表示持续）
    warning_requested = Signal(str)          # QMessageBox.warning 正文
    info_requested = Signal(str)             # QMessageBox.information 正文
    station_panel_refresh_requested = Signal()
    manual_pick_requested = Signal(str, str)     # (source_cloud, target_cloud)
    manual_icp_prompt_requested = Signal(int)    # 已选对应点对数

    def __init__(self, station_service, render_service, pool,
                 context_provider, parent=None):
        super().__init__(parent)
        self.station_service = station_service
        self.render_service = render_service
        self._pool = pool
        # context_provider() -> (project_generation, project_uuid, closing)，
        # 每次调用实时读取，用于识别过期的配准结果。
        self._context_provider = context_provider
        self._worker = None
        self._rows = None
        self._prompted_pairs = 0
        self._proxy_snapshot = None

    def run_station_registration(self):
        if self._worker is not None:
            self.info_requested.emit('已有配准任务正在执行，请稍候。')
            return
        try:
            self.station_service.refresh()
            rows = [x for x in self.station_service.list_stations() if x.is_selected]
            if len(rows) != 2:
                raise ValueError('点云配准需要恰好选择两个 PLY 站点')
            # PLY 已在 FLS 导出阶段应用 transformToGlobal。默认直接执行
            # GPS 全局坐标上的残差 ICP；人工选点仅作为失败后的兜底工具。
            self._rows = tuple(x.id for x in rows)
            self._start_auto_registration(self._rows)
        except Exception as exc:
            self.warning_requested.emit(str(exc))

    def _start_auto_registration(self, station_ids):
        """Run residual ICP on already-globalized PLY clouds off the GUI thread."""
        generation, project_uuid, _ = self._context_provider()

        def run_registration():
            payload = self.station_service.register_selected(update_viewport=False)
            payload['_project_generation'] = generation
            payload['_project_uuid'] = project_uuid
            payload['_station_ids_snapshot'] = tuple(station_ids)
            return payload

        worker = RegistrationWorker(run_registration)
        self._worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.status_message.emit('正在执行 GPS 全局坐标残差 ICP，请稍候...', 0)
        self.busy_changed.emit(False)
        self._pool.start(worker)

    def prepare_manual_fallback(self):
        """Prepare the optional correspondence workflow after auto ICP fails."""
        rows = [x for x in self.station_service.list_stations()
                if x.is_selected]
        if len(rows) != 2:
            return
        try:
            rows, cloud_names = self.station_service.prepare_registration_view(rows)
            source_cloud, target_cloud = cloud_names[1], cloud_names[0]
            self._rows = tuple(row.id for row in rows)
            self._prompted_pairs = 0
            self.manual_pick_requested.emit(source_cloud, target_cloud)
            self._proxy_snapshot = (
                np.asarray(self.render_service.get_cloud_data(source_cloud)['pos'],
                           dtype=np.float64).copy(),
                np.asarray(self.render_service.get_cloud_data(target_cloud)['pos'],
                           dtype=np.float64).copy())
            self.status_message.emit(
                '自动配准未达到质量门限。请按源站点、目标站点交替选择至少 3 对同名点', 0)
        except Exception as exc:
            self.warning_requested.emit(f'进入人工选点失败：{exc}')

    def handle_pick_updated(self, source_next):
        """视口每次选点回调：汇报进度，凑满 3 对时请求人工 ICP 确认弹窗。"""
        src, tgt = self.render_service.registration_pick_points()
        pairs = min(len(src), len(tgt))
        self.status_message.emit(
            f'配准选点：已完成 {pairs} 对，请继续点击'
            f'{"源站点" if source_next else "目标站点"}同名点', 0)
        if pairs >= 3 and pairs != self._prompted_pairs:
            self._prompted_pairs = pairs
            self.manual_icp_prompt_requested.emit(pairs)

    def start_manual_registration(self):
        source_points, target_points = self.render_service.registration_pick_points()
        self.render_service.exit_pick_mode()
        generation, project_uuid, _ = self._context_provider()
        station_ids = tuple(self._rows)
        proxy_snapshot = tuple(
            np.asarray(points, dtype=np.float64).copy()
            for points in self._proxy_snapshot)

        def run_registration():
            payload = self.station_service.register_selected(
                update_viewport=False,
                manual_points=(np.asarray(source_points), np.asarray(target_points)),
                proxy_clouds=proxy_snapshot)
            payload['_project_generation'] = generation
            payload['_project_uuid'] = project_uuid
            payload['_station_ids_snapshot'] = station_ids
            return payload

        worker = RegistrationWorker(run_registration)
        self._worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.status_message.emit('正在执行人工初值代理域 ICP，请稍候...', 0)
        self.busy_changed.emit(False)
        self._pool.start(worker)

    @Slot(object)
    def _on_registration_finished(self, payload):
        generation = payload.pop('_project_generation', None)
        project_uuid = payload.pop('_project_uuid', None)
        station_snapshot = tuple(payload.pop('_station_ids_snapshot', ()))
        current_generation, current_uuid, closing = self._context_provider()
        if (closing or generation != current_generation or
                project_uuid != current_uuid or
                station_snapshot != tuple(x.id for x in self.station_service.list_stations()
                                           if x.is_selected)):
            self._discard_registration_result(payload)
            self._worker = None
            self.busy_changed.emit(True)
            return
        try:
            self.station_service.commit_registration(payload)
            self.station_panel_refresh_requested.emit()
            self.status_message.emit('点云配准完成，已显示注册合并结果。', 5000)
        except Exception as exc:
            self.warning_requested.emit(f'配准结果提交失败：{exc}')
        finally:
            self._worker = None
            self.busy_changed.emit(True)

    @Slot(str)
    def _on_registration_failed(self, message):
        self._worker = None
        self.busy_changed.emit(True)
        if self._rows:
            self.prepare_manual_fallback()
            self.status_message.emit(f'自动配准失败：{message}', 5000)
            return
        self.status_message.emit('点云配准失败', 5000)
        self.warning_requested.emit(message)

    @staticmethod
    def _discard_registration_result(payload):
        path = payload.get('result_path') if isinstance(payload, dict) else None
        if not path:
            return
        result = Path(path)
        if result.name.startswith('registration_') and result.exists():
            try:
                result.unlink()
            except OSError:
                pass
