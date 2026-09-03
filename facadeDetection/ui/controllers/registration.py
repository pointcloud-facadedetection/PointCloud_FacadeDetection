from pathlib import Path

import numpy as np

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox

from utils.workers import RegistrationWorker


class RegistrationMixin:
    def _run_station_registration(self):
        if getattr(self, '_registration_worker', None) is not None:
            QMessageBox.information(self, '点云配准', '已有配准任务正在执行，请稍候。')
            return
        try:
            self.station_service.refresh()
            rows = [x for x in self.station_service.list_stations() if x.is_selected]
            if len(rows) != 2:
                raise ValueError('点云配准需要恰好选择两个 PLY 站点')
            # PLY 已在 FLS 导出阶段应用 transformToGlobal。默认直接执行
            # GPS 全局坐标上的残差 ICP；人工选点仅作为失败后的兜底工具。
            self._registration_rows = tuple(x.id for x in rows)
            self._start_auto_registration(self._registration_rows)
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', str(exc))

    def _start_auto_registration(self, station_ids):
        """Run residual ICP on already-globalized PLY clouds off the GUI thread."""
        generation = self._project_generation
        project_uuid = getattr(self.current_project, 'project_id', None)

        def run_registration():
            payload = self.station_service.register_selected(update_viewport=False)
            payload['_project_generation'] = generation
            payload['_project_uuid'] = project_uuid
            payload['_station_ids_snapshot'] = tuple(station_ids)
            return payload

        worker = RegistrationWorker(run_registration)
        self._registration_worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.statusBar().showMessage('正在执行 GPS 全局坐标残差 ICP，请稍候...')
        self._set_registration_buttons_enabled(False)
        self._registration_pool.start(worker)

    def _enter_manual_registration_fallback(self):
        """Prepare the optional correspondence workflow after auto ICP fails."""
        rows = [x for x in self.station_service.list_stations()
                if x.is_selected]
        if len(rows) != 2:
            return
        try:
            rows, cloud_names = self.station_service.prepare_registration_view(rows)
            source_cloud, target_cloud = cloud_names[1], cloud_names[0]
            self._registration_rows = tuple(row.id for row in rows)
            self._registration_prompted_pairs = 0

            def on_pick(_picked, source_next):
                src, tgt = self.render_service.registration_pick_points()
                pairs = min(len(src), len(tgt))
                self.statusBar().showMessage(
                    f'配准选点：已完成 {pairs} 对，请继续点击'
                    f'{"源站点" if source_next else "目标站点"}同名点')
                if pairs >= 3 and pairs != self._registration_prompted_pairs:
                    self._registration_prompted_pairs = pairs
                    answer = QMessageBox.question(
                        self, '点云配准',
                        f'已选择 {pairs} 对对应点，是否执行人工初值 ICP？',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if answer == QMessageBox.StandardButton.Yes:
                        self._start_manual_registration(src, tgt)

            self.render_service.enter_registration_pick_mode(
                source_cloud, target_cloud, on_pick, pick_radius=10)
            self._registration_proxy_snapshot = (
                np.asarray(self.viewport.get_cloud_data(source_cloud)['pos'],
                           dtype=np.float64).copy(),
                np.asarray(self.viewport.get_cloud_data(target_cloud)['pos'],
                           dtype=np.float64).copy())
            self.statusBar().showMessage(
                '自动配准未达到质量门限。请按源站点、目标站点交替选择至少 3 对同名点')
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', f'进入人工选点失败：{exc}')

    def _start_manual_registration(self, source_points, target_points):
        self.viewport.exit_pick_mode()
        generation = self._project_generation
        project_uuid = getattr(self.current_project, 'project_id', None)
        station_ids = tuple(self._registration_rows)
        proxy_snapshot = tuple(
            np.asarray(points, dtype=np.float64).copy()
            for points in self._registration_proxy_snapshot)

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
        self._registration_worker = worker
        worker.signals.finished.connect(self._on_registration_finished)
        worker.signals.failed.connect(self._on_registration_failed)
        self.statusBar().showMessage('正在执行人工初值代理域 ICP，请稍候...')
        self._set_registration_buttons_enabled(False)
        self._registration_pool.start(worker)

    def _set_registration_buttons_enabled(self, enabled):
        button = self.header_buttons.get('btn_registration')
        if button is not None:
            button.setEnabled(enabled)

    @Slot(object)
    def _on_registration_finished(self, payload):
        generation = payload.pop('_project_generation', None)
        project_uuid = payload.pop('_project_uuid', None)
        station_snapshot = tuple(payload.pop('_station_ids_snapshot', ()))
        current_uuid = getattr(self.current_project, 'project_id', None)
        if (getattr(self, '_closing', False) or generation != self._project_generation or
                project_uuid != current_uuid or
                station_snapshot != tuple(x.id for x in self.station_service.list_stations()
                                           if x.is_selected)):
            self._discard_registration_result(payload)
            self._registration_worker = None
            self._set_registration_buttons_enabled(True)
            return
        try:
            self.station_service.commit_registration(payload)
            self._refresh_station_panel()
            self.statusBar().showMessage('点云配准完成，已显示注册合并结果。', 5000)
        except Exception as exc:
            QMessageBox.warning(self, '点云配准', f'配准结果提交失败：{exc}')
        finally:
            self._registration_worker = None
            self._set_registration_buttons_enabled(True)

    @Slot(str)
    def _on_registration_failed(self, message):
        self._registration_worker = None
        self._set_registration_buttons_enabled(True)
        if getattr(self, '_registration_rows', None):
            self._enter_manual_registration_fallback()
            self.statusBar().showMessage(f'自动配准失败：{message}', 5000)
            return
        self.statusBar().showMessage('点云配准失败', 5000)
        QMessageBox.warning(self, '点云配准', message)

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
