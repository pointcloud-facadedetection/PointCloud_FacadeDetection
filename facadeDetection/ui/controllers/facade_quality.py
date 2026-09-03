from dataclasses import replace
import numpy as np

from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QWidget,
)

from config.storage import Storage
from ui.dialogs.facade_quality_dialog import FacadeQualityDialog
from services.result_export_service import ResultExportService
from services.dal.results_repo import ResultsRepo
from utils.workers import QualityWorker


class FacadeQualityMixin:
    @staticmethod
    def _quality_double(value, minimum, maximum, step):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setDecimals(3)
        box.setValue(value)
        return box

    def _reset_quality_parameters(self):
        profile = getattr(self, '_inspection_profile', None)
        if profile is None:
            return
        self.quality_length_spin.setValue(profile.measure_height_m)
        self.quality_step_spin.setValue(profile.scan_step_m)
        self.quality_width_spin.setValue(profile.ruler_width_m)
        self.quality_select_band_spin.setValue(profile.select_band_m)
        self.quality_hole_band_spin.setValue(profile.hole_band_m)
        self.quality_bin_size_spin.setValue(profile.bin_size_m)
        self.quality_top_q_spin.setValue(profile.top_q)
        self.quality_sor_check.setChecked(profile.sor_enabled)
        self.quality_sor_sigma_spin.setValue(profile.sor_sigma)
        self.quality_sor_k_spin.setValue(profile.sor_k)
        method_index = self.quality_sor_method_combo.findData(profile.sor_method)
        if method_index >= 0:
            self.quality_sor_method_combo.setCurrentIndex(method_index)
        self.quality_sor_w_weight_spin.setValue(profile.sor_w_weight)
        self.quality_max_hole_ratio_spin.setValue(profile.max_hole_ratio)
        self.quality_min_points_spin.setValue(profile.min_points)

    def _quality_profile_snapshot(self, profile):
        if profile is None or not hasattr(self, 'quality_length_spin'):
            return profile
        return replace(
            profile,
            measure_height_m=self.quality_length_spin.value(),
            scan_step_m=self.quality_step_spin.value(),
            ruler_width_m=self.quality_width_spin.value(),
            select_band_m=self.quality_select_band_spin.value(),
            hole_band_m=self.quality_hole_band_spin.value(),
            bin_size_m=self.quality_bin_size_spin.value(),
            top_q=self.quality_top_q_spin.value(),
            sor_enabled=self.quality_sor_check.isChecked(),
            sor_sigma=self.quality_sor_sigma_spin.value(),
            sor_k=self.quality_sor_k_spin.value(),
            sor_method=str(self.quality_sor_method_combo.currentData() or 'local'),
            sor_w_weight=self.quality_sor_w_weight_spin.value(),
            max_hole_ratio=self.quality_max_hole_ratio_spin.value(),
            min_points=self.quality_min_points_spin.value())

    def _show_facade_results(self, results: list[dict]):
        self._quality_result_cache.clear()
        try:
            cloud = (self.pointcloud_service.resolve_processing_cloud()
                     if self.pointcloud_service is not None else None)
            if cloud and self.render_service is not None:
                self.render_facade.highlight_facades(cloud, results or [])
        except Exception as exc:
            print(f'[PCFD] facade.color_refresh_failed error={exc!r}', flush=True)
        
        unique = {}
        for facade in results or []:
            fid = int(facade.get('id', len(unique)))
            if fid not in unique:
                unique[fid] = facade
            else:
                unique[fid].update({k: v for k, v in facade.items()
                                    if k not in ('preview_status', 'preview_status_source')})
        results = list(unique.values())
        count = len(results)
        self.lbl_facade_summary.setText(f'检测立面数量：{count}')
        if not results:
            self.list_facades.clear()
            self._refresh_heatmap_button_state()
            self._latest_facade_results = []
            self.project_operation_service._last_facade_results = []
            self._refresh_report_preview()
            return
        self.list_facades.clear()
        for index, f in enumerate(results, 1):
            display_no = int(f.get('display_no') or index)
            f['display_no'] = display_no
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setSizeHint(QSize(0, 40))
            self.list_facades.addItem(item)
            
            row = QWidget()
            row.setStyleSheet("""
                QWidget { background: transparent; }
                QLabel { font-size: 12px; color: #334155; }
            """)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 6, 8, 6)
            row_layout.setSpacing(8)
            
            info = QLabel(f"立面{display_no}　点数 {int(f.get('point_count') or 0):,}")
            info.setToolTip(f"业务索引 facade_id={int(f.get('id', 0))}")
            info.setStyleSheet('font-size: 12px; color: #334155;')
            row_layout.addWidget(info, 1)
            
            color = self.render_facade.facade_color(f, display_no)
            swatch = QFrame()
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(
                'background-color: rgb(%d,%d,%d); border: 1px solid #94a3b8; border-radius: 3px;' %
                tuple(int(max(0, min(1, x)) * 255) for x in color)
            )
            swatch.setToolTip('该立面在视口中的显示颜色')
            row_layout.addWidget(swatch)
            
            status = self._facade_review_status(f)
            action_button = QPushButton(
                '完整' if status == 'complete' else '确认完整')
            action_button.setFixedWidth(72)
            action_button.setMinimumHeight(26)
            action_button.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    padding: 2px 8px;
                    border-radius: 4px;
                    border: 1px solid #cbd5e1;
                    background: #ffffff;
                    color: #475569;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    border-color: #94a3b8;
                }
            """)
            action_button.setToolTip('点击切换完整/不完整；仅完整立面允许质量计算')
            action_button.clicked.connect(
                lambda _=False, obj=f, button=action_button:
                self._toggle_facade_review_status(obj, button))
            row_layout.addWidget(action_button)
            
            row.setMaximumHeight(48)
            self.list_facades.setItemWidget(item, row)

        self._latest_facade_results = results
        self.project_operation_service._last_facade_results = results
        self._refresh_heatmap_button_state()
        self._refresh_report_preview()

    def _compatible_quality_results(self):
        results = getattr(self.project_operation_service, '_last_facade_results', None) or []
        cloud = self._active_cloud_name()
        if not cloud or self.render_service is None:
            return []
        return self.render_facade.compatible_quality_reports(
            cloud, results, index_service=self.facade_service._index_service)

    def _refresh_heatmap_button_state(self):
        button = getattr(self, 'btn_heatmap_toggle', None)
        if button is None:
            return
        enabled = bool(self._compatible_quality_results())
        button.setEnabled(enabled)
        mode = getattr(self, '_heatmap_mode', 'flatness')
        current = '平整度' if mode == 'flatness' else '垂直度'
        next_mode = '垂直度' if mode == 'flatness' else '平整度'
        button.setText(f'热力切换显示（当前：{current}）')
        button.setToolTip(f'点击切换至{next_mode}热力映射')

    def _toggle_heatmap_display(self):
        if not self._compatible_quality_results():
            self._refresh_heatmap_button_state()
            return
        self._heatmap_mode = ('verticality' if self._heatmap_mode == 'flatness'
                              else 'flatness')
        cloud = self._active_cloud_name()
        self.render_facade.quality_reports(
            cloud, getattr(self.project_operation_service, '_last_facade_results', None) or [],
            index_service=self.facade_service._index_service,
            heatmap_mode=self._heatmap_mode)
        self._refresh_heatmap_button_state()

    def _set_facade_preview_status(self, facade, button, status):
        # review_status 是唯一的规范运行时字段。
        facade['review_status'] = status
        for row in range(self.list_facades.count()):
            item = self.list_facades.item(row)
            payload = item.data(Qt.ItemDataRole.UserRole) or {}
            if int(payload.get('id', -1)) == int(facade.get('id', -2)):
                self.list_facades.setCurrentItem(item)
                item.setData(Qt.ItemDataRole.UserRole, facade)
                break
        for current in (getattr(self.project_operation_service, '_last_facade_results', None) or []):
            if int(current.get('id', -1)) == int(facade.get('id', -2)):
                current['review_status'] = status
                facade = current
                break
        item = self.list_facades.currentItem()
        if item is not None and int((item.data(Qt.ItemDataRole.UserRole) or {}).get('id', -1)) == int(facade.get('id', -2)):
            item.setData(Qt.ItemDataRole.UserRole, facade)
        button.setText('完整' if status == 'complete' else '不完整')
        if hasattr(self.project_operation_service, 'persist_facade_review_status'):
            self.project_operation_service.persist_facade_review_status(facade)

    def _toggle_facade_review_status(self, facade, button):
        """Toggle pending/incomplete -> complete, complete -> incomplete."""
        current = self._facade_review_status(facade)
        target = 'incomplete' if current == 'complete' else 'complete'
        self._set_facade_preview_status(facade, button, target)

    @staticmethod
    def _facade_review_status(facade):
        """Canonical status reader; tolerate legacy/null review_status."""
        value = (facade or {}).get('review_status')
        if value not in {'complete', 'incomplete'}:
            value = (facade or {}).get('preview_status')
        return value if value in {'complete', 'incomplete'} else 'pending'

    def _evaluate_selected_facade(self):
        item = self.list_facades.currentItem()
        if item is None:
            QMessageBox.information(self, '质量评估', '请先在右侧结果列表中选择一个立面。')
            return
        facade = item.data(Qt.ItemDataRole.UserRole)
        if facade:
            if self._facade_review_status(facade) != 'complete':
                QMessageBox.information(self, '质量评估', '请先人工确认该立面为完整立面。')
                return
            current = next((f for f in (getattr(self.project_operation_service, '_last_facade_results', None) or [])
                            if int(f.get('id', -1)) == int(facade.get('id', -2))), facade)
            self._evaluate_facade(current)

    def _active_cloud_name(self) -> str | None:
        try:
            service = getattr(self, 'pointcloud_service', None)
            if service is not None:
                resolved = service.resolve_processing_cloud()
                if resolved:
                    return resolved
            names = self.viewport.get_cloud_names()
            return names[-1] if names else None
        except Exception:
            return None

    def _quality_cache_key(self, cloud, facade, profile, grid_size):
        facade_id = int((facade or {}).get('id', 0))
        standard_id = getattr(profile, 'standard_id', None)
        flatness_limit_mm = getattr(profile, 'flatness_limit_mm', None)
        verticality_limit_mm = getattr(profile, 'verticality_limit_mm', None)
        window_size_m = getattr(profile, 'window_size_m', None)
        step_size_m = getattr(profile, 'step_size_m', None)
        measure_height_m = getattr(profile, 'measure_height_m', None)
        domain = facade.get('measurement_indices') or facade.get('voxel_ids') or []
        runtime_profile = self._quality_profile_snapshot(profile)
        parameter_snapshot = (runtime_profile.snapshot()
                              if runtime_profile is not None else {})
        return (
            str(cloud or ''), facade_id, standard_id,
            float(grid_size), flatness_limit_mm, verticality_limit_mm,
            window_size_m, step_size_m, measure_height_m,
            tuple(sorted(parameter_snapshot.items())),
            len(domain), hash(tuple(domain[:32])),
        )

    def _show_quality_dialog(self, cloud, facade, quality):
        facade_id = int(facade.get('id', 0))
        facade_no = int(facade.get('display_no', facade_id))

        print(f'[PCFD] ui.show_dialog facade_id={facade_id} facade_no={facade_no}', flush=True)

        def _export_context(display_quality):
            context = display_quality.get('__export_context') or {}
            if context.get('points') is not None and context.get('results_dir'):
                return context
            # 历史报告会刻意不保留大型点数组。从当前处理中的数据集重建导出输入。
            try:
                dataset = self.facade_service._index_service._get_dataset(cloud)
                proxy_ids = np.asarray(
                    facade.get('proxy_indices') or facade.get('inlier_indices') or [],
                    dtype=np.int64)
                if len(proxy_ids) and dataset.index.has_source_mapping():
                    raw_indices = dataset.index.proxy_to_source_ids(
                        proxy_ids, deduplicate=True)
                else:
                    raw_indices = proxy_ids
                raw_indices = raw_indices[(raw_indices >= 0) &
                                          (raw_indices < len(dataset.processed_raw_points))]
                if len(raw_indices) == 0:
                    print(f'[PCFD] export_context_failed facade_id={facade_id} '
                          'reason=no_facade_source_indices', flush=True)
                    return context
                points = np.asarray(dataset.processed_raw_points)[raw_indices]
                source_colors = dataset.index.get_source_colors()
                colors = (np.asarray(source_colors)[raw_indices]
                          if source_colors is not None and
                          len(source_colors) > int(raw_indices.max()) else None)
                if colors is None:
                    colors = np.tile(np.asarray(
                        self.render_service.facade_color_for(facade), dtype=float),
                        (len(points), 1))
                else:
                    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
                    # 确保分段立面的颜色在导出的基础图层中可见，且不受源RGB数据是否可用影响。
                    colors[:] = np.asarray(
                        self.render_service.facade_color_for(facade), dtype=float)
                project_uuid = getattr(self.current_project, 'project_id', None)
                results_dir = (Storage.ensure_project_dirs(project_uuid)['results']
                               if project_uuid else None)
                return {'results_dir': results_dir, 'points': points,
                        'colors': colors}
            except Exception as exc:
                print(f'[PCFD] export_context_failed facade_id={facade_id} '
                      f'error={exc!r}', flush=True)
                return context

        def _show_effect(mode='flatness'):
            try:
                display_quality = dict(quality) if isinstance(quality, dict) else {}
                display_quality['heatmap_mode'] = mode
                self.render_facade.apply_quality_colors(
                    cloud, display_quality,
                    index_service=self.facade_service._index_service)
                context = _export_context(display_quality)
                exported = ResultExportService().export_heatmap(
                    context.get('results_dir'), facade_no,
                    context.get('points'), context.get('colors'), display_quality)
                if exported and exported.get('heatmap'):
                    # 仅持久化小型、可移植的工件元数据。
                    # 运行时点数组保存在 __export_context 中，不会被存储。
                    artifact = {key: exported.get(key) for key in
                                ('mode', 'title', 'heatmap', 'overlay', 'report', 'legend')}
                    quality_report = facade.get('quality_report')
                    if isinstance(quality_report, dict):
                        artifacts = quality_report.setdefault('heatmap_artifacts', {})
                        artifacts[exported.get('mode', mode)] = artifact
                        self._refresh_report_preview()
                    self.statusBar().showMessage(
                        f'热力图已保存：{exported["heatmap"]}', 6000)
                else:
                    self.statusBar().showMessage('热力图显示成功，但导出失败，请检查日志。', 5000)
            except Exception as e:
                print(f'[PCFD] ui.show_effect_error facade_id={facade_id} error={e}', flush=True)

        def _restore():
            try:
                results = getattr(self.project_operation_service,
                                  '_last_facade_results', None)
                self.render_facade.restore_highlight(cloud, results or [])
            except Exception as e:
                print(f'[PCFD] ui.restore_error facade_id={facade_id} error={e}', flush=True)

        label = f'立面 {facade_no}'
        project_name = getattr(self.current_project, 'name', '') if self.current_project else ''

        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_not_dict facade_id={facade_id} type={type(quality)}', flush=True)
            quality = {}

        try:
            previous = self._quality_dialog
            if previous is not None and previous.isVisible():
                previous.close()

            dlg = FacadeQualityDialog(self, label, quality,
                                      project_name=project_name,
                                      on_show_colors=_show_effect,
                                      on_restore_colors=_restore)
            self._quality_dialog = dlg

            def _dialog_finished(result_code, dialog=dlg):
                if self._quality_dialog is dialog:
                    self._quality_dialog = None
                print(
                    f'[PCFD] ui.dialog_closed facade_id={facade_id} '
                    f'result={result_code}',
                    flush=True,
                )

            dlg.finished.connect(_dialog_finished)
            dlg.open()
        except Exception as e:
            print(f'[PCFD] ui.dialog_exception facade_id={facade_id} error={e}', flush=True)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, '质量评估', 
                f'显示质量结果时出错：\n{e}')

    def _on_facade_item_clicked(self, item):
        f = item.data(Qt.ItemDataRole.UserRole)
        if not f:
            return
        cloud = self._active_cloud_name()
        if not cloud:
            return
        
        self.render_facade.select_facade(cloud, int(f.get('id', 0)))
        self.statusBar().showMessage(f"已选中立面 {int(f.get('display_no', 1))}，请使用“评估”按钮执行质量检测", 3000)
        return

    def _evaluate_facade(self, f):
        cloud = self._active_cloud_name()
        if not cloud:
            return
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            QMessageBox.warning(self, '质量评估', '请先选择项目。')
            return
        # 高质量的 PNG 导出文件将保存到当前项目的结果文件夹中。
        results_dir = Storage.ensure_project_dirs(project_uuid)['results']

        facade_copy = dict(f)
        facade_id = int(facade_copy.get('id', 0))
        facade_no = int(facade_copy.get('display_no', facade_id))
        facade_copy['id'] = facade_id
        facade_copy['display_no'] = facade_no

        active_station_id = getattr(self.station_service, '_active_station_id', None)
        dataset = self.facade_service._index_service._get_dataset(cloud)
        facade_copy['__quality_request_context'] = {
            'project_uuid': project_uuid,
            'project_generation': self._project_generation,
            'station_id': facade_copy.get('station_id', active_station_id),
            'dataset_id': getattr(dataset, 'dataset_id', None),
            'dataset_revision': getattr(dataset, 'revision', None),
            'cloud_name': cloud,
        }

        # 已完成的历史结果本身就是一份有效的报告。直接重新打开它
        historical_quality = facade_copy.get('quality_report')
        if facade_copy.get('quality_status') == 'complete' and isinstance(historical_quality, dict):
            self._show_quality_dialog(cloud, facade_copy, historical_quality)
            return

        print(f'[PCFD] ui.evaluate_start facade_id={facade_id} facade_no={facade_no} '
              f'cloud={cloud}', flush=True)

        profile = self._quality_profile_snapshot(
            getattr(self, '_inspection_profile', None))
        grid_size = float(self.interval_combo.currentData())
        cache_key = self._quality_cache_key(cloud, facade_copy, profile, grid_size)
        cached_quality = self._quality_result_cache.get(cache_key)
        if cached_quality:
            self.statusBar().showMessage('已命中质量结果缓存', 3000)
            self._show_quality_dialog(cloud, facade_copy, cached_quality)
            return
        kwargs = {'profile': profile,
                  'grid_size': grid_size,
                  'results_dir': results_dir}
        self._quality_request_token = getattr(self, '_quality_request_token', 0) + 1
        token = self._quality_request_token
        self._quality_request_cache_key = cache_key
        self.statusBar().showMessage(f'正在计算立面 #{facade_no} 质量指标...')

        worker = QualityWorker(self.facade_service, cloud, facade_copy, kwargs)
        self._active_quality_worker = worker
        worker.signals.finished.connect(
            lambda facade, quality: self._on_quality_finished(token, cloud, facade, quality))
        self._quality_pool.start(worker)
        return

    def _on_quality_failed(self, token, error):
        if token != getattr(self, '_quality_request_token', -1):
            return

        self._active_quality_worker = None
        self._refresh_report_preview()
        self.statusBar().showMessage('质量计算失败')
        QMessageBox.warning(self, '质量评估', f'质量计算失败：{error}')

    def _on_quality_finished(self, token, cloud, f, quality):
        """Handle quality computation completion with full state machine."""
        facade_no = int(f.get('display_no', f.get('id', 0)))

        print(f'[PCFD] ui.quality_finished token={token} '
              f'facade_no={facade_no}', flush=True)

        if token != getattr(self, '_quality_request_token', -1):
            print(f'[PCFD] ui.quality_stale token={token} ignored', flush=True)
            return

        # 用户可以在工作者正在计算时切换工作站，因此在显示或持久化其结果之前，请验证请求上下文。
        request_context = f.get('__quality_request_context') or {}
        current_project = getattr(self.current_project, 'project_id', None)
        current_station = getattr(self.station_service, '_active_station_id', None)
        expected_station = request_context.get('station_id')
        current_generation = getattr(self, '_project_generation', 0)
        current_facades = (getattr(self.project_operation_service,
                                   '_last_facade_results', None) or [])
        request_db_id = f.get('facade_db_id')
        canonical = next(
            (item for item in current_facades
             if request_db_id is not None and
             str(item.get('facade_db_id')) == str(request_db_id)),
            None)
        if canonical is None:
            canonical = next(
                (item for item in current_facades
                 if str(item.get('id')) == str(f.get('id'))),
                None)
        current_dataset = self.facade_service._index_service._get_dataset(cloud)
        if (request_context.get('project_uuid') != current_project or
                request_context.get('project_generation') != current_generation or
                request_context.get('cloud_name') != cloud or
                canonical is None or
                (request_db_id is not None and
                 str(canonical.get('facade_db_id')) != str(request_db_id)) or
                (request_context.get('dataset_id') is not None and
                 getattr(current_dataset, 'dataset_id', None) !=
                 request_context.get('dataset_id')) or
                (request_context.get('dataset_revision') is not None and
                 getattr(current_dataset, 'revision', None) !=
                 request_context.get('dataset_revision')) or
                (expected_station is not None and current_station is not None and
                 int(expected_station) != int(current_station))):
            print(f'[PCFD] ui.quality_stale_identity facade_id={f.get("id")} '
                  f'expected_station={expected_station} current_station={current_station} '
                  f'expected_cloud={request_context.get("cloud_name")} current_cloud={cloud} '
                  f'expected_db_id={request_db_id} canonical_db_id='
                  f'{(canonical or {}).get("facade_db_id")} '
                  f'expected_dataset={request_context.get("dataset_id")} '
                  f'current_dataset={getattr(current_dataset, "dataset_id", None)}',
                  flush=True)
            self._active_quality_worker = None
            self.statusBar().showMessage('质量结果已过期，当前站点已变化，未保存。', 5000)
            return

        self.statusBar().clearMessage()

        if quality is None:
            print(f'[PCFD] ui.quality_none facade_id={facade_no}', flush=True)
            QMessageBox.warning(self, '质量评估', 
                f'立面 #{facade_no} 质量计算失败：未返回结果。请检查日志。')
            self._refresh_report_preview()
            return

        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_invalid_type facade_id={facade_no} '
                  f'type={type(quality)}', flush=True)
            QMessageBox.warning(self, '质量评估',
                f'立面 #{facade_no} 质量计算返回异常类型：{type(quality)}')
            self._refresh_report_preview()
            return

        cache_key = getattr(self, '_quality_request_cache_key', None)
        if cache_key is not None:
            self._quality_result_cache[cache_key] = quality

        # Update reports list
        self._quality_reports = [r for r in self._quality_reports
                                 if (r.get('facade') or {}).get('display_no') != facade_no]
        self._quality_reports.append({'facade': f, 'quality': quality})

        if not quality.get('ok', True):
            error_reason = quality.get('reason', 'unknown')
            error_message = quality.get('message', f'质量计算失败：{error_reason}')
            print(f'[PCFD] ui.quality_error facade_id={facade_no} '
                  f'reason={error_reason} message={error_message}', flush=True)

            QMessageBox.information(self, '质量评估', 
                f'立面 #{facade_no} 质量评估结果：\n\n{error_message}')

            QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))
            self._refresh_report_preview()
            return

        overall = quality.get('overall') or {}
        window_count = int(overall.get('candidate_window_count', 0) or 0)
        valid_count = int(overall.get('quality_valid_window_count', 0) or 0)

        if valid_count <= 0:
            print(f'[PCFD] ui.quality_no_valid_windows facade_id={facade_no} '
                  f'candidates={window_count}', flush=True)
            QMessageBox.information(self, '质量评估',
                f'立面 #{facade_no} 质量计算完成，但未找到有效检测窗口。\n'
                f'候选窗口数：{window_count}\n'
                f'可能原因：立面尺寸过小、点云密度不足或存在大面积空洞。')
            QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))
            self._refresh_report_preview()
            return

        print(f'[PCFD] ui.quality_success facade_id={facade_no} '
              f'windows={window_count} valid={valid_count} '
              f'intervals={len(quality.get("intervals", []))}', flush=True)
        # 仅在算法生成有效报告后才进行持久化。
        try:
            project_uuid = getattr(self.current_project, 'project_id', None)
            if not project_uuid:
                raise RuntimeError('当前项目已失效，无法保存质量结果')
            dataset = self.facade_service._index_service._get_dataset(cloud)
            artifact_path = None
            ResultsRepo.commit_quality_success(
                project_uuid, int(f.get('id', 0)), quality,
                display_no=facade_no,
                facade_data=f,
                dataset_revision=getattr(dataset, 'revision', None),
                quality_artifact_path=artifact_path,
                color=self.render_service.facade_color_for(f, facade_no),
            )
            f['quality_status'] = 'complete'
            f['quality_report'] = quality
            for current in (getattr(self.project_operation_service,
                                    '_last_facade_results', None) or []):
                if int(current.get('id', -1)) == int(f.get('id', -2)):
                    current.update({'quality_status': 'complete',
                                    'quality_report': quality,
                                    'dataset_revision': getattr(dataset, 'revision', None)})
                    break
            self._refresh_heatmap_button_state()
        except Exception as exc:
            print(f'[PCFD] quality.persist_failed facade_id={facade_no} error={exc!r}', flush=True)
            QMessageBox.warning(self, '质量评估', f'算法已完成，但结果保存失败：{exc}')
            self._refresh_report_preview()
            return
        self._refresh_report_preview()
        QTimer.singleShot(0, lambda: self._show_quality_dialog(cloud, f, quality))
