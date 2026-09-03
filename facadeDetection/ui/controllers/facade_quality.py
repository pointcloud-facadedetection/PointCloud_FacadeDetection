from PySide6.QtCore import QObject, QTimer, Signal

from config.storage import Storage
from utils.workers import QualityWorker


class FacadeQualityController(QObject):
    """立面质量编排：持有质量缓存/报告/worker 状态，UI 反馈一律经 Signal 发回。"""

    status_message = Signal(str, int)        # 状态栏文本与超时（毫秒，0 表示持续）
    status_cleared = Signal()                # 清空状态栏
    warning_requested = Signal(str)          # QMessageBox.warning 正文（标题：质量评估）
    info_requested = Signal(str)             # QMessageBox.information 正文（标题：质量评估）
    report_preview_refresh_requested = Signal()
    heatmap_button_refresh_requested = Signal()
    show_dialog_requested = Signal(str, object, object)  # (cloud, facade, quality)

    def __init__(self, facade_service, render_facade, render_service,
                 pointcloud_service, project_operation_service,
                 station_service, pool, context_provider, profile_provider,
                 grid_size_provider, parent=None):
        super().__init__(parent)
        self.facade_service = facade_service
        self.render_facade = render_facade
        self.render_service = render_service
        self.pointcloud_service = pointcloud_service
        self.project_operation_service = project_operation_service
        self.station_service = station_service
        self._pool = pool
        # context_provider() -> (project_uuid, project_generation)，实时读取，
        # 用于识别过期的质量结果。
        self._context_provider = context_provider
        # profile_provider() -> 当前质量参数快照；grid_size_provider() -> 当前
        # 采样间距。两者都依赖 UI 控件读数，由窗口以可调用对象注入。
        self._profile_provider = profile_provider
        self._grid_size_provider = grid_size_provider
        self.quality_reports = []
        self.quality_result_cache = {}
        self._latest_facade_results = []
        self.heatmap_mode = 'flatness'
        self._quality_request_token = 0
        self._quality_request_cache_key = None
        self.active_quality_worker = None

    def process_facade_results(self, results):
        """清空质量缓存并刷新立面着色，返回按 id 去重后的立面列表。"""
        self.quality_result_cache.clear()
        try:
            cloud = (self.pointcloud_service.resolve_processing_cloud()
                     if self.pointcloud_service is not None else None)
            if cloud and self.render_facade is not None:
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
        return list(unique.values())

    def set_latest_results(self, results):
        self._latest_facade_results = results
        self.project_operation_service.last_facade_results = results

    def compatible_quality_results(self):
        results = self.project_operation_service.last_facade_results or []
        cloud = self.active_cloud_name()
        if not cloud or self.render_service is None:
            return []
        return self.render_facade.compatible_quality_reports(
            cloud, results, index_service=self.facade_service.index_service)

    def toggle_heatmap_display(self):
        if not self.compatible_quality_results():
            self.heatmap_button_refresh_requested.emit()
            return
        self.heatmap_mode = ('verticality' if self.heatmap_mode == 'flatness'
                             else 'flatness')
        cloud = self.active_cloud_name()
        self.render_facade.quality_reports(
            cloud, self.project_operation_service.last_facade_results or [],
            index_service=self.facade_service.index_service,
            heatmap_mode=self.heatmap_mode)
        self.heatmap_button_refresh_requested.emit()

    def active_cloud_name(self):
        try:
            service = self.pointcloud_service
            if service is not None:
                resolved = service.resolve_processing_cloud()
                if resolved:
                    return resolved
            names = self.render_service.get_cloud_names()
            return names[-1] if names else None
        except Exception:
            return None

    def evaluate_facade(self, f):
        cloud = self.active_cloud_name()
        if not cloud:
            return
        project_uuid, generation = self._context_provider()
        if not project_uuid:
            self.warning_requested.emit('请先选择项目。')
            return
        # 高质量的 PNG 导出文件将保存到当前项目的结果文件夹中。
        results_dir = Storage.ensure_project_dirs(project_uuid)['results']

        facade_copy = dict(f)
        facade_id = int(facade_copy.get('id', 0))
        facade_no = int(facade_copy.get('display_no', facade_id))
        facade_copy['id'] = facade_id
        facade_copy['display_no'] = facade_no

        active_station_id = getattr(self.station_service, '_active_station_id', None)
        dataset = self.facade_service.get_dataset(cloud)
        facade_copy['__quality_request_context'] = {
            'project_uuid': project_uuid,
            'project_generation': generation,
            'station_id': facade_copy.get('station_id', active_station_id),
            'dataset_id': getattr(dataset, 'dataset_id', None),
            'dataset_revision': getattr(dataset, 'revision', None),
            'cloud_name': cloud,
        }

        # 已完成的历史结果本身就是一份有效的报告。直接重新打开它
        historical_quality = facade_copy.get('quality_report')
        if facade_copy.get('quality_status') == 'complete' and isinstance(historical_quality, dict):
            self.show_dialog_requested.emit(cloud, facade_copy, historical_quality)
            return

        print(f'[PCFD] ui.evaluate_start facade_id={facade_id} facade_no={facade_no} '
              f'cloud={cloud}', flush=True)

        profile = self._profile_provider()
        grid_size = float(self._grid_size_provider())
        cache_key = self._quality_cache_key(cloud, facade_copy, profile, grid_size)
        cached_quality = self.quality_result_cache.get(cache_key)
        if cached_quality:
            self.status_message.emit('已命中质量结果缓存', 3000)
            self.show_dialog_requested.emit(cloud, facade_copy, cached_quality)
            return
        kwargs = {'profile': profile,
                  'grid_size': grid_size,
                  'results_dir': results_dir}
        self._quality_request_token += 1
        token = self._quality_request_token
        self._quality_request_cache_key = cache_key
        self.status_message.emit(f'正在计算立面 #{facade_no} 质量指标...', 0)

        worker = QualityWorker(self.facade_service, cloud, facade_copy, kwargs)
        self.active_quality_worker = worker
        worker.signals.finished.connect(
            lambda facade, quality: self._on_quality_finished(token, cloud, facade, quality))
        self._pool.start(worker)

    def on_quality_failed(self, token, error):
        if token != self._quality_request_token:
            return

        self.active_quality_worker = None
        self.report_preview_refresh_requested.emit()
        self.status_message.emit('质量计算失败', 0)
        self.warning_requested.emit(f'质量计算失败：{error}')

    def _quality_cache_key(self, cloud, facade, profile, grid_size):
        facade_id = int((facade or {}).get('id', 0))
        standard_id = getattr(profile, 'standard_id', None)
        flatness_limit_mm = getattr(profile, 'flatness_limit_mm', None)
        verticality_limit_mm = getattr(profile, 'verticality_limit_mm', None)
        window_size_m = getattr(profile, 'window_size_m', None)
        step_size_m = getattr(profile, 'step_size_m', None)
        measure_height_m = getattr(profile, 'measure_height_m', None)
        domain = facade.get('measurement_indices') or facade.get('voxel_ids') or []
        runtime_profile = self._profile_provider()
        parameter_snapshot = (runtime_profile.snapshot()
                              if runtime_profile is not None else {})
        return (
            str(cloud or ''), facade_id, standard_id,
            float(grid_size), flatness_limit_mm, verticality_limit_mm,
            window_size_m, step_size_m, measure_height_m,
            tuple(sorted(parameter_snapshot.items())),
            len(domain), hash(tuple(domain[:32])),
        )

    def _on_quality_finished(self, token, cloud, f, quality):
        """Handle quality computation completion with full state machine."""
        facade_no = int(f.get('display_no', f.get('id', 0)))

        print(f'[PCFD] ui.quality_finished token={token} '
              f'facade_no={facade_no}', flush=True)

        if token != self._quality_request_token:
            print(f'[PCFD] ui.quality_stale token={token} ignored', flush=True)
            return

        # 用户可以在工作者正在计算时切换工作站，因此在显示或持久化其结果之前，请验证请求上下文。
        request_context = f.get('__quality_request_context') or {}
        current_project, current_generation = self._context_provider()
        current_station = getattr(self.station_service, '_active_station_id', None)
        expected_station = request_context.get('station_id')
        current_facades = (self.project_operation_service.last_facade_results or [])
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
        current_dataset = self.facade_service.get_dataset(cloud)
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
            self.active_quality_worker = None
            self.status_message.emit('质量结果已过期，当前站点已变化，未保存。', 5000)
            return

        self.status_cleared.emit()

        if quality is None:
            print(f'[PCFD] ui.quality_none facade_id={facade_no}', flush=True)
            self.warning_requested.emit(
                f'立面 #{facade_no} 质量计算失败：未返回结果。请检查日志。')
            self.report_preview_refresh_requested.emit()
            return

        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_invalid_type facade_id={facade_no} '
                  f'type={type(quality)}', flush=True)
            self.warning_requested.emit(
                f'立面 #{facade_no} 质量计算返回异常类型：{type(quality)}')
            self.report_preview_refresh_requested.emit()
            return

        cache_key = self._quality_request_cache_key
        if cache_key is not None:
            self.quality_result_cache[cache_key] = quality

        # Update reports list
        self.quality_reports = [r for r in self.quality_reports
                                if (r.get('facade') or {}).get('display_no') != facade_no]
        self.quality_reports.append({'facade': f, 'quality': quality})

        if not quality.get('ok', True):
            error_reason = quality.get('reason', 'unknown')
            error_message = quality.get('message', f'质量计算失败：{error_reason}')
            print(f'[PCFD] ui.quality_error facade_id={facade_no} '
                  f'reason={error_reason} message={error_message}', flush=True)

            self.info_requested.emit(
                f'立面 #{facade_no} 质量评估结果：\n\n{error_message}')

            QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
            self.report_preview_refresh_requested.emit()
            return

        overall = quality.get('overall') or {}
        window_count = int(overall.get('candidate_window_count', 0) or 0)
        valid_count = int(overall.get('quality_valid_window_count', 0) or 0)

        if valid_count <= 0:
            print(f'[PCFD] ui.quality_no_valid_windows facade_id={facade_no} '
                  f'candidates={window_count}', flush=True)
            self.info_requested.emit(
                f'立面 #{facade_no} 质量计算完成，但未找到有效检测窗口。\n'
                f'候选窗口数：{window_count}\n'
                f'可能原因：立面尺寸过小、点云密度不足或存在大面积空洞。')
            QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
            self.report_preview_refresh_requested.emit()
            return

        print(f'[PCFD] ui.quality_success facade_id={facade_no} '
              f'windows={window_count} valid={valid_count} '
              f'intervals={len(quality.get("intervals", []))}', flush=True)
        # 仅在算法生成有效报告后才进行持久化。
        try:
            project_uuid, _ = self._context_provider()
            if not project_uuid:
                raise RuntimeError('当前项目已失效，无法保存质量结果')
            dataset = self.facade_service.get_dataset(cloud)
            artifact_path = None
            self.facade_service.commit_quality_success(
                project_uuid, int(f.get('id', 0)), quality,
                display_no=facade_no,
                facade_data=f,
                dataset_revision=getattr(dataset, 'revision', None),
                quality_artifact_path=artifact_path,
                color=self.render_service.facade_color_for(f, facade_no),
            )
            f['quality_status'] = 'complete'
            f['quality_report'] = quality
            for current in (self.project_operation_service.last_facade_results or []):
                if int(current.get('id', -1)) == int(f.get('id', -2)):
                    current.update({'quality_status': 'complete',
                                    'quality_report': quality,
                                    'dataset_revision': getattr(dataset, 'revision', None)})
                    break
            self.heatmap_button_refresh_requested.emit()
        except Exception as exc:
            print(f'[PCFD] quality.persist_failed facade_id={facade_no} error={exc!r}', flush=True)
            self.warning_requested.emit(f'算法已完成，但结果保存失败：{exc}')
            self.report_preview_refresh_requested.emit()
            return
        self.report_preview_refresh_requested.emit()
        QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
