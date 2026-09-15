"""立面质量编排控制器。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from config.storage import Storage
from ui.controllers.task_progress import (
    TASK_QUALITY,
    TASK_QUALITY_BATCH,
    TaskProgressController,
)
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
    # 批量链路：仅供状态栏/外部观察使用。进度窗由 controller 自己驱动
    batch_progress = Signal(int, int)
    batch_finished = Signal(list)
    # 单立面评估的终态信号（批量链路走 batch_finished）。
    quality_completed = Signal()
    quality_failed = Signal()

    def __init__(self, facade_service, render_facade, render_service,
                 pointcloud_service, project_operation_service,
                 station_service, pool, context_provider, profile_provider,
                 grid_size_provider, parent=None,
                 progress: Optional[TaskProgressController] = None):
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
        # 统一进度控制器：本控制器自行驱动进度窗的 begin / report / finish。
        self.progress = progress
        self.quality_reports = []
        self.quality_result_cache = {}
        self._latest_facade_results = []
        self.heatmap_mode = 'flatness'
        self._quality_request_token = 0
        self._quality_request_cache_key = None
        self.active_quality_worker = None
        self._batch_queue = []
        self._batch_results = []
        self._batch_total = 0
        self._batch_current = 0

    # ------------------------------------------------------------------
    # 进度窗：begin / report / finish
    # ------------------------------------------------------------------
    def _begin_quality_progress(self, task_key: str) -> None:
        """在**真正要启动 worker** 之前调用。

        历史结果直开、缓存命中、前置校验失败等路径严禁调用本方法，
        否则会立起一个永远等不到 finish 的空进度窗。
        """
        if self.progress is None:
            return
        self.progress.begin(task_key, '质量评估', '任务处理中')

    def _report_quality_progress(self, task_key: str, percent: int,
                                 text: str = '') -> None:
        if self.progress is None:
            return
        self.progress.report(task_key, percent, text or '任务处理中')

    def _finish_quality_progress(self, task_key: str, success: bool,
                                 message: str = '') -> None:
        if self.progress is None:
            return
        self.progress.finish(task_key, success, message)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _quality_results_dir(self, project_uuid, facade):
        """返回 results/站点名/立面ID，站点名仅用于目录显示且安全化。"""
        base = Path(Storage.ensure_project_dirs(project_uuid)['results'])
        station_id = facade.get('station_id')
        name = None
        try:
            for station in self.station_service.list_stations():
                if station_id is not None and int(station.id) == int(station_id):
                    name = station.display_name
                    break
        except Exception:
            pass
        safe = re.sub(r'[\\/*?:"<>|]', '_',
                      str(name or f'station_{station_id or "unknown"}'))
        safe = safe.strip(' ._')[:80] or 'unknown'
        # ResultExportService 在此目录下再建立 facade_xxx 子目录。
        return base / safe

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

    # ------------------------------------------------------------------
    # 单立面评估
    # ------------------------------------------------------------------
    def evaluate_facade(self, f):
        cloud = self.active_cloud_name()
        if not cloud:
            return
        project_uuid, generation = self._context_provider()
        if not project_uuid:
            self.warning_requested.emit('请先选择项目。')
            return
        # 高质量的 PNG 导出文件将保存到当前项目的结果文件夹中。
        results_dir = self._quality_results_dir(project_uuid, f)

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
        if (facade_copy.get('quality_status') == 'complete'
                and isinstance(historical_quality, dict)):
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
        self._begin_quality_progress(TASK_QUALITY)
        worker = QualityWorker(self.facade_service, cloud, facade_copy, kwargs)
        self.active_quality_worker = worker
        worker.signals.finished.connect(
            lambda facade, quality: self._on_quality_finished(
                token, cloud, facade, quality))
        # worker 的真实百分比直接喂进度窗（5 秒节流由 GUI 线程侧的
        # ThrottledReporter 承担，这里只做一次缓存写入）。
        worker.signals.progress.connect(
            lambda percent, text='':
            self._report_quality_progress(TASK_QUALITY, percent, text))
        self._pool.start(worker)

    # ------------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------------
    def evaluate_facades_batch(self, facades):
        """顺序复用 QualityWorker 执行多个完整立面，批量链路不发结果 Dialog。"""
        if self._batch_queue:
            self.warning_requested.emit('已有批量质量检测正在执行。')
            return
        cloud = self.active_cloud_name()
        project_uuid, generation = self._context_provider()
        if not cloud or not project_uuid or not facades:
            return
        profile = self._profile_provider()
        grid_size = float(self._grid_size_provider())
        self._batch_results = []
        self._batch_current = 0
        self._batch_total = len(facades)
        self._batch_queue = []
        for facade in facades:
            item = dict(facade)
            dataset = self.facade_service.get_dataset(cloud)
            item['__quality_request_context'] = {
                'project_uuid': project_uuid, 'project_generation': generation,
                'station_id': item.get(
                    'station_id',
                    getattr(self.station_service, '_active_station_id', None)),
                'dataset_id': getattr(dataset, 'dataset_id', None),
                'dataset_revision': getattr(dataset, 'revision', None),
                'cloud_name': cloud,
            }
            self._batch_queue.append((cloud, item, {
                'profile': profile, 'grid_size': grid_size,
                'results_dir': self._quality_results_dir(project_uuid, item),
            }))

        self._begin_quality_progress(TASK_QUALITY_BATCH)
        self.status_message.emit(
            f'开始批量质量检测，共 {self._batch_total} 个立面...', 0)
        self._run_next_batch()

    def _run_next_batch(self):
        if not self._batch_queue:
            results = self._batch_results
            self._batch_results = []
            self._finish_quality_progress(
                TASK_QUALITY_BATCH, True,
                f'已完成 {len(results)} 个立面')
            self.batch_finished.emit(results)
            self.report_preview_refresh_requested.emit()
            self.status_message.emit(
                f'批量检测完成，共 {len(results)} 个立面。', 5000)
            return
        cloud, facade, kwargs = self._batch_queue.pop(0)
        self._batch_current += 1
        self.batch_progress.emit(self._batch_current, self._batch_total)
        # 批量进度以 (current / total) 为唯一口径上报，避免与 worker
        # 内部"当前立面的窗口百分比"互相覆盖导致进度条乱跳。
        total = max(int(self._batch_total), 1)
        percent = int(round(100.0 * self._batch_current / total))
        self._report_quality_progress(TASK_QUALITY_BATCH, percent)
        self.status_message.emit(
            f'[{self._batch_current}/{self._batch_total}] 正在检测立面 '
            f'#{facade.get("display_no", facade.get("id", 0))}...', 0)
        worker = QualityWorker(self.facade_service, cloud, facade, kwargs)
        self.active_quality_worker = worker
        worker.signals.finished.connect(
            lambda f, q: self._on_batch_finished(cloud, f, q))
        # 注意：批量时不转发 worker.signals.progress —— 见上方注释。
        self._pool.start(worker)

    def _upsert_facade_to_aggregated(self, facade):
        """将单/批量完成的立面结果按 ID 更新到项目级聚合存储。"""
        station_id = facade.get('station_id')
        if (station_id is None
                or not hasattr(self.project_operation_service,
                               'set_facade_results_for_station')):
            return
        all_results = self.project_operation_service.all_facade_results.get(
            int(station_id), [])
        fid = int(facade.get('id', -1))
        updated = False
        for i, item in enumerate(list(all_results)):
            if int(item.get('id', -2)) == fid:
                all_results[i] = dict(facade)
                updated = True
                break
        if not updated:
            all_results.append(dict(facade))
        self.project_operation_service.set_facade_results_for_station(
            int(station_id), all_results)

    def _on_batch_finished(self, cloud, facade, quality):
        """批量完成处理：成功结果入库；不发单立面 Dialog 信号。"""
        try:
            valid = isinstance(quality, dict) and quality.get('ok', True)
            valid = valid and int(
                (quality.get('overall') or {})
                .get('quality_valid_window_count', 0) or 0) > 0
            if valid:
                self._export_batch_heatmaps(cloud, facade, quality)
                dataset = self.facade_service.get_dataset(cloud)
                project_uuid, _ = self._context_provider()
                self.facade_service.commit_quality_success(
                    project_uuid, int(facade.get('id', 0)), quality,
                    display_no=facade.get('display_no'), facade_data=facade,
                    dataset_revision=getattr(dataset, 'revision', None),
                    color=self.render_service.facade_color_for(
                        facade, facade.get('display_no')))
                facade['quality_status'] = 'complete'
                facade['quality_report'] = quality
                for current in (self.project_operation_service
                                .last_facade_results or []):
                    if int(current.get('id', -1)) == int(facade.get('id', -2)):
                        current.update(quality_status='complete',
                                       quality_report=quality)
                        break
                # 同步到项目级聚合存储，确保 PDF 报告预览即时刷新
                self._upsert_facade_to_aggregated(facade)
            self.quality_reports = [
                r for r in self.quality_reports
                if (r.get('facade') or {}).get('id') != facade.get('id')]
            self.quality_reports.append({'facade': facade, 'quality': quality})
            self._batch_results.append({'facade': facade, 'quality': quality})
        except Exception as exc:
            print(f'[PCFD] batch quality handling failed: {exc!r}', flush=True)
        finally:
            self.active_quality_worker = None
            self.report_preview_refresh_requested.emit()
            self._run_next_batch()

    def _export_batch_heatmaps(self, cloud, facade, quality):
        """批量模式生成与单立面相同的 PNG 资产，但不打开结果窗口。"""
        if not isinstance(quality, dict) or quality.get('__auto_exported'):
            return
        dataset = self.facade_service.get_dataset(cloud)
        ids = np.asarray(facade.get('proxy_indices') or
                         facade.get('inlier_indices') or [], dtype=np.int64)
        if len(ids) and dataset.index.has_source_mapping():
            ids = dataset.index.proxy_to_source_ids(ids, deduplicate=True)
        ids = ids[(ids >= 0) & (ids < len(dataset.processed_raw_points))]
        if not len(ids):
            return
        from services.result_export_service import ResultExportService
        points = np.asarray(dataset.processed_raw_points)[ids]
        colors = np.tile(np.asarray(self.render_service.facade_color_for(
            facade, facade.get('display_no')), dtype=float),
            (len(points), 1))
        context = quality.setdefault('__export_context', {})
        results_dir = context.get('results_dir')
        if not results_dir:
            project_uuid, _ = self._context_provider()
            results_dir = self._quality_results_dir(project_uuid, facade)
        exported = ResultExportService().export_all_heatmaps(
            results_dir, int(facade.get('display_no', facade.get('id', 0))),
            points, colors, quality)
        context['results_dir'] = str(results_dir)
        context['heatmaps'] = exported
        quality['__auto_exported'] = True

    # ------------------------------------------------------------------
    # 取消
    # ------------------------------------------------------------------
    def cancel_active_batch(self):
        """清空待执行队列 + 放弃当前 worker 的本地引用。  """
        self._batch_queue = []
        self._batch_results = []
        self._batch_total = 0
        self._batch_current = 0
        worker = self.active_quality_worker
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
            self.active_quality_worker = None

    # ------------------------------------------------------------------
    # 失败回调（外部入口）
    # ------------------------------------------------------------------
    def on_quality_failed(self, token, error):
        if token != self._quality_request_token:
            return

        self.active_quality_worker = None
        self._finish_quality_progress(TASK_QUALITY, False, '质量计算失败')
        self.report_preview_refresh_requested.emit()
        self.status_message.emit('质量计算失败', 0)
        self.warning_requested.emit(f'质量计算失败：{error}')
        self.quality_failed.emit()

    # ------------------------------------------------------------------
    # 缓存键
    # ------------------------------------------------------------------
    def _quality_cache_key(self, cloud, facade, profile, grid_size):
        facade_id = int((facade or {}).get('id', 0))
        standard_id = getattr(profile, 'standard_id', None)
        flatness_limit_mm = getattr(profile, 'flatness_limit_mm', None)
        verticality_limit_mm = getattr(profile, 'verticality_limit_mm', None)
        window_size_m = getattr(profile, 'window_size_m', None)
        step_size_m = getattr(profile, 'step_size_m', None)
        measure_height_m = getattr(profile, 'measure_height_m', None)
        domain = (facade.get('measurement_indices')
                  or facade.get('voxel_ids') or [])
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

    # ------------------------------------------------------------------
    # 单立面完成回调
    # ------------------------------------------------------------------
    def _on_quality_finished(self, token, cloud, f, quality):
        """Handle quality computation completion with full state machine. """
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
            self._finish_quality_progress(TASK_QUALITY, False, '结果已过期')
            self.status_message.emit('质量结果已过期，当前站点已变化，未保存。', 5000)
            return

        self.status_cleared.emit()

        if quality is None:
            print(f'[PCFD] ui.quality_none facade_id={facade_no}', flush=True)
            self._finish_quality_progress(TASK_QUALITY, False, '未返回结果')
            self.warning_requested.emit(
                f'立面 #{facade_no} 质量计算失败：未返回结果。请检查日志。')
            self.report_preview_refresh_requested.emit()
            self.quality_failed.emit()
            return

        if not isinstance(quality, dict):
            print(f'[PCFD] ui.quality_invalid_type facade_id={facade_no} '
                  f'type={type(quality)}', flush=True)
            self._finish_quality_progress(TASK_QUALITY, False, '返回类型异常')
            self.warning_requested.emit(
                f'立面 #{facade_no} 质量计算返回异常类型：{type(quality)}')
            self.report_preview_refresh_requested.emit()
            self.quality_failed.emit()
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

            # 算法层返回 ok=False：算法正常结束，只是结果不可用。
            self._finish_quality_progress(TASK_QUALITY, True, '评估完成')

            self.info_requested.emit(
                f'立面 #{facade_no} 质量评估结果：\n\n{error_message}')

            QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
            self.report_preview_refresh_requested.emit()
            self.quality_completed.emit()
            return

        overall = quality.get('overall') or {}
        window_count = int(overall.get('candidate_window_count', 0) or 0)
        valid_count = int(overall.get('quality_valid_window_count', 0) or 0)

        if valid_count <= 0:
            print(f'[PCFD] ui.quality_no_valid_windows facade_id={facade_no} '
                  f'candidates={window_count}', flush=True)
            self._finish_quality_progress(TASK_QUALITY, True, '评估完成（无有效窗口）')
            self.info_requested.emit(
                f'立面 #{facade_no} 质量计算完成，但未找到有效检测窗口。\n'
                f'候选窗口数：{window_count}\n'
                f'可能原因：立面尺寸过小、点云密度不足或存在大面积空洞。')
            QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
            self.report_preview_refresh_requested.emit()
            self.quality_completed.emit()
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
            # 同步到项目级聚合存储，确保 PDF 报告预览即时刷新
            self._upsert_facade_to_aggregated(f)
        except Exception as exc:
            print(f'[PCFD] quality.persist_failed facade_id={facade_no} error={exc!r}', flush=True)
            self._finish_quality_progress(TASK_QUALITY, False, '结果保存失败')
            self.warning_requested.emit(f'算法已完成，但结果保存失败：{exc}')
            self.report_preview_refresh_requested.emit()
            self.quality_failed.emit()
            return
        self._finish_quality_progress(TASK_QUALITY, True)
        self.report_preview_refresh_requested.emit()
        QTimer.singleShot(0, lambda: self.show_dialog_requested.emit(cloud, f, quality))
        self.quality_completed.emit()