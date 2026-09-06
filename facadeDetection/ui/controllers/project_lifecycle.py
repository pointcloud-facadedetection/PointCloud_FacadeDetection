from PySide6.QtCore import QObject, Signal

from ui.main_window_config import PAGE_DEFINITIONS


class ProjectLifecycleController(QObject):
    """项目加载/激活/销毁编排：持有加载与项目代际状态，UI 反馈经 Signal 发回。"""

    info_requested = Signal(str, str)        # (标题, 正文)
    warning_requested = Signal(str, str)     # (标题, 正文)
    status_message = Signal(str, int)        # 状态栏文本与超时（毫秒，0 表示持续）
    status_cleared = Signal()                # 清空状态栏
    station_panel_refresh_requested = Signal(object)   # active_station_id 或 None
    project_list_refresh_requested = Signal()
    facade_list_reset_requested = Signal()   # 清空立面列表（UI 侧含控件存在性保护）
    station_list_reset_requested = Signal()  # 清空站点列表（UI 侧含控件存在性保护）
    current_project_change_requested = Signal(object)
    report_preview_refresh_requested = Signal()
    facade_results_refresh_requested = Signal(object)  # 立面结果列表
    page_change_requested = Signal(int)
    load_started = Signal()               # upload/fls 后台计算段开始（显示加载窗口）
    load_progress = Signal(int, str)      # (百分比, 进度文本)
    load_finished = Signal()              # 加载会话结束：完成/失败/被取消（关闭加载窗口）

    def __init__(self, project_overview_service, pointcloud_service,
                 station_service, project_operation_service, render_service,
                 facade_quality_controller, context_provider, parent=None,
                 load_pool=None):
        super().__init__(parent)
        self.project_overview_service = project_overview_service
        self.pointcloud_service = pointcloud_service
        self.station_service = station_service
        self.project_operation_service = project_operation_service
        self.render_service = render_service
        # 项目销毁需要直接复位质量域状态，显式持有质量 controller。
        self.facade_quality_controller = facade_quality_controller
        # context_provider() -> (closing, current_project_id)，实时读取。
        self._context_provider = context_provider
        # upload/fls 的计算段经该池（maxThreadCount=1）后台执行；为 None 时
        # worker 内联同步运行（测试/无池环境），信号直连保证回调仍在调用线程。
        self.load_pool = load_pool
        self.project_generation = 0
        self._load_in_progress = False
        self._active_load_worker = None

    def start_load(self, operation, project_id, *, file_paths=None,
                   directory=None, project=None):
        """upload/fls：计算段（解析/注册/转换）后台执行，Open3D 提交段回 GUI。

        worker 只做准备段；完成信号经队列投递回 GUI 线程后，由
        on_load_finished 执行提交段（show_point_cloud）与站点展示
        （show_single），project_generation 丢弃迟到结果。
        activate 保留 GUI 线程同步事务：激活有"先成功再改 UI"的回滚语义，
        且 restore_view 的渲染与站点代理加载交织，拆分风险大于收益。
        """
        closing, _ = self._context_provider()
        if closing:
            return

        if self._load_in_progress:
            self.info_requested.emit('点云加载', '已有加载任务正在执行，请稍候。')
            return
        self._load_in_progress = True
        self.status_message.emit('正在加载点云，请稍候...', 0)
        if operation == 'activate':
            try:
                self.activate_project(project)
                self.project_list_refresh_requested.emit()
            except Exception as exc:
                self.on_load_failed(self.project_generation, str(exc))
            finally:
                self._load_in_progress = False
                self.status_cleared.emit()
            return

        before_ids = {row.id for row in self.station_service.list_stations()}
        generation = self.project_generation
        try:
            worker = self.project_overview_service.create_load_worker(
                operation, project_id,
                file_paths=file_paths, directory=directory)
        except Exception as exc:
            self._load_in_progress = False
            self.on_load_failed(generation, str(exc))
            return
        self._active_load_worker = worker
        worker.signals.progress.connect(
            lambda pct, text: (
                self.status_message.emit(text, 0),
                self.load_progress.emit(pct, text),
            ))
        worker.signals.finished.connect(
            lambda result: self.on_load_finished(
                generation, operation, project_id, result, before_ids))
        worker.signals.failed.connect(
            lambda error: self.on_load_failed(generation, error))
        self.load_started.emit()
        if self.load_pool is None:
            worker.run()
        else:
            self.load_pool.start(worker)

    def on_load_failed(self, generation, error):
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self._load_in_progress = False
        self.load_finished.emit()
        self.status_message.emit('点云加载失败', 5000)
        self.warning_requested.emit('点云加载', error)

    def on_load_finished(self, generation, operation, project_id, result,
                         before_ids=None):
        """GUI 线程完成回调：提交段 + 站点展示。代际不符的迟到结果直接丢弃。"""
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self._load_in_progress = False
        self.load_finished.emit()
        self.status_cleared.emit()
        try:
            result = result or {}
            if operation == 'upload':
                uploaded = result.get('uploaded') or []
                if uploaded:
                    # Open3D 提交段必须回 GUI 线程（open3d_adapter 的 owner 校验）
                    self.project_overview_service.commit_prepared_uploads(
                        result.get('prepared') or [])
                    # 只同步站点投影；已有运行时 dataset 保持不变。
                    self.station_service.refresh()
                    new_station = self._find_new_station(before_ids)
                    if new_station is not None:
                        self.station_service.show_single(new_station)
                        self.station_panel_refresh_requested.emit(new_station.id)
                else:
                    self.warning_requested.emit('直接上传文件', '未成功绑定任何点云文件。')
            elif operation == 'fls':
                payload = result.get('result') or {}
                if payload.get('success'):
                    # FLS 导入功能已实现资产的持久化存储和同步
                    self.station_service.refresh()
                    new_station = self._find_new_station(before_ids)
                    if new_station is not None:
                        self.station_service.show_single(new_station)
                        self.station_panel_refresh_requested.emit(new_station.id)
                else:
                    self.warning_requested.emit('FLS 导入', payload.get('message', '导入失败'))
            self.project_list_refresh_requested.emit()
        except Exception as exc:
            self.on_load_failed(generation, str(exc))

    def _find_new_station(self, before_ids):
        stations = self.station_service.list_stations()
        return next(
            (row for row in reversed(stations)
             if before_ids is None or row.id not in before_ids),
            None,
        )

    def activate_project(self, project):
        # TODO(生命周期): _activate_project：审查代码的生命周期和异常处理，确保在项目切换、导入和恢复时不会泄漏资源或导致 GUI 状态不一致。
        self.facade_list_reset_requested.emit()
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            raise ValueError('项目标识为空，无法恢复项目')
        # 所有入口点（打开目录、项目选择器、上传和 FLS）均使用相同的严格激活事务。
        # 请在修改current_project 之前执行此操作，以免恢复失败时导致用户界面处于错误的激活状态。
        self.project_overview_service.activate_project(project_uuid)
        try:
            self.pointcloud_service.set_project(project_uuid)
            self.station_service.set_project(project_uuid)
            self.station_panel_refresh_requested.emit(None)
            if not self.station_service.list_stations():
                self.render_service.clear_scene_display()
                self.current_project_change_requested.emit(project)
                self.status_message.emit('项目已打开，但未发现可用 PLY 站点。', 5000)
                return
            self.station_service.restore_view()
        except Exception as exc:
            self.render_service.clear_scene_display()
            raise RuntimeError(f'站点恢复失败：{exc}') from exc
        self.current_project_change_requested.emit(project)
        self.report_preview_refresh_requested.emit()
        try:
            # 将活动项目的 UUID 传递给操作调度程序，以实现 DAL 持久化
            self.project_operation_service.set_active_project_uuid(project_uuid)
        except Exception:
            pass
        try:
            if project_uuid:
                active_station_id = getattr(self.station_service, '_active_station_id', None)
                historical = self.project_overview_service.load_historical_facades(
                    project_uuid, active_station_id)
                # 通过与新检测相同的状态路径恢复历史立面。
                # 这将同步列表、渲染器缓存、热力图可用性及报告快照
                self.project_operation_service.last_facade_results = historical or []
                self.facade_results_refresh_requested.emit(historical or [])
                self.report_preview_refresh_requested.emit()
        except Exception as exc:
            self.status_message.emit(f'项目历史数据恢复部分失败：{exc}', 5000)
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.page_change_requested.emit(operation_index)

    def prepare_project_activation(self, project_id):
        # TODO(生命周期): _prepare_project_activation：需要统一旧项目异步任务取消、等待和资源释放顺序，核查重复分支及切换竞态。
        """Dispose the old session before a restore/import loads new arrays."""
        _, current_id = self._context_provider()
        if current_id == project_id and current_id is not None:
            self.dispose_project_runtime()
        else:
            self.dispose_project_runtime()
        self.project_generation += 1

    def dispose_project_runtime(self):
        # TODO(内存/生命周期): _dispose_project_runtime：建立可验证的项目资源释放清单。
        """Single GUI-thread disposal gate for project switches and close."""
        self._load_in_progress = False
        # 取消仍在后台执行的准备段；其迟到结果由 project_generation 门控丢弃。
        worker = self._active_load_worker
        if worker is not None:
            worker.cancel()
            self._active_load_worker = None
            # 被取消的 worker 不会再走完成/失败回调，加载窗口须在此关闭
            self.load_finished.emit()
        try:
            self.project_operation_service.invalidate_async_jobs()
        except Exception:
            pass
        self.facade_quality_controller.active_quality_worker = None
        self.facade_quality_controller.quality_result_cache.clear()
        self.facade_quality_controller.quality_reports.clear()
        try:
            self.project_operation_service.clear_processing_state()
        except Exception:
            pass
        try:
            self.render_service.clear_runtime()
        except Exception:
            pass
        try:
            self.pointcloud_service.close_project()
        except Exception:
            pass
        try:
            self.render_service.clear_viewport()
        except Exception:
            pass
        self.facade_list_reset_requested.emit()
        self.station_list_reset_requested.emit()
