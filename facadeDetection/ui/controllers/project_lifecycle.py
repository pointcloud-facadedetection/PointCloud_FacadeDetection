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

    def __init__(self, project_overview_service, pointcloud_service,
                 station_service, project_operation_service, render_service,
                 facade_quality_controller, context_provider, parent=None):
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
        self.project_generation = 0
        self._load_in_progress = False
        self._active_load_worker = None

    def start_load(self, operation, project_id, *, file_paths=None,
                   directory=None, project=None):
        # TODO(性能/响应性): _start_load：优化点云加载。
        """在GUI线程上运行合并后的传统管道。

        FileService 目前会在加载过程中注册 Open3D 几何体。
        在 QRunnable 中运行它会触发 Open3D 的 GUI 线程保护机制。保留
        面向未来拆分管道的 worker API。
        """
        closing, _ = self._context_provider()
        if closing:
            return

        if self._load_in_progress:
            self.info_requested.emit('点云加载', '已有加载任务正在执行，请稍候。')
            return
        self._load_in_progress = True
        try:
            self.status_message.emit('正在加载点云，请稍候...', 0)
            if operation == 'activate':
                self.activate_project(project)
            elif operation == 'upload':
                before_ids = {row.id for row in self.station_service.list_stations()}
                uploaded = self.project_overview_service.upload_files(file_paths, project_id)
                if uploaded:
                    # 只同步站点投影；已有运行时 dataset 保持不变。
                    self.station_service.refresh()
                    stations = self.station_service.list_stations()
                    new_station = next(
                        (row for row in reversed(stations) if row.id not in before_ids),
                        None,
                    )
                    if new_station is not None:
                        self.station_service.show_single(new_station)
                        self.station_panel_refresh_requested.emit(new_station.id)
                else:
                    self.warning_requested.emit('直接上传文件', '未成功绑定任何点云文件。')
            elif operation == 'fls':
                before_ids = {row.id for row in self.station_service.list_stations()}
                payload = self.project_overview_service.import_fls_directory(directory, project_id)
                if payload.get('success'):
                    # FLS 导入功能已实现资产的持久化存储和同步
                    self.station_service.refresh()
                    new_station = next(
                        (row for row in reversed(self.station_service.list_stations())
                         if row.id not in before_ids), None)
                    if new_station is not None:
                        self.station_service.show_single(new_station)
                        self.station_panel_refresh_requested.emit(new_station.id)
                else:
                    self.warning_requested.emit('FLS 导入', payload.get('message', '导入失败'))
            self.project_list_refresh_requested.emit()
        except Exception as exc:
            self.on_load_failed(self.project_generation, str(exc))
        finally:
            self._load_in_progress = False
            self.status_cleared.emit()

    def on_load_failed(self, generation, error):
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self.status_message.emit('点云加载失败', 5000)
        self.warning_requested.emit('点云加载', error)

    def on_load_finished(self, generation, operation, project_id, project, result):
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self.status_cleared.emit()
        if operation == 'activate' and project is not None:
            self.project_list_refresh_requested.emit()
            self.activate_project(project)
        elif operation == 'upload':
            uploaded = result.get('uploaded') or []
            if uploaded:
                self.project_list_refresh_requested.emit()
                # 上传完成后只同步站点投影，不重新激活项目
                self.station_service.refresh()
                self.station_panel_refresh_requested.emit(None)
                self.status_message.emit(
                    f'已增量添加 {len(uploaded)} 个文件，已有站点资源未重新加载。', 5000)
            else:
                self.warning_requested.emit('直接上传文件', '未成功绑定任何点云文件。')
        elif operation == 'fls':
            payload = result.get('result') or {}
            if payload.get('success'):
                self.project_list_refresh_requested.emit()
                self.station_service.refresh()
                self.station_panel_refresh_requested.emit(None)
                self.status_message.emit(
                    f'已增量导入 {payload.get("uploaded", 0)} 个站点，已有资源未重新加载。', 5000)
            else:
                self.warning_requested.emit('FLS 导入', payload.get('message', '导入失败'))

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
