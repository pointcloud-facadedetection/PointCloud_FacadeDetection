from PySide6.QtCore import QObject, Signal

from ui.main_window_config import PAGE_DEFINITIONS
from utils.workers import PointCloudLoadWorker


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
    load_started = Signal()               # 后台计算段开始（显示加载窗口）
    load_progress = Signal(int, str)      # (百分比, 进度文本)
    # 加载会话结束（success, message）：完成 / 失败 / 被取消都发一次，
    # UI 侧据此决定进度条是刷到 100% 还是保留当前值，然后自动关闭弹窗。
    load_finished = Signal(bool, str)

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
        # upload/fls/activate 的计算段经该池（maxThreadCount=1）后台执行；
        # 为 None 时 worker 内联同步运行（测试/无池环境），信号直连保证回调仍在调用线程。
        self.load_pool = load_pool
        self.project_generation = 0
        self._load_in_progress = False
        self._active_load_worker = None

    def start_load(self, operation, project_id, *, file_paths=None,
                   directory=None, directories=None, project=None):
        """upload/fls/activate：计算段后台执行，Open3D 提交段回 GUI。

        worker 只做准备段；完成信号经队列投递回 GUI 线程后，由
        on_load_finished 执行提交段（upload/fls：show_point_cloud 与
        show_single；activate：commit_restore_view 与项目级 UI 状态切换），
        project_generation 丢弃迟到结果。
        activate 的事务语义：DB 激活失败不得改动 UI/渲染状态；
        set_project/站点恢复失败由 GUI 失败回调清空场景并走统一错误路径。
        """
        closing, _ = self._context_provider()
        if closing:
            return

        if self._load_in_progress:
            self.info_requested.emit('点云加载', '已有加载任务正在执行，请稍候。')
            return
        self._load_in_progress = True
        self.status_message.emit('正在加载点云，请稍候...', 0)
        generation = self.project_generation
        if operation == 'activate':
            self.facade_list_reset_requested.emit()
            project_uuid = getattr(project, 'project_id', None)
            if not project_uuid:
                self._load_in_progress = False
                self.status_cleared.emit()
                self.on_load_failed(generation, '项目标识为空，无法恢复项目')
                return
            worker = self._create_activate_worker(project, project_uuid)
            self._active_load_worker = worker
            worker.signals.progress.connect(
                lambda pct, text: (
                    self.status_message.emit(text, 0),
                    self.load_progress.emit(pct, text),
                ))
            worker.signals.finished.connect(
                lambda result: self.on_load_finished(
                    generation, operation, project_uuid, result))
            worker.signals.failed.connect(
                lambda error: self._on_activate_failed(generation, error))
            self.load_started.emit()
            if self.load_pool is None:
                worker.run()
            else:
                self.load_pool.start(worker)
            return

        before_ids = {row.id for row in self.station_service.list_stations()}
        try:
            worker = self.project_overview_service.create_load_worker(
                operation, project_id,
                file_paths=file_paths, directory=directory, directories=directories)
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

    def _create_activate_worker(self, project, project_uuid):
        """activate 计算段 worker：DB 激活 + set_project + 站点视图数据准备。

        DB 激活失败原样抛出（GUI 侧不得改动 UI/渲染状态）；set_project 与
        站点准备失败包装为“站点恢复失败”，GUI 失败回调据此清空场景。
        """
        def run(worker):
            worker.check_cancelled()
            worker.signals.progress.emit(-1, '正在激活项目数据')
            self.project_overview_service.activate_project(project_uuid)
            try:
                worker.signals.progress.emit(-1, '正在准备站点数据')
                self.pointcloud_service.set_project(project_uuid)
                self.station_service.set_project(project_uuid)
                worker.signals.progress.emit(-1, '正在恢复站点视图')
                prepared_view = self.station_service.prepare_restore_view()
            except Exception as exc:
                raise RuntimeError(f'站点恢复失败：{exc}') from exc
            return {'operation': 'activate', 'project_uuid': project_uuid,
                    'project': project, 'prepared_view': prepared_view}
        return PointCloudLoadWorker(run)

    def _on_activate_failed(self, generation, error):
        """activate 失败回调（GUI 线程）：站点恢复阶段的失败清空场景
        （与旧同步事务一致）；DB 激活失败不得触碰 UI/渲染状态。"""
        if '站点恢复失败' in str(error):
            self.render_service.clear_scene_display()
        self.on_load_failed(generation, error)

    def _commit_activate(self, result):
        """activate 提交段（GUI 线程）：渲染提交与项目级 UI 状态切换。

        站点提交失败先清空场景再抛出，由 on_load_finished 统一走
        on_load_failed；历史立面恢复失败仅降级为状态栏提示，不影响激活。
        """
        project = result.get('project')
        project_uuid = result.get('project_uuid')
        self.station_panel_refresh_requested.emit(None)
        prepared = result.get('prepared_view')
        # E57 从不在项目恢复阶段重新解析；如果其生成的 cache PLY 被清理，
        # 保留站点记录并明确提示操作者通过原始 E57 重导/重转换。
        unavailable = [row.display_name for row in self.station_service.list_stations()
                       if getattr(row, 'last_error', None) and
                       'cache PLY' in str(row.last_error)]
        if unavailable:
            self.warning_requested.emit(
                'E57 缓存缺失',
                '以下 E57 站点的 cache PLY 不存在，已跳过恢复：\n- ' +
                '\n- '.join(unavailable) +
                '\n\n请在项目中重新导入对应原始 E57 以重新生成缓存。')
        if prepared is not None and prepared[0] is None:
            # 空项目：与旧同步路径一致，跳过后续项目级状态切换
            self.render_service.clear_scene_display()
            self.current_project_change_requested.emit(project)
            self.status_message.emit('项目已打开，但未发现可用 PLY 站点。', 5000)
            return
        if prepared is not None:
            try:
                self.station_service.commit_restore_view(prepared)
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
                # 全量恢复：加载所有站点的历史立面检测结果到聚合存储
                all_historical = self.project_overview_service.load_all_historical_facades(
                    project_uuid)
                for station_id, facades in all_historical.items():
                    self.project_operation_service.set_facade_results_for_station(
                        station_id, facades)
                # 当前活动站点的立面列表同步到 UI
                active_station_id = getattr(self.station_service, '_active_station_id', None)
                active_facades = all_historical.get(int(active_station_id), []) if active_station_id else []
                self.project_operation_service.last_facade_results = active_facades
                self.facade_results_refresh_requested.emit(active_facades)
                self.report_preview_refresh_requested.emit()
        except Exception as exc:
            self.status_message.emit(f'项目历史数据恢复部分失败：{exc}', 5000)
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.page_change_requested.emit(operation_index)

    def on_load_failed(self, generation, error):
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self._load_in_progress = False
        # 失败同样落终态：保留当前百分比，便于判断停在哪一阶段。
        self.load_finished.emit(False, '处理失败')
        self.status_message.emit('点云加载失败', 5000)
        self.warning_requested.emit('点云加载', error)

    def on_load_finished(self, generation, operation, project_id, result,
                         before_ids=None):
        """GUI 线程完成回调：提交段 + 站点展示。代际不符的迟到结果直接丢弃。

        忙碌条在提交段（Open3D 提交/立面着色）完成后才关闭：先关窗再提交
        会让"弹窗消失到模型出现"之间出现一段无遮挡的 GUI 卡顿。
        """
        if generation != self.project_generation:
            return
        self._active_load_worker = None
        self._load_in_progress = False
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
            elif operation == 'activate':
                self._commit_activate(result)
            self.project_list_refresh_requested.emit()
        except Exception as exc:
            self.on_load_failed(generation, str(exc))
            return
        self.load_finished.emit(True, '处理完成')
        self.status_cleared.emit()

    def _find_new_station(self, before_ids):
        stations = self.station_service.list_stations()
        return next(
            (row for row in reversed(stations)
             if before_ids is None or row.id not in before_ids),
            None,
        )

    def activate_project(self, project):
        """打开项目的统一入口：与 start_load('activate') 同一 worker 事务。

        所有入口（打开目录、项目选择器、项目卡片）共享同一条后台激活
        管道：计算段（DB 激活/站点域准备）离开 GUI 线程，渲染提交与
        项目级 UI 状态切换在 GUI 完成回调执行，并显示加载窗口。
        """
        self.start_load('activate', getattr(project, 'project_id', None),
                        project=project)

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
            self.load_finished.emit(False, '已中止')
        if self.load_pool is not None:
            # 清掉池里排队的任务，避免重开时被已取消的僵尸 worker 堵在
            # maxThreadCount=1 的池子后面（运行中的任务由检查点自行退出）。
            self.load_pool.clear()
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
