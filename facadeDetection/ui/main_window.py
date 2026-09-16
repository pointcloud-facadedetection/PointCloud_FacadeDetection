from dataclasses import replace
from pathlib import Path
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QMainWindow,
    QMessageBox,
)

from .main_window_config import APPLICATION_TITLE
from .pages.overview_page import OverviewPageMixin
from .pages.scaffold_page import ScaffoldPageMixin
from .pages.operation_page import OperationPageMixin
from .pages.inspection_review_page import InspectionReviewPageMixin
from .pages.report_page import ReportPageMixin
from .controllers.facade_quality import FacadeQualityController
from .controllers.registration import RegistrationController
from .controllers.project_lifecycle import ProjectLifecycleController
from .controllers.model_export import ModelExportController
from .controllers.step_navigation import StepNavigationController
from .controllers.task_progress import (
    TASK_LOAD,
    TaskProgressController,
)
from services.inspection_review import InspectionReviewService
from services.project_operation import ProjectOperationService
from services.project_overview import ProjectOverviewService
from services.viewport_render_service import ViewportRenderService
from services.viewport_render_facade import ViewportRenderFacade
from runtime.lifecycle import RuntimeLifecycle
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from services.facade.facade_service import FacadeService
from services.report_export import ReportExportService
from config.storage import Storage
from view3d.open3d_viewport import Open3DViewport




class MainWindow(OverviewPageMixin, OperationPageMixin,
                 InspectionReviewPageMixin, ReportPageMixin,
                 ScaffoldPageMixin,
                 QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName('mainWindow')
        # 完全隐藏 Windows 非客户区，避免缩放边框在深色标题栏上方露出白条。
        # 窗口缩放由八个透明 WindowResizeHandle 继续交给系统完成。
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        self.setWindowTitle(APPLICATION_TITLE)
        self.resize(1600, 900)
        # 三维工作台在过窄尺寸下失去可用性；该下限同时保证命令栏和四页签不溢出。
        self.setMinimumSize(960, 640)
        self.viewport = Open3DViewport()
        # Unified render service for business modules
        self.render_service = ViewportRenderService(self.viewport, db=None)
        self.render_facade = ViewportRenderFacade(self.render_service)
        self.project_overview_service = ProjectOverviewService(self.viewport, self.render_service, db=None)
        # 使用 ruiqi_dev 最新服务编排，UI 只负责展示与交互。
        self.pointcloud_service = PointCloudService(self.viewport, self.render_service)
        self.facade_service = FacadeService(
            self.viewport,
            db=None,
            render_service=self.render_service,
            pointcloud_service=self.pointcloud_service,
        )
        self.facade_service.set_pointcloud_service(self.pointcloud_service)
        self.render_service.pointcloud_service = self.pointcloud_service
        self.project_operation_service = ProjectOperationService(
            self.viewport,
            facade_service=self.facade_service,
            pointcloud_service=self.pointcloud_service,
            render_service=self.render_service,
        )
        self.station_service = PointCloudStationService(
            self.render_service, pointcloud_service=self.pointcloud_service)
        self.project_operation_service.set_station_service(self.station_service)
        self.inspection_review_service = InspectionReviewService()
        self.report_export_service = ReportExportService()
        self.current_project = None
        self.header_buttons = {}
        self.page_header_layouts = {}
        # 左侧"站点/点云显示"面板默认收拢：外立面检测的主视线是三维视口，
        # 左栏只在需要切换站点或调点云显示参数时展开。右栏（检测结果）
        # 是常态可用的复核面板，因此保持展开。
        self._sidebar_collapsed = {'left': True, 'right': False}
        # 默认定位到 data/projects，便于跨机迁移
        try:
            Storage.ensure_base_dirs()
            self._last_upload_directory = str(Storage.PROJECTS_ROOT)
        except Exception:
            self._last_upload_directory = str(Path.home())
        self._header_resize_pending = set()
        self._current_report_pdf_name = None
        self._report_navigation_index = 0
        self._report_webview_error = None
        # 质量域状态（_quality_reports/_quality_result_cache/_active_quality_worker
        # 等）由 FacadeQualityController 持有，经下方同名 property 委托访问。
        self._report_snapshot = {'project': {}, 'facades': []}
        # 质量结果窗口采用非阻塞打开方式；必须由主窗口持有引用，避免窗口被
        # Python 垃圾回收，同时避免再次进入 QDialog.exec() 的嵌套事件循环。
        self._quality_dialog = None
        self._runtime = RuntimeLifecycle(self, max_thread_count=1)
        self._quality_pool = self._runtime.pool('quality')
        self._registration_pool = self._runtime.pool('registration')
        self._load_pool = self._runtime.pool('load')
        self._load_cancel_button = None
        # 加载/项目代际状态全局唯一的模态进度控制器：所有重要耗时操作共享
        # 同一套"模态弹窗 + 每 5 秒刷新一次进度 + 终态强制收尾"的进度体验。
        self.task_progress = TaskProgressController(self)
        self.step_nav_controller = StepNavigationController(
            self, self.task_progress, parent=self)
        # 【导出模型】编排控制器：站点选择、后台 voxel=0.2 下采样导出 PLY、
        # 进度弹窗与结果提示都收拢在控制器内；报告页只保留一次点击的入口。
        self.model_export_controller = ModelExportController(
            station_service=self.station_service,
            pool=self._load_pool,
            project_provider=lambda: getattr(self, 'current_project', None),
            progress=self.task_progress,
            parent=self,
        )
        self.model_export_controller.status_message.connect(
            lambda message, timeout: self.statusBar().showMessage(
                message, timeout))
        self.model_export_controller.warning_requested.connect(
            lambda message: QMessageBox.warning(self, '导出模型', message))
        self.model_export_controller.info_requested.connect(
            lambda message: QMessageBox.information(self, '导出模型', message))
        self._pending_station_selection = {}
        self._station_selection_timer = QTimer(self)
        self._station_selection_timer.setSingleShot(True)
        self._station_selection_timer.setInterval(160)
        self._station_selection_timer.timeout.connect(self._flush_station_selection)
        # 配准编排移入独立 controller；按钮使能、弹窗、视口选点等 UI 反馈
        # 全部经信号接回本窗口，worker 状态由 controller 自行持有。
        self.registration_controller = RegistrationController(
            self.station_service,
            self.render_service,
            self._registration_pool,
            self._registration_context,
            parent=self,
        )
        self._connect_registration_controller()
        # 质量编排移入独立 controller；对话框、立面列表、热力图按钮等 UI 反馈
        # 全部经信号接回本窗口。参数快照与采样间距依赖 UI 控件读数，
        # 以可调用对象注入，由 controller 在计算发起时实时取数。
        self.facade_quality_controller = FacadeQualityController(
            facade_service=self.facade_service,
            render_facade=self.render_facade,
            render_service=self.render_service,
            pointcloud_service=self.pointcloud_service,
            project_operation_service=self.project_operation_service,
            station_service=self.station_service,
            pool=self._quality_pool,
            context_provider=self._quality_context,
            profile_provider=self._quality_profile_provider,
            grid_size_provider=self._quality_grid_size,
            progress=self.task_progress,
            parent=self,
        )
        self._connect_facade_quality_controller()
        # 项目加载/激活/销毁编排移入独立 controller；项目列表、页签、标题、
        # 状态栏等 UI 反馈全部经信号接回本窗口。销毁链路需要直接复位质量域
        # 状态，故显式注入 facade_quality_controller。
        self.lifecycle_controller = ProjectLifecycleController(
            project_overview_service=self.project_overview_service,
            pointcloud_service=self.pointcloud_service,
            station_service=self.station_service,
            project_operation_service=self.project_operation_service,
            render_service=self.render_service,
            facade_quality_controller=self.facade_quality_controller,
            context_provider=self._lifecycle_context,
            load_pool=self._load_pool,
            parent=self,
        )
        self._connect_lifecycle_controller()
        # service 层的信息弹窗与取色交互上移到本窗口（时机与文案不变）。
        self.project_operation_service.info_requested.connect(
            self._show_operation_info)
        self.project_operation_service.color_pick_requested.connect(
            self._pick_scene_color)
        self._setup_ui()
        self._create_resize_handles()
        self._connect_buttons()
        # 步骤 √ 只有在业务真正完成后才回填，故绑定放在控件与命令连接就绪之后。
        self._bind_step_completion_signals()
        # Hook: 当结果准备就绪时，在右侧停靠栏中显示立面统计数据
        try:
            self.project_operation_service.on_facade_results = self._show_facade_results
        except Exception:
            pass
        self._refresh_project_list()
        self._set_current_project(None)

    # 步骤导航常量别名：operation_page 等经 self.STEP_* 访问。
    STEP_DATA = StepNavigationController.STEP_DATA
    STEP_BOX_DETECT = StepNavigationController.STEP_BOX_DETECT
    STEP_QUALITY = StepNavigationController.STEP_QUALITY

    def _install_step_nav(self, header_layout):
        """装配步骤导航栏（实现见 StepNavigationController）。"""
        self.step_nav_controller.install(header_layout)

    def _begin_step_task(self, index):
        """为耗时步骤立起忙碌条（operation_page 复用此入口）。"""
        self.step_nav_controller._begin_step_task(index)

    def _bind_step_completion_signals(self):
        """把 √/✗ 徽标与进度窗收尾挂到服务层（StepNavigationController）。"""
        self.step_nav_controller.bind_completion_signals()


    def _connect_buttons(self):
        overview_actions = {
            'btn_open_project': self._open_project_directory,
            'btn_select_project': self._select_project,
            'btn_new_project': self._create_project,
        }
        pointcloud_actions = {
            'btn_denoise': self.project_operation_service.denoise,
            'btn_registration': self.registration_controller.run_station_registration,
            'btn_select_detection_area': (
                self.project_operation_service.select_detection_area
            ),
            'btn_facade_detection': (
                self.project_operation_service.facade_detection
            ),
            'btn_quality_inspection': (
                self.project_operation_service.quality_inspection
            ),
            'btn_box_segmentation': (
                self.project_operation_service.box_segmentation
            ),
            'btn_calculate_detail': (
                self.project_operation_service.calculate_detail
            ),
            'btn_align_2d_3d': self.project_operation_service.align_2d_3d,
        }
        report_actions = {
            'btn_open_report_pdf': self._open_report_pdf,
            'btn_export_quality_report': self._export_quality_report,
        }
        all_actions = {
            **overview_actions,
            **pointcloud_actions,
            **report_actions,
        }
        for button_name, callback in all_actions.items():
            self.header_buttons[button_name].clicked.connect(callback)

        self.left_sidebar_button.clicked.connect(
            lambda: self._toggle_sidebar('left')
        )
        self.right_sidebar_button.clicked.connect(
            lambda: self._toggle_sidebar('right')
        )

    def _registration_context(self):
        # 供 RegistrationController 实时判断配准结果是否已过期。
        return (
            self._project_generation,
            getattr(self.current_project, 'project_id', None),
            getattr(self, '_closing', False),
        )

    def _connect_registration_controller(self):
        controller = self.registration_controller
        controller.busy_changed.connect(self._set_registration_buttons_enabled)
        controller.status_message.connect(self._show_registration_status)
        controller.warning_requested.connect(self._show_registration_warning)
        controller.info_requested.connect(self._show_registration_info)
        controller.station_panel_refresh_requested.connect(
            self._refresh_station_panel)
        controller.manual_pick_requested.connect(
            self._enter_registration_pick_mode)
        controller.manual_icp_prompt_requested.connect(
            self._prompt_manual_registration_icp)

    def _set_registration_buttons_enabled(self, enabled):
        button = self.header_buttons.get('btn_registration')
        if button is not None:
            button.setEnabled(enabled)

    def _show_registration_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def _show_registration_warning(self, message):
        QMessageBox.warning(self, '点云配准', message)

    def _show_registration_info(self, message):
        QMessageBox.information(self, '点云配准', message)

    def _enter_registration_pick_mode(self, source_cloud, target_cloud):
        self.render_service.enter_registration_pick_mode(
            source_cloud, target_cloud, self._on_registration_pick,
            pick_radius=10)

    def _on_registration_pick(self, _picked, source_next):
        self.registration_controller.handle_pick_updated(source_next)

    def _prompt_manual_registration_icp(self, pairs):
        answer = QMessageBox.question(
            self, '点云配准',
            f'已选择 {pairs} 对对应点，是否执行人工初值 ICP？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.registration_controller.start_manual_registration()

    # 项目代际计数由 ProjectLifecycleController 持有。注册/质量 controller 的
    # context_provider 经此 property 读取，语义与原实例属性一致。
    @property
    def _project_generation(self):
        return self.lifecycle_controller.project_generation

    @_project_generation.setter
    def _project_generation(self, value):
        self.lifecycle_controller.project_generation = value

    def _lifecycle_context(self):
        # 供 ProjectLifecycleController 实时读取关闭标志与当前项目标识。
        return (
            getattr(self, '_closing', False),
            getattr(self.current_project, 'project_id', None),
        )

    def _connect_lifecycle_controller(self):
        controller = self.lifecycle_controller
        controller.info_requested.connect(self._show_lifecycle_info)
        controller.warning_requested.connect(self._show_lifecycle_warning)
        controller.status_message.connect(self._show_lifecycle_status)
        controller.status_cleared.connect(self._clear_lifecycle_status)
        controller.load_started.connect(self._show_loading_dialog)
        controller.load_progress.connect(self._update_loading_dialog)
        controller.load_finished.connect(self._hide_loading_dialog)
        controller.station_panel_refresh_requested.connect(
            self._refresh_station_panel)
        controller.project_list_refresh_requested.connect(
            self._refresh_project_list)
        controller.facade_list_reset_requested.connect(self._reset_facade_list)
        controller.station_list_reset_requested.connect(self._reset_station_list)
        controller.current_project_change_requested.connect(
            self._set_current_project)
        controller.report_preview_refresh_requested.connect(
            self._refresh_report_preview)
        controller.facade_results_refresh_requested.connect(
            self._show_facade_results)
        controller.page_change_requested.connect(self.set_current_page)

    def _show_lifecycle_info(self, title, message):
        QMessageBox.information(self, title, message)

    def _show_lifecycle_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def _show_lifecycle_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def _clear_lifecycle_status(self):
        self.statusBar().clearMessage()

    # 点云加载/导入/激活：统一走 TaskProgressController 的 TASK_LOAD 任务
    def _show_loading_dialog(self):
        self.task_progress.begin(TASK_LOAD, '点云加载', '任务处理中')
        # 兼容旧属性：历史代码/测试通过 _loading_dialog 探测控件。
        self._loading_dialog = self.task_progress.dialog(TASK_LOAD)

    def _update_loading_dialog(self, percent, text):
        # 忙碌条只透传阶段文案，百分比忽略。
        self.task_progress.report(TASK_LOAD, percent, text or '任务处理中')

    def _hide_loading_dialog(self, success=True, message='处理完成'):
        viewport = getattr(self, 'viewport', None)
        if (success and viewport is not None
                and hasattr(viewport, 'has_clouds') and viewport.has_clouds()):
            # 有点云时等首帧真正上屏再收弹窗，避免"弹窗消失到模型出现"卡顿
            self._finish_task_after_first_frame(TASK_LOAD, True, message)
        else:
            self.task_progress.finish(TASK_LOAD, bool(success), message)

    def _finish_task_after_first_frame(self, task_key, success=True,
                                       message='处理完成', timeout_ms=2500):
        """等视口首帧渲染完成后收尾任务弹窗；超时兜底防挂死。"""
        viewport = getattr(self, 'viewport', None)
        adapter = getattr(viewport, '_adapter', None)
        if adapter is None:
            self.task_progress.finish(task_key, success, message)
            return
        start_frames = getattr(adapter, '_frames_rendered', 0)
        started_at = time.monotonic()

        def _check():
            if (getattr(adapter, '_frames_rendered', 0) > start_frames
                    or (time.monotonic() - started_at) * 1000 >= timeout_ms):
                self.task_progress.finish(task_key, success, message)
                return
            QTimer.singleShot(50, _check)

        QTimer.singleShot(50, _check)


    def _reset_facade_list(self):
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()

    def _reset_station_list(self):
        if hasattr(self, 'station_list'):
            self.station_list.blockSignals(True)
            self.station_list.clear()
            self.station_list.blockSignals(False)

    def _start_load(self, operation, project_id, *, file_paths=None,
                    directory=None, directories=None, project=None):
        self.lifecycle_controller.start_load(
            operation, project_id,
            file_paths=file_paths, directory=directory, directories=directories,
            project=project)

    def _prepare_project_activation(self, project_id):
        self.lifecycle_controller.prepare_project_activation(project_id)

    def _activate_project(self, project):
        self.lifecycle_controller.activate_project(project)

    def _show_operation_info(self, title, message):
        QMessageBox.information(self, title, message)

    def _pick_scene_color(self):
        # 从 service 上移的取色弹窗；取消或选色无效时不做任何事。
        dlg = QColorDialog()
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        if not dlg.exec():
            return
        qcol = dlg.selectedColor()
        if not qcol.isValid():
            return
        color = (qcol.redF(), qcol.greenF(), qcol.blueF())
        self.project_operation_service.apply_global_color(color)

    def _set_current_project(self, project):
        self.current_project = project
        has_project = project is not None

        for page_key, button in self.page_buttons.items():
            button.setEnabled(page_key == 'project_overview' or has_project)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(
                f'{project.name}\n{project.directory_path}')
            self._apply_project_inspection_params(project)
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_workspace()
        self._refresh_report_preview()
        self._update_window_title()

    def _apply_project_inspection_params(self, project):
        """当项目被激活时，将创建项目时录入的检测参数恢复到 UI 控件。"""
        import json
        from services.inspection_profile import InspectionProfileService

        raw = getattr(project, 'inspection_params_json', None)
        if not raw:
            return
        try:
            p = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return

        # 标准
        std_id = p.get('standard_id')
        if std_id:
            idx = self.standard_combo.findData(std_id)
            if idx >= 0:
                self.standard_combo.blockSignals(True)
                self.standard_combo.setCurrentIndex(idx)
                self.standard_combo.blockSignals(False)
                self._on_standard_changed(idx)

        # 区间
        interval = p.get('interval_size_m')
        if interval is not None:
            idx = self.interval_combo.findData(float(interval))
            if idx >= 0:
                self.interval_combo.blockSignals(True)
                self.interval_combo.setCurrentIndex(idx)
                self.interval_combo.blockSignals(False)

        # 质量参数控件
        def _set_spin(spin, value):
            if value is not None and hasattr(spin, 'setValue'):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

        _set_spin(self.quality_length_spin, p.get('ruler_length_m'))
        # 操作页“滑移步距”对应创建弹窗的纵向采样步长；横向步长
        # 对应 InspectionProfile.step_size_m，不应覆盖滑移步距。
        _set_spin(self.quality_step_spin, p.get('step_longitudinal_m'))
        _set_spin(self.quality_width_spin, p.get('ruler_width_m'))
        _set_spin(self.quality_select_band_spin, p.get('select_band_m'))
        _set_spin(self.quality_hole_band_spin, p.get('hole_band_m'))
        _set_spin(self.quality_bin_size_spin, p.get('bin_size_m'))
        _set_spin(self.quality_top_q_spin, p.get('top_q'))
        _set_spin(self.quality_sor_sigma_spin, p.get('sor_sigma'))
        _set_spin(self.quality_sor_k_spin, p.get('sor_k'))
        _set_spin(self.quality_sor_w_weight_spin, p.get('sor_w_weight'))
        _set_spin(self.quality_max_hole_ratio_spin, p.get('max_hole_ratio'))
        _set_spin(self.quality_min_points_spin, p.get('min_points'))

        # SOR 启用
        sor_enabled = p.get('sor_enabled')
        if sor_enabled is not None and hasattr(self, 'quality_sor_check'):
            self.quality_sor_check.blockSignals(True)
            self.quality_sor_check.setChecked(bool(sor_enabled))
            self.quality_sor_check.blockSignals(False)

        # SOR 方法
        sor_method = p.get('sor_method')
        if sor_method and hasattr(self, 'quality_sor_method_combo'):
            idx = self.quality_sor_method_combo.findData(str(sor_method))
            if idx >= 0:
                self.quality_sor_method_combo.blockSignals(True)
                self.quality_sor_method_combo.setCurrentIndex(idx)
                self.quality_sor_method_combo.blockSignals(False)

        # 同步 _inspection_profile 到项目参数快照
        profile = InspectionProfileService.get(std_id) if std_id else None
        if profile is not None:
            aliases = {
                'ruler_length_m': 'measure_height_m',
                'step_longitudinal_m': 'scan_step_m',
                'step_transverse_m': 'step_size_m',
            }
            snapshot = {}
            for source, target in aliases.items():
                if p.get(source) is not None:
                    snapshot[target] = p[source]
            for key in (
                    'interval_size_m', 'ruler_width_m', 'select_band_m',
                    'hole_band_m', 'flatness_limit_mm',
                    'verticality_limit_mm', 'sor_enabled', 'sor_sigma',
                    'sor_k', 'sor_method', 'sor_w_weight'):
                if p.get(key) is not None and hasattr(profile, key):
                    snapshot[key] = p[key]
            self._inspection_profile = replace(profile, **snapshot)

    def closeEvent(self, event):
        # TODO(生命周期/稳定性): closeEvent：核查关闭期间线程池短超时、原生窗口销毁和 Qt 退出顺序，避免后台任务继续访问已销毁视口导致未响应。
        """严格按顺序销毁，先停 timer，再断信号，再销毁 viewport，最后退出 app。"""
        if getattr(self, '_closing', False):
            event.accept()
            return
        self._closing = True
        # Step 1: 停止 viewport timer，防止 poll_events 在销毁中调用
        timer = getattr(self.viewport, '_timer', None)
        if timer is not None:
            timer.stop()
        # Step 2: 断开 viewport 的渲染队列信号
        if hasattr(self.viewport, '_render_queue'):
            rq = self.viewport._render_queue
            rq.color.disconnect()
            rq.points.disconnect()
        # Step 3: 关闭质量结果对话框
        quality_dialog = getattr(self, '_quality_dialog', None)
        if quality_dialog is not None:
            quality_dialog.close()
            self._quality_dialog = None
        # Step 4: 由 runtime 统一清理线程池（不再接受新任务）
        runtime = getattr(self, '_runtime', None)
        if runtime is not None:
            runtime.clear()

        # Step 5: 释放项目运行时资源
        self.lifecycle_controller.dispose_project_runtime()

        # Step 6: 销毁 viewport（关键：在 Qt 窗口销毁前完成）
        try:
            if hasattr(self, 'viewport') and self.viewport is not None:
                self.viewport.destroy()
        except Exception:
            pass

        # Step 7: 等待 runtime 线程池完成（短超时）
        if runtime is not None:
            runtime.stop(100)

        # Step 8: 调用父类关闭事件
        try:
            super().closeEvent(event)
        except Exception:
            event.accept()

        # Step 9: 确保 Qt 事件循环退出（最关键）
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                    app.quit()
        except Exception:
            pass

