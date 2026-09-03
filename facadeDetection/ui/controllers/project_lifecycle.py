from PySide6.QtWidgets import QMessageBox

from ui.main_window_config import PAGE_DEFINITIONS


class ProjectLifecycleMixin:
    def _start_load(self, operation, project_id, *, file_paths=None,
                    directory=None, project=None):
        # TODO(性能/响应性): _start_load：优化点云加载。
        """在GUI线程上运行合并后的传统管道。

        FileService 目前会在加载过程中注册 Open3D 几何体。
        在 QRunnable 中运行它会触发 Open3D 的 GUI 线程保护机制。保留
        面向未来拆分管道的 worker API。
        """
        if getattr(self, '_closing', False):
            return
        
        if getattr(self, '_load_in_progress', False):
            QMessageBox.information(self, '点云加载', '已有加载任务正在执行，请稍候。')
            return
        self._load_in_progress = True
        try:
            self.statusBar().showMessage('正在加载点云，请稍候...')
            if operation == 'activate':
                self._activate_project(project)
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
                        self._refresh_station_panel(new_station.id)
                else:
                    QMessageBox.warning(self, '直接上传文件', '未成功绑定任何点云文件。')
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
                        self._refresh_station_panel(new_station.id)
                else:
                    QMessageBox.warning(self, 'FLS 导入', payload.get('message', '导入失败'))
            self._refresh_project_list()
        except Exception as exc:
            self._on_load_failed(self._project_generation, str(exc))
        finally:
            self._load_in_progress = False
            self.statusBar().clearMessage()

    def _on_load_failed(self, generation, error):
        if generation != self._project_generation:
            return
        self._active_load_worker = None
        self.statusBar().showMessage('点云加载失败', 5000)
        QMessageBox.warning(self, '点云加载', error)

    def _on_load_finished(self, generation, operation, project_id, project, result):
        if generation != self._project_generation:
            return
        self._active_load_worker = None
        self.statusBar().clearMessage()
        if operation == 'activate' and project is not None:
            self._refresh_project_list()
            self._activate_project(project)
        elif operation == 'upload':
            uploaded = result.get('uploaded') or []
            if uploaded:
                self._refresh_project_list()
                # 上传完成后只同步站点投影，不重新激活项目
                self.station_service.refresh()
                self._refresh_station_panel()
                self.statusBar().showMessage(
                    f'已增量添加 {len(uploaded)} 个文件，已有站点资源未重新加载。', 5000)
            else:
                QMessageBox.warning(self, '直接上传文件', '未成功绑定任何点云文件。')
        elif operation == 'fls':
            payload = result.get('result') or {}
            if payload.get('success'):
                self._refresh_project_list()
                self.station_service.refresh()
                self._refresh_station_panel()
                self.statusBar().showMessage(
                    f'已增量导入 {payload.get("uploaded", 0)} 个站点，已有资源未重新加载。', 5000)
            else:
                QMessageBox.warning(self, 'FLS 导入', payload.get('message', '导入失败'))

    def _activate_project(self, project):
        # TODO(生命周期): _activate_project：审查代码的生命周期和异常处理，确保在项目切换、导入和恢复时不会泄漏资源或导致 GUI 状态不一致。
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            raise ValueError('项目标识为空，无法恢复项目')
        # 所有入口点（打开目录、项目选择器、上传和 FLS）均使用相同的严格激活事务。
        # 请在修改current_project 之前执行此操作，以免恢复失败时导致用户界面处于错误的激活状态。
        self.project_overview_service.activate_project(project_uuid)
        try:
            self.pointcloud_service.set_project(project_uuid)
            self.station_service.set_project(project_uuid)
            self._refresh_station_panel()
            if not self.station_service.list_stations():
                self.render_service.clear_scene_display()
                self._set_current_project(project)
                self.statusBar().showMessage('项目已打开，但未发现可用 PLY 站点。', 5000)
                return
            self.station_service.restore_view()
        except Exception as exc:
            self.render_service.clear_scene_display()
            raise RuntimeError(f'站点恢复失败：{exc}') from exc
        self._set_current_project(project)
        self._refresh_report_preview()
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
                self.project_operation_service._last_facade_results = historical or []
                self._show_facade_results(historical or [])
                self._refresh_report_preview()
        except Exception as exc:
            self.statusBar().showMessage(f'项目历史数据恢复部分失败：{exc}', 5000)
        operation_index = next(
            index
            for index, (_title, key) in enumerate(PAGE_DEFINITIONS)
            if key == 'project_operation'
        )
        self.set_current_page(operation_index)

    def _prepare_project_activation(self, project_id):
        # TODO(生命周期): _prepare_project_activation：需要统一旧项目异步任务取消、等待和资源释放顺序，核查重复分支及切换竞态。
        """Dispose the old session before a restore/import loads new arrays."""
        current_id = getattr(self.current_project, 'project_id', None)
        if current_id == project_id and project_id is not None:
            self._dispose_project_runtime()
        else:
            self._dispose_project_runtime()
        self._project_generation += 1

    def _dispose_project_runtime(self):
        # TODO(内存/生命周期): _dispose_project_runtime：建立可验证的项目资源释放清单。
        """Single GUI-thread disposal gate for project switches and close."""
        self._load_in_progress = False
        try:
            self.project_operation_service.invalidate_async_jobs()
        except Exception:
            pass
        self._active_quality_worker = None
        self._quality_result_cache.clear()
        self._quality_reports.clear()
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
            self.viewport.clear()
        except Exception:
            pass
        if hasattr(self, 'list_facades'):
            self.list_facades.clear()
            self.lbl_facade_summary.setText('未检测')
            self._refresh_heatmap_button_state()
        if hasattr(self, 'station_list'):
            self.station_list.blockSignals(True)
            self.station_list.clear()
            self.station_list.blockSignals(False)

    def _set_current_project(self, project):
        self.current_project = project
        has_project = project is not None

        for page_key, button in self.page_buttons.items():
            button.setEnabled(page_key == 'project_overview' or has_project)

        if has_project:
            self.current_project_label.setText(f'当前项目：{project.name}')
            self.current_project_label.setToolTip(
                f'{project.name}\n{project.directory_path}'
            )
        else:
            self.current_project_label.setText('当前项目：未选择')
            self.current_project_label.setToolTip('')
            self.set_current_page(0)
        self._update_overview_workspace()
        self._refresh_report_preview()
        self._update_window_title()
