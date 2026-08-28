from __future__ import annotations
import time
from typing import Optional
import numpy as np
from PySide6.QtWidgets import QColorDialog, QMessageBox
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from config.storage import Storage
import json


class _GuiDispatcher(QObject):
    """Queued signal boundary owned by the Qt GUI thread."""
    denoise_finished = Signal(object)
    denoise_failed = Signal(str)


class ProjectOperationService:
    """将项目操作页的按钮事件转交给视口或后续算法实现。"""

    def __init__(self, viewport, facade_service=None, pointcloud_service=None, render_service=None):
        self._viewport = viewport
        self._facade_service = facade_service
        self._pointcloud_service = pointcloud_service
        self._render_service = render_service
        self._last_roi_indices: Optional[list[int]] = None
        self._last_roi_bounds = None
        self.on_facade_results = None
        self._project_uuid: Optional[str] = None
        self._last_facade_results: Optional[list[dict]] = None
        self._station_service = None
        # ProjectOperationService is intentionally not a QObject.  Connecting
        # a worker signal directly to its Python method can therefore invoke
        # the slot in the worker thread.  Use a real GUI QObject as the queued
        # dispatch boundary for every Open3D-facing completion.
        parent = viewport.get_widget() if hasattr(viewport, 'get_widget') else None
        self._gui_dispatcher = _GuiDispatcher(parent)
        self._gui_dispatcher.denoise_finished.connect(
            self._on_denoise_finished, Qt.ConnectionType.QueuedConnection)
        self._gui_dispatcher.denoise_failed.connect(
            self._on_denoise_failed, Qt.ConnectionType.QueuedConnection)

    def set_station_service(self, service):
        self._station_service = service

    def set_active_project_uuid(self, project_uuid: Optional[str]):
        self._project_uuid = project_uuid
        # A project switch invalidates all proxy-indexed process results.
        self.clear_processing_state()
        try:
            if not project_uuid:
                return
            from pathlib import Path
            prefs_path = Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME / "ui_prefs.json"
            if prefs_path.exists():
                with open(prefs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                col = data.get('scene_color')
                if col and hasattr(self._viewport, 'update_cloud_color') and hasattr(self._viewport, 'get_cloud_names'):
                    render = self._get_render_service()
                    if render is not None:
                        render.set_global_point_color(tuple(col))
                    else:
                        self._apply_global_color_direct(tuple(col))
        except Exception:
            pass

    def invalidate_async_jobs(self):
        """Invalidate completion callbacks without blocking the GUI thread."""
        self._quality_request_token = getattr(self, '_quality_request_token', 0) + 1
        thread = getattr(self, '_denoise_thread', None)
        if thread is not None and thread.isRunning():
            thread.requestInterruption()

    def clear_processing_state(self):
        """Drop transient facade/ROI state; never remove raw/source data."""
        self._last_facade_results = None
        self._last_roi_indices = None
        self._last_roi_bounds = None
        self._quality_result_cache = getattr(self, '_quality_result_cache', {})
        self._quality_result_cache.clear()
        render = self._get_render_service()
        if render is not None:
            try:
                render.clear_selected_facade()
            except Exception:
                pass
        self._clear_roi_visuals()

    def _apply_global_color_direct(self, color):
        try:
            names = self._viewport.get_cloud_names()
            if not names:
                return
            for name in names:
                data = self._viewport.get_cloud_data(name)
                if data is None:
                    continue
                pos = data.get('pos')
                if pos is None or len(pos) == 0:
                    continue
                n = len(pos)
                colors = np.tile(np.asarray(color, dtype=float).reshape(1, 3), (n, 1))
                # Patch: 优先使用队列更新方法，避免主线程阻塞
                queue = getattr(self._viewport, 'queue_update_cloud_color', None)
                if callable(queue):
                    queue(name, colors)
                else:
                    self._viewport.update_cloud_color(name, colors)
        except Exception:
            pass

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

    def reset_view(self):
        """保留 UI 的视图重置入口，并委托给最新视口实现。"""
        self._notify('reset_view')
        reset = getattr(self._viewport, 'reset_view', None)
        if callable(reset):
            reset()

    def change_color(self):
        self._notify('change_color')
        # 弹出颜色选择器，选择后全局应用并持久化到项目缓存
        try:
            dlg = QColorDialog()
            dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
            if not dlg.exec():
                return
            qcol = dlg.selectedColor()
            if not qcol.isValid():
                return
            color = (qcol.redF(), qcol.greenF(), qcol.blueF())
            # 首选通过渲染服务应用（若可用）
            try:
                render = self._get_render_service()
                if render is not None:
                    render.set_global_point_color(color)
                else:
                    self._apply_global_color_direct(color)
            except Exception:
                self._apply_global_color_direct(color)

            # 持久化到项目缓存
            try:
                if self._project_uuid:
                    dirs = Storage.ensure_project_dirs(self._project_uuid)
                    prefs_path = dirs['cache'] / 'ui_prefs.json'
                    data = {}
                    if prefs_path.exists():
                        try:
                            with open(prefs_path, 'r', encoding='utf-8') as f:
                                data = json.load(f) or {}
                        except Exception:
                            data = {}
                    data['scene_color'] = list(map(float, color))
                    with open(prefs_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception as e:
            print(f"颜色更改失败: {e}", flush=True)

    def denoise(self, **denoise_options):
        self._notify('denoise')
        try:
            if self._pointcloud_service is None:
                print('PointCloudService 未注入，跳过去噪。', flush=True)
                return
            if getattr(self, '_denoise_thread', None) is not None:
                return
            service = self._pointcloud_service
            class Worker(QObject):
                finished = Signal(object)
                failed = Signal(str)
                @Slot()
                def run(self):
                    try:
                        print('[PCFD] denoise.worker_start', flush=True)
                        self.finished.emit(
                            service.denoise(
                                method='adaptive',
                                update_viewport=False,
                                **denoise_options,
                            )
                        )
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        self.failed.emit(str(exc))
            self._denoise_thread = QThread()
            self._denoise_worker = Worker()
            self._denoise_worker.moveToThread(self._denoise_thread)
            self._denoise_thread.started.connect(self._denoise_worker.run)
            self._denoise_worker.finished.connect(self._gui_dispatcher.denoise_finished)
            self._denoise_worker.failed.connect(self._gui_dispatcher.denoise_failed)
            self._denoise_worker.finished.connect(self._denoise_thread.quit)
            self._denoise_worker.failed.connect(self._denoise_thread.quit)
            self._denoise_thread.finished.connect(self._clear_denoise_worker)
            self._denoise_thread.start()
        except Exception as e:
            print(f"去噪失败: {e}", flush=True)

    @Slot(object)
    def _on_denoise_finished(self, stats):
        try:
            if stats:
                # A station review layer is never authoritative for processing.
                # Remove it before publishing the new proxy snapshot so the
                # viewport cannot display stale source geometry over the proxy.
                render = self._get_render_service()
                if render is not None and hasattr(render, 'clear_station_scene'):
                    render.clear_station_scene()
                points = np.empty((0, 3), dtype=np.float32)
                data = self._viewport.get_cloud_data(stats['name'])
                if data is not None:
                    points = np.asarray(stats.get('proxy_points'), dtype=np.float32).reshape(-1, 3)
                    colors = stats.get('proxy_colors')
                    if colors is not None:
                        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
                        if len(colors) != len(points):
                            colors = None
                    # This slot runs in the GUI thread.  Keep the viewport
                    # metadata and geometry update atomic from the UI's point
                    # of view; the worker must never touch Open3D.
                    metadata = {'proxy_ids': np.arange(len(points), dtype=np.int32),
                                'domain': 'proxy', 'index_space': 'proxy_global',
                                'is_processing_cloud': True}
                    if hasattr(self._viewport, "replace_cloud_snapshot"):
                        self._viewport.replace_cloud_snapshot(stats['name'], points, colors, metadata)
                    elif hasattr(self._viewport, "queue_update_cloud_points"):
                        data.update(metadata)
                        self._viewport.queue_update_cloud_points(stats['name'], points, colors)
                    elif hasattr(self._viewport, "update_cloud_points"):
                        data.update(metadata)
                        self._viewport.update_cloud_points(stats['name'], points, colors)
                    # A replacement invalidates proxy-indexed facade working
                    # data.  The next detection must recolor from the new
                    # proxy rows and new VoxelCascadeIndex revision.
                    self._last_facade_results = None
                    if render is not None:
                        render._facades_cache.pop(stats['name'], None)
                        render.clear_selected_facade(stats['name'])
                    if len(points) == 0:
                        print(f"[PCFD] denoise.viewport_cleared cloud={stats['name']}", flush=True)
                # Keep the result payload intact for diagnostics and for any
                # downstream snapshot consumer; never mutate worker output.
                print(f"去噪完成: cloud={stats.get('name')} "
                      f"proxy={len(points)} raw={stats.get('raw_count', 0)}", flush=True)
                callback = getattr(self, 'denoise_completed_callback', None)
                if callable(callback):
                    callback(stats)
            else:
                print('去噪未产生结果。', flush=True)
        finally:
            self._finish_denoise_state()

    @Slot(str)
    def _on_denoise_failed(self, message):
        print(f"去噪失败: {message}", flush=True)
        callback = getattr(self, 'denoise_completed_callback', None)
        if callable(callback):
            callback(None)
        self._finish_denoise_state()

    @Slot()
    def _clear_denoise_worker(self):
        worker = self._denoise_worker
        thread = self._denoise_thread
        self._denoise_worker = None
        self._denoise_thread = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    def _finish_denoise_state(self):
        """Allow a new job even if a queued cleanup callback was delayed."""
        # Do not delete QObjects here; QThread.finished owns that lifecycle.
        if self._denoise_thread is not None and not self._denoise_thread.isRunning():
            self._clear_denoise_worker()

    def registration(self):
        self._notify('registration')
        if self._station_service is not None:
            return self._station_service.register_selected()

    # ------------------------------------------------------------------
    # ROI 视觉辅助：统一清除与渲染
    # ------------------------------------------------------------------
    def _clear_roi_visuals(self):
        """清除 ROI 视觉辅助。"""
        try:
            if hasattr(self._viewport, 'clear_roi_visuals'):
                self._viewport.clear_roi_visuals()
        except Exception as e:
            print(f"清除 ROI 视觉失败: {e}", flush=True)

    def _render_roi_bbox(self, cloud_name: str, indices: list[int]):
        """根据 ROI 索引计算 AABB 并在视口渲染3D框。"""
        try:
            data = self._viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(pos))]
            if len(idx) == 0:
                return
            roi_pts = pos[idx]
            min_b = np.min(roi_pts, axis=0)
            max_b = np.max(roi_pts, axis=0)
            # 更小的外扩，避免框体视觉偏大：1% + 0.005m
            margin = np.max(max_b - min_b) * 0.01 + 0.005
            min_b -= margin
            max_b += margin
            if hasattr(self._viewport, 'show_roi_bbox'):
                self._viewport.show_roi_bbox(min_b, max_b, color=(1.0, 1.0, 1.0))
        except Exception as e:
            print(f"渲染 ROI 包围盒失败: {e}", flush=True)

    # ------------------------------------------------------------------
    # 框选检测区域
    # ------------------------------------------------------------------
    def select_detection_area(self):
        """激活建筑外立面 ROI 框选模式（拦截相机交互，Qt 覆盖层实时绘制矩形）。"""
        cloud = self._active_cloud_name()
        if not cloud:
            try:
                QMessageBox.information(None, '框选检测区域', '请先加载点云数据。')
            except Exception:
                print('请先加载点云数据。', flush=True)
            return
        # 清除之前的立面高亮和 ROI 显示
        try:
            if hasattr(self._viewport, 'clear_roi_visuals'):
                self._viewport.clear_roi_visuals()
        except Exception:
            pass
        try:
            # 通过渲染服务清除立面选中高亮（若存在）
            render = self._get_render_service()
            if render is not None:
                render.clear_selected_facade(cloud)
        except Exception:
            pass
        # 进入 ROI 框选模式
        try:
            self._viewport.enter_roi_selection(
                cloud_name=cloud,
                on_complete=self._on_roi_selected
            )
        except Exception as exc:
            print(f"进入 ROI 框选模式失败: {exc}", flush=True)

    def _on_roi_selected(self, min_bound, max_bound, indices, p1=None, p2=None):
        """ROI 框选完成回调"""
        t0 = time.monotonic()
        try:
            names = self._viewport.get_cloud_names()
            cloud = names[-1] if names else None
            
            if indices is None:
                n_indices = 0
                indices_list = []
            elif isinstance(indices, np.ndarray):
                n_indices = len(indices)
                indices_list = indices.tolist() if indices.ndim == 1 else []
            else:
                n_indices = len(indices)
                indices_list = list(indices)
            
            print(
                f"[ROI] 开始: cloud={cloud}, 索引={n_indices}",
                flush=True,
            )

            if n_indices == 0:
                if min_bound is not None and max_bound is not None:
                    data = self._viewport.get_cloud_data(cloud)
                    if data is not None and data.get('pos') is not None:
                        pos = np.asarray(data['pos'], dtype=float)
                        bmin = np.asarray(min_bound, dtype=float).reshape(1, 3)
                        bmax = np.asarray(max_bound, dtype=float).reshape(1, 3)
                        mask = np.all((pos >= bmin) & (pos <= bmax), axis=1)
                        indices_list = np.where(mask)[0].astype(int).tolist()
                        n_indices = len(indices_list)

            if not cloud or n_indices == 0:
                try:
                    QMessageBox.information(
                        None,
                        '框选检测区域',
                        '未选中有效区域。提示：请确保在视口中拖拽框选建筑立面区域!',
                    )
                except Exception:
                    print('未选中有效区域，请重试。', flush=True)
                return

            render = self._get_render_service()
            if render is None:
                print(
                    "[ROI] 警告: RenderService不可用，仅保存索引",
                    flush=True,
                )
                self.set_detection_roi(None, None, indices_list)
                return

            screen_rect = (p1, p2) if p1 is not None and p2 is not None else None

            bmin, bmax = render.compute_building_bbox_from_selection(
                cloud,
                indices_list, 
                screen_rect=screen_rect,
                plane=None,
                tol=None,
            )

            if bmin is None or bmax is None:
                print('[ROI] BBox生成失败，使用AABB兜底', flush=True)
                try:
                    data = self._viewport.get_cloud_data(cloud)
                    pos = np.asarray(data['pos'], dtype=float)
                    idx_arr = np.asarray(indices_list, dtype=int)
                    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(pos))]
                    sel_pts = pos[idx_arr]
                    bmin = np.min(sel_pts, axis=0)
                    bmax = np.max(sel_pts, axis=0)
                    margin = np.max(bmax - bmin) * 0.05 + 0.1
                    bmin -= margin
                    bmax += margin
                except Exception as e:
                    print(f"[ROI] AABB失败: {e}", flush=True)
                    return

            try:
                render.visualize_building_bbox(bmin, bmax, color=(1.0, 0.2, 0.2))
            except Exception as e:
                print(f"[ROI] 可视化失败: {e}", flush=True)

            self.set_detection_roi(bmin, bmax, None)
            elapsed = time.monotonic() - t0
            print(
                f"[ROI] 完成 (耗时{elapsed:.3f}s): "
                f"[{bmin[0]:.2f},{bmin[1]:.2f},{bmin[2]:.2f}] ~ "
                f"[{bmax[0]:.2f},{bmax[1]:.2f},{bmax[2]:.2f}]",
                flush=True,
            )
            self._last_roi_bounds = (bmin, bmax)

        except Exception as exc:
            print(f"[ROI] 异常: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    def _get_render_service(self):
        """获取渲染服务实例的统一入口。"""
        if self._render_service is not None:
            return self._render_service
        if hasattr(self._pointcloud_service, 'render_service'):
            return self._pointcloud_service.render_service
        return None

    def set_detection_roi(self, min_bound, max_bound, indices=None):
        """设置当前检测 ROI。
        - min_bound, max_bound: 3D AABB（世界坐标）
        - indices: 可选，已选点的全局索引；若未提供，则根据 AABB 计算。
        结果存入 self._last_roi_indices，供 detect_on_roi 使用。
        """
        try:
            import numpy as np
            names = self._viewport.get_cloud_names()
            if not names:
                self._last_roi_indices = None
                self._last_roi_bounds = None
                return
            cloud = names[-1]
            data = self._viewport.get_cloud_data(cloud)
            if data is None or (data.get('pos') is None) or len(data['pos']) == 0:
                self._last_roi_indices = None
                self._last_roi_bounds = None
                return
            pos = np.asarray(data['pos'], dtype=float)
            if indices is None or len(indices) == 0:
                if min_bound is None or max_bound is None:
                    idx = np.array([], dtype=int)
                else:
                    bmin = np.asarray(min_bound, dtype=float).reshape(1, 3)
                    bmax = np.asarray(max_bound, dtype=float).reshape(1, 3)
                    mask = np.all((pos >= bmin) & (pos <= bmax), axis=1)
                    idx = np.where(mask)[0].astype(int)
            else:
                idx = np.asarray(indices, dtype=int)
                idx = idx[(idx >= 0) & (idx < len(pos))]
            # 去重并存档
            if len(idx) == 0:
                self._last_roi_indices = None
            else:
                self._last_roi_indices = np.unique(idx).tolist()
            # 保留 BBox 作为检测输入。索引是渲染点云空间的索引，可能与
            # detection dataset 的 proxy 索引不同，因此 BBox 是更稳定的兼容路径。
            if min_bound is not None and max_bound is not None:
                bmin = np.asarray(min_bound, dtype=float).reshape(3)
                bmax = np.asarray(max_bound, dtype=float).reshape(3)
                self._last_roi_bounds = (np.minimum(bmin, bmax), np.maximum(bmin, bmax))
            else:
                self._last_roi_bounds = None
        except Exception:
            self._last_roi_indices = None
            self._last_roi_bounds = None


    def facade_detection(self):
        self._notify('facade_detection')
        # 检测前清除旧的 ROI 视觉（仅清除 2D/3D 视觉，不重置当前检测条件）
        self._clear_roi_visuals()

        if self._facade_service is None:
            print('FacadeService未注入，跳过检测。', flush=True)
            return
        cloud_name = self._pointcloud_service.resolve_processing_cloud() \
            if self._pointcloud_service is not None else None
        if not cloud_name:
            print('无已注册的代理处理点云。', flush=True)
            return
        try:
            roi_indices = self._last_roi_indices
            roi_bounds = self._last_roi_bounds
            if roi_indices or roi_bounds is not None:
                # 框选检测使用 seed 模式：全场景识别，按 ROI 选择相交立面，
                # 避免把建筑墙体截断；没有 ROI 时保持全局检测行为。
                results = self._facade_service.detect_on_roi(
                    cloud_name=cloud_name,
                    roi_indices=roi_indices,
                    roi_bounds=roi_bounds,
                    project_uuid=self._project_uuid,
                    roi_scope='seed')
            else:
                results = self._facade_service.detect(
                    cloud_name=cloud_name, project_uuid=self._project_uuid)
            self._last_facade_results = results
            if callable(self.on_facade_results):
                try:
                    self.on_facade_results(results)
                except Exception:
                    pass
            # 启用点击选择：在视口中点击立面点，高亮对应立面
            try:
                self._enable_facade_click_select(cloud_name, results or [])
            except Exception:
                pass
            # The UI result callback performs the single authoritative color
            # commit.  Do not submit a second 1.9M-row color buffer here.
        except Exception as e:
            print(f'立面检测失败: {e}', flush=True)

    def quality_inspection(self):
        self._notify('quality_inspection')
        try:
            if self._facade_service is None:
                return
            names = []
            try:
                names = self._viewport.get_cloud_names()
            except Exception:
                pass
            if not names:
                return
            cloud_name = self._pointcloud_service.resolve_processing_cloud() \
                if self._pointcloud_service is not None else None
            if not cloud_name:
                print('无已注册的代理处理点云。', flush=True)
                return
            if not self._last_facade_results:
                print('尚无检测结果用于热力着色', flush=True)
                return
            render = self._get_render_service()
            if render is not None:
                render.render_flatness_heatmap(
                    cloud_name, self._last_facade_results,
                    index_service=getattr(self._facade_service, '_index_service', None))
        except Exception as e:
            print(f'质量检测渲染失败: {e}', flush=True)

    def persist_facade_review_status(self, facade):
        """Persist operator review when a project result repository is available."""
        if not self._project_uuid or not facade:
            return
        # Detection IDs are transient ordinals.  Only a formally persisted
        # facade may be addressed by the database primary key.
        facade_id = facade.get('facade_db_id') or facade.get('database_id')
        if not facade_id:
            return
        try:
            from services.dal.results_repo import ResultsRepo
            ResultsRepo.update_facade_review_status(
                self._project_uuid, int(facade_id),
                facade.get('review_status', facade.get('preview_status', 'pending')))
        except (AttributeError, NotImplementedError):
            # Older databases do not yet expose this optional column; the
            # in-memory result remains authoritative for the current session.
            pass
        except Exception as exc:
            print(f'[PCFD] facade.review.persist_failed reason={exc}', flush=True)

    def box_segmentation(self):
        self._notify('box_segmentation')

    def calculate_detail(self):
        self._notify('calculate_detail')

    def align_2d_3d(self):
        """进入/退出三维点云手动标点模式。"""
        self._notify('align_2d_3d')
        cloud = self._active_cloud_name()
        if not cloud:
            try:
                QMessageBox.information(None, '二维-三维对齐', '请先加载点云数据。')
            except Exception:
                print('请先加载点云数据。', flush=True)
            return

        render = self._get_render_service()
        viewport = self._viewport
        if render is None and not hasattr(viewport, 'enter_pick_mode'):
            print('[Pick] 当前视口不支持点选', flush=True)
            return

        if render is not None and render.is_pick_mode():
            render.exit_pick_mode()
            count = render.picked_count()
            try:
                QMessageBox.information(
                    None,
                    '二维-三维对齐',
                    f'已退出标点模式，当前共 {count} 个三维标注点。',
                )
            except Exception:
                print(f'[Pick] 退出标点模式，点数={count}', flush=True)
            return

        if render is not None:
            render.clear_pick_markers()

        def _on_pick(picked):
            if not picked:
                return
            point = picked.get('point')
            if point is None:
                return
            xyz = np.asarray(point, dtype=float).reshape(3)
            if render is not None:
                render.add_pick_marker(xyz)
                count = render.picked_count()
            else:
                count = 0
            print(
                f'[Pick] 标注点 #{count}: '
                f'X={xyz[0]:.6f}, Y={xyz[1]:.6f}, Z={xyz[2]:.6f} '
                f'(点云索引={picked.get("index")})',
                flush=True,
            )

        pick_radius = 32
        try:
            if render is not None:
                render.enter_pick_mode(
                    callback=_on_pick,
                    cloud_name=cloud,
                    pick_radius=pick_radius,
                )
            elif hasattr(viewport, 'enter_pick_mode'):
                viewport.enter_pick_mode(
                    cloud_name=cloud,
                    pick_radius=pick_radius,
                    callback=_on_pick,
                )
        except Exception as exc:
            print(f'[Pick] 进入标点模式失败: {exc}', flush=True)
            return
        print('[Pick] 进入三维标点模式，左键点击加点', flush=True)

    # ---------------- Internal: click-to-select facade ----------------
    def _enable_facade_click_select(self, cloud_name: str, facades: list[dict]):
        if not facades:
            return
        # 构建索引到立面ID映射（仅包含内点）
        try:
            idx2fid = {}
            for f in facades:
                fid = int(f.get('id', -1))
                if fid < 0:
                    continue
                idx = np.asarray(f.get('inlier_indices') or [], dtype=int)
                for i in idx.tolist():
                    if i not in idx2fid:
                        idx2fid[i] = fid
        except Exception:
            idx2fid = None

        def _on_pick(picked):
            try:
                if not picked:
                    return
                i = int(picked.get('index'))
                fid = None
                if idx2fid is not None:
                    fid = idx2fid.get(i)
                if fid is None:
                    # 退化：线性扫描
                    for f in facades:
                        if i in (f.get('inlier_indices') or []):
                            fid = int(f.get('id', -1))
                            break
                if fid is None or fid < 0:
                    return
                # 通过渲染服务触发高亮选择
                render = self._get_render_service()
                if render is not None and hasattr(render, 'select_facade'):
                    render.select_facade(cloud_name, fid)
            except Exception:
                pass

        # 进入拾取模式（不影响旋转：拖动仍可旋转，点按会拾取）
        try:
            if hasattr(self._viewport, 'enter_pick_mode'):
                self._viewport.enter_pick_mode(cloud_name=cloud_name, pick_radius=10, callback=_on_pick)
        except Exception:
            pass

    # ---------------- Helpers ----------------
    def _active_cloud_name(self) -> str | None:
        try:
            if self._pointcloud_service is not None:
                resolved = self._pointcloud_service.resolve_processing_cloud()
                if resolved:
                    return resolved
            names = self._viewport.get_cloud_names()
            return names[-1] if names else None
        except Exception:
            return None
