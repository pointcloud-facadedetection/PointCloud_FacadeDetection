from __future__ import annotations
from typing import Optional
import numpy as np
from PySide6.QtWidgets import QColorDialog, QMessageBox
from config.storage import Storage
import json


class ProjectOperationService:
    """将项目操作页的按钮事件转交给视口或后续算法实现。"""

    def __init__(self, viewport, facade_service=None, pointcloud_service=None):
        self._viewport = viewport
        self._facade_service = facade_service
        self._pointcloud_service = pointcloud_service
        self._last_roi_indices: Optional[list[int]] = None
        self.on_facade_results = None
        self._project_uuid: Optional[str] = None
        self._last_facade_results: Optional[list[dict]] = None

    def set_active_project_uuid(self, project_uuid: Optional[str]):
        self._project_uuid = project_uuid
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
                    if hasattr(self, '_render_global_color'):
                        # use service helper if available
                        try:
                            from services.viewport_render_service import ViewportRenderService  # noqa
                            if hasattr(self._pointcloud_service, 'render_service'):
                                self._pointcloud_service.render_service.set_global_point_color(tuple(col))
                        except Exception:
                            self._apply_global_color_direct(tuple(col))
                    else:
                        self._apply_global_color_direct(tuple(col))
        except Exception:
            pass

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
                self._viewport.update_cloud_color(name, colors)
        except Exception:
            pass

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

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
                from services.viewport_render_service import ViewportRenderService  # noqa
                if hasattr(self, '_pointcloud_service') and self._pointcloud_service is not None and hasattr(self._pointcloud_service, 'render_service'):
                    self._pointcloud_service.render_service.set_global_point_color(color)
                elif hasattr(self, '_facade_service') and self._facade_service is not None and hasattr(self._facade_service, '_render') and self._facade_service._render is not None:
                    self._facade_service._render.set_global_point_color(color)
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

    def denoise(self):
        self._notify('denoise')
        try:
            if self._pointcloud_service is None:
                print('PointCloudService 未注入，跳过去噪。', flush=True)
                return
            stats = self._pointcloud_service.denoise(method='radius')
            if stats is not None:
                print(f"去噪完成: {stats}", flush=True)
        except Exception as e:
            print(f"去噪失败: {e}", flush=True)

    def registration(self):
        self._notify('registration')

    # ------------------------------------------------------------------
    # ROI 视觉辅助：统一清除与渲染
    # ------------------------------------------------------------------
    def _clear_roi_visuals(self):
        """清除 2D 白框与 3D 包围盒，并重置 ROI 索引记录。"""
        try:
            if hasattr(self._viewport, 'clear_roi_visuals'):
                self._viewport.clear_roi_visuals()
            else:
                # 兼容降级：直接操作 adapter
                if hasattr(self._viewport, '_adapter'):
                    self._viewport._adapter.remove_geometry("__roi_selection_bbox")
                if hasattr(self._viewport, '_interactor'):
                    self._viewport._interactor.clear_selection_rect()
        except Exception as e:
            print(f"清除 ROI 视觉失败: {e}", flush=True)
        self._last_roi_indices = None

    def _render_roi_bbox(self, cloud_name: str, indices: list[int]):
        """根据 ROI 索引计算 AABB 并在视口渲染 3D 白框。"""
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
            else:
                # 兼容降级
                from view3d.geometry_factory import make_bbox
                self._viewport._adapter.remove_geometry("__roi_selection_bbox")
                self._viewport._adapter.add_geometry("__roi_selection_bbox", make_bbox(min_b, max_b, color=(1.0, 1.0, 1.0)), reset_bounding_box=False)
                if hasattr(self._viewport, '_overlay'):
                    self._viewport._overlay.update()
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
            if self._facade_service is not None and getattr(self._facade_service, '_render', None) is not None:
                self._facade_service._render.clear_selected_facade(cloud)
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

    def _on_roi_selected(self, min_bound, max_bound, indices):
        """
        ROI 框选完成后的回调：根据框选索引估计整栋建筑 BBOX，存档并可视化。

        修复要点：
        1. 如果 indices 为空，先尝试用 min_bound/max_bound 计算
        2. 添加更详细的调试日志
        3. 如果仍然为空，给出更具体的错误提示
        """
        try:
            names = self._viewport.get_cloud_names()
            cloud = names[-1] if names else None

            # 尝试从 indices 获取，如果为空则尝试从 BBox 计算
            if not isinstance(indices, (list, tuple, np.ndarray)) or len(indices) == 0:
                if min_bound is not None and max_bound is not None:
                    # 从 BBox 计算索引
                    data = self._viewport.get_cloud_data(cloud)
                    if data is not None and data.get('pos') is not None:
                        pos = np.asarray(data['pos'], dtype=float)
                        bmin = np.asarray(min_bound, dtype=float).reshape(1, 3)
                        bmax = np.asarray(max_bound, dtype=float).reshape(1, 3)
                        mask = np.all((pos >= bmin) & (pos <= bmax), axis=1)
                        indices = np.where(mask)[0].astype(int).tolist()

            if not cloud or not isinstance(indices, (list, tuple, np.ndarray)) or len(indices) == 0:
                try:
                    QMessageBox.information(
                        None, 
                        '框选检测区域', 
                        '未选中有效区域。提示：请确保在视口中拖拽框选建筑立面区域!'
                    )
                except Exception:
                    print('未选中有效区域，请重试。', flush=True)
                return

            # 找到渲染服务实例
            render = self._get_render_service()
            if render is None:
                # 渲染服务不可用时，仅记录原始索引
                self.set_detection_roi(None, None, indices)
                return

            # 1) 尝试根据框选索引拟合立面平面
            plane, sel_pts = render.fit_plane_on_selection(cloud, indices)

            # 距离容差（米）：使用 DETECT_DIST_TOL_MM（毫米）转换为米，避免将倍率 2.5 误作米值
            try:
                tol = float(getattr(__import__('config.settings', fromlist=['Config']).Config, 'DETECT_DIST_TOL_MM', 20.0)) / 1000.0
            except Exception:
                tol = 0.02

            # 2) 在平面约束下计算整栋建筑 BBox
            if plane is not None:
                bmin, bmax = render.compute_building_bbox_from_selection(cloud, indices, plane=plane, tol=None)
            else:
                bmin, bmax = render.compute_building_bbox_from_selection(cloud, indices)

            if bmin is None or bmax is None:
                # 主墙面识别失败时不保存被遮挡物控制的伪 BBox。
                print('未能识别稳定的主墙面，请扩大框选区域后重试。', flush=True)
                return

            # 可视化 ROI BBOX（红色）
            try:
                render.visualize_building_bbox(bmin, bmax, color=(1.0, 0.2, 0.2))
            except Exception:
                pass

            # 以 BBOX 计算并存储 ROI 索引
            self.set_detection_roi(bmin, bmax, None)
            print(
                f"ROI 已设定：建筑 BBox [{bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f}] ~ "
                f"[{bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f}]",
                flush=True,
            )
            self._last_roi_bounds = (bmin, bmax)
        except Exception as exc:
            print(f"ROI 计算失败: {exc}", flush=True)

    def _get_render_service(self):
        """获取渲染服务实例的统一入口。"""
        if self._facade_service is not None and getattr(self._facade_service, '_render', None) is not None:
            return self._facade_service._render
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
                return
            cloud = names[-1]
            data = self._viewport.get_cloud_data(cloud)
            if data is None or (data.get('pos') is None) or len(data['pos']) == 0:
                self._last_roi_indices = None
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
        except Exception:
            self._last_roi_indices = None


    def facade_detection(self):
        self._notify('facade_detection')
        # 检测前清除旧的 ROI 视觉（仅清除 2D/3D 视觉，不重置相机）
        self._clear_roi_visuals()

        if self._facade_service is None:
            print('FacadeService未注入，跳过检测。', flush=True)
            return
        names = []
        try:
            names = self._viewport.get_cloud_names()
        except Exception:
            pass
        if not names:
            print('无活动点云。', flush=True)
            return
        cloud_name = names[-1]
        roi = self._last_roi_indices if self._last_roi_indices else None
        roi_bounds = getattr(self, '_last_roi_bounds', None)
        try:
            results = self._facade_service.detect_on_roi(
                cloud_name=cloud_name,
                roi_indices=roi,
                roi_bounds=roi_bounds,
                project_uuid=self._project_uuid,
            )
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
            cloud_name = names[-1]
            if not self._last_facade_results:
                print('尚无检测结果用于热力着色', flush=True)
                return
            self._facade_service.render_flatness_heatmap(cloud_name, self._last_facade_results)
        except Exception as e:
            print(f'质量检测渲染失败: {e}', flush=True)

    def box_segmentation(self):
        self._notify('box_segmentation')

    def calculate_detail(self):
        self._notify('calculate_detail')

    def align_2d_3d(self):
        self._notify('align_2d_3d')

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
            names = self._viewport.get_cloud_names()
            return names[-1] if names else None
        except Exception:
            return None