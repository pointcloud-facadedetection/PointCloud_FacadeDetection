from __future__ import annotations

from typing import Optional, Tuple
from algorithms.facade.facade_detection import detect_facades
from config.settings import Config
import numpy as np
from services.dal.results_repo import ResultsRepo

class FacadeService:
    """
    外立面检测和持久化服务。
    所有DAL交互均使用project_uuid，以确保各层之间地址的一致性。
    """

    def __init__(self, viewport, db=None, render_service=None):
        self._viewport = viewport
        self._db = db
        self._render = render_service

    def detect(self, cloud_name: str, project_uuid: Optional[str] = None) -> list[dict]:
        """运行全场景立面检测，并可选地将结果保存到指定的项目 UUID 中。
        """
        pcd = self._viewport.get_cloud_data(cloud_name)
        if pcd is None:
            return []
        import open3d as o3d
        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(pcd["pos"].astype(float))
        if (pcd.get("color") is not None) and len(pcd["color"]) == len(pcd["pos"]):
            geo.colors = o3d.utility.Vector3dVector(pcd["color"].astype(float))

        try:
            vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))
        except Exception:
            vsize = 0.05
        try:
            min_area = float(getattr(Config, 'MIN_FACADE_AREA', 10.0))
        except Exception:
            min_area = 10.0
        result = detect_facades(
            geo,
            voxel_size=vsize,
            min_facade_area=float(getattr(Config, 'MIN_FACADE_AREA', min_area)),
        )
        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])

        if project_uuid:
            self._persist_results(project_uuid, facades)

        try:
            if self._render is not None:
                self._render.highlight_facades(cloud_name, facades)
        except Exception:
            pass

        return facades

    def detect_on_roi(self, cloud_name: str, roi_indices: list[int] | None = None, 
                      roi_bounds: tuple | None = None,  # 新增参数
                      project_uuid: Optional[str] = None,
                      min_facade_area: float | None = None) -> list[dict]:
        """对 ROI 子集运行检测。当提供 project_uuid 时，保存结果."""
        pcd = self._viewport.get_cloud_data(cloud_name)
        if pcd is None:
            return []
        import open3d as o3d
        pos = pcd["pos"]
        col = pcd.get("color")
        if min_facade_area is None:
            min_facade_area = 5.0
        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(pos.astype(float))
        if col is not None:
            geo.colors = o3d.utility.Vector3dVector(col.astype(float))

        try:
            vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))
        except Exception:
            vsize = 0.05
        
        # 同时传入 roi_bounds 和 roi_indices，并允许在bounds内生长
        roi_bounds_param = None
        if roi_bounds is not None:
            roi_bounds_param = roi_bounds
        elif roi_indices is not None and len(roi_indices) > 0:
            # 若未传入bounds但传了indices，从indices计算bounds作为安全边界
            pts = pos[np.asarray(roi_indices, dtype=int)]
            bmin = np.min(pts, axis=0)
            bmax = np.max(pts, axis=0)
            # 小扩展确保边界完整
            margin = 0.5
            roi_bounds_param = (bmin - margin, bmax + margin)

        result = detect_facades(
            geo,
            voxel_size=vsize,
            min_facade_area=float(getattr(Config, 'MIN_FACADE_AREA', min_facade_area or 10.0)),
            roi_indices=roi_indices if roi_indices else None,
            roi_bounds=roi_bounds_param,  # 新增
            enable_grow=True,  # 【修复】改为True，让算法在bounds内补全立面
        )
        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])

        if project_uuid:
            self._persist_results(project_uuid, facades)

        try:
            if self._render is not None:
                self._render.highlight_facades(cloud_name, facades)
        except Exception:
            pass

        return facades

    @staticmethod
    def project_results_dir(project_uuid: str):
        """统一项目结果目录，避免调用方各自拼接 Windows 路径。"""
        from pathlib import Path
        return Path(__file__).resolve().parents[2] / 'data' / 'projects' / str(project_uuid) / 'results'

    # ---------------- 渲染质量热力贴图 ----------------
    def render_flatness_heatmap(self, cloud_name: str, facades: list[dict], vmin: float | None = None,
                                vmax: float | None = None) -> None:
        """计算每个立面内各点的绝对平面距离，并以热力图形式呈现。"""
        try:
            if self._render is None:
                return
            data = self._viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            import numpy as np
            n_total = len(data.get('pos') or [])
            if n_total == 0:
                return
            all_indices = []
            all_values = []
            for f in facades or []:
                idx = np.asarray(f.get('measurement_indices', f.get('inlier_indices')) or [], dtype=int)
                idx = idx[(idx >= 0) & (idx < n_total)]
                if len(idx) == 0:
                    continue
                model = np.asarray(f.get('plane_model') or [], dtype=float)
                if model.shape[0] != 4:
                    continue
                pts = np.asarray(data['pos'])[idx]

                n = model[:3]
                n = n / (np.linalg.norm(n) + 1e-12)
                d = float(model[3])
                dist = np.abs(pts @ n + d)
                all_indices.append(idx)
                all_values.append(dist.astype(float))
            if not all_indices:
                return
            idx_cat = np.concatenate(all_indices, axis=0)
            val_cat = np.concatenate(all_values, axis=0)
            self._render.colorize_by_scalar(cloud_name, idx_cat, val_cat, vmin=vmin, vmax=vmax, cmap='turbo')
        except Exception as e:
            print(f'FacadeService: 渲染质量热力贴图失败 : {e}', flush=True)

    @staticmethod
    def compute_verticality_deg(facade: dict) -> float:
        try:
            import numpy as np
            n = np.asarray(facade.get('normal') or [], dtype=float)
            if n.shape[0] != 3:
                return 0.0
            n = n / (np.linalg.norm(n) + 1e-12)
            nz = abs(float(n[2]))
            return float(np.degrees(np.arcsin(np.clip(nz, 0.0, 1.0))))
        except Exception:
            return 0.0

    # ---------------- 质量评估计算 ----------------
    def compute_quality(self, cloud_name: str, facade: dict,
                        grid_size: float | None = None,
                        flatness_limit: float = 0.004,
                        verticality_limit_mm: float = 4.0,
                        ruler_size: float | None = None,
                        ruler_step: float | None = None,
                        max_windows: int | None = None,
                        profile=None, results_dir=None) -> Optional[dict]:
        """使用 algorithms.facade.quality 计算单个立面的平整度/垂直度质量指标，并返回结果字典。"""
        try:
            if profile is not None:
                flatness_limit = profile.flatness_limit_mm / 1000.0
                verticality_limit_mm = profile.verticality_limit_mm
                ruler_size = profile.window_size_m
                ruler_step = profile.step_size_m
                max_windows = None
            from algorithms.facade.quality import compute_facade_quality
            from algorithms.geometry import (
                split_by_depth_histogram, fit_plane_irls,
                plane_basis_from_normal, project_to_plane, estimate_uv_bbox_area
            )
            import numpy as np
            import open3d as o3d
            data = self._viewport.get_cloud_data(cloud_name)
            if data is None:
                return None
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return None
            idx = np.asarray(facade.get('measurement_indices', facade.get('inlier_indices')) or [], dtype=int)
            idx = idx[(idx >= 0) & (idx < len(pos))]
            if len(idx) == 0:
                return None
            pts = np.asarray(pos)[idx]

            # 参考平面：优先使用 facade 提供的 plane_model；若无则后续算法处理
            model = np.asarray(facade.get('plane_model') or [], dtype=float)
            if model.shape[0] != 4:
                model = None

            # 1) 剥离凸出层：沿法向的有符号距离直方图分割，只保留主墙片层
            try:
                tol_m = float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0)) / 1000.0
                if model is not None:
                    masks = split_by_depth_histogram(pts, model, tol_m, min_points=80)
                    if masks and len(masks) > 1:
                        best_i = 0
                        best_area = -1.0
                        for i, mk in enumerate(masks):
                            sub = pts[mk]
                            try:
                                nrm = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
                                cen = np.mean(sub, axis=0)
                                u, v = plane_basis_from_normal(nrm)
                                uv = project_to_plane(sub, cen, u, v)
                                area = float(estimate_uv_bbox_area(uv))
                            except Exception:
                                area = float(len(sub))
                            if area > best_area:
                                best_area = area
                                best_i = i
                        mk = masks[best_i]
                        pts = pts[mk]
                        idx = idx[mk]
            except Exception:
                pass

            # 2) 小步 IRLS 在主墙片层上精修参考面（若有初值）
            try:
                if model is not None and len(pts) >= 3:
                    model = fit_plane_irls(pts, init_model=model, max_iters=int(getattr(Config, 'FACADE_IRLS_ITERS', 3)),
                                           huber_delta=float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0))/1000.0)
            except Exception:
                pass

            # 3) 可选：垂直约束参考面（仅用于平整度，垂直度仍按自由法向）
            try:
                use_vertical_ref = bool(getattr(Config, 'QUALITY_VERTICAL_REF_PLANE', False))
            except Exception:
                use_vertical_ref = False

            if use_vertical_ref and len(pts) >= 3:
                try:
                    cen = np.mean(pts, axis=0)
                    if model is None:
                        # fallback: 基于当前点集拟合
                        model = fit_plane_irls(pts, init_model=None, max_iters=int(getattr(Config, 'FACADE_IRLS_ITERS', 3)))
                    n = np.asarray(model[:3], dtype=float)
                    n_xy = np.array([float(n[0]), float(n[1]), 0.0], dtype=float)
                    if np.linalg.norm(n_xy) < 1e-6:
                        n_xy = np.array([1.0, 0.0, 0.0], dtype=float)
                    n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-12)
                    d_xy = -float(np.dot(n_xy, cen))
                    model = np.array([n_xy[0], n_xy[1], n_xy[2], d_xy], dtype=float)
                except Exception:
                    pass

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(float))
            gsize = float(Config.QUALITY_GRID_SIZE if grid_size is None else grid_size)
            rsize = float(Config.RULER_SIZE if ruler_size is None else ruler_size)
            rstep = float(Config.RULER_STEP if ruler_step is None else ruler_step)
            # 使用更新后的参考面与中心（若已计算）
            facade_ref = dict(facade)
            try:
                if model is not None:
                    facade_ref['plane_model'] = [float(x) for x in model]
                    facade_ref['center'] = [float(x) for x in (np.mean(pts, axis=0) if len(pts) else np.asarray(facade.get('center') or [0,0,0], dtype=float))]
            except Exception:
                pass

            result = compute_facade_quality(
                facade_info=facade_ref,
                pcd=pcd,
                grid_size=gsize,
                flatness_limit=float(flatness_limit),
                verticality_limit_mm=float(verticality_limit_mm),
                ruler_size=rsize,
                ruler_step=rstep,
                max_windows=max_windows,
            )
            # 将映射重新关联到全局索引，以便于着色
            result = dict(result)
            result['__global_indices'] = idx.tolist()
            result['__index_space'] = 'facade_local_to_cloud_global'
            if profile is not None:
                result['profile_snapshot'] = profile.snapshot()
            if results_dir:
                try:
                    from services.result_export_service import ResultExportService
                    colors = data.get('color')
                    result['artifacts'] = ResultExportService().export_facade(
                        results_dir, facade.get('id', 0), np.asarray(pos), colors, result)
                except Exception as export_error:
                    result['export_error'] = str(export_error)
            return result
        except Exception as e:
            print(f'立面检测(FacadeService: compute_quality)操作失败: {e}', flush=True)
            return None

    def export_quality_result(self, cloud_name: str, facade: dict, quality_result: dict,
                              results_dir, pixel_size: float = 0.01) -> Optional[dict]:
        """显式导出质量结果；与 compute_quality 使用完全相同的局部索引映射。"""
        data = self._viewport.get_cloud_data(cloud_name)
        if data is None:
            return None
        from services.result_export_service import ResultExportService
        return ResultExportService().export_facade(
            results_dir, facade.get('id', 0), np.asarray(data['pos']),
            data.get('color'), quality_result, pixel_size=pixel_size)

    def apply_quality_colors(self, cloud_name: str, quality_result: dict,
                              base_color: Tuple[float, float, float] = (0.75, 0.75, 0.75)) -> None:
        """对立面子集应用逐点质量颜色，其余点保留为base_color."""
        try:
            if self._render is None:
                pass
            data = self._viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return
            import numpy as np
            n = len(pos)
            original = data.get('color')
            colors = (np.asarray(original, dtype=np.float32).copy()
                      if original is not None and np.asarray(original).shape == (n, 3)
                      else np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1)))
            idx = np.asarray(quality_result.get('__global_indices') or [], dtype=int).reshape(-1)
            local = np.asarray(quality_result.get('defect_local_indices') or [], dtype=int).reshape(-1)
            qcols = np.asarray(quality_result.get('defect_colors') or [], dtype=np.float32).reshape(-1, 3)
            valid = (local >= 0) & (local < len(idx))
            gi = idx[local[valid]] if len(local) == len(qcols) else np.empty(0, dtype=int)
            valid_g = (gi >= 0) & (gi < n)
            if len(gi) == len(qcols[valid]) :
                colors[gi[valid_g]] = qcols[valid][valid_g]
            self._viewport.update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f'立面检测(FacadeService: apply_quality_colors)操作失败: {e}', flush=True)

    def restore_highlight(self, cloud_name: str, facades: list[dict]) -> None:
        try:
            if self._render is not None:
                self._render.highlight_facades(cloud_name, facades)
        except Exception:
            pass

    # ---------------- Internal helpers ----------------
    def _persist_results(self, project_uuid: str, facades: list[dict]) -> None:
        """将检测结果交由仓储层保存，服务层不直接管理 ORM 会话。"""
        try:
            ResultsRepo.save_detected_facades(project_uuid, facades)
        except Exception as e:
            print(f'FacadeService: 持久化失败: {e}', flush=True)
