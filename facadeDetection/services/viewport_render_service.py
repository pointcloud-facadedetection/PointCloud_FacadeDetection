from __future__ import annotations

from typing import Optional, Callable, Tuple, Dict

import numpy as np
from config.settings import Config


class ViewportRenderService:
    """
    封装视口的渲染交互逻辑，提供统一的 API 给上层服务使用。
    """

    def __init__(self, viewport, db):
        self.viewport = viewport
        self.db = db
        self._pick_mode = False
        self._picked_points = []
        # Facade coloring scheme and selection state
        self._facade_colors = dict(Config.FACADE_TYPE_COLORS)
        self._highlight_color = tuple(Config.HIGHLIGHT_COLOR)
        self._selected_facade_id: Optional[int] = None
        self._facades_cache: Dict[str, list[dict]] = {}

    # 通知渲染器：显示点云，可选颜色
    def show_point_cloud(self, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        # 给定视口应提供用于添加点数据的 API。
        if hasattr(self.viewport, 'add_point_cloud'):
            self.viewport.add_point_cloud(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud_numpy'):
            self.viewport.add_cloud_numpy(name=name, points=points, colors=colors)
        elif hasattr(self.viewport, 'add_cloud'):
            self.viewport.add_cloud(name, points, colors)
        else:
            raise RuntimeError('Viewport does not support adding point cloud data')

    def show_image(self, name: str, image: np.ndarray):
        if hasattr(self.viewport, 'show_image'):
            self.viewport.show_image(name, image)
        else:
            raise RuntimeError('Viewport does not support image display')

    def get_widget(self):
        if hasattr(self.viewport, 'get_widget'):
            return self.viewport.get_widget()
        if hasattr(self.viewport, 'native') and hasattr(self.viewport.native, 'widget'):
            return self.viewport.native.widget
        raise RuntimeError('Viewport has no Qt widget to embed')

    def enter_pick_mode(self, callback: Optional[Callable] = None,
                        cloud_name: Optional[str] = None,
                        pick_radius: int = 8):
        if not hasattr(self.viewport, 'enter_pick_mode'):
            raise RuntimeError('Viewport does not support pick mode')
        self.viewport.enter_pick_mode(
            cloud_name=cloud_name,
            pick_radius=pick_radius,
            callback=callback
        )
        self._pick_mode = True

    def exit_pick_mode(self):
        if hasattr(self.viewport, 'exit_pick_mode'):
            self.viewport.exit_pick_mode()
        self._pick_mode = False

    def is_pick_mode(self) -> bool:
        return self._pick_mode

    def add_pick_marker(self, point):
        """记录拾取点并刷新视口标记。"""
        self._picked_points.append(point)
        if hasattr(self.viewport, 'update_pick_markers'):
            self.viewport.update_pick_markers(src_points=self._picked_points)

    def clear_pick_markers(self):
        """清除所有拾取标记并重置记录。"""
        if hasattr(self.viewport, 'clear_pick_markers'):
            self.viewport.clear_pick_markers()
        self._picked_points.clear()

    # ---- Facade highlighting ----
    def highlight_facades(self, cloud_name: str, facades: list[dict], base_color=(0.75, 0.75, 0.75)):
        """
        立面着色策略（统一颜色规则 + 选中高亮）：
        - 非立面点使用基础色 base_color。
        - 所有垂直立面统一使用 Config.FACADE_TYPE_COLORS['vertical_facade']。
        - 所有水平面统一使用 Config.FACADE_TYPE_COLORS['horizontal']。
        - 其它（如倾斜面）统一使用 Config.FACADE_TYPE_COLORS['inclined']（若存在）。
        - 当存在选中的立面 self._selected_facade_id 时，该立面的点使用 Config.HIGHLIGHT_COLOR。
        """
        try:
            if not hasattr(self.viewport, 'get_cloud_data'):
                return
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return

            n = len(pos)
            colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))

            try:
                self._facades_cache[cloud_name] = facades or []
            except Exception:
                pass

            for f in facades or []:
                ftype = str(f.get('type') or f.get('type_label') or '').lower()
                col = None
                if 'vertical' in ftype:
                    col = self._facade_colors.get('vertical_facade')
                elif 'horizontal' in ftype:
                    col = self._facade_colors.get('horizontal')
                else:
                    col = self._facade_colors.get('inclined') or self._facade_colors.get('vertical_facade')

                # Apply highlight if selected
                try:
                    fid = int(f.get('id', -1))
                except Exception:
                    fid = -1
                if self._selected_facade_id is not None and fid == self._selected_facade_id:
                    col = self._highlight_color

                if col is None:
                    continue
                col = np.asarray(col, dtype=np.float32)

                idx = np.asarray(f.get('inlier_indices') or [], dtype=int)
                m = (idx >= 0) & (idx < n)
                idx = idx[m]
                if len(idx):
                    colors[idx] = col

            self.viewport.update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"highlight_facades failed: {e}", flush=True)

    def set_facade_colors(self,
                          vertical: Tuple[float, float, float] | None = None,
                          horizontal: Tuple[float, float, float] | None = None,
                          inclined: Tuple[float, float, float] | None = None,
                          highlight: Tuple[float, float, float] | None = None) -> None:
        """更新立面配色方案."""
        if vertical is not None:
            self._facade_colors['vertical_facade'] = tuple(vertical)
        if horizontal is not None:
            self._facade_colors['horizontal'] = tuple(horizontal)
        if inclined is not None:
            self._facade_colors['inclined'] = tuple(inclined)
        if highlight is not None:
            self._highlight_color = tuple(highlight)

    def select_facade(self, cloud_name: str, facade_id: int) -> None:
        """根据 ID 突出显示单个立面，同时保持其他立面的文字颜色统一。"""
        try:
            self._selected_facade_id = int(facade_id)
        except Exception:
            self._selected_facade_id = None
        facades = self._facades_cache.get(cloud_name) or []
        if facades:
            self.highlight_facades(cloud_name, facades)

    def clear_selected_facade(self, cloud_name: str | None = None) -> None:
        self._selected_facade_id = None
        if cloud_name is None:
            return
        facades = self._facades_cache.get(cloud_name) or []
        if facades:
            self.highlight_facades(cloud_name, facades)

    def set_global_point_color(self, color: Tuple[float, float, float]) -> None:
        """在视口内将整个点云场景统一着色."""
        try:
            if not hasattr(self.viewport, 'get_cloud_names'):
                return
            names = self.viewport.get_cloud_names()
            if not names:
                return
            for name in names:
                data = self.viewport.get_cloud_data(name)
                if data is None:
                    continue
                pos = data.get('pos')
                if pos is None or len(pos) == 0:
                    continue
                n = len(pos)
                col = np.tile(np.asarray(color, dtype=np.float32).reshape(1, 3), (n, 1))
                self.viewport.update_cloud_color(name, col)
        except Exception as e:
            print(f"set_global_point_color failed: {e}", flush=True)

    # ---- 热力图渲染 ----
    def colorize_by_rgb(self, cloud_name: str, indices: np.ndarray, colors_rgb: np.ndarray,
                        base_color=(0.18, 0.20, 0.24)) -> None:
        """按算法已生成的 RGB 颜色着色，避免再次做标量归一化/色带计算。"""
        try:
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None or data.get('pos') is None or len(data['pos']) == 0:
                return
            n = len(data['pos'])
            idx = np.asarray(indices, dtype=int).reshape(-1)
            rgb = np.asarray(colors_rgb, dtype=np.float32).reshape(-1, 3)
            if len(idx) != len(rgb):
                return
            valid = (idx >= 0) & (idx < n)
            if not np.any(valid):
                return
            original = data.get('color')
            # Keep RGB only when it is sufficiently dark; bright source RGB
            # makes defect colors unreadable in the 3D view.
            if original is not None and np.asarray(original).shape == (n, 3):
                source = np.asarray(original, dtype=np.float32)
                colors = np.clip(source * 0.35, 0.0, 0.42)
            else:
                colors = np.tile(np.asarray(base_color, dtype=np.float32), (n, 1))
            colors[idx[valid]] = np.clip(rgb[valid] * 1.15, 0.0, 1.0)
            self.viewport.update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"colorize_by_rgb failed: {e}", flush=True)

    def colorize_by_scalar(self, cloud_name: str, indices: np.ndarray, values: np.ndarray,
                            vmin: float | None = None, vmax: float | None = None,
                            base_color=(0.75, 0.75, 0.75), cmap: str = 'turbo') -> None:
        """
        根据给定的标量值对指定点进行热力着色，其余点使用 base_color。
        - indices: 全局点索引（0..N-1）的一维数组
        - values: 与 indices 对齐的浮点数组
        """
        try:
            if not hasattr(self.viewport, 'get_cloud_data'):
                return
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return
            n = len(pos)
            idx = np.asarray(indices, dtype=int).reshape(-1)
            val = np.asarray(values, dtype=float).reshape(-1)
            if len(idx) == 0 or len(idx) != len(val):
                return
            m = (idx >= 0) & (idx < n)
            idx = idx[m]
            val = val[m]
            if len(idx) == 0:
                return
            if vmin is None:
                vmin = float(np.min(val))
            if vmax is None:
                vmax = float(np.max(val))
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-6
            # normalize to 0..1
            t = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
            colors = np.tile(np.asarray(base_color, dtype=np.float32).reshape(1, 3), (n, 1))
            colors[idx] = self._colormap(t, cmap)
            self.viewport.update_cloud_color(cloud_name, colors)
        except Exception as e:
            print(f"colorize_by_scalar failed: {e}", flush=True)

    @staticmethod
    def _colormap(t: np.ndarray, cmap: str = 'turbo') -> np.ndarray:
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if cmap in ('diverging', 'diverging_blue_white_red', 'signed'):
            # Signed deviation: -1=recessed (blue), 0=reference (white),
            # +1=protruding (red).  This is deliberately symmetric.
            t = np.clip(t, 0.0, 1.0)
            blue = np.array([0.05, 0.35, 0.95], dtype=np.float32)
            white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            red = np.array([0.95, 0.08, 0.04], dtype=np.float32)
            arr = np.empty((len(t), 3), dtype=np.float32)
            left = t <= 0.5
            q = (t[left] * 2.0)[:, None]
            arr[left] = blue + (white - blue) * q
            q = ((t[~left] - 0.5) * 2.0)[:, None]
            arr[~left] = white + (red - white) * q
            return np.clip(arr, 0.0, 1.0)
        if cmap == 'turbo':
            # Lightweight Turbo approximation
            r = 0.135 - 0.157*t + 2.776*t**2 - 2.443*t**3
            g = 0.091 - 1.33*t + 3.51*t**2 - 1.84*t**3
            b = 0.106 + 1.097*t - 2.295*t**2 + 1.98*t**3
            arr = np.stack([r, g, b], axis=1)
        else:
            # simple blue->cyan->yellow->red
            arr = np.empty((len(t), 3), dtype=np.float32)
            arr[:, 0] = np.clip(2*t - 0.5, 0.0, 1.0)
            arr[:, 1] = np.clip(2*t, 0.0, 1.0)
            arr[:, 2] = np.clip(1.5 - 2*t, 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    # ---- ROI utilities (selection -> facade plane -> building BBOX) ----
    def fit_plane_on_selection(self, cloud_name: str, indices: list[int],
                               prefer_frontmost: bool = True,
                               front_fraction: float = 0.25,
                               mode: str = 'dominant') -> tuple[np.ndarray | None, np.ndarray | None]:
        """根据选定的索引拟合一个平面，并返回 (plane_model[4], selected_points)。
        plane_model：[nx, ny, nz, d]，包含单位法向量；若操作失败，则返回 (None, None)。

        prefer_frontmost:
            当为 True 时，优先使用靠近相机前方向的前景点进行拟合（基于投影深度的前 20%~30% 片层），
            以减少后景干扰，更符合“最近立面”需求；若无法获取投影或结果过少，则回退到全部选中点。
        front_fraction:
            前景分位阈值（0~1），默认 0.25 表示使用最靠前的 25% 深度点。
        mode:
            选择候选平面的策略：
            - 'auto': 前景可用且面积足够时选前景，否则选面积最大者（可绕过脚手架等遮挡层）
            - 'front': 总是选前景（若不可用则回退全部）
            - 'dominant': 总是选面积最大者
        """
        try:
            data = self.viewport.get_cloud_data(cloud_name)
            if data is None:
                return None, None
            pos = data.get('pos')
            if pos is None or len(pos) == 0:
                return None, None
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(pos))]
            if len(idx) < 3:
                return None, None
            pts = np.asarray(pos)[idx]

            # 依据投影深度选择“前景点”子集以拟合最近立面（可选）
            front_pts = None
            front_idx = None
            if hasattr(self.viewport, 'project_points'):
                try:
                    proj = self.viewport.project_points(pts)
                    if proj is not None:
                        screen, valid = proj
                        if screen is not None and len(screen) == len(pts):
                            z = screen[:, 2]
                            m = np.asarray(valid, dtype=bool)
                            z = z[m]
                            if z.size >= 10:
                                # 统一“越小越近”：若范围更贴近上界说明符号反了，则取相反分位
                                frac = float(np.clip(front_fraction, 0.05, 0.5))
                                zmin, zmax = float(np.min(z)), float(np.max(z))
                                use_small = (np.median(z) - zmin) < (zmax - np.median(z))
                                thresh = zmin + frac * (zmax - zmin) if use_small else zmax - frac * (zmax - zmin)
                                sel = (screen[m, 2] <= thresh) if use_small else (screen[m, 2] >= thresh)
                                keep_mask_full = np.zeros(len(pts), dtype=bool)
                                keep_mask_full[m] = sel
                                # 过少则选择前 K 个“最靠前”的点
                                if keep_mask_full.sum() < 30:
                                    order = np.argsort(screen[m, 2]) if use_small else np.argsort(-screen[m, 2])
                                    k = max(30, min(len(order)//3, 2000))
                                    pick = np.zeros(len(pts), dtype=bool)
                                    true_idx = np.nonzero(m)[0]
                                    pick[true_idx[order[:k]]] = True
                                    keep_mask_full = pick
                                if keep_mask_full.any():
                                    front_pts = pts[keep_mask_full]
                                    front_idx = idx[keep_mask_full]
                except Exception:
                    pass

            # 计算两个候选：前景（若可用）与全部选中点
            candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            def _fit_and_area(points: np.ndarray) -> tuple[np.ndarray, float]:
                if points is None or len(points) < 3:
                    return None, 0.0
                try:
                    from algorithms.geometry import fit_plane_irls
                    model = fit_plane_irls(points, init_model=None, max_iters=int(getattr(Config, 'FACADE_IRLS_ITERS', 3)))
                except Exception:
                    from algorithms.geometry import fit_plane_svd
                    model = fit_plane_svd(points)
                # 估计 UV 面积用于偏好大尺度墙面
                try:
                    from algorithms.geometry import plane_basis_from_normal, project_to_plane, estimate_uv_bbox_area
                    nrm = model[:3]
                    nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
                    cen = np.mean(points, axis=0)
                    u, v = plane_basis_from_normal(nrm)
                    uv = project_to_plane(points, cen, u, v)
                    area = float(estimate_uv_bbox_area(uv))
                except Exception:
                    area = 0.0
                # 归一化法向
                n = model[:3]
                n = n / (np.linalg.norm(n) + 1e-12)
                model[:3] = n
                return np.asarray(model, dtype=float), area

            # 前景候选
            if prefer_frontmost and front_pts is not None and len(front_pts) >= 3:
                m_front, _ = _fit_and_area(front_pts)
                candidates.append((m_front, front_pts, front_idx))

            # 全部选中点候选
            m_all, _ = _fit_and_area(pts)
            candidates.append((m_all, pts, idx))

            # 选择策略
            chosen_model = None
            chosen_pts = None
            chosen_idx = None
            try:
                # 垂直筛选阈值
                nz_thr = float(getattr(Config, 'VERTICAL_NZ_THR', 0.20))
            except Exception:
                nz_thr = 0.20

            def _is_vertical(m: np.ndarray) -> bool:
                if m is None:
                    return False
                n = m[:3]
                n = n / (np.linalg.norm(n) + 1e-12)
                return abs(float(n[2])) <= nz_thr

            # 面积比较（优先大尺度主墙面）
            areas = []
            for m, p, _ in candidates:
                if m is None or p is None:
                    areas.append(0.0)
                else:
                    try:
                        from algorithms.geometry import plane_basis_from_normal, project_to_plane, estimate_uv_bbox_area
                        nrm = m[:3] / (np.linalg.norm(m[:3]) + 1e-12)
                        cen = np.mean(p, axis=0)
                        u, v = plane_basis_from_normal(nrm)
                        uv = project_to_plane(p, cen, u, v)
                        areas.append(float(estimate_uv_bbox_area(uv)))
                    except Exception:
                        areas.append(0.0)

            # 选择模型
            if mode == 'front' and len(candidates) >= 1:
                # 选择第一个（前景）有效且垂直，否则回退到面积最大
                m, p, id_arr = candidates[0]
                if _is_vertical(m):
                    chosen_model, chosen_pts, chosen_idx = m, p, id_arr
                else:
                    k = int(np.argmax(areas))
                    chosen_model, chosen_pts, chosen_idx = candidates[k]
            else:
                # auto/dominant
                if mode == 'auto' and prefer_frontmost and len(candidates) >= 2 and _is_vertical(candidates[0][0]) and areas[0] >= max(areas[1] * 0.5, 5.0):
                    chosen_model, chosen_pts, chosen_idx = candidates[0]
                else:
                    k = int(np.argmax(areas))
                    chosen_model, chosen_pts, chosen_idx = candidates[k]

            if chosen_model is None or chosen_pts is None:
                return None, None

            # 使用深度剥离进一步稳定主墙（避免凸出物干扰）
            try:
                from algorithms.geometry import split_by_depth_histogram
                # 自适应距离容差（米）
                tol_m = float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0)) / 1000.0
                masks = split_by_depth_histogram(chosen_pts, chosen_model, tol_m, min_points=80)
                if masks and len(masks) > 1:
                    # 选择 UV 面积最大的片层
                    best_i = 0
                    best_area = -1.0
                    for i, mk in enumerate(masks):
                        sub = chosen_pts[mk]
                        try:
                            from algorithms.geometry import plane_basis_from_normal, project_to_plane, estimate_uv_bbox_area
                            nrm = chosen_model[:3] / (np.linalg.norm(chosen_model[:3]) + 1e-12)
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
                    chosen_pts = chosen_pts[mk]
                    chosen_idx = chosen_idx[mk]
                    # 小步 IRLS 精修
                    try:
                        from algorithms.geometry import fit_plane_irls
                        chosen_model = fit_plane_irls(chosen_pts, init_model=chosen_model, max_iters=3,
                                                      huber_delta=float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0))/1000.0)
                    except Exception:
                        pass
            except Exception:
                pass

            # 归一化法向并返回
            n = chosen_model[:3]
            n = n / (np.linalg.norm(n) + 1e-12)
            chosen_model[:3] = n
            return np.asarray(chosen_model, dtype=float), chosen_pts
        except Exception:
            return None, None

    def compute_building_bbox_from_selection(self, cloud_name: str, indices: list[int],
                                             fixed_margin: float | None = None, max_iter: int = 8,
                                             max_volume_ratio: float | None = None,
                                             plane: np.ndarray | None = None,
                                             tol: float | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            data = self.viewport.get_cloud_data(cloud_name)
            if not data or data.get('pos') is None:
                return None, None
            all_pos = np.asarray(data['pos'], dtype=float).reshape(-1, 3)
            idx = np.unique(np.asarray(indices, dtype=int).reshape(-1))
            idx = idx[(idx >= 0) & (idx < len(all_pos))]
            if len(idx) < 3:
                return None, None
            seed = all_pos[idx]
            scale = self._estimate_scene_scale(cloud_name)

            if plane is None:
                plane, _ = self.fit_plane_on_selection(cloud_name, idx.tolist(),
                                                       prefer_frontmost=False, mode='dominant')
            if plane is None:
                return None, None
            n = np.asarray(plane[:3], dtype=float)
            n /= np.linalg.norm(n) + 1e-12
            d = float(plane[3])
            if abs(n[2]) > float(getattr(Config, 'VERTICAL_NZ_THR', 0.20)):
                return None, None

            from algorithms.geometry import plane_axes, connected_components_2d_grid
            u, v = plane_axes(n, 'vertical_facade')
            origin = np.mean(seed, axis=0)
            seed_uv = np.column_stack(((seed-origin) @ u, (seed-origin) @ v))

            if tol is None:
                tol = float(np.clip(scale * 0.01, 0.08, 0.5))
            else:
                tol = float(np.clip(tol, 0.05, max(scale * 0.03, 0.5)))
            signed = all_pos @ n + d
            near = np.abs(signed) <= tol
            near_ids = np.flatnonzero(near)
            if len(near_ids) < 20:
                return None, None
            near_uv = np.column_stack(((all_pos[near_ids]-origin) @ u,
                                       (all_pos[near_ids]-origin) @ v))

            comps = connected_components_2d_grid(near_uv, grid_size=None, min_cells=3,
                                                  adaptive_ratio=2.5, sample_ratio=0.1,
                                                  connectivity=8, close_radius_cells=3)
            if comps:
                def score(mask):
                    pts = near_uv[mask]
                    overlap = np.sum((seed_uv[:, 0] >= pts[:, 0].min()) &
                                     (seed_uv[:, 0] <= pts[:, 0].max()) &
                                     (seed_uv[:, 1] >= pts[:, 1].min()) &
                                     (seed_uv[:, 1] <= pts[:, 1].max()))
                    return (overlap / max(len(seed), 1)) * max(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]), 1e-6)
                component = comps[int(np.argmax([score(m) for m in comps]))]
                wall_ids = near_ids[component]
            else:
                wall_ids = near_ids
            wall_uv = np.column_stack(((all_pos[wall_ids]-origin) @ u,
                                       (all_pos[wall_ids]-origin) @ v))
            umin, vmin = np.min(wall_uv, axis=0)
            umax, vmax = np.max(wall_uv, axis=0)

            footprint = (((all_pos-origin) @ u >= umin) & ((all_pos-origin) @ u <= umax) &
                         ((all_pos-origin) @ v >= vmin) & ((all_pos-origin) @ v <= vmax))
            ids = np.flatnonzero(footprint)
            if len(ids) < 3:
                ids = wall_ids
            coords = np.column_stack(((all_pos[ids]-origin) @ u,
                                       (all_pos[ids]-origin) @ v,
                                       (all_pos[ids]-origin) @ n))
            lo = np.min(coords, axis=0); hi = np.max(coords, axis=0)
            result = origin + np.array([u, v, n]).T @ np.array([[(lo[0]+hi[0])/2],
                                                                  [(lo[1]+hi[1])/2],
                                                                  [(lo[2]+hi[2])/2]])[:, 0]
            half = np.array([(hi[0]-lo[0])/2, (hi[1]-lo[1])/2, (hi[2]-lo[2])/2])
            axes = np.column_stack([u, v, n])
            corners = result + np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]) * half @ axes.T
            return np.min(corners, axis=0), np.max(corners, axis=0)
        except Exception as e:
            print(f"compute_building_bbox_from_selection failed: {e}", flush=True)
            return None, None

    def _estimate_scene_scale(self, cloud_name: str | None = None) -> float:
        """估计场景尺度（点云最大范围）。"""
        try:
            if cloud_name:
                data = self.viewport.get_cloud_data(cloud_name)
                if data and data.get('pos') is not None and len(data['pos']) > 0:
                    pos = np.asarray(data['pos'])
                    extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                    return max(float(np.max(extent)), 1.0)
            # 如果没有指定 cloud_name，尝试所有点云
            max_extent = 0.0
            for name in self.viewport.get_cloud_names():
                data = self.viewport.get_cloud_data(name)
                if data and data.get('pos') is not None and len(data['pos']) > 0:
                    pos = np.asarray(data['pos'])
                    extent = np.max(pos, axis=0) - np.min(pos, axis=0)
                    max_extent = max(max_extent, float(np.max(extent)))
            return max(max_extent, 1.0)
        except Exception:
            return 50.0  # 默认 50 米

    def visualize_building_bbox(self, min_bound, max_bound, color=(1.0, 0.2, 0.2)) -> None:
        """Render a 3D bbox for the building ROI in the viewport."""
        try:
            if hasattr(self.viewport, 'show_roi_bbox'):
                self.viewport.show_roi_bbox(min_bound, max_bound, color=color)
        except Exception:
            pass