from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.storage import Storage


class TwoDMatchingService:
    """2D-3D 图片匹配服务：照片-点云立面配准 + 透明热力图叠加。"""

    _DISPLAY_MODES = [
        'ruler_flatness_area',
        'ruler_verticality_area',
        'global_plane_flatness_area',
        'global_plane_verticality_area',
    ]

    # 与 photo_match_service 保持一致的 intensity 属性候选名
    _INTENSITY_ATTRIBUTE_KEYS = (
        'intensity', 'intensities', 'scalar_intensity', 'Intensity',
    )

    # ==================================================================
    # 主入口
    # ==================================================================
    @staticmethod
    def match(facade_data: dict, **deps) -> dict:
        """执行 2D 图片与 3D 点云匹配，输出照片-热力叠加图。

        核心链路：
          1) 提取平面参考系 plane_frame（与点云热力图同源）
          2) estimate_photo_cloud_match(..., plane_frame=...) 输出
             平面单应 H_uv_to_photo
          3) 四组模式统一走"热力图 warp"：透明热力图 → 原始照片
        """
        facade_no = int(facade_data.get('display_no') or facade_data.get('id', 0))
        print(f'[PCFD][PhotoMatch] start facade_no={facade_no}', flush=True)

        # ---------- 依赖解析 ----------
        project = deps.get('project')
        station_service = deps.get('station_service')
        pointcloud_service = deps.get('pointcloud_service')
        facade_service = deps.get('facade_service')

        if project is None:
            print(f'[PCFD][PhotoMatch] fail reason=no_project', flush=True)
            return {'success': False, 'reason': '当前未选择项目'}

        # ---------- 1. 读取照片 ----------
        photo_path = TwoDMatchingService._resolve_photo_path(
            project, facade_data, station_service)
        print(f'[PCFD][PhotoMatch] photo_path={photo_path}', flush=True)
        if not photo_path:
            print(f'[PCFD][PhotoMatch] fail reason=no_photo', flush=True)
            return {'success': False,
                    'reason': '未找到该站点的原始照片，请先上传现场照片'}
        photo_bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        if photo_bgr is None:
            print(f'[PCFD][PhotoMatch] fail reason=photo_read_fail path={photo_path}', flush=True)
            return {'success': False, 'reason': f'无法读取照片：{photo_path}'}

        photo_h, photo_w = photo_bgr.shape[:2]
        print(f'[PCFD][PhotoMatch] photo_shape={photo_bgr.shape}', flush=True)

        # ---------- 1b. 按照片宽高比推导映射图尺寸 ----------
        mapping_size = TwoDMatchingService._mapping_size_for_photo(
            photo_bgr, max_side=1600, min_side=768)
        print(f'[PCFD][PhotoMatch] mapping_size={mapping_size} '
              f'(photo={photo_w}x{photo_h})', flush=True)

        # ---------- 2. 读取点云与位姿 ----------
        cloud = TwoDMatchingService._resolve_cloud_name(
            facade_data, pointcloud_service)
        print(f'[PCFD][PhotoMatch] cloud_name={cloud}', flush=True)
        if not cloud or facade_service is None:
            print(f'[PCFD][PhotoMatch] fail reason=no_cloud '
                  f'facade_service={facade_service}', flush=True)
            return {'success': False, 'reason': '无法获取点云数据'}
        dataset = facade_service.get_dataset(cloud)

        source_points = np.asarray(dataset.processed_raw_points, dtype=np.float64)
        print(f'[PCFD][PhotoMatch] source_points_shape={source_points.shape}',
              flush=True)

        # ★ 用与 photo_match_service 相同的方式读取原始 intensity/RGB
        source_values = TwoDMatchingService._load_source_values(
            dataset, pointcloud_service, cloud, len(source_points))
        has_intensity = source_values is not None
        print(f'[PCFD][PhotoMatch] source_intensity_exists={has_intensity}',
              flush=True)

        # 下采样点云（保留 intensity）
        if has_intensity:
            point_cloud, point_values = TwoDMatchingService._voxel_downsample_with_values(
                source_points, source_values, voxel_size=0.1)
        else:
            point_cloud = TwoDMatchingService._voxel_downsample(
                source_points, voxel_size=0.1)
            point_values = None
        print(f'[PCFD][PhotoMatch] point_cloud_downsampled_shape={point_cloud.shape} '
              f'values_downsampled={"yes" if point_values is not None else "no"}',
              flush=True)

        scan_pose = TwoDMatchingService._resolve_scan_pose(
            dataset, station_service, facade_data)
        print(f'[PCFD][PhotoMatch] scan_pose_exists={scan_pose is not None}',
              flush=True)
        if scan_pose is None:
            print(f'[PCFD][PhotoMatch] fail reason=no_scan_pose', flush=True)
            return {'success': False,
                    'reason': '该站点缺少扫描仪位姿（transformToGlobal），'
                              '无法执行照片-点云匹配'}

        # ---------- 3. 照片-点云配准（含平面单应） ----------
        from utils.photo_matching import (
            generate_cloud_mapping_image,
            estimate_photo_cloud_match,
        )

        quality = facade_data.get('quality_report') or {}

        # ★ 提取平面参考系（与点云热力图同源）
        plane_frame = TwoDMatchingService._extract_plane_frame(
            facade_data, quality)
        print(f'[PCFD][PhotoMatch] plane_frame_ready={plane_frame is not None}',
              flush=True)

        try:
            print(f'[PCFD][PhotoMatch] generate_cloud_mapping...', flush=True)
            mapping = generate_cloud_mapping_image(
                point_cloud, scan_pose,
                colors=point_values,          # ★ 传入原始 intensity
                image_size=mapping_size,
                crop_subject=False)            # ★ 裁到主体，减少背景干扰
            print(f'[PCFD][PhotoMatch] mapping_size_actual='
                  f'{mapping.mapping_image.shape}', flush=True)
            print(f'[PCFD][PhotoMatch] estimate_photo_cloud_match...', flush=True)
            match_result = estimate_photo_cloud_match(
                point_cloud,
                photo_bgr,
                mapping,
                mapping.camera_matrix,
                mapping.extrinsic,
                depth_image=mapping.depth_image,
                pixel_point_index=mapping.pixel_point_index,
                plane_frame=plane_frame,        # ★ 让 photo_matching 输出 H
                photo_path=photo_path,          # ★ EXIF 焦距/Orientation
            )
            print(f'[PCFD][PhotoMatch] match_inliers={match_result.inlier_count} '
                  f'rmse={match_result.reprojection_rmse_px:.2f}px '
                  f'H_ready={match_result.homography_uv_to_photo is not None} '
                  f'H_inliers={match_result.homography_inlier_count} '
                  f'H_rmse={match_result.homography_rmse_px:.2f}px',
                  flush=True)
            # ---------- 3b. 对齐质量门禁 ----------
            try:
                from utils.photo_matching import _quality_gate_metrics
                gate = _quality_gate_metrics(match_result, plane_frame)
                print(f'[PCFD][PhotoMatch] quality_gate inliers={gate["inlier_count"]} '
                      f'rmse={gate["reprojection_rmse_px"]:.2f}px '
                      f'H_rmse={gate["homography_rmse_px"]:.2f}px '
                      f'corner_rmse={gate["corner_reprojection_rmse_px"]}',
                      flush=True)
                quality['__alignment_quality'] = gate
                if (gate['reprojection_rmse_px'] > 20.0
                        or gate['inlier_count'] < 8):
                    print('[PCFD][PhotoMatch] WARNING alignment quality low: '
                          '贴图可能出现偏移', flush=True)
            except Exception as exc:
                print(f'[PCFD][PhotoMatch] quality_gate error={exc}', flush=True)
        except Exception as exc:
            print(f'[PCFD][PhotoMatch] fail reason=match_error error={exc}',
                  flush=True)
            return {'success': False, 'reason': f'特征匹配失败：{exc}'}

        # ---------- 4. 逐模式叠加（四组统一路径：热力图 warp） ----------
        results_dir = TwoDMatchingService._resolve_results_dir(
            project, facade_data, station_service)
        facade_no = int(facade_data.get('display_no') or facade_data.get('id', 0))

        exported = {}
        for mode in TwoDMatchingService._DISPLAY_MODES:
            try:
                print(f'[PCFD][PhotoMatch] overlay mode={mode} ...', flush=True)
                overlay_path = TwoDMatchingService._generate_matched_overlay(
                    mode=mode,
                    facade_no=facade_no,
                    photo_bgr=photo_bgr,
                    match_result=match_result,
                    facade_data=facade_data,
                    quality=quality,
                    results_dir=results_dir,
                    dataset=dataset,
                )
                if overlay_path:
                    exported[mode] = str(overlay_path)
                    print(f'[PCFD][PhotoMatch] mode={mode} saved={overlay_path}',
                          flush=True)
                else:
                    print(f'[PCFD][PhotoMatch] mode={mode} skipped', flush=True)
            except Exception as exc:
                print(f'[PCFD][PhotoMatch] mode={mode} error={exc}', flush=True)
                continue

        # ---------- 5. 更新 quality_report 中的图片路径 ----------
        TwoDMatchingService._update_export_context(quality, exported)
        print(f'[PCFD][PhotoMatch] finish success={bool(exported)} '
              f'modes={list(exported.keys())}', flush=True)

        return {
            'success': bool(exported),
            'paths': exported,
            'reason': '' if exported else '所有模式的热力图叠加均失败',
        }

    # ==================================================================
    # 原始 intensity / RGB 读取（对齐 photo_match_service）
    # ==================================================================
    @staticmethod
    def _extract_tensor_intensity(cloud):
        """从 Open3D 张量点云读取 intensity；缺失时回退到 RGB 灰度。"""
        try:
            names = [str(name) for name in cloud.point]
        except Exception:
            names = []
        candidates = list(dict.fromkeys(names + list(
            TwoDMatchingService._INTENSITY_ATTRIBUTE_KEYS)))
        for key in candidates:
            if 'intens' not in key.lower():
                continue
            try:
                return cloud.point[key].numpy().reshape(-1).astype(np.float32)
            except Exception:
                continue
        try:
            colors = cloud.point['colors'].numpy().astype(np.float32)
        except Exception:
            return None
        if colors.ndim != 2 or colors.shape[1] < 3:
            return None
        return (0.299 * colors[:, 0]
                + 0.587 * colors[:, 1]
                + 0.114 * colors[:, 2])

    @staticmethod
    def _resolve_cloud_file(dataset, pointcloud_service, cloud_name):
        """解析点云物理文件路径（多个候选字段）。"""
        candidates = []
        for attr in ('source_path', 'path', 'file_path', 'cloud_path'):
            val = getattr(dataset, attr, None)
            if val:
                candidates.append(str(val))
        meta = getattr(dataset, 'metadata', None)
        if isinstance(meta, dict):
            for key in ('source_path', 'path', 'file_path', 'cloud_path'):
                if meta.get(key):
                    candidates.append(str(meta[key]))
        try:
            if pointcloud_service is not None:
                for method_name in (
                    'resolve_processing_cloud_path',
                    'resolve_processing_cloud_file',
                    'resolve_cloud_path',
                ):
                    method = getattr(pointcloud_service, method_name, None)
                    if callable(method):
                        try:
                            resolved = method(cloud_name)
                        except TypeError:
                            resolved = method()
                        if resolved:
                            candidates.append(str(resolved))
                            break
        except Exception:
            pass
        for raw in candidates:
            p = Path(raw)
            if p.is_file():
                return p
        return None

    @staticmethod
    def _load_source_values(dataset, pointcloud_service, cloud_name, point_count):
        """加载原始 intensity / RGB 灰度值；与 photo_match_service 顺序一致。

        优先顺序：
          1) 直接从点云文件读取（张量 intensity → 张量 RGB → legacy RGB）
          2) dataset.index.get_source_colors() 缓存
        """
        # ---- 1) 点云文件直读 ----
        cloud_path = TwoDMatchingService._resolve_cloud_file(
            dataset, pointcloud_service, cloud_name)
        if cloud_path is not None:
            try:
                import open3d as o3d
                try:
                    tensor_cloud = o3d.t.io.read_point_cloud(str(cloud_path))
                    values = TwoDMatchingService._extract_tensor_intensity(tensor_cloud)
                    if values is not None and len(values) == point_count:
                        print(f'[PCFD][PhotoMatch] intensity source=tensor_file '
                              f'n={len(values)} path={cloud_path}', flush=True)
                        return values
                except Exception as exc:
                    print(f'[PCFD][PhotoMatch] tensor read failed: {exc}',
                          flush=True)
                try:
                    legacy = o3d.io.read_point_cloud(str(cloud_path))
                    if legacy.has_colors():
                        colors = np.asarray(legacy.colors, dtype=np.float32)
                        if len(colors) == point_count and colors.shape[1] >= 3:
                            values = (0.299 * colors[:, 0]
                                      + 0.587 * colors[:, 1]
                                      + 0.114 * colors[:, 2])
                            print(f'[PCFD][PhotoMatch] intensity source=legacy_rgb '
                                  f'n={len(values)} path={cloud_path}', flush=True)
                            return values.astype(np.float32)
                except Exception as exc:
                    print(f'[PCFD][PhotoMatch] legacy read failed: {exc}',
                          flush=True)
            except ImportError:
                pass

        # ---- 2) dataset 缓存 ----
        try:
            cached = dataset.index.get_source_colors()
            if cached is not None:
                cached = np.asarray(cached)
                if (cached.ndim == 2 and cached.shape[1] >= 3
                        and len(cached) == point_count):
                    values = (0.299 * cached[:, 0]
                              + 0.587 * cached[:, 1]
                              + 0.114 * cached[:, 2])
                    print(f'[PCFD][PhotoMatch] intensity source=dataset_rgb '
                          f'n={len(values)}', flush=True)
                    return values.astype(np.float32)
                if cached.ndim == 1 and len(cached) == point_count:
                    print(f'[PCFD][PhotoMatch] intensity source=dataset_scalar '
                          f'n={len(cached)}', flush=True)
                    return cached.astype(np.float32)
        except Exception as exc:
            print(f'[PCFD][PhotoMatch] dataset colors read failed: {exc}',
                  flush=True)

        print('[PCFD][PhotoMatch] intensity source=none (fallback to depth)',
              flush=True)
        return None

    # ==================================================================
    # 平面参考系提取
    # ==================================================================
    @staticmethod
    def _extract_plane_frame(facade_data, quality):
        """提取平面参考系 (origin, u_axis, v_axis, plane_model)。

        唯一数据源：quality.projection_*（与点云热力图渲染完全同源）。
        正交性/共面性校验失败时显式报错并记录，禁止静默重建，
        防止热力图与单应 H 使用不同平面参考系导致整体偏移。
        """
        origin = quality.get('projection_origin')
        u_axis = quality.get('projection_u_axis')
        v_axis = quality.get('projection_v_axis')
        plane_model = facade_data.get('plane_model')
        if plane_model is None:
            plane_model = quality.get('plane_model')

        if (origin is None or u_axis is None
                or v_axis is None or plane_model is None):
            print('[PCFD][PhotoMatch] plane_frame missing: '
                  'quality.projection_* 或 plane_model 缺失', flush=True)
            return None
        try:
            o = np.asarray(origin, dtype=np.float64).reshape(3)
            u = np.asarray(u_axis, dtype=np.float64).reshape(3)
            v = np.asarray(v_axis, dtype=np.float64).reshape(3)
            pm = np.asarray(plane_model, dtype=np.float64).reshape(4)
            u = u / max(np.linalg.norm(u), 1e-12)
            v = v / max(np.linalg.norm(v), 1e-12)
            n = pm[:3] / max(np.linalg.norm(pm[:3]), 1e-12)
            if not (np.isfinite(o).all() and np.isfinite(u).all()
                    and np.isfinite(v).all() and np.isfinite(pm).all()):
                raise ValueError('plane_frame 包含无效数值')
            if not (abs(float(np.dot(u, v))) < 0.02
                    and abs(float(np.dot(u, n))) < 0.02
                    and abs(float(np.dot(v, n))) < 0.02):
                raise ValueError(
                    f'plane_frame 正交/共面校验失败: '
                    f'|u·v|={abs(float(np.dot(u, v))):.4f} '
                    f'|u·n|={abs(float(np.dot(u, n))):.4f} '
                    f'|v·n|={abs(float(np.dot(v, n))):.4f}'
                )
            print(f'[PCFD][PhotoMatch] plane_frame ok '
                  f'u=({u[0]:.2f},{u[1]:.2f},{u[2]:.2f}) '
                  f'v=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})', flush=True)
            return o, u, v, pm
        except Exception as exc:
            print(f'[PCFD][PhotoMatch] plane_frame invalid: {exc}', flush=True)
            return None

    # ==================================================================
    # 逐模式叠加（四组统一：热力图 warp）
    # ==================================================================
    @staticmethod
    def _generate_matched_overlay(mode, facade_no, photo_bgr, match_result,
                                   facade_data, quality, results_dir, dataset):
        """四组模式统一路径：透明热力图 warp 到原始照片。"""
        from services.heatmap_spec import heatmap_spec
        from services.heatmap_renderer import FacadeHeatmapTripletRenderer

        spec = heatmap_spec(mode)
        method = spec.get('method', 'ruler')
        metric = spec.get('metric', 'flatness')

        comparison = quality.get('quality_comparison', {}) or {}
        methods_data = comparison.get('methods', {}) or {}
        method_dict = methods_data.get(method, {}) or {}
        metric_data = method_dict.get(metric, {}) or {}
        windows = metric_data.get('windows', []) or []
        if not windows:
            print(f'[PCFD][PhotoMatch] mode={mode} no_windows', flush=True)
            return None

        # 以完整 quality 为基底，方法级字段独立覆盖，共享字段
        renderer_quality = dict(quality)
        renderer_quality['defect_samples'] = method_dict.get(
            'defect_samples', {}) or {}
        renderer_quality['__global_indices'] = quality.get(
            '__global_indices', [])
        renderer_quality['__defect_index_space'] = quality.get(
            '__defect_index_space', 'raw_global_rows')
        if isinstance(method_dict.get('parameters'), dict):
            renderer_quality['parameters'] = method_dict['parameters']
        for key in ('projection_origin', 'projection_u_axis',
                    'projection_v_axis'):
            if method_dict.get(key) is not None:
                renderer_quality[key] = method_dict[key]
        renderer_quality['quality_comparison'] = (
            quality.get('quality_comparison', {}) or {})
        renderer_quality['quality_domain'] = (
            quality.get('quality_domain', {}) or {})

        ds = renderer_quality.get('defect_samples') or {}
        ds_info = {
            k: (int(len(v.get('raw_ids', []))) if isinstance(v, dict) else 0)
            for k, v in ds.items()
        }
        print(f'[PCFD][PhotoMatch] mode={mode} '
              f'defect_samples_sizes={ds_info}', flush=True)

        renderer = FacadeHeatmapTripletRenderer()
        facade_points = np.asarray(dataset.processed_raw_points, dtype=np.float64)

        transparent = renderer.render_transparent_heatmap(
            mode=mode,
            points=facade_points,
            colors=None,
            windows=windows,
            plane_model=facade_data.get('plane_model'),
            quality=renderer_quality,
            pixel_size=0.01,
            return_metadata=True,
        )
        if transparent is None:
            print(f'[PCFD][PhotoMatch] mode={mode} transparent_heatmap_failed',
                  flush=True)
            return None

        heatmap_rgba = np.ascontiguousarray(transparent['image'])
        uv_bounds = tuple(float(v) for v in transparent['uv_bounds'])
        # shape 应为 (height, width, 4)：ndim 是 3，不是 4。
        if (heatmap_rgba.ndim != 3 or heatmap_rgba.shape[2] != 4
                or len(uv_bounds) != 4
                or not np.all(np.isfinite(uv_bounds))):
            print(f'[PCFD][PhotoMatch] mode={mode} invalid_heatmap_metadata '
                  f'shape={getattr(heatmap_rgba, "shape", None)} '
                  f'uv_bounds={uv_bounds}', flush=True)
            return None
        print(f'[PCFD][PhotoMatch] mode={mode} heatmap_shape={heatmap_rgba.shape} '
              f'uv_bounds={uv_bounds}', flush=True)

        # ---- 2. 将“热力图 UV 系”安全变换到“配准 UV 系”，再 warp 到照片 ----
        # 根因修复：estimate_photo_cloud_match() 输出的 homography_uv_to_photo
        # 属于 match 阶段的平面参考系；而 transparent 热力图使用 renderer 返回的
        # projection_*。两套 (origin, u, v) 在方法级数据中可能仅 u 轴反向，
        # 旧代码直接用 renderer 参考系重解 H，会把 u 反向解释为 UV 反向，
        # 最终造成照片叠加热力图左右镜像。这里必须保留配准 H，只做两个正交
        # 平面坐标系之间的 2D 刚体/镜像变换，不能对同一批 correspondences
        # 换一套坐标系重新估计。
        render_origin = transparent.get('projection_origin')
        render_u = transparent.get('projection_u_axis')
        render_v = transparent.get('projection_v_axis')
        H_match = getattr(match_result, 'homography_uv_to_photo', None)
        H = None
        if H_match is not None:
            H = np.asarray(H_match, dtype=np.float64).reshape(3, 3)
            if (render_origin is not None and render_u is not None
                    and render_v is not None):
                try:
                    match_frame = TwoDMatchingService._extract_plane_frame(
                        facade_data, quality)
                    render_frame = (np.asarray(render_origin, dtype=np.float64),
                                    np.asarray(render_u, dtype=np.float64),
                                    np.asarray(render_v, dtype=np.float64),
                                    np.asarray(facade_data.get('plane_model'),
                                               dtype=np.float64))
                    T_render_to_match = TwoDMatchingService._uv_frame_change_matrix(
                        src_frame=render_frame,
                        dst_frame=match_frame,
                    )
                    H = H @ T_render_to_match
                    print(f'[PCFD][PhotoMatch] mode={mode} '
                          f'H_frame_composed det_T={np.linalg.det(T_render_to_match):.3f}',
                          flush=True)
                except Exception as exc:
                    # 不静默退回未变换的 H_match：两套 UV 系一旦不一致，
                    # 退回旧 H 恰好会复现左右镜像。交由 K/R/t 逐点投影兜底。
                    print(f'[PCFD][PhotoMatch] mode={mode} '
                          f'H_frame_compose_failed error={exc}', flush=True)
                    H = None
        elif render_origin is not None and render_u is not None and render_v is not None:
            # 极少见的无配准 H 路径：保持旧的重估能力。
            try:
                from utils.photo_matching import _solve_plane_homography_robust
                H, _h_meta = _solve_plane_homography_robust(
                    correspondences=list(match_result.correspondences or []),
                    plane_frame=(render_origin, render_u, render_v,
                                 facade_data.get('plane_model')),
                    camera_matrix=np.asarray(match_result.camera_matrix,
                                             dtype=np.float64).reshape(3, 3),
                    rotation=np.asarray(match_result.rotation_matrix,
                                        dtype=np.float64).reshape(3, 3),
                    translation=np.asarray(match_result.translation_vector,
                                           dtype=np.float64).reshape(3),
                )
            except Exception as exc:
                print(f'[PCFD][PhotoMatch] mode={mode} '
                      f'H_rebuild_failed error={exc}', flush=True)
                H = None

        blended = None
        if H is not None and np.all(np.isfinite(H)):
            try:
                warped_rgba = TwoDMatchingService._warp_heatmap_rgba_to_photo(
                    heatmap_rgba=heatmap_rgba,
                    uv_bounds=uv_bounds,
                    uv_to_photo=H,
                    photo_size=(photo_bgr.shape[1], photo_bgr.shape[0]),
                )
                blended = TwoDMatchingService._alpha_blend(
                    photo_bgr, warped_rgba, alpha=0.72)
                # ---- ★ 在照片上叠加 0.5m 物理网格线 ----
                try:
                    blended = TwoDMatchingService._draw_uv_grid_on_photo(
                        blended,
                        uv_bounds,
                        H,
                        grid_step_m=0.5,
                        line_color=(180, 190, 205),
                        thickness=1,
                    )
                    print(f'[PCFD][PhotoMatch] mode={mode} photo_grid_drawn',
                          flush=True)
                except Exception as exc:
                    print(f'[PCFD][PhotoMatch] mode={mode} '
                          f'photo_grid_draw_failed error={exc}', flush=True)
                print(f'[PCFD][PhotoMatch] mode={mode} basis=heatmap_warp '
                      f'H_inliers={match_result.homography_inlier_count} '
                      f'H_rmse={match_result.homography_rmse_px:.2f}px',
                      flush=True)
            except Exception as exc:
                print(f'[PCFD][PhotoMatch] mode={mode} '
                      f'heatmap_warp_failed error={exc}', flush=True)
                blended = None

        # ---- 3. 回退：K/R/t 逐点投影 ----
        if blended is None:
            blended = TwoDMatchingService._fallback_krt_overlay(
                photo_bgr, match_result, windows, spec, renderer_quality)
            print(f'[PCFD][PhotoMatch] mode={mode} basis=K_R_t_fallback',
                  flush=True)

        if blended is None:
            return None

        # ---- 4. 保存 ----
        prefix = f'facade_{facade_no:03d}_{mode}'
        root = results_dir / f'facade_{facade_no:03d}'
        root.mkdir(parents=True, exist_ok=True)
        overlay_path = root / f'{prefix}_photo_overlay.png'
        success, encoded = cv2.imencode('.png', blended)
        if success:
            overlay_path.write_bytes(encoded.tobytes())
            return overlay_path
        return None

    # ==================================================================
    # 平面 UV 参考系变换与热力图 warp（镜像修复）
    # ==================================================================
    @staticmethod
    def _uv_frame_change_matrix(src_frame, dst_frame) -> np.ndarray:
        """返回 src UV 坐标 → dst UV 坐标的 3×3 变换。

        src/dst frame 均为 (origin, u_axis, v_axis, plane_model)。只利用
        origin/u/v；不要求两个系的 handedness 相同，因此能正确表达方法级
        projection_* 中可能出现的 u 轴反向（图像左右镜像），而不是静默重估 H。
        """
        src_origin, src_u, src_v, _ = src_frame
        dst_origin, dst_u, dst_v, _ = dst_frame
        so = np.asarray(src_origin, dtype=np.float64).reshape(3)
        do = np.asarray(dst_origin, dtype=np.float64).reshape(3)
        su = np.asarray(src_u, dtype=np.float64).reshape(3)
        sv = np.asarray(src_v, dtype=np.float64).reshape(3)
        du = np.asarray(dst_u, dtype=np.float64).reshape(3)
        dv = np.asarray(dst_v, dtype=np.float64).reshape(3)

        su /= max(np.linalg.norm(su), 1e-12)
        sv /= max(np.linalg.norm(sv), 1e-12)
        du /= max(np.linalg.norm(du), 1e-12)
        dv /= max(np.linalg.norm(dv), 1e-12)

        src_basis = np.column_stack((su, sv))
        dst_basis = np.column_stack((du, dv))
        if (not np.all(np.isfinite(src_basis))
                or not np.all(np.isfinite(dst_basis))):
            raise ValueError('plane basis contains non-finite values')
        if (not np.allclose(src_basis.T @ src_basis, np.eye(2), atol=1e-3)
                or not np.allclose(dst_basis.T @ dst_basis, np.eye(2),
                                   atol=1e-3)):
            raise ValueError('plane basis is not orthonormal')

        T = np.eye(3, dtype=np.float64)
        T[:2, :2] = dst_basis.T @ src_basis
        T[:2, 2] = dst_basis.T @ (so - do)
        if not np.all(np.isfinite(T)):
            raise ValueError('uv frame transform contains non-finite values')
        return T

    @staticmethod
    def _warp_heatmap_rgba_to_photo(heatmap_rgba: np.ndarray,
                                    uv_bounds: tuple[float, float, float, float],
                                    uv_to_photo: np.ndarray,
                                    photo_size: tuple[int, int]) -> np.ndarray:
        """按 rasterize_facade 的真实像素→UV 约定，将透明热力图 warp 到照片。

        rasterize_facade() 中：
            col ↑  => u 增大
            row ↓  => v 减小（第 0 行是 v_hi，最后一行是 v_lo）
        因此不能只把 uv_bounds 当作“左上=(u0,v0)”交给通用 warp。这里显式
        使用四个像素角点建立映射，避免参考系 handedness 或 v 轴约定变化时
        再次发生镜像/上下翻转。
        """
        from utils.photo_matching import plane_uv_to_photo

        h, w = heatmap_rgba.shape[:2]
        if h < 2 or w < 2:
            raise ValueError(f'heatmap too small: {w}x{h}')
        u0, v0, u1, v1 = (float(v) for v in uv_bounds)
        if not (np.isfinite([u0, v0, u1, v1]).all() and u1 > u0 and v1 > v0):
            raise ValueError(f'invalid uv_bounds={uv_bounds}')

        # 源图角点：左上、右上、右下、左下。
        src_px = np.array([[0.0, 0.0],
                           [w - 1.0, 0.0],
                           [w - 1.0, h - 1.0],
                           [0.0, h - 1.0]], dtype=np.float32)

        # 与 rasterize_facade 的列/行公式严格对应（第 0 行为 v_hi）。
        uv_corners = np.array([[u0, v1],
                               [u1, v1],
                               [u1, v0],
                               [u0, v0]], dtype=np.float64)
        dst_px = np.asarray(
            plane_uv_to_photo(uv_corners,
                              np.asarray(uv_to_photo, dtype=np.float64).reshape(3, 3)),
            dtype=np.float32,
        )
        if not np.all(np.isfinite(dst_px)):
            raise ValueError('uv corners map to non-finite photo pixels')

        warp_matrix = cv2.getPerspectiveTransform(src_px, dst_px)
        warped = cv2.warpPerspective(
            heatmap_rgba,
            warp_matrix,
            (int(photo_size[0]), int(photo_size[1])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if warped.shape[2] == 4:
            # warp 后清除完全不可见像素，避免半透明杂边进入 alpha blend。
            warped[warped[:, :, 3] < 4] = 0
        return warped

    # ==================================================================
    # K/R/t 回退：逐点画圆（保留旧路径）
    # ==================================================================
    @staticmethod
    def _fallback_krt_overlay(photo_bgr, match_result, windows, spec, quality):
        """H 不可用时，回退到 K/R/t 逐点投影叠加。"""
        from services.heatmap_spec import (
            defect_colormap, bipolar_colormap, cold_defect_colormap,
        )
        from services.heatmap_renderer import FacadeHeatmapTripletRenderer

        method = spec.get('method', 'ruler')
        metric = spec.get('metric', 'flatness')

        result = None
        if method == 'ruler' and metric == 'flatness':
            result = FacadeHeatmapTripletRenderer._prepare_ruler_flatness_grid(
                windows, spec, quality)
        if result is None:
            result = FacadeHeatmapTripletRenderer._prepare_subgrid(
                windows, spec, quality)
        if result is None or len(result[0]) == 0:
            return None

        defect_points = np.ascontiguousarray(result[0], dtype=np.float64)
        values = np.ascontiguousarray(result[1], dtype=np.float32)

        # ★ 优先复用样本准备阶段已计算好的色带，避免方法间色带不一致
        prepared_colors = result[2] if len(result) > 2 else None
        if prepared_colors is not None:
            defect_colors_rgb = np.asarray(prepared_colors, dtype=np.float32)
        else:
            limit_mm = TwoDMatchingService._resolve_limit_mm(quality, spec)
            excess = np.maximum(np.abs(values) - limit_mm, 0.0)
            finite = excess[np.isfinite(excess)]
            scale = max(
                float(np.percentile(finite, 98)) if finite.size
                else limit_mm * 0.15,
                limit_mm * 0.15, 1e-6,
            )
            t = np.clip(excess / scale, 0.0, 1.0)
            if method == 'global_plane' and metric == 'flatness':
                defect_colors_rgb = bipolar_colormap(values, limit_mm)
            elif method == 'ruler':
                # ★ 靠尺法统一使用单极冷色，与图例、点云叠加保持一致
                defect_colors_rgb = cold_defect_colormap(t)
            else:
                defect_colors_rgb = defect_colormap(t)

        defect_colors_bgr = (
            np.asarray(defect_colors_rgb)[:, ::-1] * 255.0
        ).clip(0, 255).astype(np.uint8)

        from utils.photo_matching import _overlay_points_on_rectified
        blended, _ = _overlay_points_on_rectified(
            photo_bgr,
            defect_points,
            values,
            np.asarray(match_result.rotation_matrix, dtype=np.float64),
            np.asarray(match_result.translation_vector, dtype=np.float64),
            np.asarray(match_result.camera_matrix, dtype=np.float64),
            point_radius=8,
            external_colors_bgr=defect_colors_bgr,
        )
        return blended

    # ==================================================================
    # 基础工具
    # ==================================================================
    @staticmethod
    def _alpha_blend(photo_bgr: np.ndarray, heatmap_rgba: np.ndarray,
                     alpha: float = 0.72) -> np.ndarray:
        """将透明热力图叠加到 BGR 照片上。"""
        if photo_bgr.shape[:2] != heatmap_rgba.shape[:2]:
            heatmap_rgba = cv2.resize(
                heatmap_rgba,
                (photo_bgr.shape[1], photo_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        photo = photo_bgr.astype(np.float32)
        rgb = heatmap_rgba[:, :, :3].astype(np.float32)
        mask = heatmap_rgba[:, :, 3:4].astype(np.float32) / 255.0
        weight = mask * float(np.clip(alpha, 0.0, 1.0))
        blended = np.clip(photo * (1.0 - weight) + rgb * weight, 0, 255)
        return blended.astype(np.uint8)

    @staticmethod
    def _draw_uv_grid_on_photo(
        image_bgr: np.ndarray,
        uv_bounds: tuple[float, float, float, float],
        uv_to_photo: np.ndarray,
        grid_step_m: float = 0.5,
        line_color: tuple[int, int, int] = (200, 210, 220),
        thickness: int = 1,
    ) -> np.ndarray:
        """在照片叠加图上绘制 UV 物理网格线。

        利用单应 H_uv_to_photo 将 0.5m 间隔的 UV 直线端点映射到照片像素，
        由于单应保持直线性，端点映射后直接用 cv2.line 绘制。
        """
        from utils.photo_matching import plane_uv_to_photo

        u0, v0, u1, v1 = map(float, uv_bounds)
        H = np.asarray(uv_to_photo, dtype=np.float64).reshape(3, 3)

        # 生成网格线端点（UV 空间）
        lines_uv = []
        # u 方向竖线（固定 u，v 变化）
        u_start = np.ceil(u0 / grid_step_m) * grid_step_m
        for u in np.arange(u_start, u1 + 1e-6, grid_step_m):
            lines_uv.append(((u, v0), (u, v1)))
        # v 方向横线（固定 v，u 变化）
        v_start = np.ceil(v0 / grid_step_m) * grid_step_m
        for v in np.arange(v_start, v1 + 1e-6, grid_step_m):
            lines_uv.append(((u0, v), (u1, v)))

        if not lines_uv:
            return image_bgr

        # 批量映射端点到照片像素
        endpoints = np.asarray(
            [pt for seg in lines_uv for pt in seg], dtype=np.float64
        ).reshape(-1, 2)
        try:
            pixels = plane_uv_to_photo(endpoints, H)
        except Exception:
            return image_bgr

        img = image_bgr.copy()
        h, w = img.shape[:2]
        for i, ((u_a, v_a), (u_b, v_b)) in enumerate(lines_uv):
            p0 = pixels[i * 2]
            p1 = pixels[i * 2 + 1]
            # 仅当线段至少部分在图像内时才绘制（简单裁剪判断）
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            # Cohen–Sutherland 风格边界框快速拒绝
            if (max(x0, x1) < 0 or min(x0, x1) >= w or
                    max(y0, y1) < 0 or min(y0, y1) >= h):
                continue
            cv2.line(
                img,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                line_color, thickness, cv2.LINE_AA,
            )
        return img

    @staticmethod
    def _resolve_limit_mm(quality, spec):
        profile = quality.get('profile_snapshot', {}) or {}
        return float(profile.get(
            spec['limit_key'],
            (quality.get('thresholds') or {}).get(
                spec['limit_key'],
                (quality.get('parameters') or {}).get(spec['limit_key'], 4.0),
            ),
        ))

    @staticmethod
    def _mapping_size_for_photo(photo_bgr: np.ndarray,
                               max_side: int = 1024,
                               min_side: int = 768) -> tuple[int, int]:
        """按照片宽高比推导点云映射图尺寸，保证与照片同方向、同比例。"""
        h, w = photo_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return int(max_side), int(min_side)
        ar = float(w) / float(h)
        if ar >= 1.0:                       # 横构图
            width = int(max_side)
            height = int(round(width / ar))
            height = max(int(min_side), height)
        else:                               # 竖构图
            height = int(max_side)
            width = int(round(height * ar))
            width = max(int(min_side), width)
        width -= (width % 2)
        height -= (height % 2)
        return width, height

    @staticmethod
    def _resolve_photo_path(project, facade_data, station_service):
        """从项目 cache/photos/ 按站点名解析照片路径；缺失时从资产自动缓存。"""
        project_uuid = getattr(project, 'project_id', None)
        if not project_uuid:
            return None

        dirs = Storage.ensure_project_dirs(project_uuid)
        photo_cache = dirs['cache'] / 'photos'
        photo_cache.mkdir(parents=True, exist_ok=True)

        station_id = facade_data.get('station_id')
        name = None
        try:
            for station in (station_service.list_stations() if station_service else []):
                if station_id is not None and int(station.id) == int(station_id):
                    name = station.display_name
                    break
        except Exception:
            pass

        safe_name = str(name or f'station_{station_id or "unknown"}')

        # 1) 按站点名匹配缓存
        for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            candidate = photo_cache / (safe_name + ext)
            if candidate.is_file():
                return candidate

        # 2) 缓存目录已有图片：优先匹配站点名子串，避免多站点时拿错照片
        cached_images = [
            p for p in photo_cache.iterdir()
            if p.is_file() and p.suffix.lower() in (
                '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        ]
        if cached_images:
            name_key = safe_name.strip().lower()
            if name_key:
                for candidate in cached_images:
                    if name_key in candidate.stem.strip().lower():
                        return candidate
            print(f'[PCFD][PhotoMatch] photo_cache no station match: '
                  f'station={safe_name} candidates='
                  f'{[p.name for p in cached_images]}', flush=True)
            return None

        # 3) 缓存为空：从项目 raw_image 资产自动复制到缓存
        try:
            from models.enums import FileKind
            from services.dal.file_repo import FileRepo
            assets = FileRepo.list_assets_by_kind(project_uuid, FileKind.raw_image)
            if not assets:
                return None
            import shutil
            # ★ 多站点：按站点名匹配资产原始文件名，并记录文件指纹
            name_key = safe_name.strip().lower()
            selected = None
            if name_key:
                for asset in assets:
                    if name_key in Path(str(asset.path)).stem.strip().lower():
                        selected = asset
                        break
            if selected is None and len(assets) == 1:
                selected = assets[0]
            if selected is None:
                print(f'[PCFD][PhotoMatch] raw_image no station match: '
                      f'station={safe_name} assets='
                      f'{[Path(str(a.path)).name for a in assets]}', flush=True)
                return None
            src = Path(selected.path)
            if not src.is_file():
                return None
            dst_name = safe_name + src.suffix.lower()
            dst = photo_cache / dst_name
            shutil.copy2(str(src), str(dst))
            if dst.is_file():
                try:
                    digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
                    print(f'[PCFD][PhotoMatch] photo_cache_fingerprint '
                          f'station={safe_name} sha256_16={digest}', flush=True)
                except Exception:
                    pass
                return dst
        except Exception as exc:
            print(f'[PCFD] photo_cache_copy_failed project={project_uuid} '
                  f'error={exc}', flush=True)

        return None

    @staticmethod
    def _resolve_cloud_name(facade_data, pointcloud_service):
        """解析当前立面所属点云名称。"""
        if pointcloud_service is None:
            return facade_data.get('cloud_name')
        try:
            resolved = pointcloud_service.resolve_processing_cloud()
            if resolved:
                return resolved
        except Exception:
            pass
        return facade_data.get('cloud_name')

    @staticmethod
    def _resolve_scan_pose(dataset, station_service, facade_data):
        """从数据集元数据或站点服务解析 transformToGlobal。"""
        meta = getattr(dataset, 'metadata', None) or {}
        if isinstance(meta, dict):
            scan_poses = meta.get('scan_poses')
            if isinstance(scan_poses, list) and scan_poses:
                pose = scan_poses[0]
                transform = (pose.get('transform_to_global')
                             or pose.get('transformToGlobal'))
                if transform is not None:
                    return transform
            transform = (meta.get('transform_to_global')
                         or meta.get('transformToGlobal'))
            if transform is not None:
                return transform
        return (facade_data.get('scan_pose')
                or facade_data.get('transform_to_global'))

    @staticmethod
    def _resolve_results_dir(project, facade_data, station_service):
        """遵循既有规范：results/站点名/。"""
        project_uuid = getattr(project, 'project_id', None)
        base = Path(Storage.ensure_project_dirs(project_uuid)['results'])
        station_id = facade_data.get('station_id')
        name = None
        try:
            for station in (station_service.list_stations() if station_service else []):
                if station_id is not None and int(station.id) == int(station_id):
                    name = station.display_name
                    break
        except Exception:
            pass
        safe = re.sub(r'[\\/*?:"<>|]', '_',
                      str(name or f'station_{station_id or "unknown"}'))
        safe = safe.strip(' ._')[:80] or 'unknown'
        return base / safe

    # ==================================================================
    # 下采样（含 intensity 同步）
    # ==================================================================
    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """Open3D voxel 下采样，纯内存操作不写入磁盘。"""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(points, dtype=np.float64))
            down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
            return np.asarray(down.points, dtype=np.float64)
        except Exception:
            return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _voxel_downsample_with_values(points: np.ndarray, values: np.ndarray,
                                       voxel_size: float = 0.1
                                       ) -> tuple[np.ndarray, np.ndarray | None]:
        """体素下采样时保留强度值：强度编码进颜色 → voxel → 解码回来。"""
        try:
            import open3d as o3d
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            vals = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(vals) != len(pts):
                return TwoDMatchingService._voxel_downsample(
                    points, voxel_size), None
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pts)
            finite = np.isfinite(vals)
            offset, scale = 0.0, 1.0
            gray = np.zeros(len(vals), dtype=np.float64)
            if finite.any():
                offset = float(np.min(vals[finite]))
                peak = float(np.max(vals[finite]))
                scale = peak - offset if peak > offset else 1.0
                gray[finite] = np.clip((vals[finite] - offset) / scale, 0.0, 1.0)
            cloud.colors = o3d.utility.Vector3dVector(
                np.repeat(gray[:, None], 3, axis=1))
            down = cloud.voxel_down_sample(float(voxel_size))
            pts_out = np.asarray(down.points, dtype=np.float64)
            if not down.has_colors():
                return pts_out, None
            recovered = np.asarray(down.colors)[:, 0] * scale + offset
            return pts_out, recovered.astype(np.float32)
        except Exception as exc:
            print(f'[PCFD][PhotoMatch] voxel_downsample_with_values failed: {exc}',
                  flush=True)
            return TwoDMatchingService._voxel_downsample(points, voxel_size), None

    @staticmethod
    def _update_export_context(quality: dict, exported: dict[str, str]):
        """将新生成的 photo_overlay 路径写回 quality 的 export_context，
        供 ReportDataService._images() 下次读取时直接使用。"""
        if not isinstance(quality, dict) or not exported:
            return
        context = quality.setdefault('__export_context', {})
        if not isinstance(context, dict):
            context = quality['__export_context'] = {}
        heatmaps = context.setdefault('heatmaps', {})
        if not isinstance(heatmaps, dict):
            heatmaps = context['heatmaps'] = {}

        for mode, path in exported.items():
            artifact = heatmaps.setdefault(mode, {})
            if not isinstance(artifact, dict):
                artifact = heatmaps[mode] = {}
            artifact['photo'] = path