"""自动照片-立面匹配：正射渲染 → 2D 匹配 → 3D 回溯 → PnP。"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .backProjector import backproject_matches_to_3d
from .facade_2D_matcher import match_photo_to_ortho, scale_photo_keypoints_to_original
from .ortho_renderer import render_facade_orthographic
from .facade_render import DEFAULT_ORTHO_POINT_FRACTION
from .photo_rectifier import PhotoPerspectiveRectifier
from .pose_estimator import estimate_camera_pose_ransac


class AutoMatchError(ValueError):
    """自动匹配失败，但可携带已完成的 2D/2D-3D 可视化数据。"""

    def __init__(self, message, match_visualization=None):
        super().__init__(message)
        self.match_visualization = match_visualization or {}


def _decode_ortho_rgb(ortho_result):
    raw = base64.b64decode(ortho_result['image_base64'])
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('正射图解码失败')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ortho_image_for_matching(ortho_result):
    """
    用于 2D 特征匹配的正射图输入（单张 RGB）。

    自动匹配固定使用 color 模式渲染，并取边缘+CLAHE 增强后的
    `_match_image_rgb`（与 UI 上「颜色」视图一致；深度/法线仅供预览）。
    """
    match_rgb = ortho_result.get('_match_image_rgb')
    if match_rgb is not None:
        img = np.asarray(match_rgb)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    return _decode_ortho_rgb(ortho_result)


def _build_match_visualization(
    match2d,
    kpts_photo,
    kpts_render,
    ortho,
    backproj=None,
    pnp_inlier_indices=None,
):
    """组装前端标注用的 2D 匹配与 2D-3D 对应点。"""
    kpts_photo = np.asarray(kpts_photo, dtype=np.float64).reshape(-1, 2)
    kpts_render = np.asarray(kpts_render, dtype=np.float64).reshape(-1, 2)
    raw_scores = match2d.get('scores')
    scores = np.asarray(
        [] if raw_scores is None else raw_scores, dtype=np.float64
    ).reshape(-1)
    point_count = int(match2d.get('point_match_count', len(kpts_photo)))
    match_kinds = list(match2d.get('match_kinds') or [])

    match2d_items = []
    for i in range(len(kpts_photo)):
        kind = (
            str(match_kinds[i])
            if i < len(match_kinds)
            else ('point' if i < point_count else 'line_endpoint')
        )
        match2d_items.append({
            'index': int(i + 1),
            'photo_px': [float(kpts_photo[i, 0]), float(kpts_photo[i, 1])],
            'ortho_px': [float(kpts_render[i, 0]), float(kpts_render[i, 1])],
            'kind': kind,
            'score': float(scores[i]) if i < len(scores) else None,
        })

    line_pairs = []
    line_start = point_count
    line_end = point_count + int(match2d.get('line_endpoint_count', 0))
    for j in range(line_start, line_end, 2):
        if j + 1 < line_end:
            line_pairs.append([int(j + 1), int(j + 2)])

    corr_items = []
    inlier_set = set(int(x) for x in (pnp_inlier_indices or []))
    if backproj is not None:
        for i in range(int(backproj['pair_count'])):
            detail = backproj['details'][i]
            corr_items.append({
                'index': int(i + 1),
                'photo_px': backproj['image_points'][i].tolist(),
                'ortho_px': detail.get('ortho_px'),
                'object_point': backproj['object_points'][i].tolist(),
                'lookup_method': detail.get('lookup_method'),
                'kind': (
                    str(match_kinds[i])
                    if i < len(match_kinds)
                    else ('point' if i < point_count else 'line_endpoint')
                ),
                'pnp_inlier': bool(i in inlier_set),
            })

    return {
        'match2d': match2d_items,
        'line_pairs': line_pairs,
        'correspondences_3d': corr_items,
        'ortho_width_px': int(ortho.get('width_px', 0)),
        'ortho_height_px': int(ortho.get('height_px', 0)),
        'point_match_count': point_count,
        'line_endpoint_count': int(match2d.get('line_endpoint_count', 0)),
    }


def _coarse_guided_match_mask(photo_points, render_points, coarse_alignment, local_frame):
    """把粗匹配预测从全局正射像素转换到局部正射像素，筛除明显不一致候选。"""
    if not coarse_alignment:
        return None, None
    try:
        homography = np.asarray(
            coarse_alignment['homography_matrix'], dtype=np.float64
        ).reshape(3, 3)
        coarse_frame = coarse_alignment['ortho_frame']
        predicted_global = cv2.perspectiveTransform(
            np.asarray(photo_points, dtype=np.float32).reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        coarse_res = float(coarse_frame['resolution_m'])
        world_u = (
            float(coarse_frame['u_min']) + predicted_global[:, 0] * coarse_res
        )
        world_v = (
            float(coarse_frame['v_max']) - predicted_global[:, 1] * coarse_res
        )
        local_res = float(local_frame['resolution_m'])
        predicted_local = np.column_stack([
            (world_u - float(local_frame['u_min'])) / local_res,
            (float(local_frame['v_max']) - world_v) / local_res,
        ])
        errors = np.linalg.norm(
            predicted_local - np.asarray(render_points, dtype=np.float64),
            axis=1,
        )
        tolerance = max(
            80.0,
            0.08 * max(
                float(local_frame['width_px']),
                float(local_frame['height_px']),
            ),
        )
        return errors <= tolerance, {
            'guided_tolerance_px': float(tolerance),
            'guided_error_mean_px': float(np.mean(errors)),
        }
    except (KeyError, TypeError, ValueError, cv2.error):
        return None, None


def _prepare_photo_ortho_matches(
    photo_path,
    facade,
    points,
    colors=None,
    ply_path=None,
    resolution_m=0.012,
    point_fraction=DEFAULT_ORTHO_POINT_FRACTION,
    margin_left_ratio=None,
    margin_right_ratio=None,
    margin_bottom_ratio=None,
    margin_top_ratio=None,
    resize_max=1024,
    min_match_confidence=0.1,
    min_correspondences=6,
    rectified_photo_path=None,
    rectify_h_inv=None,
    precomputed_match2d=None,
    homography_filter=False,
    coarse_alignment=None,
):
    """生成正射图，并获得统一到原始照片/原始正射图像素坐标的 2D 匹配。"""
    ortho = render_facade_orthographic(
        points,
        facade,
        colors=colors,
        ply_path=ply_path,
        resolution_m=float(resolution_m),
        point_fraction=float(point_fraction),
        margin_left_ratio=margin_left_ratio,
        margin_right_ratio=margin_right_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
        margin_top_ratio=margin_top_ratio,
        include_point_mappings=True,
    )

    using_rectified = bool(rectified_photo_path)
    if precomputed_match2d is not None:
        match2d = {
            'kpts_photo': np.asarray(
                precomputed_match2d.get('kpts_photo') or [], dtype=np.float32
            ).reshape(-1, 2),
            'kpts_render': np.asarray(
                precomputed_match2d.get('kpts_render') or [], dtype=np.float32
            ).reshape(-1, 2),
            'scores': np.asarray(
                precomputed_match2d.get('scores') or [], dtype=np.float32
            ).reshape(-1),
            'match_kinds': list(precomputed_match2d.get('match_kinds') or []),
            'render_scale': 1.0,
            'coordinates_are_original': True,
            'device': precomputed_match2d.get('device'),
            'matcher_backend': precomputed_match2d.get('matcher_backend', 'precomputed'),
            'used_perspective_rectify': bool(
                precomputed_match2d.get('used_perspective_rectify', False)
            ),
        }
        if len(match2d['kpts_photo']) != len(match2d['kpts_render']):
            raise ValueError('预计算的照片与正射图匹配点数量不一致')
        match2d['match_count'] = int(len(match2d['kpts_photo']))
        match2d['raw_match_count'] = int(
            precomputed_match2d.get('raw_match_count', match2d['match_count'])
        )
        match2d['point_match_count'] = int(sum(
            kind == 'point' for kind in match2d['match_kinds']
        ))
        match2d['line_endpoint_count'] = (
            match2d['match_count'] - match2d['point_match_count']
        )
        kpts_photo_orig = match2d['kpts_photo']
        using_rectified = match2d['used_perspective_rectify']
    else:
        match_photo_path = str(rectified_photo_path or photo_path)
        h_inv = None
        if using_rectified:
            if rectify_h_inv is None:
                raise ValueError('使用矫正照片匹配时必须提供 rectify_h_inv')
            h_inv = np.asarray(rectify_h_inv, dtype=np.float64).reshape(3, 3)

        match2d = match_photo_to_ortho(
            match_photo_path,
            ortho_image_for_matching(ortho),
            resize_max=int(resize_max),
            min_confidence=(
                float(min_match_confidence)
                if min_match_confidence is not None else None
            ),
        )
        if match2d.get('coordinates_are_original'):
            kpts_photo_orig = np.asarray(match2d['kpts_photo'], dtype=np.float32)
        else:
            kpts_photo_orig, _photo_scale = scale_photo_keypoints_to_original(
                match2d['kpts_photo'],
                match_photo_path,
                resize_max=int(resize_max),
            )
        if using_rectified:
            kpts_photo_orig = PhotoPerspectiveRectifier.map_keypoints_back(
                kpts_photo_orig, h_inv
            )

    raw_match_count = int(match2d.get('raw_match_count', match2d['match_count']))
    if match2d['match_count'] < int(min_correspondences):
        backend = match2d.get('matcher_backend', 'lightglue')
        raise ValueError(
            f'2D 匹配点不足（{backend} {raw_match_count} 对，'
            f'有效 {match2d["match_count"]} 对），至少需要 {min_correspondences} 对'
        )

    homography_stats = None
    if homography_filter:
        render_points = np.asarray(match2d['kpts_render'], dtype=np.float32)
        photo_points_for_h = np.asarray(kpts_photo_orig, dtype=np.float32)
        candidate_indices = np.arange(len(render_points), dtype=np.int32)
        guidance_stats = None
        guided_mask, guidance_stats = _coarse_guided_match_mask(
            photo_points_for_h,
            render_points,
            coarse_alignment,
            ortho['mapping']['frame'],
        )
        if (
            guided_mask is not None
            and int(np.count_nonzero(guided_mask)) >= int(min_correspondences)
        ):
            candidate_indices = np.flatnonzero(guided_mask)

        threshold = 5.0 * max(
            max(float(ortho['width_px']), float(ortho['height_px']))
            / float(resize_max or 1024),
            1.0,
        )
        homography, inlier_mask = cv2.findHomography(
            photo_points_for_h[candidate_indices],
            render_points[candidate_indices],
            cv2.RANSAC,
            threshold,
        )
        if homography is None or inlier_mask is None:
            raise ValueError('照片与正射图的单应一致性检查失败')
        keep = np.zeros(len(render_points), dtype=bool)
        keep[candidate_indices] = inlier_mask.reshape(-1).astype(bool)
        original_kinds = list(match2d.get('match_kinds') or [])
        # 线段的两个端点作为一组保留，避免过滤后把不同线段的单个端点误连成一条线。
        line_indices = [
            i for i, kind in enumerate(original_kinds) if kind == 'line_endpoint'
        ]
        for pair_start in range(0, len(line_indices) - 1, 2):
            i0, i1 = line_indices[pair_start:pair_start + 2]
            pair_keep = bool(keep[i0] and keep[i1])
            keep[i0] = pair_keep
            keep[i1] = pair_keep
        inlier_count = int(np.count_nonzero(keep))
        if inlier_count < int(min_correspondences):
            raise ValueError(
                f'照片与正射图的几何一致内点不足（{inlier_count}/'
                f'{match2d["match_count"]}）'
            )
        kpts_photo_orig = np.asarray(kpts_photo_orig)[keep]
        match2d['kpts_photo'] = np.asarray(kpts_photo_orig, dtype=np.float32)
        match2d['kpts_render'] = render_points[keep]
        raw_scores = match2d.get('scores')
        scores = np.asarray(
            [] if raw_scores is None else raw_scores, dtype=np.float32
        )
        if len(scores) == len(keep):
            match2d['scores'] = scores[keep]
        kinds = original_kinds
        if len(kinds) == len(keep):
            match2d['match_kinds'] = [
                kind for kind, selected in zip(kinds, keep) if selected
            ]
        match2d['match_count'] = inlier_count
        match2d['point_match_count'] = int(sum(
            kind == 'point' for kind in match2d.get('match_kinds', [])
        ))
        match2d['line_endpoint_count'] = (
            inlier_count - match2d['point_match_count']
        )
        homography_stats = {
            'homography_matrix': homography.tolist(),
            'inlier_count': inlier_count,
            'total_count': int(len(keep)),
            'inlier_ratio': float(inlier_count / len(keep)),
            'ransac_threshold_px': float(threshold),
            'guided_candidate_count': int(len(candidate_indices)),
            **(guidance_stats or {}),
        }

    match2d['kpts_photo'] = np.asarray(kpts_photo_orig, dtype=np.float32)
    match2d['coordinates_are_original'] = True
    match2d['render_scale'] = 1.0
    return ortho, match2d, using_rectified, homography_stats


def match_photo_to_facade_2d(
    photo_path,
    facade,
    points,
    colors=None,
    ply_path=None,
    resolution_m=0.012,
    point_fraction=DEFAULT_ORTHO_POINT_FRACTION,
    margin_left_ratio=None,
    margin_right_ratio=None,
    margin_bottom_ratio=None,
    margin_top_ratio=None,
    resize_max=1024,
    min_match_confidence=0.1,
    min_correspondences=6,
    rectified_photo_path=None,
    rectify_h_inv=None,
    coarse_alignment=None,
):
    """第一阶段：只匹配照片与立面正射图，并返回绿色 2D 匹配点。"""
    ortho, match2d, using_rectified, homography_stats = _prepare_photo_ortho_matches(
        photo_path,
        facade,
        points,
        colors=colors,
        ply_path=ply_path,
        resolution_m=resolution_m,
        point_fraction=point_fraction,
        margin_left_ratio=margin_left_ratio,
        margin_right_ratio=margin_right_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
        margin_top_ratio=margin_top_ratio,
        resize_max=resize_max,
        min_match_confidence=min_match_confidence,
        min_correspondences=min_correspondences,
        rectified_photo_path=rectified_photo_path,
        rectify_h_inv=rectify_h_inv,
        homography_filter=True,
        coarse_alignment=coarse_alignment,
    )
    viz = _build_match_visualization(
        match2d,
        match2d['kpts_photo'],
        match2d['kpts_render'],
        ortho,
    )
    return {
        'matches_2d': {
            'kpts_photo': match2d['kpts_photo'].tolist(),
            'kpts_render': np.asarray(match2d['kpts_render']).tolist(),
            'scores': np.asarray(
                [] if match2d.get('scores') is None else match2d['scores']
            ).tolist(),
            'match_kinds': list(match2d.get('match_kinds') or []),
            'raw_match_count': int(match2d.get('raw_match_count', match2d['match_count'])),
            'matcher_backend': match2d.get('matcher_backend'),
            'device': match2d.get('device'),
            'used_perspective_rectify': bool(using_rectified),
        },
        'match_visualization': viz,
        'homography': homography_stats,
        'alignment': {
            'homography_matrix': homography_stats['homography_matrix'],
            'ortho_frame': ortho['mapping']['frame'],
        },
        'match_count': int(match2d['match_count']),
        'ortho_width_px': int(ortho['width_px']),
        'ortho_height_px': int(ortho['height_px']),
    }


def match_photo_to_facade(
    photo_path,
    facade,
    points,
    image_width,
    image_height,
    colors=None,
    ply_path=None,
    horizontal_fov_deg=60.0,
    resolution_m=0.012,
    margin_m=None,
    point_fraction=DEFAULT_ORTHO_POINT_FRACTION,
    margin_left_ratio=None,
    margin_right_ratio=None,
    margin_bottom_ratio=None,
    margin_top_ratio=None,
    resize_max=1024,
    min_match_confidence=0.1,
    min_correspondences=6,
    reprojection_error_px=5.0,
    pnp_confidence=0.999,
    camera_matrix=None,
    distortion_coefficients=None,
    rectified_photo_path=None,
    rectify_h_inv=None,
    precomputed_match2d=None,
):
    """
    自动 2D-3D 立面匹配全流程。

    步骤 1: 立面正射渲染 + index_map
    步骤 2: GlueStick 点线联合匹配（回退 LightGlue）照片↔正射图
    步骤 3: 正射像素回溯 3D 物理坐标
    步骤 4: PnP RANSAC 估计相机内外参
    """
    if facade is None:
        raise ValueError('未指定待检测立面')
    if not photo_path:
        raise ValueError('照片路径无效')

    plane = np.asarray(facade.get('plane_model', []), dtype=float)
    if plane.size != 4:
        raise ValueError('立面平面模型无效')

    inlier_raw = facade.get('inlier_indices')
    if inlier_raw is None or len(inlier_raw) < 3:
        raise ValueError('立面有效点数不足，无法自动匹配')

    points = np.asarray(points, dtype=float)

    ortho, match2d, using_rectified, _homography_stats = _prepare_photo_ortho_matches(
        photo_path,
        facade,
        points,
        colors=colors,
        ply_path=ply_path,
        resolution_m=resolution_m,
        point_fraction=point_fraction,
        margin_left_ratio=margin_left_ratio,
        margin_right_ratio=margin_right_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
        margin_top_ratio=margin_top_ratio,
        resize_max=resize_max,
        min_match_confidence=min_match_confidence,
        min_correspondences=min_correspondences,
        rectified_photo_path=rectified_photo_path,
        rectify_h_inv=rectify_h_inv,
        precomputed_match2d=precomputed_match2d,
    )
    render_scale = float(match2d.get('render_scale') or 1.0)
    raw_match_count = int(match2d.get('raw_match_count', match2d['match_count']))

    kpts_photo_orig = np.asarray(match2d['kpts_photo'], dtype=np.float32)

    backproj = backproject_matches_to_3d(
        kpts_photo_orig,
        match2d['kpts_render'],
        ortho['mapping'],
        points,
        render_scale=render_scale,
    )
    lookup_counts = backproj.get('lookup_method_counts') or {}
    raw_scores = match2d.get('scores')
    scores = np.asarray(
        [] if raw_scores is None else raw_scores, dtype=float
    ).reshape(-1)
    reliable_candidates = [
        i for i, detail in enumerate(backproj['details'])
        if detail.get('lookup_method') in ('index_map', 'neighborhood')
    ]
    # 同一个正射点只能提供一个独立 3D 约束；优先保留置信度更高的匹配。
    reliable_candidates.sort(
        key=lambda i: float(scores[i]) if i < len(scores) else 0.0,
        reverse=True,
    )
    reliable_indices = []
    seen_3d = set()
    for i in reliable_candidates:
        xyz = np.asarray(backproj['object_points'][i], dtype=float)
        key = tuple(np.round(xyz, decimals=4).tolist())
        if key in seen_3d:
            continue
        seen_3d.add(key)
        reliable_indices.append(i)
    reliable_indices.sort()
    reliable_count = len(reliable_indices)

    viz = _build_match_visualization(
        match2d,
        kpts_photo_orig,
        match2d['kpts_render'],
        ortho,
        backproj=backproj,
    )

    if reliable_count < int(min_correspondences):
        raise AutoMatchError(
            f'2D-3D 可靠对应点不足（index_map/neighborhood {reliable_count} 对，'
            f'几何回退 {lookup_counts.get("geometric", 0)} 对），'
            f'至少需要 {min_correspondences} 对',
            match_visualization=viz,
        )
    if backproj['pair_count'] < int(min_correspondences):
        raise AutoMatchError(
            f'2D-3D 对应点不足（{backproj["pair_count"]}），至少需要 {min_correspondences} 对',
            match_visualization=viz,
        )

    try:
        pose = estimate_camera_pose_ransac(
            object_points=backproj['object_points'][reliable_indices],
            image_points=backproj['image_points'][reliable_indices],
            image_width=float(image_width),
            image_height=float(image_height),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            horizontal_fov_deg=float(horizontal_fov_deg),
            reproj_error=float(reprojection_error_px) * max(
                max(float(image_width), float(image_height)) / float(resize_max or 1024),
                1.0,
            ),
            confidence=float(pnp_confidence),
        )
    except ValueError as exc:
        raise AutoMatchError(str(exc), match_visualization=viz) from exc

    filtered_inliers = pose.get('inlier_indices') or []
    inlier_indices = [
        int(reliable_indices[int(i)])
        for i in filtered_inliers
        if 0 <= int(i) < len(reliable_indices)
    ]
    pose['inlier_indices'] = inlier_indices
    viz = _build_match_visualization(
        match2d,
        kpts_photo_orig,
        match2d['kpts_render'],
        ortho,
        backproj=backproj,
        pnp_inlier_indices=inlier_indices,
    )

    return {
        **pose,
        'auto_match': {
            'match2d_count': int(match2d['match_count']),
            'match2d_raw_count': raw_match_count,
            'correspondence_count': int(backproj['pair_count']),
            'reliable_correspondence_count': reliable_count,
            'pnp_correspondence_count': reliable_count,
            'lookup_method_counts': lookup_counts,
            'render_scale': render_scale,
            'ortho_width_px': int(ortho['width_px']),
            'ortho_height_px': int(ortho['height_px']),
            'device': match2d.get('device'),
            'matcher_backend': match2d.get('matcher_backend', 'lightglue'),
            'point_match_count': int(match2d.get('point_match_count', match2d['match_count'])),
            'line_endpoint_count': int(match2d.get('line_endpoint_count', 0)),
            'used_perspective_rectify': using_rectified,
            'ortho_match_mode': 'color_enhanced',
            'ortho_resolution_m': float(resolution_m),
        },
        'match_visualization': viz,
        'correspondences': [
            {
                'image_point': backproj['image_points'][i].tolist(),
                'object_point': backproj['object_points'][i].tolist(),
                **backproj['details'][i],
            }
            for i in range(backproj['pair_count'])
        ],
    }


__all__ = ['AutoMatchError', 'match_photo_to_facade', 'ortho_image_for_matching']
