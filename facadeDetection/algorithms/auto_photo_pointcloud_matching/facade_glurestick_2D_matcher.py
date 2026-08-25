"""2D 现场照片 ↔ 立面正射图特征匹配（GlueStick 点线联合）。"""

from __future__ import annotations

# OpenCV Python 接口由二进制扩展动态导出。
# pylint: disable=no-member

from pathlib import Path

import cv2
import numpy as np
import torch

from .facade_2D_matcher import (
    _as_rgb_uint8,
    letterbox_ortho_for_matching,
    preprocess_for_superpoint,
    read_image,
    resize_image,
    unletterbox_render_keypoints,
)

try:
    from gluestick import numpy_image_to_torch
    from gluestick.models.two_view_pipeline import TwoViewPipeline

    GLUESTICK_AVAILABLE = True
except ImportError:
    GLUESTICK_AVAILABLE = False
    TwoViewPipeline = None
    numpy_image_to_torch = None


def _require_gluestick():
    if not GLUESTICK_AVAILABLE:
        raise ImportError(
            'GlueStick 未安装。Windows 可先执行: '
            'pip install pytlsd==0.0.2 && '
            'pip install git+https://github.com/cvg/GlueStick.git --no-deps'
        )


def _to_match_gray(img_input, *, from_path: bool = False) -> np.ndarray:
    """CLAHE 增强后转灰度 uint8。"""
    if from_path:
        rgb = preprocess_for_superpoint(read_image(str(img_input)))
    else:
        rgb = preprocess_for_superpoint(_as_rgb_uint8(img_input))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


class FacadeGlueStickMatcher:
    """GlueStick 点线联合匹配：SuperPoint 角点 + LSD 线段。"""

    def __init__(self, max_pts=2048, max_lines=500, device=None):
        _require_gluestick()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'[GlueStick] 使用计算设备: {self.device}')

        conf = {
            'name': 'two_view_pipeline',
            'use_lines': True,
            'extractor': {
                'name': 'wireframe',
                'sp_params': {
                    'max_num_keypoints': int(max_pts),
                    'detection_threshold': 0.0005,
                    'force_num_keypoints': False,
                },
                'wireframe_params': {
                    'merge_points': True,
                    'merge_line_endpoints': True,
                },
                'max_n_lines': int(max_lines),
            },
            'matcher': {
                'name': 'gluestick',
                'weights': 'outdoor',
                'filter_threshold': 0.1,
            },
        }
        self.pipeline = TwoViewPipeline(conf).to(self.device).eval()

    def match(
        self,
        photo_path,
        render_img,
        resize_max=1024,
        min_confidence=None,
    ) -> dict:
        photo_path = Path(photo_path)
        if not photo_path.is_file():
            raise FileNotFoundError(f'照片不存在: {photo_path}')

        photo_gray = _to_match_gray(photo_path, from_path=True)
        render_gray = _to_match_gray(render_img, from_path=False)
        render_rgb = preprocess_for_superpoint(_as_rgb_uint8(render_img))
        render_rgb, letterbox = letterbox_ortho_for_matching(render_rgb)
        render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)

        photo_resized, _ = resize_image(photo_gray, int(resize_max), fn='max')
        render_resized, scale_render = resize_image(
            render_gray, int(resize_max), fn='max'
        )
        render_scale = float(scale_render[0])

        image0 = numpy_image_to_torch(photo_resized).to(self.device)[None]
        image1 = numpy_image_to_torch(render_resized).to(self.device)[None]
        data = {'image0': image0, 'image1': image1}

        with torch.inference_mode():
            pred = self.pipeline(data)

        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches_pts = pred['matches0'][0].cpu().numpy()
        match_scores_pts = pred['match_scores0'][0].cpu().numpy()

        valid_pt_mask = matches_pts > -1
        matched_pts0 = kpts0[valid_pt_mask]
        matched_pts1 = kpts1[matches_pts[valid_pt_mask]]
        scores_pts = match_scores_pts[valid_pt_mask]

        lines0 = pred['lines0'][0].cpu().numpy()
        lines1 = pred['lines1'][0].cpu().numpy()
        line_matches = pred['line_matches0'][0].cpu().numpy()
        line_match_scores = pred['line_match_scores0'][0].cpu().numpy()

        matched_line_pts0 = []
        matched_line_pts1 = []
        scores_line = []

        valid_line_mask = line_matches > -1
        for idx0 in np.where(valid_line_mask)[0]:
            idx1 = int(line_matches[idx0])
            l0_start, l0_end = lines0[idx0, 0], lines0[idx0, 1]
            l1_start, l1_end = lines1[idx1, 0], lines1[idx1, 1]
            direction0 = l0_end - l0_start
            direction1 = l1_end - l1_start
            if float(np.dot(direction0, direction1)) < 0.0:
                l1_start, l1_end = l1_end, l1_start
            line_score = float(line_match_scores[idx0])
            matched_line_pts0.extend([l0_start, l0_end])
            matched_line_pts1.extend([l1_start, l1_end])
            scores_line.extend([line_score, line_score])

        point_match_count = int(len(matched_pts0))
        line_endpoint_count = int(len(matched_line_pts0))

        if line_endpoint_count > 0:
            matched_line_pts0 = np.asarray(matched_line_pts0, dtype=np.float64)
            matched_line_pts1 = np.asarray(matched_line_pts1, dtype=np.float64)
            scores_line = np.asarray(scores_line, dtype=np.float32)
            if point_match_count > 0:
                all_pts0 = np.vstack([matched_pts0, matched_line_pts0])
                all_pts1 = np.vstack([matched_pts1, matched_line_pts1])
                all_scores = np.concatenate([scores_pts, scores_line])
            else:
                all_pts0, all_pts1, all_scores = (
                    matched_line_pts0,
                    matched_line_pts1,
                    scores_line,
                )
        elif point_match_count > 0:
            all_pts0, all_pts1, all_scores = matched_pts0, matched_pts1, scores_pts
        else:
            all_pts0 = np.empty((0, 2), dtype=np.float32)
            all_pts1 = np.empty((0, 2), dtype=np.float32)
            all_scores = np.empty((0,), dtype=np.float32)

        raw_match_count = int(len(all_pts0))
        match_kinds = np.array(
            ['point'] * point_match_count
            + ['line_endpoint'] * line_endpoint_count,
            dtype=object,
        )
        print(
            f'[GlueStick] 点匹配: {point_match_count} 对 | '
            f'线段端点: {line_endpoint_count} 对 | 合计: {raw_match_count} 对'
        )

        if raw_match_count == 0:
            return {
                'kpts_photo': all_pts0.astype(np.float32),
                'kpts_render': all_pts1.astype(np.float32),
                'scores': all_scores.astype(np.float32),
                'match_count': 0,
                'raw_match_count': 0,
                'point_match_count': 0,
                'line_endpoint_count': 0,
                'render_scale': render_scale,
                'coordinates_are_original': False,
                'match_kinds': [],
                'device': str(self.device),
                'matcher_backend': 'gluestick',
            }

        if min_confidence is not None:
            keep = all_scores >= float(min_confidence)
            all_pts0 = all_pts0[keep]
            all_pts1 = all_pts1[keep]
            all_scores = all_scores[keep]
            match_kinds = match_kinds[keep]

        point_match_count = int(np.count_nonzero(match_kinds == 'point'))
        line_endpoint_count = int(np.count_nonzero(match_kinds == 'line_endpoint'))

        all_pts1 = unletterbox_render_keypoints(all_pts1, render_scale, letterbox)
        render_scale = 1.0

        return {
            'kpts_photo': all_pts0.astype(np.float32),
            'kpts_render': all_pts1.astype(np.float32),
            'scores': all_scores.astype(np.float32),
            'match_count': int(len(all_pts0)),
            'raw_match_count': raw_match_count,
            'point_match_count': point_match_count,
            'line_endpoint_count': line_endpoint_count,
            'render_scale': render_scale,
            'coordinates_are_original': False,
            'match_kinds': match_kinds.tolist(),
            'device': str(self.device),
            'matcher_backend': 'gluestick',
        }


def match_photo_to_ortho(
    photo_path,
    render_img,
    max_num_keypoints=2048,
    max_lines=500,
    resize_max=1024,
    min_confidence=None,
    matcher=None,
) -> dict:
    """函数式入口：照片 ↔ 正射图 2D 匹配（GlueStick）。"""
    engine = matcher or FacadeGlueStickMatcher(
        max_pts=max_num_keypoints,
        max_lines=max_lines,
    )
    return engine.match(
        photo_path,
        render_img,
        resize_max=resize_max,
        min_confidence=min_confidence,
    )


__all__ = [
    'GLUESTICK_AVAILABLE',
    'FacadeGlueStickMatcher',
    'match_photo_to_ortho',
]
