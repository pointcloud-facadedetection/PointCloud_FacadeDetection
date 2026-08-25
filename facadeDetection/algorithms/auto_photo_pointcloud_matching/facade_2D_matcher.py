"""2D 现场照片 ↔ 立面正射图特征匹配（SuperPoint + LightGlue）。"""

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np
import torch

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch, rbd, read_image, resize_image
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LightGlue = SuperPoint = None
    numpy_image_to_torch = rbd = None
    LIGHTGLUE_AVAILABLE = False

    def read_image(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"照片不存在或无法读取: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def resize_image(image, size, fn='max'):
        image = np.asarray(image)
        h, w = image.shape[:2]
        edge = max(h, w) if fn == 'max' else min(h, w)
        scale = min(float(size) / max(float(edge), 1.0), 1.0)
        if scale >= 1.0:
            return image, np.array([1.0, 1.0], dtype=np.float32)
        resized = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, np.array([scale, scale], dtype=np.float32)


_MATCHER_CACHE = {}
_MATCHER_CACHE_LOCK = threading.Lock()
_MATCH_INFERENCE_LOCK = threading.Lock()


def scale_photo_keypoints_to_original(kpts_photo, photo_path, resize_max=1024):
    """将 LightGlue 在缩放照片上的关键点映射回原始像素坐标。"""
    kpts = np.asarray(kpts_photo, dtype=np.float64).reshape(-1, 2)
    if kpts.size == 0 or not resize_max:
        return kpts.astype(np.float32), 1.0

    image = read_image(photo_path)
    h, w = image.shape[:2]
    if max(h, w) <= float(resize_max):
        return kpts.astype(np.float32), 1.0

    _, scale = resize_image(image, int(resize_max), fn='max')
    scaled = kpts.copy()
    scaled[:, 0] /= float(scale[0])
    scaled[:, 1] /= float(scale[1])
    return scaled.astype(np.float32), float(scale[0])


def _as_rgb_uint8(image) -> np.ndarray:
    """统一为 HWC RGB uint8。"""
    img = np.asarray(image)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def letterbox_ortho_for_matching(render_rgb, min_width_ratio=0.35):
    """
    极窄正射图左右填黑边，避免按长边缩放到 1024 后宽度仅百余像素、特征极少。
    返回 padded 图像及 pad 偏移（供关键点映射回原始正射像素）。
    """
    img = _as_rgb_uint8(render_rgb)
    h, w = img.shape[:2]
    target_w = max(w, int(np.ceil(h * float(min_width_ratio))))
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    padded = cv2.copyMakeBorder(
        img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, {'pad_left': pad_left, 'pad_top': 0, 'orig_w': w, 'orig_h': h}


def unletterbox_render_keypoints(kpts_render, render_scale, letterbox):
    """将匹配器输出的正射关键点还原到原始正射图像素坐标。"""
    kpts = np.asarray(kpts_render, dtype=np.float64).reshape(-1, 2).copy()
    if kpts.size == 0:
        return kpts.astype(np.float32)
    scale = float(render_scale) if render_scale else 1.0
    if scale > 0:
        kpts /= scale
    kpts[:, 0] -= float((letterbox or {}).get('pad_left', 0))
    kpts[:, 1] -= float((letterbox or {}).get('pad_top', 0))
    return kpts.astype(np.float32)


def preprocess_for_superpoint(img_numpy: np.ndarray) -> np.ndarray:
    """CLAHE 增强局部对比度，拉齐跨模态图像表现。"""
    img = np.asarray(img_numpy)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)


class FacadeFeatureMatcher:
    """SuperPoint 提特征 + LightGlue 跨图匹配。"""

    def __init__(self, max_num_keypoints=2048, device=None):
        if not LIGHTGLUE_AVAILABLE:
            raise ImportError(
                'LightGlue 未安装，且 GlueStick 不可用；请安装 lightglue 后重试'
            )
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.extractor = SuperPoint(
            max_num_keypoints=max_num_keypoints,
            keypoint_threshold=0.0005,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.1,
        ).eval().to(self.device)

    def match(
        self,
        photo_path,
        render_img,
        resize_max=1024,
        min_confidence=None,
    ) -> dict:
        """
        匹配现场照片与立面正射图。

        参数:
            photo_path: 2D 照片路径
            render_img: 正射图 numpy（RGB/灰度/BGR uint8 或 float）
            resize_max: 推理时长边上限（像素）
            min_confidence: 可选，过滤 LightGlue 匹配置信度

        返回:
            dict:
              - kpts_photo: (N, 2) 照片像素坐标
              - kpts_render: (N, 2) 正射图像素坐标
              - scores: (N,) 匹配置信度
              - match_count: 匹配对数
              - device: 推理设备
        """
        photo_path = Path(photo_path)
        if not photo_path.is_file():
            raise FileNotFoundError(f"照片不存在: {photo_path}")

        render_rgb = preprocess_for_superpoint(_as_rgb_uint8(render_img))
        render_rgb, letterbox = letterbox_ortho_for_matching(render_rgb)
        preprocess = {'resize': int(resize_max)} if resize_max else {}

        photo_rgb = preprocess_for_superpoint(read_image(photo_path))
        image0 = numpy_image_to_torch(photo_rgb).to(self.device)
        image1 = numpy_image_to_torch(render_rgb).to(self.device)

        with torch.inference_mode():
            feats0 = self.extractor.extract(image0, **preprocess)
            feats1 = self.extractor.extract(image1, **preprocess)
            print(f"[DEBUG] 2D 照片提取特征点数: {len(feats0['keypoints'][0])}")
            print(f"[DEBUG] 3D 正射图提取特征点数: {len(feats1['keypoints'][0])}")
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
        match_indices = matches01.get('matches')
        scores_tensor = matches01.get('scores')
        raw_match_count = int(match_indices.shape[0]) if match_indices is not None else 0
        print(f"[DEBUG] LightGlue 原始匹配对数: {raw_match_count}")

        if match_indices is None or match_indices.numel() == 0:
            return {
                'kpts_photo': np.empty((0, 2), dtype=np.float32),
                'kpts_render': np.empty((0, 2), dtype=np.float32),
                'scores': np.empty((0,), dtype=np.float32),
                'match_count': 0,
                'raw_match_count': 0,
                'render_scale': 1.0,
                'coordinates_are_original': True,
                'match_kinds': [],
                'point_match_count': 0,
                'line_endpoint_count': 0,
                'device': str(self.device),
                'matcher_backend': 'lightglue',
            }

        kpts_photo = feats0['keypoints'][match_indices[:, 0]].cpu().numpy()
        kpts_render = feats1['keypoints'][match_indices[:, 1]].cpu().numpy()
        # LightGlue Extractor.extract 已把内部 resize 后的关键点恢复到输入图坐标。
        # 此处仅移除原始 padded 正射图的 letterbox 偏移，不能再次除 resize 比例。
        kpts_render = unletterbox_render_keypoints(kpts_render, 1.0, letterbox)
        scores = (
            scores_tensor.cpu().numpy()
            if scores_tensor is not None
            else np.ones(len(match_indices), dtype=np.float32)
        )

        if min_confidence is not None:
            keep = scores >= float(min_confidence)
            kpts_photo = kpts_photo[keep]
            kpts_render = kpts_render[keep]
            scores = scores[keep]

        return {
            'kpts_photo': kpts_photo.astype(np.float32),
            'kpts_render': kpts_render.astype(np.float32),
            'scores': scores.astype(np.float32),
            'match_count': int(len(kpts_photo)),
            'raw_match_count': raw_match_count,
            'render_scale': 1.0,
            'coordinates_are_original': True,
            'match_kinds': ['point'] * int(len(kpts_photo)),
            'point_match_count': int(len(kpts_photo)),
            'line_endpoint_count': 0,
            'device': str(self.device),
            'matcher_backend': 'lightglue',
        }


def _default_photo_ortho_matcher(max_num_keypoints=2048):
    """优先 GlueStick，不可用时回退 SuperPoint + LightGlue。"""
    cache_key = int(max_num_keypoints)
    with _MATCHER_CACHE_LOCK:
        cached = _MATCHER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from .facade_glurestick_2D_matcher import (
                GLUESTICK_AVAILABLE,
                FacadeGlueStickMatcher,
            )

            if GLUESTICK_AVAILABLE:
                engine = FacadeGlueStickMatcher(max_pts=max_num_keypoints)
                _MATCHER_CACHE[cache_key] = engine
                return engine
        except Exception as exc:
            print(f'[match_photo_to_ortho] GlueStick 不可用，回退 LightGlue: {exc}')
        engine = FacadeFeatureMatcher(max_num_keypoints=max_num_keypoints)
        _MATCHER_CACHE[cache_key] = engine
        return engine


def match_photo_to_ortho(
    photo_path,
    render_img,
    max_num_keypoints=2048,
    resize_max=1024,
    min_confidence=None,
    matcher=None,
) -> dict:
    """函数式入口：照片 ↔ 正射图 2D 匹配（默认 GlueStick）。"""
    engine = matcher or _default_photo_ortho_matcher(max_num_keypoints)
    with _MATCH_INFERENCE_LOCK:
        return engine.match(
            photo_path,
            render_img,
            resize_max=resize_max,
            min_confidence=min_confidence,
        )


def _perspective_map_xy(homography, points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return mapped


def sample_verify_matches(
    photo_path,
    render_img,
    sample_count=8,
    max_num_keypoints=2048,
    resize_max=1024,
    min_confidence=0.1,
    random_seed=None,
    matcher=None,
) -> dict:
    """
    随机抽取若干匹配点，并基于单应矩阵计算估计位置，供 UI 验证。

    每个 sample:
      - photo_actual / ortho_actual: LightGlue 实际匹配点（实心圆）
      - photo_estimated / ortho_estimated: 单应变换估计点（空心圆）
    """
    match = match_photo_to_ortho(
        photo_path,
        render_img,
        max_num_keypoints=max_num_keypoints,
        resize_max=resize_max,
        min_confidence=min_confidence,
        matcher=matcher,
    )

    kpts_photo = match['kpts_photo']
    kpts_render = match['kpts_render']
    scores = match['scores']
    total = len(kpts_photo)

    if total < 4:
        raise ValueError(f'有效匹配点不足（{total}），无法验证')

    homography, inlier_mask = cv2.findHomography(
        kpts_photo,
        kpts_render,
        cv2.RANSAC,
        5.0,
    )
    if homography is None:
        raise ValueError('单应矩阵估计失败，无法验证匹配')

    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_indices = np.flatnonzero(inlier_mask)
    if inlier_indices.size < 4:
        raise ValueError('RANSAC 内点不足，无法验证匹配')

    rng = np.random.default_rng(random_seed)
    pick_count = min(int(sample_count), int(inlier_indices.size))
    chosen = rng.choice(inlier_indices, size=pick_count, replace=False)

    inv_h = np.linalg.inv(homography)
    est_render_all = _perspective_map_xy(homography, kpts_photo)
    est_photo_all = _perspective_map_xy(inv_h, kpts_render)

    samples = []
    for idx in chosen:
        idx = int(idx)
        photo_actual = kpts_photo[idx]
        ortho_actual = kpts_render[idx]
        photo_estimated = est_photo_all[idx]
        ortho_estimated = est_render_all[idx]
        samples.append({
            'index': idx,
            'score': float(scores[idx]),
            'photo_actual': [float(photo_actual[0]), float(photo_actual[1])],
            'photo_estimated': [float(photo_estimated[0]), float(photo_estimated[1])],
            'ortho_actual': [float(ortho_actual[0]), float(ortho_actual[1])],
            'ortho_estimated': [float(ortho_estimated[0]), float(ortho_estimated[1])],
            'error_photo_px': float(np.linalg.norm(photo_actual - photo_estimated)),
            'error_ortho_px': float(np.linalg.norm(ortho_actual - ortho_estimated)),
        })

    mean_photo_err = float(np.mean([s['error_photo_px'] for s in samples]))
    mean_ortho_err = float(np.mean([s['error_ortho_px'] for s in samples]))

    return {
        'match_count': int(total),
        'inlier_count': int(inlier_indices.size),
        'sample_count': int(len(samples)),
        'samples': samples,
        'mean_error_photo_px': mean_photo_err,
        'mean_error_ortho_px': mean_ortho_err,
        'device': match['device'],
    }


__all__ = [
    'FacadeFeatureMatcher',
    'letterbox_ortho_for_matching',
    'match_photo_to_ortho',
    'sample_verify_matches',
    'scale_photo_keypoints_to_original',
    'unletterbox_render_keypoints',
]
