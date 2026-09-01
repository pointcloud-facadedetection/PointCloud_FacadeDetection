"""2D 现场照片读取与校验（上传 2D 照片）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PHOTO_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def validate_photo_path(photo_path: str | Path) -> Path:
    """校验照片路径与格式。"""
    path = Path(photo_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'照片不存在: {path}')
    if path.suffix.lower() not in PHOTO_SUFFIXES:
        raise ValueError(f'不支持的照片格式: {path.suffix}')
    return path


def read_bgr_image(photo_path: str | Path) -> np.ndarray:
    """读取 BGR 图像，兼容 Windows 下含中文/空格的路径。"""
    path = validate_photo_path(photo_path)
    try:
        with open(path, 'rb') as handle:
            data = np.frombuffer(handle.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
    except OSError:
        pass
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'无法读取照片: {path}')
    return bgr


def load_photo_info(photo_path: str | Path) -> dict:
    """读取照片并返回宽高等元信息。"""
    path = validate_photo_path(photo_path)
    bgr = read_bgr_image(path)
    height, width = bgr.shape[:2]
    return {
        'photo_path': str(path),
        'width': int(width),
        'height': int(height),
        'bgr': bgr,
    }


__all__ = [
    'PHOTO_SUFFIXES',
    'validate_photo_path',
    'read_bgr_image',
    'load_photo_info',
]
