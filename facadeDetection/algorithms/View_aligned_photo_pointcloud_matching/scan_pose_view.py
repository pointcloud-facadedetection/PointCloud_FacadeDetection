"""扫描仪位姿 JSON 解析与 3D 视口视角初始化。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_POSE_KEYS = (
    'transformToGlobal',
    'transform_to_global',
    'scan_origin',
    'scanOrigin',
    'scannerPosition',
    'camera_matrix',
    'rotation_matrix',
)


def validate_scan_pose_json(json_path: str | Path) -> dict:
    """校验扫描位姿 JSON 并返回解析摘要（上传扫描仪位姿）。"""
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'位姿文件不存在: {path}')
    if path.suffix.lower() != '.json':
        raise ValueError('请选择 .json 格式的扫描位姿文件')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'JSON 解析失败: {exc}') from exc
    if not _has_scan_pose_fields(data):
        raise ValueError(
            'JSON 中缺少扫描位姿字段（需含 transformToGlobal / scan_origin 等）'
        )
    pose_src = _first_pose_dict(data)
    origin = _extract_scan_origin(pose_src)
    up = _extract_scan_up(pose_src)
    transform = _extract_transform_matrix(pose_src, required=False)
    meta = {
        'json_path': str(path),
        'origin': origin.tolist(),
        'up': up.tolist(),
        'has_transform': transform is not None,
        'has_camera_matrix': _get_first_present(pose_src, ('camera_matrix',)) is not None,
    }
    if transform is not None:
        meta['transform_to_global'] = transform.tolist()
        meta['global_to_scanner'] = np.linalg.inv(transform).tolist()
    return meta


def _has_scan_pose_fields(data) -> bool:
    if not isinstance(data, dict):
        return False
    if any(key in data for key in _POSE_KEYS):
        return True
    scans = data.get('scans')
    return isinstance(scans, list) and any(_has_scan_pose_fields(item) for item in scans)


def _first_pose_dict(data: dict) -> dict:
    if _has_scan_pose_fields(data) and 'scans' not in data:
        return data
    for item in data.get('scans') or []:
        if isinstance(item, dict) and _has_scan_pose_fields(item):
            return item
    return data


def _get_first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _extract_transform_matrix(pose_src: dict, *, required: bool) -> np.ndarray | None:
    transform = _get_first_present(pose_src, ('transformToGlobal', 'transform_to_global'))
    if transform is None:
        if required:
            raise ValueError('JSON 中缺少 transformToGlobal')
        return None
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f'transformToGlobal 应为 4x4，实际元素数 {matrix.size}')
    return matrix.reshape(4, 4)


def _extract_scan_origin(pose_src: dict) -> np.ndarray:
    origin = _get_first_present(
        pose_src,
        ('scan_origin', 'scanOrigin', 'scannerPosition'),
    )
    transform = _extract_transform_matrix(pose_src, required=False)
    if transform is not None:
        origin = transform[:3, 3]
    if origin is None:
        raise ValueError('JSON 中缺少 transformToGlobal 或 scan_origin')
    return np.asarray(origin, dtype=np.float64).reshape(3)


def _extract_scan_up(pose_src: dict) -> np.ndarray:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    transform = _extract_transform_matrix(pose_src, required=False)
    if transform is not None:
        axis_up = transform[:3, 2]
        if np.linalg.norm(axis_up) > 1e-6:
            world_up = axis_up / (np.linalg.norm(axis_up) + 1e-12)
    rotation_value = _get_first_present(pose_src, ('rotation_matrix',))
    if rotation_value is not None:
        rotation = np.asarray(rotation_value, dtype=np.float64).reshape(3, 3)
        up = -rotation[1]
        if np.linalg.norm(up) > 1e-6:
            return up / (np.linalg.norm(up) + 1e-12)
    return world_up


def _extract_camera_pose(pose_src: dict) -> tuple[np.ndarray, np.ndarray] | None:
    camera_matrix = _get_first_present(pose_src, ('camera_matrix',))
    rotation_matrix = _get_first_present(pose_src, ('rotation_matrix',))
    if camera_matrix is None or rotation_matrix is None:
        return None
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    translation_value = _get_first_present(pose_src, ('translation_vector', 'tvec'))
    if translation_value is None:
        translation_value = [0, 0, 0]
    translation = np.asarray(translation_value, dtype=np.float64).reshape(3)
    origin = -rotation.T @ translation
    up = -rotation[1]
    return origin, up


def _fallback_forward_from_pose(pose_src: dict) -> np.ndarray:
    transform = _extract_transform_matrix(pose_src, required=False)
    if transform is not None:
        # 扫描仪是 360 度设备，JSON 往往没有唯一光轴；这里用局部 Y 轴反向
        # 作为无法从点云中心确定观察目标时的稳定兜底方向。
        forward = -transform[:3, 1]
        if np.linalg.norm(forward) > 1e-6:
            return forward / (np.linalg.norm(forward) + 1e-12)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _fallback_up_for_forward(forward: np.ndarray) -> np.ndarray:
    candidates = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    return min(candidates, key=lambda axis: abs(float(np.dot(axis, forward))))


def build_scan_viewport_camera(
    json_path: str | Path,
    lookat,
    *,
    image_size=(960, 720),
    horizontal_fov_deg=70.0,
) -> dict:
    """
    根据扫描位姿与观察目标，计算 3D 视口相机参数（初始化点云角度）。
    """
    data = json.loads(Path(json_path).expanduser().read_text(encoding='utf-8'))
    pose_src = _first_pose_dict(data)
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)

    camera_pose = _extract_camera_pose(pose_src)
    transform = _extract_transform_matrix(pose_src, required=False)
    if camera_pose is not None:
        origin, up = camera_pose
    else:
        origin = _extract_scan_origin(pose_src)
        up = _extract_scan_up(pose_src)

    forward = lookat - origin
    if np.linalg.norm(forward) < 1e-6:
        forward = _fallback_forward_from_pose(pose_src)
        lookat = origin + forward
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    up = up - forward * float(np.dot(forward, up))
    if np.linalg.norm(up) < 1e-6:
        up = _fallback_up_for_forward(forward)
        up = up - forward * float(np.dot(forward, up))
    up = up / (np.linalg.norm(up) + 1e-12)

    result = {
        'eye': origin.tolist(),
        'lookat': lookat.tolist(),
        'up': up.tolist(),
        'front': forward.tolist(),
        'origin': origin.tolist(),
        'json_path': str(json_path),
        'image_size': tuple(image_size),
        'horizontal_fov_deg': float(horizontal_fov_deg),
    }
    if transform is not None:
        result['transform_to_global'] = transform.tolist()
        result['global_to_scanner'] = np.linalg.inv(transform).tolist()
    return result


__all__ = [
    'validate_scan_pose_json',
    'build_scan_viewport_camera',
]
