"""E57 point-cloud reader with a stable ``(points, colors)`` contract.

E57 is not consistently supported by the Open3D wheels used on Windows, so
the optional pye57 reader is preferred.  Keeping this adapter at the file
boundary lets the existing proxy, denoise, detection and report pipelines
remain format agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
import json
import time

import numpy as np
from utils.ply_writer import write_point_cloud_ply_atomic
from utils.logging_utils import log_event


def _field(raw, name: str):
    """Read a structured-array field while tolerating pye57 field aliases."""
    if isinstance(raw, dict):
        for candidate in (name, name.lower(), name.upper()):
            if candidate in raw:
                return np.asarray(raw[candidate])
        return None
    names = getattr(getattr(raw, "dtype", None), "names", None) or ()
    for candidate in (name, name.lower(), name.upper()):
        if candidate in names:
            return np.asarray(raw[candidate])
    return None


def _safe_int(value, default: int = 0) -> int:
    """Convert value to int, returning default if value is None or invalid."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _scan_count(reader):
    """Robustly extract scan count from pye57 reader with version fallbacks."""
    count = _safe_int(getattr(reader, 'scan_count', None), 0)
    if count:
        return count
    root = getattr(reader, 'root', None)
    if root is not None:
        count = _safe_int(getattr(root, 'header_count', None), 0)
        if count:
            return count
        # pye57 may expose data3D as a dict/list attribute or via __getitem__
        data3d = getattr(root, 'data3D', None)
        if data3d is not None:
            try:
                return len(data3d)
            except Exception:
                pass
        try:
            data3d = root['data3D']
            if data3d is not None:
                return len(data3d)
        except Exception:
            pass
    return 0


def _normalise_colour(values, count: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != count:
        return None
    arr = arr.astype(np.float32, copy=False)
    if np.nanmax(arr) > 1.0:
        arr /= 255.0
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)


def _extract_scan_poses(reader):
    """Extract scan poses (rotation_matrix + scan_position) from E57 headers.

    Returns a list of dicts conforming to the unified pose schema shared
    with the FLS converter.  rotation_matrix is 3x3 double; scan_position
    is [x, y, z] double.  Both are in the global frame (same as the
    transformed point coordinates returned by read_e57).

    If a scan has no pose, has_pose is False and rotation_matrix /
    scan_position are identity / zero so downstream never receives None.
    """
    if reader is None:
        return []
    poses = []
    scan_count = _scan_count(reader)
    for i in range(scan_count):
        header = reader.get_header(i)
        has_pose = bool(getattr(header, 'has_pose', lambda: False)())
        name = str(getattr(header, 'name', f'scan_{i}'))
        if has_pose:
            try:
                from pyquaternion import Quaternion
                rotation = getattr(header, 'rotation', None)
                translation = getattr(header, 'translation', None)
                if rotation is not None and translation is not None:
                    rot_mat = Quaternion(rotation).rotation_matrix
                    scan_pos = np.asarray(translation, dtype=float).flatten()[:3]
                    # Build full 4x4 transform_to_global
                    transform = np.eye(4, dtype=float)
                    transform[:3, :3] = rot_mat
                    transform[:3, 3] = scan_pos
                    poses.append({
                        'scan_index': i,
                        'scan_name': name,
                        'has_pose': True,
                        'rotation_matrix': rot_mat.tolist(),
                        'scan_position': scan_pos.tolist(),
                        'transform_to_global': transform.tolist(),
                    })
                    continue
            except Exception:
                pass
        # Fallback: no pose or extraction failed
        poses.append({
            'scan_index': i,
            'scan_name': name,
            'has_pose': False,
            'rotation_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'scan_position': [0.0, 0.0, 0.0],
            'transform_to_global': [[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]],
        })
    return poses


def read_e57(path: str):
    """Return finite E57 points and optional RGB colors as float32 arrays."""
    source = Path(path).expanduser().resolve()
    # 待后续启用：E57 原始元数据、强度、距离及姿态字段的完整反序列化。
    # 当前转换边界只启用已验证稳定的坐标/颜色，并通过 transform=True
    # 将 pose 固化到 PLY 坐标，避免恢复项目时再次接触 E57 元数据。
    try:
        import pye57
    except ImportError as exc:
        raise RuntimeError(
            '读取 E57 需要安装 pye57；请执行 pip install pye57') from exc

    try:
        reader = pye57.E57(str(source))
        chunks = []
        color_chunks = []
        have_color = True
        scan_count = _scan_count(reader)
        for scan_index in range(scan_count):
            # read_scan applies the scan pose when transform=True and exposes
            # RGB fields through the same dictionary contract.  This keeps
            # E57 coordinates in the global frame used by existing operators.
            raw = reader.read_scan(
                scan_index, colors=True, transform=True,
                ignore_missing_fields=True)
            x = _field(raw, 'cartesianX')
            y = _field(raw, 'cartesianY')
            z = _field(raw, 'cartesianZ')
            if x is None or y is None or z is None:
                continue
            points = np.column_stack((x, y, z)).astype(np.float32, copy=False)

            # 过滤 ASTM E57 cartesianInvalidState（1/2 表示无效坐标）
            invalid = _field(raw, 'cartesianInvalidState')
            if invalid is not None:
                try:
                    invalid_mask = np.asarray(invalid, dtype=np.int32) != 0
                    points = points[~invalid_mask]
                except Exception:
                    pass

            # 再过滤 NaN/Inf
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]

            red = _field(raw, 'colorRed')
            green = _field(raw, 'colorGreen')
            blue = _field(raw, 'colorBlue')
            if red is None or green is None or blue is None:
                have_color = False
            elif have_color:
                colors = np.column_stack((red, green, blue))
                # 颜色必须与过滤后点数量对齐
                if len(colors) == len(points) + len(finite):
                    colors = colors[finite]
                elif len(colors) == len(points):
                    pass
                else:
                    have_color = False
                    continue
                color_chunks.append(colors)
            chunks.append(points)
        if not chunks:
            raise ValueError('E57 中没有可读取的笛卡尔坐标点')
        points = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        colors = (_normalise_colour(np.concatenate(color_chunks), len(points))
                  if have_color and color_chunks and
                  sum(len(chunk) for chunk in color_chunks) == len(points)
                  else None)
        return points, colors
    except Exception:
        # Do not silently reinterpret malformed E57 data as an empty cloud.
        raise


def convert_e57_to_ply(source_path, cache_path, project_uuid=None, metadata=None):
    """一次性将 E57 标准化为 PLY；业务处理从此只接收 PLY。

    Sidecar JSON (cache_path.json) 现在额外包含 scan_poses 和
    scan_origins，与 FLS 转换器的元数据 schema 统一，供下游配准流程使用。
    """
    started = time.perf_counter()
    source_path, cache_path = Path(source_path).resolve(), Path(cache_path).resolve()
    log_event(project_uuid, 'e57.convert.begin', source_path=str(source_path),
              cache_path=str(cache_path))

    # 尝试创建 reader 提取位姿；失败时降级（兼容测试打桩或损坏文件）
    reader = None
    try:
        import pye57
        reader = pye57.E57(str(source_path))
    except Exception:
        pass

    try:
        points, colors = read_e57(str(source_path))
        scan_poses = _extract_scan_poses(reader)
        scan_origins = [p['scan_position'] for p in scan_poses if p['has_pose']]
        elapsed = time.perf_counter() - started
        log_event(project_uuid, 'e57.convert.read_done', source_path=str(source_path),
                  cache_path=str(cache_path), point_count=int(len(points)),
                  has_colors=bool(colors is not None), elapsed_seconds=elapsed)
        write_point_cloud_ply_atomic(cache_path, points, colors)
        info = dict(metadata or {}, cache_version=2, source_format='e57',
                    cache_format='ply', source_path=str(source_path),
                    cache_path=str(cache_path), point_count=int(len(points)),
                    has_colors=bool(colors is not None), pose_applied=True,
                    scan_count=len(scan_poses),
                    scan_poses=scan_poses,
                    scan_origins=scan_origins if scan_origins else None,
                    conversion_seconds=time.perf_counter() - started)
        cache_path.with_suffix('.json').write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        log_event(project_uuid, 'e57.convert.write_done', source_path=str(source_path),
                  cache_path=str(cache_path), point_count=int(len(points)),
                  scan_count=len(scan_poses),
                  elapsed_seconds=info['conversion_seconds'])
        return info
    except Exception as exc:
        log_event(project_uuid, 'e57.convert.failed', source_path=str(source_path),
                  cache_path=str(cache_path), error=repr(exc),
                  elapsed_seconds=time.perf_counter() - started)
        raise