from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _first_pose(payload):
    poses = (payload or {}).get('scan_poses') or []
    for pose in poses:
        if not pose.get('has_pose'):
            continue
        try:
            position = np.asarray(pose['scan_position'], dtype=float).reshape(3)
            if np.isfinite(position).all():
                return pose
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _camera(pose, target=None):
    if pose is None:
        return None
    camera = {
        'position': np.asarray(pose['scan_position'], dtype=float).reshape(3).tolist(),
        'focal_point': None,
        'up': [0.0, 0.0, 1.0],
        'source': 'e57_station_pose',
        'scan_index': int(pose.get('scan_index', 0)),
    }
    if target is not None:
        target = np.asarray(target, dtype=float).reshape(3)
        if np.isfinite(target).all():
            camera['focal_point'] = target.tolist()
    return camera


def initial_view_from_asset(asset):
    if Path(asset.path or '').suffix.lower() != '.e57':
        return None, None
    metadata = dict(asset.meta_json or {})
    cache_path = Path(metadata.get('cache_path') or '')
    if cache_path.is_file():
        sidecar = next((path for path in (
            cache_path.with_suffix('.ply.json'), cache_path.with_suffix('.json'))
            if path.is_file()), None)
        if sidecar is not None:
            try:
                metadata.update(json.loads(sidecar.read_text(encoding='utf-8')))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    poses = metadata.get('scan_poses') or []
    pose = _first_pose({'scan_poses': poses})
    payload = {'scan_poses': poses, 'source': 'e57'} if poses else None
    camera = _camera(pose)
    if camera is not None:
        try:
            low = np.asarray(metadata['bounds_min'], dtype=float).reshape(3)
            high = np.asarray(metadata['bounds_max'], dtype=float).reshape(3)
            camera['focal_point'] = ((low + high) / 2).tolist()
        except (KeyError, TypeError, ValueError):
            pass
    return payload, camera


def initial_view_from_station(station, points):
    pose = _first_pose(getattr(station, 'initial_pose_json', None))
    points = np.asarray(points)
    if pose is None or points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    target = (np.nanmin(points, axis=0) + np.nanmax(points, axis=0)) / 2
    return _camera(pose, target)
