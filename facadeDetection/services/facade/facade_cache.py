"""立面检测结果的项目级 JSON 缓存。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.storage import Storage


SNAPSHOT_NAME = 'facade_detection_latest.json'


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def snapshot_path(project_uuid: str) -> Path:
    return Storage.ensure_project_dirs(project_uuid)['results'] / SNAPSHOT_NAME


def save_facade_snapshot(
    project_uuid: str,
    cloud_name: str,
    facades: list[dict],
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
    roi_key: dict | None = None,
) -> Path:
    """原子写入最近一次立面检测结果。"""
    target = snapshot_path(project_uuid)
    first_facade = facades[0] if facades else {}
    payload = {
        'schema': 1,
        'kind': 'facade_detection',
        'cloud_name': str(cloud_name),
        'dataset_id': first_facade.get('dataset_id') or dataset_id,
        'dataset_revision': (
            first_facade.get('dataset_revision') or dataset_revision),
        'roi_key': _jsonable(roi_key),
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'facade_count': len(facades or []),
        'facades': _jsonable(facades or []),
    }
    temporary = target.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding='utf-8',
    )
    temporary.replace(target)
    return target


def load_facade_snapshot(project_uuid: str) -> Optional[dict]:
    """读取最近一次有效缓存；空检测结果也视为有效快照。"""
    target = snapshot_path(project_uuid)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('kind') != 'facade_detection':
        return None
    if not isinstance(payload.get('facades'), list):
        return None
    return payload
