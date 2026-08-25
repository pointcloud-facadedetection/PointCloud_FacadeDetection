"""立面检测结果暂存：关闭项目后可直接加载最近一次检测结果。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.storage import Storage

SNAPSHOT_NAME = "facade_detection_latest.json"


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
    dirs = Storage.ensure_project_dirs(project_uuid)
    return dirs["results"] / SNAPSHOT_NAME


def save_facade_snapshot(project_uuid: str, cloud_name: str, facades: list[dict]) -> Path:
    path = snapshot_path(project_uuid)
    payload = {
        "schema": 1,
        "kind": "facade_detection",
        "cloud_name": cloud_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "facade_count": len(facades or []),
        "facades": _jsonable(facades or []),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_facade_snapshot(project_uuid: str) -> Optional[dict]:
    path = snapshot_path(project_uuid)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("facades"):
        return None
    return data
