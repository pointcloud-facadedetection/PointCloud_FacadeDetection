"""立面检测结果暂存：关闭项目后可直接加载最近一次检测结果。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.storage import Storage

SNAPSHOT_NAME = "facade_detection_latest.json"
SNAPSHOT_PREFIX = "facade_detection_"


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


def results_dir(project_uuid: str) -> Path:
    """立面检测结果固定写到本仓库 data/projects/<项目名>/results。"""
    project_name = Storage.project_root(project_uuid).name or project_uuid
    folder = (
        Storage.REPO_ROOT
        / "data"
        / "projects"
        / project_name
        / Storage.RESULTS_DIRNAME
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def snapshot_path(project_uuid: str) -> Path:
    return results_dir(project_uuid) / SNAPSHOT_NAME


def _legacy_snapshot_path(project_uuid: str) -> Optional[Path]:
    legacy_root = Storage.LEGACY_PROJECTS_ROOT
    if not legacy_root.exists():
        return None
    current = Storage.project_root(project_uuid)
    for candidate in (legacy_root / current.name, legacy_root / project_uuid):
        path = candidate / Storage.RESULTS_DIRNAME / SNAPSHOT_NAME
        if path.exists():
            return path
    try:
        for sub in legacy_root.iterdir():
            idx = Storage.pcfd_index_path(sub)
            if not idx.exists():
                continue
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
            except Exception:
                continue
            puid = str(((data or {}).get("project") or {}).get("uuid") or "")
            if puid == project_uuid:
                path = sub / Storage.RESULTS_DIRNAME / SNAPSHOT_NAME
                if path.exists():
                    return path
    except Exception:
        return None
    return None


def save_facade_snapshot(project_uuid: str, cloud_name: str, facades: list[dict]) -> Path:
    payload = {
        "schema": 1,
        "kind": "facade_detection",
        "cloud_name": cloud_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "facade_count": len(facades or []),
        "facades": _jsonable(facades or []),
    }
    text = json.dumps(payload, ensure_ascii=False)
    latest = snapshot_path(project_uuid)
    latest.write_text(text, encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = results_dir(project_uuid) / f"{SNAPSHOT_PREFIX}{stamp}.json"
    if archive.resolve() != latest.resolve():
        archive.write_text(text, encoding="utf-8")
    return latest


def load_facade_snapshot_file(path: str | Path) -> Optional[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("kind") not in (None, "facade_detection"):
        return None
    if not data.get("facades"):
        return None
    return data


def load_facade_snapshot(project_uuid: str) -> Optional[dict]:
    latest = load_facade_snapshot_file(snapshot_path(project_uuid))
    if latest:
        return latest
    legacy = _legacy_snapshot_path(project_uuid)
    if legacy is None:
        return None
    return load_facade_snapshot_file(legacy)


def list_facade_snapshots(project_uuid: str) -> list[Path]:
    folders = [results_dir(project_uuid)]
    legacy = _legacy_snapshot_path(project_uuid)
    if legacy is not None:
        folders.append(legacy.parent)
    files = []
    seen = set()
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.glob("facade_detection*.json"):
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)
