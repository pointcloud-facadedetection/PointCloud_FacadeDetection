from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import select
from contextlib import contextmanager
import msvcrt

from config.storage import Storage
from db.connection import project_session
from models import FileAsset, Project, ResultScene
from models.enums import PersistPolicy


def _retry_unlink(path: Path, retries: int = 3, delay: float = 0.2) -> None:
    for i in range(retries):
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            return
        except PermissionError:  # Windows sharing violation
            time.sleep(delay)
    # last try
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception:
        pass


def ensure_project_folders(project_uuid: str) -> dict[str, Path]:
    Storage.ensure_base_dirs()
    return Storage.ensure_project_dirs(project_uuid)


@contextmanager
def _gc_lock(project_root: Path):
    lock_path = project_root / ".gc.lock"
    f = open(lock_path, "a+b")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        f.close()


def gc_project(project_uuid: str, active_scene_id: Optional[int] = None, *, dry_run: bool = False) -> dict[str, int]:
    """
    删除除当前活动场景外的 CACHE 文件
    """
    removed_files = 0
    marked_records = 0
    proj_root = Storage.project_root(project_uuid)
    with _gc_lock(proj_root), project_session(project_uuid) as s:
        if active_scene_id is None:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            # Resolve active scene by is_active flag
            active = s.execute(
                select(ResultScene).where(ResultScene.project_id == proj.id, ResultScene.is_active == True)
            ).scalar_one_or_none() if proj else None
            active_scene_id = active.id if active else None
        q = s.execute(
            select(FileAsset).where(
                (FileAsset.persist_policy == PersistPolicy.CACHE.value) | (FileAsset.persist_policy == PersistPolicy.PERSIST.value)
            )
        ).scalars()
        for fa in q:
            p = Path(fa.path) if fa.path else None
            # Orphan mark: if path missing (absolute only), mark record deleted
            if not p or (p.is_absolute() and not p.exists()):
                fa.is_deleted = True
                marked_records += 1
                continue
            # If CACHE and not active -> delete file and mark
            if fa.persist_policy == PersistPolicy.CACHE.value and fa.scene_id not in (None, active_scene_id):
                if not dry_run:
                    _retry_unlink(p)
                    removed_files += 1
                fa.is_deleted = True
        s.flush()
    return {"removed_files": removed_files, "marked_records": marked_records}


def validate_paths(project_uuid: str, kinds: Optional[set[str]] = None) -> list[tuple[int, str]]:
    """返回处理文件 (file_id, missing_path) 列表。可选择按类型过滤。"""
    missing: list[tuple[int, str]] = []
    with project_session(project_uuid) as s:
        q = select(FileAsset)
        if kinds:
            q = q.where(FileAsset.kind.in_(list(kinds)))
        assets = s.execute(q).scalars().all()
        for a in assets:
            if not a.path:
                continue
            # relative paths are considered present inside packaged projects; skip check here
            if not Path(a.path).is_absolute():
                continue
            if not Path(a.path).exists():
                missing.append((a.id, a.path))
    return missing


def remove_records(project_uuid: str, file_ids: Iterable[int]) -> int:
    count = 0
    with project_session(project_uuid) as s:
        for fid in file_ids:
            obj = s.get(FileAsset, fid)
            if obj is None:
                continue
            obj.is_deleted = True
            count += 1
        s.flush()
    return count
