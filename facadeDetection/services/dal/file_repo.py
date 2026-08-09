from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from config.storage import Storage
from db.connection import project_session
from models import FileAsset, Project, ResultScene
from models.enums import FileKind, PersistPolicy
from utils.file_types import guess_file_kind


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FileRepo:
    @staticmethod
    def import_file(project_uuid: str, src_path: str, kind: FileKind, *, copy_into_project: bool = False) -> Optional[FileAsset]:
        """
        Default behavior: do NOT copy raw files, only record external absolute path.
        If copy_into_project=True, copy into project's RAW directory and store the new path.
        For cache artifacts (non-raw kinds), write into CACHE directory.
        """
        src = Path(src_path).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return None
            root = Path(proj.root_dir)
            # Resolve active scene by is_active flag; create a default if missing.
            active_scene = s.execute(
                select(ResultScene).where(ResultScene.project_id == proj.id, ResultScene.is_active == True)
            ).scalar_one_or_none()
            if active_scene is None:
                active_scene = ResultScene(project_id=proj.id, name="Scene 1", is_active=True)
                s.add(active_scene)
                s.flush()
            active_scene_id = active_scene.id

            if kind in (FileKind.raw_pointcloud, FileKind.raw_image):
                policy = PersistPolicy.PERSIST
                if copy_into_project:
                    dst_dir = root / Storage.RAW_DIRNAME
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / src.name
                    if dst.resolve() != src:
                        shutil.copy2(src, dst)
                    path_to_store = str(dst.resolve())
                    size = dst.stat().st_size
                    sha = _sha256(dst)
                else:
                    path_to_store = str(src)
                    size = src.stat().st_size
                    sha = _sha256(src)
            else:
                # cache artifact
                policy = PersistPolicy.CACHE
                dst_dir = root / Storage.CACHE_DIRNAME
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                if dst.resolve() != src:
                    shutil.copy2(src, dst)
                path_to_store = str(dst.resolve())
                size = dst.stat().st_size
                sha = _sha256(dst)

            asset = FileAsset(
                project_id=proj.id,
                scene_id=active_scene_id,
                kind=(kind.value if kind else guess_file_kind(str(src))),
                persist_policy=policy.value,
                path=path_to_store,
                original_name=src.name,
                ext=src.suffix.lower(),
                size_bytes=size,
                sha256=sha,
            )
            s.add(asset)
            s.flush()
            return asset

    @staticmethod
    def delete_file(project_uuid: str, file_id: int, delete_disk: bool = False) -> bool:
        with project_session(project_uuid) as s:
            asset = s.get(FileAsset, file_id)
            if not asset:
                return False
            if delete_disk:
                try:
                    Path(asset.path).unlink(missing_ok=True)
                except Exception:
                    pass
            asset.is_deleted = True
            s.flush()
            return True

    @staticmethod
    def get_latest_raw_pointcloud(project_uuid: str) -> Optional[FileAsset]:
        """Return the latest non-deleted raw point cloud asset for the active scene of the project."""
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return None
            active = s.execute(select(ResultScene).where(ResultScene.project_id == proj.id, ResultScene.is_active == True)).scalar_one_or_none()
            if active is None:
                return None
            # newest by id DESC
            q = select(FileAsset).where(
                FileAsset.project_id == proj.id,
                FileAsset.scene_id == active.id,
                FileAsset.is_deleted == False,
                FileAsset.kind == FileKind.raw_pointcloud.value,
            ).order_by(FileAsset.id.desc())
            return s.execute(q).scalar_one_or_none()
