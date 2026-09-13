from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select, or_, and_

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
    def get_asset(project_uuid: str, asset_id: int) -> Optional[FileAsset]:
        with project_session(project_uuid) as s:
            asset = s.get(FileAsset, int(asset_id))
            return asset if asset is not None and not asset.is_deleted else None

    @staticmethod
    def update_cache_metadata(project_uuid: str, asset_id: int, metadata: dict) -> None:
        """Persist conversion metadata after the cache artifact is atomically ready."""
        with project_session(project_uuid) as s:
            asset = s.get(FileAsset, int(asset_id))
            if asset is None or asset.is_deleted:
                raise ValueError(f'点云资产不存在: {asset_id}')
            asset.meta_json = dict(asset.meta_json or {}, **(metadata or {}))

    @staticmethod
    def list_assets(project_uuid: str, *, pointcloud_only: bool = False) -> list[FileAsset]:
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return []
            q = select(FileAsset).where(FileAsset.project_id == proj.id, FileAsset.is_deleted == False)
            if pointcloud_only:
                # Keep old projects readable: early versions stored PLY assets
                # with inconsistent kind values, while the extension remained
                # reliable.  This must match station synchronization.
                q = q.where(or_(
                    FileAsset.kind == FileKind.raw_pointcloud.value,
                    FileAsset.ext.ilike('.ply'),
                    FileAsset.path.ilike('%.ply'),
                    FileAsset.ext.ilike('.e57'),
                    FileAsset.path.ilike('%.e57'),
                ))
            return s.execute(q.order_by(FileAsset.id.asc())).scalars().all()

    @staticmethod
    def list_assets_by_kind(project_uuid: str, kind: FileKind) -> list[FileAsset]:
        """列出项目中指定原始资源；路径仍以绝对路径返回，便于编辑回显。"""
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return []
            return list(s.execute(select(FileAsset).where(
                FileAsset.project_id == proj.id,
                FileAsset.kind == kind.value,
                FileAsset.is_deleted == False,
            ).order_by(FileAsset.id.asc())).scalars().all())

    @staticmethod
    def validate_asset(asset: FileAsset) -> tuple[bool, str]:
        path = Path(asset.path)
        if not path.is_file():
            return False, "missing"
        stat = path.stat()
        if asset.size_bytes is not None and stat.st_size != asset.size_bytes:
            return False, "size_mismatch"
        # size 与 mtime 均未变时跳过全量哈希；mtime 变化才重新校验内容
        meta = asset.meta_json or {}
        if asset.sha256 and meta.get('mtime_ns') == stat.st_mtime_ns:
            return True, "ok"
        if asset.sha256 and _sha256(path) != asset.sha256:
            return False, "sha256_mismatch"
        asset.meta_json = dict(meta, mtime_ns=stat.st_mtime_ns)
        return True, "ok"

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

            # Import is idempotent within a project.  Prefer the canonical
            # source path, then use the content fingerprint so a renamed copy
            # is not registered as a second station.
            existing = s.execute(select(FileAsset).where(
                FileAsset.project_id == proj.id,
                FileAsset.is_deleted == False,
                FileAsset.kind == (kind.value if kind else guess_file_kind(str(src))),
                or_(FileAsset.path == path_to_store,
                    and_(FileAsset.sha256 == sha, FileAsset.size_bytes == size)),
            ).order_by(FileAsset.id.asc())).scalars().first()
            mtime_ns = Path(path_to_store).stat().st_mtime_ns
            if existing is not None:
                existing.path = path_to_store
                existing.size_bytes = size
                existing.sha256 = sha
                existing.original_name = src.name
                existing.ext = src.suffix.lower()
                existing.meta_json = dict(existing.meta_json or {}, mtime_ns=mtime_ns)
                return existing

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
                meta_json={'mtime_ns': mtime_ns},
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
