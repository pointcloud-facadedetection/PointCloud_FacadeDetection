from __future__ import annotations

from pathlib import Path
import uuid as _uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from config.storage import Storage
from db.connection import (
    init_index_db,
    upsert_index_project,
    index_session,
    list_index_projects,
    project_session,
)
from db.connection import IndexProject  # type: ignore
from models import Project, ResultScene, FileAsset
from models.enums import FileKind
from utils.file_lifecycle import ensure_project_folders, gc_project, validate_paths


class ProjectRepo:
    @staticmethod
    def create_project(name: str, org_unit: str | None = None, address: str | None = None,
                       remarks: str | None = None, building_floor: str | None = None) -> dict:
        project_uuid = str(_uuid.uuid4())
        # New: create dirs by pinyin abbreviation of name and initialize pcfd index
        try:
            dirs = Storage.ensure_project_dirs_by_name(project_uuid, name)
        except Exception:
            # Fallback to legacy UUID-named root
            dirs = ensure_project_folders(project_uuid)

        # Initialize pcfd index json EARLY (before DB) so that resolve_project_root can find the pinyin folder
        try:
            from datetime import datetime
            pcfd = {
                "schema": 1,
                "project": {
                    "uuid": project_uuid,
                    "name_cn": name,
                    "pinyin_abbr": dirs.get("dirname") or "",
                    "org_unit": org_unit,
                    "address": address,
                    "building_floor": building_floor,
                    "remarks": remarks,
                    "created_at": datetime.now().isoformat(timespec='seconds')
                },
                "paths": {
                    "root_dir": str(dirs["root"]),
                    "raw": Storage.RAW_DIRNAME + "/",
                    "cache": Storage.CACHE_DIRNAME + "/",
                    "results": Storage.RESULTS_DIRNAME + "/",
                    "reports": Storage.REPORTS_DIRNAME + "/",
                },
                "assets": {
                    "raw_pointclouds": [],
                    "raw_images": [],
                    "fls_folders": [],
                    "generated_ply": [],
                    "result_pdfs": []
                }
            }
            Storage.save_pcfd_index(dirs["root"], pcfd)
            # Prime in-memory uuid->root cache to avoid legacy UUID folder creation
            try:
                Storage._uuid_root_cache[project_uuid] = dirs["root"]  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        # Create per-project DB and seed Project + default scene
        with project_session(project_uuid) as s:
            proj = Project(
                uuid=project_uuid,
                name=name,
                org_unit=org_unit,
                address=address,
                remarks=remarks,
                building_floor=building_floor,
                root_dir=str(dirs["root"]),
            )
            s.add(proj)
            s.flush()
            scene = ResultScene(project_id=proj.id, name="Scene 1", is_active=True)
            s.add(scene)
            s.flush()

        # Update global index DB
        init_index_db()
        upsert_index_project(project_uuid, name, str(dirs["root"]))

        # pcfd index already initialized before DB creation

        return {"project_uuid": project_uuid, "name": name, "root_dir": str(dirs["root"]) }

    @staticmethod
    def update_project(project_uuid: str, **fields) -> Optional[dict]:
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return None
            for k, v in fields.items():
                if k not in {'id', 'uuid', 'root_dir', 'directory_path'} and hasattr(proj, k):
                    setattr(proj, k, v)
            proj.updated_at = datetime.now()
            s.flush()
            upsert_index_project(project_uuid, proj.name, proj.root_dir)
            try:
                index_data = Storage.load_pcfd_index(Path(proj.root_dir)) or {}
                project_data = index_data.setdefault("project", {})
                project_data.update({
                    "uuid": proj.uuid,
                    "name_cn": proj.name,
                    "org_unit": proj.org_unit,
                    "address": proj.address,
                    "building_floor": proj.building_floor,
                    "remarks": proj.remarks,
                })
                Storage.save_pcfd_index(Path(proj.root_dir), index_data)
            except Exception:
                pass
            return {
                "project_id": proj.uuid,
                "name": proj.name,
                "directory_path": proj.root_dir,
                "org_unit": proj.org_unit,
                "address": proj.address,
                "building_floor": proj.building_floor,
                "remarks": proj.remarks,
            }

    @staticmethod
    def delete_project(project_uuid: str, hard: bool = False) -> bool:
        from pathlib import Path
        import shutil

        # Resolve and validate the authoritative root before removing any row.
        init_index_db()
        root_dir = None
        with index_session() as isess:
            row = isess.execute(select(IndexProject).where(IndexProject.project_uuid == project_uuid)).scalar_one_or_none()
            root_dir = None
            if row:
                root_dir = row.root_dir
        root = Path(root_dir) if root_dir else Storage.project_root(project_uuid)
        if root.resolve().parent != Storage.PROJECTS_ROOT.resolve():
            raise ValueError(f'拒绝删除项目目录之外的路径: {root}')

        # Project deletion is intentionally always hard: the project database
        # and all generated/raw assets must not survive a UI delete operation.
        engine = None
        try:
            from db.connection import _project_engine
            engine = _project_engine(project_uuid)
            engine.dispose()
            _project_engine.cache_clear()
        except Exception:
            pass
        if root.exists():
            shutil.rmtree(root, ignore_errors=False)
        with index_session() as isess:
            row = isess.execute(select(IndexProject).where(IndexProject.project_uuid == project_uuid)).scalar_one_or_none()
            if row:
                isess.delete(row)
        Storage._uuid_root_cache.pop(project_uuid, None)
        return True

    @staticmethod
    def list_projects(limit: int | None = None) -> list[dict]:
        init_index_db()
        items = list_index_projects()
        data: list[dict] = []
        for r in items:
            # Prefer pcfd project name if present
            name = r.name
            try:
                idx = Storage.load_pcfd_index(Path(r.root_dir))
                pc = (idx or {}).get('project') or {}
                name = str(pc.get('name_cn') or r.name)
            except Exception:
                pass
            details = None
            try:
                with project_session(r.project_uuid) as s:
                    details = s.execute(
                        select(Project).where(Project.uuid == r.project_uuid)
                    ).scalar_one_or_none()
            except Exception:
                pass
            data.append({
                "project_id": r.project_uuid,
                "name": name,
                "directory_path": r.root_dir,
                "org_unit": getattr(details, "org_unit", None),
                "address": getattr(details, "address", None),
                "building_floor": getattr(details, "building_floor", None),
                "remarks": getattr(details, "remarks", None),
            })
        return data[:limit] if limit else data

    @staticmethod
    def load_and_activate(project_uuid: str, scene_id: Optional[int] = None) -> dict:
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one()
            # ensure at least one scene
            scenes = s.execute(select(ResultScene).where(ResultScene.project_id == proj.id)).scalars().all()
            if not scenes:
                sc = ResultScene(project_id=proj.id, name="Scene 1", is_active=True)
                s.add(sc)
                s.flush()
                active_id = sc.id
            else:
                if scene_id is None:
                    # find current active
                    active = next((sc for sc in scenes if sc.is_active), None)
                    if not active:
                        scenes[0].is_active = True
                        s.flush()
                        active = scenes[0]
                    active_id = active.id
                else:
                    # toggle flags
                    for sc in scenes:
                        sc.is_active = (sc.id == scene_id)
                    s.flush()
                    active_id = scene_id
        # GC cache according to active scene
        gc_project(project_uuid, active_scene_id=active_id)
        # Validate external raw paths (non-blocking)
        missing = validate_paths(project_uuid, kinds={FileKind.raw_pointcloud.value, FileKind.raw_image.value})
        return {"project_uuid": project_uuid, "active_scene_id": active_id, "missing_raw": missing}
