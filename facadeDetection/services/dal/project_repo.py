from __future__ import annotations

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
                       remarks: str | None = None) -> dict:
        project_uuid = str(_uuid.uuid4())
        dirs = ensure_project_folders(project_uuid)

        # Create per-project DB and seed Project + default scene
        with project_session(project_uuid) as s:
            proj = Project(
                uuid=project_uuid,
                name=name,
                org_unit=org_unit,
                address=address,
                remarks=remarks,
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

        return {"project_uuid": project_uuid, "name": name, "root_dir": str(dirs["root"]) }

    @staticmethod
    def update_project(project_uuid: str, **fields) -> Optional[dict]:
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not proj:
                return None
            for k, v in fields.items():
                if hasattr(proj, k) and v is not None:
                    setattr(proj, k, v)
            proj.updated_at = datetime.now()
            s.flush()
            upsert_index_project(project_uuid, proj.name, proj.root_dir)
            return {"project_uuid": proj.uuid, "name": proj.name, "root_dir": proj.root_dir}

    @staticmethod
    def delete_project(project_uuid: str, hard: bool = False) -> bool:
        from pathlib import Path
        import shutil

        # Remove index entry
        init_index_db()
        with index_session() as isess:
            row = isess.execute(select(IndexProject).where(IndexProject.project_uuid == project_uuid)).scalar_one_or_none()
            root_dir = None
            if row:
                root_dir = row.root_dir
                isess.delete(row)
                isess.flush()

        if hard:
            root = Path(root_dir) if root_dir else Storage.project_root(project_uuid)
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
        return True

    @staticmethod
    def list_projects(limit: int | None = None) -> list[dict]:
        init_index_db()
        items = list_index_projects()
        data = [
            {"project_uuid": r.project_uuid, "name": r.name, "root_dir": r.root_dir}
            for r in items
        ]
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
