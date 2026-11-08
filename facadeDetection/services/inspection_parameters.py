from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from config.storage import Storage
from db.connection import project_session
from models import Facade, Project


class InspectionParametersService:
    @staticmethod
    def resolve(project_params, facade_params=None) -> tuple[dict, str]:
        def decode(raw):
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
            return dict(raw) if isinstance(raw, dict) else {}

        project_values = decode(project_params)
        facade_values = decode(facade_params)
        overrides = {key: value for key, value in facade_values.items()
                     if key != 'building_info'}
        if overrides:
            if 'building_info' in project_values:
                overrides['building_info'] = project_values['building_info']
            return overrides, 'facade'
        return project_values, 'default'

    @staticmethod
    def save(project_uuid: str, values: dict, facade_id: int | None = None) -> None:
        raw = json.dumps(dict(values), ensure_ascii=False)
        with project_session(project_uuid) as session:
            project = session.execute(
                select(Project).where(Project.uuid == project_uuid)
            ).scalar_one_or_none()
            if project is None:
                raise ValueError('项目不存在或已被删除。')
            project.inspection_params_json = raw
            if facade_id is not None:
                facade = session.get(Facade, int(facade_id))
                if facade is None or facade.project_id != project.id:
                    raise ValueError('立面不存在或不属于当前项目。')
                facade_values = {key: value for key, value in values.items()
                                 if key != 'building_info'}
                facade.inspection_params_json = (
                    json.dumps(facade_values, ensure_ascii=False)
                    if facade_values else None)

        root = Path(project.root_dir)
        index = Storage.load_pcfd_index(root) or {}
        index.setdefault('project', {})['inspection_params_json'] = raw
        Storage.save_pcfd_index(root, index)
        from services.building_info_cache import BuildingInfoCache
        try:
            BuildingInfoCache.save_inspection_params(raw)
        except (OSError, ValueError, TypeError) as error:
            print(f'[PCFD] building_info.cache_failed: {error!r}', flush=True)

    @staticmethod
    def use_project_default(project_uuid: str, facade_id: int) -> dict:
        with project_session(project_uuid) as session:
            project = session.execute(
                select(Project).where(Project.uuid == project_uuid)
            ).scalar_one_or_none()
            facade = session.get(Facade, int(facade_id))
            if project is None or facade is None or facade.project_id != project.id:
                raise ValueError('立面不存在或不属于当前项目。')
            facade.inspection_params_json = None
            values, _ = InspectionParametersService.resolve(
                project.inspection_params_json)
            return values
