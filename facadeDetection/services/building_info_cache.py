from __future__ import annotations

import json
import uuid

from config.storage import Storage
from services.building_info_service import (
    parse_building_info,
    validate_building_info,
)


class BuildingInfoCache:
    FILE_NAME = 'building_info_cache.json'
    BASIC_KEYS = ('org_unit', 'address', 'building_floor', 'remarks')
    REPORT_KEYS = (
        'construction_unit', 'construction_unit_executor', 'inspection_unit',
        'supervision_unit', 'client_unit', 'inspection_date', 'report_date',
    )

    @classmethod
    def _path(cls):
        return Storage.DATA_DIR / cls.FILE_NAME

    @staticmethod
    def _field(source, section, key):
        if isinstance(source, dict):
            nested = source.get(section)
            if isinstance(nested, dict):
                return nested.get(key)
            return source.get(key)
        return getattr(source, key, None)

    @classmethod
    def _normalise_project(cls, source):
        def date_value(value):
            return value.isoformat() if hasattr(value, 'isoformat') else value

        params = cls._field(source, None, 'inspection_params_json')
        if params is None and isinstance(source, dict) and 'building_info' in source:
            params = {'building_info': source['building_info']}
        return {
            'schema_version': 2,
            'building_info': parse_building_info(params),
            'basic_info': {
                key: cls._field(source, 'basic_info', key) or ''
                for key in cls.BASIC_KEYS
            },
            'report_info': {
                key: date_value(cls._field(source, 'report_info', key))
                for key in cls.REPORT_KEYS
            },
        }

    @classmethod
    def _read_payload(cls):
        try:
            payload = json.loads(cls._path().read_text(encoding='utf-8'))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @classmethod
    def load_project_defaults(cls):
        payload = cls._read_payload()
        return cls._normalise_project(payload) if payload is not None else None

    @classmethod
    def load(cls):
        defaults = cls.load_project_defaults()
        return defaults.get('building_info') if defaults is not None else None

    @classmethod
    def _write(cls, payload):
        path = cls._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def save_inspection_params(cls, params):
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (TypeError, ValueError, json.JSONDecodeError):
                return
        if not isinstance(params, dict) or not isinstance(
                params.get('building_info'), dict):
            return
        info = parse_building_info(params)
        valid, message = validate_building_info(info)
        if not valid:
            raise ValueError(message)
        payload = cls.load_project_defaults() or cls._normalise_project({})
        payload['building_info'] = info
        cls._write(payload)

    @classmethod
    def save_project(cls, project):
        payload = cls._normalise_project(project)
        if not cls._field(project, None, 'inspection_params_json'):
            previous = cls.load_project_defaults()
            if previous is not None:
                payload['building_info'] = previous['building_info']
        valid, message = validate_building_info(payload['building_info'])
        if not valid:
            raise ValueError(message)
        cls._write(payload)

    @classmethod
    def latest_project_defaults(cls):
        payload = cls._read_payload()
        if payload is not None and payload.get('schema_version', 1) >= 2:
            return cls._normalise_project(payload)
        from services.dal.project_repo import ProjectRepo
        try:
            project = ProjectRepo.latest_project()
        except Exception as error:
            print(f'[PCFD] building_info.fallback_failed: {error!r}', flush=True)
            project = None
        if project is None:
            return cls._normalise_project(payload) if payload is not None else None
        try:
            cls.save_project(project)
            return cls.load_project_defaults()
        except (OSError, ValueError, TypeError) as error:
            print(f'[PCFD] building_info.cache_failed: {error!r}', flush=True)
            return cls._normalise_project(project)

    @classmethod
    def latest(cls):
        defaults = cls.latest_project_defaults()
        return defaults.get('building_info') if defaults is not None else None
