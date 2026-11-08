from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace

from services.dal.project_repo import ProjectRepo


def test_project_list_is_ordered_by_creation_time_descending(monkeypatch):
    base = datetime(2026, 1, 1)
    rows = [
        SimpleNamespace(project_uuid='old', name='old', root_dir='old',
                        created_at=base, id=1),
        SimpleNamespace(project_uuid='new', name='new', root_dir='new',
                        created_at=base + timedelta(days=2), id=3),
        SimpleNamespace(project_uuid='middle', name='middle', root_dir='middle',
                        created_at=base + timedelta(days=1), id=2),
    ]
    monkeypatch.setattr('services.dal.project_repo.list_index_projects', lambda: rows)
    monkeypatch.setattr('services.dal.project_repo.init_index_db', lambda: None)
    monkeypatch.setattr('services.dal.project_repo.Storage.load_pcfd_index', lambda _path: {})

    @contextmanager
    def session(_uuid):
        yield SimpleNamespace(execute=lambda _query: SimpleNamespace(
            scalar_one_or_none=lambda: None))

    monkeypatch.setattr('services.dal.project_repo.project_session', session)
    monkeypatch.setattr(
        'services.dal.project_repo.FileRepo.list_assets_by_kind', lambda *_args: [])

    projects = ProjectRepo.list_projects(limit=2)
    assert [project['project_id'] for project in projects] == ['new', 'middle']
