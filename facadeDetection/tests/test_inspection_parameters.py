import json
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import services.inspection_parameters as parameters
from models import Base, Facade, Project, ResultScene
from services.inspection_profile import InspectionProfileService


@pytest.fixture
def project_db(monkeypatch, tmp_path):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def session_scope(_project_uuid):
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(parameters, 'project_session', session_scope)
    monkeypatch.setattr(parameters.Storage, 'load_pcfd_index', lambda _root: {})
    saved = []
    monkeypatch.setattr(parameters.Storage, 'save_pcfd_index',
                        lambda root, data: saved.append(data))
    with session_scope('project-1') as session:
        project = Project(uuid='project-1', name='测试', root_dir=str(tmp_path),
                          inspection_params_json=json.dumps({'standard_id': 'plaster_normal'}))
        session.add(project)
        session.flush()
        session.add(ResultScene(project_id=project.id, name='Scene 1', is_active=True))
        session.flush()
        session.add_all([
            Facade(project_id=project.id, scene_id=1, label='Facade 1'),
            Facade(project_id=project.id, scene_id=1, label='Facade 2'),
        ])
    return Session, saved


def test_latest_settings_are_default_for_unconfigured_facades(project_db):
    Session, saved = project_db
    service = parameters.InspectionParametersService
    service.save('project-1', {'standard_id': 'plaster_normal', 'ruler_length_m': 3.0}, 1)
    with Session() as session:
        first, second = session.execute(select(Facade).order_by(Facade.id)).scalars().all()
        fallback, source = service.resolve(session.get(Project, 1).inspection_params_json,
                                           second.inspection_params_json)
        assert (fallback['ruler_length_m'], source) == (3.0, 'default')
        assert first.inspection_params_json

    service.save('project-1', {'standard_id': 'tile_finish', 'ruler_length_m': 2.5}, 2)
    with Session() as session:
        default = session.get(Project, 1).inspection_params_json
        first = session.get(Facade, 1)
        fallback, source = service.resolve(default, first.inspection_params_json)
        assert (fallback['ruler_length_m'], source) == (3.0, 'facade')
        assert service.resolve(default, None)[0]['ruler_length_m'] == 2.5
    assert saved[-1]['project']['inspection_params_json']

    assert service.use_project_default('project-1', 1)['ruler_length_m'] == 2.5
    with Session() as session:
        assert session.get(Facade, 1).inspection_params_json is None


def test_snapshot_mapping_supports_facade_overrides():
    profile = InspectionProfileService.from_snapshot({
        'standard_id': 'plaster_advanced',
        'ruler_length_m': 3.0,
        'step_longitudinal_m': 1.5,
        'flatness_limit_mm': 2.0,
    })
    assert (profile.measure_height_m, profile.scan_step_m,
            profile.flatness_limit_mm) == (3.0, 1.5, 2.0)


def test_operation_parameter_dialog_only_exposes_building_and_quality(qapp):
    from services.project_overview.project_overview_service import ProjectCard
    from ui.dialogs.inspection_parameters_dialog import InspectionParametersDialog
    project = ProjectCard(project_id='p', name='项目', directory_path='',
                          inspection_params_json=json.dumps({
                              'standard_id': 'tile_finish', 'ruler_length_m': 3.0,
                          }))
    dialog = InspectionParametersDialog(project, facade_no=2)
    visible = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())
               if dialog.tabs.isTabVisible(i)]
    assert visible == ['楼宇信息', '检测参数']
    assert dialog.cb_ruler_length.currentData() == 3.0
    values = dialog.inspection_values()
    assert values['standard_id'] == 'tile_finish'
    dialog.close()
