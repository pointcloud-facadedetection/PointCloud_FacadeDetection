import json
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import services.inspection_parameters as parameters
from models import Base, Facade, Project, ResultScene
from services.inspection_profile import InspectionProfileService


@pytest.fixture(autouse=True)
def local_building_cache(monkeypatch, tmp_path):
    from config.storage import Storage
    monkeypatch.setattr(Storage, 'DATA_DIR', tmp_path / 'local-cache')


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


def test_latest_project_uses_most_recently_created_project(monkeypatch):
    from datetime import datetime
    from db.connection import IndexBase, IndexProject
    import services.dal.project_repo as repo
    engine = create_engine('sqlite:///:memory:')
    IndexBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def index_scope():
        session = Session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    monkeypatch.setattr(repo, 'index_session', index_scope)
    monkeypatch.setattr(repo, 'init_index_db', lambda: None)
    with index_scope() as session:
        session.add_all([
            IndexProject(project_uuid='old', name='旧', root_dir='',
                         created_at=datetime(2026, 1, 1)),
            IndexProject(project_uuid='new', name='新', root_dir='',
                         created_at=datetime(2026, 2, 1)),
        ])
    monkeypatch.setattr(repo.ProjectRepo, 'list_projects', staticmethod(lambda: [
        {'project_id': 'old'}, {'project_id': 'new'},
    ]))
    assert repo.ProjectRepo.latest_project()['project_id'] == 'new'


def test_new_project_dialog_inherits_latest_building_info(qapp, monkeypatch):
    from services.dal.project_repo import ProjectRepo
    from ui.dialogs.project_create_dialog import ProjectCreateDialog
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(lambda: {
        'inspection_params_json': json.dumps({
            'building_info': {
                'base_elevation_m': 12.5, 'base_floor_no': 3,
                'standard_floor_height_m': 4.35, 'total_floor_count': 18,
                'special_floors': [{'floor_no': 3, 'height_m': 5.0, 'label': '首层'}],
            },
        }),
    }))
    dialog = ProjectCreateDialog()
    assert dialog.spin_total_floor_count.value() == 18
    assert dialog.spin_standard_floor_height.value() == 4.35
    assert dialog.spin_base_elevation.value() == 12.5
    assert dialog.spin_base_floor_no.value() == 3
    assert dialog.special_floor_table.count() == 1
    assert dialog.cb_ruler_length.currentData() == 2.0
    dialog.close()


def test_local_cache_precedes_project_and_fallback_recovers_it(qapp, monkeypatch):
    from config.storage import Storage
    from services.building_info_cache import BuildingInfoCache
    from services.dal.project_repo import ProjectRepo
    from ui.dialogs.project_create_dialog import ProjectCreateDialog

    def building(floors, height):
        return {'building_info': {
            'base_elevation_m': 0.0, 'base_floor_no': 1,
            'standard_floor_height_m': height,
            'total_floor_count': floors, 'special_floors': [],
        }}

    BuildingInfoCache.save_inspection_params(building(21, 4.5))
    BuildingInfoCache.save_inspection_params({'standard_id': 'plaster_normal'})
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(
        lambda: (_ for _ in ()).throw(AssertionError('缓存存在时不应查询项目索引'))))
    dialog = ProjectCreateDialog()
    assert dialog.spin_total_floor_count.value() == 21
    assert dialog.spin_standard_floor_height.value() == 4.5
    dialog.close()

    BuildingInfoCache._path().unlink()
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(lambda: {
        'inspection_params_json': json.dumps(building(17, 3.5)),
    }))
    dialog = ProjectCreateDialog()
    assert dialog.spin_total_floor_count.value() == 17
    assert dialog.spin_standard_floor_height.value() == 3.5
    dialog.close()
    assert BuildingInfoCache.load()['total_floor_count'] == 17

    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(lambda: None))
    dialog = ProjectCreateDialog()
    assert dialog.spin_total_floor_count.value() == 17
    assert dialog.spin_standard_floor_height.value() == 3.5
    dialog.close()
    assert BuildingInfoCache._path().parent == Storage.DATA_DIR


def test_project_defaults_include_basic_and_report_but_not_identity(qapp, monkeypatch):
    from datetime import date
    from services.building_info_cache import BuildingInfoCache
    from services.dal.project_repo import ProjectRepo
    from services.project_overview.project_overview_service import ProjectCard
    from ui.dialogs.project_create_dialog import ProjectCreateDialog
    project = ProjectCard(
        project_id='p1', name='旧项目', directory_path='',
        org_unit='单位A', address='北京市海淀区中关村1号',
        building_floor='1号楼', remarks='备注A',
        construction_unit='建设单位A', construction_unit_executor='施工单位A',
        inspection_unit='检测单位A', supervision_unit='监理单位A',
        client_unit='委托单位A', report_no='RJ-OLD',
        inspection_date=date(2026, 3, 4), report_date=date(2026, 3, 5),
        inspection_params_json=json.dumps({'building_info': {
            'base_elevation_m': 1.5, 'base_floor_no': 1,
            'standard_floor_height_m': 4.2, 'total_floor_count': 22,
            'special_floors': [],
        }}),
        pointcloud_files=['not-inherited.e57'], photo_files=['not-inherited.jpg'],
        fls_directories=['not-inherited'],
    )
    BuildingInfoCache.save_project(project)
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(
        lambda: (_ for _ in ()).throw(AssertionError('不应查询项目索引'))))
    dialog = ProjectCreateDialog()
    assert dialog.edt_name.text() == ''
    assert dialog.edt_report_no.text() == ''
    assert dialog.edt_org.text() == '单位A'
    assert '中关村' in dialog.edt_address.text()
    assert dialog.edt_building.text() == '1号楼'
    assert dialog.edt_remarks.toPlainText() == '备注A'
    assert dialog.edt_construction_unit.text() == '建设单位A'
    assert dialog.edt_construction_unit_executor.text() == '施工单位A'
    assert dialog.edt_inspection_unit.text() == '检测单位A'
    assert dialog.edt_supervision_unit.text() == '监理单位A'
    assert dialog.edt_client_unit.text() == '委托单位A'
    assert dialog.date_inspection.date().toString('yyyy-MM-dd') == '2026-03-04'
    assert dialog.date_report.date().toString('yyyy-MM-dd') == '2026-03-05'
    assert dialog.spin_total_floor_count.value() == 22
    assert dialog.spin_standard_floor_height.value() == 4.2
    assert dialog.values()['pointcloud_files'] == []
    assert dialog.values()['photo_files'] == []
    assert dialog.values()['fls_directories'] == []
    dialog.close()


def test_legacy_cache_backfills_basic_and_report(qapp, monkeypatch):
    from datetime import date
    from services.building_info_cache import BuildingInfoCache
    from services.dal.project_repo import ProjectRepo
    from services.project_overview.project_overview_service import ProjectCard
    from ui.dialogs.project_create_dialog import ProjectCreateDialog
    path = BuildingInfoCache._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 1, 'building_info': {
        'base_elevation_m': 2.0, 'base_floor_no': 1,
        'standard_floor_height_m': 4.5, 'total_floor_count': 21,
        'special_floors': [],
    }}), encoding='utf-8')
    latest = ProjectCard(
        project_id='latest', name='最近项目', directory_path='',
        org_unit='最近单位', address='北京市海淀区中关村1号',
        building_floor='最近楼栋', remarks='最近备注',
        construction_unit='最近建设', construction_unit_executor='最近施工',
        inspection_unit='最近检测', supervision_unit='最近监理',
        client_unit='最近委托', report_no='RJ-OLD',
        inspection_date=date(2026, 4, 6), report_date=date(2026, 4, 7),
        inspection_params_json=json.dumps({'building_info': {
            'base_elevation_m': 2.0, 'base_floor_no': 1,
            'standard_floor_height_m': 4.5, 'total_floor_count': 21,
            'special_floors': [],
        }}),
    )
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(lambda: latest))
    dialog = ProjectCreateDialog()
    assert dialog.edt_org.text() == '最近单位'
    assert dialog.edt_construction_unit.text() == '最近建设'
    assert dialog.edt_construction_unit_executor.text() == '最近施工'
    assert dialog.spin_total_floor_count.value() == 21
    assert dialog.edt_name.text() == '' and dialog.edt_report_no.text() == ''
    dialog.close()
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['schema_version'] == 2
    assert saved['basic_info']['org_unit'] == '最近单位'
    assert saved['report_info']['construction_unit'] == '最近建设'
    monkeypatch.setattr(ProjectRepo, 'latest_project', staticmethod(lambda: None))
    dialog = ProjectCreateDialog()
    assert dialog.edt_org.text() == '最近单位'
    assert dialog.edt_construction_unit.text() == '最近建设'
    dialog.close()


def test_building_fields_have_required_marks(qapp):
    from PySide6.QtWidgets import QLabel
    from ui.dialogs.project_create_dialog import ProjectCreateDialog
    dialog = ProjectCreateDialog()
    page = dialog.tabs.widget(1)
    marks = [label for label in page.findChildren(QLabel)
             if label.property('uiRole') == 'requiredMark']
    assert len(marks) == 4
    dialog.close()


def test_operation_page_has_no_parameter_setting(qapp):
    from PySide6.QtWidgets import QFrame, QWidget
    from ui.pages.operation_page import OperationPageMixin

    class Harness(OperationPageMixin):
        pass

    harness = Harness()
    harness.right_dock = QFrame()
    panel = QWidget(harness.right_dock)
    panel.setObjectName('rightDockPanel')
    harness._init_right_panel_widgets()
    assert not hasattr(harness, 'btn_inspection_params')
    assert not hasattr(harness, 'lbl_inspection_params')
    assert not hasattr(harness, 'btn_use_default_inspection_params')


def test_building_info_is_shared_from_project_not_facade(project_db):
    Session, _saved = project_db
    service = parameters.InspectionParametersService
    service.save('project-1', {
        'ruler_length_m': 3.0,
        'building_info': {'total_floor_count': 18},
    }, 1)
    with Session() as session:
        facade = session.get(Facade, 1)
        assert 'building_info' not in json.loads(facade.inspection_params_json)
        facade.inspection_params_json = json.dumps({
            'ruler_length_m': 2.0, 'building_info': {'total_floor_count': 3},
        })
        session.commit()
        project = session.get(Project, 1)
        values, source = service.resolve(
            project.inspection_params_json, facade.inspection_params_json)
        assert source == 'facade'
        assert values['ruler_length_m'] == 2.0
        assert values['building_info']['total_floor_count'] == 18


def test_snapshot_mapping_supports_facade_overrides():
    profile = InspectionProfileService.from_snapshot({
        'standard_id': 'plaster_advanced',
        'ruler_length_m': 3.0,
        'step_longitudinal_m': 1.5,
        'flatness_limit_mm': 2.0,
    })
    assert (profile.measure_height_m, profile.scan_step_m,
            profile.flatness_limit_mm) == (3.0, 1.5, 2.0)


def test_facade_dialog_only_exposes_quality_and_project_dialog_keeps_building(qapp):
    from services.project_overview.project_overview_service import ProjectCard
    from ui.dialogs.inspection_parameters_dialog import InspectionParametersDialog
    project = ProjectCard(project_id='p', name='项目', directory_path='',
                          inspection_params_json=json.dumps({
                              'standard_id': 'tile_finish', 'ruler_length_m': 3.0,
                          }))
    dialog = InspectionParametersDialog(project, facade_no=2)
    visible = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())
               if dialog.tabs.isTabVisible(i)]
    assert visible == ['检测参数']
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == '检测参数'
    assert dialog.cb_ruler_length.currentData() == 3.0
    values = dialog.inspection_values()
    assert values['standard_id'] == 'tile_finish'
    dialog.close()

    default_dialog = InspectionParametersDialog(project)
    visible = [default_dialog.tabs.tabText(i)
               for i in range(default_dialog.tabs.count())
               if default_dialog.tabs.isTabVisible(i)]
    assert visible == ['楼宇信息', '检测参数']
    default_dialog.close()
