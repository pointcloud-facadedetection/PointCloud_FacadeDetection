import json
import queue
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import Base, FileAsset, PointCloudStation, Project
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.initial_camera_service import initial_view_from_asset
from utils.e57_reader import convert_e57_to_ply


class FakeHeader:
    has_pose = True
    name = '测站一'
    rotation = [1.0, 0.0, 0.0, 0.0]
    translation = [5.0, 6.0, 7.0]


class FakeE57:
    scan_count = 2

    def __init__(self, _path):
        pass

    def get_header(self, _index):
        return FakeHeader()

    def read_scan(self, index, **_kwargs):
        count = 4 + index
        return {
            'cartesianX': np.arange(count, dtype=float),
            'cartesianY': np.full(count, index + 1.0),
            'cartesianZ': np.full(count, 2.0),
            'colorRed': np.full(count, 128),
            'colorGreen': np.full(count, 128),
            'colorBlue': np.full(count, 128),
        }


def test_native_intensity_write_read_and_progress(monkeypatch, tmp_path):
    class E57WithIntensity(FakeE57):
        def read_scan(self, index, **kwargs):
            raw = super().read_scan(index, **kwargs)
            raw['intensity'] = np.arange(len(raw['cartesianX']),
                                         dtype=np.float32) + index
            return raw

    monkeypatch.setitem(sys.modules, 'pye57',
                        SimpleNamespace(E57=E57WithIntensity))
    progress = []
    cache = tmp_path / 'native.ply'
    info = convert_e57_to_ply(
        tmp_path / 'scan.e57', cache,
        progress_cb=lambda percent, text: progress.append(percent))
    from utils.ply_fast_reader import read_ply_features_fast
    _points, _colors, intensities = read_ply_features_fast(cache)
    assert info['has_intensity'] is True
    assert info['intensity_valid_ratio'] == 1.0
    np.testing.assert_array_equal(
        intensities, np.r_[np.arange(4), np.arange(5) + 1].astype(np.float32))
    assert progress[-1] == 100
    _pose, camera = initial_view_from_asset(SimpleNamespace(
        path='scan.e57', meta_json={'cache_path': str(cache)}))
    assert camera['position'] == [5.0, 6.0, 7.0]
    assert camera['focal_point'] == [2.0, 1.5, 2.0]


def test_conversion_progress_and_pose_sidecar(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'pye57',
                        SimpleNamespace(E57=FakeE57))
    progress = []
    info = convert_e57_to_ply(
        tmp_path / 'scan.e57', tmp_path / 'cache.ply',
        progress_cb=lambda percent, text: progress.append((percent, text)))
    assert info['point_count'] == 9
    assert info['scan_poses'][0]['has_pose'] is True
    assert info['scan_poses'][0]['scan_position'] == [5.0, 6.0, 7.0]
    assert info['bounds_min'] == [0.0, 1.0, 2.0]
    assert info['bounds_max'] == [4.0, 2.0, 2.0]
    assert progress[-1][0] == 100
    assert all(first[0] <= second[0] for first, second in zip(progress, progress[1:]))
    assert any('2/2' in text for _percent, text in progress)
    assert (tmp_path / 'cache.ply').stat().st_size > 0
    assert json.loads((tmp_path / 'cache.ply.json').read_text(encoding='utf-8'))['scan_poses']


def test_parent_forwards_subprocess_progress(monkeypatch, tmp_path):
    import services.e57_cache_service as service
    source = tmp_path / 'scan.e57'
    source.touch()

    class Process:
        exitcode = 0

        def __init__(self, target, args):
            self.target, self.args = target, args
            self.alive = False

        def start(self):
            self.target(*self.args)
            self.alive = False

        def is_alive(self):
            return self.alive

        def join(self):
            return None

    def worker(source_path, cache_path, project_uuid, metadata, output):
        from pathlib import Path
        Path(cache_path).write_bytes(b'cache')
        output.put(('progress', 40, '读取完成'))
        output.put(('ok', dict(metadata, cache_path=cache_path)))

    class Queue(queue.Queue):
        def close(self):
            return None

        def join_thread(self):
            return None

    monkeypatch.setattr(service.multiprocessing, 'get_context',
                        lambda _method: SimpleNamespace(Queue=Queue, Process=Process))
    monkeypatch.setattr(service, '_e57_convert_worker', worker)
    monkeypatch.setattr(service.Storage, 'ensure_project_dirs',
                        lambda _uuid: {'cache': tmp_path})
    updates = []
    monkeypatch.setattr(service.FileRepo, 'update_cache_metadata',
                        lambda *args: updates.append(args))
    asset = SimpleNamespace(id=1, path=str(source), sha256='sha', size_bytes=0,
                            meta_json={})
    progress = []
    result = service.ensure_e57_cache(
        'uuid', asset, progress_cb=lambda percent, text: progress.append(percent))
    assert result.is_file()
    assert 0 in progress and 40 in progress and progress[-1] == 100
    assert updates


def test_upload_adapter_forwards_conversion_and_index_progress(monkeypatch, tmp_path):
    import services.file_service as module
    source = tmp_path / 'scan.e57'
    source.touch()
    cache = tmp_path / 'cache.ply'
    asset = SimpleNamespace(id=1, path=str(source), original_name=source.name)
    monkeypatch.setattr(module.FileRepo, 'import_file', lambda **_kwargs: asset)

    def convert(_uuid, _asset, progress_cb=None):
        assert progress_cb is not None
        progress_cb(40, '已读取测站')
        cache.write_bytes(b'cache')
        return cache

    monkeypatch.setattr(module, 'ensure_e57_cache', convert)
    monkeypatch.setattr(module.Storage, 'resolve_project_root', lambda _uuid: tmp_path)
    monkeypatch.setattr(module.Storage, 'append_pcfd_asset_for_uuid',
                        lambda *_args: None)
    service = module.FileService(None, None, SimpleNamespace(pointcloud_service=None))
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    service._load_point_cloud_features = lambda _path: (points, None, None)
    progress = []
    prepared = service.prepare_upload(
        'uuid', str(source),
        progress_cb=lambda percent, text: progress.append((percent, text)))
    assert prepared.asset is asset
    assert progress[0][0] == 34
    assert progress[-1][0] == 100
    assert any(percent == 92 for percent, _text in progress)


def test_project_edit_can_defer_e57_to_background_activation(monkeypatch, tmp_path):
    import services.project_overview.project_overview_service as module
    source = tmp_path / 'new.e57'
    source.touch()
    service = module.ProjectOverviewService()
    monkeypatch.setattr(service, 'get_project_assets', lambda _uuid: {})
    monkeypatch.setattr(module.FileRepo, 'list_assets_by_kind', lambda *_args: [])
    monkeypatch.setattr(module.FileRepo, 'import_file',
                        lambda *_args, **_kwargs: SimpleNamespace(path=str(source)))
    monkeypatch.setattr(module.PointCloudStationRepo, 'sync_assets', lambda _uuid: None)
    from config.storage import Storage
    monkeypatch.setattr(Storage, 'project_root', lambda _uuid: tmp_path)
    monkeypatch.setattr(Storage, 'load_pcfd_index', lambda _root: {})
    monkeypatch.setattr(Storage, 'save_pcfd_index', lambda *_args: None)
    calls = []
    monkeypatch.setattr(module, 'ensure_e57_cache',
                        lambda *_args: calls.append('convert'))
    service.update_project_assets('uuid', [], [str(source)], [], defer_e57=True)
    assert calls == []
    service.update_project_assets('uuid', [], [str(source)], [])
    assert calls == ['convert']


@contextmanager
def memory_session(Session, _uuid):
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_station_pose_and_camera_survive_db_reload(monkeypatch, tmp_path):
    import services.dal.pointcloud_station_repo as repo_module
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(repo_module, 'project_session',
                        lambda uuid: memory_session(Session, uuid))
    monkeypatch.setattr(repo_module.FileRepo, 'validate_asset', lambda _asset: (True, 'ok'))
    cache = tmp_path / 'cache.ply'
    cache.write_bytes(b'cached')
    cache.with_suffix('.json').write_text(json.dumps({
        'scan_poses': [{'has_pose': True, 'scan_index': 0,
                        'scan_position': [5, 6, 7], 'rotation_matrix': np.eye(3).tolist()}],
        'bounds_min': [0, 0, 0], 'bounds_max': [10, 20, 30],
    }), encoding='utf-8')
    with memory_session(Session, 'uuid') as session:
        session.add(Project(uuid='uuid', name='E57', root_dir=str(tmp_path)))
        session.flush()
        session.add(FileAsset(project_id=1, kind='raw_pointcloud', path='scan.e57',
                              original_name='scan.e57', sha256='sha', size_bytes=1,
                              meta_json={'cache_path': str(cache)}))
    PointCloudStationRepo.sync_assets('uuid')
    with memory_session(Session, 'uuid') as session:
        project = session.execute(select(Project)).scalar_one()
        station = session.execute(select(PointCloudStation)).scalar_one()
        assert project.initial_camera_json['position'] == [5.0, 6.0, 7.0]
        assert project.initial_camera_json['focal_point'] == [5.0, 10.0, 15.0]
        assert station.initial_pose_json['scan_poses'][0]['has_pose'] is True


def test_viewport_applies_saved_station_position(qapp):
    from view3d.pyvista_viewport import PyVistaViewport
    viewport = PyVistaViewport()
    try:
        viewport.set_initial_camera([5, 6, 7], [1, 2, 3])
        viewport.add_cloud('cloud', np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        assert np.allclose(viewport._plotter.camera.position, [5, 6, 7])
        assert np.allclose(viewport._plotter.camera.focal_point, [1, 2, 3])
        viewport._plotter.camera.position = [100, 100, 100]
        qapp.processEvents()
        updates = []
        viewport._plotter.update = lambda: updates.append(1)
        viewport.reset_view()
        assert np.allclose(viewport._plotter.camera.position, [5, 6, 7])
        assert viewport._render_pending
        qapp.processEvents()
        assert updates == [1]
    finally:
        viewport.destroy()


def test_plane_positive_side_faces_e57_camera_without_pose_fallback():
    from services.initial_camera_service import orient_plane_to_camera

    plane = np.array([1.0, 0.0, 0.0, -10.0])
    camera = {'position': [0.0, 0.0, 0.0]}
    oriented = orient_plane_to_camera(plane, [10.0, 0.0, 0.0], camera)
    assert np.allclose(oriented, [-1.0, 0.0, 0.0, 10.0])
    assert np.allclose(orient_plane_to_camera(plane, [10.0, 0.0, 0.0], None), plane)


def test_station_camera_is_attached_to_quality_dataset(monkeypatch):
    from services.pointcloud_station_service import PointCloudStationService

    class Render:
        def __init__(self):
            self.camera = None

        def set_initial_camera(self, camera):
            self.camera = camera

    render = Render()
    service = PointCloudStationService(render, project_uuid='project')
    station = SimpleNamespace(initial_pose_json={'scan_poses': [{
        'has_pose': True, 'scan_index': 0, 'scan_position': [5.0, 6.0, 7.0],
    }]})
    dataset = SimpleNamespace(
        proxy_points=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]), metadata={})
    monkeypatch.setattr(PointCloudStationRepo, 'save_initial_camera',
                        lambda *_args: None)
    service._apply_initial_camera(station, dataset)
    assert dataset.metadata['initial_camera']['position'] == [5.0, 6.0, 7.0]
    assert render.camera['position'] == [5.0, 6.0, 7.0]
