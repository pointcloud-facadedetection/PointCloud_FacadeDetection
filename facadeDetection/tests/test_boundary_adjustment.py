from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from algorithms.facade.boundary_adjustment import (
    incremental_proxy_ids, resize_uv_bounds)
from models import Base, Facade, Project, ResultScene
from services.facade.facade_detection_service import FacadeDetectionService
from view3d.pyvista_viewport import PyVistaViewport


GEOMETRY = {
    'plane_model': [1.0, 0.0, 0.0, 0.0],
    'origin': [0.0, 0.0, 0.0],
    'u_axis': [0.0, 1.0, 0.0],
    'v_axis': [0.0, 0.0, 1.0],
    'uv_bounds': [0.0, 0.0, 2.0, 2.0],
}


def test_resize_edges_support_expand_shrink_and_minimum_size():
    bounds = np.array([0.0, 0.0, 2.0, 2.0])
    np.testing.assert_array_equal(
        resize_uv_bounds(bounds, 'right', 3.0, 0.0), [0, 0, 3, 2])
    np.testing.assert_array_equal(
        resize_uv_bounds(bounds, 'left', 0.5, 0.0), [0.5, 0, 2, 2])
    np.testing.assert_array_equal(
        resize_uv_bounds(bounds, 'top', 0.0, 0.05, min_size=0.1), [0, 0, 2, 0.1])
    np.testing.assert_array_equal(
        resize_uv_bounds(bounds, 'bottom', 0.0, -1.0), [0, -1, 2, 2])


def test_incremental_ids_only_add_newly_exposed_surface_points():
    points = np.array([
        [0.0, 1.0, 1.0],
        [0.0, 2.5, 1.0],
        [0.0, 3.5, 1.0],
        [0.05, 2.5, 1.0],
        [0.0, 1.0, 2.5],
        [0.0, 1.0, 4.0],
        [0.0, 1.5, 1.5],
        [0.0, 1.7, 1.7],
    ])
    ids = incremental_proxy_ids(
        points, GEOMETRY, [0], [0.0, 0.0, 3.0, 3.0],
        surface_band_m=0.01)
    np.testing.assert_array_equal(ids, [0, 1, 4])
    retained = incremental_proxy_ids(
        points, GEOMETRY, [0, 6, 7], [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_array_equal(retained, [0, 6, 7])
    with pytest.raises(ValueError, match='有效点不足'):
        incremental_proxy_ids(points, GEOMETRY, ids, [0, 0, 0.1, 0.1])


def test_resize_service_updates_only_selected_facade_and_keeps_plane():
    points = np.array([
        [0.0, 1.0, 1.0], [0.0, 1.5, 1.0], [0.0, 1.0, 1.5],
        [0.0, 2.5, 1.0], [0.0, 1.0, 2.5],
    ])
    dataset = SimpleNamespace(
        proxy_points=points, revision='r1', dataset_id='ds', metadata={})
    viewport = SimpleNamespace(get_cloud_data=lambda _name: {
        'dataset_id': 'ds', 'station_id': 1})
    pointcloud = SimpleNamespace(get_dataset=lambda _id: dataset)
    index = SimpleNamespace(
        populate_voxel_ids=lambda items, _cloud: None,
        get_raw_count_for_facade=lambda _cloud, facade: facade['point_count'])
    service = FacadeDetectionService(viewport, pointcloud, index)
    facade = {
        'id': 9, 'facade_db_id': 9, 'dataset_revision': 'r1',
        'plane_model': [1.0, 0, 0, 0], 'proxy_indices': [0, 1, 2],
        'quality_status': 'complete',
        'quality_report': {'facade_geometry': dict(GEOMETRY)},
    }
    adjusted = service.resize_facade(
        'cloud', facade, [0.0, 0.0, 3.0, 3.0])
    assert adjusted['id'] == 9
    assert adjusted['proxy_indices'] == [0, 1, 2, 3, 4]
    assert adjusted['plane_model'] == facade['plane_model']
    assert adjusted['quality_status'] == 'pending'
    assert adjusted['edit_revision'] == 1
    assert adjusted['quality_report']['facade_geometry']['uv_bounds'] == [
        0.0, 0.0, 3.0, 3.0]


def test_resize_move_updates_live_bounds_without_rebuilding_cloud():
    viewport = PyVistaViewport.__new__(PyVistaViewport)
    viewport._quad_resize = {
        'origin': np.zeros(3), 'u_axis': np.array([0.0, 1, 0]),
        'v_axis': np.array([0.0, 0, 1]), 'plane_model': np.array([1.0, 0, 0, 0]),
        'bounds': np.array([0.0, 0.0, 2.0, 2.0]),
    }
    viewport._quad_resize_drag = 'right'
    viewport.ray_plane_pick = lambda *_args: np.array([0.0, 3.0, 1.0])
    viewport._refresh_heatmap_resize = lambda: None
    assert viewport._quad_resize_move(QPoint(10, 10)) is True
    np.testing.assert_array_equal(
        viewport._quad_resize['bounds'], [0.0, 0.0, 3.0, 2.0])


def test_resize_release_waits_for_explicit_confirmation():
    viewport = PyVistaViewport.__new__(PyVistaViewport)
    completed = []
    viewport._quad_resize = {
        'bounds': np.array([0.0, 0.0, 3.0, 2.0]),
        'on_complete': completed.append,
    }
    viewport._quad_resize_drag = 'right'
    viewport._quad_resize_move = lambda _xy: True

    for _ in range(4):
        assert viewport._quad_resize_release(QPoint(10, 10)) is True
        viewport._quad_resize_drag = 'right'
    assert completed == []
    assert viewport._quad_resize is not None

    viewport.exit_heatmap_resize = lambda: setattr(
        viewport, '_quad_resize', None)
    assert viewport.confirm_heatmap_resize() is True
    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0], [0.0, 0.0, 3.0, 2.0])


def test_resize_refresh_reuses_actors_and_skips_unchanged_move(qapp):
    viewport = PyVistaViewport()
    try:
        viewport.enter_heatmap_resize(GEOMETRY)
        actors = list(viewport._quad_resize_actors)
        assert len(actors) == 2
        viewport.ray_plane_pick = lambda *_args: np.array([0.0, 3.0, 1.0])
        viewport._quad_resize_drag = 'right'
        viewport._quad_resize_move(QPoint(10, 10))
        assert viewport._quad_resize_actors == actors

        renders = []
        viewport._render = lambda: renders.append(1)
        viewport._quad_resize_move(QPoint(10, 10))
        assert renders == []
    finally:
        viewport.destroy()


def test_resized_facade_persistence_invalidates_old_quality(monkeypatch):
    import services.dal.results_repo as repo_module

    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        project = Project(uuid='p', name='项目')
        session.add(project)
        session.flush()
        scene = ResultScene(project_id=project.id, name='Scene', is_active=True)
        session.add(scene)
        session.flush()
        row = Facade(
            project_id=project.id, scene_id=scene.id, label='Facade 1',
            display_no=1, plane_json={'proxy_indices': [0, 1, 2]},
            quality_status='complete', quality_report_json={'ok': True},
            point_count=3, raw_point_count=3)
        session.add(row)
        session.commit()
        facade_id = row.id

    @contextmanager
    def project_session(uuid):
        assert uuid == 'p'
        with Session() as session:
            yield session
            session.commit()

    monkeypatch.setattr(repo_module, 'project_session', project_session)
    adjusted = {
        'point_count': 5, 'raw_point_count': 5,
        'proxy_indices': [0, 1, 2, 3, 4], 'inlier_indices': [0, 1, 2, 3, 4],
        'voxel_ids': [1], 'edit_revision': 1, 'manually_adjusted': True,
        'bbox_2d': {}, 'area': 9.0,
        'quality_report': {'facade_geometry': dict(GEOMETRY)},
    }
    repo_module.ResultsRepo.update_resized_facade('p', facade_id, adjusted)
    with Session() as session:
        row = session.execute(
            select(Facade).where(Facade.id == facade_id)).scalar_one()
        assert row.point_count == 5
        assert row.quality_status == 'pending'
        assert row.quality_report_json is None
        assert row.plane_json['proxy_indices'] == [0, 1, 2, 3, 4]
        assert row.plane_json['edit_revision'] == 1
        assert row.plane_json['facade_geometry']['uv_bounds'] == [0, 0, 2, 2]
