"""B4：service 弹窗上移回归——ProjectOperationService 的对话框类已全部移除，
交互改经 info_requested / color_pick_requested 信号与 apply_global_color 回调。
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from services.project_operation.project_operation_service import ProjectOperationService


@pytest.fixture
def service(qapp):
    viewport = SimpleNamespace(get_cloud_names=lambda: ['cloud1'])
    pointcloud = SimpleNamespace(resolve_processing_cloud=lambda: 'cloud1')
    render = mock.Mock()
    svc = ProjectOperationService(
        viewport, facade_service=SimpleNamespace(),
        pointcloud_service=pointcloud, render_service=render)
    return svc, render


def test_quality_inspection_without_results_emits_info(service):
    svc, _ = service
    svc._last_facade_results = None
    events = []
    svc.info_requested.connect(lambda t, m: events.append((t, m)))
    svc.quality_inspection()
    assert events == [('质量检测', '还未进行质量检测。')]


def test_select_detection_area_without_cloud_emits_info(service):
    svc, _ = service
    svc._active_cloud_name = lambda: None
    events = []
    svc.info_requested.connect(lambda t, m: events.append((t, m)))
    svc.select_detection_area()
    assert events == [('框选检测区域', '请先加载点云数据。')]


def test_change_color_only_emits_pick_request(service):
    """change_color 入口只发取色请求；取消（无回调）时不触碰渲染与存储。"""
    svc, render = service
    events = []
    svc.color_pick_requested.connect(lambda: events.append('pick'))
    svc.change_color()
    assert events == ['pick']
    render.set_global_point_color.assert_not_called()


def test_apply_global_color_confirm_branch(service):
    """确认选色后的回调：经渲染服务全局应用（无项目时不写持久化文件）。"""
    svc, render = service
    svc._project_uuid = None
    svc.apply_global_color((1.0, 0.0, 0.0))
    render.set_global_point_color.assert_called_once_with((1.0, 0.0, 0.0))


def test_apply_global_color_falls_back_without_render(service):
    svc, _ = service
    svc._render_service = None
    svc._pointcloud_service = SimpleNamespace()  # 无 render_service 属性
    svc._apply_global_color_direct = mock.Mock()
    svc.apply_global_color((0.0, 1.0, 0.0))
    svc._apply_global_color_direct.assert_called_once_with((0.0, 1.0, 0.0))
