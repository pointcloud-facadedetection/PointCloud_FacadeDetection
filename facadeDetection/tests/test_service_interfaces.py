"""A 组：2a 接口公开化回归（services 层纯 Python 委托链）。"""
from unittest import mock

from services.project_operation.project_operation_service import ProjectOperationService
from services.facade.facade_service import FacadeService
from services.facade.facade_quality_service import FacadeQualityService
from services.report_export.report_export_service import ReportExportService


class TestLastFacadeResultsProperty:
    """ProjectOperationService.last_facade_results 公开 property。"""

    def _make_service(self):
        service = ProjectOperationService.__new__(ProjectOperationService)
        service._last_facade_results = None
        return service

    def test_initial_none(self):
        assert self._make_service().last_facade_results is None

    def test_write_read_roundtrip_matches_backing_field(self):
        service = self._make_service()
        results = [{'id': 1}, {'id': 2}]
        service.last_facade_results = results
        assert service.last_facade_results is results
        assert service._last_facade_results is results

    def test_reset_to_none(self):
        service = self._make_service()
        service.last_facade_results = [{'id': 1}]
        service.last_facade_results = None
        assert service.last_facade_results is None
        assert service._last_facade_results is None


class TestFacadeServicePublicAccessors:
    """FacadeService.index_service / get_dataset 委托正确性。"""

    def test_index_service_property_returns_backing_service(self):
        service = FacadeService.__new__(FacadeService)
        sentinel = object()
        service._index_service = sentinel
        assert service.index_service is sentinel

    def test_get_dataset_delegates_with_cloud_name(self):
        service = FacadeService.__new__(FacadeService)
        index = mock.Mock()
        index._get_dataset.return_value = 'dataset-1'
        service._index_service = index
        assert service.get_dataset('cloud-a') == 'dataset-1'
        index._get_dataset.assert_called_once_with('cloud-a')


class TestFacadeQualityServiceCommit:
    """FacadeQualityService.commit_quality_success 参数原样透传到 DAL。"""

    def test_passthrough_to_results_repo(self):
        service = FacadeQualityService()
        quality = {'ok': True, 'overall': {}}
        facade_data = {'id': 7}
        with mock.patch('services.dal.results_repo.ResultsRepo') as repo:
            service.commit_quality_success(
                'uuid-1', 7, quality,
                display_no=3, facade_data=facade_data, color=(1.0, 0.0, 0.0),
                dataset_revision=5, quality_artifact_path=None)
            repo.commit_quality_success.assert_called_once_with(
                'uuid-1', 7, quality,
                display_no=3, facade_data=facade_data, color=(1.0, 0.0, 0.0),
                dataset_revision=5, quality_artifact_path=None)

    def test_default_kwargs_passthrough(self):
        service = FacadeQualityService()
        with mock.patch('services.dal.results_repo.ResultsRepo') as repo:
            service.commit_quality_success('uuid-2', 1, {'ok': True})
            repo.commit_quality_success.assert_called_once_with(
                'uuid-2', 1, {'ok': True},
                display_no=None, facade_data=None, color=None,
                dataset_revision=None, quality_artifact_path=None)


class TestReportExportServiceRegisterPdf:
    """ReportExportService.register_pdf 透传到 ReportRepo。"""

    def test_passthrough_to_report_repo(self):
        service = ReportExportService()
        with mock.patch('services.dal.report_repo.ReportRepo') as repo:
            repo.register_pdf.return_value = 42
            out = service.register_pdf('uuid-1', '/tmp/a.pdf', '建筑外立面质量检测报告')
            assert out == 42
            repo.register_pdf.assert_called_once_with(
                'uuid-1', '/tmp/a.pdf', '建筑外立面质量检测报告')

    def test_title_defaults_to_none(self):
        service = ReportExportService()
        with mock.patch('services.dal.report_repo.ReportRepo') as repo:
            service.register_pdf('uuid-1', '/tmp/a.pdf')
            repo.register_pdf.assert_called_once_with('uuid-1', '/tmp/a.pdf', None)
