"""质量评估后台任务：阶段进度转发、后处理（热力图导出）失败不影响评估结果、
结果导出复用已渲染的透明热力层。
"""
from __future__ import annotations

import numpy as np


def test_quality_worker_forwards_stages_and_runs_postprocess(qapp):
    from utils.workers import QualityWorker

    class Service:
        def compute_quality(self, cloud, facade, **kwargs):
            kwargs['progress_cb'](50, '真实计算阶段')
            return {'ok': True, 'overall': {}}

    worker = QualityWorker(
        Service(), 'cloud', {'id': 1}, {},
        postprocess=lambda result: result.update(exported=True))
    progress = []
    finished = []
    worker.signals.progress.connect(
        lambda percent, text: progress.append((percent, text)))
    worker.signals.finished.connect(
        lambda facade, result: finished.append(result))
    worker.run()

    assert (50, '真实计算阶段') in progress
    assert (95, '正在生成热力图') in progress
    assert finished[0]['exported'] is True


def test_quality_worker_heatmap_failure_is_best_effort(qapp):
    from utils.workers import QualityWorker

    class Service:
        def compute_quality(self, cloud, facade, **kwargs):
            return {'ok': True, 'overall': {}}

    def fail_export(_result):
        raise RuntimeError('模拟图片写入失败')

    worker = QualityWorker(
        Service(), 'cloud', {'id': 1}, {}, postprocess=fail_export)
    finished = []
    worker.signals.finished.connect(
        lambda facade, result: finished.append(result))
    worker.run()
    assert len(finished) == 1
    assert finished[0]['ok'] is True


def test_heatmap_export_reuses_triplet_transparent(tmp_path):
    from services.result_export_service import ResultExportService

    class Renderer:
        def __init__(self):
            self.render_calls = 0

        def render(self, **_kwargs):
            self.render_calls += 1
            image = np.ones((2, 2, 4), dtype=np.uint8)
            return {key: image.copy() for key in (
                'overlay', 'heatmap_grid', 'photo', 'transparent')}

        def fit_report_image(self, image):
            return image

        def render_transparent_heatmap(self, **_kwargs):
            raise AssertionError('透明图不得重复光栅化')

    service = ResultExportService()
    renderer = Renderer()
    service._renderer = renderer
    methods = {
        method: {
            metric: {'windows': [{'center_xyz': [0, 0, 0]}]}
            for metric in ('flatness', 'verticality')
        } for method in ('ruler', 'global_plane')
    }
    quality = {
        'overall': {'plane_model': [1.0, 0, 0, 0]},
        'quality_comparison': {'methods': methods},
        '__global_indices': np.array([0], dtype=np.int64),
    }
    exported = service.export_all_heatmaps(
        tmp_path, 1, np.zeros((1, 3)), np.ones((1, 3)), quality)
    assert renderer.render_calls == 4
    assert len(exported) == 4
    assert all(item['transparent'] for item in exported.values())
