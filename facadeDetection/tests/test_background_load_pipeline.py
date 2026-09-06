"""点云加载管道后台化的实效验证。

每个用例同时断言三类事实，防止"全部跳过、计时归零"的假优化：
1. 机制：计算段真实发生在后台线程（threading.get_ident 计数），且计算段
   不做任何 Open3D 提交（渲染调用计数为 0）；提交段在 GUI 线程执行。
2. 数据：prepared 对象、dataset 代理数组、source 资产真实驻留内存
   （shape/nbytes/对象身份），证明数据真实流过管道。
3. 时间：真实磁盘读取与代理构建有可测耗时。

仅使用 tmp_path 下的普通文件；Storage 根目录一律打桩到 tmp_path。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

import services.file_service as file_service_mod
from config.storage import Storage
from models.enums import FileKind
from services.file_service import FileService, PreparedUpload
from services.pointcloud_service import PointCloudService


def _write_ply(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    assert o3d.io.write_point_cloud(str(path), cloud)
    return path


class FakeViewport:
    """最小视口替身：记录每次状态写入发生的线程。"""

    def __init__(self):
        self.clouds = {}
        self.access_threads = []

    def add_point_cloud(self, name, points, colors=None):
        self.access_threads.append(threading.get_ident())
        self.clouds[name] = {'pos': points, 'color': colors}

    def get_cloud_data(self, name):
        self.access_threads.append(threading.get_ident())
        return self.clouds.get(name)


class FakeRenderService:
    """记录 Open3D 提交调用与其发生线程，并像真实渲染服务一样写入视口。"""

    def __init__(self, pointcloud_service=None, viewport=None):
        self.pointcloud_service = pointcloud_service
        self.viewport = viewport
        self.calls = []

    def show_point_cloud(self, name, points, colors=None):
        self.calls.append(('cloud', name, threading.get_ident(), points, colors))
        if self.viewport is not None:
            self.viewport.add_point_cloud(name=name, points=points, colors=colors)

    def show_image(self, name, image):
        self.calls.append(('image', name, threading.get_ident()))


def _make_service():
    pointcloud = PointCloudService()
    viewport = FakeViewport()
    render = FakeRenderService(pointcloud, viewport)
    svc = FileService(viewport, None, render)
    return svc, pointcloud, viewport, render


# ---------------------------------------------------------------------------
# 1. upload 拆分：计算段（prepare_upload）可离 GUI 线程执行且零渲染调用，
#    提交段（commit_prepared）在 GUI 线程恰好提交一次
# ---------------------------------------------------------------------------
class TestUploadPrepareCommitSplit:
    def test_prepare_off_gui_commit_on_gui(self, tmp_path):
        gui_ident = threading.get_ident()
        n_points = 150_000
        rng = np.random.default_rng(11)
        pts = (rng.normal(size=(n_points, 3)) * 20).astype(np.float32)
        cols = rng.random((n_points, 3), dtype=np.float32)
        ply = _write_ply(tmp_path / 'scan.ply', pts, cols)
        # 配套 .dist（二进制 float32，长度等于点数）走完整的距离分层管线
        ranges = np.linalg.norm(pts, axis=1).astype('<f4')
        (tmp_path / 'scan.dist').write_bytes(ranges.tobytes())

        svc, pointcloud, viewport, render = _make_service()

        load_threads = []
        real_load = svc._load_point_cloud
        def counting_load(path):
            load_threads.append(threading.get_ident())
            return real_load(path)
        svc._load_point_cloud = counting_load

        holder = {}
        def work():
            t0 = time.perf_counter()
            holder['prepared'] = svc.prepare_upload(None, str(ply))
            holder['elapsed'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw_pts, raw_cols = real_load(str(ply))  # 基线：真实读盘耗时（不计入后台线程断言）
        t_real_read = time.perf_counter() - t0
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

        prepared = holder['prepared']
        # 机制：读取/解析/代理构建全部发生在后台线程，且计算段零渲染提交
        assert load_threads and all(ident != gui_ident for ident in load_threads)
        assert render.calls == []
        assert viewport.clouds == {}
        assert isinstance(prepared, PreparedUpload)
        assert prepared.kind == FileKind.raw_pointcloud

        # 数据：dataset 与 source 资产真实驻留，prepared 持有的就是代理数组本体
        dataset = pointcloud.get_dataset('local:scan.ply')
        assert dataset is not None
        assert prepared.dataset_id == 'local:scan.ply'
        assert prepared.points is dataset.proxy_points
        assert len(prepared.points) > 0
        assert dataset.proxy_points.nbytes > 0
        source = pointcloud.get_source_asset('local:source:scan')
        assert source['points'].nbytes == pts.nbytes
        offsets = dataset.index.source_raw_offsets
        assert offsets is not None and int(offsets[-1]) > 0

        # 时间：后台计算段做的是真实工作（读盘 + 高度角 + 分层代理）
        print(f'\n[perf] real_read={t_real_read*1e3:.2f}ms '
              f'prepare_off_gui={holder["elapsed"]*1e3:.2f}ms')
        assert t_real_read > 0
        assert holder['elapsed'] > t_real_read  # 读盘之外还有真实代理构建

        # 提交段：GUI 线程恰好提交一次，视口状态写回同一 dataset
        svc.commit_prepared(prepared)
        assert len(render.calls) == 1
        kind, name, call_thread, shown_pts, _shown_cols = render.calls[0]
        assert (kind, name) == ('cloud', 'scan.ply')
        assert call_thread == gui_ident
        assert shown_pts is dataset.proxy_points
        assert viewport.access_threads and all(
            ident == gui_ident for ident in viewport.access_threads)
        data = viewport.clouds['scan.ply']
        assert data['dataset_id'] == 'local:scan.ply'
        assert data['domain'] == 'proxy'
        assert len(data['proxy_ids']) == len(dataset.proxy_points)

    def test_sync_entry_equals_prepare_plus_commit(self, tmp_path):
        n_points = 50_000
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(n_points, 3)).astype(np.float32)
        cols = rng.random((n_points, 3), dtype=np.float32)
        ply = _write_ply(tmp_path / 'plain.ply', pts, cols)

        svc, pointcloud, viewport, render = _make_service()
        t0 = time.perf_counter()
        asset = svc.upload_files(None, str(ply))
        elapsed = time.perf_counter() - t0
        assert asset is None  # project_uuid=None 时不落 FileAsset
        # 同步入口走完同一条管道：一次渲染提交 + dataset 注册 + 视口绑定
        assert len(render.calls) == 1
        dataset = pointcloud.get_dataset('local:plain.ply')
        assert dataset is not None
        assert viewport.clouds['plain.ply']['dataset_id'] == 'local:plain.ply'
        # 无 .dist 时走标准体素路径：代理是体素重心，数量不超过原始点数
        assert 0 < len(dataset.proxy_points) <= n_points
        print(f'\n[perf] sync_upload_50k={elapsed*1e3:.2f}ms')
        assert elapsed > 0


# ---------------------------------------------------------------------------
# 2. FLS 导入：转换 + 逐站读取/注册全程不触碰渲染，可在后台线程完成，
#    逐站 progress_cb 真实回报
# ---------------------------------------------------------------------------
class TestFlsImportBackgroundable:
    def test_import_runs_off_gui_with_zero_render_calls(self, tmp_path, monkeypatch):
        gui_ident = threading.get_ident()
        monkeypatch.setattr(Storage, 'DATA_DIR', tmp_path / 'data')
        monkeypatch.setattr(Storage, 'PROJECTS_ROOT', tmp_path / 'data' / 'projects')

        rng = np.random.default_rng(5)
        station_sizes = (80_000, 60_000)
        convert_calls = []

        def fake_convert(fls_folder, output_dir, project_name):
            convert_calls.append((fls_folder, output_dir, project_name))
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            scans = []
            for index, size in enumerate(station_sizes, start=1):
                pts = (rng.normal(size=(size, 3)) * 15).astype(np.float32)
                cols = rng.random((size, 3), dtype=np.float32)
                ply = _write_ply(out / f'st{index}.ply', pts, cols)
                scans.append(SimpleNamespace(ply_path=str(ply.resolve())))
            return SimpleNamespace(output_dir=str(out), scans=scans, message='ok')

        monkeypatch.setattr(file_service_mod, 'convert_fls_to_ply', fake_convert)

        svc, pointcloud, viewport, render = _make_service()
        load_threads = []
        real_load = svc._load_point_cloud
        def counting_load(path):
            load_threads.append(threading.get_ident())
            return real_load(path)
        svc._load_point_cloud = counting_load

        progress = []
        holder = {}
        def work():
            t0 = time.perf_counter()
            holder['result'] = svc.import_fls_directory(
                str(tmp_path), None,
                progress_cb=lambda done, total, name: progress.append(
                    (done, total, name, threading.get_ident())))
            holder['elapsed'] = time.perf_counter() - t0
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

        result = holder['result']
        # 机制：转换器被子进程式调用一次；逐站读取发生在后台线程；零渲染提交
        assert len(convert_calls) == 1
        assert len(load_threads) == len(station_sizes)
        assert all(ident != gui_ident for ident in load_threads)
        assert all(p[3] != gui_ident for p in progress)
        assert render.calls == []
        assert viewport.clouds == {}

        # 数据：两站 dataset + source 资产真实驻留
        assert result['success'] is True
        assert result['uploaded'] == len(station_sizes)
        assert len(result['metadata']) == len(station_sizes)
        total_raw = 0
        for index, size in enumerate(station_sizes, start=1):
            dataset = pointcloud.get_dataset(f'local:st{index}.ply')
            assert dataset is not None
            assert len(dataset.proxy_points) > 0
            source = pointcloud.get_source_asset(f'local:source:st{index}')
            assert source['points'].nbytes == size * 3 * 4
            total_raw += source['points'].nbytes
        assert total_raw == sum(station_sizes) * 12

        # 进度逐站递增且计数与站点数一致
        assert [p[0] for p in progress] == [1, 2]
        assert all(p[1] == len(station_sizes) for p in progress)
        print(f'\n[perf] fls_import_2_stations_off_gui={holder["elapsed"]*1e3:.2f}ms')
        assert holder['elapsed'] > 0


# ---------------------------------------------------------------------------
# 3. ProjectOverviewService：批量准备/提交分离，create_load_worker 产出
#    携带 prepared 的结果并逐文件回报进度、响应取消
# ---------------------------------------------------------------------------
from services.dal.file_repo import FileRepo
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.project_overview.project_overview_service import ProjectOverviewService


class _FakeWorker:
    """prepare_upload_files 需要的最小 worker 协议（同步记录进度）。"""

    def __init__(self):
        self.progress = []
        self.signals = SimpleNamespace(
            progress=SimpleNamespace(
                emit=lambda pct, text: self.progress.append(
                    (pct, text, threading.get_ident()))))
        self.cancelled = False

    def check_cancelled(self):
        if self.cancelled:
            raise RuntimeError('点云加载任务已取消')


def _stub_persistence(monkeypatch, tmp_path):
    """把 DB/索引持久化打桩到计数器，管道其余部分全部真实执行。"""
    calls = {'import_file': 0, 'sync_assets': 0, 'pcfd_append': 0}
    monkeypatch.setattr(
        FileRepo, 'import_file',
        staticmethod(lambda **kwargs: calls.__setitem__(
            'import_file', calls['import_file'] + 1) or None))
    monkeypatch.setattr(
        PointCloudStationRepo, 'sync_assets',
        staticmethod(lambda uuid: calls.__setitem__(
            'sync_assets', calls['sync_assets'] + 1) or {}))
    monkeypatch.setattr(Storage, 'resolve_project_root',
                        classmethod(lambda cls, uuid: tmp_path))
    monkeypatch.setattr(
        Storage, 'append_pcfd_asset_for_uuid',
        classmethod(lambda cls, uuid, kind, rel: calls.__setitem__(
            'pcfd_append', calls['pcfd_append'] + 1)))
    return calls


def _make_overview():
    pointcloud = PointCloudService()
    viewport = FakeViewport()
    render = FakeRenderService(pointcloud, viewport)
    overview = ProjectOverviewService(viewport=viewport, render_service=render)
    return overview, pointcloud, viewport, render


class TestOverviewPrepareCommitSplit:
    def test_batch_prepare_off_gui_commit_on_gui(self, tmp_path, monkeypatch):
        gui_ident = threading.get_ident()
        calls = _stub_persistence(monkeypatch, tmp_path)
        overview, pointcloud, viewport, render = _make_overview()

        rng = np.random.default_rng(21)
        paths = []
        for index, size in enumerate((120_000, 90_000), start=1):
            pts = (rng.normal(size=(size, 3)) * 25).astype(np.float32)
            cols = rng.random((size, 3), dtype=np.float32)
            ply = _write_ply(tmp_path / f'up{index}.ply', pts, cols)
            ranges = np.linalg.norm(pts, axis=1).astype('<f4')
            (tmp_path / f'up{index}.dist').write_bytes(ranges.tobytes())
            paths.append(str(ply))
        # .dist 与 PLY 一起传入：必须被归并为一个上传任务而非独立资产
        upload_list = [paths[0], str(tmp_path / 'up1.dist'),
                       paths[1], str(tmp_path / 'up2.dist')]

        svc = overview._ensure_file_service()
        prepare_threads = []
        real_prepare = svc.prepare_upload
        def recording_prepare(*args, **kwargs):
            prepare_threads.append(threading.get_ident())
            return real_prepare(*args, **kwargs)
        monkeypatch.setattr(svc, 'prepare_upload', recording_prepare)

        holder = {}
        worker = _FakeWorker()
        def work():
            t0 = time.perf_counter()
            holder['out'] = overview.prepare_upload_files(
                upload_list, 'u1', worker=worker)
            holder['elapsed'] = time.perf_counter() - t0
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()

        prepared, uploaded = holder['out']
        # 机制：两个文件的准备都发生在后台线程，计算段零渲染提交；
        # 逐文件进度真实回报且发生在后台线程
        assert prepare_threads == [thread.ident] * 2 or all(
            ident != gui_ident for ident in prepare_threads)
        assert render.calls == []
        assert len(worker.progress) >= 2
        assert all(p[2] != gui_ident for p in worker.progress)
        assert [p[0] for p in worker.progress] == sorted(p[0] for p in worker.progress)

        # 数据：.dist 被归并（2 个任务而非 4 个），dataset/source 真实驻留
        assert len(prepared) == 2 and len(uploaded) == 2
        assert calls['import_file'] == 2       # 每个点云资产恰好持久化一次
        assert calls['sync_assets'] == 1       # 批尾一次同步
        total_proxy = 0
        for index, item in enumerate(prepared, start=1):
            dataset = pointcloud.get_dataset(f'u1:up{index}.ply')
            assert dataset is not None
            assert item.points is dataset.proxy_points
            assert len(item.points) > 0
            total_proxy += len(item.points)
        assert total_proxy > 0
        print(f'\n[perf] batch_prepare_2x100k_off_gui={holder["elapsed"]*1e3:.2f}ms')
        assert holder['elapsed'] > 0

        # 提交段：GUI 线程逐文件提交，增量语义（不清空已有云）
        viewport.clouds['existing'] = {'pos': np.zeros((1, 3), np.float32)}
        overview.commit_prepared_uploads(prepared)
        assert len(render.calls) == 2
        assert all(call[2] == gui_ident for call in render.calls)
        assert 'existing' in viewport.clouds   # 增量上传没有清空场景
        for index in (1, 2):
            data = viewport.clouds[f'up{index}.ply']
            assert data['dataset_id'] == f'u1:up{index}.ply'
            assert data['domain'] == 'proxy'

    def test_create_load_worker_result_and_progress(self, tmp_path, monkeypatch):
        calls = _stub_persistence(monkeypatch, tmp_path)
        overview, pointcloud, viewport, render = _make_overview()

        rng = np.random.default_rng(31)
        pts = (rng.normal(size=(60_000, 3)) * 10).astype(np.float32)
        cols = rng.random((60_000, 3), dtype=np.float32)
        ply = _write_ply(tmp_path / 'w.ply', pts, cols)

        worker = overview.create_load_worker('upload', 'u1', file_paths=[str(ply)])
        events = {'progress': [], 'finished': [], 'failed': []}
        worker.signals.progress.connect(
            lambda pct, text: events['progress'].append(pct))
        worker.signals.finished.connect(lambda r: events['finished'].append(r))
        worker.signals.failed.connect(lambda e: events['failed'].append(e))
        worker.run()  # 内联执行：信号直连，同步投递

        assert events['failed'] == []
        assert len(events['finished']) == 1
        payload = events['finished'][0]
        # 结果携带 prepared 对象与上传清单，供 GUI 完成回调提交
        assert payload['operation'] == 'upload'
        assert payload['uploaded'] == [str(Path(str(ply)).resolve())] or \
            payload['uploaded'] == [str(ply)]
        assert len(payload['prepared']) == 1
        dataset = pointcloud.get_dataset('u1:w.ply')
        assert payload['prepared'][0].points is dataset.proxy_points
        assert events['progress'][0] == 2       # worker 起始进度
        assert events['progress'][-1] == 100    # worker 完成进度
        assert render.calls == []               # worker 只做计算段

    def test_cancelled_worker_skips_compute(self, tmp_path, monkeypatch):
        calls = _stub_persistence(monkeypatch, tmp_path)
        overview, pointcloud, _, render = _make_overview()
        rng = np.random.default_rng(41)
        ply = _write_ply(tmp_path / 'c.ply',
                         rng.normal(size=(10_000, 3)).astype(np.float32),
                         rng.random((10_000, 3), dtype=np.float32))
        worker = overview.create_load_worker('upload', 'u1', file_paths=[str(ply)])
        events = {'finished': [], 'failed': []}
        worker.signals.finished.connect(lambda r: events['finished'].append(r))
        worker.signals.failed.connect(lambda e: events['failed'].append(e))
        worker.cancel()
        worker.run()
        # 机制：取消的任务不做任何读取/注册/提交
        assert events['finished'] == []
        assert pointcloud.get_dataset('u1:c.ply') is None
        assert render.calls == []
        assert calls['import_file'] == 0
