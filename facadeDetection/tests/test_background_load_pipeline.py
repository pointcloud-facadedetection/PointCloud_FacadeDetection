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
