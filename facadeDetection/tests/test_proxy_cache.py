"""站点两层磁盘缓存（原始点 + 分层代理）的实效验证。

每个用例同时断言三类事实，防止"全部跳过、计时归零"的假优化：
1. 机制：self._load（Open3D 读盘）/ stratified_proxy_build / np.load 的
   调用计数符合命中或未命中预期；
2. 数据：raw 三件套（points.npy/colors.npy/指纹 json）与 proxy npz 真实写盘
   （存在且 >0 字节）、缓存读出的 raw/proxy 与首轮结果 np.array_equal
   完全一致、CSR offsets 一致、nbytes>0；
3. 时间：首轮与缓存轮都有真实毫秒级耗时，缓存路径可测地更短。

.dist 使用真实二进制 float32 文件（utils/dist_reader.py 的 mmap 二进制分支：
文件大小 % 4 == 0、元素数与点数一致、前 64 字节含 NUL），不做 monkeypatch。
仅使用 tmp_path 下的普通文件；Storage.project_root 打桩到 tmp_path，
不创建真实项目目录或数据库。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

import services.pointcloud_station_service as pss_mod
from algorithms.geometry import stratified_proxy_build as real_proxy_build
from config.storage import Storage
from services import proxy_cache
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService


def _write_ply(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    assert o3d.io.write_point_cloud(str(path), cloud)
    return path


def _write_dist(path, n_points, seed=11):
    """真实二进制 .dist：float32 距离值，落在 dist_reader 的 mmap 分支。"""
    rng = np.random.default_rng(seed)
    values = rng.uniform(2.0, 60.0, n_points).astype('<f4')
    values.tofile(path)
    assert path.stat().st_size == n_points * 4
    return path


def _make_scan(n_side=550, spacing=0.01, seed=7):
    """约 30 万点的仿真墙体（x≈0 平面），float32 保证 PLY 往返精确相等。"""
    rng = np.random.default_rng(seed)
    grid = (np.arange(n_side) - n_side / 2) * spacing
    yy, zz = np.meshgrid(grid, grid)
    pts = np.stack([
        rng.normal(0, 0.002, n_side * n_side),
        yy.ravel(),
        zz.ravel(),
    ], axis=1).astype(np.float32)
    cols = np.full_like(pts, 0.6)
    return pts, cols


def _real_fingerprint(path):
    data = Path(path).read_bytes()
    return (str(path), hashlib.sha256(data).hexdigest(), len(data))


class _Harness:
    """装配真实文件 + 计数代理 + 打桩 Storage/Repo 的测试环境。"""

    def __init__(self, tmp_path, monkeypatch, n_side=550):
        pts, cols = _make_scan(n_side=n_side)
        self.n_points = len(pts)
        self.ply = _write_ply(tmp_path / 'scan.ply', pts, cols)
        self.dist = _write_dist(tmp_path / 'scan.dist', self.n_points)
        self.project_root = tmp_path / 'proj'
        self.cache_path = (self.project_root / Storage.CACHE_DIRNAME
                           / 'proxy' / '7.npz')
        self.raw_points_path = (self.project_root / Storage.CACHE_DIRNAME
                                / 'raw' / '7.points.npy')
        self.raw_colors_path = (self.project_root / Storage.CACHE_DIRNAME
                                / 'raw' / '7.colors.npy')
        self.raw_meta_path = (self.project_root / Storage.CACHE_DIRNAME
                              / 'raw' / '7.json')
        self.station = SimpleNamespace(
            id=7, source_path=str(self.ply), display_name='s7')

        monkeypatch.setattr(Storage, 'project_root',
                            classmethod(lambda cls, u: self.project_root))
        monkeypatch.setattr(
            PointCloudStationRepo, 'get_asset_fingerprint',
            staticmethod(lambda u, s: _real_fingerprint(self.ply)))
        self.denoise_state = {'value': None}
        monkeypatch.setattr(
            PointCloudStationRepo, 'get_denoise_state',
            staticmethod(lambda u, s: self.denoise_state['value']))

        self.build_calls = []
        def counting_build(*args, **kwargs):
            self.build_calls.append(time.perf_counter())
            return real_proxy_build(*args, **kwargs)
        monkeypatch.setattr(pss_mod, 'stratified_proxy_build', counting_build)

        self.ply_load_calls = []
        real_load = PointCloudStationService._load
        def counting_load(service, path):
            self.ply_load_calls.append(time.perf_counter())
            return real_load(service, path)
        monkeypatch.setattr(PointCloudStationService, '_load', counting_load)

        self.np_load_calls = []
        real_np_load = np.load
        def counting_np_load(*args, **kwargs):
            self.np_load_calls.append(args[0] if args else None)
            return real_np_load(*args, **kwargs)
        # proxy_cache 通过 np.load 读盘；临时替换 numpy 全局符号统计真实读盘次数
        monkeypatch.setattr(np, 'load', counting_np_load)

    def new_service(self):
        """全新 service + 全新 PointCloudService，模拟重开项目后的空运行时。"""
        return PointCloudStationService(
            render_service=SimpleNamespace(), project_uuid='u',
            pointcloud_service=PointCloudService())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return _Harness(tmp_path, monkeypatch)


class TestProxyCache:

    def test_first_build_writes_cache_reopen_hits_cache(self, harness):
        # ---- 第一次加载：真实读盘、代理真实重建，raw 三件套与 proxy npz 真实写盘 ----
        svc1 = harness.new_service()
        t0 = time.perf_counter()
        ds1 = svc1._load_proxy_domain(harness.station)
        t_first = time.perf_counter() - t0
        assert len(harness.ply_load_calls) == 1       # 机制：PLY 读取真实发生
        assert len(harness.build_calls) == 1          # 机制：重建真实发生
        assert harness.cache_path.exists()
        assert harness.cache_path.stat().st_size > 0  # 数据：proxy npz 真实写盘
        for raw_file in (harness.raw_points_path, harness.raw_colors_path,
                         harness.raw_meta_path):
            assert raw_file.exists()
            assert raw_file.stat().st_size > 0        # 数据：raw 三件套真实写盘
        assert 'proxy_cache' not in ds1.metadata      # 首次走 dist 重建分支
        assert ds1.metadata['distance_source'] == 'dist'
        assert len(ds1.proxy_points) > 0
        assert ds1.proxy_points.nbytes > 0
        first_proxy = np.array(ds1.proxy_points)      # 快照，防止后续被改
        first_offsets = np.asarray(ds1.metadata['proxy_source_offsets'])
        first_colors = np.array(ds1.proxy_colors)
        first_raw = np.asarray(
            svc1.pointcloud.get_source_asset('u:7:source')['points'],
            dtype=np.float32)
        assert first_raw.nbytes > 0
        assert first_offsets[0] == 0
        assert np.all(np.diff(first_offsets) > 0)     # CSR 严格递增

        # ---- 全新 service 实例（模拟重开项目）：两层缓存命中，零读盘零重建 ----
        svc2 = harness.new_service()
        loads_before = len(harness.np_load_calls)
        t0 = time.perf_counter()
        ds2 = svc2._load_proxy_domain(harness.station)
        t_second = time.perf_counter() - t0
        assert len(harness.ply_load_calls) == 1       # 机制：未再读盘
        assert len(harness.build_calls) == 1          # 机制：没有再重建
        assert len(harness.np_load_calls) > loads_before  # 机制：缓存真实读盘
        assert ds2.metadata['proxy_cache'] == 'restored'
        # 数据：缓存恢复的 raw/proxy/颜色/CSR 与第一次结果完全一致
        second_raw = svc2.pointcloud.get_source_asset('u:7:source')['points']
        assert second_raw.nbytes > 0
        assert np.array_equal(np.asarray(second_raw, dtype=np.float32),
                              first_raw)
        assert np.array_equal(ds2.proxy_points, first_proxy)
        assert np.array_equal(ds2.proxy_colors, first_colors)
        assert ds2.proxy_points.nbytes > 0
        assert np.array_equal(
            np.asarray(ds2.metadata['proxy_source_offsets']), first_offsets)
        assert np.array_equal(
            np.asarray(ds2.metadata['proxy_source_indices']),
            np.asarray(ds1.metadata['proxy_source_indices']))
        assert (ds2.metadata['scan_origins'] ==
                ds1.metadata['scan_origins'])
        assert ds2.metadata['distance_source'] == 'dist'
        # 时间：缓存路径可测地快于真实读盘+重建
        print(f'\n[perf] first_load={t_first*1e3:.2f}ms '
              f'cache_restore={t_second*1e3:.2f}ms '
              f'proxy={len(first_proxy)} raw={harness.n_points}')
        assert t_first > 0 and t_second > 0
        assert t_second < t_first

    def test_fingerprint_mismatch_rebuilds_and_overwrites(self, harness):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 1
        assert len(harness.build_calls) == 1
        old_bytes = harness.cache_path.read_bytes()
        old_raw_bytes = harness.raw_points_path.read_bytes()
        old_offsets = np.asarray(ds1.metadata['proxy_source_offsets'])
        old_fingerprint = _real_fingerprint(harness.ply)

        # 修改源文件：指纹（sha256+size）随之变化，两层旧缓存必须失效
        pts, cols = _make_scan(n_side=400, seed=99)
        _write_ply(harness.ply, pts, cols)
        _write_dist(harness.dist, len(pts), seed=12)
        assert _real_fingerprint(harness.ply) != old_fingerprint
        assert proxy_cache.load_raw_cache('u', 7, old_fingerprint) is not None
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is None
        assert proxy_cache.load_proxy_cache(
            'u', 7, _real_fingerprint(harness.ply),
            source_count=len(pts)) is None

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 2     # 机制：重新读盘
        assert len(harness.build_calls) == 2        # 机制：指纹不符 → 重建
        assert 'proxy_cache' not in ds2.metadata
        # 数据：proxy npz 与 raw 三件套被真实覆盖，且新内容与新结果一致
        assert harness.cache_path.read_bytes() != old_bytes
        assert harness.raw_points_path.read_bytes() != old_raw_bytes
        raw_cached = proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply))
        assert raw_cached is not None
        new_raw = svc2.pointcloud.get_source_asset('u:7:source')['points']
        assert np.array_equal(raw_cached[0],
                              np.asarray(new_raw, dtype=np.float32))
        cached = proxy_cache.load_proxy_cache(
            'u', 7, _real_fingerprint(harness.ply), source_count=len(pts))
        assert cached is not None
        assert np.array_equal(cached['offsets'],
                              np.asarray(ds2.metadata['proxy_source_offsets']))
        assert not np.array_equal(cached['offsets'], old_offsets) or \
            len(cached['offsets']) != len(old_offsets)

    def test_corrupt_cache_falls_back_to_rebuild(self, harness):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        assert len(harness.build_calls) == 1
        first_proxy = np.array(ds1.proxy_points)

        # 写入垃圾字节破坏 npz
        harness.cache_path.write_bytes(b'\x00\xffgarbage' * 64)
        assert proxy_cache.load_proxy_cache(
            'u', 7, _real_fingerprint(harness.ply),
            source_count=harness.n_points) is None

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)  # 不得崩溃
        assert len(harness.ply_load_calls) == 1     # 机制：raw 缓存仍命中
        assert len(harness.build_calls) == 2        # 机制：回退真实重建
        assert 'proxy_cache' not in ds2.metadata
        assert np.array_equal(ds2.proxy_points, first_proxy)  # 结果确定
        # 重建后缓存被重新写盘，恢复为可读状态
        assert proxy_cache.load_proxy_cache(
            'u', 7, _real_fingerprint(harness.ply),
            source_count=harness.n_points) is not None

    def test_corrupt_raw_cache_falls_back_to_ply(self, harness):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 1
        assert len(harness.build_calls) == 1
        first_raw = np.asarray(
            svc1.pointcloud.get_source_asset('u:7:source')['points'],
            dtype=np.float32)
        first_proxy = np.array(ds1.proxy_points)

        # 写入垃圾字节破坏 raw points.npy；proxy npz 保持完好
        harness.raw_points_path.write_bytes(b'\x00\xffgarbage' * 64)
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is None

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)  # 不得崩溃
        # 机制：回退真实读盘；proxy 缓存仍命中，不重建
        assert len(harness.ply_load_calls) == 2
        assert len(harness.build_calls) == 1
        assert ds2.metadata['proxy_cache'] == 'restored'
        # 数据：回退读盘的 raw 与首轮逐点一致，raw 缓存被重新写盘
        second_raw = svc2.pointcloud.get_source_asset('u:7:source')['points']
        assert np.array_equal(np.asarray(second_raw, dtype=np.float32),
                              first_raw)
        assert np.array_equal(ds2.proxy_points, first_proxy)
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is not None

    def test_denoise_restored_direct_beats_cache(self, harness):
        # 预置一个与指纹匹配的合法缓存（哨兵 distance_source 证明未被采用）
        fp = _real_fingerprint(harness.ply)
        assert proxy_cache.save_proxy_cache(
            'u', 7, fp,
            offsets=np.array([0, 1], dtype=np.int64),
            indices=np.array([0], dtype=np.int32),
            ranges=np.array([0.0], dtype=np.float32),
            scan_origins=np.zeros((1, 3), dtype=np.float32),
            distance_source='sentinel',
            representative_ids=np.array([0], dtype=np.int64))
        # 合法去噪快照：CSR 指向真实源点行
        harness.denoise_state['value'] = {
            'enabled': True,
            'proxy_count': 2,
            'proxy_source_offsets': [0, 2, 5],
            'proxy_source_indices': [10, 20, 30, 40, 50],
        }
        svc = harness.new_service()
        loads_before = len(harness.np_load_calls)
        ds = svc._load_proxy_domain(harness.station)
        # 机制：restored_direct 优先，缓存连读盘都没发生，重建也没发生
        assert len(harness.build_calls) == 0
        assert len(harness.np_load_calls) == loads_before
        assert ds.metadata['denoise_restored'] is True
        assert 'proxy_cache' not in ds.metadata
        assert ds.metadata.get('distance_source') != 'sentinel'
        # 数据：proxy 真实来自源点云按去噪 CSR 代表行采集
        raw = svc.pointcloud.get_source_asset('u:7:source')['points']
        assert np.array_equal(np.asarray(ds.proxy_points, dtype=np.float32),
                              raw[[10, 30]])

    def test_delete_selected_removes_cache_file(self, harness, monkeypatch):
        svc = harness.new_service()
        svc._load_proxy_domain(harness.station)
        assert harness.cache_path.exists()
        assert harness.raw_points_path.exists()
        proxy_dir = harness.cache_path.parent
        raw_dir = harness.raw_points_path.parent

        rows = [SimpleNamespace(id=7, is_selected=True, registered_path=None)]
        list_results = [rows, []]  # 删除前选中列表、删除后剩余列表
        monkeypatch.setattr(svc, 'list_stations',
                            lambda: list_results.pop(0))
        monkeypatch.setattr(PointCloudStationRepo, 'delete',
                            staticmethod(lambda u, ids: None))
        monkeypatch.setattr(PointCloudStationRepo, 'save_view',
                            staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(pss_mod, 'log_event', lambda *a, **k: None)
        svc.render = SimpleNamespace(clear_scene_display=lambda: None)

        svc.delete_selected()
        assert not harness.cache_path.exists()        # 数据：proxy npz 真实删除
        assert not harness.raw_points_path.exists()   # 数据：raw 三件套真实删除
        assert not harness.raw_colors_path.exists()
        assert not harness.raw_meta_path.exists()
        assert not proxy_dir.exists()                 # 空目录一并清理
        assert not raw_dir.exists()
