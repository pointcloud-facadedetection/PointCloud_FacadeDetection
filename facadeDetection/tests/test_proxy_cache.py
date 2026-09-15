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
import json
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

import services.pointcloud_station_service as pss_mod
from algorithms.geometry import build_proxy_domain as real_domain_build
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
            return real_domain_build(*args, **kwargs)
        monkeypatch.setattr(pss_mod, 'build_proxy_domain', counting_build)

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

        rows = [SimpleNamespace(id=7, is_selected=True, registered_path=None,
                                file_asset_id=None)]
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


class TestProxyDirectArrays:
    """proxy 缓存直存代理数组：重开轮完全不触碰 raw memmap。"""

    def test_reopen_uses_cached_proxy_arrays_without_raw_gather(
            self, harness, monkeypatch):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        assert len(harness.build_calls) == 1
        first_proxy = np.array(ds1.proxy_points)
        first_colors = np.array(ds1.proxy_colors)

        # 数据：proxy npz 真实包含代理数组字段，且与首轮结果逐点一致
        with np.load(harness.cache_path) as data:
            assert 'proxy_points' in data.files
            assert 'proxy_colors' in data.files
            cached_points = np.asarray(data['proxy_points'])
            cached_colors = np.asarray(data['proxy_colors'])
        assert cached_points.shape == first_proxy.shape
        assert np.array_equal(cached_points, first_proxy)
        assert np.array_equal(cached_colors, first_colors)

        # 机制：重开轮对 raw memmap 的任何取元素（gather/切片）计数必须为 0
        getitem_calls = []
        real_getitem = np.memmap.__getitem__
        def counting_getitem(self, key):
            getitem_calls.append(key)
            return real_getitem(self, key)
        monkeypatch.setattr(np.memmap, '__getitem__', counting_getitem)

        svc2 = harness.new_service()
        t0 = time.perf_counter()
        ds2 = svc2._load_proxy_domain(harness.station)
        t_second = time.perf_counter() - t0
        assert ds2.metadata['proxy_cache'] == 'restored'
        assert getitem_calls == []              # 零 gather：raw 页未被换入
        assert len(harness.ply_load_calls) == 1
        assert len(harness.build_calls) == 1
        # 数据：恢复的代理与首轮逐点一致；raw 仍是只读映射
        assert np.array_equal(ds2.proxy_points, first_proxy)
        assert np.array_equal(ds2.proxy_colors, first_colors)
        asset = svc2.pointcloud.get_source_asset('u:7:source')
        assert not asset['points'].flags.owndata
        print(f'\n[perf] reopen_with_proxy_arrays={t_second*1e3:.2f}ms '
              f'proxy={len(first_proxy)}')
        assert t_second > 0

    def test_legacy_cache_without_proxy_arrays_falls_back_to_gather(
            self, harness, monkeypatch):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        first_proxy = np.array(ds1.proxy_points)

        # 剥掉 proxy 数组字段，模拟旧格式缓存（其余字段原样保留）
        with np.load(harness.cache_path) as data:
            kept = {key: data[key] for key in data.files
                    if key not in ('proxy_points', 'proxy_colors')}
        np.savez(harness.cache_path, **kept)
        loaded = proxy_cache.load_proxy_cache(
            'u', 7, _real_fingerprint(harness.ply),
            source_count=harness.n_points)
        assert loaded is not None
        assert loaded['proxy_points'] is None   # 旧缓存无 proxy 数组字段

        getitem_calls = []
        real_getitem = np.memmap.__getitem__
        def counting_getitem(self, key):
            getitem_calls.append(key)
            return real_getitem(self, key)
        monkeypatch.setattr(np.memmap, '__getitem__', counting_getitem)

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)
        # 机制：旧缓存回退按代表行采集，gather 真实发生
        assert ds2.metadata['proxy_cache'] == 'restored'
        assert len(getitem_calls) > 0
        assert len(harness.build_calls) == 1    # 回退采集不等于重建
        # 数据：回退结果与首轮逐点一致
        assert np.array_equal(ds2.proxy_points, first_proxy)


class TestCsrNdarrayMetadata:
    """运行期 metadata 的 CSR/ranges 一律 ndarray；list 只在 JSON 边界出现。"""

    CSR_KEYS = ('proxy_source_offsets', 'proxy_source_indices', 'ranges')

    def test_metadata_csr_keys_are_ndarray_all_branches(self, harness):
        # dist 重建分支
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        for key in self.CSR_KEYS:
            assert isinstance(ds1.metadata[key], np.ndarray), key
        # 数据：register_dataset 的索引与 metadata 数组逐点一致
        assert np.array_equal(ds1.index.source_raw_offsets,
                              ds1.metadata['proxy_source_offsets'])
        assert np.array_equal(ds1.index.source_raw_indices,
                              ds1.metadata['proxy_source_indices'])
        assert int(ds1.index.source_raw_offsets[-1]) == \
            len(ds1.metadata['proxy_source_indices'])

        # proxy 缓存恢复分支
        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)
        assert ds2.metadata['proxy_cache'] == 'restored'
        for key in self.CSR_KEYS:
            assert isinstance(ds2.metadata[key], np.ndarray), key
        assert np.array_equal(ds2.metadata['proxy_source_offsets'],
                              ds1.metadata['proxy_source_offsets'])

        # 去噪直恢复分支（state 来自 JSON，运行期必须转回 ndarray）
        harness.denoise_state['value'] = {
            'enabled': True,
            'proxy_count': 2,
            'proxy_source_offsets': [0, 2, 5],
            'proxy_source_indices': [10, 20, 30, 40, 50],
            'ranges': [1.5, 2.5],
        }
        svc3 = harness.new_service()
        ds3 = svc3._load_proxy_domain(harness.station)
        assert ds3.metadata['denoise_restored'] is True
        for key in self.CSR_KEYS:
            assert isinstance(ds3.metadata[key], np.ndarray), key
        assert np.array_equal(ds3.metadata['proxy_source_offsets'], [0, 2, 5])
        assert ds3.metadata['proxy_source_indices'].dtype == np.int32

    def test_register_dataset_accepts_both_ndarray_and_list(self):
        # register_dataset 兼容性：ndarray 直传与旧 list 调用方都能注册
        offsets = np.array([0, 2, 5], dtype=np.int64)
        indices = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        source = np.zeros((64, 3), dtype=np.float32)
        proxy = np.zeros((2, 3), dtype=np.float32)
        pointcloud = PointCloudService()
        pointcloud.register_source_asset('src', source, None, {})
        ds_arr = pointcloud.register_dataset(
            'd1', proxy, None,
            metadata={'source_id': 'src',
                      'proxy_source_offsets': offsets,
                      'proxy_source_indices': indices})
        ds_list = pointcloud.register_dataset(
            'd2', proxy, None,
            metadata={'source_id': 'src',
                      'proxy_source_offsets': offsets.tolist(),
                      'proxy_source_indices': indices.tolist()})
        for ds in (ds_arr, ds_list):
            assert np.array_equal(ds.index.source_raw_offsets, offsets)
            assert np.array_equal(ds.index.source_raw_indices, indices)
            assert ds.index.source_raw_offsets.dtype == np.int64
            assert ds.index.source_raw_indices.dtype == np.int32

    def test_state_boundary_produces_ndarrays(self):
        # save_denoise_state 的持久化边界：大数组以 ndarray 交给 sidecar，
        # 运行期零 list 转换。
        from services.project_operation.project_operation_service import (
            _as_state_array)
        offsets = np.array([0, 3, 7], dtype=np.int64)
        indices = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        ranges = np.array([0.5, 1.5], dtype=np.float32)
        state = {
            'proxy_source_offsets': _as_state_array(offsets, np.int64),
            'proxy_source_indices': _as_state_array(indices, np.int64),
            'ranges': _as_state_array(ranges, np.float32),
        }
        for value in state.values():
            assert isinstance(value, np.ndarray)
        assert state['proxy_source_offsets'].tolist() == [0, 3, 7]
        assert state['proxy_source_indices'].tolist() == [1, 2, 3, 4, 5, 6, 7]
        # None 归一为空数组
        empty = _as_state_array(None, np.int64)
        assert isinstance(empty, np.ndarray) and len(empty) == 0

    def test_ndarray_metadata_avoids_python_int_roundtrip(self):
        # 内存证明：大 CSR 的 .tolist() 产生海量 Python int 临时对象
        # （pymalloc 不还给 OS），ndarray 直传的追踪峰值必须远低于它。
        n_proxy, group = 200_000, 20   # indices 4M
        offsets = np.arange(0, n_proxy * group + 1, group, dtype=np.int64)
        indices = np.arange(n_proxy * group, dtype=np.int32)
        ranges = np.zeros(n_proxy, dtype=np.float32)
        source = np.zeros((n_proxy * group, 3), dtype=np.float32)
        proxy = np.zeros((n_proxy, 3), dtype=np.float32)

        tracemalloc.start()
        meta_list = {'proxy_source_offsets': offsets.tolist(),
                     'proxy_source_indices': indices.tolist(),
                     'ranges': ranges.tolist()}
        peak_list = tracemalloc.get_traced_memory()[1]
        assert len(meta_list['proxy_source_indices']) == n_proxy * group
        del meta_list
        tracemalloc.reset_peak()

        pointcloud = PointCloudService()
        pointcloud.register_source_asset('src', source, None, {})
        dataset = pointcloud.register_dataset(
            'd', proxy, None,
            metadata={'source_id': 'src',
                      'proxy_source_offsets': offsets,
                      'proxy_source_indices': indices,
                      'ranges': ranges})
        peak_arr = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # 数据：ndarray 路径注册的索引与输入逐点一致
        assert np.array_equal(dataset.index.source_raw_offsets, offsets)
        assert np.array_equal(dataset.index.source_raw_indices, indices)
        # 内存：list 往返的 Python 分配峰值真实存在且远超 ndarray 路径
        print(f'\n[mem] tolist_peak={peak_list/1e6:.1f}MB '
              f'ndarray_peak={peak_arr/1e6:.1f}MB')
        assert peak_list > 50 * 1024 * 1024  # 4M Python int 必然超过 50MB
        assert peak_arr < 0.1 * peak_list


class TestProxyNormalsPersistence:
    """proxy 法向随缓存持久化：大点云检测估计一次，重开/再检测直接复用。

    三类事实：
    1. 机制：_estimate_geo_normals 调用计数、缓存命中/拒绝写入门控；
    2. 数据：npz 内 proxy_normals 与内存 float64 原值逐点一致，
       恢复后 dataset.proxy_normals 逐点一致；
    3. 时间：首次估计有可测耗时，复用路径可测地更快。
    """

    def _make_synthetic_cache(self, tmp_path, monkeypatch, n_proxy):
        """打桩 Storage.project_root 并预写一份带合法 CSR 的 proxy 缓存。"""
        project_root = tmp_path / 'proj'
        monkeypatch.setattr(Storage, 'project_root',
                            classmethod(lambda cls, u: project_root))
        fp = ('fp', 100, 200)
        offsets = np.arange(n_proxy + 1, dtype=np.int64)  # 每组 1 个源点
        assert proxy_cache.save_proxy_cache(
            'u', 7, fp, offsets=offsets,
            indices=np.arange(n_proxy, dtype=np.int32),
            ranges=np.zeros(n_proxy, dtype=np.float32),
            scan_origins=np.zeros((1, 3), dtype=np.float32),
            distance_source='dist',
            representative_ids=np.arange(n_proxy, dtype=np.int64),
            proxy_points=np.zeros((n_proxy, 3), dtype=np.float32))
        return fp

    def test_save_and_load_proxy_normals_roundtrip(self, tmp_path,
                                                   monkeypatch):
        n_proxy = 100
        fp = self._make_synthetic_cache(tmp_path, monkeypatch, n_proxy)
        # 未写法向的旧格式缓存：proxy_normals 为 None，缓存本体仍有效
        cached = proxy_cache.load_proxy_cache('u', 7, fp, source_count=n_proxy)
        assert cached is not None and cached['proxy_normals'] is None

        rng = np.random.default_rng(1)
        normals = rng.normal(size=(n_proxy, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        assert proxy_cache.save_proxy_normals('u', 7, fp, normals)
        cached = proxy_cache.load_proxy_cache('u', 7, fp, source_count=n_proxy)
        # 数据：float64 原值逐点往返；CSR 字段不受重写影响
        assert cached['proxy_normals'] is not None
        assert cached['proxy_normals'].dtype == np.float64
        assert np.array_equal(cached['proxy_normals'], normals)
        assert len(cached['offsets']) == n_proxy + 1
        assert np.array_equal(cached['indices'],
                              np.arange(n_proxy, dtype=np.int32))
        # 机制：指纹不符 / 长度不符 / 缓存缺失一律拒绝写入
        assert not proxy_cache.save_proxy_normals(
            'u', 7, ('other', 1, 2), normals)
        assert not proxy_cache.save_proxy_normals('u', 7, fp, normals[:-1])
        assert not proxy_cache.save_proxy_normals('u', 999, fp, normals)

    def test_station_restore_attaches_cached_normals(self, harness):
        svc1 = harness.new_service()
        ds1 = svc1._load_proxy_domain(harness.station)
        n_proxy = len(ds1.proxy_points)
        fp = _real_fingerprint(harness.ply)
        rng = np.random.default_rng(2)
        normals = rng.normal(size=(n_proxy, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        assert proxy_cache.save_proxy_normals('u', 7, fp, normals)

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)
        # 机制：仍命中缓存（未再读盘、未重建）
        assert ds2.metadata['proxy_cache'] == 'restored'
        assert len(harness.build_calls) == 1
        # 数据：恢复的法向挂到 dataset 且与写入逐点一致
        assert ds2.proxy_normals is not None
        assert ds2.proxy_normals.dtype == np.float64
        assert np.array_equal(ds2.proxy_normals, normals)

        # 去噪直恢复子集：长度与缓存代理数不符，不得挂缓存法向
        harness.denoise_state['value'] = {
            'enabled': True,
            'proxy_count': 2,
            'proxy_source_offsets': [0, 2, 5],
            'proxy_source_indices': [10, 20, 30, 40, 50],
        }
        svc3 = harness.new_service()
        ds3 = svc3._load_proxy_domain(harness.station)
        assert ds3.metadata['denoise_restored'] is True
        assert ds3.proxy_normals is None

    def test_detect_estimates_once_and_reopen_reuses_cached_normals(
            self, tmp_path, monkeypatch):
        import services.facade.facade_detection_service as fds_mod
        from services.pointcloud_index.core import (
            PointCloudDataset, RawPointStore)

        n_proxy = 520_000  # 触发 >=50 万的大点云估计+持久化分支
        fp = self._make_synthetic_cache(tmp_path, monkeypatch, n_proxy)
        rng = np.random.default_rng(4)
        proxy = rng.uniform(0, 3, (n_proxy, 3)).astype(np.float32)

        class _IndexStub:
            proxy_points = proxy
            proxy_colors = None

        def make_dataset(pointcloud):
            dataset = PointCloudDataset(
                dataset_id='u:cloud',
                raw=RawPointStore.from_arrays(proxy[:8]),
                index=_IndexStub(),
                metadata={'station_id': 7, 'asset_fingerprint': list(fp)})
            pointcloud.datasets['u:cloud'] = dataset
            return dataset

        seen_normals = []
        estimate_calls = []
        real_estimate = fds_mod._estimate_geo_normals

        def spy_estimate(geo, voxel_size):
            estimate_calls.append(time.perf_counter())
            return real_estimate(geo, voxel_size)

        def stub_detect(geo, **kwargs):
            seen_normals.append(
                np.asarray(geo.normals).copy() if geo.has_normals() else None)
            return {'facades': [], 'remaining': geo,
                    'total_points': len(geo.points)}

        monkeypatch.setattr(fds_mod, '_estimate_geo_normals', spy_estimate)
        monkeypatch.setattr(fds_mod, 'detect_facades_adaptive', stub_detect)
        monkeypatch.setattr(fds_mod.ResultsRepo, 'save_detected_facades',
                            staticmethod(lambda *a, **k: None))

        viewport = SimpleNamespace(
            get_cloud_data=lambda name: {'dataset_id': 'u:cloud',
                                         'station_id': 7})

        # 第一次检测：大点云法向真实估计一次并持久化
        pc1 = PointCloudService()
        ds1 = make_dataset(pc1)
        svc1 = fds_mod.FacadeDetectionService(viewport, pc1,
                                              index_service=None)
        t0 = time.perf_counter()
        svc1.detect('cloud', project_uuid='u')
        t_first = time.perf_counter() - t0
        # 机制：估计真实发生一次
        assert len(estimate_calls) == 1
        # 数据：float64 单位法向原值写入 dataset 并传给算法
        assert ds1.proxy_normals is not None
        assert ds1.proxy_normals.dtype == np.float64
        assert seen_normals[0] is not None
        assert np.array_equal(seen_normals[0], ds1.proxy_normals)
        norms = np.linalg.norm(ds1.proxy_normals, axis=1)
        assert np.all(np.abs(norms - 1.0) <= 1e-12)

        # 数据：proxy npz 真实追加 proxy_normals，与内存逐点一致
        cached = proxy_cache.load_proxy_cache('u', 7, fp,
                                              source_count=n_proxy)
        assert cached is not None and cached['proxy_normals'] is not None
        assert np.array_equal(cached['proxy_normals'], ds1.proxy_normals)

        # 模拟重开：全新 service + 从缓存恢复法向的 dataset
        pc2 = PointCloudService()
        ds2 = make_dataset(pc2)
        ds2.proxy_normals = cached['proxy_normals']
        svc2 = fds_mod.FacadeDetectionService(viewport, pc2,
                                              index_service=None)
        t0 = time.perf_counter()
        svc2.detect('cloud', project_uuid='u')
        t_second = time.perf_counter() - t0
        # 机制：estimate_normals 计 0 次，算法收到的就是缓存法向（逐点一致）
        assert len(estimate_calls) == 1
        assert np.array_equal(seen_normals[1], cached['proxy_normals'])
        # 时间：复用路径可测地快于真实估计
        print(f'\n[perf] proxy_normals first_detect={t_first:.2f}s '
              f'reuse_detect={t_second:.2f}s proxy={n_proxy}')
        assert t_second < t_first
