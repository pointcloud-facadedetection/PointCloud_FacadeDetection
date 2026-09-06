"""性能优化的实效验证。

每个用例同时断言三类事实，防止"全部跳过、计时归零"的假优化：
1. 机制：被省掉的昂贵操作确实没有再发生（读取/哈希/法线估计/GLFW 轮询计数）；
2. 数据：内存中的缓存/数组真实存在且内容正确（shape、nbytes、范数、弱引用释放）；
3. 时间：真实机器操作有可测耗时，复用路径可测地更快。

仅使用 tmp_path 下的普通文件，不创建真实项目目录或数据库。
"""
from __future__ import annotations

import gc
import hashlib
import os
import threading
import time
import weakref
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from services.dal.file_repo import FileRepo
import services.dal.file_repo as file_repo_mod
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.facade.facade_detection_service import FacadeDetectionService
import services.facade.facade_detection_service as fds_mod
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from view3d.open3d_adapter import Open3DAdapter


def _write_ply(path, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    assert o3d.io.write_point_cloud(str(path), cloud)
    return path


def _make_wall(n_side=100, spacing=0.05, seed=7):
    """生成一面 5m x 5m 的仿真墙体点云（x≈0 平面），含微小噪声。"""
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


# ---------------------------------------------------------------------------
# 1. 指纹门控：上传链注册的 dataset 必须被复用，同一文件不得二次读取
# ---------------------------------------------------------------------------
class TestStationDatasetReuse:
    def test_upload_chain_dataset_reused_without_second_read(self, tmp_path, monkeypatch):
        n_points = 200_000
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(n_points, 3)).astype(np.float32)
        cols = rng.random((n_points, 3)).astype(np.float32)
        ply = _write_ply(tmp_path / 'scan.ply', pts, cols)
        raw_nbytes = pts.nbytes + cols.nbytes
        assert raw_nbytes > 1_000_000  # 确保后续读的是真实大体量数据

        pointcloud = PointCloudService()
        # 模拟 file_service.upload_files 的上传链注册（legacy 键 + source 资产）
        legacy_id = 'u:scan.ply'
        source_id = 'u:source:scan'
        pointcloud.register_source_asset(source_id, pts, cols, {})
        upload_dataset = pointcloud.register_dataset(
            legacy_id, pts, cols, metadata={'source_id': source_id})
        assert pointcloud.get_source_asset(source_id)['points'].nbytes == pts.nbytes

        # 真实磁盘读取基线：证明 _load 真的会从磁盘读出全部点
        svc = PointCloudStationService(
            render_service=SimpleNamespace(), project_uuid='u',
            pointcloud_service=pointcloud)
        t0 = time.perf_counter()
        base_pts, base_cols = svc._load(str(ply))
        t_real_read = time.perf_counter() - t0
        assert len(base_pts) == n_points and base_cols is not None
        # _load 返回 float64，字节数翻倍；总量一致证明机器真的读了全部点
        assert base_pts.nbytes + base_cols.nbytes == raw_nbytes * 2
        assert t_real_read > 0

        monkeypatch.setattr(PointCloudStationRepo, 'get_asset_fingerprint',
                            staticmethod(lambda u, s: ('fp', 100, 200)))
        monkeypatch.setattr(PointCloudStationRepo, 'get_denoise_state',
                            staticmethod(lambda u, s: None))

        read_calls = []
        real_load = svc._load
        def counting_load(path):
            read_calls.append(path)
            return real_load(path)
        monkeypatch.setattr(svc, '_load', counting_load)

        station = SimpleNamespace(id=1, source_path=str(ply))
        t0 = time.perf_counter()
        ds1 = svc._load_proxy_domain(station)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        ds2 = svc._load_proxy_domain(station)
        t_second = time.perf_counter() - t0

        # 机制：两次展示都没有再读磁盘（旧实现此处会整站重读 + 重建代理）
        assert read_calls == []
        # 缓存：复用的就是上传链注册的同一个 dataset 对象，键已迁移到站点域
        assert ds1 is upload_dataset and ds2 is upload_dataset
        assert pointcloud.get_dataset('u:1') is upload_dataset
        assert pointcloud.get_dataset(legacy_id) is None
        assert ds1.metadata['station_id'] == 1
        assert svc._station_fingerprints[1] == ('fp', 100, 200)
        # 数据：代理与原始数组真实驻留，不是空壳
        assert len(ds1.proxy_points) > 0
        assert ds1.raw.points.nbytes > 0
        assert pointcloud.get_source_asset(source_id)['points'].nbytes == pts.nbytes
        # 时间：复用路径可测地快于一次真实磁盘读取
        print(f'\n[perf] real_read={t_real_read*1e3:.2f}ms '
              f'reuse1={t_first*1e3:.2f}ms reuse2={t_second*1e3:.2f}ms')
        assert t_first < t_real_read
        assert t_second < t_real_read


# ---------------------------------------------------------------------------
# 2. proxy_normals：法线只估计一次并缓存，后续检测真实复用缓存数组
# ---------------------------------------------------------------------------
class TestProxyNormalsCache:
    def test_normals_estimated_once_and_cache_actually_used(self, monkeypatch):
        pts, cols = _make_wall()
        pointcloud = PointCloudService()
        dataset = pointcloud.register_dataset('u:cloud', pts, cols, metadata={})
        assert dataset.proxy_normals is None

        viewport = SimpleNamespace(
            get_cloud_data=lambda name: {'dataset_id': 'u:cloud'})
        svc = FacadeDetectionService(
            viewport=viewport, pointcloud_service=pointcloud, index_service=None)

        seen = []
        real_detect = fds_mod.detect_facades_adaptive
        def spy_detect(geo, **kwargs):
            normals = np.asarray(geo.normals) if geo.has_normals() else None
            seen.append(normals)
            return real_detect(geo, **kwargs)
        monkeypatch.setattr(fds_mod, 'detect_facades_adaptive', spy_detect)

        svc.detect('cloud')
        # 第一次检测：服务级法线估计发生，缓存写入且为单位法线（真实计算）
        assert dataset.proxy_normals is not None
        cached = dataset.proxy_normals
        assert cached.shape == (len(dataset.proxy_points), 3)
        assert cached.dtype == np.float32
        norms = np.linalg.norm(cached, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        assert seen and seen[0] is not None
        assert np.allclose(seen[0], cached, atol=1e-5)

        # 用哨兵数组替换缓存：若第二次检测用的是缓存，算法收到的必须是哨兵
        sentinel = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                           (len(cached), 1))
        dataset.proxy_normals = sentinel
        svc.detect('cloud')
        assert len(seen) == 2 and seen[1] is not None
        assert np.array_equal(seen[1], sentinel)  # 证明复用缓存而非重新估计

    def test_estimate_normals_benchmark_is_real_work(self):
        """基准：被缓存省掉的 estimate_normals 在真实机器上有可观耗时。"""
        pts, _ = _make_wall(n_side=200)  # 40k 点
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        t0 = time.perf_counter()
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        elapsed = time.perf_counter() - t0
        normals = np.asarray(cloud.normals)
        assert normals.shape == (len(pts), 3)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
        print(f'\n[perf] estimate_normals 40k pts = {elapsed*1e3:.2f}ms')
        assert elapsed > 1e-3  # 若接近 0 说明什么都没算，测试即失效


# ---------------------------------------------------------------------------
# 3. release_station_domain：上传链 source 资产真正从内存释放
# ---------------------------------------------------------------------------
class TestReleaseStationDomain:
    def test_upload_chain_source_asset_memory_freed(self):
        n_points = 1_000_000  # ~24MB，释放效果可测
        pts = np.random.default_rng(2).random((n_points, 3), dtype=np.float32)
        cols = np.random.default_rng(3).random((n_points, 3), dtype=np.float32)

        pointcloud = PointCloudService()
        source_id = 'u:source:scan'  # 上传链键（不含站点 id，旧实现永不匹配）
        pointcloud.register_source_asset(source_id, pts, cols, {})
        dataset = pointcloud.register_dataset(
            'u:1', pts[:1000], cols[:1000], metadata={'source_id': source_id})

        asset = pointcloud.get_source_asset(source_id)
        assert asset['points'].nbytes == pts.nbytes  # 释放前真实驻留
        pts_ref = weakref.ref(asset['points'])
        cols_ref = weakref.ref(asset['colors'])
        assert pts_ref() is not None

        pointcloud.release_station_domain(1)

        assert 'u:1' not in pointcloud.datasets
        assert source_id not in pointcloud.source_assets
        del pts, cols, asset, dataset
        gc.collect()
        # 内存证明：弱引用失效 = 数组对象真正被回收，而非仅摘除键名
        assert pts_ref() is None
        assert cols_ref() is None


# ---------------------------------------------------------------------------
# 4. validate_asset：mtime 未变跳过全量哈希，mtime 变化必须重新哈希
# ---------------------------------------------------------------------------
class TestValidateAssetMtimeShortcircuit:
    def test_unchanged_file_skips_sha256_changed_file_rehashes(
            self, tmp_path, monkeypatch):
        payload = os.urandom(16 * 1024 * 1024)  # 16MB，哈希耗时真实可测
        target = tmp_path / 'scan.ply'
        target.write_bytes(payload)
        real_sha = hashlib.sha256(payload).hexdigest()
        stat = target.stat()

        hash_calls = []
        real_sha256 = file_repo_mod._sha256
        def counting_sha256(path):
            t0 = time.perf_counter()
            digest = real_sha256(path)
            hash_calls.append((str(path), digest, time.perf_counter() - t0))
            return digest
        monkeypatch.setattr(file_repo_mod, '_sha256', counting_sha256)

        asset = SimpleNamespace(
            path=str(target), size_bytes=stat.st_size, sha256=real_sha,
            meta_json={'mtime_ns': stat.st_mtime_ns})

        # mtime 命中：一次哈希都不应发生
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert hash_calls == []

        # 无 mtime 记录：回退到真实哈希，且哈希值必须等于真实内容摘要
        asset.meta_json = {}
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert len(hash_calls) == 1
        assert hash_calls[0][1] == real_sha  # 机器真的读了 16MB 并算出正确摘要
        assert hash_calls[0][2] > 0
        assert asset.meta_json['mtime_ns'] == stat.st_mtime_ns  # 哈希后回写 mtime

        # 再次校验：mtime 已记录，恢复短路
        ok, _ = FileRepo.validate_asset(asset)
        assert ok and len(hash_calls) == 1

        # mtime 改变：必须重新哈希（防止短路掩盖真实修改）
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        ok, reason = FileRepo.validate_asset(asset)
        assert (ok, reason) == (True, 'ok')
        assert len(hash_calls) == 2
        print(f'\n[perf] sha256 16MB = {hash_calls[0][2]*1e3:.2f}ms, '
              f'skipped_calls_saved=2')


# ---------------------------------------------------------------------------
# 5. 渲染门控：页面不可见时 GLFW 轮询与帧提交真实停止，恢复后补一帧
# ---------------------------------------------------------------------------
class _FakeVis:
    def __init__(self):
        self.poll_events_count = 0
        self.update_renderer_count = 0

    def poll_events(self):
        self.poll_events_count += 1

    def update_renderer(self):
        self.update_renderer_count += 1


class TestRenderGate:
    def _adapter(self):
        adapter = Open3DAdapter()
        adapter.vis = _FakeVis()
        adapter._owner_thread_id = threading.get_ident()
        return adapter

    def test_hidden_page_stops_polling_visible_page_resumes(self):
        adapter = self._adapter()
        adapter.set_render_enabled(False)
        adapter._last_event_poll_time = 0.0
        time.sleep(0.04)  # 超过 30Hz 轮询间隔，若未门控必然触发 poll_events
        t0 = time.perf_counter()
        for _ in range(10):
            assert adapter.poll() is False
        t_disabled = time.perf_counter() - t0
        assert adapter.vis.poll_events_count == 0      # GLFW 轮询真实停止
        assert adapter.vis.update_renderer_count == 0  # 帧提交真实停止

        adapter.set_render_enabled(True)  # 恢复时应挂起一次刷新
        assert adapter._render_pending is True
        adapter._last_event_poll_time = 0.0
        adapter._last_render_time = 0.0
        assert adapter.poll() is True
        assert adapter.vis.poll_events_count == 1
        assert adapter.vis.update_renderer_count == 1
        assert adapter._render_pending is False  # 恢复帧被真实提交而非丢弃

        # 节流未被破坏：30Hz 间隔内重复 poll 不再触发轮询
        assert adapter.poll() is False
        assert adapter.vis.poll_events_count == 1
        print(f'\n[perf] 10x gated poll = {t_disabled*1e3:.3f}ms, '
              f'glfw_calls_while_hidden=0')
