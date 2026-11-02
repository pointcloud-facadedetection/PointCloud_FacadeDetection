"""站点数据集复用与释放的实效验证。

每个用例同时断言三类事实，防止"全部跳过、计时归零"的假优化：
1. 机制：被省掉的昂贵操作确实没有再发生（读取计数）；
2. 数据：内存中的数组真实存在且内容正确（shape、nbytes、弱引用释放）；
3. 时间：真实机器操作有可测耗时，复用路径可测地更快。

仅使用 tmp_path 下的普通文件，不创建真实项目目录或数据库。
"""
from __future__ import annotations

import gc
import time
import weakref
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService


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
# 指纹门控：上传链注册的 dataset 必须被复用，同一文件不得二次读取
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
        # _load 命中 PLY memmap 快读，返回 float32；总量一致证明真的读了全部点
        assert base_pts.dtype == np.float32 and base_cols.dtype == np.float32
        assert base_pts.nbytes + base_cols.nbytes == raw_nbytes
        assert np.allclose(base_pts, pts, rtol=1e-6, atol=1e-7)
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
# release_station_domain：上传链 source 资产真正从内存释放
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
