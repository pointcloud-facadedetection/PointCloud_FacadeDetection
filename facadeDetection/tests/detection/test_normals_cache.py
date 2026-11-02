"""检测法线缓存的实效验证：代理点云法线只估计一次并缓存，后续检测真实复用。

断言三类事实：估计次数（机制）、法线数组内容（数据）、估计确有可测耗时（时间）。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from services.facade.facade_detection_service import FacadeDetectionService
import services.facade.facade_detection_service as fds_mod
from services.pointcloud_service import PointCloudService


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
# proxy_normals：法线只估计一次并缓存，后续检测真实复用缓存数组
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
