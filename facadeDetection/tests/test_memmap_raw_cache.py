"""raw 缓存 .npy memmap 按需换页与 PLY 免解析快读的实效验证。

每个用例同时断言三类事实，防止"全部跳过、计时归零"的假优化：
1. 机制：np.load(mmap_mode='r') / np.clip / self._load 的调用计数与参数
   符合命中、回退或保映射预期；
2. 数据：memmap 恢复的数组与首轮逐点一致（np.array_equal）、快读结果与
   Open3D np.allclose、三件套真实写盘；
3. 内存：source asset 的 points/colors 保持只读映射（不可写、不拥有数据）、
   300MB 映射未触碰时 RSS 增量远小于数组体积、值域内颜色不触发整份拷贝。

Storage.project_root 打桩到 tmp_path，不创建真实项目目录或数据库。
"""
from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import psutil
import pytest

from config.storage import Storage
from services import proxy_cache
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from services.pointcloud_service import PointCloudService
from services.pointcloud_station_service import PointCloudStationService
from utils.ply_fast_reader import read_ply_fast


def _write_o3d_ply(path, points, colors=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    assert o3d.io.write_point_cloud(str(path), cloud)
    return path


def _write_intensity_ply(path, points, colors01, intensity):
    """bllygg01 同构记录：3×float32 xyz + 3×uchar rgb + 1×float32 intensity。"""
    n = len(points)
    header = (b'ply\nformat binary_little_endian 1.0\n'
              b'element vertex ' + str(n).encode() + b'\n'
              b'property float x\nproperty float y\nproperty float z\n'
              b'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              b'property float intensity\nend_header\n')
    records = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                 ('intensity', '<f4')])
    records['x'], records['y'], records['z'] = (points[:, 0], points[:, 1],
                                                points[:, 2])
    rgb = np.round(colors01 * 255).astype(np.uint8)
    records['red'], records['green'], records['blue'] = (rgb[:, 0], rgb[:, 1],
                                                         rgb[:, 2])
    records['intensity'] = intensity
    with open(path, 'wb') as stream:
        stream.write(header)
        stream.write(records.tobytes())
    return path


def _make_scan(n_side=320, spacing=0.01, seed=7):
    """约 10 万点的仿真墙体（x≈0 平面），float32 保证 PLY 往返精确相等。"""
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


def _write_dist(path, n_points, seed=11):
    """真实二进制 .dist：float32 距离值，落在 dist_reader 的 mmap 分支。"""
    rng = np.random.default_rng(seed)
    values = rng.uniform(2.0, 60.0, n_points).astype('<f4')
    values.tofile(path)
    assert path.stat().st_size == n_points * 4
    return path


def _real_fingerprint(path):
    data = Path(path).read_bytes()
    return (str(path), hashlib.sha256(data).hexdigest(), len(data))


def _is_readonly_mapping(arr):
    """raw 缓存命中的标志：数组不可写且不拥有数据（页由 OS 映射按需换入）。"""
    return isinstance(arr, np.memmap) or (
        not arr.flags.writeable and not arr.flags.owndata)


class TestPlyFastReader:

    def test_matches_open3d_on_o3d_written_ply(self, tmp_path):
        pts, cols = _make_scan()
        ply = _write_o3d_ply(tmp_path / 'scan.ply', pts, cols)
        t0 = time.perf_counter()
        fast = read_ply_fast(ply)
        t_fast = time.perf_counter() - t0
        # 机制：o3d 写的 binary_little_endian 双精度 PLY 必须命中快读
        assert fast is not None
        points, colors = fast
        t0 = time.perf_counter()
        cloud = o3d.io.read_point_cloud(str(ply))
        t_o3d = time.perf_counter() - t0
        # 数据：与 Open3D 路径逐点一致（uchar rgb /255 归一化语义相同）
        assert points.dtype == np.float32 and points.shape == (len(pts), 3)
        assert colors.dtype == np.float32 and colors.shape == (len(pts), 3)
        assert np.allclose(points, np.asarray(cloud.points),
                           rtol=1e-6, atol=1e-7)
        assert np.allclose(colors, np.asarray(cloud.colors),
                           rtol=1e-5, atol=1e-6)
        assert np.array_equal(points, pts)  # float32 源往返精确相等
        # 内存：快读结果是自有 C 序内存数组，不把文件映射泄漏给调用方
        # （F 序会让下游 ascontiguousarray 整份拷贝，破坏 memmap 保映射链路）
        assert points.flags.owndata and colors.flags.owndata
        assert points.flags.c_contiguous and colors.flags.c_contiguous
        print(f'\n[perf] o3d_read={t_o3d*1e3:.2f}ms '
              f'fast_read={t_fast*1e3:.2f}ms points={len(points)}')
        assert t_fast > 0 and t_o3d > 0

    def test_matches_open3d_on_intensity_ply(self, tmp_path):
        pts, cols = _make_scan(n_side=200)
        rng = np.random.default_rng(3)
        cols = rng.uniform(0.0, 1.0, cols.shape).astype(np.float32)
        intensity = rng.uniform(0, 2048, len(pts)).astype(np.float32)
        ply = _write_intensity_ply(tmp_path / 'bllygg01_like.ply', pts, cols,
                                   intensity)
        t0 = time.perf_counter()
        fast = read_ply_fast(ply)
        t_fast = time.perf_counter() - t0
        # 机制：16+ 字节混合记录（含 intensity）必须命中快读
        assert fast is not None
        points, colors = fast
        t0 = time.perf_counter()
        cloud = o3d.io.read_point_cloud(str(ply))
        t_o3d = time.perf_counter() - t0
        # 数据：逐点一致（颜色以 uchar 量化后的真值为准）
        assert np.array_equal(points, pts)
        expected_colors = (np.round(cols * 255).astype(np.uint8)
                           .astype(np.float32) / np.float32(255.0))
        assert np.allclose(colors, expected_colors, rtol=1e-5, atol=1e-6)
        assert np.allclose(points, np.asarray(cloud.points),
                           rtol=1e-6, atol=1e-7)
        assert np.allclose(colors, np.asarray(cloud.colors),
                           rtol=1e-5, atol=1e-6)
        print(f'\n[perf] intensity_ply o3d_read={t_o3d*1e3:.2f}ms '
              f'fast_read={t_fast*1e3:.2f}ms points={len(points)}')

    def test_no_color_ply_returns_none_colors(self, tmp_path):
        pts, _ = _make_scan(n_side=120)
        ply = _write_o3d_ply(tmp_path / 'plain.ply', pts)
        fast = read_ply_fast(ply)
        # 机制：命中快读；数据：无颜色返回 None，与 Open3D has_colors=False 一致
        assert fast is not None
        points, colors = fast
        assert colors is None
        assert np.array_equal(points, pts)

    def test_rejects_unsupported_and_falls_back(self, tmp_path):
        pts, cols = _make_scan(n_side=60)
        n = len(pts)
        # ASCII PLY → None（调用方回退 Open3D）
        ascii_ply = tmp_path / 'ascii.ply'
        body = '\n'.join(f'{p[0]} {p[1]} {p[2]} 128 128 128' for p in pts)
        ascii_ply.write_text(
            'ply\nformat ascii 1.0\nelement vertex ' + str(n) + '\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n' + body + '\n', encoding='utf-8')
        assert read_ply_fast(ascii_ply) is None
        # 大端 / 多 element+list 属性 / 畸形头 / 非 PLY → 全部 None
        big_endian = tmp_path / 'be.ply'
        big_endian.write_bytes(
            b'ply\nformat binary_big_endian 1.0\nelement vertex 1\n'
            b'property float x\nproperty float y\nproperty float z\n'
            b'end_header\n' + b'\x00' * 12)
        assert read_ply_fast(big_endian) is None
        with_faces = tmp_path / 'mesh.ply'
        with_faces.write_bytes(
            b'ply\nformat binary_little_endian 1.0\nelement vertex 1\n'
            b'property float x\nproperty float y\nproperty float z\n'
            b'element face 1\nproperty list uchar int vertex_indices\n'
            b'end_header\n' + b'\x00' * 32)
        assert read_ply_fast(with_faces) is None
        garbage = tmp_path / 'garbage.ply'
        garbage.write_bytes(b'\x00\xffnot a ply at all' * 16)
        assert read_ply_fast(garbage) is None
        assert read_ply_fast(tmp_path / 'missing.ply') is None
        # 机制：回退路径行为与现状完全一致 —— Open3D 仍能读出 ASCII PLY
        cloud = o3d.io.read_point_cloud(str(ascii_ply))
        assert len(cloud.points) == n
        assert np.allclose(np.asarray(cloud.points), pts, rtol=1e-5, atol=1e-7)


class _Harness:
    """装配真实文件 + 计数代理 + 打桩 Storage/Repo 的 memmap 测试环境。"""

    def __init__(self, tmp_path, monkeypatch, with_colors=True):
        pts, cols = _make_scan()
        self.n_points = len(pts)
        self.ply = _write_o3d_ply(tmp_path / 'scan.ply', pts,
                                  cols if with_colors else None)
        self.dist = _write_dist(tmp_path / 'scan.dist', self.n_points)
        self.project_root = tmp_path / 'proj'
        self.station = SimpleNamespace(
            id=7, source_path=str(self.ply), display_name='s7')

        monkeypatch.setattr(Storage, 'project_root',
                            classmethod(lambda cls, u: self.project_root))
        monkeypatch.setattr(
            PointCloudStationRepo, 'get_asset_fingerprint',
            staticmethod(lambda u, s: _real_fingerprint(self.ply)))
        monkeypatch.setattr(
            PointCloudStationRepo, 'get_denoise_state',
            staticmethod(lambda u, s: None))

        self.ply_load_calls = []
        real_load = PointCloudStationService._load
        def counting_load(service, path):
            self.ply_load_calls.append(time.perf_counter())
            return real_load(service, path)
        monkeypatch.setattr(PointCloudStationService, '_load', counting_load)

        self.np_load_calls = []
        real_np_load = np.load
        def counting_np_load(*args, **kwargs):
            self.np_load_calls.append((args[0] if args else None,
                                       dict(kwargs)))
            return real_np_load(*args, **kwargs)
        monkeypatch.setattr(np, 'load', counting_np_load)

    def raw_paths(self):
        return proxy_cache.raw_cache_paths('u', 7)

    def mmap_loads(self):
        """raw 三件套上以 mmap_mode='r' 发生的真实 np.load 调用。"""
        return [(p, kw) for p, kw in self.np_load_calls
                if kw.get('mmap_mode') == 'r']

    def new_service(self):
        """全新 service + 全新 PointCloudService，模拟重开项目后的空运行时。"""
        return PointCloudStationService(
            render_service=SimpleNamespace(), project_uuid='u',
            pointcloud_service=PointCloudService())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return _Harness(tmp_path, monkeypatch)


class TestMemmapRawCache:

    def test_restore_keeps_readonly_mapping(self, harness):
        svc1 = harness.new_service()
        svc1._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 1
        first_raw = np.asarray(
            svc1.pointcloud.get_source_asset('u:7:source')['points'],
            dtype=np.float32)
        first_colors = np.asarray(
            svc1.pointcloud.get_source_asset('u:7:source')['colors'],
            dtype=np.float32)

        svc2 = harness.new_service()
        svc2._load_proxy_domain(harness.station)
        # 机制：重开轮未再读 PLY；np.load 以 mmap_mode='r' 真实调用两次
        assert len(harness.ply_load_calls) == 1
        mmap_loads = harness.mmap_loads()
        assert len(mmap_loads) == 2
        assert {tuple(str(p).rsplit('.', 2)[-2:]) for p, _ in mmap_loads} == \
            {('points', 'npy'), ('colors', 'npy')}
        # 内存：source asset 两个数组都保持只读映射，未被物化
        asset = svc2.pointcloud.get_source_asset('u:7:source')
        assert _is_readonly_mapping(asset['points'])
        assert _is_readonly_mapping(asset['colors'])
        dataset = svc2.pointcloud.get_dataset('u:7')
        assert not dataset.index.source_points.flags.writeable
        # 数据：与首轮逐点一致
        assert np.array_equal(np.asarray(asset['points']), first_raw)
        assert np.array_equal(np.asarray(asset['colors']), first_colors)

    def test_clip_skipped_for_in_range_memmap_colors(self, harness,
                                                     monkeypatch):
        svc1 = harness.new_service()
        svc1._load_proxy_domain(harness.station)

        big_clip_calls = []
        real_clip = np.clip
        def counting_clip(a, *args, **kwargs):
            if isinstance(a, np.ndarray) and a.nbytes > 1_000_000:
                big_clip_calls.append(a.nbytes)
            return real_clip(a, *args, **kwargs)
        monkeypatch.setattr(np, 'clip', counting_clip)

        svc2 = harness.new_service()
        svc2._load_proxy_domain(harness.station)
        # 机制：memmap 恢复路径对 raw 尺寸数组 0 次 np.clip（恒等拷贝被跳过）
        assert big_clip_calls == []
        # 内存：颜色对象身份保留映射（不可写、不拥有数据）
        asset = svc2.pointcloud.get_source_asset('u:7:source')
        assert _is_readonly_mapping(asset['colors'])
        # 数据：值域内颜色与首轮逐点一致
        first_colors = np.asarray(
            svc1.pointcloud.get_source_asset('u:7:source')['colors'],
            dtype=np.float32)
        assert np.array_equal(np.asarray(asset['colors']), first_colors)

    def test_out_of_range_colors_still_clipped(self, harness):
        # 越界颜色必须仍然拷贝+裁剪，语义与旧路径一致
        pointcloud = PointCloudService()
        pts = np.zeros((4, 3), dtype=np.float32)
        cols = np.array([[0.5, 1.2, -0.3]] * 4, dtype=np.float32)
        pointcloud.register_source_asset('src', pts, cols)
        stored = pointcloud.get_source_asset('src')['colors']
        assert stored.flags.writeable  # 确实发生了拷贝
        assert np.allclose(stored, [[0.5, 1.0, 0.0]] * 4)

    def test_no_color_station_roundtrip(self, tmp_path, monkeypatch):
        harness = _Harness(tmp_path, monkeypatch, with_colors=False)
        svc1 = harness.new_service()
        svc1._load_proxy_domain(harness.station)
        _, colors_path, _ = harness.raw_paths()
        # 数据：无颜色站点真实落盘 (0,3) 标记
        marker = np.load(colors_path)
        assert marker.shape == (0, 3)

        svc2 = harness.new_service()
        svc2._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 1      # 机制：缓存命中
        assert len(harness.mmap_loads()) == 2
        asset = svc2.pointcloud.get_source_asset('u:7:source')
        assert asset['colors'] is None               # 数据：还原为 None
        assert _is_readonly_mapping(asset['points'])

    def test_corrupt_meta_json_falls_back_and_rewrites(self, harness):
        svc1 = harness.new_service()
        svc1._load_proxy_domain(harness.station)
        _, _, meta_path = harness.raw_paths()
        meta_path.write_bytes(b'\x00\xffgarbage')
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is None

        svc2 = harness.new_service()
        ds2 = svc2._load_proxy_domain(harness.station)  # 不得崩溃
        # 机制：回退真实读盘；proxy 缓存仍命中不重建；raw 缓存被覆写恢复
        assert len(harness.ply_load_calls) == 2
        assert ds2.metadata['proxy_cache'] == 'restored'
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is not None

    def test_missing_colors_file_falls_back_and_rewrites(self, harness):
        svc1 = harness.new_service()
        svc1._load_proxy_domain(harness.station)
        _, colors_path, _ = harness.raw_paths()
        colors_path.unlink()
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is None

        svc2 = harness.new_service()
        svc2._load_proxy_domain(harness.station)
        assert len(harness.ply_load_calls) == 2      # 机制：回退真实读盘
        assert colors_path.exists()                  # 数据：三件套被覆写恢复
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is not None

    def test_legacy_npz_is_migrated(self, harness):
        legacy = (harness.project_root / Storage.CACHE_DIRNAME
                  / 'raw' / '7.npz')
        legacy.parent.mkdir(parents=True, exist_ok=True)
        np.savez(legacy, points=np.zeros((3, 3), np.float32))
        assert legacy.exists()
        # 旧格式绝不命中，且遇到即清理（一次性迁移）
        assert proxy_cache.load_raw_cache(
            'u', 7, _real_fingerprint(harness.ply)) is None
        assert not legacy.exists()

        svc = harness.new_service()
        svc._load_proxy_domain(harness.station)
        assert not legacy.exists()
        assert all(p.exists() for p in harness.raw_paths())


class TestMemmapRSS:

    def test_untouched_mapping_does_not_raise_rss(self, tmp_path):
        # 300MB+ 真实 .npy 落盘
        n = 300 * 1024 * 1024 // 12
        rng = np.random.default_rng(5)
        array = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
        path = tmp_path / 'big.npy'
        np.save(path, array)
        expected_sum = float(array.sum())
        head = array[:1000].copy()
        del array
        gc.collect()

        process = psutil.Process()
        rss_before = process.memory_info().rss
        mapped = np.load(path, mmap_mode='r')
        # 机制：建立的是只读映射而非整读
        assert isinstance(mapped, np.memmap)
        assert not mapped.flags.writeable
        assert mapped.nbytes >= 300 * 1024 * 1024
        rss_mapped = process.memory_info().rss
        # 内存：映射不触碰页，RSS 增量远小于数组体积（阈值 30%）
        assert rss_mapped - rss_before < 0.3 * mapped.nbytes
        # 只切一小片：换页以页为单位，RSS 仍远低于整数组
        assert np.array_equal(np.asarray(mapped[:1000]), head)
        rss_slice = process.memory_info().rss
        assert rss_slice - rss_before < 0.3 * mapped.nbytes
        # 数据：全量读取对照，证明映射背后数据真实可读
        assert abs(float(mapped.sum()) - expected_sum) < 1e-2 * n
        print(f'\n[rss] before={rss_before/1e6:.1f}MB '
              f'mapped={rss_mapped/1e6:.1f}MB slice={rss_slice/1e6:.1f}MB '
              f'array={mapped.nbytes/1e6:.1f}MB')
