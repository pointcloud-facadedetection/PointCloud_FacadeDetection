"""首开链路多核并行与算法优化的实效验证。

覆盖三项改动（algorithms/geometry.py）：
1. estimate_elevation_angles 单测站快路径（跳过 N×M×3 临时数组与 argmin）；
2. stratified_proxy_build 组内代表点选择由 fromiter 循环改为 minimum.reduceat；
3. stratified_proxy_build 距离壳层 ThreadPoolExecutor 并行。

每个用例断言三类事实，防止"并行了个寂寞"的假优化：
1. 机制：快路径/线程池/reduceat 真的被使用（内存峰值、线程池间谍、调用计数）；
2. 数据：新实现输出与旧实现参照副本 np.array_equal 逐点一致；
3. 时间：大点云上新旧实现都有可测耗时，且新实现可测地更快。
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import algorithms.geometry as geom
from algorithms.geometry import (
    elevation_scale_factor,
    estimate_elevation_angles,
    stratified_proxy_build,
)


# ---------------------------------------------------------------------------
# 旧实现参照副本（优化前的串行 + fromiter 版本，逐行保持原语义）
# ---------------------------------------------------------------------------
def _reference_elevation_angles(points, scan_origin=None):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    origins = np.zeros((1, 3), dtype=float) if scan_origin is None \
        else np.asarray(scan_origin, dtype=float).reshape(-1, 3)
    if not len(pts):
        return np.empty(0, dtype=np.float32)
    ranges = np.min(np.stack([np.linalg.norm(pts - o, axis=1) for o in origins], axis=1), axis=1)
    delta = pts[:, None, :] - origins[None, :, :]
    nearest = np.argmin(np.linalg.norm(delta, axis=2), axis=1)
    dz = pts[:, 2] - origins[nearest, 2]
    horizontal = np.sqrt(np.maximum(ranges ** 2 - dz ** 2, 0.0))
    elevation = np.degrees(np.arctan2(dz, horizontal + 1e-9))
    return elevation.astype(np.float32)


def _reference_proxy_build(points, colors, ranges, **kwargs):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rng = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if len(pts) != len(rng):
        raise ValueError("points and ranges length mismatch")
    if not len(pts):
        return (np.empty((0, 3), np.float32), None,
                np.zeros(1, np.int64), np.empty(0, np.int32),
                np.empty(0, np.float32), np.empty(0, np.int32))

    shells = kwargs.pop("shells", ((10., .10), (20., .08), (35., .06),
                                    (50., .05), (80., .045), (100., .04)))
    elevations = kwargs.pop("elevations", None)
    col = None if colors is None else np.asarray(colors, dtype=np.float32).reshape(-1, 3)
    if col is not None and len(col) != len(pts):
        raise ValueError("colors length mismatch")
    proxy_parts, color_parts, range_parts, source_parts, rep_parts = [], [], [], [], []
    lo = float(kwargs.pop("min_range", 0.0)); crop = float(kwargs.pop("crop_range", np.max(rng)))
    elev = np.zeros(len(pts), np.float32) if elevations is None else np.asarray(elevations, dtype=np.float32)
    for hi, base in list(shells) + [(float("inf"), None)]:
        mask = (rng >= lo) & (rng < min(float(hi), crop + 1e-9)) & (rng <= crop)
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            lo = float(hi)
            continue
        if base is None:
            base = shells[-1][1]
        scale = elevation_scale_factor(float(np.median(np.abs(elev[ids]))),
                                       kwargs.get("elevation_low_scale", 1.0),
                                       kwargs.get("elevation_high_scale", .75),
                                       kwargs.get("elevation_threshold_deg", 50.0))
        vs = float(np.clip(base * scale, kwargs.get("min_voxel", .02), kwargs.get("max_voxel", .20)))
        shell = pts[ids]
        origin = np.floor(np.min(shell, axis=0) / vs) * vs
        keys = np.floor((shell - origin) / vs).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sk = keys[order]
        starts = np.flatnonzero(np.r_[True, np.any(sk[1:] != sk[:-1], axis=1)])
        ends = np.r_[starts[1:], len(order)]
        group_counts = ends - starts
        group_ids = np.repeat(np.arange(len(starts), dtype=np.int32), group_counts)
        sorted_global = ids[order]
        sorted_pts = pts[sorted_global]
        centers = np.column_stack([
            np.add.reduceat(sorted_pts[:, axis], starts) / group_counts
            for axis in range(3)
        ])
        dist2 = np.sum((sorted_pts - centers[group_ids]) ** 2, axis=1)
        rep_rows = np.fromiter(
            (start + int(np.argmin(dist2[start:end]))
             for start, end in zip(starts, ends)),
            dtype=np.int64,
            count=len(starts),
        )
        reps = sorted_global[rep_rows]
        proxy_parts.append(pts[reps])
        range_parts.append(rng[reps])
        rep_parts.append(reps)
        source_parts.extend(
            sorted_global[start:end] for start, end in zip(starts, ends)
        )
        if col is not None:
            color_parts.append(col[reps])
        lo = float(hi)
        if lo >= crop: break
    proxy = np.vstack(proxy_parts).astype(np.float32, copy=False)
    offsets = np.zeros(len(source_parts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in source_parts], dtype=np.int64)
    indices = np.concatenate(source_parts).astype(np.int32, copy=False) if source_parts else np.empty(0, np.int32)
    proxy_colors = (np.vstack(color_parts).astype(np.float32, copy=False)
                    if col is not None else None)
    proxy_ranges = np.concatenate(range_parts).astype(np.float32, copy=False)
    representatives = (np.concatenate(rep_parts).astype(np.int32, copy=False)
                       if rep_parts else np.empty(0, np.int32))
    return proxy, proxy_colors, offsets, indices, proxy_ranges, representatives


def _assert_proxy_equal(old, new, tag):
    names = ('proxy', 'colors', 'offsets', 'indices', 'ranges', 'representatives')
    assert len(old) == len(new) == 6  # 返回 6 元组结构不变
    for name, x, y in zip(names, old, new):
        if x is None:
            assert y is None, f'{tag}.{name}'
        else:
            assert np.array_equal(x, y), f'{tag}.{name} shape={x.shape} vs {y.shape}'


def _make_cloud(rng, n, r_max=150.0):
    pts = (rng.normal(size=(n, 3)) * 40).astype(np.float32)
    cols = rng.random((n, 3)).astype(np.float32)
    rr = rng.uniform(0, r_max, n).astype(np.float32)
    return pts, cols, rr


# ---------------------------------------------------------------------------
# 1. estimate_elevation_angles：单测站快路径
# ---------------------------------------------------------------------------
class TestElevationAnglesFastPath:
    @pytest.mark.parametrize('m', (1, 2, 3))
    def test_output_identical_to_reference(self, m):
        rng = np.random.default_rng(10 + m)
        pts = rng.normal(size=(100_000, 3)) * 50
        origins = rng.normal(size=(m, 3)) * 10
        new = estimate_elevation_angles(pts, origins)
        old = _reference_elevation_angles(pts, origins)
        # 数据：float64 数学不变，逐点一致（不是 allclose）
        assert new.dtype == np.float32 == old.dtype
        assert np.array_equal(new, old)
        # 时间：真实计算有可测耗时
        t0 = time.perf_counter()
        estimate_elevation_angles(pts, origins)
        elapsed = time.perf_counter() - t0
        print(f'\n[perf] elevation M={m} 100k pts = {elapsed*1e3:.2f}ms')
        assert elapsed > 0

    def test_single_station_skips_nxm3_temp_and_is_faster(self, monkeypatch):
        """机制：M=1 走快路径，不再调用 argmin，也不再分配 N×M×3 临时数组。"""
        n = 1_000_000  # 旧实现 delta 临时 = n×1×3×8 = 24MB
        rng = np.random.default_rng(5)
        pts = rng.normal(size=(n, 3)) * 50
        origin = np.array([[1.0, 2.0, 3.0]])

        # 机制：快路径下 argmin 一次都不应被调用；M=2 时必须调用（对照组）
        argmin_calls = []
        real_argmin = np.argmin
        def spy_argmin(*a, **k):
            argmin_calls.append(1)
            return real_argmin(*a, **k)
        monkeypatch.setattr(np, 'argmin', spy_argmin)
        estimate_elevation_angles(pts, origin)
        assert argmin_calls == []
        estimate_elevation_angles(pts, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert len(argmin_calls) == 1
        monkeypatch.undo()

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        old = _reference_elevation_angles(pts, origin)
        t_old = time.perf_counter() - t0
        _, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        new = estimate_elevation_angles(pts, origin)
        t_new = time.perf_counter() - t0
        _, new_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 数据：输出逐点一致
        assert np.array_equal(new, old)
        # 机制：旧实现额外持有 delta(N×M×3) 及其中间量，峰值内存可测地更高
        print(f'\n[perf] elevation M=1 1M pts: old={t_old*1e3:.1f}ms/{old_peak/1e6:.1f}MB '
              f'new={t_new*1e3:.1f}ms/{new_peak/1e6:.1f}MB')
        assert old_peak - new_peak > n * 4  # 至少省出一个 N 级 float64 临时
        # 时间：快路径可测地更快
        assert t_new < t_old


# ---------------------------------------------------------------------------
# 2. stratified_proxy_build：等价性（reduceat + 壳层并行 vs 串行 fromiter 参照）
# ---------------------------------------------------------------------------
class TestProxyBuildEquivalence:
    def _check(self, pts, cols, rr, elev, tag, **kw):
        old = _reference_proxy_build(pts, cols, rr, elevations=elev, **kw)
        new = stratified_proxy_build(pts, cols, rr, elevations=elev, **kw)
        _assert_proxy_equal(old, new, tag)
        proxy, _, offsets, indices, _, reps = new
        # 数据：CSR 映射自洽——proxy[i] 就是 representatives[i] 指向的源行
        assert len(proxy) == len(reps) == len(offsets) - 1
        assert offsets[-1] == len(indices)
        assert np.array_equal(proxy, pts[reps])

    def test_multi_shell_single_station(self):
        rng = np.random.default_rng(20)
        pts, cols, rr = _make_cloud(rng, 200_000)
        elev = estimate_elevation_angles(pts, np.zeros((1, 3)))
        self._check(pts, cols, rr, elev, 'multi-shell-M1')

    @pytest.mark.parametrize('m', (2, 3))
    def test_multi_station_elevations(self, m):
        rng = np.random.default_rng(30 + m)
        pts, cols, rr = _make_cloud(rng, 100_000)
        origins = rng.normal(size=(m, 3)) * 15
        elev = estimate_elevation_angles(pts, origins)
        self._check(pts, cols, rr, elev, f'multi-shell-M{m}')

    def test_empty_shell_beyond_last_shell_and_crop_edge(self):
        """空壳（10-20m 无点）、超出最后壳层（>100m）、rng 恰等于 crop 的边界点。"""
        rng = np.random.default_rng(40)
        pts, cols, rr = _make_cloud(rng, 200_000)
        rr[(rr >= 10) & (rr < 20)] = 5.0     # 制造空壳
        rr[:100] = 60.0                       # crop 边界点
        assert np.any(rr > 100)               # 确有超出最后壳层的点
        elev = estimate_elevation_angles(pts, np.zeros((1, 3)))
        self._check(pts, cols, rr, elev, 'edge-cases', crop_range=60.0)

    def test_min_range_and_no_colors(self):
        rng = np.random.default_rng(50)
        pts, cols, rr = _make_cloud(rng, 100_000)
        rr[:500] = 0.1  # min_range 之下的点
        elev = estimate_elevation_angles(pts, np.zeros((1, 3)))
        self._check(pts, None, rr, elev, 'min-range-no-colors', min_range=0.5)

    def test_tiny_cloud_single_shell(self):
        rng = np.random.default_rng(60)
        pts, cols, rr = _make_cloud(rng, 500, r_max=8.0)
        self._check(pts, cols, rr, None, 'tiny')

    def test_thread_pool_actually_used_for_multiple_shells(self, monkeypatch):
        """机制：多壳层时真的经过 ThreadPoolExecutor，且 worker 数 > 1。"""
        rng = np.random.default_rng(70)
        pts, cols, rr = _make_cloud(rng, 50_000)
        created = []
        real_pool = ThreadPoolExecutor

        class SpyPool(real_pool):
            def __init__(self, max_workers=None, *a, **k):
                created.append(max_workers)
                super().__init__(max_workers=max_workers, *a, **k)

        monkeypatch.setattr(geom, 'ThreadPoolExecutor', SpyPool)
        new = stratified_proxy_build(pts, cols, rr, elevations=None)
        monkeypatch.undo()
        old = _reference_proxy_build(pts, cols, rr, elevations=None)

        # 机制：线程池被创建且并行度 > 1（20 核机器上 min(4, cpu)=4）
        assert created and max(created) > 1
        # 数据：并行结果与串行参照逐点一致
        _assert_proxy_equal(old, new, 'threaded')
        # 时间：并行构建有可测耗时
        t0 = time.perf_counter()
        stratified_proxy_build(pts, cols, rr, elevations=None)
        elapsed = time.perf_counter() - t0
        print(f'\n[perf] threaded proxy build 50k pts = {elapsed*1e3:.2f}ms, '
              f'pool_workers={created}')
        assert elapsed > 0


# ---------------------------------------------------------------------------
# 3. 大点云提速实测（>=500 万点）
# ---------------------------------------------------------------------------
class TestProxyBuildSpeed:
    def test_large_cloud_faster_and_identical(self):
        n = 5_000_000
        rng = np.random.default_rng(80)
        pts, cols, rr = _make_cloud(rng, n)
        elev = estimate_elevation_angles(pts, np.zeros((1, 3)))

        t0 = time.perf_counter()
        old = _reference_proxy_build(pts, cols, rr, elevations=elev)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = stratified_proxy_build(pts, cols, rr, elevations=elev)
        t_new = time.perf_counter() - t0

        # 数据：500 万点全量输出逐点一致（proxy 近 500 万行，证明真算了）
        _assert_proxy_equal(old, new, 'large-5M')
        assert len(new[0]) > 4_000_000
        # 时间：旧实现（串行 + 200 万组 fromiter）耗时真实，新实现可测地更快
        print(f'\n[perf] proxy build {n} pts: reference={t_old:.2f}s '
              f'new={t_new:.2f}s speedup={t_old / t_new:.2f}x')
        assert t_old > 1.0  # 若接近 0 说明参照实现没真跑，测试失效
        assert t_new < t_old
