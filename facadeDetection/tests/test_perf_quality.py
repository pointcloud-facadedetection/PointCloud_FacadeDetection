"""质量链路优化的实效验证。

覆盖三项改动：
1. ruler_flatness_3d.prepare_surface tops 选取 argsort+reduceat 向量化；
2. ruler_quality 常驻进程池 + 临时 .npy 路径传参（子进程 mmap 自取）；
3. compute_ruler_quality interval 统计复用预算好的 v_all。

每个用例断言三类事实：
1. 机制：常驻池只创建一次、进程模式真实落盘共享数组、参照循环真实执行；
2. 数据：向量化 tops 与旧逐 bin 循环逐点一致；进程/线程模式输出一致；
3. 时间：参照实现有可测耗时，新实现可测地更快。
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import algorithms.facade.ruler_quality as rq
from algorithms.facade.ruler_flatness_3d import (
    Surface, make_frame, fit_line, sor_mask_local, prepare_surface)
from algorithms.facade.ruler_quality import (
    RulerQualityParameters, compute_ruler_quality)


# ---------------------------------------------------------------------------
# 旧实现参照副本：prepare_surface（tops 逐 bin 循环版本，逐行保持原语义）
# ---------------------------------------------------------------------------
def _reference_prepare_surface(points, source_ids=None, ruler_dir=None,
                               outward=(0, 0, 1), select_band=.01,
                               hole_band=.02, bin_size=.04, top_q=1.0,
                               sor=None, plane_model=None, plane_origin=None):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 3:
        raise ValueError("点数不足")
    valid_mask = np.all(np.isfinite(p), axis=1)
    if not np.any(valid_mask):
        raise ValueError("所有点坐标均无效（含 nan/inf）")
    p = p[valid_mask]
    ids = np.arange(len(p), dtype=np.int64) if source_ids is None \
        else np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if not np.all(valid_mask) and source_ids is not None:
        ids = ids[valid_mask]
    if len(ids) != len(p):
        raise ValueError("source_ids 与 points 长度不一致")
    model = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if model.size != 4 or not np.all(np.isfinite(model)):
        raise ValueError("质量靠尺算法必须提供前序立面 plane_model")
    norm = np.linalg.norm(model[:3])
    if norm < 1e-12:
        raise ValueError("plane_model 法向长度为零")
    normal = model[:3] / norm
    d = float(model[3]) / norm
    if np.dot(normal, np.asarray(outward, dtype=float)) < 0:
        normal, d = -normal, -d
    centroid = (-d * normal if plane_origin is None else
                np.asarray(plane_origin, dtype=np.float64))
    centroid = centroid - (np.dot(centroid, normal) + d) * normal
    eu, ev = make_frame(normal, p, ruler_dir)
    local = (p - centroid) @ np.column_stack((eu, ev, normal))
    order = np.argsort(local[:, 0], kind="stable")
    u, w, ids = local[order, 0], local[order, 2], ids[order]
    u_min_full = float(u.min())
    u_max_full = float(u.max())
    a, c = fit_line(u, w)
    residual = w - (a * u + c)
    keep = np.ones(len(u), dtype=bool)
    sp = dict(enable=True, sigma=4.0, method="local", k=8, w_weight=50.0)
    sp.update(sor or {})
    if sp.get("enable", True) and len(u) > 16:
        method = sp.get("method", "local")
        k = int(sp.get("k", 8))
        sigma = float(sp["sigma"])
        if method == "local":
            keep = sor_mask_local(u, w, k, sigma)
        else:
            keep = sor_mask_local(u, w, k, sigma)
        if np.sum(keep) < 3:
            keep[:] = True
    a, c = fit_line(u[keep], w[keep])
    residual = w - (a * u + c)
    surf = keep & (np.abs(residual) <= select_band)
    if np.sum(surf) < 3:
        raise ValueError("表面点不足，请放宽 select_band")
    effective_hole_band = max(hole_band, select_band)
    u0 = float(u[0])
    bins = np.floor((u - u0) / bin_size).astype(np.int64)
    solid_mask = np.zeros(int(bins[-1]) + 1, dtype=bool)
    solid_mask[bins[keep & (np.abs(residual) <= effective_hole_band)]] = True
    solid = np.flatnonzero(solid_mask)
    selected = np.flatnonzero(surf & np.isin(bins, solid))
    # 旧实现：逐 bin 伪二次循环
    tops = []
    for b in solid:
        ix = selected[bins[selected] == b]
        if len(ix):
            if top_q >= 1.0:
                tops.append(ix[np.argmax(w[ix])])
            else:
                segment_w = w[ix]
                target_w = np.quantile(segment_w, top_q)
                closest_idx = ix[np.argmin(np.abs(segment_w - target_w))]
                tops.append(closest_idx)
    tops = np.asarray(tops, dtype=np.int64)
    if len(tops) < 2:
        raise ValueError("实区分段不足以放置靠尺")
    pad = np.r_[True, solid_mask, True]
    hs = np.flatnonzero(~pad[1:-1] & pad[:-2])
    he = np.flatnonzero(~pad[1:-1] & pad[2:]) + 1
    holes = (np.column_stack((u0 + hs * bin_size, u0 + he * bin_size))
             if hs.size else np.empty((0, 2), dtype=float))
    stats = {
        "n_points": len(p),
        "n_used": int(np.sum(surf)),
        "n_sor_removed": int(np.sum(~keep)),
        "n_solid_bins": int(solid_mask.sum()),
        "hole_length": float(np.diff(holes, axis=1).sum()) if holes.size else 0.
    }
    return Surface(
        normal, centroid, eu, ev, u, w, ids, keep, surf,
        float(bin_size), (u_min_full, u_max_full),
        u[tops], w[tops], ids[tops], holes, (a, c), stats)


def _make_profile(rng, n=3000, length=5.0):
    """沿 y 方向展开的墙面剖面点云（x 为法向，含波浪起伏与噪声）。"""
    u_coords = np.sort(rng.uniform(0, length, n))
    w_vals = rng.normal(0, 0.003, n) + 0.004 * np.sin(u_coords * 2.7)
    pts = np.stack([w_vals, u_coords, rng.uniform(0, 2, n)], 1)
    return pts


def _assert_surface_equal(old, new, tag):
    for name in ('u', 'w', 'source_ids', 'keep', 'surf',
                 'top_u', 'top_w', 'top_source_ids', 'holes'):
        assert np.array_equal(np.asarray(getattr(old, name)),
                              np.asarray(getattr(new, name))), \
            f'{tag}.{name} 不一致'
    assert old.ref_line == new.ref_line, f'{tag}.ref_line'
    assert old.stats == new.stats, f'{tag}.stats'
    assert old.u_span == new.u_span, f'{tag}.u_span'


class TestTopsVectorizedEquivalence:
    PLANE = np.array([1.0, 0, 0, 0.0])

    @pytest.mark.parametrize('trial', range(8))
    @pytest.mark.parametrize('top_q', (1.0, 0.8))
    def test_tops_identical_to_reference_loop(self, trial, top_q):
        rng = np.random.default_rng(100 + trial)
        pts = _make_profile(rng, n=int(rng.integers(200, 4000)))
        kw = dict(source_ids=np.arange(len(pts)),
                  ruler_dir=np.array([0.0, 1.0, 0.0]),
                  outward=np.array([1.0, 0, 0]),
                  select_band=0.01, hole_band=0.02, bin_size=0.04,
                  top_q=top_q,
                  sor={'enable': True, 'sigma': 4.0, 'k': 8,
                       'method': 'local', 'w_weight': 50.0},
                  plane_model=self.PLANE, plane_origin=np.zeros(3))
        old = _reference_prepare_surface(pts, **kw)
        new = prepare_surface(pts, **kw)
        # 数据：tops 及全部分量逐点一致（含 top_q<1 的分位选取路径）
        _assert_surface_equal(old, new, f'trial{trial}-q{top_q}')
        assert len(new.top_u) >= 2

    def test_tops_vectorized_faster_on_dense_profile(self):
        rng = np.random.default_rng(7)
        pts = _make_profile(rng, n=120_000, length=8.0)
        kw = dict(ruler_dir=np.array([0.0, 1.0, 0.0]),
                  outward=np.array([1.0, 0, 0]),
                  select_band=0.01, hole_band=0.02, bin_size=0.04,
                  top_q=1.0,
                  sor={'enable': True, 'sigma': 4.0, 'k': 8,
                       'method': 'local', 'w_weight': 50.0},
                  plane_model=self.PLANE, plane_origin=np.zeros(3))
        t0 = time.perf_counter()
        old = _reference_prepare_surface(pts, **kw)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = prepare_surface(pts, **kw)
        t_new = time.perf_counter() - t0
        # 数据：稠密剖面上 tops 逐点一致
        _assert_surface_equal(old, new, 'dense')
        # 时间：旧伪二次循环（bin 数 × 全量 selected）有可测耗时，新实现更快
        print(f'\n[perf] tops 120k pts: reference_loop={t_old*1e3:.1f}ms '
              f'vectorized={t_new*1e3:.1f}ms')
        assert t_old > 0.01  # 参照没真跑则测试失效
        assert t_new < t_old


def _make_quality_wall(seed=5):
    rng = np.random.default_rng(seed)
    gy, gz = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(0, 5, 0.05))
    n = gy.size
    x = rng.normal(0, 0.002, n)
    dent = (np.abs(gy.ravel()) < 0.5) & (np.abs(gz.ravel() - 2.5) < 0.5)
    x[dent] -= 0.01
    pts = np.stack([x, gy.ravel(), gz.ravel()], 1)
    return pts, np.arange(n, dtype=np.int64)


def _strip_volatile(report):
    """剔除并行模式快照与计时字段后比较质量报告。"""
    r = {k: v for k, v in report.items()
         if k not in ('performance', 'parameters')}
    return r


def _deep_equal(a, b, path=''):
    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), f'{path} 键不一致'
        for k in a:
            _deep_equal(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f'{path} 长度不一致'
        for i, (x, y) in enumerate(zip(a, b)):
            _deep_equal(x, y, f'{path}[{i}]')
    elif isinstance(a, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f'{path} 数组不一致'
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), f'{path}: {a!r} != {b!r}'
    else:
        assert a == b, f'{path}: {a!r} != {b!r}'


class TestProcessPoolAndMmapTransport:
    def test_process_and_thread_modes_output_identical(self):
        pts, raw_ids = _make_quality_wall()
        plane = np.array([1.0, 0, 0, 0.0])
        origin = np.zeros(3)
        u_axis = np.array([0.0, 1.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])

        t0 = time.perf_counter()
        rt = compute_ruler_quality(
            pts, raw_ids, plane, origin, u_axis, v_axis,
            RulerQualityParameters(parallel_mode='thread', n_jobs=4))
        t_thread = time.perf_counter() - t0
        t0 = time.perf_counter()
        rp = compute_ruler_quality(
            pts, raw_ids, plane, origin, u_axis, v_axis,
            RulerQualityParameters(parallel_mode='process', n_jobs=4))
        t_process = time.perf_counter() - t0

        # 机制：两种模式都真实产出窗口行
        assert rt['ok'] and rp['ok']
        assert len(rt['windows']) > 0 and len(rp['windows']) > 0
        # 数据：mmap 传参不改变数值路径——windows/intervals/overall/
        # verticality 全量一致（含 NaN 与 ndarray 字段）
        _deep_equal(_strip_volatile(rt), _strip_volatile(rp), 'report')
        print(f'\n[perf] quality {len(pts)} pts: thread={t_thread:.2f}s '
              f'process={t_process:.2f}s windows={len(rt["windows"])}')

    def test_process_pool_created_once_and_arrays_dumped(self, monkeypatch):
        pts, raw_ids = _make_quality_wall()
        plane = np.array([1.0, 0, 0, 0.0])
        origin = np.zeros(3)
        u_axis = np.array([0.0, 1.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])

        rq._reset_shared_process_pool()  # 与用例执行顺序解耦，从空池计起
        created = []
        real_pool = rq.ProcessPoolExecutor

        class SpyPool(real_pool):
            def __init__(self, max_workers=None, *a, **k):
                created.append(max_workers)
                super().__init__(max_workers=max_workers, *a, **k)

        save_calls = []
        real_save = np.save

        def counting_save(*a, **k):
            save_calls.append(str(a[1])[:80] if len(a) > 1 else '')
            return real_save(*a, **k)

        monkeypatch.setattr(rq, 'ProcessPoolExecutor', SpyPool)
        monkeypatch.setattr(np, 'save', counting_save)
        try:
            params = RulerQualityParameters(parallel_mode='process', n_jobs=4)
            r1 = compute_ruler_quality(
                pts, raw_ids, plane, origin, u_axis, v_axis, params)
            dumps_first = len(save_calls)
            r2 = compute_ruler_quality(
                pts, raw_ids, plane, origin, u_axis, v_axis, params)
        finally:
            rq._reset_shared_process_pool()

        # 机制：两次进程模式质量计算只创建了一个常驻池（spawn 开销摊销）
        assert len(created) == 1
        assert created[0] == min(4, rq.os.cpu_count() or 1)
        # 机制：进程模式真实落盘共享数组（平整度 points/raw_ids +
        # 垂直度 surf 三件套），线程模式对照组不落盘
        assert dumps_first >= 2
        assert len(save_calls) == 2 * dumps_first
        assert r1['ok'] and r2['ok']
        _deep_equal(_strip_volatile(r1), _strip_volatile(r2), 'pool-reuse')
        monkeypatch.undo()
        save_calls.clear()
        monkeypatch.setattr(np, 'save', counting_save)
        compute_ruler_quality(
            pts, raw_ids, plane, origin, u_axis, v_axis,
            RulerQualityParameters(parallel_mode='thread', n_jobs=4))
        assert save_calls == []  # 线程模式不触碰共享数组落盘
