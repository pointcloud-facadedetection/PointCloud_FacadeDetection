"""检测链路优化的实效验证：去冗余 deepcopy（inplace 法向）+ 输出逐点一致。

旧实现参照副本 _reference_ensure_normals 逐行保持优化前语义
（copy.deepcopy + estimate_normals + orient_normals_consistent_tangent_plane(30)），
通过 monkeypatch 注入 facade_detection 模块得到"优化前"基线。

关于 orient_normals_consistent_tangent_plane：前期报告曾建议跳过
（检测链路对法向符号"全不敏感"），实测证伪——粗层 voxel_down_sample
对法向做带符号平均，平均方向依赖一致定向（TestOrientNotSkippable 固定
这条机制事实），高密度点云上跳 orient 会改变粗层 Hough 投票与最终
facades 数值，违反逐点一致硬边界，因此生产代码保留 orient。

每个用例断言三类事实：
1. 机制：inplace 后 deepcopy 0 次、orient 每次检测真实执行 1 次；
2. 数据：新旧检测的 facades 列表数值逐点一致（JSON 全量比对）；
3. 时间：新旧实现都有可测耗时（orient 在百万点级为数十秒真实工作）。
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import open3d as o3d
import pytest

import algorithms.geometry as geom
import algorithms.facade.facade_detection as fd


# ---------------------------------------------------------------------------
# 旧实现参照副本（优化前的 ensure_normals，逐行保持原语义）
# ---------------------------------------------------------------------------
def _reference_ensure_normals(pcd, voxel_size=0.05, inplace=False):
    if len(pcd.points) == 0:
        return pcd
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        existing = np.asarray(pcd.normals)
        if existing.ndim == 2 and existing.shape == (len(pcd.points), 3) \
                and np.all(np.isfinite(existing)):
            norms = np.linalg.norm(existing, axis=1)
            if np.all(norms > 1e-8):
                pcd_work = pcd if inplace else copy.deepcopy(pcd)
                pcd_work.normals = o3d.utility.Vector3dVector(
                    existing / norms[:, None])
                return pcd_work
    pcd_work = pcd if inplace else copy.deepcopy(pcd)
    radius = max(float(voxel_size) * 4.0, 0.2)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=50))
    try:
        pcd_work.orient_normals_consistent_tangent_plane(30)
    except Exception:
        pcd_work.orient_normals_towards_camera_location([0, 0, 0])
    return pcd_work


def _make_building(spacing, seed=3):
    """两面墙（含窗洞）+ 地面条带的合成建筑点云。"""
    rng = np.random.default_rng(seed)
    parts = []
    gy, gz = np.meshgrid(np.arange(-6, 6, spacing), np.arange(0, 8, spacing))
    n = gy.size
    x = rng.normal(0, 0.003, n)
    mask = np.ones(n, bool)
    for wy in (-3, 0, 3):
        for wz in (2, 5):
            mask &= ~((np.abs(gy.ravel() - wy) < 0.6) &
                      (np.abs(gz.ravel() - wz) < 0.7))
    parts.append(np.stack([x[mask], gy.ravel()[mask], gz.ravel()[mask]], 1))
    gx, gz2 = np.meshgrid(np.arange(0, 10, spacing), np.arange(0, 8, spacing))
    y = 8 + rng.normal(0, 0.003, gx.size)
    parts.append(np.stack([gx.ravel(), y, gz2.ravel()], 1))
    gx2, gy2 = np.meshgrid(np.arange(-6, 10, spacing * 1.6),
                           np.arange(-6, 8, spacing * 1.6))
    z = rng.normal(0, 0.003, gx2.size)
    parts.append(np.stack([gx2.ravel(), gy2.ravel(), z], 1))
    return np.vstack(parts).astype(np.float64) + 10.0


def _run_detect(pts, ensure_normals_fn, monkeypatch, seed=42):
    """以指定 ensure_normals 运行检测（np.random 固定种子保证可比）。

    ensure_normals_fn 为 None 时显式装回生产版（不能用 monkeypatch.undo()
    恢复——那会把 orient_spy fixture 的补丁一并撤销，导致后续计数失效）。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    monkeypatch.setattr(fd, 'ensure_normals',
                        ensure_normals_fn or _PRODUCTION_ENSURE_NORMALS)
    np.random.seed(seed)
    t0 = time.perf_counter()
    result = fd.detect_facades_adaptive(pcd, voxel_size=0.05, metadata={})
    return result, time.perf_counter() - t0, pcd


def _facades_json(result):
    return json.dumps(result['facades'], sort_keys=True)


# 生产版 ensure_normals（模块导入时快照，供 _run_detect 显式装回）
_PRODUCTION_ENSURE_NORMALS = fd.ensure_normals


@pytest.fixture
def orient_spy(monkeypatch):
    """orient_normals_consistent_tangent_plane 调用计数间谍。"""
    calls = []
    real = o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane

    def spy(self, k):
        calls.append(k)
        return real(self, k)

    monkeypatch.setattr(
        o3d.geometry.PointCloud,
        'orient_normals_consistent_tangent_plane', spy)
    return calls


class TestInplaceDetectEquivalence:
    @pytest.mark.parametrize('seed', (3, 17))
    def test_output_identical_to_reference(self, monkeypatch, orient_spy,
                                           seed):
        """10 万级点云（粗层已启用）：inplace 新实现与旧基线逐点一致。"""
        pts = _make_building(0.05, seed=seed)
        assert len(pts) > 100_000  # 确保粗层分支真实启用

        base, t_old, _ = _run_detect(pts, _reference_ensure_normals,
                                     monkeypatch)
        orient_old = len(orient_spy)
        new, t_new, pcd = _run_detect(pts, None, monkeypatch)

        # 机制：新旧都真实执行了一次 orient（定向保留）；法向就地写回输入
        assert orient_old >= 1
        assert len(orient_spy) >= orient_old + 1
        assert pcd.has_normals()
        # 数据：facades 列表数值逐点一致（含粗层投票与合并后处理）
        assert len(new['facades']) > 0
        assert _facades_json(new) == _facades_json(base)
        print(f'\n[perf] detect {len(pts)} pts seed={seed}: '
              f'reference={t_old:.2f}s inplace={t_new:.2f}s')
        assert t_old > 0 and t_new > 0

    def test_million_points_identical_to_reference(self, monkeypatch,
                                                   orient_spy):
        """≥100 万点合成云：inplace 新实现与旧基线 facades 逐点一致。"""
        pts = _make_building(0.015)
        assert len(pts) >= 1_000_000

        base, t_old, _ = _run_detect(pts, _reference_ensure_normals,
                                     monkeypatch)
        orient_old = len(orient_spy)
        new, t_new, pcd = _run_detect(pts, None, monkeypatch)

        # 机制：新旧各执行一次 orient；法向就地写回（无外层 deepcopy）
        assert orient_old >= 1
        assert len(orient_spy) >= orient_old + 1
        assert pcd.has_normals()
        assert len(pcd.normals) == len(pts)
        # 数据：facades 列表数值与优化前逐点一致
        assert len(new['facades']) > 0
        assert _facades_json(new) == _facades_json(base)
        # 时间：百万点级新旧检测都是数十秒级真实工作（若接近 0 测试失效）
        print(f'\n[perf] detect {len(pts)} pts: '
              f'reference={t_old:.2f}s inplace={t_new:.2f}s')
        assert t_old > 1.0 and t_new > 1.0


class TestOrientNotSkippable:
    def test_coarse_voxel_normal_averaging_is_sign_sensitive(self):
        """机制事实：voxel_down_sample 对法向做带符号平均，平均方向依赖
        orient 的一致定向——这是"跳 orient 保持逐点一致"被证伪的根因，
        固定下来防止再次尝试跳过。"""
        pts = _make_building(0.05)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.2, max_nn=50))
        raw = copy.deepcopy(pcd)
        pcd.orient_normals_consistent_tangent_plane(30)
        coarse_oriented = pcd.voxel_down_sample(0.2)
        coarse_raw = raw.voxel_down_sample(0.2)
        # 数据：粗层点逐点一致（降采样与法向无关），但平均法向方向不同
        assert np.array_equal(np.asarray(coarse_oriented.points),
                              np.asarray(coarse_raw.points))
        dots = np.abs(np.sum(np.asarray(coarse_oriented.normals) *
                             np.asarray(coarse_raw.normals), axis=1))
        diverged = int(np.sum(dots < 0.999))
        print(f'\n[perf] coarse voxel normals diverged without orient: '
              f'{diverged}/{len(dots)}')
        assert diverged > 0  # 符号敏感的平均真实存在


class TestNoRedundantDeepcopy:
    def test_detect_runs_inplace_without_deepcopy(self, monkeypatch):
        pts = _make_building(0.05)

        deepcopy_calls = []
        real_deepcopy = copy.deepcopy

        def counting_deepcopy(obj):
            deepcopy_calls.append(type(obj).__name__)
            return real_deepcopy(obj)

        monkeypatch.setattr(geom.copy, 'deepcopy', counting_deepcopy)
        result, _, pcd = _run_detect(pts, None, monkeypatch)

        # 机制：ensure_normals(inplace=True) 一次 deepcopy 都没发生
        assert deepcopy_calls == []
        # 机制：法向就地写回调用方的 pcd（证明工作对象就是输入对象本身）
        assert pcd.has_normals()
        assert len(pcd.normals) == len(pcd.points)
        # 数据：检测真实产出了结果结构
        assert result['total_points'] == len(pts)
        assert isinstance(result['facades'], list)
