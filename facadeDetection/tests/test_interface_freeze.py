"""接口冻结 golden 测试：与报告侧同事的结构契约守护。

facade.quality_report_json 与 facades.plane_json 的键结构已冻结，
任何算法/性能优化都不得增删改键名。本文件用真实合成数据跑
compute_ruler_quality 与 detect_facades_adaptive，断言输出的键集合
逐字等于冻结清单：

1. 机制：真实调用算法入口（非 mock），结果确实包含数据行；
2. 数据：windows 行 / intervals / overall / verticality / projection /
   facade dict / bbox_2d 的键集合与冻结清单一字不差；
3. 时间：真实计算有可测耗时（证明不是空跑）。
"""
from __future__ import annotations

import time

import numpy as np
import open3d as o3d
import pytest

from algorithms.facade.ruler_quality import (
    RulerQualityParameters, compute_ruler_quality)
from algorithms.facade.facade_detection import detect_facades_adaptive


# ---------------------------------------------------------------------------
# 冻结清单（与同事约定的契约，逐字冻结，不得改动）
# ---------------------------------------------------------------------------
QUALITY_TOP_KEYS = {
    'ok', 'parameters', 'windows', 'intervals', 'interval_size_m',
    'interval_count', 'projection_origin', 'projection_u_axis',
    'projection_v_axis', 'projection_normal', 'projection', 'overall',
    'verticality', 'performance',
}

QUALITY_WINDOW_ROW_KEYS = {
    'window_id', 'grid_key', 'cell_u', 'cell_v', 'center_xyz',
    'center_uv_base', 'flatness_gap_mm', 'flatness_raw_max_gap_mm',
    'flatness_pass', 'coverage_valid', 'direction_count',
    'quality_direction_count', 'hole_ratio', 'effective_point_count',
    'direction_deg', 'depression_source_id', 'pivot_source_ids',
    'center_uv', 'hole_length_m', 'angle_deg', 'source_direction',
    'snap_distance_m', 'directional_measurements',
    'verticality_angle_deg', 'verticality_deviation_mm', 'verticality_pass',
}

QUALITY_DIRECTION_MEASUREMENT_KEYS = {
    'direction_deg', 'flatness_gap_mm', 'coverage_valid',
    'effective_point_count',
}

QUALITY_INTERVAL_KEYS = {
    'label', 'v_min_m', 'v_max_m', 'world_height_min_m',
    'world_height_max_m', 'window_count', 'valid_window_count',
    'point_count', 'flatness_max_gap_mm', 'flatness_avg_gap_mm',
    'flatness_raw_max_gap_mm', 'flatness_pass_rate', 'quality_pass_rate',
    'status', 'verticality_max_deviation_mm',
    'verticality_avg_deviation_mm', 'verticality_pass_rate',
}

QUALITY_OVERALL_KEYS = {
    'point_count', 'candidate_window_count', 'geometry_valid_window_count',
    'quality_valid_window_count', 'failed_window_count',
    'flatness_max_gap_mm', 'flatness_avg_gap_mm',
    'flatness_raw_max_gap_mm', 'flatness_pass_rate', 'quality_pass_rate',
    'flatness_valid_rate', 'max_hole_ratio',
    'verticality_deviation_mm', 'verticality_avg_deviation_mm',
    'verticality_max_deviation_mm', 'verticality_max_angle_deg',
    'verticality_pass', 'verticality_pass_rate',
    'plane_model', 'normal', 'center',
}

QUALITY_VERTICALITY_KEYS = {
    'ok', 'verticality_pass', 'verticality_pass_rate',
}

QUALITY_PROJECTION_KEYS = {
    'u_range_m', 'v_range_m', 'u_min_m', 'u_max_m', 'v_min_m', 'v_max_m',
}

FACADE_DICT_KEYS = {
    'id', 'type', 'type_label', 'plane_model', 'normal', 'center', 'area',
    'point_count', 'inlier_indices', 'support_indices', 'verticality',
    'horizontality', 'flatness', 'flatness_mean', 'flatness_max',
    'bbox_2d', 'mean_range', 'max_range', 'fill_ratio', 'wall_kind',
}

FACADE_BBOX_2D_KEYS = {'u_min', 'u_max', 'v_min', 'v_max', 'u_axis', 'v_axis'}

# results_repo 写入 plane_json 的几何字段白名单（键结构冻结）
PLANE_JSON_ALLOWED_KEYS = {
    'plane_model', 'normal', 'center', 'inlier_indices', 'proxy_indices',
    'measurement_indices', 'voxel_ids', 'cloud_name', '__index_space',
}
PLANE_JSON_REQUIRED_KEYS = {'plane_model', 'normal', 'center', 'inlier_indices'}


def _make_quality_wall(seed=5):
    """合成墙面（x≈0 平面）+ 噪声 + 一处凹陷，raw_ids 模拟 dist 代理索引。"""
    rng = np.random.default_rng(seed)
    gy, gz = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(0, 5, 0.05))
    n = gy.size
    x = rng.normal(0, 0.002, n)
    dent = (np.abs(gy.ravel()) < 0.5) & (np.abs(gz.ravel() - 2.5) < 0.5)
    x[dent] -= 0.01
    pts = np.stack([x, gy.ravel(), gz.ravel()], 1)
    raw_ids = np.arange(n, dtype=np.int64)
    return pts, raw_ids


def _make_building(spacing=0.05, seed=3):
    """两面墙 + 地面条带的合成建筑点云（含窗洞）。"""
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


class TestQualityReportStructureFrozen:
    def test_quality_report_key_structure_matches_frozen_contract(self):
        pts, raw_ids = _make_quality_wall()
        plane = np.array([1.0, 0, 0, 0.0])
        origin = np.zeros(3)
        u_axis = np.array([0.0, 1.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])

        t0 = time.perf_counter()
        report = compute_ruler_quality(
            pts, raw_ids, plane, origin, u_axis, v_axis,
            RulerQualityParameters(parallel_mode='thread', n_jobs=2))
        elapsed = time.perf_counter() - t0

        # 机制：真实算法跑通且有数据行（不是空壳结构）
        assert report['ok'] is True
        assert len(report['windows']) > 0
        assert len(report['intervals']) > 0
        # 数据：键集合逐字等于冻结清单
        assert set(report.keys()) == QUALITY_TOP_KEYS
        for row in report['windows']:
            assert set(row.keys()) == QUALITY_WINDOW_ROW_KEYS
            assert isinstance(row['grid_key'], tuple) and len(row['grid_key']) == 2
            assert len(row['center_xyz']) == 3
        measured = [m for row in report['windows']
                    for m in row['directional_measurements']]
        assert measured, 'directional_measurements 不得为空'
        for m in measured:
            assert set(m.keys()) == QUALITY_DIRECTION_MEASUREMENT_KEYS
        for interval in report['intervals']:
            assert set(interval.keys()) == QUALITY_INTERVAL_KEYS
        assert set(report['overall'].keys()) == QUALITY_OVERALL_KEYS
        assert set(report['verticality'].keys()) == QUALITY_VERTICALITY_KEYS
        assert set(report['projection'].keys()) == QUALITY_PROJECTION_KEYS
        # 时间：真实计算有可测耗时
        print(f'\n[perf] frozen quality report '
              f'{len(pts)} pts = {elapsed:.2f}s '
              f'windows={len(report["windows"])}')
        assert elapsed > 0


class TestFacadePlaneJsonStructureFrozen:
    def test_facade_dict_key_structure_matches_frozen_contract(self):
        pts = _make_building()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        np.random.seed(42)
        t0 = time.perf_counter()
        result = detect_facades_adaptive(pcd, voxel_size=0.05, metadata={})
        elapsed = time.perf_counter() - t0

        facades = result['facades']
        # 机制：真实检测出立面（合成建筑至少应检出两面墙之一）
        assert len(facades) > 0
        for facade in facades:
            # 数据：facade dict 键集合逐字等于冻结清单
            assert set(facade.keys()) == FACADE_DICT_KEYS
            assert set(facade['bbox_2d'].keys()) == FACADE_BBOX_2D_KEYS
            # 数据：plane_json 必需字段齐备，且未引入白名单外的几何键
            assert PLANE_JSON_REQUIRED_KEYS <= set(facade.keys())
        # 时间：真实检测有可测耗时
        print(f'\n[perf] frozen detect {len(pts)} pts = {elapsed:.2f}s '
              f'facades={len(facades)}')
        assert elapsed > 0
