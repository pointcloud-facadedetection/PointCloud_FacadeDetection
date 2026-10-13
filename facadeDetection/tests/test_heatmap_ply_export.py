"""带热力图的下采样 PLY 导出：纯逻辑（UV 取色 / 写盘往返）测试。"""
import numpy as np

from config.settings import Config
from services.heatmap_point_coloring import (
    colorize_points_with_quality_reports,
    rasterize_quality_heatmap,
    sample_heatmap_colors,
)
from utils.ply_fast_reader import read_ply_fast
from utils.ply_writer import write_point_cloud_ply_atomic

GRAY = np.asarray(Config.FACADE_BASE_COLOR, dtype=np.float32)


def _make_quality(all_pass=False):
    """x=0 立面：u=+Y、v=+Z，2m x 2m 窗口网格，超限 8mm（限值 4mm）。"""
    windows = []
    for iy in range(5):
        for iz in range(5):
            windows.append({
                'center_xyz': [0.0, 0.25 + 0.5 * iy, 0.25 + 0.5 * iz],
                'flatness_gap_mm': 0.0 if all_pass else 8.0,
                'flatness_pass': bool(all_pass),
                'verticality_deviation_mm': 0.0 if all_pass else 8.0,
                'verticality_pass': bool(all_pass),
            })
    return {
        'overall': {'plane_model': [1.0, 0.0, 0.0, 0.0]},
        'thresholds': {'flatness_limit_mm': 4.0, 'verticality_limit_mm': 4.0},
        'projection_origin': [0.0, 0.0, 0.0],
        'projection_u_axis': [0.0, 1.0, 0.0],
        'projection_v_axis': [0.0, 0.0, 1.0],
        'quality_comparison': {'methods': {'ruler': {
            'flatness': {'windows': windows},
            'verticality': {'windows': windows},
        }}},
    }


def test_rasterize_returns_none_when_no_failed_windows():
    assert rasterize_quality_heatmap(
        _make_quality(all_pass=True), 'ruler_flatness_area') is None
    assert rasterize_quality_heatmap({}, 'ruler_flatness_area') is None
    assert rasterize_quality_heatmap(None, 'ruler_flatness_area') is None


def test_sample_heatmap_colors_covers_failed_window_neighborhood():
    raster = rasterize_quality_heatmap(_make_quality(), 'ruler_flatness_area')
    assert raster is not None
    assert np.any(raster['overlay_rgba'][:, :, 3] > 0)

    points = np.array([
        [0.01, 1.25, 1.25],   # 命中失败窗口 splat 邻域
        [0.0, 50.0, 50.0],    # 画布外：不得被边缘截断错染
    ])
    colors, covered = sample_heatmap_colors(raster, points)
    assert covered[0]
    assert not covered[1]
    assert np.all(colors[0] >= 0.0) and np.all(colors[0] <= 1.0)


def test_colorize_keeps_explicit_base_color_for_uncovered_points():
    quality = _make_quality()
    points = np.array([
        [0.01, 1.25, 1.25],
        [0.0, 50.0, 50.0],
    ])
    base = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (2, 1))
    colors, stats = colorize_points_with_quality_reports(
        points, base, [quality], 'ruler_flatness_area')
    assert stats['applied_facades'] == 1
    assert stats['colored_points'] == 1
    # 显式基准色：未覆盖点保留原色
    assert np.allclose(colors[1], base[1], atol=1e-6)
    # 覆盖点被热力色替换
    assert not np.allclose(colors[0], base[0], atol=1e-3)


def test_colorize_fills_facade_gray_when_base_colors_missing():
    colors, stats = colorize_points_with_quality_reports(
        np.array([[0.0, 50.0, 50.0]]), None, [_make_quality()],
        'ruler_flatness_area')
    assert stats['colored_points'] == 0
    assert np.allclose(colors[0], GRAY, atol=1e-6)


def test_heatmap_colored_ply_round_trip(tmp_path):
    quality = _make_quality()
    rng = np.random.default_rng(7)
    # 立面域内的密集点 + 少量域外点，模拟下采样后的点云
    in_domain = np.column_stack((
        rng.normal(0, 0.02, 2000),
        rng.uniform(0, 2.5, 2000),
        rng.uniform(0, 2.5, 2000),
    ))
    out_domain = np.array([[0.0, 50.0, 50.0], [100.0, 0.0, 0.0]])
    points = np.vstack((in_domain, out_domain)).astype(np.float32)

    # 与 export_station_heatmap_ply 一致：灰底（base_colors=None）+ 热力染色
    colors, stats = colorize_points_with_quality_reports(
        points, None, [quality], 'ruler_verticality_area')
    assert stats['colored_points'] > 0

    target = tmp_path / 'heatmap.ply'
    result = write_point_cloud_ply_atomic(target, points, colors)
    assert result['point_count'] == len(points)
    assert result['has_colors']

    read = read_ply_fast(target)
    assert read is not None
    r_points, r_colors = read
    assert np.allclose(r_points, points, atol=1e-6)
    # 写盘 uchar 量化误差 ≤ 1/255
    assert np.allclose(r_colors, colors, atol=1.0 / 255.0 + 1e-6)
    # 未被热力覆盖的点是统一灰（Config.FACADE_BASE_COLOR）
    assert np.allclose(r_colors[-2:], GRAY, atol=1.0 / 255.0 + 1e-6)
