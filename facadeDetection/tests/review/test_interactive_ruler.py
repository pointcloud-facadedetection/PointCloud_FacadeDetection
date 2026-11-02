"""交互靠尺纯逻辑测试（algorithms.facade.interactive_ruler）。

合成一面垂直墙点云（x=0 平面，u 轴沿 +y），验证：
- resolve_ruler_direction：质量报告 u 轴优先，否则由平面法向推导；
- extract_facade_points：索引域提取与 proxy→source 映射口径；
- place_ruler：平整墙 gap≈0，5mm 凹陷处 gap≈5mm 且按限差判超限，
  渲染规格（尺面四角 / 缺陷分段）形状与几何正确；
- ray_plane_intersect：正交/斜射求交、平行与反方向无交；
- point_in_facade_bounds：内/外/边界与 quality 包装字典；
- 平面交点驱动 place_ruler 的端到端（复核页新拾取链路）。
"""
import numpy as np
import pytest

from algorithms.facade import interactive_ruler

FACADE_GEOMETRY = {
    'plane_model': [1.0, 0.0, 0.0, 0.0],
    'origin': [0.0, 0.0, 0.0],
    'u_axis': [0.0, 1.0, 0.0],
    'v_axis': [0.0, 0.0, 1.0],
    'uv_bounds': [0.0, 0.0, 4.0, 3.0],
    'corners_3d': [[0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]],
    'normal': [1.0, 0.0, 0.0],
}


def _wall_points(recess=False, rng_seed=7):
    """x=0 垂直墙：y∈[0,4]，z∈[0,3]，法向 (1,0,0)。"""
    rng = np.random.default_rng(rng_seed)
    ys = np.arange(0.0, 4.0, 0.05)
    zs = np.arange(0.0, 3.0, 0.05)
    y, z = np.meshgrid(ys, zs)
    x = rng.uniform(-0.0005, 0.0005, size=y.shape)
    if recess:
        # 5mm 凹陷带（小于 select_band，保持为表面点参与靠尺）
        band = (y >= 1.9) & (y <= 2.1)
        x = np.where(band, x - 0.005, x)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


PLANE_MODEL = np.array([1.0, 0.0, 0.0, 0.0])


class TestResolveRulerDirection:
    def test_quality_report_u_axis_wins(self):
        facade = {'plane_model': PLANE_MODEL.tolist(),
                  'quality_report': {'projection_u_axis': [0.0, 2.0, 0.0]}}
        direction = interactive_ruler.resolve_ruler_direction(facade)
        np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-9)

    def test_fallback_from_plane_normal(self):
        direction = interactive_ruler.resolve_ruler_direction(
            {'plane_model': PLANE_MODEL.tolist()})
        assert abs(float(direction @ PLANE_MODEL[:3])) < 1e-9
        assert np.linalg.norm(direction) == pytest.approx(1.0)

    def test_missing_plane_returns_none(self):
        assert interactive_ruler.resolve_ruler_direction({}) is None
        assert interactive_ruler.resolve_ruler_direction(None) is None


class TestExtractFacadePoints:
    def test_inlier_indices_without_mapping(self):
        points = np.arange(15, dtype=float).reshape(5, 3)
        facade = {'inlier_indices': [1, 3, 4]}
        out = interactive_ruler.extract_facade_points(facade, points)
        np.testing.assert_array_equal(out, points[[1, 3, 4]])

    def test_proxy_indices_mapped_to_source(self):
        class _Index:
            def has_source_mapping(self):
                return True

            def proxy_to_source_ids(self, ids, deduplicate=True):
                return np.asarray(ids) * 2

        points = np.arange(15, dtype=float).reshape(5, 3)
        facade = {'proxy_indices': [0, 1, 2]}
        out = interactive_ruler.extract_facade_points(
            facade, points, index=_Index())
        np.testing.assert_array_equal(out, points[[0, 2, 4]])

    def test_too_few_points_returns_none(self):
        points = np.arange(15, dtype=float).reshape(5, 3)
        assert interactive_ruler.extract_facade_points(
            {'inlier_indices': [0, 1]}, points) is None
        assert interactive_ruler.extract_facade_points({}, points) is None


class TestPlaceRuler:
    def _surface(self, recess=False):
        return interactive_ruler.build_interactive_surface(
            _wall_points(recess=recess), PLANE_MODEL,
            ruler_dir=np.array([0.0, 1.0, 0.0]))

    def test_flat_wall_gap_near_zero(self):
        result = interactive_ruler.place_ruler(
            self._surface(), click_point=np.array([0.0, 2.0, 1.5]),
            threshold_mm=4.0)
        assert result['ok'] is True
        assert result['gap_mm'] < 2.0
        assert result['exceeded'] is False

    def test_recess_detected_and_flagged(self):
        """反测：5mm 凹陷必须报出约 5mm 的 gap 且按 4mm 限差判超限。"""
        result = interactive_ruler.place_ruler(
            self._surface(recess=True), click_point=np.array([0.0, 2.0, 1.5]),
            threshold_mm=4.0)
        assert result['ok'] is True
        assert result['gap_mm'] == pytest.approx(5.0, abs=1.5)
        assert result['exceeded'] is True
        segments = result['render']['defect_segments']
        assert segments
        assert any(seg['exceed'] for seg in segments)

    def test_render_geometry_matches_ruler_size(self):
        result = interactive_ruler.place_ruler(
            self._surface(), click_point=np.array([0.0, 2.0, 1.5]),
            ruler_length=2.0, ruler_width=0.055, threshold_mm=4.0)
        corners = np.asarray(result['render']['corners'])
        assert corners.shape == (4, 3)
        assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(2.0)
        assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(0.055)
        assert result['render']['depression_xyz'].shape == (3,)

    def test_no_threshold_means_no_defect_grid(self):
        result = interactive_ruler.place_ruler(
            self._surface(recess=True), click_point=np.array([0.0, 2.0, 1.5]))
        assert result['ok'] is True
        assert result['exceeded'] is False
        assert result['render']['defect_segments'] == []

    def test_click_outside_span_clamps_to_edge(self):
        surface = self._surface()
        result = interactive_ruler.place_ruler(
            surface, click_point=np.array([0.0, 99.0, 1.5]))
        assert result['ok'] is True
        assert result['u_center'] <= surface.u_span[1]

    def test_insufficient_points_raises(self):
        with pytest.raises(ValueError):
            interactive_ruler.build_interactive_surface(
                np.zeros((2, 3)), PLANE_MODEL)


class TestRayPlaneIntersect:
    def test_orthogonal_hit(self):
        point = interactive_ruler.ray_plane_intersect(
            origin=[5.0, 2.0, 1.0], direction=[-1.0, 0.0, 0.0],
            plane_model=PLANE_MODEL)
        np.testing.assert_allclose(point, [0.0, 2.0, 1.0], atol=1e-9)

    def test_oblique_hit(self):
        point = interactive_ruler.ray_plane_intersect(
            origin=[5.0, 0.0, 0.0], direction=[-1.0, 0.5, 0.25],
            plane_model=PLANE_MODEL)
        np.testing.assert_allclose(point, [0.0, 2.5, 1.25], atol=1e-9)

    def test_parallel_returns_none(self):
        assert interactive_ruler.ray_plane_intersect(
            origin=[5.0, 0.0, 0.0], direction=[0.0, 1.0, 0.0],
            plane_model=PLANE_MODEL) is None

    def test_behind_ray_returns_none(self):
        assert interactive_ruler.ray_plane_intersect(
            origin=[5.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0],
            plane_model=PLANE_MODEL) is None

    def test_unnormalized_plane_model(self):
        point = interactive_ruler.ray_plane_intersect(
            origin=[10.0, 1.0, 1.0], direction=[-1.0, 0.0, 0.0],
            plane_model=[2.0, 0.0, 0.0, -4.0])
        np.testing.assert_allclose(point, [2.0, 1.0, 1.0], atol=1e-9)


class TestPointInFacadeBounds:
    def test_inside(self):
        u, v, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 2.0, 1.5], FACADE_GEOMETRY)
        assert (u, v) == pytest.approx((2.0, 1.5))
        assert inside is True

    def test_outside(self):
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 5.0, 1.5], FACADE_GEOMETRY)
        assert inside is False
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 2.0, -0.5], FACADE_GEOMETRY)
        assert inside is False

    def test_boundary_counts_as_inside(self):
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 4.0, 3.0], FACADE_GEOMETRY)
        assert inside is True

    def test_accepts_quality_report_wrapper(self):
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 2.0, 1.5], {'facade_geometry': FACADE_GEOMETRY})
        assert inside is True

    def test_missing_geometry_returns_none_flag(self):
        u, v, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 2.0, 1.5], None)
        assert (u, v, inside) == (None, None, None)
        u, v, inside = interactive_ruler.point_in_facade_bounds(
            [0.0, 2.0, 1.5], {'quality_report': {}})
        assert (u, v, inside) == (None, None, None)


class TestPlaneIntersectionDrivenRuler:
    def test_ray_hit_point_drives_place_ruler(self):
        """复核页链路：射线交点 → place_ruler（凹陷处判超限）。"""
        points = _wall_points(recess=True)
        surface = interactive_ruler.build_interactive_surface(
            points, PLANE_MODEL, ruler_dir=np.array([0.0, 1.0, 0.0]))
        hit = interactive_ruler.ray_plane_intersect(
            origin=[8.0, 2.0, 1.5], direction=[-1.0, 0.0, 0.0],
            plane_model=PLANE_MODEL)
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            hit, FACADE_GEOMETRY)
        assert inside is True
        result = interactive_ruler.place_ruler(
            surface, hit, threshold_mm=4.0)
        assert result['ok'] is True
        assert result['gap_mm'] == pytest.approx(5.0, abs=1.5)
        assert result['exceeded'] is True


class TestTransparentPngUvRect:
    """复刻导出链路画布+裁剪：domain 外扩 5%、max_size 像素步长、
    facade_mask 2% 行列覆盖率裁剪。合成 u∈[0,4] v∈[0,3] 密墙，
    裁剪框应精确还原为 (0,0,4,3)。"""

    def _quality(self):
        return {
            'quality_comparison': {'domain': {
                'u_min_m': 0.0, 'v_min_m': 0.0,
                'u_max_m': 4.0, 'v_max_m': 3.0}},
            'projection_origin': [0.0, 0.0, 0.0],
            'projection_u_axis': [0.0, 1.0, 0.0],
            'projection_v_axis': [0.0, 0.0, 1.0],
        }

    def _wall(self):
        ys = np.arange(0.0, 4.0 + 1e-9, 0.05)
        zs = np.arange(0.0, 3.0 + 1e-9, 0.05)
        y, z = np.meshgrid(ys, zs)
        return np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])

    def test_crop_rect_recovers_wall_extent(self):
        rect = interactive_ruler.transparent_png_uv_rect(
            self._wall(), self._quality())
        assert rect is not None
        # 画布：domain 外扩 5%（u ±0.2, v ±0.15），size=0.01
        assert rect['canvas_wh'] == (441, 331)
        assert rect['crop_xy'] == (20, 15, 420, 315)
        u0, v0, u1, v1 = rect['uv_bounds']
        assert (u0, v0, u1, v1) == pytest.approx((0.0, 0.0, 4.0, 3.0))

    def test_max_size_downscales_pixel(self):
        # 90m 长边触发 max_size=3200 降采样：size = 99/3199（含 5% 边距）
        quality = self._quality()
        quality['quality_comparison']['domain'].update(
            {'u_max_m': 4.0, 'v_max_m': 90.0})
        ys = np.arange(0.0, 4.0 + 1e-9, 0.05)
        zs = np.arange(0.0, 90.0 + 1e-9, 0.05)
        y, z = np.meshgrid(ys, zs)
        pts = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
        rect = interactive_ruler.transparent_png_uv_rect(pts, quality)
        assert rect['pixel_size'] == pytest.approx(99.0 / 3199)
        u0, v0, u1, v1 = rect['uv_bounds']
        assert (u0, u1) == pytest.approx((0.0, 4.0), abs=0.05)
        assert (v0, v1) == pytest.approx((0.0, 90.0), abs=0.05)

    def test_missing_domain_returns_none(self):
        assert interactive_ruler.transparent_png_uv_rect(
            self._wall(), {}) is None
        assert interactive_ruler.transparent_png_uv_rect(
            self._wall(), None) is None


class TestRulerDirectionFromPoints:
    def test_along_u_axis(self):
        d = interactive_ruler.ruler_direction_from_points(
            [0.0, 1.0, 1.0], [0.0, 3.0, 1.0], PLANE_MODEL)
        np.testing.assert_allclose(d, [0.0, 1.0, 0.0], atol=1e-9)

    def test_out_of_plane_component_removed(self):
        """第二击的射线交点必在平面内；此处直接给带法向分量的点，
        验证投影仍把方向压回平面。"""
        d = interactive_ruler.ruler_direction_from_points(
            [0.0, 1.0, 1.0], [0.4, 1.0, 2.0], PLANE_MODEL)
        assert abs(float(d @ PLANE_MODEL[:3])) < 1e-9
        np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-9)

    def test_diagonal_normalized(self):
        d = interactive_ruler.ruler_direction_from_points(
            [0.0, 0.0, 0.0], [0.0, 1.0, 1.0], PLANE_MODEL)
        np.testing.assert_allclose(d, [0.0, 1.0, 1.0] / np.sqrt(2.0),
                                   atol=1e-9)

    def test_degenerate_too_close_returns_none(self):
        assert interactive_ruler.ruler_direction_from_points(
            [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], PLANE_MODEL) is None
        assert interactive_ruler.ruler_direction_from_points(
            [0.0, 1.0, 1.0], [0.0, 1.0, 1.0001], PLANE_MODEL) is None


class TestTwoClickRulerPlacement:
    """点选方向驱动的端到端：条带取样（与批量靠尺同物理语义）。"""

    @staticmethod
    def _dense_wall():
        """u∈[0,4] v∈[0,3] 密墙，局部凹陷补丁（非通长条带）：
        y∈[1.9,2.1] 且 z∈[1.4,1.6]，深 6mm。"""
        rng = np.random.default_rng(7)
        ys = np.arange(0.0, 4.0, 0.02)
        zs = np.arange(0.0, 3.0, 0.02)
        y, z = np.meshgrid(ys, zs)
        x = rng.uniform(-0.0005, 0.0005, size=y.shape)
        patch = (y >= 1.9) & (y <= 2.1) & (z >= 1.4) & (z <= 1.6)
        x = np.where(patch, x - 0.006, x)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    def test_strip_catches_localized_recess(self):
        """尺线穿过局部凹陷 → 检出超限；对照：全立面宽剖面（旧交互路径）
        会把同一缺陷稀释掉（同 bin 被条带外正常点顶掉）。"""
        points = self._dense_wall()
        center = np.array([0.0, 2.0, 1.5])
        direction = np.array([0.0, 1.0, 0.0])
        result = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, center, direction, threshold_mm=4.0)
        assert result['ok'] is True
        assert result['gap_mm'] == pytest.approx(6.0, abs=1.5)
        assert result['exceeded'] is True
        assert result['strip_point_count'] > 0

        surface = interactive_ruler.build_interactive_surface(
            points, PLANE_MODEL, ruler_dir=direction)
        wide = interactive_ruler.place_ruler(
            surface, center, threshold_mm=4.0)
        assert wide['ok'] is True
        assert wide['gap_mm'] < 4.0, '全宽剖面应稀释局部凹陷'

    def test_body_line_matches_pick_direction(self):
        points = self._dense_wall()
        center = np.array([0.0, 2.0, 1.5])
        second = np.array([0.0, 3.0, 1.5])     # 沿 u 轴
        direction = interactive_ruler.ruler_direction_from_points(
            center, second, PLANE_MODEL)
        result = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, center, direction, ruler_length=2.0,
            threshold_mm=4.0)
        assert result['ok'] is True
        body = np.asarray(result['render']['body_line'])
        assert body.shape == (2, 3)
        body_dir = body[1] - body[0]
        assert np.linalg.norm(body_dir) == pytest.approx(2.0)
        np.testing.assert_allclose(
            body_dir / np.linalg.norm(body_dir), direction, atol=1e-4)
        mid = (body[0] + body[1]) * 0.5
        assert abs(float((mid - center) @ direction)) < 1e-6

    def test_strip_too_sparse_returns_not_ok(self):
        points = self._dense_wall()[::2000]      # 抽稀到条带不足 3 点
        result = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, np.array([0.0, 2.0, 1.5]),
            np.array([0.0, 1.0, 0.0]), threshold_mm=4.0)
        assert result['ok'] is False
        assert result.get('reason')

    def test_first_click_is_body_start_not_center(self):
        points = self._dense_wall()
        start = np.array([0.0, 1.0, 1.5])
        second = np.array([0.0, 3.0, 1.5])
        direction = interactive_ruler.ruler_direction_from_points(
            start, second, PLANE_MODEL)
        center = interactive_ruler.ruler_center_from_start(start, direction, 2.0)
        result = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, center, direction, ruler_length=2.0,
            clamp_center=False)
        assert result['ok'] is True
        body = np.asarray(result['render']['body_line'])
        assert float((body[0] - start) @ direction) == pytest.approx(0.0, abs=1e-6)
        assert float((body[1] - start) @ direction) == pytest.approx(2.0, abs=1e-6)

    def test_isolated_spike_is_removed_before_ruler_placement(self):
        points = self._dense_wall()
        center = np.array([0.0, 2.0, 1.5])
        direction = np.array([0.0, 1.0, 0.0])
        baseline = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, center, direction, threshold_mm=4.0)
        points = np.vstack((points, np.array([[0.2, 2.0, 1.5]])))
        result = interactive_ruler.place_ruler_on_strip(
            points, PLANE_MODEL, center, direction, threshold_mm=4.0)
        assert result['ok'] is True
        assert result['sor_removed_count'] >= 1
        assert result['gap_mm'] == pytest.approx(baseline['gap_mm'])
