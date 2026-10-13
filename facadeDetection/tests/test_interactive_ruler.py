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
