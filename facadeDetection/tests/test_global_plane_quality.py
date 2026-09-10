import numpy as np

from algorithms.facade.global_plane_quality import (
    compute_global_plane_quality,
    fit_global_plane,
)


def test_global_plane_fit_rejects_large_outliers_and_preserves_orientation():
    rng = np.random.default_rng(10)
    u = rng.uniform(0, 4, 3000)
    v = rng.uniform(0, 6, 3000)
    points = np.column_stack((u, v, rng.normal(0, 0.001, len(u))))
    points[:20, 2] += 0.3

    # New API: keyword-only reference_plane argument
    result = fit_global_plane(
        points,
        reference_plane=np.array([0, 0, 1, 0]),
        huber_delta_m=0.01,
    )
    normal = result['plane_model'][:3]
    assert abs(normal[2]) > 0.99
    assert result['residual_mad_mm'] < 2.0
    assert result['fit_accepted'] is True
    assert result['inlier_ratio'] >= 0.70


def test_global_windows_return_both_pass_rates():
    rng = np.random.default_rng(11)
    points = np.column_stack((
        rng.uniform(0, 2.1, 1200),
        rng.uniform(0, 4.1, 1200),
        rng.normal(0, 0.0005, 1200),
    ))
    result = compute_global_plane_quality(
        points, [0, 0, 1, 0], np.zeros(3), [1, 0, 0], [0, 1, 0],
        length_m=2.0, width_m=0.055, min_points=2,
    )
    assert result['windows']
    assert 0.0 <= result['overall']['flatness_primary_area_rate'] <= 1.0
    assert 0.0 <= result['overall']['verticality_secondary_point_rate'] <= 1.0
    assert all('depression_mm' in row and 'protrusion_mm' in row
               for row in result['windows'])
    # Per-window verticality should be present
    assert all('verticality_deviation_mm' in row for row in result['windows'])
    # New fields: actual_area and is_clipped
    assert all('actual_area_m2' in row and row['actual_area_m2'] > 0
               for row in result['windows'])
    assert all('is_clipped' in row for row in result['windows'])
    # Valid facade area via convex hull
    assert result['overall']['valid_detection_area_m2'] > 0


def test_verticality_is_zero_for_vertical_facade_and_uses_tangent_for_tilt():
    points = np.column_stack((
        np.zeros(400),
        np.repeat(np.linspace(0, 1, 20), 20),
        np.tile(np.linspace(0, 4, 20), 20),
    ))
    vertical = compute_global_plane_quality(
        points, [1, 0, 0, 0], np.zeros(3), [0, 1, 0], [0, 0, 1],
        length_m=2.0, width_m=.5, min_points=2)
    # For a vertical facade, each window's verticality should be near zero
    assert all(abs(row['verticality_deviation_mm']) < 0.01
               for row in vertical['windows'])

    # Plane normal tilted 1 degree from horizontal: 2m*tan(1deg).
    theta = np.deg2rad(1.0)
    tilted = compute_global_plane_quality(
        points, [np.cos(theta), 0, np.sin(theta), 0], np.zeros(3),
        [0, 1, 0], [0, 0, 1], length_m=2.0, width_m=.5, min_points=2)
    expected = 2.0 * np.tan(theta) * 1000.0
    # Global method now uses per-window fit_line; the first window should be close
    first_vert = tilted['windows'][0]['verticality_deviation_mm']
    assert np.isfinite(first_vert)
    assert np.isclose(first_vert, expected, atol=0.5)


def test_huber_irls_angular_constraint_prevents_drift():
    """Test that the angular gate prevents the fitted plane from drifting
    away from the reference normal."""
    rng = np.random.default_rng(42)
    n = 5000
    # Create a clean plane
    points = np.column_stack((
        rng.uniform(0, 10, n),
        rng.uniform(0, 10, n),
        rng.normal(0, 0.002, n),
    ))
    # Reference normal is z-up
    reference = np.array([0, 0, 1, 0])

    result = fit_global_plane(
        points,
        reference_plane=reference,
        angle_limit_deg=5.0,
    )
    assert result['fit_accepted']
    # Normal should stay very close to reference
    angle = result['normal_angle_to_reference_deg']
    assert angle < 1.0  # Clean data should fit almost perfectly


def test_global_plane_fit_rejects_with_low_inlier_ratio():
    """Test that extremely noisy data gets rejected."""
    rng = np.random.default_rng(99)
    n = 1000
    points = rng.normal(0, 0.5, (n, 3))  # Random cloud, no plane
    reference = np.array([0, 0, 1, 0])

    result = fit_global_plane(
        points,
        reference_plane=reference,
    )
    # Random points should not pass the 75% inlier threshold
    assert result['fit_accepted'] is False