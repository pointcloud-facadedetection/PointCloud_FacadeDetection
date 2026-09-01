"""Canonical measurement mapping shared by viewport, image export and reports."""

import numpy as np


HEATMAP_SPECS = {
    'flatness': {
        'title': '平整度热力图',
        'value_key': 'flatness_gap_mm',
        'pass_key': 'flatness_pass',
        'limit_key': 'flatness_limit_mm',
        'file_key': 'flatness',
    },
    'verticality': {
        'title': '垂直度热力图',
        'value_key': 'verticality_deviation_mm',
        'pass_key': 'verticality_pass',
        'limit_key': 'verticality_limit_mm',
        'file_key': 'verticality',
    },
}


def normalize_heatmap_mode(mode) -> str:
    """Only expose production-supported display modes to downstream services."""
    return 'verticality' if str(mode or '').lower() == 'verticality' else 'flatness'


def heatmap_spec(mode):
    return HEATMAP_SPECS[normalize_heatmap_mode(mode)]


def heatmap_limit_and_scale_mm(quality, spec, default_limit=4.0):
    """Return (limit_mm, scale_mm) from the active inspection profile.

    Colour is anchored to the standard limit, not the sample maximum.
    ``scale_mm`` is the absolute millimetre value that maps to red
    (default ``fail_ratio`` × limit, i.e. 2×limit).
    """
    params = quality.get('parameters') or {}
    thresholds = quality.get('thresholds') or {}
    raw_limit = params.get(spec['limit_key'], thresholds.get(spec['limit_key'], default_limit))
    try:
        limit = abs(float(raw_limit))
    except (TypeError, ValueError):
        limit = float(default_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(default_limit)
    try:
        fail_ratio = float(params.get('fail_ratio', 2.0) or 2.0)
    except (TypeError, ValueError):
        fail_ratio = 2.0
    scale = max(limit * max(fail_ratio, 0.5), 1.0)
    return limit, scale


def heatmap_error_colors(values_mm, limit_mm, scale_mm=None):
    """Grey at 0 mm, yellow at the standard limit, red at ``scale_mm``.

    ``scale_mm`` defaults to 2×limit so the pass/fail boundary sits at yellow.
    """
    mag = np.abs(np.asarray(values_mm, dtype=np.float64).reshape(-1))
    if mag.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    limit = max(float(limit_mm), 1e-6)
    top = max(float(scale_mm) if scale_mm is not None else (2.0 * limit), limit * 1.01)
    gray = np.array((0.58, 0.58, 0.58), dtype=np.float32)
    yellow = np.array((1.0, 0.90, 0.12), dtype=np.float32)
    red = np.array((0.90, 0.10, 0.08), dtype=np.float32)
    colors = np.empty((len(mag), 3), dtype=np.float32)
    below = mag <= limit
    u = (mag[below] / limit).astype(np.float32)[:, None]
    colors[below] = gray + (yellow - gray) * u
    span = max(top - limit, 1e-6)
    u = np.clip((mag[~below] - limit) / span, 0.0, 1.0).astype(np.float32)[:, None]
    colors[~below] = yellow + (red - yellow) * u
    return colors


def defect_excess_colors(excess_mm, scale_mm):
    """Backward-compatible wrapper: treat excess as error above a unit limit."""
    scale = max(float(scale_mm), 1e-6)
    return heatmap_error_colors(np.asarray(excess_mm) + scale, scale, scale)