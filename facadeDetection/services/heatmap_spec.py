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
    """Return (limit_mm, excess_scale_mm) from the active inspection profile.

    Colour must be anchored to the standard limit, not the sample maximum.
    A 570 mm outlier would otherwise crush typical 10–30 mm defects to grey.
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


def defect_excess_colors(excess_mm, scale_mm):
    """Yellow at just-over-limit, red at ``scale_mm``. Failed cells stay visible."""
    excess = np.asarray(excess_mm, dtype=np.float64).reshape(-1)
    t = np.clip(excess / max(float(scale_mm), 1e-6), 0.0, 1.0).astype(np.float32)
    colors = np.empty((len(t), 3), dtype=np.float32)
    colors[:, 0] = 1.0
    colors[:, 1] = 1.0 - t
    colors[:, 2] = 0.0
    return colors