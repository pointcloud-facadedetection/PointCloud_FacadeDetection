"""Canonical measurement mapping shared by viewport, image export and reports."""
from __future__ import annotations

import numpy as np

# 5-node ramp: cyan -> green -> yellow -> orange -> red, over normalised t.
COLOR_STOPS = (
    (0.00, (0.0, 0.7, 1.0)),
    (0.25, (0.2, 0.9, 0.2)),
    (0.50, (1.0, 1.0, 0.0)),
    (0.75, (1.0, 0.5, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
)


def defect_colormap(t):
    """Map normalised t (0..1) to RGB (N,3) float32 using COLOR_STOPS."""
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    t = np.clip(t, 0.0, 1.0)
    out = np.empty((len(t), 3), dtype=np.float32)
    stops = np.asarray([c for _, c in COLOR_STOPS], dtype=np.float32)
    positions = np.asarray([p for p, _ in COLOR_STOPS], dtype=np.float32)
    out[:, 0] = np.interp(t, positions, stops[:, 0])
    out[:, 1] = np.interp(t, positions, stops[:, 1])
    out[:, 2] = np.interp(t, positions, stops[:, 2])
    return out


def excess_uniform(values_mm, limit_mm):
    """Return normalised t for |excess| above the limit.

    t = clip((|value| - limit) / scale, 0, 1), with a robust 98th-percentile
    scale floor so a single extreme residual cannot crush the dynamic range.
    """
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    excess = np.maximum(np.abs(values) - float(limit_mm), 0.0)
    finite = excess[np.isfinite(excess)]
    if finite.size and np.any(finite > 0):
        p98 = float(np.percentile(finite, 98))
        scale = max(p98, float(limit_mm) * 0.15, 1e-6)
    else:
        scale = max(float(limit_mm) * 0.15, 1e-6)
    return np.clip(excess / scale, 0.0, 1.0)


def apply_excess_color(values_mm, limit_mm):
    """Return (N,3) float32 RGB for a list of mm deviations."""
    heat = defect_colormap(excess_uniform(values_mm, limit_mm))
    # t<=limit region is "pass"; neutralise it so only defects are coloured.
    # Keep the ramp start (cyan) out of the pass band.
    return heat

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
    'global_flatness': {
        'title': '全局平面平整度热力图',
        'value_key': 'flatness_gap_mm',
        'pass_key': 'flatness_pass',
        'limit_key': 'flatness_limit_mm',
        'file_key': 'global_flatness',
    },
    'global_verticality': {
        'title': '全局平面垂直度热力图',
        'value_key': 'verticality_deviation_mm',
        'pass_key': 'verticality_pass',
        'limit_key': 'verticality_limit_mm',
        'file_key': 'global_verticality',
    },
}

# Explicit display/export contract: two algorithms x two measurements x two
# rate views. Rate views use the same window deviation, but differ in how
# values are rasterized (window area versus raw-point coverage).
for _method, _label in (('ruler', '米字/I字靠尺'), ('global_plane', '全局平面基准')):
    for _metric, _metric_label, _value, _passed in (
        ('flatness', '平整度', 'flatness_gap_mm', 'flatness_pass'),
        ('verticality', '垂直度', 'verticality_deviation_mm', 'verticality_pass'),
    ):
        for _view, _view_label in (('area', '面积'), ('point', '点数')):
            _key = f'{_method}_{_metric}_{_view}'
            HEATMAP_SPECS[_key] = {
                'title': f'{_label}{_metric_label}{_view_label}热力图',
                'value_key': _value,
                'pass_key': _passed,
                'limit_key': f'{_metric}_limit_mm',
                'file_key': _key,
                'method': _method,
                'metric': _metric,
                'view': _view,
            }


def normalize_heatmap_mode(mode) -> str:
    """Only expose production-supported display modes to downstream services."""
    value = str(mode or '').lower()
    aliases = {
        'flatness': 'ruler_flatness_area',
        'verticality': 'ruler_verticality_area',
        'global_flatness': 'global_flatness_area',
        'global_verticality': 'global_verticality_area',
    }
    value = aliases.get(value, value)
    if value in HEATMAP_SPECS:
        return value
    return 'ruler_verticality_area' if value == 'verticality' else 'ruler_flatness_area'


def heatmap_spec(mode):
    return HEATMAP_SPECS[normalize_heatmap_mode(mode)]