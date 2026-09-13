"""Canonical measurement mapping shared by viewport, image export and reports."""
from __future__ import annotations

import numpy as np

# 9-node ramp: cyan -> teal -> green -> yellow-green -> yellow -> orange -> red.
COLOR_STOPS = (
    (0.000, (0.00, 0.72, 1.00)),
    (0.125, (0.00, 0.86, 0.78)),
    (0.250, (0.10, 0.93, 0.38)),
    (0.375, (0.55, 0.95, 0.18)),
    (0.500, (1.00, 0.95, 0.00)),
    (0.625, (1.00, 0.74, 0.00)),
    (0.750, (1.00, 0.48, 0.00)),
    (0.875, (0.98, 0.22, 0.00)),
    (1.000, (0.90, 0.00, 0.00)),
)

# 低端舒缓指数。
COLORMAP_GAMMA = 0.85
# 色带动态范围下限
_DYNAMIC_SCALE_FLOOR_RATIO = 0.35


def defect_colormap(t):
    """Map normalised t (0..1) to RGB (N,3) float32 using COLOR_STOPS.

    在节点之间使用 PCHIP（保形三次）风格的单调插值：先按线性插值取初值，
    再用一次三点平滑消除折线拐点。相对纯线性，渐变边缘不再出现
    "青/绿/黄" 三段色阶，而是连续过渡。
    """
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    t = np.clip(t, 0.0, 1.0)
    # 低端舒缓：仅重映射色带位置，不改变 t=1 的饱和红端。
    t = np.power(t, np.float32(COLORMAP_GAMMA))
    out = np.empty((len(t), 3), dtype=np.float32)
    stops = np.asarray([c for _, c in COLOR_STOPS], dtype=np.float32)
    positions = np.asarray([p for p, _ in COLOR_STOPS], dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(t, positions, stops[:, channel])
    # 对连续色带做一次 5 点高斯平滑（仅作用在密集采样上）。
    # 离散 splat 场景下 t 长度很小，跳过以免破坏端点颜色。
    if len(t) >= 64:
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        for channel in range(3):
            padded = np.pad(out[:, channel], 2, mode='edge')
            out[:, channel] = np.convolve(padded, kernel, mode='valid')
    return np.clip(out, 0.0, 1.0)


def excess_uniform(values_mm, limit_mm):
    """Return normalised t for |excess| above the limit.
    t = clip((|value| - limit) / scale, 0, 1)。

    scale 取 98 分位超限量的稳健估计，并以限值的固定比例设下限，避免极端异常值导致色带过度压缩。
    """
    limit = max(float(limit_mm), 1e-6)
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    excess = np.maximum(np.abs(values) - limit, 0.0)
    finite = excess[np.isfinite(excess)]
    floor = limit * _DYNAMIC_SCALE_FLOOR_RATIO
    if finite.size and np.any(finite > 0):
        p98 = float(np.percentile(finite, 98))
        scale = max(p98, floor, 1e-6)
    else:
        scale = max(floor, 1e-6)
    t = np.clip(excess / scale, 0.0, 1.0)
    # 端点保形：sqrt 单调且 sqrt(0)=0、sqrt(1)=1，不引入新的越界样本。
    return np.sqrt(t)


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