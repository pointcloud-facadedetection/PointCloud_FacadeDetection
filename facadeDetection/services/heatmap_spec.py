"""Canonical measurement mapping shared by viewport, image export and reports."""
from __future__ import annotations

import numpy as np

# 9-node ramp: cyan -> teal -> green -> yellow-green -> yellow -> orange -> red.
COLOR_STOPS = (
    # Both one-way and bipolar scales start from the same low-saturation
    # neutral tone at the acceptance boundary.  This keeps the legend and
    # raster pixels semantically consistent instead of making threshold
    # defects appear immediately cyan.
    (0.000, (158 / 255.0, 178 / 255.0, 158 / 255.0)),
    (0.100, (0.00, 0.72, 1.00)),
    (0.200, (0.00, 0.86, 0.78)),
    (0.325, (0.10, 0.93, 0.38)),
    (0.450, (0.55, 0.95, 0.18)),
    (0.575, (1.00, 0.95, 0.00)),
    (0.700, (1.00, 0.74, 0.00)),
    (0.825, (1.00, 0.48, 0.00)),
    (0.925, (0.98, 0.22, 0.00)),
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


def cold_defect_colormap(t):
    """单极冷色调色图：合格区浅灰，超限从浅青过渡到深蓝。

    用于靠尺法平整度/垂直度，语义与凹陷一致（只看偏差量，不区分方向）。
    """
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    t = np.clip(t, 0.0, 1.0)
    # 低端舒缓
    t = np.power(t, np.float32(COLORMAP_GAMMA))
    out = np.empty((len(t), 3), dtype=np.float32)
    # 节点：灰 -> 浅青 -> 青 -> 蓝 -> 深蓝
    stops = np.asarray([
        (0.000, (220 / 255.0, 225 / 255.0, 230 / 255.0)),  # 合格灰
        (0.150, (180 / 255.0, 220 / 255.0, 240 / 255.0)),  # 浅青
        (0.350, (100 / 255.0, 190 / 255.0, 245 / 255.0)),  # 青
        (0.600, ( 40 / 255.0, 130 / 255.0, 230 / 255.0)),  # 蓝
        (0.850, ( 15 / 255.0,  70 / 255.0, 200 / 255.0)),  # 深蓝
        (1.000, (  5 / 255.0,  30 / 255.0, 160 / 255.0)),  # 极深蓝
    ], dtype=object)
    positions = np.asarray([p for p, _ in stops], dtype=np.float32)
    colors = np.asarray([c for _, c in stops], dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(t, positions, colors[:, channel])
    if len(t) >= 64:
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
        for channel in range(3):
            padded = np.pad(out[:, channel], 2, mode='edge')
            out[:, channel] = np.convolve(padded, kernel, mode='valid')
    return np.clip(out, 0.0, 1.0)


def apply_cold_excess_color(values_mm, limit_mm):
    """单极冷色调：合格区浅灰，超限青->蓝。"""
    return cold_defect_colormap(excess_uniform(values_mm, limit_mm))


def bipolar_colormap(signed_values_mm, limit_mm, scale_mm=None):
    """双向色带：暖色 = 凹陷（负偏差，朝建筑内），冷色 = 凸起（正偏差，朝外）。

    t 为 0..1 的超限严重度 (|value| - limit) / scale。
    暖端（凹陷）：黄 (255,220,60) → 红 (220,30,20)
    冷端（凸起）：青 (60,220,240) → 深蓝 (20,80,220)
    合格区：低饱和灰绿 (158,178,158)
    """
    values = np.asarray(signed_values_mm, dtype=float).reshape(-1)
    limit = max(float(limit_mm), 1e-6)

    excess = np.abs(values) - limit
    finite = excess[np.isfinite(excess)]
    if scale_mm is None:
        pos = finite[finite > 0]
        p98 = float(np.percentile(pos, 98)) if pos.size else limit * 0.15
        scale = max(p98, limit * 0.15, 1e-6)
    else:
        scale = max(float(scale_mm), 1e-6)

    t = np.clip(excess / scale, 0.0, 1.0)

    # Keep a visible neutral buffer around the acceptance boundary.  This is
    # deliberately continuous: colour starts at neutral and only reaches a
    # saturated warm/cool hue after the excess becomes material.
    neutral_width = 0.20
    q = np.clip(t / neutral_width, 0.0, 1.0)
    neutral = np.array((158 / 255.0, 178 / 255.0, 158 / 255.0), dtype=np.float32)

    out = np.empty((len(values), 3), dtype=np.float32)

    out[:] = neutral

    # 暖色区（凹陷，负值）：黄 → 红
    warm_mask = values < -limit
    if np.any(warm_mask):
        tw = q[warm_mask][:, None]
        warm = np.array((1.0, 0.86, 0.24), dtype=np.float32)
        hot = np.array((220.0, 30.0, 20.0), dtype=np.float32) / 255.0
        out[warm_mask] = neutral * (1.0 - tw) + (warm * (1.0 - tw) + hot * tw) * tw

    # 冷色区（凸起，正值）：青 → 深蓝
    cold_mask = values > limit
    if np.any(cold_mask):
        tc = q[cold_mask][:, None]
        cool = np.array((60.0, 220.0, 240.0), dtype=np.float32) / 255.0
        cold = np.array((20.0, 80.0, 220.0), dtype=np.float32) / 255.0
        out[cold_mask] = neutral * (1.0 - tc) + (cool * (1.0 - tc) + cold * tc) * tc

    return np.clip(out, 0.0, 1.0)

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
        'title': '模拟墙面垂直度热力图（外倾/内陷）',
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