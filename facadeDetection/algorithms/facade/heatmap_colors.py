"""有符号偏差热力着色：平整灰、凹陷 Blues、凸起 autumn_r。"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_NEUTRAL_MM = 5.0
_GRAY = np.array((0.55, 0.55, 0.55), dtype=np.float64)
GRAY_BGR = (140, 140, 140)
_CMAP_CACHE: dict[str, object] = {}


def _sample_colormap(name: str, samples):
    values = np.clip(np.asarray(samples, dtype=np.float64).reshape(-1), 0.0, 1.0)
    cmap = _CMAP_CACHE.get(name)
    if cmap is None:
        try:
            from matplotlib import colormaps
            cmap = colormaps[name]
        except Exception:
            try:
                import matplotlib.cm as mpl_cm
                cmap = mpl_cm.get_cmap(name)
            except Exception:
                cmap = name
        _CMAP_CACHE[name] = cmap
    if callable(cmap):
        return np.asarray(cmap(values), dtype=np.float64)[..., :3]
    if name == 'autumn_r':
        rgb = np.empty((len(values), 3), dtype=np.float64)
        rgb[:, 0] = 1.0
        rgb[:, 1] = 1.0 - values
        rgb[:, 2] = 0.0
        return rgb
    # Blues：从浅蓝到深蓝，对应 plt.cm.Blues 的大致走向
    light = np.array((0.78, 0.86, 0.94), dtype=np.float64)
    dark = np.array((0.03, 0.19, 0.42), dtype=np.float64)
    return light[None, :] + (dark - light)[None, :] * values[:, None]


def signed_deviation_colors(values, threshold, vmin, vmax):
    """
    按偏差值着色。

    |value| <= threshold 为灰色；
    value < -threshold 用 Blues(0.3 + 0.7 * t)；
    value > threshold 用 autumn_r(t)。
    """
    z = np.asarray(values, dtype=np.float64).reshape(-1)
    rgb = np.broadcast_to(_GRAY, (len(z), 3)).copy()
    threshold = float(threshold)
    vmin = float(vmin)
    vmax = float(vmax)

    mask_neg = z < -threshold
    if np.any(mask_neg):
        span = max(abs(vmin) - threshold, 1e-12)
        neg = np.clip((-z[mask_neg] - threshold) / span, 0.0, 1.0)
        rgb[mask_neg] = _sample_colormap('Blues', 0.3 + 0.7 * neg)

    mask_pos = z > threshold
    if np.any(mask_pos):
        span = max(vmax - threshold, 1e-12)
        pos = np.clip((z[mask_pos] - threshold) / span, 0.0, 1.0)
        rgb[mask_pos] = _sample_colormap('autumn_r', pos)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def compute_heatmap_scale(values_mm, neutral_mm=DEFAULT_NEUTRAL_MM):
    """对称色标：灰色 ±5 mm，上下限取 |偏差| 的 98 分位（至少 ±10 mm）。"""
    values = np.asarray(values_mm, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        limit = 10.0
    else:
        limit = max(10.0, float(np.percentile(np.abs(finite), 98.0)))
    threshold = max(0.1, min(float(neutral_mm), limit * 0.5))
    return float(threshold), -float(limit), float(limit)


def signed_deviation_colors_bgr(values, threshold, vmin, vmax):
    rgb = signed_deviation_colors(values, threshold, vmin, vmax)
    return np.clip(rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)


def colorbar_ticks(threshold, vmin, vmax):
    span = max(float(vmax) - float(vmin), 1e-6)
    candidates = (
        vmax,
        vmax * 0.5 if vmax > 0 else None,
        threshold,
        0.0,
        -threshold,
        vmin * 0.5 if vmin < 0 else None,
        vmin,
    )
    ticks = []
    for value in candidates:
        if value is None:
            continue
        value = float(value)
        if value < vmin - 1e-6 or value > vmax + 1e-6:
            continue
        if any(abs(value - existing) < max(0.2, span * 0.05) for existing, _ in ticks):
            continue
        if abs(value) < 1e-6:
            ticks.append((0.0, '0 mm'))
        elif abs(value) >= 1.0:
            ticks.append((value, f'{value:+.0f} mm'))
        else:
            ticks.append((value, f'{value:+.1f} mm'))
    return ticks


def draw_signed_colorbar(
    image,
    x,
    y,
    bar_width,
    bar_height,
    threshold,
    limit=None,
    *,
    vmin=None,
    vmax=None,
    text_color=(40, 40, 40),
    font_scale=0.72,
    thickness=2,
    output_bgr=True,
):
    """在图像上绘制与热力着色一致的毫米色标。"""
    if vmin is None or vmax is None:
        bound = abs(float(limit if limit is not None else 10.0))
        vmin, vmax = -bound, bound
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    threshold = float(threshold)
    height = max(2, int(bar_height))
    width = max(8, int(bar_width))
    values = np.linspace(vmax, vmin, height, dtype=np.float32)
    bar = signed_deviation_colors(values, threshold, vmin, vmax)
    bar = np.clip(bar * 255.0, 0, 255).astype(np.uint8)
    if output_bgr:
        bar = bar[:, ::-1]
    bar = np.repeat(bar.reshape(height, 1, 3), width, axis=1)
    image[y:y + height, x:x + width] = bar
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height - 1),
        text_color,
        1,
        cv2.LINE_AA,
    )
    span = vmax - vmin
    for value, label in colorbar_ticks(threshold, vmin, vmax):
        ty = int(round(y + (vmax - value) / span * (height - 1)))
        ty = int(np.clip(ty, y + 16, y + height - 8))
        cv2.putText(
            image,
            label,
            (x + width + 10, ty + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
    return image
