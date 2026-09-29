import math

import numpy as np

try:
    from config.settings import Config
    _FALLBACK_GRAY = float(getattr(Config, 'DEFAULT_CLOUD_GRAY', 0.45))
except Exception:
    _FALLBACK_GRAY = 0.45


def normalize_colors(colors, count):
    if colors is None:
        return np.full((count, 3), _FALLBACK_GRAY, dtype=np.float32)

    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1 and colors.shape[0] == 3:
        return np.tile(colors, (count, 1))

    try:
        colors = colors.reshape(-1, 3)
    except ValueError:
        return np.full((count, 3), _FALLBACK_GRAY, dtype=np.float32)
    if not np.all(np.isfinite(colors)):
        return np.full((count, 3), _FALLBACK_GRAY, dtype=np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    if len(colors) != count:
        fallback = np.full((count, 3), _FALLBACK_GRAY, dtype=np.float32)
        fallback[: min(len(colors), count)] = colors[: min(len(colors), count)]
        return fallback

    return colors

def display_arrays(data):
    return data["pos"], data["color"]