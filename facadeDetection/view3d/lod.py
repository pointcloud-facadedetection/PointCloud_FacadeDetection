import math

import numpy as np


def normalize_colors(colors, count):
    if colors is None:
        return np.ones((count, 3), dtype=np.float64) * 0.7

    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1 and colors.shape[0] == 3:
        return np.tile(colors, (count, 1))

    colors = colors.reshape(-1, 3)
    colors = np.clip(colors, 0.0, 1.0)

    if len(colors) != count:
        fallback = np.ones((count, 3), dtype=np.float64) * 0.7
        fallback[: min(len(colors), count)] = colors[: min(len(colors), count)]
        return fallback

    return colors


def build_lod_indices(count, max_points):
    if count <= max_points:
        return None
    step = int(math.ceil(count / max_points))
    return np.arange(0, count, step, dtype=np.int64)


def display_arrays(data):
    idx = data.get("render_indices")
    if idx is None:
        return data["pos"], data["color"]
    return data["pos"][idx], data["color"][idx]


def sample_step(count, max_points):
    return max(1, int(math.ceil(count / max_points)))
