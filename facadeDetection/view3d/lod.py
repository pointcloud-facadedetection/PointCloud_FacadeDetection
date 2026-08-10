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

def display_arrays(data):
    return data["pos"], data["color"]


