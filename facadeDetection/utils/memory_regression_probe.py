from __future__ import annotations

import gc
import json
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.facade.boundary_adjustment import incremental_proxy_ids
from ui.controllers.facade_quality import FacadeQualityController


def run():
    process = psutil.Process()
    rng = np.random.default_rng(42)
    points = np.column_stack([
        np.zeros(1_500_000, dtype=np.float32),
        rng.random(1_500_000, dtype=np.float32) * 4.0,
        rng.random(1_500_000, dtype=np.float32) * 4.0,
    ])
    geometry = {
        'plane_model': [1.0, 0.0, 0.0, 0.0],
        'origin': [0.0, 0.0, 0.0],
        'u_axis': [0.0, 1.0, 0.0],
        'v_axis': [0.0, 0.0, 1.0],
        'uv_bounds': [0.0, 0.0, 2.0, 2.0],
    }
    old_ids = np.arange(1000, dtype=np.int64)
    gc.collect()
    rss_before = process.memory_info().rss
    tracemalloc.start()
    for extent in (3.0, 3.5, 3.0, 3.5, 3.0):
        result = incremental_proxy_ids(
            points, geometry, old_ids, [0.0, 0.0, extent, extent])
        del result
    _current, boundary_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    boundary_growth = process.memory_info().rss - rss_before

    controller = SimpleNamespace(quality_result_cache={})
    gc.collect()
    rss_before = process.memory_info().rss
    for revision in range(20):
        quality = {'__global_indices': np.arange(1_000_000, dtype=np.int64)}
        FacadeQualityController._store_quality_result(
            controller, ('cloud', 1, revision), 'cloud', 1, quality)
    gc.collect()
    cache_growth = process.memory_info().rss - rss_before
    cached = controller.quality_result_cache[('cloud', 1, 19)]
    result = {
        'proxy_points': len(points),
        'boundary_peak_mib': round(boundary_peak / 1024**2, 2),
        'boundary_rss_growth_mib': round(boundary_growth / 1024**2, 2),
        'quality_cache_entries': len(controller.quality_result_cache),
        'quality_cache_indices': len(cached['__global_indices']),
        'quality_cache_rss_growth_mib': round(cache_growth / 1024**2, 2),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert boundary_peak < 150 * 1024**2, result
    assert len(controller.quality_result_cache) == 1, result
    assert len(cached['__global_indices']) == 1_000_000, result
    assert cache_growth < 30 * 1024**2, result
    return result


if __name__ == '__main__':
    run()
