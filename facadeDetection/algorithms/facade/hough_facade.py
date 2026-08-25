"""Fast, deterministic Hough voting primitives for facade candidates.

The functions in this module only propose directions and signed plane offsets;
they deliberately do not label a candidate as a facade.  The caller must still
apply normal, residual, occupancy and plane-quality checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HoughPeak:
    value: float
    support: int
    prominence: float = 0.0


def _canonical_normals(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    length = np.linalg.norm(n, axis=1)
    valid = np.isfinite(n).all(axis=1) & (length > 1e-8)
    out = n[valid] / length[valid, None]
    if not len(out):
        return out
    flip = (out[:, 2] < 0) | ((np.abs(out[:, 2]) < 1e-8) & (out[:, 0] < 0))
    out[flip] *= -1.0
    return out


def normal_hough_peaks(normals, min_support=80, bin_deg=2.0,
                       merge_deg=5.0, max_peaks=8, vertical_nz=0.40):
    """Return dominant unoriented normal directions and their support counts."""
    raw = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    unit = _canonical_normals(raw)
    if not len(unit):
        return []
    # Vertical facade candidates only; keep the index-free voting primitive cheap.
    unit = unit[np.abs(unit[:, 2]) <= float(vertical_nz)]
    if not len(unit):
        return []
    step = np.deg2rad(np.clip(float(bin_deg), 0.5, 10.0))
    theta = (np.arctan2(unit[:, 1], unit[:, 0]) + np.pi) / step
    phi = np.arccos(np.clip(unit[:, 2], -1., 1.)) / step
    ntheta = max(4, int(np.ceil(2 * np.pi / step)))
    keys = np.floor(theta).astype(np.int64) + ntheta * np.floor(phi).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    peaks = []
    cos_merge = np.cos(np.deg2rad(max(float(merge_deg), float(bin_deg))))
    threshold = max(int(min_support), int(len(unit) * 0.001))
    for pos in order:
        if counts[pos] < threshold or len(peaks) >= int(max_peaks):
            break
        key = unique[pos]
        cell = np.floor(phi).astype(np.int64) * ntheta + np.floor(theta).astype(np.int64)
        members = unit[cell == key]
        if not len(members):
            continue
        direction = np.mean(members, axis=0)
        direction /= np.linalg.norm(direction) + 1e-12
        if any(abs(direction @ p["normal"]) >= cos_merge for p in peaks):
            continue
        support_mask = np.abs(unit @ direction) >= np.cos(np.deg2rad(max(merge_deg, bin_deg)))
        peaks.append({"normal": direction, "support": int(np.count_nonzero(support_mask))})
    return peaks


def rho_hough_peaks(points, normal, bin_size=0.05, min_support=80,
                    min_peak_distance=0.15, prominence_ratio=0.20,
                    max_peaks=16):
    """Find prominent parallel-plane offsets ``rho = dot(point, normal)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n /= np.linalg.norm(n) + 1e-12
    rho = pts @ n
    rho = rho[np.isfinite(rho)]
    if len(rho) < max(3, int(min_support)):
        return []
    lo, hi = np.percentile(rho, [0.5, 99.5])
    if hi <= lo:
        return []
    bs = max(float(bin_size), 1e-3)
    bins = max(1, int(np.ceil((hi - lo) / bs)))
    counts, edges = np.histogram(rho, bins=bins, range=(lo, hi))
    if len(counts) >= 3:
        smooth = np.convolve(counts.astype(float), np.array([1., 2., 1.]) / 4., mode='same')
    else:
        smooth = counts.astype(float)
    candidates = []
    # Include histogram boundaries: an almost perfectly planar wall produces a
    # narrow peak at the lower/upper rho edge, and skipping edges previously
    # made Hough return no seed for exactly the easiest wall case.
    peak_indices = range(len(smooth))
    for i in peak_indices:
        left = smooth[i - 1] if i > 0 else 0.0
        right = smooth[i + 1] if i + 1 < len(smooth) else 0.0
        if smooth[i] < left or smooth[i] < right:
            continue
        # For a 1-D histogram, use the immediate saddle around the peak.
        # Comparing against the global left/right maxima would make two
        # equally large parallel planes report zero prominence.
        base = max(float(left), float(right))
        prominence = float(smooth[i] - base)
        ratio_ok = smooth[i] >= max(1.0, base) * (1.0 + float(prominence_ratio))
        if counts[i] >= int(min_support) and (prominence >= 2. or ratio_ok):
            value = float((edges[i] + edges[i + 1]) * 0.5)
            candidates.append(HoughPeak(value, int(counts[i]), prominence))
    candidates.sort(key=lambda p: (-p.prominence, -p.support))
    selected = []
    for peak in candidates:
        if any(abs(peak.value - old.value) < float(min_peak_distance) for old in selected):
            continue
        selected.append(peak)
        if len(selected) >= int(max_peaks):
            break
    return sorted(selected, key=lambda p: p.value)