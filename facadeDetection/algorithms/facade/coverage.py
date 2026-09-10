"""UV-domain coverage masks for deduplicated pass-rate computation.

The ruler (米字/I字) method sweeps overlapping windows across a facade, so a
plain arithmetic sum of per-window area double-counts the overlap.  A boolean
grid over the UV plane makes the union of covered cells idempotent, which
guarantees pass-rate in [0, 1].
"""
from __future__ import annotations

import numpy as np


class CoverageMask:
    """Discretise the UV plane and accumulate a boolean occupancy mask.

    Parameters
    ----------
    u0, v0, u1, v1 : float
        Bounding box of the facade domain in the projection (u, v) frame.
    resolution : float
        Grid cell size in metres (default 1 cm).
    """

    def __init__(self, u0, v0, u1, v1, resolution=0.01):
        self.u0 = float(u0)
        self.v0 = float(v0)
        self.u1 = float(u1)
        self.v1 = float(v1)
        self.res = max(float(resolution), 1e-4)
        self.nu = max(1, int(np.ceil((self.u1 - self.u0) / self.res)))
        self.nv = max(1, int(np.ceil((self.v1 - self.v0) / self.res)))
        self.mask = np.zeros((self.nu, self.nv), dtype=bool)

    def _clip_cell(self, u_lo, v_lo, u_hi, v_hi):
        i0 = int(max(0, (u_lo - self.u0) / self.res))
        i1 = int(min(self.nu - 1, np.floor((u_hi - self.u0) / self.res)))
        j0 = int(max(0, (v_lo - self.v0) / self.res))
        j1 = int(min(self.nv - 1, np.floor((v_hi - self.v0) / self.res)))
        return i0, i1, j0, j1

    def mark_rect(self, u_lo, v_lo, u_hi, v_hi):
        """Mark a rectangle (window footprint) as occupied."""
        i0, i1, j0, j1 = self._clip_cell(u_lo, v_lo, u_hi, v_hi)
        if i0 <= i1 and j0 <= j1:
            self.mask[i0:i1 + 1, j0:j1 + 1] = True

    def mark_pts(self, u_arr, v_arr):
        """Mark the cells that contain actual points (the true facade area)."""
        u_arr = np.asarray(u_arr, dtype=float).reshape(-1)
        v_arr = np.asarray(v_arr, dtype=float).reshape(-1)
        if u_arr.size == 0:
            return
        i = np.clip(((u_arr - self.u0) / self.res).astype(int), 0, self.nu - 1)
        j = np.clip(((v_arr - self.v0) / self.res).astype(int), 0, self.nv - 1)
        self.mask[i, j] = True

    def area_m2(self):
        """Return the covered area (union of marked cells), in m^2."""
        return float(self.mask.sum()) * self.res * self.res

    def intersection_area_m2(self, other):
        """Area of the intersection with another mask."""
        n = min(self.nu, other.nu)
        m = min(self.nv, other.nv)
        return float(np.count_nonzero(self.mask[:n, :m] & other.mask[:n, :m])) \
            * self.res * self.res


def cell_occupancy(point_count_per_cell, pass_cells):
    """Deduplicate point counts by cell.

    Parameters
    ----------
    point_count_per_cell : (K,) int array
        Number of points owned by each unique cell (already aggregated once).
    pass_cells : (K,) bool array
        Whether the representative window of each cell passed.

    Returns
    -------
    (total_points, pass_points)
    """
    total = int(np.sum(point_count_per_cell))
    passed = int(np.sum(point_count_per_cell[pass_cells]))
    return total, passed