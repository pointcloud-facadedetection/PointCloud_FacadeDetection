from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

RULER_KERNEL_VERSION = "ruler-flatness-3d-2.0"


def make_frame(normal, points, ruler_dir=None):
    normal = np.asarray(normal, dtype=float)
    if ruler_dir is None:
        a = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
        e1 = np.cross(normal, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        xy = (np.asarray(points) - np.mean(points, axis=0)) @ np.column_stack((e1, e2))
        e = np.linalg.eigh(xy.T @ xy)[1][:, -1]
        u = e[0] * e1 + e[1] * e2
    else:
        u = np.asarray(ruler_dir, dtype=float)
        u = u - np.dot(u, normal) * normal
    if np.linalg.norm(u) < 1e-12:
        raise ValueError("靠尺方向与平面法向平行")
    u /= np.linalg.norm(u)
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return u, np.cross(normal, u)


def fit_line(u, w):
    u, w = np.asarray(u, float), np.asarray(w, float)
    if len(u) < 2 or np.ptp(u) < 1e-12:
        raise ValueError("剖面长度不足以拟合参考线")
    m = np.ones(len(u), dtype=bool)
    for _ in range(3):
        a, c = np.polyfit(u[m], w[m], 1)
        r = w - (a * u + c)
        mad = 1.4826 * np.median(np.abs(r[m] - np.median(r[m]))) + 1e-9
        new = np.abs(r) <= max(2.5 * mad, 1e-5)
        if np.sum(new) < 3 or np.array_equal(new, m):
            break
        m = new
    return float(a), float(c)


def sor_mask_local(u, w, k=8, sigma=4.0, subset=None):
    """Linear-time ordered-profile SOR approximation."""
    n = len(w)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    block = 4 * int(k) + 1
    count = (n + block - 1) // block
    padded = np.pad(np.asarray(w, float), (0, count * block - n), mode='edge')
    dev = np.abs(padded.reshape(count, block) -
                 np.median(padded.reshape(count, block), axis=1)[:, None]).ravel()[:n]
    values = dev if subset is None else dev[np.asarray(subset)]
    med = np.median(values)
    threshold = med + sigma * (1.4826 * np.median(np.abs(values - med)) + 1e-12)
    keep = np.ones(n, dtype=bool)
    if subset is None:
        keep = dev <= threshold
    else:
        sel = np.asarray(subset)
        keep[sel] = values <= threshold
    return keep


def sor_mask_grid(u, w, k=8, sigma=4.0, subset=None, w_weight=50.0):
    """Grid-based SOR using 2D density estimation."""
    n = len(w)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    
    x = u.astype(np.float64)
    y = (w * w_weight).astype(np.float64)
    
    xlo, xhi = np.percentile(x, [0.1, 99.9])
    ylo, yhi = np.percentile(y, [0.1, 99.9])
    rx, ry = max(xhi - xlo, 1e-12), max(yhi - ylo, 1e-12)
    cell = np.sqrt(k * rx * ry / n)
    cell = max(cell, max(rx, ry) / 1e6)
    nx = int(rx / cell) + 3
    ny = int(ry / cell) + 3
    
    ix = np.clip(((x - xlo) / cell).astype(np.int64) + 1, 0, nx - 1)
    iy = np.clip(((y - ylo) / cell).astype(np.int64) + 1, 0, ny - 1)
    cnt = np.bincount(ix * ny + iy, minlength=nx * ny).reshape(nx, ny)
    
    csx = np.cumsum(np.pad(cnt, ((1, 1), (0, 0))), axis=0)
    sx = csx[2:] - csx[:-2] + cnt
    csy = np.cumsum(np.pad(sx, ((0, 0), (1, 1))), axis=1)
    S = csy[:, 2:] - csy[:, :-2] + sx
    
    m = S[ix, iy].astype(np.float64)
    rho = np.maximum(m - 1.0, 0.5) / (9.0 * cell * cell)
    md = (2.0 / 3.0) * np.sqrt(k / (np.pi * rho))
    
    if subset is not None:
        sel = np.asarray(subset)
        mds = md[sel]
    else:
        sel, mds = None, md
    
    med = np.median(mds)
    threshold = med + sigma * (1.4826 * np.median(np.abs(mds - med)) + 1e-12)
    keep = np.ones(n, dtype=bool)
    if sel is None:
        keep = md <= threshold
    else:
        keep[sel] = mds <= threshold
    return keep


def sor_mask_exact(u, w, k=8, sigma=4.0, subset=None, w_weight=50.0):
    """Exact k-nearest neighbor SOR using sliding window."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(w)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    W = max(2 * k, 8)
    up = np.empty(n + 2 * W, np.float32)
    up[:W] = 1e12
    up[-W:] = 1e12
    up[W:W + n] = u
    wp = np.zeros(n + 2 * W, np.float32)
    wp[W:W + n] = w * w_weight
    ctr = np.arange(n) if subset is None else np.asarray(subset)
    Vu = sliding_window_view(up, 2 * W + 1)
    Vw = sliding_window_view(wp, 2 * W + 1)
    md = np.empty(ctr.size, np.float32)
    chunk = 500_000
    for s0 in range(0, ctr.size, chunk):
        r = ctr[s0:s0 + chunk]
        d2 = Vu[r] - up[W + r, None]
        d2 *= d2
        dw = Vw[r] - wp[W + r, None]
        dw *= dw
        d2 += dw
        del dw
        d2[:, W] = np.inf
        md[s0:s0 + chunk] = np.sqrt(
            np.partition(d2, k - 1, axis=1)[:, :k]).mean(axis=1)
    med = np.median(md)
    threshold = med + sigma * (1.4826 * np.median(np.abs(md - med)) + 1e-12)
    keep = np.ones(n, dtype=bool)
    keep[ctr] = md <= threshold
    return keep


def upper_hull(u, w):
    stack = []
    for i in range(len(u)):
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            cross = (u[b]-u[a])*(w[i]-w[a]) - (w[b]-w[a])*(u[i]-u[a])
            if cross >= 0:
                stack.pop()
            else:
                break
        stack.append(i)
    return np.asarray(stack, dtype=np.int64)


@dataclass(frozen=True)
class Surface:
    normal: np.ndarray
    centroid: np.ndarray
    e_u: np.ndarray
    e_v: np.ndarray
    u: np.ndarray
    w: np.ndarray
    source_ids: np.ndarray
    keep: np.ndarray
    surf: np.ndarray
    bin_size: float
    u_span: tuple
    top_u: np.ndarray
    top_w: np.ndarray
    top_source_ids: np.ndarray
    holes: np.ndarray
    ref_line: tuple
    stats: dict = field(default_factory=dict)

    def to_world(self, u, w=0.0, v=0.0):
        return (self.centroid + np.asarray(u)[..., None] * self.e_u +
                np.asarray(v)[..., None] * self.e_v +
                np.asarray(w)[..., None] * self.normal)

    def compact(self):
        """Drop full profile arrays before cross-process transfer."""
        e = np.empty(0)
        return Surface(self.normal, self.centroid, self.e_u, self.e_v,
                       e, e, e.astype(np.int64), e.astype(bool), e.astype(bool),
                       self.bin_size, self.u_span,
                       self.top_u, self.top_w, self.top_source_ids, 
                       self.holes, self.ref_line, self.stats)


def prepare_surface(points, source_ids=None, ruler_dir=None, outward=(0, 0, 1),
                    select_band=.01, hole_band=.02, bin_size=.04,
                    top_q=1.0, sor=None, plane_model=None,
                    plane_origin=None):
    # Validate and filter input
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 3:
        raise ValueError("点数不足")
    
    valid_mask = np.all(np.isfinite(p), axis=1)
    if not np.any(valid_mask):
        raise ValueError("所有点坐标均无效（含 nan/inf）")
    p = p[valid_mask]
    
    ids = np.arange(len(p), dtype=np.int64) if source_ids is None else np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if not np.all(valid_mask) and source_ids is not None:
        ids = ids[valid_mask]
    if len(ids) != len(p):
        raise ValueError("source_ids 与 points 长度不一致")
    
    # Validate plane_model
    model = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if model.size != 4 or not np.all(np.isfinite(model)):
        raise ValueError("质量靠尺算法必须提供前序立面 plane_model")
    
    norm = np.linalg.norm(model[:3])
    if norm < 1e-12:
        raise ValueError("plane_model 法向长度为零")
    normal = model[:3] / norm
    d = float(model[3]) / norm
    
    if np.dot(normal, np.asarray(outward, dtype=float)) < 0:
        normal, d = -normal, -d
    
    centroid = (-d * normal if plane_origin is None else
                np.asarray(plane_origin, dtype=np.float64))
    centroid = centroid - (np.dot(centroid, normal) + d) * normal
    
    eu, ev = make_frame(normal, p, ruler_dir)
    local = (p - centroid) @ np.column_stack((eu, ev, normal))
    order = np.argsort(local[:, 0], kind="stable")
    u, w, ids = local[order, 0], local[order, 2], ids[order]
    
    # Store full u_span before any filtering
    u_min_full = float(u.min())
    u_max_full = float(u.max())
    
    a, c = fit_line(u, w)
    residual = w - (a * u + c)
    keep = np.ones(len(u), dtype=bool)
    
    sp = dict(enable=True, sigma=4.0, method="local", k=8, w_weight=50.0)
    sp.update(sor or {})
    
    if sp.get("enable", True) and len(u) > 16:
        method = sp.get("method", "local")
        k = int(sp.get("k", 8))
        sigma = float(sp["sigma"])
        w_weight = float(sp.get("w_weight", 50.0))
        
        if method == "local":
            keep = sor_mask_local(u, w, k, sigma)
        elif method == "grid":
            keep = sor_mask_grid(u, w, k, sigma, w_weight=w_weight)
        elif method == "exact":
            keep = sor_mask_exact(u, w, k, sigma, w_weight=w_weight)
        else:
            keep = sor_mask_local(u, w, k, sigma)
            
        if np.sum(keep) < 3:
            keep[:] = True
    
    a, c = fit_line(u[keep], w[keep])
    residual = w - (a*u+c)
    surf = keep & (np.abs(residual) <= select_band)
    if np.sum(surf) < 3:
        raise ValueError("表面点不足，请放宽 select_band")
    
    # Hole detection with proper hole_band >= select_band check
    effective_hole_band = max(hole_band, select_band)
    u0 = float(u[0])
    bins = np.floor((u-u0)/bin_size).astype(np.int64)
    solid_mask = np.zeros(int(bins[-1]) + 1, dtype=bool)
    solid_mask[bins[keep & (np.abs(residual) <= effective_hole_band)]] = True
    solid = np.flatnonzero(solid_mask)
    selected = np.flatnonzero(surf & np.isin(bins, solid))
    
    # Correct top_q implementation
    tops = []
    for b in solid:
        ix = selected[bins[selected] == b]
        if len(ix):
            if top_q >= 1.0:
                tops.append(ix[np.argmax(w[ix])])
            else:
                segment_w = w[ix]
                target_w = np.quantile(segment_w, top_q)
                closest_idx = ix[np.argmin(np.abs(segment_w - target_w))]
                tops.append(closest_idx)
    
    tops = np.asarray(tops, dtype=np.int64)
    if len(tops) < 2:
        raise ValueError("实区分段不足以放置靠尺")
    
    # Holes
    pad = np.r_[True, solid_mask, True]
    hs = np.flatnonzero(~pad[1:-1] & pad[:-2])
    he = np.flatnonzero(~pad[1:-1] & pad[2:]) + 1
    holes = (np.column_stack((u0 + hs * bin_size, u0 + he * bin_size))
             if hs.size else np.empty((0, 2), dtype=float))
    
    stats = {
        "n_points": len(p),
        "n_used": int(np.sum(surf)),
        "n_sor_removed": int(np.sum(~keep)),
        "n_solid_bins": int(solid_mask.sum()),
        "hole_length": float(np.diff(holes, axis=1).sum()) if holes.size else 0.,
        "kernel_version": RULER_KERNEL_VERSION
    }
    
    return Surface(
        normal, centroid, eu, ev, u, w, ids, keep, surf,
        float(bin_size), (u_min_full, u_max_full),
        u[tops], w[tops], ids[tops], holes, (a, c), stats)


def ruler_at(surface: Surface, u_center=None, ruler_length=2.0, max_hole_ratio=0.2):
    """Compute ruler flatness at given u_center position.
    
    Returns cleaned dict with only essential fields for quality aggregation.
    """
    if u_center is None:
        u_center = float(0.5 * (surface.u_span[0] + surface.u_span[1]))
    
    lo, hi = np.searchsorted(surface.top_u, [u_center-ruler_length/2, u_center+ruler_length/2])
    tu, tw = surface.top_u[lo:hi], surface.top_w[lo:hi]
    ts = surface.top_source_ids[lo:hi]
    
    if len(tu) < 2:
        return {
            "ok": False, 
            "reason": "靠尺范围内实区分段不足", 
            "u_center": float(u_center),
            "coverage_valid": False,
            "hole_ratio": 1.0,
        }
    
    hull = upper_hull(tu, tw)
    # Center of gravity moment balance: find pivot pair straddling u_center
    cg = float(np.clip(u_center, tu[hull[0]], tu[hull[-1]]))
    j = int(np.clip(np.searchsorted(tu[hull], cg, side="right"), 1, len(hull)-1))
    p, q = hull[j-1], hull[j]
    slope = (tw[q]-tw[p])/(tu[q]-tu[p])
    intercept = tw[p]-slope*tu[p]
    
    # Gap: positive = depression (ruler above surface), negative = protrusion
    # Consistent with rulermeasure.txt: (a*u + c - w) / hypot(1, a)
    gap = (slope * tu + intercept - tw) / np.hypot(1, slope)
    gi = int(np.argmax(gap))
    
    # Hole coverage
    overlap = (np.clip(np.minimum(surface.holes[:, 1], u_center+ruler_length/2) -
                       np.maximum(surface.holes[:, 0], u_center-ruler_length/2), 0, None)
               if surface.holes.size else np.empty(0))
    hole_length = float(overlap.sum())
    hole_ratio = hole_length / ruler_length
    coverage_valid = hole_ratio <= max_hole_ratio
    
    signed_gap = float(gap[gi])
    
    # Return minimal essential fields only
    return {
        "ok": True,
        "u_center": float(u_center),
        "gap_mm": float(max(signed_gap, 0.0)) * 1000.0,  # defect value in mm (always >= 0)
        "signed_gap_mm": signed_gap * 1000.0,  # signed for advanced analysis
        "depression_u": float(tu[gi]),
        "depression_xyz": surface.to_world(tu[gi], tw[gi]),
        "depression_source_id": int(ts[gi]),
        "angle_deg": float(np.degrees(np.arctan(abs(slope)))),
        "slope": float(slope),
        "pivots_u": (float(tu[p]), float(tu[q])),
        "pivots_xyz": np.vstack((surface.to_world(tu[p], tw[p]), surface.to_world(tu[q], tw[q]))),
        "pivot_source_ids": (int(ts[p]), int(ts[q])),
        "n_segments": int(len(tu)),
        "n_hull": int(len(hull)),
        "hole_length": hole_length,
        "hole_ratio": hole_ratio,
        "coverage_valid": coverage_valid,
        "covered_source_ids": ts.copy(),
        "kernel_version": RULER_KERNEL_VERSION
    }


def ruler_flatness(points, source_ids=None, center=None, **kwargs):
    ruler_length = kwargs.pop("ruler_length", 2.0)
    max_hole_ratio = kwargs.pop("max_hole_ratio", 0.2)
    surface = prepare_surface(points, source_ids=source_ids, **kwargs)
    uc = None if center is None else float((np.asarray(center)-surface.centroid) @ surface.e_u)
    result = ruler_at(surface, uc, ruler_length, max_hole_ratio)
    result["surface"] = surface
    return result