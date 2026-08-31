import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

EYE_HEIGHT = 1.75          # camera height, held fixed (never touched by auto-fit)


# ------------------------------------------------------------------ loading ----
def load_local_points(ply_file, json_file, voxel_size=0.1):
    """Load points (+ intensity if present) into the LiDAR frame (x fwd, y left, z up)."""
    with open(json_file, "r") as f:
        meta = json.load(f)
    T_local = np.linalg.inv(np.array(meta["transformToGlobal"]))

    try:
        pcd = o3d.t.io.read_point_cloud(ply_file)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        pts = pcd.point["positions"].numpy().astype(np.float64)
    except Exception as e:
        print(f"Tensor PLY read failed ({e}); using legacy reader, no intensity.")
        legacy = o3d.io.read_point_cloud(ply_file)
        if voxel_size > 0:
            legacy = legacy.voxel_down_sample(voxel_size)
        pts = np.asarray(legacy.points).astype(np.float64)
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        return (T_local @ pts_h.T).T[:, :3], None

    try:
        keys = list(pcd.point)
    except Exception:
        keys = []
    keys = list(dict.fromkeys(keys + ["intensity", "intensities",
                                      "scalar_intensity", "Intensity"]))

    intensity = None
    for key in keys:
        if "intens" not in key.lower():
            continue
        try:
            intensity = pcd.point[key].numpy().reshape(-1).astype(np.float32)
            break
        except Exception:
            continue
    
    if intensity is None and "colors" in keys:
        colors = pcd.point["colors"].numpy().astype(np.float32)
        intensity = (
            0.299 * colors[:, 0]
            + 0.587 * colors[:, 1]
            + 0.114 * colors[:, 2]
        )
        print("Using PLY colors as intensity")

    if intensity is None:
        print("No intensity attribute found -> depth colouring will be used")

    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    return (T_local @ pts_h.T).T[:, :3], intensity


# ---------------------------------------------------------------- rotations ----
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------- projection ----
def focal_from_fov(W, H, fov_deg):
    """Focal length in px. FOV is measured on the LONG side, so switching between
    landscape and portrait rotates the same lens instead of changing it."""
    return (max(W, H) / 2) / np.tan(np.radians(fov_deg / 2))


def render_perspective(points, values, W, H, fov_deg, yaw, pitch, roll,
                       tx, ty, tz, near, far, splat=1, center_h=True, center_v=False):
    """Pinhole camera looking down +x. Returns (image with NaN holes, hfov, vfov, n_px)."""
    R_cam = rot_z(np.radians(yaw)) @ rot_y(np.radians(pitch)) @ rot_x(np.radians(roll))
    p = (points - np.array([tx, ty, tz])) @ R_cam

    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    m = (x > near) & (d < far)
    x, y, z, d, val = x[m], y[m], z[m], d[m], values[m]

    img = np.full((H, W), np.nan, np.float32)
    fx = fy = focal_from_fov(W, H, fov_deg)
    hfov = 2 * np.degrees(np.arctan((W / 2) / fx))
    vfov = 2 * np.degrees(np.arctan((H / 2) / fx))
    if x.size == 0:
        return img, hfov, vfov, 0

    px, py = -fx * y / x, -fy * z / x
    # Points very close to the lens produce huge pixel offsets; clip before taking
    # percentiles so a handful of them cannot drag the whole frame sideways.
    pxc, pyc = np.clip(px, -2 * W, 2 * W), np.clip(py, -2 * H, 2 * H)
    cx = W / 2 - 0.5 * (np.percentile(pxc, 5) + np.percentile(pxc, 95)) if center_h else W / 2
    cy = H / 2 - 0.5 * (np.percentile(pyc, 5) + np.percentile(pyc, 95)) if center_v else H / 2

    u = np.round(cx + px).astype(np.int32)
    v = np.round(cy + py).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d, val = u[inb], v[inb], d[inb], val[inb]

    order = np.argsort(d)[::-1]          # far first, near overwrites
    u, v, val = u[order], v[order], val[order]

    r = int(splat) // 2
    if r == 0:
        img[v, u] = val
    else:
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                img[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = val
    return img, hfov, vfov, int(u.size)


# ----------------------------------------------------------------- auto-fit ----
def auto_fit_params(points, W, H, cam_z=EYE_HEIGHT, hfov_list=(100, 110, 120, 130),
                    back_range=(1.0, 3.0), top_margin=0.07, near=0.3, core_pct=88):
    """
    Derive yaw / pitch / camera XY / HFOV from the data itself.

    The camera stays essentially where the scanner was: it only steps back
    `back_range` metres (1-3 m by default), at a fixed height of cam_z.

    Framing is decided from the SUBJECT only - the dense core of the cloud - not
    from every last point. Far-flung stray points (the ragged left/right edges of
    a scan, stuff seen through a doorway, isolated outliers) are excluded, so they
    can no longer drag the view direction or force a silly wide FOV. Points
    outside the frame are expected and fine; a photographer crops too.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], min(120_000, points.shape[0]), replace=False)
    P = points[idx]

    # ---- isolate the subject: trim by radius from the scanner and by height ----
    r = np.linalg.norm(P[:, :2], axis=1)
    z_lo, z_hi = np.percentile(P[:, 2], [0.5, 99.5])
    core = (r <= np.percentile(r, core_pct)) & (P[:, 2] >= z_lo) & (P[:, 2] <= z_hi)
    C = P[core]
    xy, cz_pts = C[:, :2], C[:, 2]

    def evaluate(cam_xy, yaw_deg, fov):
        """-> (fraction of the SUBJECT in frame, pitch in deg, top gap fraction)"""
        d2 = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
        perp = np.array([-d2[1], d2[0]])
        rel = xy - cam_xy
        xf, yf = rel @ d2, rel @ perp
        dz = cz_pts - cam_z
        fx = focal_from_fov(W, H, fov)
        tan_h = (W / 2) / fx                      # actual horizontal half-FOV

        # only points that are actually in front AND within the horizontal FOV
        # get a say in the vertical framing - that is what stops the left/right
        # fringes of the scan from setting the pitch.
        front = (xf > near) & (np.abs(yf) <= xf * tan_h)
        if front.sum() < 100:
            return -1.0, 0.0, 0.0
        e_top = np.percentile(np.arctan2(dz[front], xf[front]), 99.5)
        phi = e_top - np.arctan(((0.5 - top_margin) * H) / fx)
        phi = np.clip(phi, np.radians(-85), np.radians(85))

        xr = xf * np.cos(phi) + dz * np.sin(phi)
        zr = -xf * np.sin(phi) + dz * np.cos(phi)
        ok = xr > near
        u = W / 2 - fx * (yf[ok] / xr[ok])
        v = H / 2 - fx * (zr[ok] / xr[ok])
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inside.sum() < 100:
            return -1.0, np.degrees(phi), 0.0
        gap = float(np.clip(v[inside].min(), 0, H)) / H
        return inside.sum() / xy.shape[0], np.degrees(phi), gap

    # ---- yaw: point at the densest direction, per FOV width ----
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    yaw_grid = np.arange(-180, 180, 5.0)

    def best_yaw(fov):
        half = np.degrees(np.arctan((W / 2) / focal_from_fov(W, H, fov)))
        counts = [np.sum(np.abs((az - y + 180) % 360 - 180) <= half) for y in yaw_grid]
        return float(yaw_grid[int(np.argmax(counts))])

    best = None
    for fov in hfov_list:
        yaw0 = best_yaw(fov)
        for yaw in yaw0 + np.arange(-10, 11, 5.0):        # local refinement
            d2 = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
            for D in np.linspace(back_range[0], back_range[1], 9):
                cam_xy = -d2 * D                          # step back from the origin
                frac, phi_deg, gap = evaluate(cam_xy, yaw, fov)
                if frac < 0:
                    continue
                score = frac + 0.35 * min(gap / top_margin, 1.0)
                cand = dict(hfov=fov, yaw=yaw, pitch=-phi_deg, tx=cam_xy[0],
                            ty=cam_xy[1], D=D, frac=frac, gap=gap, score=score)
                if best is None or score > best["score"]:
                    best = cand

    far = float(np.percentile(np.linalg.norm(C - np.array([best["tx"], best["ty"], cam_z]),
                                             axis=1), 99) * 1.15)
    best["far"] = float(np.clip(far, 10, 300))
    print(f"[auto-fit] HFOV {best['hfov']}°, yaw {best['yaw']:.1f}°, pitch {best['pitch']:.1f}°, "
          f"cam ({best['tx']:.2f}, {best['ty']:.2f}, {cam_z}), stepped back {best['D']:.1f} m, "
          f"subject in frame {best['frac']*100:.1f}%, top gap {best['gap']*100:.1f}%")
    return best


# ------------------------------------------------------------------- viewer ----
def launch_viewer(ply_file, json_file, long_side=1024, short_side=576, voxel_size=0.1):
    points, intensity = load_local_points(ply_file, json_file, voxel_size)
    print(f"Loaded {points.shape[0]} points, intensity: {intensity is not None}")

    if intensity is not None:
        lo, hi = np.percentile(intensity, [1, 99])
        inten01 = np.clip((intensity - lo) / (hi - lo + 1e-9), 0, 1)
    else:
        inten01 = None

    S = dict(orient="Landscape", color="Intensity" if inten01 is not None else "Depth",
             cmap="gray", center_h=True, center_v=False)

    def dims():
        return (long_side, short_side) if S["orient"] == "Landscape" else (short_side, long_side)

    W0, H0 = dims()
    fit = auto_fit_params(points, W0, H0)
    P = dict(hfov=fit["hfov"], yaw=fit["yaw"], pitch=fit["pitch"], roll=0.0,
             tx=fit["tx"], ty=fit["ty"], tz=EYE_HEIGHT,
             near=0.3, far=fit["far"], gamma=1.0, splat=3)

    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0.03, right=0.70, top=0.94, bottom=0.04)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.full((H0, W0), np.nan), cmap=S["cmap"], vmin=0, vmax=1,
                   aspect="equal", interpolation="nearest")
    im.cmap.set_bad("black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    title = ax.set_title("")

    span = float(np.percentile(np.linalg.norm(points[:, :2], axis=1), 99)) * 4 + 20
    sliders = {}
    specs = [
        ("hfov",  "FOV, long side (deg)",         40, 170, 1),
        ("yaw",   "Yaw / pan (deg)",             -180, 180, 1),
        ("pitch", "Pitch / tilt (deg)  (- = up)", -90, 90, 1),
        ("roll",  "Roll (deg)",                  -180, 180, 1),
        ("tx",    "Camera X (m)",                -span, span, 0.1),
        ("ty",    "Camera Y (m)",                -span, span, 0.1),
        ("near",  "Near clip (m)",                0.1, 10, 0.1),
        ("far",   "Far clip (m)",                 5, 300, 1),
        ("gamma", "Colour gamma",                 0.2, 3.0, 0.05),
        ("splat", "Point size (px)",              1, 15, 2),
    ]
    x0, w, h, y = 0.80, 0.17, 0.022, 0.90
    for key, label, lo_, hi_, st in specs:
        sax = fig.add_axes([x0, y, w, h])
        sliders[key] = Slider(sax, label, lo_, hi_, valinit=P[key], valstep=st)
        y -= 0.045

    y -= 0.02
    ax_or = fig.add_axes([x0, y - 0.07, w, 0.08]); ax_or.set_title("Orientation", fontsize=9)
    r_orient = RadioButtons(ax_or, ("Landscape", "Portrait"))

    ax_col = fig.add_axes([x0, y - 0.16, w, 0.08]); ax_col.set_title("Colour by", fontsize=9)
    r_color = RadioButtons(ax_col, ("Intensity", "Depth"),
                           active=0 if S["color"] == "Intensity" else 1)

    ax_cm = fig.add_axes([x0, y - 0.29, w, 0.12]); ax_cm.set_title("Colormap", fontsize=9)
    r_cmap = RadioButtons(ax_cm, ("gray", "turbo", "viridis", "magma"))

    ax_ck = fig.add_axes([x0, y - 0.37, w, 0.07])
    ck = CheckButtons(ax_ck, ["Center H", "Center V"], [S["center_h"], S["center_v"]])

    ax_af = fig.add_axes([x0, y - 0.42, w, 0.04])
    btn_af = Button(ax_af, "Auto-fit")

    updating = {"busy": False}

    def redraw(_=None):
        if updating["busy"]:
            return
        for k in sliders:
            P[k] = sliders[k].val
        P["tz"] = EYE_HEIGHT                      # z is never touched
        W, H = dims()

        if S["color"] == "Intensity" and inten01 is not None:
            values, label = inten01, "Intensity (normalised)"
        else:
            values = np.clip(np.linalg.norm(points - np.array([P["tx"], P["ty"], P["tz"]]),
                                            axis=1) / P["far"], 0, 1)
            label = f"Depth / {P['far']:.0f} m"

        img, hfov, vfov, n = render_perspective(points, values, W, H, P["hfov"], P["yaw"],
                                                P["pitch"], P["roll"], P["tx"], P["ty"], P["tz"],
                                                P["near"], P["far"], P["splat"],
                                                S["center_h"], S["center_v"])
        img = np.power(img, P["gamma"])
        im.set_data(np.ma.masked_invalid(img))
        im.set_extent((-0.5, W - 0.5, H - 0.5, -0.5))
        im.set_cmap(S["cmap"]); im.cmap.set_bad("black")
        im.set_clim(0, 1)
        cbar.set_label(label)
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        back = np.hypot(P["tx"], P["ty"])
        title.set_text(f"{S['orient']} {W}x{H}   FOV {P['hfov']:.0f}° (H {hfov:.0f}° / V {vfov:.0f}°)   "
                       f"yaw {P['yaw']:.0f}°  pitch {P['pitch']:.0f}°  roll {P['roll']:.0f}°   "
                       f"eye {EYE_HEIGHT} m, back {back:.1f} m   {n} px")
        fig.canvas.draw_idle()

    def apply_fit(_=None):
        W, H = dims()
        f = auto_fit_params(points, W, H)
        updating["busy"] = True
        for k in ("hfov", "yaw", "pitch", "tx", "ty", "far"):
            s = sliders[k]
            s.set_val(float(np.clip(f[k], s.valmin, s.valmax)))
        sliders["roll"].set_val(0.0)
        updating["busy"] = False
        redraw()

    def on_orient(lbl):
        S["orient"] = lbl
        apply_fit()                               # re-fit for the new aspect ratio

    def on_color(lbl): S["color"] = lbl; redraw()
    def on_cmap(lbl):  S["cmap"] = lbl; redraw()

    def on_check(lbl):
        S["center_h" if lbl == "Center H" else "center_v"] ^= True
        redraw()

    for s in sliders.values():
        s.on_changed(redraw)
    r_orient.on_clicked(on_orient)
    r_color.on_clicked(on_color)
    r_cmap.on_clicked(on_cmap)
    ck.on_clicked(on_check)
    btn_af.on_clicked(apply_fit)

    def on_key(ev):
        step = {"left": ("yaw", 5), "right": ("yaw", -5),
                "up": ("pitch", -3), "down": ("pitch", 3),
                "w": ("tx", 1), "s": ("tx", -1),
                "a": ("ty", 1), "d": ("ty", -1)}
        if ev.key in step:
            k, dv = step[ev.key]
            s = sliders[k]
            s.set_val(float(np.clip(s.val + dv, s.valmin, s.valmax)))
        elif ev.key == "r":
            r_orient.set_active(1 if S["orient"] == "Landscape" else 0)
        elif ev.key == "f":
            apply_fit()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    launch_viewer("C:/Users/Gao Liying/Downloads/ply/ply/bllygg01.fls/bllygg01.ply", "../data/bllygg01.json",
                  long_side=1024, short_side=576, voxel_size=0.1)
