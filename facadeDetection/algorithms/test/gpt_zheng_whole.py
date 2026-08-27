import cv2
import numpy as np
import math
import os

# ============================================================
# Whole-building perspective upright correction
# ------------------------------------------------------------
# Goal:
#   - Do NOT fronto-parallelize any single facade.
#   - Keep the original left/right viewing perspective.
#   - Only remove camera pitch/roll induced vertical convergence.
#   - Keep the full input frame content (no building-boundary crop).
#
# Core idea:
#   1) Detect many near-vertical building lines with LSD.
#   2) Estimate the vertical vanishing point with RANSAC.
#   3) Treat correction as a pure virtual-camera rotation:
#        H = K * R * K^{-1}
#      which sends the vertical vanishing point to infinity.
#   4) Warp the ENTIRE image, preserving the original side-view geometry.
# ============================================================


# -----------------------------
# User parameters
# -----------------------------
IMAGE_PATH = "../data/southeast.jpg"     # change to your image path
OUTPUT_PATH = r"whole_building_upright.jpg"
DEBUG_PATH = r"debug_vertical_lines.jpg"

# Approximate horizontal field of view of the camera.
# Smartphone normal lens: often 55~75 deg
# Smartphone wide-angle: often 75~100 deg
# If unsure, start with 75.0.
HFOV_DEG = 75.0

# Line detection / RANSAC parameters
MIN_LINE_LENGTH_RATIO = 0.035
MAX_VERTICAL_ANGLE_DEVIATION_DEG = 35.0
VP_RANSAC_ANGLE_THRESHOLD_DEG = 1.8
VP_RANSAC_ITERATIONS = 7000

# Avoid excessive output dimensions.
MAX_OUTPUT_SIDE = 4500

# Draw only RANSAC inlier vertical lines in the debug image.
DRAW_ONLY_INLIERS = True


# ============================================================
# Unicode-safe image I/O for Windows Chinese paths
# ============================================================
def imread_unicode(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image does not exist: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".jpg"
        path = path + ext
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Cannot save image: {path}")
    buf.tofile(path)


# ============================================================
# Geometry helpers
# ============================================================
def normalize_line(line):
    line = np.asarray(line, dtype=np.float64)
    n = math.hypot(line[0], line[1])
    if n < 1e-12:
        return line
    return line / n


def segment_to_line(x1, y1, x2, y2):
    p1 = np.array([x1, y1, 1.0], dtype=np.float64)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64)
    return normalize_line(np.cross(p1, p2))


def line_intersection(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-12:
        return None
    return p / p[2]


def angle_residual_to_vp(item, vp):
    """
    Angular residual between:
      - detected line direction
      - direction from segment midpoint to candidate VP

    This is much more stable than pixel point-to-line distance when
    the vanishing point is far outside the image.
    """
    x1, y1, x2, y2 = item["segment"]
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    d_line = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    d_vp = np.array([vp[0] - mx, vp[1] - my], dtype=np.float64)

    n1 = np.linalg.norm(d_line)
    n2 = np.linalg.norm(d_vp)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0

    d_line /= n1
    d_vp /= n2

    # line is unoriented, so use absolute dot product
    c = np.clip(abs(np.dot(d_line, d_vp)), 0.0, 1.0)
    return math.degrees(math.acos(c))


# ============================================================
# 1. Detect long near-vertical image line segments
# ============================================================
def detect_vertical_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight denoising improves LSD stability without destroying edges.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)[0]

    if detected is None:
        raise RuntimeError("No line segments detected.")

    h, w = image.shape[:2]
    min_length = min(h, w) * MIN_LINE_LENGTH_RATIO

    candidates = []

    for seg in detected[:, 0, :]:
        x1, y1, x2, y2 = map(float, seg)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length < min_length:
            continue

        # 0 deg = horizontal, 90 deg = vertical
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        dev = abs(angle - 90.0)

        if dev > MAX_VERTICAL_ANGLE_DEVIATION_DEG:
            continue

        line = segment_to_line(x1, y1, x2, y2)

        candidates.append({
            "segment": np.array([x1, y1, x2, y2], dtype=np.float64),
            "line": line,
            "length": length,
            "angle": angle,
        })

    if len(candidates) < 2:
        raise RuntimeError(
            "Too few vertical line candidates. "
            "Try increasing MAX_VERTICAL_ANGLE_DEVIATION_DEG or "
            "decreasing MIN_LINE_LENGTH_RATIO."
        )

    return candidates


# ============================================================
# 2. Robustly estimate the vertical vanishing point
# ============================================================
def estimate_vertical_vp(lines, image_shape):
    h, w = image_shape[:2]
    rng = np.random.default_rng(0)

    lengths = np.array([item["length"] for item in lines], dtype=np.float64)
    probabilities = lengths / lengths.sum()

    best_score = -1.0
    best_inliers = None
    best_vp = None

    n = len(lines)

    for _ in range(VP_RANSAC_ITERATIONS):
        i, j = rng.choice(n, size=2, replace=False, p=probabilities)

        vp = line_intersection(lines[i]["line"], lines[j]["line"])
        if vp is None:
            continue

        # Avoid absurd numerical intersections.
        if not np.all(np.isfinite(vp)):
            continue
        if abs(vp[0]) > 100 * w or abs(vp[1]) > 100 * h:
            continue

        residuals = np.array(
            [angle_residual_to_vp(item, vp) for item in lines],
            dtype=np.float64,
        )

        inliers = residuals < VP_RANSAC_ANGLE_THRESHOLD_DEG

        # Reward long line segments more.
        score = lengths[inliers].sum()

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_vp = vp

    if best_inliers is None or best_inliers.sum() < 2:
        raise RuntimeError("Failed to estimate a stable vertical vanishing point.")

    # Weighted least-squares refinement using all inlier lines:
    # solve L v = 0 in homogeneous coordinates.
    L = np.array(
        [item["line"] for item, flag in zip(lines, best_inliers) if flag],
        dtype=np.float64,
    )
    W = np.sqrt(
        np.array(
            [item["length"] for item, flag in zip(lines, best_inliers) if flag],
            dtype=np.float64,
        )
    )[:, None]

    _, _, Vt = np.linalg.svd(L * W, full_matrices=False)
    vp = Vt[-1]

    if abs(vp[2]) < 1e-12:
        # Already almost vertical in the image; use the RANSAC result.
        vp = best_vp
    else:
        vp = vp / vp[2]

    # Recompute inliers after refinement.
    residuals = np.array(
        [angle_residual_to_vp(item, vp) for item in lines],
        dtype=np.float64,
    )
    inliers = residuals < VP_RANSAC_ANGLE_THRESHOLD_DEG

    return vp, inliers, residuals


# ============================================================
# 3. Build a physically meaningful pure-rotation homography
# ============================================================
def rotation_from_a_to_b(a, b):
    """Return a 3x3 rotation matrix mapping unit vector a to unit vector b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float64)

        # 180-degree case: choose any axis orthogonal to a.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis - np.dot(axis, a) * a
        axis /= np.linalg.norm(axis)

        Kx = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ], dtype=np.float64)
        return np.eye(3) + 2.0 * (Kx @ Kx)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)

    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    return R


def build_upright_homography(image_shape, vp_vertical, hfov_deg):
    h, w = image_shape[:2]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Approximate focal length from horizontal FOV.
    f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K)

    # Camera-coordinate direction corresponding to the vertical VP.
    g = K_inv @ np.asarray(vp_vertical, dtype=np.float64)
    g /= np.linalg.norm(g)

    # Vanishing directions are sign-ambiguous. Choose the one pointing
    # roughly downward in image/camera Y so the result does not flip.
    if g[1] < 0:
        g = -g

    target_vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Minimum 3D rotation that makes world-vertical parallel to image Y.
    R = rotation_from_a_to_b(g, target_vertical)

    H = K @ R @ K_inv
    H = H / H[2, 2]

    return H, K, R


# ============================================================
# 4. Warp the entire original frame without cropping source pixels
# ============================================================
def transform_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    q = (H @ hom.T).T
    valid = np.abs(q[:, 2]) > 1e-12
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def make_full_frame_warp(image, H):
    h, w = image.shape[:2]

    # Dense samples along all four borders are safer than corners only.
    n = 300
    xs = np.linspace(0, w - 1, n)
    ys = np.linspace(0, h - 1, n)

    border_pts = np.vstack([
        np.column_stack([xs, np.zeros_like(xs)]),
        np.column_stack([xs, np.full_like(xs, h - 1)]),
        np.column_stack([np.zeros_like(ys), ys]),
        np.column_stack([np.full_like(ys, w - 1), ys]),
    ])

    warped_border = transform_points(H, border_pts)
    warped_border = warped_border[np.all(np.isfinite(warped_border), axis=1)]

    if len(warped_border) == 0:
        raise RuntimeError("Invalid output bounds after homography.")

    min_xy = warped_border.min(axis=0)
    max_xy = warped_border.max(axis=0)
    span = max_xy - min_xy

    if span[0] <= 1 or span[1] <= 1:
        raise RuntimeError("Degenerate output bounds.")

    # Scale down only if the output would become too large.
    scale = min(1.0, MAX_OUTPUT_SIDE / max(span[0], span[1]))

    T = np.array([
        [scale, 0, -scale * min_xy[0]],
        [0, scale, -scale * min_xy[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    H_final = T @ H

    out_w = max(1, int(math.ceil(span[0] * scale)))
    out_h = max(1, int(math.ceil(span[1] * scale)))

    warped = cv2.warpPerspective(
        image,
        H_final,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return warped, H_final


# ============================================================
# 5. Debug visualization
# ============================================================
def draw_debug(image, lines, inliers, vp):
    vis = image.copy()

    for item, flag in zip(lines, inliers):
        if DRAW_ONLY_INLIERS and not flag:
            continue

        x1, y1, x2, y2 = map(int, item["segment"])
        color = (0, 0, 255) if flag else (120, 120, 120)
        thickness = 2 if flag else 1
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # VP may be far outside the image, so only draw it if visible nearby.
    h, w = image.shape[:2]
    if (-w <= vp[0] <= 2 * w) and (-h <= vp[1] <= 2 * h):
        p = (int(round(vp[0])), int(round(vp[1])))
        cv2.circle(vis, p, 10, (255, 0, 255), -1)
        cv2.putText(
            vis,
            "Vertical VP",
            (p[0] + 12, p[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


import cv2
import numpy as np


def warp_and_auto_crop(image, H, margin_ratio=0.02):
    """
    对整张图做透视变换，然后根据有效区域 mask 自动裁剪空白区域。

    参数：
        image: 原图
        H: 原图 -> 校正图 的 3x3 Homography
        margin_ratio: crop 后保留一点边缘，默认 2%

    返回：
        cropped: 自动裁剪后的结果图
        H_crop: 原图 -> 裁剪后结果图 的 Homography
        warped_full: 未裁剪的完整 warp 结果
        warped_mask: warp 后的有效区域 mask
    """

    h, w = image.shape[:2]

    # -------------------------------------------------------
    # 1. 先把原图四个角做变换，确定 warp 后整幅图的包围盒
    # -------------------------------------------------------
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    min_xy = np.floor(warped_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(warped_corners.max(axis=0)).astype(int)

    min_x, min_y = min_xy
    max_x, max_y = max_xy

    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    # -------------------------------------------------------
    # 2. 平移，使变换后坐标落入图像正坐标系
    # -------------------------------------------------------
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float64)

    H_full = T @ H

    # -------------------------------------------------------
    # 3. warp 原图
    # -------------------------------------------------------
    warped_full = cv2.warpPerspective(
        image,
        H_full,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # -------------------------------------------------------
    # 4. warp 一个全白 mask
    #    用它表示哪些区域是真实有效像素
    # -------------------------------------------------------
    src_mask = np.ones((h, w), dtype=np.uint8) * 255

    warped_mask = cv2.warpPerspective(
        src_mask,
        H_full,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # -------------------------------------------------------
    # 5. 形态学闭运算，填掉一些小裂缝/小孔洞
    # -------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 二值化
    valid = (warped_mask > 0).astype(np.uint8)

    # -------------------------------------------------------
    # 6. 找最大连通区域
    #    避免偶然小碎片影响 crop
    # -------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)

    if num_labels <= 1:
        # 理论上不该发生
        return warped_full, H_full, warped_full, warped_mask

    # stats[0] 是背景，从 1 开始才是前景
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + np.argmax(areas)

    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w_box = stats[best_idx, cv2.CC_STAT_WIDTH]
    h_box = stats[best_idx, cv2.CC_STAT_HEIGHT]

    # -------------------------------------------------------
    # 7. 加一点 margin，避免 crop 太紧
    # -------------------------------------------------------
    margin = int(max(w_box, h_box) * margin_ratio)

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(out_w, x + w_box + margin)
    y1 = min(out_h, y + h_box + margin)

    cropped = warped_full[y0:y1, x0:x1]

    # -------------------------------------------------------
    # 8. 更新 Homography
    #    原图 -> full_warp 是 H_full
    #    full_warp -> cropped 是一个平移
    # -------------------------------------------------------
    C = np.array([
        [1, 0, -x0],
        [0, 1, -y0],
        [0, 0, 1]
    ], dtype=np.float64)

    H_crop = C @ H_full

    return cropped, H_crop, warped_full, warped_mask

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("============================================")
    print("Whole-building upright perspective correction")
    print("Preserve original side-view perspective")
    print("============================================")

    image = imread_unicode(IMAGE_PATH)
    print("Image shape:", image.shape)

    # Step 1: detect vertical structural lines over the whole image.
    vertical_lines = detect_vertical_lines(image)
    print("Vertical candidates:", len(vertical_lines))

    # Step 2: estimate the common vertical vanishing point.
    vp, inliers, residuals = estimate_vertical_vp(vertical_lines, image.shape)
    print("Vertical VP:", vp)
    print("Vertical VP inliers:", int(inliers.sum()), "/", len(inliers))
    print("Median inlier angular residual (deg):",
          float(np.median(residuals[inliers])) if inliers.any() else None)

    # Step 3: compute pure camera-rotation homography.
    H_upright, K, R = build_upright_homography(
        image.shape,
        vp,
        HFOV_DEG,
    )

    # Step 4: warp the entire input frame and retain all source pixels.
    upright, H_final = make_full_frame_warp(image, H_upright)

    # Step 5: save output and matrices.
    debug = draw_debug(image, vertical_lines, inliers, vp)

    imwrite_unicode(OUTPUT_PATH, upright)
    imwrite_unicode(DEBUG_PATH, debug)

    np.save("H_upright.npy", H_final)
    np.save("H_upright_inv.npy", np.linalg.inv(H_final))
    np.save("vertical_vp.npy", vp)
    np.save("camera_K_approx.npy", K)
    np.save("camera_rotation_correction.npy", R)

    print("\nSaved:")
    print(" ", OUTPUT_PATH)
    print(" ", DEBUG_PATH)
    print("  H_upright.npy")
    print("  H_upright_inv.npy")
    print("  vertical_vp.npy")
    print("  camera_K_approx.npy")
    print("  camera_rotation_correction.npy")

    print("\nMeaning:")
    print("  p_upright ~ H_upright * p_original")
    print("  p_original ~ H_upright_inv * p_upright")

    cv2.namedWindow("Whole building upright", cv2.WINDOW_NORMAL)
    cv2.imshow("Whole building upright", upright)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    upright_crop, H_upright_crop, upright_full, upright_mask = warp_and_auto_crop(
        image,
        H_upright,
        margin_ratio=0.03
    )

    cv2.imwrite("whole_building_upright_full.jpg", upright_full)
    cv2.imwrite("whole_building_upright_crop.jpg", upright_crop)
    cv2.imwrite("whole_building_upright_mask.png", upright_mask)

    np.save("H_upright_crop.npy", H_upright_crop)
    np.save("H_upright_crop_inv.npy", np.linalg.inv(H_upright_crop))

    print("Saved:")
    print("  whole_building_upright_full.jpg")
    print("  whole_building_upright_crop.jpg")
    print("  whole_building_upright_mask.png")
    print("  H_upright_crop.npy")
    print("  H_upright_crop_inv.npy")
