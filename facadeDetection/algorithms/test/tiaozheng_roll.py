import cv2
import numpy as np
import math


def estimate_vertical_vp(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------
    # 1. edge
    # -------------------------
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # -------------------------
    # 2. LSD
    # -------------------------
    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    detected = lsd.detect(edges)[0]

    if detected is None:
        return None, []

    H, W = gray.shape

    lines = []

    for seg in detected[:, 0]:

        x1, y1, x2, y2 = seg

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < 0.08 * H:
            continue

        angle = (
            math.degrees(
                math.atan2(dy, dx)
            )
            % 180
        )

        # 先宽松筛选接近竖直的线
        if not (55 < angle < 125):
            continue

        p1 = np.array(
            [x1, y1, 1.0]
        )

        p2 = np.array(
            [x2, y2, 1.0]
        )

        line = np.cross(p1, p2)

        n = np.linalg.norm(
            line[:2]
        )

        if n < 1e-8:
            continue

        line /= n

        lines.append({
            "line": line,
            "length": length,
            "segment": seg
        })

    if len(lines) < 2:
        return None, lines

    # -------------------------
    # 3. RANSAC VP
    # -------------------------

    L = np.array([
        x["line"]
        for x in lines
    ])

    lengths = np.array([
        x["length"]
        for x in lines
    ])

    rng = np.random.default_rng(0)

    best_score = -1
    best_inliers = None

    for _ in range(3000):

        i, j = rng.choice(
            len(lines),
            2,
            replace=False
        )

        p = np.cross(
            L[i],
            L[j]
        )

        if abs(p[2]) < 1e-10:
            continue

        p /= p[2]

        distances = np.abs(
            L @ p
        )

        inliers = distances < 4.0

        score = lengths[
            inliers
        ].sum()

        if score > best_score:

            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        return None, lines

    # -------------------------
    # 4. 最小二乘 refinement
    # -------------------------

    Lin = L[best_inliers]

    Wgt = np.sqrt(
        lengths[best_inliers]
    )[:, None]

    _, _, Vt = np.linalg.svd(
        Lin * Wgt
    )

    vp = Vt[-1]

    if abs(vp[2]) < 1e-10:
        return None, lines

    vp /= vp[2]

    return vp[:2], lines


def estimate_roll_correction(image, principal_x=None, principal_y=None):

    vp, lines = estimate_vertical_vp(
        image
    )

    if vp is None:
        raise RuntimeError(
            "Cannot estimate vertical VP"
        )

    H, W = image.shape[:2]

    cx = W / 2.0 if principal_x is None else float(principal_x)
    cy = H / 2.0 if principal_y is None else float(principal_y)

    vx, vy = vp

    dx = vx - cx
    dy = vy - cy

    roll_error = np.degrees(
        np.arctan2(
            dx,
            -dy
        )
    )

    print(
        "Vertical VP:",
        vp
    )

    print(
        "Image center:",
        (cx, cy)
    )

    print(
        "Estimated roll correction:",
        roll_error,
        "deg"
    )

    return roll_error