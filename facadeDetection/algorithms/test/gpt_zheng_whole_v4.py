import cv2
import numpy as np
import math
import os

# ============================================================
# whole_building_upright_v4.py
# ------------------------------------------------------------
# 目标：
#   1) 将整栋建筑“站直”：世界竖直线在输出图中尽量平行且竖直；
#   2) 保留原始左右立面的观察关系，不把某个立面强行拉成正视图；
#   3) 不依赖猜测 HFOV / 焦距：默认使用最小畸变的 vertical-only projective rectification；
#   4) 自动聚类选择主建筑的竖向结构线，减少左侧其他楼/树枝干扰；
#   5) 自动 crop：围绕主建筑结构范围裁剪，并去掉绝大多数 warp 黑色空白；
#   6) 最终 crop 自动恢复到与原图基本一致的宽高比，不做非等比拉伸；
#   7) 保存完整 Homography，后续可将 2D 校正图坐标映回原始图。
#
# 依赖：
#   pip install opencv-python numpy
# ============================================================


# ============================================================
# 用户参数
# ============================================================
IMAGE_PATH = r"../data/northeast.jpg"          # <-- 修改为你的图片路径

OUTPUT_FULL = r"../data/southeast_upright_v4_full.jpg"
OUTPUT_CROP = r"../data/southeast_upright_v4_crop.jpg"
OUTPUT_MASK = r"../data/southeast_upright_v4_mask.png"
DEBUG_PATH = r"../data/southeast_upright_v4_debug.jpg"
SHOW_WINDOW = True                 # 本地运行时显示结果窗口

# ------------------------------------------------------------
# LSD 直线检测
# ------------------------------------------------------------
MIN_LINE_LENGTH_RATIO = 0.025     # 相对图像短边
MAX_VERTICAL_ANGLE_DEVIATION_DEG = 45.0

# 只在主体区域内找结构线，减少边缘干扰。
# 对你的建筑图，中央主体占比很大，因此范围设得较宽。
USE_MAIN_ROI = True
ROI_X_MIN = 0.10
ROI_X_MAX = 0.95
ROI_Y_MIN = 0.00
ROI_Y_MAX = 1.00

# ------------------------------------------------------------
# Vanishing point RANSAC
# ------------------------------------------------------------
VP_RANSAC_ITERATIONS = 5000
VP_RANSAC_ANGLE_THRESHOLD_DEG = 2.2

# ------------------------------------------------------------
# 主建筑竖线聚类
# ------------------------------------------------------------
# 按竖线中点 x 聚类；相邻 cluster 的 x 间隔大于这个比例时分开。
BUILDING_CLUSTER_GAP_RATIO = 0.09

# ------------------------------------------------------------
# 校正强度
# ------------------------------------------------------------
# 1.0 = 将 vertical VP 完全送到无穷远，即严格竖直校正。
# 0.9~0.98 = 稍弱，视觉更柔和，但竖线不会完全平行。
PERSPECTIVE_STRENGTH = 1.0

# 校正时选一个 y anchor，使该高度附近的局部尺度尽量保持不变。
# 默认由主建筑竖线自动估计，然后限制在下面范围，避免极端值。
ANCHOR_Y_MIN_RATIO = 0.45
ANCHOR_Y_MAX_RATIO = 0.72

# ------------------------------------------------------------
# 输出尺寸
# ------------------------------------------------------------
MAX_OUTPUT_SIDE = 5000

# ------------------------------------------------------------
# 建筑感知 crop
# ------------------------------------------------------------
AUTO_CROP = True

# ------------------------------------------------------------
# 最终输出宽高比
# ------------------------------------------------------------
# True: 最终 crop 的 width / height 尽量与原图完全一致。
# 注意：这里只调整 crop 窗口，不对图像做横向/纵向非等比 resize，
# 因此不会破坏后续 2D-3D 几何关系。
KEEP_ORIGINAL_ASPECT_RATIO = True

# 搜索满足目标宽高比的 crop 时，允许在候选位置上做多少级搜索。
ASPECT_SEARCH_STEPS = 61

# 位置评分时对“偏离原 crop 中心”的惩罚，越大越倾向保持原构图中心。
ASPECT_CENTER_PENALTY = 0.03

# 主建筑结构点的 robust quantile，去掉少量异常线端点。
STRUCTURE_LOW_Q = 0.01
STRUCTURE_HIGH_Q = 0.99

# 围绕主建筑结构范围增加 margin。
CROP_MARGIN_X_RATIO = 0.08
CROP_MARGIN_Y_RATIO = 0.035

# 利用有效 mask 的逐行左右边界进一步收缩横向 crop。
# 0.94 / 0.06 对当前这类梯形有效区域通常能去掉 >99% 黑区，
# 又不容易切到主体建筑。
MASK_LEFT_QUANTILE = 0.94
MASK_RIGHT_QUANTILE = 0.06

# crop 边缘向内缩一点，避免插值产生 1~2 像素黑边。
CROP_INSET_PX = 2

# Debug 是否只画 RANSAC inlier
DRAW_ONLY_INLIERS = True


# ============================================================
# Unicode-safe I/O（Windows 中文路径）
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
# 基础几何
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
    p = p / p[2]
    if not np.all(np.isfinite(p)):
        return None
    return p


def transform_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    q = (H @ hom.T).T
    valid = np.abs(q[:, 2]) > 1e-12
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]
    c = np.cumsum(weights)
    if c[-1] <= 0:
        return float(np.median(values))
    pos = np.searchsorted(c, 0.5 * c[-1])
    pos = min(max(int(pos), 0), len(values) - 1)
    return float(values[pos])


# ============================================================
# 1. LSD 检测竖向结构线
# ============================================================
def detect_vertical_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)[0]
    if detected is None:
        raise RuntimeError("No line segments detected.")

    h, w = image.shape[:2]
    min_length = min(h, w) * MIN_LINE_LENGTH_RATIO

    rx0 = ROI_X_MIN * w
    rx1 = ROI_X_MAX * w
    ry0 = ROI_Y_MIN * h
    ry1 = ROI_Y_MAX * h

    vertical = []

    for seg in detected[:, 0, :]:
        x1, y1, x2, y2 = map(float, seg)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        if USE_MAIN_ROI and not (rx0 <= mx <= rx1 and ry0 <= my <= ry1):
            continue

        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        dev_v = abs(angle - 90.0)
        if dev_v > MAX_VERTICAL_ANGLE_DEVIATION_DEG:
            continue

        vertical.append({
            "segment": np.array([x1, y1, x2, y2], dtype=np.float64),
            "line": segment_to_line(x1, y1, x2, y2),
            "length": length,
            "angle": angle,
            "mid_x": mx,
            "mid_y": my,
        })

    if len(vertical) < 2:
        raise RuntimeError(
            "Too few vertical line candidates. Try decreasing "
            "MIN_LINE_LENGTH_RATIO or increasing MAX_VERTICAL_ANGLE_DEVIATION_DEG."
        )

    return vertical


# ============================================================
# 2. Weighted angular RANSAC 求竖直消失点
# ============================================================
def estimate_vertical_vp(lines, image_shape, seed=0):
    h, w = image_shape[:2]
    n = len(lines)
    rng = np.random.default_rng(seed)

    lengths = np.array([x["length"] for x in lines], dtype=np.float64)
    probs = lengths / lengths.sum()
    line_matrix = np.array([x["line"] for x in lines], dtype=np.float64)

    segs = np.array([x["segment"] for x in lines], dtype=np.float64)
    mids = np.column_stack([
        0.5 * (segs[:, 0] + segs[:, 2]),
        0.5 * (segs[:, 1] + segs[:, 3]),
    ])
    dirs = np.column_stack([
        segs[:, 2] - segs[:, 0],
        segs[:, 3] - segs[:, 1],
    ])
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    cos_thr = math.cos(math.radians(VP_RANSAC_ANGLE_THRESHOLD_DEG))

    best_score = -1.0
    best_inliers = None
    best_vp = None

    for _ in range(VP_RANSAC_ITERATIONS):
        i, j = rng.choice(n, 2, replace=False, p=probs)
        vp = line_intersection(line_matrix[i], line_matrix[j])
        if vp is None:
            continue

        # 拒绝极端 near-infinity 数值异常模型。
        if abs(vp[0]) > 300 * w or abs(vp[1]) > 300 * h:
            continue

        dv = vp[:2][None, :] - mids
        dn = np.linalg.norm(dv, axis=1)
        good = dn > 1e-12

        cosang = np.zeros(n, dtype=np.float64)
        cosang[good] = np.abs(np.sum(dirs[good] * dv[good], axis=1) / dn[good])
        inliers = cosang >= cos_thr

        score = lengths[inliers].sum()
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_vp = vp

    if best_inliers is None or best_inliers.sum() < 2:
        raise RuntimeError("Failed to estimate a stable vertical vanishing point.")

    # 加权 SVD 精化。
    L = line_matrix[best_inliers]
    W = np.sqrt(lengths[best_inliers])[:, None]
    _, _, Vt = np.linalg.svd(L * W, full_matrices=False)
    vp = Vt[-1]
    if abs(vp[2]) < 1e-12:
        vp = best_vp
    else:
        vp = vp / vp[2]

    # 最终 residual / inlier。
    dv = vp[:2][None, :] - mids
    dn = np.linalg.norm(dv, axis=1)
    good = dn > 1e-12
    cosang = np.zeros(n, dtype=np.float64)
    cosang[good] = np.abs(np.sum(dirs[good] * dv[good], axis=1) / dn[good])
    cosang = np.clip(cosang, 0.0, 1.0)
    residuals = np.degrees(np.arccos(cosang))
    inliers = residuals <= VP_RANSAC_ANGLE_THRESHOLD_DEG

    return vp, inliers, residuals


# ============================================================
# 3. 自动选择“主建筑”的竖线 cluster
# ============================================================
def select_main_building_cluster(lines, inliers, image_shape):
    h, w = image_shape[:2]
    candidates = [x for x, flag in zip(lines, inliers) if flag]
    if len(candidates) < 2:
        raise RuntimeError("Too few vertical inliers for main-building selection.")

    # 按 x 排序，用较大间隔切分 cluster。
    candidates = sorted(candidates, key=lambda x: x["mid_x"])
    gap_thr = BUILDING_CLUSTER_GAP_RATIO * w

    clusters = []
    cur = [candidates[0]]
    for item in candidates[1:]:
        if item["mid_x"] - cur[-1]["mid_x"] > gap_thr:
            clusters.append(cur)
            cur = [item]
        else:
            cur.append(item)
    clusters.append(cur)

    cx = 0.5 * (w - 1)
    best = None
    best_score = -1.0

    for cluster in clusters:
        total_len = sum(x["length"] for x in cluster)
        xs = np.array([x["mid_x"] for x in cluster], dtype=np.float64)
        cmean = float(np.average(xs, weights=[x["length"] for x in cluster]))

        # 中央偏好只是轻微 tie-break，主体仍主要由总线长决定。
        center_bonus = 0.75 + 0.25 * math.exp(-((cmean - cx) / (0.35 * w)) ** 2)
        score = total_len * center_bonus

        if score > best_score:
            best_score = score
            best = cluster

    return best, clusters


# ============================================================
# 4. Minimal-distortion vertical-only rectification
# ============================================================
def build_minimal_vertical_homography(image_shape, vp_vertical, main_lines):
    """
    核心：
      A. 先做一个很小的 2D roll rotation，使 vertical VP 位于主点正上/正下方；
      B. 再做一维 projective correction，仅把 vertical VP 沿竖直方向送到无穷远；
      C. 用 anchor_y 归一化局部尺度，避免建筑被无谓压得过窄。

    与 H = K R K^-1 的区别：本方法不需要猜焦距，
    对未知手机内参/广角照片通常视觉上更稳定。
    """
    h, w = image_shape[:2]
    cx = 0.5 * (w - 1)
    cy = 0.5 * (h - 1)

    # ---------- A. roll correction ----------
    dx = float(vp_vertical[0] - cx)
    dy = float(vp_vertical[1] - cy)
    if abs(dx) + abs(dy) < 1e-12:
        raise RuntimeError("Vertical VP is too close to image center; unstable geometry.")

    current_angle = math.atan2(dy, dx)
    target_angle = -math.pi / 2 if dy < 0 else math.pi / 2
    theta = target_angle - current_angle

    c = math.cos(theta)
    s = math.sin(theta)

    T0 = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    T1 = np.array([
        [1.0, 0.0, cx],
        [0.0, 1.0, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R2 = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_roll = T1 @ R2 @ T0

    vp_rot = H_roll @ np.asarray(vp_vertical, dtype=np.float64)
    vp_rot = vp_rot / vp_rot[2]
    vp_y_centered = float(vp_rot[1] - cy)

    if abs(vp_y_centered) < 0.15 * h:
        raise RuntimeError(
            "Vertical VP is too close to the image center after roll correction. "
            "The photo may not have enough stable vertical structure."
        )

    # ---------- B. 自动 anchor y ----------
    mids = []
    weights = []
    for item in main_lines:
        x1, y1, x2, y2 = item["segment"]
        mids.append([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        weights.append(item["length"])

    mids_rot = transform_points(H_roll, mids)
    anchor_y = weighted_median(mids_rot[:, 1], weights)
    anchor_y = np.clip(
        anchor_y,
        ANCHOR_Y_MIN_RATIO * (h - 1),
        ANCHOR_Y_MAX_RATIO * (h - 1),
    )

    anchor_y_centered = float(anchor_y - cy)

    # ---------- C. 一维 vertical projective correction ----------
    strength = float(np.clip(PERSPECTIVE_STRENGTH, 0.0, 1.0))
    p = -strength / vp_y_centered

    # 令 anchor_y 处的局部 isotropic scale ~ 1。
    anchor_den = 1.0 + p * anchor_y_centered
    scale_anchor = anchor_den

    H_proj_centered = np.array([
        [scale_anchor, 0.0, 0.0],
        [0.0, scale_anchor, 0.0],
        [0.0, p, 1.0],
    ], dtype=np.float64)

    H_base = T1 @ H_proj_centered @ T0 @ H_roll

    if abs(H_base[2, 2]) > 1e-12:
        H_base = H_base / H_base[2, 2]

    info = {
        "roll_deg": math.degrees(theta),
        "anchor_y": float(anchor_y),
        "vp_after_roll": vp_rot,
        "vp_y_centered": vp_y_centered,
    }
    return H_base, info


# ============================================================
# 5. Warp 完整画面 + valid mask
# ============================================================
def make_full_warp(image, H_base):
    h, w = image.shape[:2]

    # 用密集边界采样，而非仅四角，避免 projective near-infinity 时边界估错。
    n = 500
    xs = np.linspace(0, w - 1, n)
    ys = np.linspace(0, h - 1, n)
    border = np.vstack([
        np.column_stack([xs, np.zeros_like(xs)]),
        np.column_stack([xs, np.full_like(xs, h - 1)]),
        np.column_stack([np.zeros_like(ys), ys]),
        np.column_stack([np.full_like(ys, w - 1), ys]),
    ])

    warped_border = transform_points(H_base, border)
    warped_border = warped_border[np.all(np.isfinite(warped_border), axis=1)]
    if len(warped_border) == 0:
        raise RuntimeError("Invalid transformed image boundary.")

    # 去掉极端 near-infinity 点。
    min_xy = np.quantile(warped_border, 0.001, axis=0)
    max_xy = np.quantile(warped_border, 0.999, axis=0)
    span = max_xy - min_xy

    if np.any(span <= 1):
        raise RuntimeError("Degenerate output bounds.")

    scale = min(1.0, MAX_OUTPUT_SIDE / float(max(span)))

    T = np.array([
        [scale, 0.0, -scale * min_xy[0]],
        [0.0, scale, -scale * min_xy[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_full = T @ H_base
    out_w = max(1, int(math.ceil(span[0] * scale)))
    out_h = max(1, int(math.ceil(span[1] * scale)))

    warped = cv2.warpPerspective(
        image, H_full, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    src_mask = np.full((h, w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        src_mask, H_full, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped, mask, H_full


# ============================================================
# 6. Building-aware crop
# ============================================================
def get_main_structure_points(main_lines):
    pts = []
    for item in main_lines:
        x1, y1, x2, y2 = item["segment"]
        pts.append([x1, y1])
        pts.append([x2, y2])
    return np.asarray(pts, dtype=np.float64)


def building_aware_crop(warped, mask, H_full, main_lines):
    """
    与“按整幅 valid mask bounding box crop”不同：
      - 先用主建筑结构线确定主体 y 范围；
      - 再在这个 y 范围内，根据 mask 每一行的有效左右边界做 robust crop；
      - 因而不会被顶部/底部极端尖角拖累，也不会保留大块黑三角。
    """
    Hh, Ww = mask.shape[:2]

    pts_src = get_main_structure_points(main_lines)
    pts_dst = transform_points(H_full, pts_src)
    pts_dst = pts_dst[np.all(np.isfinite(pts_dst), axis=1)]

    if len(pts_dst) < 4:
        return warped, mask, H_full, (0, 0, Ww, Hh)

    qlo = np.quantile(pts_dst, STRUCTURE_LOW_Q, axis=0)
    qhi = np.quantile(pts_dst, STRUCTURE_HIGH_Q, axis=0)
    span = np.maximum(qhi - qlo, 1.0)

    # 主建筑纵向范围 + 少量 margin。
    y0 = int(math.floor(qlo[1] - CROP_MARGIN_Y_RATIO * span[1]))
    y1 = int(math.ceil(qhi[1] + CROP_MARGIN_Y_RATIO * span[1]))
    y0 = max(0, min(y0, Hh - 1))
    y1 = max(y0 + 1, min(y1, Hh))

    valid = mask > 0
    lefts, rights = [], []
    for y in range(y0, y1):
        xs = np.where(valid[y])[0]
        if len(xs) == 0:
            continue
        lefts.append(xs[0])
        rights.append(xs[-1])

    if len(lefts) < 4:
        return warped, mask, H_full, (0, 0, Ww, Hh)

    # mask 给出的“基本无黑边”横向范围。
    x0_mask = int(math.ceil(np.quantile(lefts, MASK_LEFT_QUANTILE)))
    x1_mask = int(math.floor(np.quantile(rights, MASK_RIGHT_QUANTILE))) + 1

    # 结构范围给出的“主体建筑 + margin”。
    x0_struct = int(math.floor(qlo[0] - CROP_MARGIN_X_RATIO * span[0]))
    x1_struct = int(math.ceil(qhi[0] + CROP_MARGIN_X_RATIO * span[0]))

    # 优先保留主体：只要 mask crop 没侵入主体核心，就采用它；
    # 若侵入，则退回结构范围，允许极少量黑角而不切建筑。
    core_x0 = int(math.floor(np.quantile(pts_dst[:, 0], 0.05)))
    core_x1 = int(math.ceil(np.quantile(pts_dst[:, 0], 0.95)))

    x0 = max(0, x0_struct)
    x1 = min(Ww, x1_struct)

    if x0_mask <= core_x0:
        x0 = max(x0, x0_mask)
    if x1_mask >= core_x1:
        x1 = min(x1, x1_mask)

    # 若上面的交集过窄，直接用 mask robust 范围。
    if x1 <= x0 + 20:
        x0, x1 = x0_mask, x1_mask

    x0 += CROP_INSET_PX
    x1 -= CROP_INSET_PX
    y0 += CROP_INSET_PX
    y1 -= CROP_INSET_PX

    x0 = max(0, min(x0, Ww - 1))
    x1 = max(x0 + 1, min(x1, Ww))
    y0 = max(0, min(y0, Hh - 1))
    y1 = max(y0 + 1, min(y1, Hh))

    cropped = warped[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1].copy()

    C = np.array([
        [1.0, 0.0, -x0],
        [0.0, 1.0, -y0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    H_crop = C @ H_full

    return cropped, cropped_mask, H_crop, (x0, y0, x1, y1)


# ============================================================
# 7. 保持原图宽高比的最终 crop
# ============================================================
def _rect_sum(integral, x0, y0, x1, y1):
    """积分图矩形求和，坐标范围 [x0,x1) x [y0,y1)。"""
    return (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def adjust_crop_box_to_aspect(mask, base_box, target_ratio):
    """
    在不裁掉 base_box 的前提下，将最终 crop 调整为目标 width/height。

    原则：
      1) 只扩大 crop，不对图像做非等比 resize；
      2) 尽量保持 base_box（主建筑主体）完整；
      3) 在可选位置中优先选择有效 mask 比例最高的位置；
      4) 轻微偏好与原 crop 中心接近的位置。

    返回：x0, y0, x1, y1
    """
    Hh, Ww = mask.shape[:2]
    bx0, by0, bx1, by1 = map(int, base_box)

    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)

    if target_ratio <= 0:
        return bx0, by0, bx1, by1

    # 最小扩张：先保持宽度，再补高度；若高度仍不足则反过来补宽度。
    tw = bw
    th = int(math.ceil(tw / target_ratio))

    if th < bh:
        th = bh
        tw = int(math.ceil(th * target_ratio))

    # 若目标框超过 full warp 尺寸，尽量在可用画布内保持目标比例。
    if tw > Ww or th > Hh:
        scale = min(Ww / max(tw, 1), Hh / max(th, 1))
        tw2 = max(bw, int(math.floor(tw * scale)))
        th2 = max(bh, int(round(tw2 / target_ratio)))

        if th2 > Hh:
            th2 = Hh
            tw2 = int(round(th2 * target_ratio))
        if tw2 > Ww:
            tw2 = Ww
            th2 = int(round(tw2 / target_ratio))

        # 如果画布确实无法在“不裁 base_box”的同时满足目标比例，
        # 则退回 base_box，避免切掉主体建筑。
        if tw2 < bw or th2 < bh or tw2 > Ww or th2 > Hh:
            print("Warning: cannot satisfy original aspect ratio without cutting the base crop.")
            return bx0, by0, bx1, by1

        tw, th = tw2, th2

    # 可选左上角范围：必须完整包含 base_box。
    x0_min = max(0, bx1 - tw)
    x0_max = min(bx0, Ww - tw)
    y0_min = max(0, by1 - th)
    y0_max = min(by0, Hh - th)

    if x0_min > x0_max or y0_min > y0_max:
        return bx0, by0, bx1, by1

    valid = (mask > 0).astype(np.uint8)
    integral = cv2.integral(valid, sdepth=cv2.CV_64F)

    base_cx = 0.5 * (bx0 + bx1)
    base_cy = 0.5 * (by0 + by1)

    nx = max(1, min(ASPECT_SEARCH_STEPS, x0_max - x0_min + 1))
    ny = max(1, min(ASPECT_SEARCH_STEPS, y0_max - y0_min + 1))

    xs = np.unique(np.round(np.linspace(x0_min, x0_max, nx)).astype(int))
    ys = np.unique(np.round(np.linspace(y0_min, y0_max, ny)).astype(int))

    best = None
    best_score = -1e30
    area = float(tw * th)

    for y0 in ys:
        y1 = y0 + th
        for x0 in xs:
            x1 = x0 + tw

            valid_ratio = float(_rect_sum(integral, x0, y0, x1, y1)) / area

            cx = x0 + 0.5 * tw
            cy = y0 + 0.5 * th
            center_dist = math.hypot(
                (cx - base_cx) / max(Ww, 1),
                (cy - base_cy) / max(Hh, 1),
            )

            score = valid_ratio - ASPECT_CENTER_PENALTY * center_dist

            if score > best_score:
                best_score = score
                best = (int(x0), int(y0), int(x1), int(y1))

    return best if best is not None else (bx0, by0, bx1, by1)


def crop_with_box(warped, mask, H_full, crop_box):
    x0, y0, x1, y1 = map(int, crop_box)

    cropped = warped[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1].copy()

    C = np.array([
        [1.0, 0.0, -x0],
        [0.0, 1.0, -y0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_crop = C @ H_full
    return cropped, cropped_mask, H_crop


# ============================================================
# 8. Debug
# ============================================================
def draw_debug(image, lines, inliers, main_lines, vp, info):
    vis = image.copy()
    h, w = image.shape[:2]

    main_ids = {id(x) for x in main_lines}

    for item, flag in zip(lines, inliers):
        if DRAW_ONLY_INLIERS and not flag:
            continue

        x1, y1, x2, y2 = map(int, item["segment"])

        if id(item) in main_ids:
            # 主建筑竖线：绿色
            color = (0, 255, 0)
            thick = 3
        elif flag:
            # 其他 vertical VP inlier：红色
            color = (0, 0, 255)
            thick = 2
        else:
            color = (120, 120, 120)
            thick = 1

        cv2.line(vis, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

    # VP 若不是特别远则画出。
    if (-2 * w <= vp[0] <= 3 * w) and (-2 * h <= vp[1] <= 3 * h):
        p = (int(round(vp[0])), int(round(vp[1])))
        cv2.circle(vis, p, 10, (255, 0, 255), -1)
        cv2.putText(vis, "Vertical VP", (p[0] + 12, p[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    if USE_MAIN_ROI:
        p0 = (int(ROI_X_MIN * w), int(ROI_Y_MIN * h))
        p1 = (int(ROI_X_MAX * w), int(ROI_Y_MAX * h))
        cv2.rectangle(vis, p0, p1, (255, 255, 0), 2)

    text1 = f"roll={info['roll_deg']:.2f} deg"
    text2 = f"anchor_y={info['anchor_y']:.1f}, strength={PERSPECTIVE_STRENGTH:.2f}"
    cv2.rectangle(vis, (10, 10), (620, 82), (0, 0, 0), -1)
    cv2.putText(vis, text1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text2, (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("====================================================")
    print("Whole-building upright perspective correction V4")
    print("- vertical-only minimal projective rectification")
    print("- no guessed HFOV required")
    print("- main-building line clustering")
    print("- building-aware auto crop")
    print("- preserve original image aspect ratio")
    print("====================================================")

    image = imread_unicode(IMAGE_PATH)
    print("Image shape:", image.shape)

    # Step 1: vertical structural lines
    vertical_lines = detect_vertical_lines(image)
    print("Vertical candidates:", len(vertical_lines))

    # Step 2: vertical vanishing point
    vp_v, inliers_v, residuals_v = estimate_vertical_vp(
        vertical_lines, image.shape, seed=0
    )
    print("Vertical VP:", vp_v)
    print("Vertical inliers:", int(inliers_v.sum()), "/", len(inliers_v))
    if np.any(inliers_v):
        print("Median vertical residual (deg):",
              float(np.median(residuals_v[inliers_v])))

    # Step 3: main building cluster
    main_lines, clusters = select_main_building_cluster(
        vertical_lines, inliers_v, image.shape
    )
    print("Vertical clusters:", [len(c) for c in clusters])
    print("Selected main-building lines:", len(main_lines))

    # Step 4: minimal vertical rectification
    H_base, rectify_info = build_minimal_vertical_homography(
        image.shape, vp_v, main_lines
    )
    print("Roll correction (deg):", rectify_info["roll_deg"])
    print("Anchor y:", rectify_info["anchor_y"])

    # Step 5: full warp + mask
    upright_full, valid_mask, H_full = make_full_warp(image, H_base)
    print("Full warp shape:", upright_full.shape)

    # Step 6: building-aware base crop
    if AUTO_CROP:
        _, _, _, base_crop_box = building_aware_crop(
            upright_full, valid_mask, H_full, main_lines
        )
    else:
        base_crop_box = (0, 0, upright_full.shape[1], upright_full.shape[0])

    # Step 7: 将最终 crop 的宽高比恢复到与原图一致。
    original_ratio = image.shape[1] / float(image.shape[0])

    if KEEP_ORIGINAL_ASPECT_RATIO:
        crop_box = adjust_crop_box_to_aspect(
            valid_mask, base_crop_box, original_ratio
        )
    else:
        crop_box = base_crop_box

    upright_crop, crop_mask, H_crop = crop_with_box(
        upright_full, valid_mask, H_full, crop_box
    )

    output_ratio = upright_crop.shape[1] / float(upright_crop.shape[0])

    print("Base crop box [x0,y0,x1,y1]:", base_crop_box)
    print("Final crop box [x0,y0,x1,y1]:", crop_box)
    print("Original aspect ratio (W/H): {:.6f}".format(original_ratio))
    print("Output aspect ratio   (W/H): {:.6f}".format(output_ratio))

    invalid_ratio = 1.0 - float(np.mean(crop_mask > 0))
    print("Invalid/black ratio inside crop: {:.3f}%".format(100.0 * invalid_ratio))

    # Step 8: debug
    debug = draw_debug(
        image, vertical_lines, inliers_v, main_lines, vp_v, rectify_info
    )

    # Step 9: save images
    imwrite_unicode(OUTPUT_FULL, upright_full)
    imwrite_unicode(OUTPUT_CROP, upright_crop)
    imwrite_unicode(OUTPUT_MASK, valid_mask)
    imwrite_unicode(DEBUG_PATH, debug)

    # Step 10: save matrices
    np.save("H_upright_v4_base.npy", H_base)
    np.save("H_upright_v4_full.npy", H_full)
    np.save("H_upright_v4_full_inv.npy", np.linalg.inv(H_full))
    np.save("H_upright_v4_crop.npy", H_crop)
    np.save("H_upright_v4_crop_inv.npy", np.linalg.inv(H_crop))
    np.save("vertical_vp_v4.npy", vp_v)

    print("\nSaved images:")
    print(" ", OUTPUT_FULL)
    print(" ", OUTPUT_CROP, " <-- recommended")
    print(" ", OUTPUT_MASK)
    print(" ", DEBUG_PATH)

    print("\nSaved matrices:")
    print("  H_upright_v4_crop.npy      <-- original -> cropped upright")
    print("  H_upright_v4_crop_inv.npy  <-- cropped upright -> original")
    print("  H_upright_v4_full.npy")
    print("  vertical_vp_v4.npy")

    print("\nCoordinate relation:")
    print("  p_crop ~ H_upright_v4_crop @ p_original")
    print("  p_original ~ H_upright_v4_crop_inv @ p_crop")

    if SHOW_WINDOW and os.environ.get("HEADLESS", "0") != "1":
        cv2.namedWindow("Whole building upright V4 - crop", cv2.WINDOW_NORMAL)
        cv2.imshow("Whole building upright V4 - crop", upright_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
