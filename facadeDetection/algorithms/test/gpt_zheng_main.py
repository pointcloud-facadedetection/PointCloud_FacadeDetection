import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN


# ============================================================
# 参数
# ============================================================

IMAGE_PATH = "../data/southeast.jpg"

# "real_edge"：优先使用检测到的真实上/下边界线
# "image_border"：如果上边界不明显，可直接用图像上边界/下边界作为可见范围
TOP_BOTTOM_MODE = "real_edge"

# 如果知道建筑立面真实宽高比，例如 0.45，可填入
# 表示 width / height = 0.45
KNOWN_ASPECT_RATIO = None

MIN_LINE_LENGTH_RATIO = 0.04
VP_RANSAC_THRESHOLD = 4.0
VP_RANSAC_ITERATIONS = 5000


# ============================================================
# 中文路径读写
# ============================================================

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def imwrite_unicode(path, image):
    ext = "." + path.split(".")[-1]
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"保存失败: {path}")
    buf.tofile(path)


# ============================================================
# 基本几何函数
# 直线表示为 ax + by + c = 0
# ============================================================

def normalize_line(line):
    line = np.asarray(line, dtype=np.float64)
    norm = math.hypot(line[0], line[1])
    if norm < 1e-12:
        return line
    return line / norm


def line_from_points(p1, p2):
    p1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    p2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    return normalize_line(line)


def intersect_lines(line1, line2):
    p = np.cross(line1, line2)
    if abs(p[2]) < 1e-10:
        raise RuntimeError("两条线平行或交点在无穷远")
    return p[:2] / p[2]


def x_on_line(line, y):
    a, b, c = line
    if abs(a) < 1e-10:
        return np.inf
    return -(b * y + c) / a


def y_on_line(line, x):
    a, b, c = line
    if abs(b) < 1e-10:
        return np.inf
    return -(a * x + c) / b


# ============================================================
# LSD 检测直线
# ============================================================

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(gray)[0]

    if detected is None:
        raise RuntimeError("没有检测到直线")

    h, w = image.shape[:2]
    min_length = min(h, w) * MIN_LINE_LENGTH_RATIO

    lines = []
    for seg in detected[:, 0, :]:
        x1, y1, x2, y2 = map(float, seg)

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length:
            continue

        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        line = line_from_points((x1, y1), (x2, y2))

        lines.append({
            "segment": np.array([x1, y1, x2, y2], dtype=np.float64),
            "length": length,
            "angle": angle,
            "line": line
        })

    return lines


# ============================================================
# 方向分类
# ============================================================

def classify_lines(lines):
    vertical = []
    horizontal = []

    for item in lines:
        angle = item["angle"]

        # 竖线候选
        if 70 <= angle <= 110:
            vertical.append(item)

        # 横线候选（透视下允许有较大倾斜）
        elif angle <= 45 or angle >= 135:
            horizontal.append(item)

    return vertical, horizontal


# ============================================================
# RANSAC 求消失点
# ============================================================

def estimate_vanishing_point(lines,
                             iterations=5000,
                             threshold=4.0,
                             seed=0):
    if len(lines) < 2:
        raise RuntimeError("用于估计消失点的直线太少")

    rng = np.random.default_rng(seed)

    line_matrix = np.array([x["line"] for x in lines], dtype=np.float64)
    lengths = np.array([x["length"] for x in lines], dtype=np.float64)
    probs = lengths / lengths.sum()

    best_score = -1
    best_inliers = None

    n = len(lines)

    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False, p=probs)

        p = np.cross(line_matrix[i], line_matrix[j])
        if abs(p[2]) < 1e-10:
            continue
        p = p / p[2]

        distances = np.abs(line_matrix @ p)
        inliers = distances < threshold

        score = lengths[inliers].sum()
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        raise RuntimeError("RANSAC 估计消失点失败")

    L = line_matrix[best_inliers]
    W = np.sqrt(lengths[best_inliers])[:, None]

    _, _, Vt = np.linalg.svd(L * W, full_matrices=False)
    vp = Vt[-1]

    if abs(vp[2]) < 1e-10:
        raise RuntimeError("得到无穷远消失点")

    vp = vp / vp[2]
    return vp, best_inliers


# ============================================================
# 横向线中选择主立面
# 通过聚类去掉左侧其他楼、树枝等干扰
# ============================================================

def select_main_facade_horizontal(horizontal_lines,
                                  horizontal_inliers,
                                  image_shape):
    h, w = image_shape[:2]

    good = [line for line, flag in zip(horizontal_lines, horizontal_inliers) if flag]
    if len(good) < 4:
        raise RuntimeError("有效横向线太少")

    centers = []
    weights = []

    for item in good:
        x1, y1, x2, y2 = item["segment"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append([cx / w, cy / h])
        weights.append(item["length"])

    centers = np.array(centers, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    clustering = DBSCAN(eps=0.12, min_samples=4).fit(centers)
    labels = clustering.labels_

    valid_labels = [lab for lab in set(labels) if lab != -1]

    if len(valid_labels) == 0:
        return good

    best_label = None
    best_score = -1
    for lab in valid_labels:
        mask = labels == lab
        score = weights[mask].sum()
        if score > best_score:
            best_score = score
            best_label = lab

    selected = [line for line, lab in zip(good, labels) if lab == best_label]
    return selected


# ============================================================
# 从竖线中选择与主立面对应的线
# ============================================================

def select_vertical_facade_lines(vertical_lines,
                                 vertical_inliers,
                                 facade_horizontal_lines,
                                 image_shape):
    h, w = image_shape[:2]

    xs = []
    ys = []

    for item in facade_horizontal_lines:
        x1, y1, x2, y2 = item["segment"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_margin = max(100, 0.25 * (x_max - x_min))
    y_margin = max(100, 0.15 * (y_max - y_min))

    selected = []

    for item, flag in zip(vertical_lines, vertical_inliers):
        if not flag:
            continue

        x1, y1, x2, y2 = item["segment"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if (x_min - x_margin <= cx <= x_max + x_margin and
            y_min - y_margin <= cy <= y_max + y_margin):
            selected.append(item)

    if len(selected) < 2:
        raise RuntimeError("主立面有效竖线太少")

    return selected


# ============================================================
# 选左右边界线
# 思想：在一个参考 y 上，左右最外侧的竖线往往是边界
# ============================================================

def choose_left_right_boundary_lines(vertical_facade_lines):
    ref_y = np.median([
        (item["segment"][1] + item["segment"][3]) / 2
        for item in vertical_facade_lines
    ])

    items = []
    for item in vertical_facade_lines:
        x_ref = x_on_line(item["line"], ref_y)
        if np.isfinite(x_ref):
            items.append((item, x_ref))

    if len(items) < 2:
        raise RuntimeError("无法确定左右边界线")

    items = sorted(items, key=lambda t: t[1])

    k = max(2, len(items) // 3)

    left_candidates = items[:k]
    right_candidates = items[-k:]

    left_line_item = max(left_candidates, key=lambda t: t[0]["length"])[0]
    right_line_item = max(right_candidates, key=lambda t: t[0]["length"])[0]

    return left_line_item, right_line_item


# ============================================================
# 选上/下边界线
# 若建筑真实顶边不明显，可切换为 image_border 模式
# ============================================================

def choose_top_bottom_lines(facade_horizontal_lines, image_shape):
    h, w = image_shape[:2]

    if TOP_BOTTOM_MODE == "image_border":
        top_line = np.array([0.0, 1.0, 0.0], dtype=np.float64)         # y = 0
        bottom_line = np.array([0.0, 1.0, -(h - 1)], dtype=np.float64) # y = h-1
        return top_line, bottom_line, None, None

    ref_x = np.median([
        (item["segment"][0] + item["segment"][2]) / 2
        for item in facade_horizontal_lines
    ])

    items = []
    for item in facade_horizontal_lines:
        y_ref = y_on_line(item["line"], ref_x)
        if np.isfinite(y_ref):
            items.append((item, y_ref))

    if len(items) < 2:
        # fallback
        top_line = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        bottom_line = np.array([0.0, 1.0, -(h - 1)], dtype=np.float64)
        return top_line, bottom_line, None, None

    items = sorted(items, key=lambda t: t[1])

    k = max(2, len(items) // 3)

    top_candidates = items[:k]
    bottom_candidates = items[-k:]

    top_item = max(top_candidates, key=lambda t: t[0]["length"])[0]
    bottom_item = max(bottom_candidates, key=lambda t: t[0]["length"])[0]

    return top_item["line"], bottom_item["line"], top_item, bottom_item


# ============================================================
# 构造四个顶点
# TL, TR, BR, BL
# ============================================================

def build_facade_quad(left_item, right_item, top_line, bottom_line):
    left_line = left_item["line"]
    right_line = right_item["line"]

    tl = intersect_lines(left_line, top_line)
    tr = intersect_lines(right_line, top_line)
    br = intersect_lines(right_line, bottom_line)
    bl = intersect_lines(left_line, bottom_line)

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return quad


# ============================================================
# 由四边形生成摆正结果
# ============================================================

def rectify_from_quad(image, src_quad, known_aspect_ratio=None):
    tl, tr, br, bl = src_quad.astype(np.float64)

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    width = max(width, 1)
    height = max(height, 1)

    if known_aspect_ratio is not None:
        height = int(width / known_aspect_ratio)
        height = max(height, 1)

    dst_quad = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad)
    rectified = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_CUBIC)

    return rectified, H, dst_quad


# ============================================================
# 调试图：画消失点 inlier 线
# ============================================================

def draw_debug_lines(image,
                     vertical_lines, vertical_inliers,
                     horizontal_lines, horizontal_inliers):
    debug = image.copy()

    for item, flag in zip(vertical_lines, vertical_inliers):
        if not flag:
            continue
        x1, y1, x2, y2 = map(int, item["segment"])
        cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for item, flag in zip(horizontal_lines, horizontal_inliers):
        if not flag:
            continue
        x1, y1, x2, y2 = map(int, item["segment"])
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return debug


# ============================================================
# 原图上画出估计边界和顶点
# ============================================================

def draw_estimated_quad(image, quad):
    vis = image.copy()
    quad_int = quad.astype(int)

    # polygon
    cv2.polylines(vis, [quad_int], True, (255, 0, 0), 3)

    names = ["TL", "TR", "BR", "BL"]
    for i, p in enumerate(quad_int):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, names[i], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return vis


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=================================")
    print("Automatic facade rectification")
    print("keep boundary + keep vertices")
    print("=================================")

    image = imread_unicode(IMAGE_PATH)
    print("Image shape:", image.shape)

    # 1) 检测直线
    lines = detect_lines(image)
    print("Detected lines:", len(lines))

    # 2) 横竖分类
    vertical_lines, horizontal_lines = classify_lines(lines)
    print("Vertical candidates:", len(vertical_lines))
    print("Horizontal candidates:", len(horizontal_lines))

    # 3) 估计两个消失点
    vp_vertical, vertical_inliers = estimate_vanishing_point(
        vertical_lines,
        iterations=VP_RANSAC_ITERATIONS,
        threshold=VP_RANSAC_THRESHOLD,
        seed=0
    )

    vp_horizontal, horizontal_inliers = estimate_vanishing_point(
        horizontal_lines,
        iterations=VP_RANSAC_ITERATIONS,
        threshold=VP_RANSAC_THRESHOLD,
        seed=1
    )

    print("Vertical VP:", vp_vertical)
    print("Horizontal VP:", vp_horizontal)

    # 4) 自动选主立面
    facade_horizontal = select_main_facade_horizontal(
        horizontal_lines, horizontal_inliers, image.shape
    )

    facade_vertical = select_vertical_facade_lines(
        vertical_lines, vertical_inliers, facade_horizontal, image.shape
    )

    print("Facade horizontal lines:", len(facade_horizontal))
    print("Facade vertical lines:", len(facade_vertical))

    # 5) 选主立面边界
    left_item, right_item = choose_left_right_boundary_lines(facade_vertical)
    top_line, bottom_line, top_item, bottom_item = choose_top_bottom_lines(
        facade_horizontal, image.shape
    )

    # 6) 由四条边界线求四个顶点
    src_quad = build_facade_quad(left_item, right_item, top_line, bottom_line)

    print("\nEstimated quad (TL, TR, BR, BL):")
    print(src_quad)

    # 7) 透视摆正
    rectified, H_quad, dst_quad = rectify_from_quad(
        image,
        src_quad,
        known_aspect_ratio=KNOWN_ASPECT_RATIO
    )

    # 8) 保存结果
    debug = draw_debug_lines(
        image,
        vertical_lines, vertical_inliers,
        horizontal_lines, horizontal_inliers
    )

    quad_vis = draw_estimated_quad(image, src_quad)

    imwrite_unicode("debug_lines.jpg", debug)
    imwrite_unicode("estimated_quad.jpg", quad_vis)
    imwrite_unicode("auto_rectified_keep_vertices.jpg", rectified)

    np.save("src_quad.npy", src_quad)
    np.save("dst_quad.npy", dst_quad)
    np.save("H_quad.npy", H_quad)

    print("\nSaved:")
    print("  debug_lines.jpg")
    print("  estimated_quad.jpg")
    print("  auto_rectified_keep_vertices.jpg")
    print("  src_quad.npy")
    print("  dst_quad.npy")
    print("  H_quad.npy")

    # 显示
    cv2.namedWindow("estimated_quad", cv2.WINDOW_NORMAL)
    cv2.namedWindow("rectified", cv2.WINDOW_NORMAL)

    cv2.imshow("estimated_quad", quad_vis)
    cv2.imshow("rectified", rectified)

    cv2.waitKey(0)
    cv2.destroyAllWindows()