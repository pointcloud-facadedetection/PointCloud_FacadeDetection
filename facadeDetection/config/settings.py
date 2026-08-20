import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DEFAULT_VOXEL_SIZE = 0.05

    QUALITY_GRID_SIZE = 20.0
    RULER_SIZE = 2.0
    RULER_STEP = 0.05

    DETECT_DIST_TOL_MM = 20.0
    # Minimum effective facade area in square meters
    MIN_FACADE_AREA = 10.0

    FACADE_MAX_ITERATIONS = 40
    FACADE_NORMAL_ANGLE_DEG = 10.0
    FACADE_MIN_POINTS_RATIO = 0.003
    # 距离容差：给种子迭代和精修留足容错空间
    FACADE_SIGNED_DIST_TOLERANCE = 2.5
    # Plane merging and fallback
    FACADE_MERGE_ANGLE_DEG = 8.0
    FACADE_MERGE_D_THRESH = 0.08
    FACADE_RANSAC_MIN_INLIERS_RATIO = 0.002
    FACADE_RANSAC_MAX_PLANES = 3

    # --- New detection tuning knobs ---
    # 垂直立面阈值：|nz| <= VERTICAL_NZ_THR 视为“近似垂直面”
    VERTICAL_NZ_THR = 0.20
    # IRLS 迭代次数（用于平面精修）
    FACADE_IRLS_ITERS = 3
    # UV 连通域桥接的半径（cells）。用于跨越门窗等小缝隙，形成大连通片
    UV_CLOSE_RADIUS_CELLS = 1
    # 平面邻域容差上限（米），用于 ROI 平面优先路径的逐步放宽
    MAX_PLANE_TOL_M = 0.5

    # --- Viewport interaction & camera settings ---
    # Use orthographic projection by forcing a very small FoV in Open3D (<=5 deg)
    ORTHO_FOV_DEG = 5.0
    # 默认沿 +Y 轴观察，Z 轴向上（正面向上）；仅保留兼容配置项。
    ORTHO_DEFAULT_VIEW = 'neg_y'
    # Mouse wheel zoom scaling factor (>1). Higher means stronger zoom per notch
    ZOOM_WHEEL_SCALE = 1.6
    # Base panning speed scaling (tuned in interactor with scene/zoom factors)
    PAN_BASE_SPEED = 0.06
    # Left-drag rotate sensitivity
    ROTATE_SPEED = 1.0
    # Ortho projection screen mapping tuning (multiplier on world_per_pixel)
    ORTHO_WPP_SCALE = 1.0
    # Enable verbose debug for selection when empty results occur
    DEBUG_SELECTION = True

    # --- Selection overlay style ---
    SELECT_BORDER_RGBA = (255, 0, 0, 240)  # red opaque border
    SELECT_FILL_RGBA = (255, 0, 0, 25)     # light red fill
    SELECT_BORDER_WIDTH = 2

    # --- Selection behavior ---
    # 深度切片相关配置已废弃，框选阶段不做深度过滤；最近立面偏好在服务层通过投影深度实现。

    SEGMENT_COLORS = [
        [1.00, 0.20, 0.20],
        [1.00, 0.85, 0.10],
        [0.10, 0.45, 1.00],
        [0.20, 0.85, 0.35],
        [0.85, 0.25, 1.00],
        [1.00, 0.50, 0.10],
        [0.10, 0.90, 0.90],
        [0.95, 0.95, 0.95],
    ]

    FACADE_TYPE_COLORS = {
        'vertical_facade': [0.95, 0.68, 0.18],
        'horizontal': [0.25, 0.65, 0.95],
        'inclined': [0.55, 0.45, 0.9],
    }

    HIGHLIGHT_COLOR = [0.2, 0.8, 0.2]

    # --- Quality evaluation constants ---
    QUALITY_PASS = 0
    QUALITY_WARN = 1
    QUALITY_FAIL = 2

    # Categorical colors for quality levels (RGB in [0,1])
    QUALITY_COLORS = {
        QUALITY_PASS: [0.20, 0.80, 0.20],  # green
        QUALITY_WARN: [0.95, 0.85, 0.20],  # yellow
        QUALITY_FAIL: [0.95, 0.35, 0.35],  # red
    }

    # 质量评估：是否采用“垂直约束参考面”用于平整度（不影响垂直度计算）
    QUALITY_VERTICAL_REF_PLANE = False
