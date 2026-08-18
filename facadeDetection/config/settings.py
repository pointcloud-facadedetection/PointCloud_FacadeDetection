import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DEFAULT_VOXEL_SIZE = 0.05

    QUALITY_GRID_SIZE = 20.0
    RULER_SIZE = 0.05
    RULER_STEP = 0.05

    DETECT_DIST_TOL_MM = 20.0
    # Minimum effective facade area in square meters
    MIN_FACADE_AREA = 5.0

    # 垂直立面法向 Z 分量阈值（|nz| < 此值视为垂直候选）
    VERTICAL_NZ_THR: float = 0.30

    # 法向一致性阈值：邻域内法向夹角标准差上限（度）
    # 墙面法向一致（低方差），树木法向杂乱（高方差）
    NORMAL_VARIANCE_DEG: float = 15.0

    # 聚类参数
    CLUSTER_ANGLE_DEG: float = 5.0          # 法向聚类角度阈值
    CLUSTER_MIN_POINTS: int = 120           # 最小聚类点数
    FACADE_SPHERE_GRID_RES: float = 5.0
    FACADE_NORMAL_ANGLE_DEG: float = 8.0

    # 平面拟合参数
    RANSAC_ITERATIONS: int = 50            # RANSAC 迭代次数
    RANSAC_THRESHOLD_RATIO: float = 1.5     # 阈值 = voxel_size * ratio

    # 立面合并参数
    MERGE_ANGLE_DEG: float = 5.0            # 合并法向夹角阈值
    MERGE_D_THRESH_M: float = 0.10          # 合并平面距离阈值
    MERGE_UV_DIST_M: float = 3.0            # 合并 UV BBox 距离阈值
    FACADE_MERGE_UV_DIST_M: float = 5.0

    # 生长参数
    GROW_NORMAL_TOL_DEG: float = 8.0        # 生长法向一致性容忍
    GROW_DIST_MULT: float = 2.0             # 生长距离倍数（相对 max_plane_dist）

    # 测量网格参数
    MEASUREMENT_GRID_M: float = 0.10        # 测量网格单元大小
    MEASUREMENT_MIN_CELL_PTS: int = 2       # 最小单元点数

    # 质量域采用检测残差和体素误差共同确定深度范围。
    FACADE_QUALITY_DEPTH_MULT: float = 3.0
    FACADE_QUALITY_DEPTH_MIN_M: float = 0.02
    FACADE_QUALITY_DEPTH_MAX_M: float = 0.50


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