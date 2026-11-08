import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    DEFAULT_VOXEL_SIZE = 0.05

    # stratified_proxy_build 距离壳层并行的线程数上限（多线程共享同一份点云数组）
    PROXY_BUILD_MAX_WORKERS: int = min(4, os.cpu_count() or 1)

    DETECT_DIST_TOL_MM = 20.0
    # Minimum effective facade area in square meters
    # 合法凹陷立面碎片，造成整墙不可恢复（扩展阶段即被丢弃）。
    MIN_FACADE_AREA = 10.0
    # 进入业务流的最大立面数量：按最终点数降序选取。
    MAX_FACADE_COUNT = 40

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
    FACADE_DETECT_METHOD = 'hybrid'

    # ============================================
    # 新增：种子生成模式切换（bucket / hough / hybrid）
    # ============================================
    # 'bucket' ：方向桶 × 深度桶确定性枚举 
    # 'hough'  ：保留原有空间化 Hough 投票路径 
    # 'hybrid' ：precluster + bucket 双通道全域非竞争（推荐）
    FACADE_SEED_MODE: str = 'hybrid'
    # 输出每个预聚类 patch 的法向/深度/intensity/密度诊断；生产默认关闭。
    FACADE_DEBUG_PATCH_FEATURES: bool = False

    # 分割仅移除有限值错误和真正孤立的稀疏噪点。
    FACADE_ENABLE_SPARSE_OUTLIER_FILTER: bool = True
    FACADE_OUTLIER_RADIUS_M: float = 0.125
    FACADE_OUTLIER_MIN_NEIGHBORS: int = 2

    # 深度+法向局部图预聚类。hybrid 模式双通道全域非竞争运行，全局去重
    # 兜底；不允许一个预聚类结果阻断完整墙面发现。
    FACADE_PRECLUSTER_RADIUS_M: float = 0.20
    FACADE_PRECLUSTER_NORMAL_ANGLE_DEG: float = 12.0
    FACADE_PRECLUSTER_MAX_ELIGIBLE: int = 64

    # 方向桶量化步长（用于 fallback 均匀桶，主桶用弧形半宽）
    FACADE_DIR_BIN_DEG: float = 5.0
    # 每个主方向桶的弧形半宽（°）；窗口不重叠，同一物理面只落进一个桶
    FACADE_DIR_ARC_HALF_DEG: float = 32.0
    FACADE_DIR_BIN_OVERLAP_RATIO = 0.25
    # 方向桶退化兜底：主朝向估计失败时的固定方位桶数量
    FACADE_FALLBACK_BUCKETS: int = 12

    # 深度桶硬间距（米）：同向异层在候选生成阶段就物理隔离
    FACADE_DEPTH_LAYER_HARD_M: float = 0.05
    # 去重深度差阈值严格 < 5cm
    FACADE_DEDUP_DEPTH_THRESH_M: float = 0.045
    # 桶边界双归属模糊带（米）：ρ mod 5cm 在 [0, ε] ∪ [5cm-ε, 5cm] 内双归属
    # 仅用于候选召回的边界模糊带；最终归属必须唯一，且不应覆盖半个硬层间距。
    FACADE_LAYER_BOUNDARY_EPS_M: float = 0.012
    # bucket 路径内启用 Hough ρ 投票校验层中心（补种通道已删除——经两次运行
    # 证明几何上不可触发；残余方向域的层发现由补充通道的 ρ-Hough 承担）
    FACADE_BUCKET_HOUGH_FUSION: bool = True
    # 候选扩张可放宽，但最终 facade 深度门控仍固定为 5cm。
    FACADE_FINAL_DEPTH_RESIDUAL_M: float = 0.03
    FACADE_FINAL_DEPTH_MAD_SCALE: float = 3.0
    FACADE_FINAL_IRLS_ITERS: int = 6
    # ============ 同层合并孔洞桥接自适应（修复大洞口同面分割）============
    # 门窗洞口桥接窗按较大片段尺度自适应。0.75 可覆盖工业建筑宽门、连续
    # 幕墙窗带；仍受 5cm 深度硬约束、法向、单峰性和投影重叠共同约束。
    FACADE_MERGE_GAP_SPAN_RATIO: float = 0.75
    FACADE_MERGE_GAP_MAX_M: float = 12.0
    # 高置信合并对（法向差 ≤2° 且双向平面偏移 ≤ max_plane_dist/2）：
    FACADE_MERGE_CONFIDENT_ANGLE_DEG: float = 2.0
    FACADE_MERGE_CONFIDENT_FILL_SLACK: float = 0.15
    # 同层合并 UV 桥接距离（米）：同层合并时桥接距离容差
    FACADE_MERGE_SAME_LAYER_GAP_M: float = 4.0
    # 同层合并桥接窗下限（米）：近于该值的片段本应是一个连通域，不作候选
    FACADE_MERGE_SAME_LAYER_MIN_GAP_M: float = 0.10
    # 跨门窗桥接必须在与间隙正交的 UV 方向保留投影重叠，避免把空间中斜对角
    # 分离、仅仅共面的独立墙体误并。
    FACADE_MERGE_OPENING_MIN_OVERLAP_M: float = 0.30
    FACADE_MERGE_OPENING_MIN_OVERLAP_RATIO: float = 0.08
    # 跨层裁决浅台阶（Δρ≤5cm）分支的 AABB 邻近窗（米），与同层桥接同尺度
    FACADE_CROSS_LAYER_MAX_AABB_GAP_M: float = 3.0
    # 可选外观特征：数据缺失时自动退化为纯几何；阈值为鲁棒归一化强度差。
    FACADE_ENABLE_INTENSITY_FEATURE: bool = True
    FACADE_MERGE_INTENSITY_DELTA: float = 0.10
    # UV morphological closing radius during single-seed expansion
    FACADE_EXPAND_CLOSE_RADIUS_M = 0.50

    # IRLS 迭代次数（逐片精化）
    IRLS_MAX_ITER : int = 10

    # 完形回收容差倍数（相对 tol）：桶化后错误吸附入口减少，可收紧
    FACADE_RECLAIM_TOL_MULT: float = 1.5
    # 完形回收 UV 膨胀带宽（米）：0.20 → 0.30，补偿边界条带漏点；
    FACADE_RECLAIM_UV_DILATE_M: float = 0.30

    # 新的 precluster 主路径默认不启用旧粗层；显式 legacy 模式可恢复。
    FACADE_ENABLE_LEGACY_FALLBACKS: bool = False
    FACADE_ENABLE_COARSE_LAYER: bool = False

    # 粗层体素倍数（相对 base voxel_size，默认 4x = 20cm）
    FACADE_COARSE_VOXEL_MULT: float = 4.0

    # 建筑外窗洞口最小跨越宽度（米），用于自适应孔洞闭合
    FACADE_MIN_OPENING_M: float = 0.50

    # UV 网格闭合半径上限（防止网格过密时爆炸）
    FACADE_UV_CLOSE_MAX_CELLS: int = 12
    # UV+Depth 三维分层连通：同一 UV 格中深度不同的面不得被孔洞闭合桥接。
    FACADE_DEPTH_LAYER_BIN_M: float = 0.05
    FACADE_DEPTH_LAYER_GAP_M: float = 0.05

    # 主立面边界弱面召回（仅在局部边界带运行，避免全场景二次扫描）。
    FACADE_ENABLE_WEAK_BOUNDARY_RECALL: bool = False
    FACADE_WEAK_BOUNDARY_BAND_M: float = 0.50
    FACADE_WEAK_MIN_AREA_M2: float = 20.00
    FACADE_WEAK_MIN_POINTS: int = 200
    FACADE_WEAK_MIN_ANGLE_DEG: float = 30.0
    FACADE_WEAK_MIN_DENSITY: float = 20.0          # 最小点密度（点数/平方米）
    FACADE_WEAK_MIN_COMPACT_RATIO: float = 0.50    # 最大连通域占比
    FACADE_WEAK_MAX_COMPONENTS: int = 4            # 最大连通域数量
    FACADE_WEAK_MAX_PER_FACADE: int = 4            # 每个主立面最多召回数
    FACADE_WEAK_EXTEND_TOL_MULT: float = 0.8       # 扩展时距离容差乘数
    FACADE_WEAK_EXTEND_ANGLE_DEG: float = 5.0      # 扩展点法向与弱平面法向最大夹角

    # 粗层 signed 距离容差倍数（相对 base）
    FACADE_COARSE_SIGNED_MULT: float = 2.0

    # 细层 signed 距离容差
    FACADE_SIGNED_DIST_TOL: float = 1.0

    # Hough 参数调优（ρ 投票层中心校验 + 补充通道层发现）
    HOUGH_MAX_VOTE_POINTS: int = 100000
    HOUGH_MAX_DIRECTION_PEAKS: int = 8
    # 【三期】4 → 12：补充通道需要对残余方向域发现更多层中心
    HOUGH_MAX_RHO_PEAKS_PER_DIRECTION: int = 12
    HOUGH_MAX_CANDIDATES: int = 24
    HOUGH_RHO_MIN_PROMINENCE: float = 0.08
    HOUGH_DENSITY_ADAPTIVE: bool = True
    HOUGH_LOW_DENSITY_THRESH: float = 5.0        # 点/立方米
    HOUGH_LOW_DENSITY_BIN_MULT: float = 2.0      # bin_size = base * mult
    HOUGH_LOW_DENSITY_MIN_SUPPORT_RATIO: float = 0.2  # 相对 min_count
    HOUGH_LOW_DENSITY_MIN_PEAK_DIST_M: float = 0.10
    HOUGH_LOW_DENSITY_PROMINENCE_RATIO: float = 0.10
    HOUGH_PEAK_MERGE_VALLEY_RATIO: float = 0.30
    HOUGH_MAX_WEAK_SEEDS: int = 16
    HOUGH_WEAK_SUPPORT_RATIO: float = 0.20

    # 平面拟合参数
    RANSAC_ITERATIONS: int = 50            # RANSAC 迭代次数
    RANSAC_THRESHOLD_RATIO: float = 1.5     # 阈值 = voxel_size * ratio

    # 立面合并参数
    MERGE_ANGLE_DEG: float = 5.0            # 合并法向夹角阈值
    MERGE_D_THRESH_M: float = 0.08          # 合并平面距离阈值
    MERGE_UV_DIST_M: float = 3.0            # 合并 UV BBox 距离阈值
    FACADE_MERGE_UV_DIST_M: float = 5.0
    FACADE_MERGE_NORMAL_RESIDUAL_DEG: float = 2.0
    FACADE_MERGE_PROTECT_WEAK_PLANES: bool = True

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

    # ============ 新增：密度估计模块 ============
    # 距离壳层（与 stratified_proxy_build 保持一致）
    DENSITY_SHELLS = ((10., .10), (20., .08), (35., .06),
                      (50., .05), (80., .045), (100., .04))
    # 高/低层分界高度角（度）；高处扫描线更稀疏，单独设阈值
    DENSITY_ELEV_SPLIT_DEG: float = 30.0
    # 壳层绝对密度仅保留为诊断与容差参考；实际等级由局部/同壳层相对密度决定。
    DENSITY_DENSE_PTS_M2: float = 200.0
    DENSITY_SPARSE_PTS_M2: float = 25.0
    # 30cm 分析格显著大于 5cm 代理体素，可识别洞口边缘和局部遮挡；
    # 每个距离壳层×高度层独立估计期望密度，避免远距/高层被近距阈值误杀。
    DENSITY_ANALYSIS_CELL_M: float = 0.30
    DENSITY_RELATIVE_DENSE_RATIO: float = 0.55
    DENSITY_RELATIVE_SPARSE_RATIO: float = 0.10
    # 区域门控比例：skip 占比 > 50% 跳过整个区域；skip+sparse > 35% 降级 sparse
    DENSITY_GATE_SKIP_RATIO: float = 0.50
    DENSITY_GATE_SPARSE_RATIO: float = 0.35
    # SPARSE 区域最小点数门槛倍率（召回稀疏窗台立面）
    DENSITY_SPARSE_MIN_COUNT_MULT: float = 0.40
    # 容差膨胀上限（密度越低 tol 越大，但不得超过此倍数——保证不突破 5cm 硬约束）
    DENSITY_TOL_MAX_SCALE: float = 1.6

    # ============ 新增：残余方向补充通道 ============
    FACADE_SUPPLEMENTAL_ENABLE: bool = True
    # 补充域总点数门槛：≥ min_count × 此比例才触发（min_count≈2000 → 1000 点）
    FACADE_SUPPLEMENTAL_MIN_TOTAL_RATIO: float = 0.5
    # 每个补充方向的最小支持度与方向数上限
    FACADE_SUPPLEMENTAL_MAX_DIRS: int = 4

    # --- Viewport interaction & camera settings ---
    ORTHO_FOV_DEG = 5.0
    ORTHO_DEFAULT_VIEW = 'neg_y'
    ZOOM_WHEEL_SCALE = 1.6
    PAN_BASE_SPEED = 0.06
    ROTATE_SPEED = 1.0
    ORTHO_WPP_SCALE = 1.0
    DEBUG_SELECTION = True

    # --- Selection overlay style ---
    SELECT_BORDER_RGBA = (255, 0, 0, 240)
    SELECT_FILL_RGBA = (255, 0, 0, 25)
    SELECT_BORDER_WIDTH = 2

    # 立面实例颜色
    FACADE_INSTANCE_COLORS = [
        [0.50, 0.05, 0.85],
        [0.05, 0.55, 0.10],
        [0.10, 0.05, 0.80],
        [0.80, 0.05, 0.55],
        [0.05, 0.40, 0.45],
        [0.55, 0.35, 0.15],
        [0.65, 0.10, 0.75],
        [0.08, 0.70, 0.12],
        [0.10, 0.20, 0.75],
        [0.85, 0.15, 0.80],
        [0.15, 0.35, 0.18],
        [0.50, 0.15, 0.35],
        [0.40, 0.28, 0.18],
        [0.25, 0.08, 0.45],
        [0.15, 0.25, 0.30],
        [0.35, 0.12, 0.55],
        [0.20, 0.18, 0.50],
        [0.55, 0.20, 0.45],
        [0.20, 0.30, 0.55],
        [0.30, 0.45, 0.20],
    ]

    FACADE_TYPE_COLORS = {
        'vertical_facade': [0.95, 0.68, 0.18],
        'horizontal': [0.25, 0.65, 0.95],
        'inclined': [0.55, 0.45, 0.9],
    }

    HIGHLIGHT_COLOR = [0.2, 0.8, 0.2]
    FACADE_BASE_COLOR = [0.55, 0.55, 0.55]
    QUALITY_HEATMAP_BASE_COLOR = [0.72, 0.78, 0.86]

    VIEWPORT_BACKGROUND = '#FFFFFF'
    DEFAULT_CLOUD_GRAY = 0.45

    QUALITY_PASS = 0
    QUALITY_WARN = 1
    QUALITY_FAIL = 2

    QUALITY_COLORS = {
        QUALITY_PASS: [0.20, 0.80, 0.20],
        QUALITY_WARN: [0.95, 0.85, 0.20],
        QUALITY_FAIL: [0.95, 0.35, 0.35],
    }

    QUALITY_VERTICAL_REF_PLANE = False

    GLOBAL_PLANE_HUBER_DELTA_M: float = 0.005
    GLOBAL_PLANE_MAX_ITERATIONS: int = 500
    GLOBAL_PLANE_ANGLE_LIMIT_DEG: float = 3.0
    GLOBAL_PLANE_MIN_INLIER_RATIO: float = 0.15
    GLOBAL_PLANE_MAX_P95_MM: float = 200.0
    GLOBAL_PLANE_ENABLE_PARTITION_FALLBACK: bool = True
    GLOBAL_PLANE_PARTITION_DEPTH_GAP_M: float = 0.03
    GLOBAL_PLANE_PARTITION_MIN_POINTS_RATIO: float = 0.10
    GLOBAL_PLANE_OUTLIER_SIGMA: float = 3.0
    GLOBAL_PLANE_FINAL_GATE_SIGMA: float = 3.5
    GLOBAL_PLANE_ENABLE_P95_REFIT: bool = True
    # 可信核心保留比例：默认 50%，字段名保留兼容旧报告和调用方。
    GLOBAL_PLANE_P95_CONFIDENCE: float = 0.50

    # ============ 效率优化参数 ============
    # 质量评估 CoverageMask 分辨率（米）：默认 5cm，权衡精度与速度
    QUALITY_COVERAGE_RESOLUTION_M: float = 0.05

    # 靠尺法默认并行模式：'thread' 避免 Windows spawn 开销
    RULER_PARALLEL_MODE: str = 'thread'