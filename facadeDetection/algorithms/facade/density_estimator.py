"""点云密度估计模块：距离壳层 × 高度层的二维密度场。

设计目标
--------
1. 与现有 ``stratified_proxy_build`` 的壳层定义保持一致（距离壳层），
   再叠加高度角分层（低层/高层），形成 (shell, elev_layer) 二维密度单元。
2. 密度以「面密度估计」为准：体素占用数 × 平均体素点数 / 单位壳层面积，
   避免稀疏区 bbox 体积虚大导致密度低估（旧 Hough 路径 pts/m³ 的缺陷）。
3. 内部三级阈值判定：
     DENSE  —— 正常处理；
     SPARSE —— 放宽种子/连通域最小点数继续处理（召回窗台等稀疏立面）；
     SKIP   —— 过于稀疏，直接跳过，不参与任何拟合与评估。
4. 提供 per-point 密度查询（供 tol/min_count 自适应）与
   per-region 门控（供 bucket 种子与连通域两级跳过）。

使用方式
--------
    field = build_density_field(points, ranges, elevations=None,
                                voxel_size=0.05)
    level = field.gate_indices(indices)      # 'dense' | 'sparse' | 'skip'
    dens  = field.point_density[indices]     # 估计面密度 pts/m²
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import numpy as np

try:
    from config.settings import Config
except Exception:  # 允许脱离工程单独测试
    Config = None


def _cfg(name, default):
    if Config is None:
        return default
    return getattr(Config, name, default)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DensityCell:
    """单个 (壳层, 高度层) 密度单元的统计量。"""
    shell_lo: float
    shell_hi: float
    elev_layer: int            # 0=低层 1=高层
    point_count: int
    area_m2: float             # 占用体素在壳层切面上的投影面积估计
    density_m2: float          # 面密度 pts/m²
    level: str                 # 'dense' | 'sparse' | 'skip'


@dataclass
class DensityField:
    """密度场：单元表 + 逐点密度 + 逐点等级。"""
    point_density: np.ndarray          # (N,) pts/m²
    point_level: np.ndarray            # (N,) int8: 0=dense 1=sparse 2=skip
    cells: list = _dc_field(default_factory=list)
    params: dict = _dc_field(default_factory=dict)

    LEVEL_DENSE = 0
    LEVEL_SPARSE = 1
    LEVEL_SKIP = 2

    def gate_indices(self, indices) -> str:
        """对一组点索引做区域门控。众数等级决定结果。"""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(idx) == 0:
            return 'skip'
        lv = self.point_level[idx]
        # skip 占比过半 → skip；sparse 占比较高 → sparse；否则 dense
        frac_skip = float(np.mean(lv == self.LEVEL_SKIP))
        frac_sparse = float(np.mean(lv == self.LEVEL_SPARSE))
        if frac_skip > float(self.params.get('gate_skip_ratio', 0.5)):
            return 'skip'
        if frac_skip + frac_sparse > float(self.params.get('gate_sparse_ratio', 0.35)):
            return 'sparse'
        return 'dense'

    def sparse_mask(self, indices) -> np.ndarray:
        """返回 indices 中处于 skip 区域的布尔掩码（True=应剔除）。"""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        return self.point_level[idx] == self.LEVEL_SKIP


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _pack_voxel_keys(keys: np.ndarray) -> np.ndarray:
    """三维体素键混合进制打包为 int64 标量，替代 np.unique(axis=0)。

    np.unique(axis=0) 对 270 万点量级需字典化排序，实测是密度场构建的
    主要耗时（15s 中的 ~70%）；打包后退化为一维 unique，提速约 5~8 倍。
    """
    k = keys - keys.min(axis=0)
    ny = int(k[:, 1].max()) + 1
    nz = int(k[:, 2].max()) + 1
    return (k[:, 0].astype(np.int64) * ny + k[:, 1]) * nz + k[:, 2]


def _voxel_occupancy_area(pts: np.ndarray, voxel: float) -> tuple:
    """占用体素数与平均体素点数 → (占用体素数, 平均每体素点数)。

    壳层切面面积 ≈ 占用体素数 × voxel²（体素面在壳层切面上的投影）。
    """
    if len(pts) == 0:
        return 0, 0.0
    keys = np.floor(pts / max(float(voxel), 1e-6)).astype(np.int64)
    _, counts = np.unique(_pack_voxel_keys(keys), return_counts=True)
    return int(len(counts)), float(np.mean(counts))


def _classify_level(density_m2: float, dense_thr: float,
                    sparse_thr: float) -> str:
    if density_m2 >= dense_thr:
        return 'dense'
    if density_m2 >= sparse_thr:
        return 'sparse'
    return 'skip'


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_density_field(points: np.ndarray,
                        ranges: np.ndarray,
                        elevations: np.ndarray | None = None,
                        voxel_size: float = 0.05,
                        shells=None,
                        elev_split_deg: float | None = None,
                        dense_thr_m2: float | None = None,
                        sparse_thr_m2: float | None = None,
                        ) -> DensityField:
    """构建 (距离壳层 × 高度层) 密度场。

    Parameters
    ----------
    points : (N,3) 原始点云（建议传 ROI 内点云，索引与返回值对齐）
    ranges : (N,) 测距；无真实测距时传 bbox 归一化代理值亦可（仅分壳）
    elevations : (N,) 高度角（度）；None 时按 z 分位近似高/低层
    shells : 距离壳层 ((hi, voxel_hint), ...)，默认与工程现有壳层一致
    elev_split_deg : 高/低层分界高度角，默认读 Config.DENSITY_ELEV_SPLIT_DEG
    dense_thr_m2 : DENSE 下限（pts/m²），默认 Config.DENSITY_DENSE_PTS_M2
    sparse_thr_m2: SPARSE 下限（pts/m²），默认 Config.DENSITY_SPARSE_PTS_M2
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rng = np.asarray(ranges, dtype=np.float64).reshape(-1)
    rng = np.where(np.isfinite(rng), rng, 0.0)
    n = len(pts)
    if n == 0:
        return DensityField(np.empty(0), np.empty(0, np.int8), [], {})

    if shells is None:
        shells = tuple(_cfg('DENSITY_SHELLS',
                            ((10., .10), (20., .08), (35., .06),
                             (50., .05), (80., .045), (100., .04),
                             (float('inf'), .04))))
    if elev_split_deg is None:
        elev_split_deg = float(_cfg('DENSITY_ELEV_SPLIT_DEG', 30.0))
    if dense_thr_m2 is None:
        dense_thr_m2 = float(_cfg('DENSITY_DENSE_PTS_M2', 200.0))
    if sparse_thr_m2 is None:
        sparse_thr_m2 = float(_cfg('DENSITY_SPARSE_PTS_M2', 25.0))

    # ---- 高度层标记 ----
    if elevations is not None and len(elevations) == n:
        elev = np.abs(np.asarray(elevations, dtype=np.float64).reshape(-1))
        elev_layer = (elev >= elev_split_deg).astype(np.int8)
    else:
        # 无测站高度角时以 z 中位数近似分层（高层扫描线更稀疏的保守近似）
        z_med = float(np.median(pts[:, 2]))
        elev_layer = (pts[:, 2] >= z_med).astype(np.int8)

    # ---- 壳层归属 ----
    bounds = [float(hi) for hi, _ in shells]
    shell_idx = np.zeros(n, dtype=np.int32)
    lo = 0.0
    for si, hi in enumerate(bounds):
        m = (rng >= lo) & (rng < hi)
        shell_idx[m] = si
        lo = hi
    shell_idx[rng >= lo] = len(bounds) - 1

    point_density = np.zeros(n, dtype=np.float64)
    point_level = np.zeros(n, dtype=np.int8)
    cells: list[DensityCell] = []

    level_map = {'dense': DensityField.LEVEL_DENSE,
                 'sparse': DensityField.LEVEL_SPARSE,
                 'skip': DensityField.LEVEL_SKIP}

    for si, (hi, voxel_hint) in enumerate(shells):
        lo_shell = 0.0 if si == 0 else float(shells[si - 1][0])
        for el in (0, 1):
            m = (shell_idx == si) & (elev_layer == el)
            cnt = int(np.count_nonzero(m))
            if cnt < 3:
                # 单元内点太少：密度不可估，但不应误杀——按保守 sparse 处理
                if cnt:
                    point_density[m] = sparse_thr_m2
                    point_level[m] = DensityField.LEVEL_SPARSE
                continue

            sub = pts[m]
            # 单元体素：取壳层建议体素与全局 voxel 的较大者，防止体素过密
            cell_voxel = max(float(voxel_size), float(voxel_hint or voxel_size))
            n_occ, avg_per_voxel = _voxel_occupancy_area(sub, cell_voxel)

            # 面密度估计：
            #   切面面积 ≈ 占用体素数 × cell_voxel²（壳层切向投影）
            #   density = cnt / area
            area_m2 = max(n_occ * cell_voxel * cell_voxel, 1e-6)
            density = cnt / area_m2

            level = _classify_level(density, dense_thr_m2, sparse_thr_m2)
            point_density[m] = density
            point_level[m] = level_map[level]
            cells.append(DensityCell(
                shell_lo=lo_shell, shell_hi=float(hi), elev_layer=el,
                point_count=cnt, area_m2=area_m2,
                density_m2=float(density), level=level))

    # ---- 局部精化：体素网格内逐点二次判定 ----
    # 壳层×高度层仍可能掩盖局部极稀疏斑块（如被树木遮挡的墙面）。
    # 在全局 voxel 网格上估计局部占用密度，覆盖单元级结果（只降不升）。
    local_voxel = max(float(voxel_size) * 4.0, 0.20)
    keys = np.floor(pts / local_voxel).astype(np.int64)
    _, inv, counts = np.unique(_pack_voxel_keys(keys), return_inverse=True,
                               return_counts=True)
    expected = max((local_voxel / max(float(voxel_size), 1e-6)) ** 2, 1.0)
    local_fill = counts[inv] / expected          # 局部填充率 ≈ 相对密度
    local_sparse_thr = float(_cfg('DENSITY_LOCAL_SPARSE_FILL', 0.05))
    local_skip_thr = float(_cfg('DENSITY_LOCAL_SKIP_FILL', 0.01))
    downgrade = np.where(local_fill < local_skip_thr,
                         DensityField.LEVEL_SKIP,
                         np.where(local_fill < local_sparse_thr,
                                  DensityField.LEVEL_SPARSE,
                                  DensityField.LEVEL_DENSE))
    point_level = np.maximum(point_level, downgrade).astype(np.int8)
    # 局部极稀疏时把逐点密度一并压低，供下游 tol 膨胀使用
    point_density = np.where(downgrade == DensityField.LEVEL_SKIP,
                             np.minimum(point_density, sparse_thr_m2 * 0.5),
                             point_density)

    return DensityField(
        point_density=point_density,
        point_level=point_level,
        cells=cells,
        params={'dense_thr_m2': dense_thr_m2,
                'sparse_thr_m2': sparse_thr_m2,
                'elev_split_deg': elev_split_deg,
                'gate_skip_ratio': float(_cfg('DENSITY_GATE_SKIP_RATIO', 0.5)),
                'gate_sparse_ratio': float(
                    _cfg('DENSITY_GATE_SPARSE_RATIO', 0.35))})


# ---------------------------------------------------------------------------
# 下游适配器：密度 → 处理参数
# ---------------------------------------------------------------------------

def adaptive_min_count(base_min_count: int, level: str) -> int:
    """按区域密度等级调整最小点数门槛。SPARSE 放宽以召回稀疏立面。"""
    if level == 'sparse':
        mult = float(_cfg('DENSITY_SPARSE_MIN_COUNT_MULT', 0.4))
        return max(20, int(base_min_count * mult))
    if level == 'skip':
        return 10 ** 9       # 实践中不可达：上游已跳过
    return int(base_min_count)


def adaptive_tol_scale(base_tol: float, density_m2: float) -> float:
    """密度越低，逐点容差适度膨胀（稀疏点平面残差天然偏大）。

    膨胀设上限，避免把退台立面重新糊在一起（硬约束 5cm 不可突破）。
    """
    dense_thr = float(_cfg('DENSITY_DENSE_PTS_M2', 200.0))
    max_scale = float(_cfg('DENSITY_TOL_MAX_SCALE', 1.6))
    if density_m2 >= dense_thr:
        return 1.0
    scale = np.sqrt(dense_thr / max(density_m2, 1.0))
    return float(min(scale, max_scale))