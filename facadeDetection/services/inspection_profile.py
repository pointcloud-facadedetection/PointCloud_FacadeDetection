"""工程质量检测标准与计算参数的统一映射。"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InspectionProfile:
    standard_id: str
    standard_name: str
    version: str
    wall_type: str
    flatness_limit_mm: float
    verticality_limit_mm: float
    interval_size_m: float = 20.0
    window_size_m: float = 0.05
    step_size_m: float = 0.05
    measure_height_m: float = 2.0
    min_points: int = 30
    pass_rate: float = 0.95
    warn_ratio: float = 1.0
    fail_ratio: float = 2.0
    algorithm_version: str = "facade-quality"
    ruler_width_m: float = 0.055
    select_band_m: float = 0.01
    hole_band_m: float = 0.02
    bin_size_m: float = 0.04
    top_q: float = 1.0
    strip_step_m: float = 0.05
    sor_enabled: bool = True
    sor_sigma: float = 4.0
    sor_k: int = 8
    sor_method: str = "local"
    sor_w_weight: float = 50.0
    scan_step_m: float = 0.05
    max_hole_ratio: float = 0.20
    verticality_bin_size_m: float = 0.05
    verticality_enabled: bool = True
    parallel_mode: str = "process"
    n_jobs: int = 4

    def snapshot(self) -> dict:
        return asdict(self)


class InspectionProfileService:
    """标准注册表；后续可将注册表替换为数据库，不影响算法接口。"""
    PRESETS = (
        InspectionProfile("structure_main", "主体", "v2.0", "structure", 8.0, 10.0),
        InspectionProfile("plaster_normal", "普通抹灰", "v2.0", "plaster", 4.0, 4.0),
        InspectionProfile("plaster_advanced", "高级抹灰", "v2.0", "plaster", 3.0, 3.0),
        InspectionProfile("tile_finish", "饰面砖", "v2.0", "tile", 4.0, 3.0),
        InspectionProfile("coating_normal", "普通涂饰", "v2.0", "coating", 4.0, 4.0),
        InspectionProfile("coating_advanced", "高级涂饰", "v2.0", "coating", 3.0, 3.0),
    )

    @classmethod
    def all(cls):
        return cls.PRESETS

    @classmethod
    def get(cls, standard_id: str):
        return next((p for p in cls.PRESETS if p.standard_id == standard_id), None)