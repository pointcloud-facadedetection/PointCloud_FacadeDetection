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
    window_size_m: float = 2.0
    step_size_m: float = 0.05
    measure_height_m: float = 2.0
    min_points: int = 30
    pass_rate: float = 0.95
    warn_ratio: float = 1.0
    fail_ratio: float = 2.0
    algorithm_version: str = "facade-quality-v1"

    def snapshot(self) -> dict:
        return asdict(self)


class InspectionProfileService:
    """标准注册表；后续可将注册表替换为数据库，不影响算法接口。"""
    PRESETS = (
        InspectionProfile("masonry_normal", "普通砌块", "v1.0", "masonry", 8.0, 5.0),
        InspectionProfile("masonry_precision", "高精砌块", "v1.0", "masonry", 4.0, 4.0),
        InspectionProfile("timber_formwork", "木模", "v1.0", "formwork", 8.0, 8.0),
        InspectionProfile("aluminum_formwork", "铝模", "v1.0", "formwork", 4.0, 4.0),
    )

    @classmethod
    def all(cls):
        return cls.PRESETS

    @classmethod
    def get(cls, standard_id: str):
        return next((p for p in cls.PRESETS if p.standard_id == standard_id), None)