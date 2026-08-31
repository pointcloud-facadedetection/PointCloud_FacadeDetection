"""Audit metadata for PLY files exported in the global coordinate frame.

The FLS converter applies ``transformToGlobal`` before writing PLY.  Runtime
registration must therefore never apply this matrix again.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GlobalTransformAudit:
    json_path: str
    matrix: np.ndarray
    applied_at_export: bool = True


def audit_exported_global_transform(json_path: str | Path) -> GlobalTransformAudit:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("transformToGlobal", data.get("transform_to_global"))
    if raw is None:
        raise ValueError(f"缺少 transformToGlobal: {path}")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"全局变换矩阵无效: {path}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"全局变换矩阵齐次行无效: {path}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"全局变换矩阵旋转部分不是正交矩阵: {path}")
    if np.linalg.det(rotation) <= 0:
        raise ValueError(f"全局变换矩阵包含镜像变换: {path}")
    return GlobalTransformAudit(str(path.resolve()), matrix, True)
