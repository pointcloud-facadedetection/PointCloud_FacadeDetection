"""通用 NumPy 数组辅助函数。"""
from __future__ import annotations

import numpy as np


def as_array(value, dtype=float):
    """将任意序列或 None 转换为指定 dtype 的 ndarray。"""
    return np.asarray([] if value is None else value, dtype=dtype)


def valid_ids(value, size, dtype=np.int32):
    """返回去重且位于 [0, size) 范围内的有效索引数组。"""
    ids = np.asarray([] if value is None else value, dtype=np.int64).reshape(-1)
    return np.unique(ids[(ids >= 0) & (ids < int(size))]).astype(dtype)
