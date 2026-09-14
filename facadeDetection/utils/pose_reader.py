"""兼容读取 E57 / FLS 两种缓存格式的测站位姿（旋转矩阵 + 平移向量）。

两种存储路径：
  1. E57:  {cache_dir}/{station_name}.json
  2. FLS:  {cache_dir}/{station_name}.fls/pointclouds/{station_name}.json

返回统一命名字段，便于下游算法直接消费。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


def _try_load_json(path: Path) -> Optional[dict]:
    """尝试读取并解析 JSON，失败返回 None。"""
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _extract_pose_from_e57(data: dict) -> Optional[Dict[str, list]]:
    """从 E57 侧载 JSON 中提取首个 scan 的位姿。

    E57 结构：
        {
          "scan_poses": [
            {
              "rotation_matrix": [[...], [...], [...]],
              "scan_position": [x, y, z],
              "transform_to_global": [[...4x4...]]
            }
          ]
        }
    """
    scan_poses = data.get("scan_poses")
    if not isinstance(scan_poses, list) or not scan_poses:
        return None
    pose = scan_poses[0]
    rot = pose.get("rotation_matrix")
    trans = pose.get("scan_position")
    transform = pose.get("transform_to_global")
    if rot is None or trans is None:
        return None
    return {
        "rotation_matrix": rot,
        "translation": trans,
        "transform_to_global": transform,
    }


def _extract_pose_from_fls(data: dict) -> Optional[Dict[str, list]]:
    """从 FLS 侧载 JSON 中提取位姿。

    FLS 结构：
        {
          "rotationMatrix": [[...], [...], [...]],
          "scanPosition": [x, y, z],
          "transformToGlobal": [[...4x4...]]
        }
    """
    rot = data.get("rotationMatrix")
    trans = data.get("scanPosition")
    transform = data.get("transformToGlobal")
    if rot is None or trans is None:
        return None
    return {
        "rotation_matrix": rot,
        "translation": trans,
        "transform_to_global": transform,
    }


def read_station_pose(
    cache_dir: Union[str, Path],
    station_name: str,
    as_numpy: bool = False,
) -> Dict[str, Union[list, np.ndarray, None]]:
    """读取指定测站的旋转矩阵与平移向量。

    优先尝试 E57 缓存路径，其次尝试 FLS 缓存路径。

    Parameters
    ----------
    cache_dir : str | Path
        项目 cache 目录，例如 ``data/projects/<uuid>/cache``。
    station_name : str
        测站名称（不含扩展名），例如 ``bllygg01``。
    as_numpy : bool, optional
        若为 True，返回的矩阵/向量自动转为 ``np.ndarray``；
        否则保持原生 ``list``。默认 False。

    Returns
    -------
    dict
        统一字段命名：
        - ``rotation_matrix`` : 3×3 旋转矩阵
        - ``translation``     : [x, y, z] 平移向量
        - ``transform_to_global`` : 可选的 4×4 齐次变换矩阵（可能为 None）

    Raises
    ------
    FileNotFoundError
        两种路径均未找到有效 JSON 文件。
    ValueError
        JSON 存在但无法解析出有效位姿字段。
    """
    cache_dir = Path(cache_dir).expanduser().resolve()

    # 1) 尝试 E57 格式
    e57_path = cache_dir / f"{station_name}.json"
    data = _try_load_json(e57_path)
    pose = _extract_pose_from_e57(data) if data is not None else None

    # 2) 尝试 FLS 格式
    if pose is None:
        fls_path = cache_dir / f"{station_name}.fls" / "pointclouds" / f"{station_name}.json"
        data = _try_load_json(fls_path)
        pose = _extract_pose_from_fls(data) if data is not None else None

    if data is None:
        raise FileNotFoundError(
            f"未找到测站 '{station_name}' 的位姿文件；"
            f"已尝试: {e57_path} 与 {fls_path}"
        )
    if pose is None:
        raise ValueError(
            f"在位姿文件中找到测站 '{station_name}'，但缺少必要的 "
            f"rotation_matrix / translation 字段。"
        )

    if as_numpy:
        pose = {
            k: np.asarray(v, dtype=float) if v is not None else None
            for k, v in pose.items()
        }

    return pose


def get_all_station_poses(
    cache_dir: Union[str, Path],
    as_numpy: bool = False,
) -> Dict[str, Dict[str, Union[list, np.ndarray, None]]]:
    """批量读取 cache 目录下所有可识别的测站位姿。

    扫描规则：
      - ``{cache_dir}/*.json``                → E57 格式
      - ``{cache_dir}/*.fls/pointclouds/*.json`` → FLS 格式

    Returns
    -------
    dict
        ``{station_name: pose_dict}``，其中 pose_dict 与
        :func:`read_station_pose` 返回值一致。
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    results: Dict[str, dict] = {}

    # E57
    for json_path in cache_dir.glob("*.json"):
        name = json_path.stem
        data = _try_load_json(json_path)
        pose = _extract_pose_from_e57(data) if data is not None else None
        if pose is not None:
            results[name] = pose

    # FLS
    for fls_dir in cache_dir.glob("*.fls"):
        name = fls_dir.stem
        json_path = fls_dir / "pointclouds" / f"{name}.json"
        data = _try_load_json(json_path)
        pose = _extract_pose_from_fls(data) if data is not None else None
        if pose is not None:
            results[name] = pose

    if as_numpy:
        for name, pose in results.items():
            results[name] = {
                k: np.asarray(v, dtype=float) if v is not None else None
                for k, v in pose.items()
            }

    return results