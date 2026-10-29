"""楼宇层高配置的解析、校验与楼层标高计算。"""
from __future__ import annotations

import json
from typing import Any


def default_building_info() -> dict:
    return {
        "schema_version": 2,
        "base_elevation_m": 0.0,
        "base_floor_no": 1,
        "standard_floor_height_m": 3.0,
        "total_floor_count": 1,
        "special_floors": [],
    }


def parse_building_info(value: Any) -> dict:
    """从项目参数 JSON/dict 中读取楼宇信息，旧项目返回默认值。"""
    params = value
    if isinstance(value, str):
        try:
            params = json.loads(value)
        except (TypeError, ValueError):
            params = {}
    source = params.get("building_info", {}) if isinstance(params, dict) else {}
    result = default_building_info()
    if isinstance(source, dict):
        result.update({k: source[k] for k in result if k in source})
        specials = source.get("special_floors", [])
        result["special_floors"] = [dict(item) for item in specials if isinstance(item, dict)]
    return result


def validate_building_info(data: dict) -> tuple[bool, str]:
    try:
        total = int(data.get("total_floor_count", 0))
        standard = float(data.get("standard_floor_height_m", 0))
        float(data.get("base_elevation_m", 0))
    except (TypeError, ValueError):
        return False, "楼宇信息中的数值格式不正确"
    try:
        base_floor = int(data.get("base_floor_no", 1))
    except (TypeError, ValueError):
        return False, "基准标高对应楼层号必须为整数"
    if base_floor < -20 or base_floor > 300:
        return False, "基准标高对应楼层号超出范围"
    if total < 1:
        return False, "楼宇总层数必须至少为 1 层"
    if standard <= 0:
        return False, "标准楼层层高必须大于 0"
    start_floor = base_floor
    end_floor = start_floor + total - 1
    seen = set()
    for item in data.get("special_floors", []):
        try:
            floor_no = int(item.get("floor_no"))
            height = float(item.get("height_m"))
        except (TypeError, ValueError, AttributeError):
            return False, "特殊楼层编号和层高必须为有效数字"
        if floor_no < start_floor or floor_no > end_floor:
            return False, f"特殊楼层 {floor_no} 超出本次标注范围（{start_floor}F-{end_floor}F）"
        if floor_no in seen:
            return False, f"特殊楼层 {floor_no} 重复配置"
        if height <= 0:
            return False, f"特殊楼层 {floor_no} 的层高必须大于 0"
        seen.add(floor_no)
    return True, ""


def floor_height(floor_no: int, data: dict) -> float:
    for item in data.get("special_floors", []):
        if int(item["floor_no"]) == int(floor_no):
            return float(item["height_m"])
    return float(data["standard_floor_height_m"])


def floor_elevations(data: dict) -> list[dict]:
    """返回每层起始标高、层高和顶部标高；特殊层高度归属于该层。

    ``base_elevation_m`` 与 ``base_floor_no`` 是同一个工程基准锚点：
    基准标高对应的楼层号由配置直接指定，后续楼层只按实际层高累加。
    本函数有意不接收点云坐标，也不根据世界 Z 推断楼层号；点云世界坐标
    与工程标高的转换属于几何投影边界，不能改变这里的楼层编号语义。
    """
    current = float(data.get("base_elevation_m", 0.0))
    rows = []
    start = int(data.get("base_floor_no", 1))
    for number in range(start, start + int(data.get("total_floor_count", 0))):
        height = floor_height(number, data)
        rows.append({"floor_no": number, "elevation_m": current,
                     "height_m": height, "top_elevation_m": current + height})
        current += height
    return rows
