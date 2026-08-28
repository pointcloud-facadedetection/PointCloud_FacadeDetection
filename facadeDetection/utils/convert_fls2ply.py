from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from utils.logging_utils import trace
except Exception:  # pragma: no cover - keep converter usable as a standalone script
    def trace(stage: str, **fields):
        msg = f"[PCFD] {stage}"
        if fields:
            msg += " " + " ".join(f"{key}={value}" for key, value in fields.items())
        print(msg, flush=True)

# 默认 pybind 模块所在目录（可通过环境变量 FLS_CONVERTER_DLL_DIR 覆盖）。
DEFAULT_DLL_DIR = str(Path(__file__).resolve().parents[3] / "FlsConverter_package" / "FlsConverter")


@dataclass
class ScanMeta:
    scan_name: str = ""
    scanner_type: str = ""
    scanner_serial: str = ""
    num_cols: int = 0
    num_rows: int = 0
    total_num_rows: int = 0
    row_start_angle_rad: float = 0.0
    row_end_angle_rad: float = 0.0
    col_end_angle_rad: float = 0.0
    scanner_range_m: float = 0.0
    dist_offset_m: float = 0.0
    dist_factor: float = 0.0
    scan_time: str = ""
    scan_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scan_angle_rad: float = 0.0
    transform_to_global: List[List[float]] = field(default_factory=lambda: [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    ])
    gps: Dict = field(default_factory=dict)
    compass: Dict = field(default_factory=dict)
    sensor_usage: Dict = field(default_factory=dict)
    ply_path: str = ""
    json_path: str = ""
    point_count: int = 0
    has_color: bool = False
    has_intensity: bool = False
    has_distance: bool = False
    scan_origin: Optional[List[float]] = None
    scan_origins: Optional[List[List[float]]] = None


@dataclass
class ConversionResult:
    success: bool = False
    fls_path: str = ""
    output_dir: str = ""
    scans: List[ScanMeta] = field(default_factory=list)
    stdout_log: List[str] = field(default_factory=list)
    stderr_log: List[str] = field(default_factory=list)
    message: str = ""
    elapsed_sec: float = 0.0
    scan_count: int = 0
    exported_count: int = 0


def _emit(lines: List[str], line: str, callback: Optional[Callable[[str], None]] = None) -> None:
    lines.append(line)
    if callback:
        callback(line)
    else:
        print(line, flush=True)


def _load_pybind_converter(dll_dir: str | None = None):
    dll_path = Path(dll_dir or os.getenv("FLS_CONVERTER_DLL_DIR", DEFAULT_DLL_DIR))
    if not dll_path.exists():
        raise FileNotFoundError(
            f"FlsConverter DLL 目录不存在: {dll_path}. "
            "请设置环境变量 FLS_CONVERTER_DLL_DIR 指向 pybind 包目录。"
        )
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path))
    os.environ["PATH"] = str(dll_path) + os.pathsep + os.environ.get("PATH", "")
    if str(dll_path) not in sys.path:
        sys.path.insert(0, str(dll_path))
    return importlib.import_module("FlsConverter")


def _load_scan_meta(json_path: Path, ply_path: Path, point_count: int) -> ScanMeta:
    meta = ScanMeta()
    meta.ply_path = str(ply_path.resolve())
    meta.json_path = str(json_path.resolve()) if json_path.exists() else ""
    meta.point_count = int(point_count or 0)
    if not json_path.exists():
        return meta
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        meta.scan_name = data.get("scanName", data.get("scan_name", meta.scan_name))
        meta.scanner_type = data.get("scannerType", "")
        meta.scanner_serial = data.get("scannerSerial", "")
        meta.num_cols = data.get("numCols", 0)
        meta.num_rows = data.get("numRows", 0)
        meta.total_num_rows = data.get("totalNumRows", 0)
        meta.row_start_angle_rad = data.get("rowStartAngle_rad", 0.0)
        meta.row_end_angle_rad = data.get("rowEndAngle_rad", 0.0)
        meta.col_end_angle_rad = data.get("colEndAngle_rad", 0.0)
        meta.scanner_range_m = data.get("scannerRange_m", 0.0)
        meta.dist_offset_m = data.get("distOffset_m", 0.0)
        meta.dist_factor = data.get("distFactor", 0.0)
        meta.scan_time = data.get("scanTime", "")
        meta.scan_axis = data.get("scanAxis", meta.scan_axis)
        meta.scan_angle_rad = data.get("scanAngle_rad", 0.0)
        meta.transform_to_global = data.get("transformToGlobal", data.get("transform_to_global", meta.transform_to_global))
        meta.gps = data.get("gps", {})
        meta.compass = data.get("compass", {})
        meta.sensor_usage = data.get("sensorUsage", {})
        origin = data.get("scan_origin", data.get("scanOrigin"))
        if origin is None and meta.transform_to_global:
            try:
                origin = [
                    float(meta.transform_to_global[0][3]),
                    float(meta.transform_to_global[1][3]),
                    float(meta.transform_to_global[2][3]),
                ]
            except Exception:
                origin = None
        if origin is not None:
            meta.scan_origin = list(origin)
        origins = data.get("scan_origins", data.get("scanOrigins"))
        if origins is not None:
            meta.scan_origins = origins
    except Exception as exc:
        trace("fls.convert.metadata_warning", json=json_path, error=exc)
    return meta


def _meta_from_pybind_scan(scan, meta_dir: Path) -> ScanMeta:
    ply_path = Path(str(getattr(scan, "ply_path", ""))).resolve()
    json_attr = str(getattr(scan, "json_path", "") or "")
    json_path = Path(json_attr).resolve() if json_attr else ply_path.with_suffix(".json")
    meta = _load_scan_meta(json_path, ply_path, int(getattr(scan, "point_count", 0) or 0))
    meta.scan_name = str(getattr(scan, "scan_name", "") or meta.scan_name or ply_path.stem)
    meta.has_color = bool(getattr(scan, "has_color", False))
    meta.has_intensity = bool(getattr(scan, "has_intensity", False))
    meta.has_distance = bool(getattr(scan, "has_distance", False))
    if json_path.exists():
        dest_json = meta_dir / json_path.name
        if json_path.resolve() != dest_json.resolve():
            shutil.copy2(json_path, dest_json)
        meta.json_path = str(dest_json.resolve())
    return meta


def _write_project_index(project_dir: Path, result: ConversionResult) -> None:
    index = {
        "project_name": project_dir.name,
        "created_at": datetime.now().isoformat(),
        "source_fls": result.fls_path,
        "output_dir": result.output_dir,
        "total_scans": len(result.scans),
        "total_points": sum(s.point_count for s in result.scans),
        "elapsed_sec": round(result.elapsed_sec, 2),
        "converter": "pybind:FlsConverter",
        "scans": [
            {
                "scan_name": s.scan_name,
                "ply_path": os.path.relpath(s.ply_path, project_dir),
                "json_path": os.path.relpath(s.json_path, project_dir) if s.json_path else "",
                "point_count": s.point_count,
                "scanner_type": s.scanner_type,
                "scanner_serial": s.scanner_serial,
                "scan_time": s.scan_time,
                "has_gps": s.gps.get("hasGps", False),
                "gps_lat": s.gps.get("latitude", 0.0),
                "gps_lon": s.gps.get("longitude", 0.0),
                "has_color": s.has_color,
                "has_intensity": s.has_intensity,
                "has_distance": s.has_distance,
                "transform_to_global": s.transform_to_global,
            }
            for s in result.scans
        ],
    }
    index_path = project_dir / "project.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    trace("fls.convert.index_written", path=index_path)


def convert_fls_to_ply(
    *,
    fls_folder: str,
    output_dir: str,
    project_name: Optional[str] = None,
    on_stdout: Optional[Callable[[str], None]] = None,
    merge_scans: bool = True,
    use_gps: bool = True,
    dll_dir: Optional[str] = None,
) -> ConversionResult:
    """
    使用 pybind11 封装的 FlsConverter 执行 FLS -> PLY/JSON 转换。

    参数：
        fls_folder: FLS 文件夹或文件的路径
        output_dir: 输出根目录
        project_name: 项目名称（可选，默认为 fls_folder 的名称）
        on_stdout: 标准输出回调函数（每行日志回调）
        merge_scans: 是否合并所有扫描为单个 PLY/DIST 文件
        use_gps: 是否使用 GPS 信息对齐扫描
        dll_dir: 指向包含 FlsConverter.pyd 及依赖 DLL 的目录（可选，默认使用 DEFAULT_DLL_DIR）

    返回：
        ConversionResult 对象
    """
    result = ConversionResult(fls_path=str(Path(fls_folder).resolve()))
    fls_path = Path(fls_folder).resolve()
    if not fls_path.exists():
        result.message = f"FLS 路径不存在: {fls_path}"
        return result

    if project_name is None:
        project_name = fls_path.stem if fls_path.is_file() else fls_path.name

    project_dir = Path(output_dir).resolve() / project_name
    ply_dir = project_dir / "pointclouds"
    meta_dir = project_dir / "metadata"
    ply_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    result.output_dir = str(project_dir)

    trace("fls.convert.begin", source=fls_path, output=ply_dir, merge_scans=merge_scans, use_gps=use_gps)
    _emit(result.stdout_log, f"[PCFD] fls.convert.begin source={fls_path} output={ply_dir}", on_stdout)
    start = time.perf_counter()

    try:
        fls = _load_pybind_converter(dll_dir)
        py_result = fls.convert(
            fls_folder=str(fls_path),
            output_dir=str(ply_dir),
            merge_scans=merge_scans,
            use_gps=use_gps,
        )
    except Exception as exc:
        result.elapsed_sec = time.perf_counter() - start
        result.success = False
        result.message = f"FlsConverter 调用失败: {exc}"
        trace("fls.convert.failed", error=exc, seconds=f"{result.elapsed_sec:.2f}")
        return result

    result.elapsed_sec = float(getattr(py_result, "elapsed_sec", time.perf_counter() - start) or 0.0)
    result.success = bool(getattr(py_result, "success", False))
    result.scan_count = int(getattr(py_result, "scan_count", 0) or 0)
    result.exported_count = int(getattr(py_result, "exported_count", 0) or 0)
    result.scans = [_meta_from_pybind_scan(scan, meta_dir) for scan in list(getattr(py_result, "scans", []) or [])]

    # 补偿可能缺失的计数
    if not result.exported_count:
        result.exported_count = len(result.scans)
    if not result.scan_count:
        result.scan_count = len(result.scans)

    for scan in result.scans:
        line = f"[PCFD] fls.convert.scan name={scan.scan_name} points={scan.point_count} ply={scan.ply_path}"
        _emit(result.stdout_log, line, on_stdout)
        trace("fls.convert.scan", name=scan.scan_name, points=scan.point_count, ply=scan.ply_path)

    if result.success and result.exported_count > 0:
        result.message = (
            f"成功导出 {result.exported_count}/{result.scan_count} 个扫描，"
            f"总点数 {sum(s.point_count for s in result.scans):,}，耗时 {result.elapsed_sec:.1f} 秒"
        )
        _write_project_index(project_dir, result)
    elif not result.message:
        result.success = False
        result.message = "未生成 PLY，请检查 FLS 路径或 FlsConverter 输出"

    trace(
        "fls.convert.done",
        success=result.success,
        scan_count=result.scan_count,
        exported_count=result.exported_count,
        seconds=f"{result.elapsed_sec:.2f}",
    )
    _emit(result.stdout_log, f"[PCFD] fls.convert.done success={result.success} exported={result.exported_count}", on_stdout)
    return result