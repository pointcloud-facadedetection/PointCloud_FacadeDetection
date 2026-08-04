import os
import sys
import json
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import open3d as o3d
HAS_OPEN3D = True

# ============================================================================
# 配置常量（保留全局引用）
# ============================================================================
EXE_PATH = r"D:\\ElevationDetect\\FlsRead\\x64\\Debug\\FlSRead.exe"
DEFAULT_OUTPUT = r"D:\\ElevationDetect\\faro\\output"

# ============================================================================
# 数据结构
# ============================================================================
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
        [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    ])
    gps: Dict = field(default_factory=dict)
    compass: Dict = field(default_factory=dict)
    sensor_usage: Dict = field(default_factory=dict)
    ply_path: str = ""
    json_path: str = ""
    point_count: int = 0

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

# ============================================================================
# 内部私有工具函数（原类方法剥离）
# ============================================================================
def _load_scan_meta(json_path: Path, ply_path: Path, point_count: int) -> ScanMeta:
    meta = ScanMeta()
    meta.ply_path = str(ply_path.resolve())
    meta.json_path = str(json_path.resolve()) if json_path.exists() else ""
    meta.point_count = point_count

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta.scan_name = data.get("scanName", "")
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
            meta.scan_axis = data.get("scanAxis", [0.0, 0.0, 0.0])
            meta.scan_angle_rad = data.get("scanAngle_rad", 0.0)
            meta.transform_to_global = data.get("transformToGlobal", meta.transform_to_global)
            meta.gps = data.get("gps", {})
            meta.compass = data.get("compass", {})
            meta.sensor_usage = data.get("sensorUsage", {})
        except Exception as e:
            print(f"[WARN] 解析 JSON 失败 {json_path}: {e}")
    return meta

def _write_project_index(project_dir: Path, result: ConversionResult):
    index = {
        "project_name": project_dir.name,
        "created_at": datetime.now().isoformat(),
        "source_fls": result.fls_path,
        "output_dir": result.output_dir,
        "total_scans": len(result.scans),
        "total_points": sum(s.point_count for s in result.scans),
        "elapsed_sec": round(result.elapsed_sec, 2),
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
                "transform_to_global": s.transform_to_global,
            }
            for s in result.scans
        ],
    }
    index_path = project_dir / "project.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 项目索引已写入: {index_path}")

# ============================================================================
# 对外暴露的转换主函数（关键字传参）
# ============================================================================
def convert_fls_to_ply(
    *,
    fls_folder: str,
    output_dir: str,
    project_name: Optional[str] = None,
    exe_path: str = EXE_PATH,
    timeout_sec: int = 3600,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
) -> ConversionResult:
    """
    执行 FLS -> PLY/JSON 转换的简化工具函数。
    保留原有的 subprocess 调用逻辑与输出解析，不做业务耦合。
    """
    exe = Path(exe_path)
    if not exe.exists():
        raise FileNotFoundError(f"FlSRead.exe 未找到: {exe}")

    result = ConversionResult()
    result.fls_path = str(Path(fls_folder).resolve())

    fls_path = Path(fls_folder)
    if not fls_path.exists():
        result.message = f"FLS 路径不存在: {fls_folder}"
        return result

    if project_name is None:
        project_name = fls_path.stem if fls_path.is_file() else fls_path.name

    # 项目目录结构
    project_dir = Path(output_dir) / project_name
    ply_dir = project_dir / "pointclouds"
    meta_dir = project_dir / "metadata"
    ply_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    result.output_dir = str(project_dir.resolve())
    cmd = [str(exe), str(fls_path.resolve()), str(ply_dir.resolve())]

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    scan_metas: List[ScanMeta] = []

    print(f"[INFO] 启动转换...")
    print(f"[INFO] EXE: {exe}")
    print(f"[INFO] FLS: {fls_path}")
    print(f"[INFO] OUT: {ply_dir}")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(exe.parent),
        )

        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip("\n")
                stdout_lines.append(line)
                if on_stdout:
                    on_stdout(line)
                else:
                    print(f"[STDOUT] {line}")

        if proc.stderr:
            for line in proc.stderr:
                line = line.rstrip("\n")
                stderr_lines.append(line)
                if on_stderr:
                    on_stderr(line)
                else:
                    print(f"[STDERR] {line}")
        proc.wait(timeout=timeout_sec)
        returncode = proc.returncode

    except subprocess.TimeoutExpired:
        proc.kill()
        result.message = f"转换超时（>{timeout_sec}秒）"
        result.stdout_log = stdout_lines
        result.stderr_log = stderr_lines
        return result
    except Exception as e:
        result.message = f"子进程异常: {e}"
        result.stdout_log = stdout_lines
        result.stderr_log = stderr_lines
        return result

    elapsed = time.time() - start_time
    result.elapsed_sec = elapsed
    result.stdout_log = stdout_lines
    result.stderr_log = stderr_lines

    # 解析控制台输出提取扫描信息
    scan_count = 0
    exported_count = 0
    scan_name_pattern = re.compile(r"\[OK\] Exported: (.+?) \((\d+) points")
    found_pattern = re.compile(r"\[INFO\] Found (\d+) scan\(s\)")

    for line in stdout_lines:
        m = found_pattern.search(line)
        if m:
            scan_count = int(m.group(1))
        m = scan_name_pattern.search(line)
        if m:
            exported_count += 1
            ply_file = Path(m.group(1))
            pts = int(m.group(2))
            json_file = ply_file.with_suffix(".json")
            meta = _load_scan_meta(json_file, ply_file, pts)
            scan_metas.append(meta)
            if json_file.exists():
                dest_json = meta_dir / json_file.name
                shutil.copy2(str(json_file), str(dest_json))
    result.scans = scan_metas

    if returncode != 0:
        result.success = False
        result.message = f"FlSRead.exe 返回非零退出码: {returncode} (0x{returncode:08X})"
        if returncode == 3221225781 or returncode == -1073741515:
            result.message += (
                "\n[HINT] STATUS_DLL_NOT_FOUND: 缺少运行时依赖\n"
                "[HINT] 请确保已安装以下组件：\n"
                "  1. SCENE Redistributable Package（位于 SDK bin/ 目录）\n"
                "  2. vcredist_x64_2022（位于 SDK bin/ 目录）"
            )
        elif returncode == 3221225477 or returncode == -1073741819:
            result.message += "\n[HINT] STATUS_ACCESS_VIOLATION: 访问冲突，可能是 FARO API 调用异常"
    elif exported_count > 0:
        result.success = True
        result.message = (
            f"成功导出 {exported_count}/{scan_count} 个扫描，"
            f"总点数 {sum(s.point_count for s in scan_metas):,}，"
            f"耗时 {elapsed:.1f} 秒"
        )
    else:
        result.success = False
        result.message = "未找到扫描数据，请检查 FLS 路径是否正确"

    if result.success:
        _write_project_index(project_dir, result)
    return result