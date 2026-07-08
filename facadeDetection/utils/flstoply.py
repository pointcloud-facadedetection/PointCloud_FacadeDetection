#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
借助 CloudCompare 将 FARO FLS 扫描文件转换为 PLY 点云文件。

常用示例：
    python flstoply.py "faro\\bllygg01.fls" "output\\bllygg01.ply" --overwrite
    python flstoply.py "faro" "output_folder" --overwrite
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# CloudCompare 在 Windows 上的常见安装位置。
# 如果安装在其他位置，可以用 --cloudcompare 参数或 CLOUDCOMPARE_EXE 环境变量指定。
DEFAULT_CLOUDCOMPARE_PATHS = (
    r"C:\Program Files\CloudCompare\CloudCompare.exe",
    r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
)


@dataclass
class ConvertJob:
    """一个转换任务：包含一个输入扫描和一个输出 PLY 路径。"""

    scan: Path
    output: Path


@dataclass
class ConvertResult:
    """单个转换任务的结果信息，用于最后汇总输出。"""

    scan: Path
    output: Path
    status: str
    vertex_count: int | None = None
    message: str = ""


def is_expanded_fls_dir(path: Path) -> bool:
    """判断路径是否为 FARO 展开的 .fls 扫描包文件夹。"""

    return path.is_dir() and path.suffix.lower() == ".fls"


def is_inside_fls_package(path: Path, root: Path) -> bool:
    """避免把 .fls 扫描包内部的子文件或子目录误认为独立扫描。"""

    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        rel_parts = path.parts[:-1]

    current = root
    for part in rel_parts:
        current = current / part
        if current.is_dir() and current.suffix.lower() == ".fls":
            return True
    return False


def skip_plain_duplicate_dir(path: Path) -> bool:
    """如果同名 .fls 扫描包已存在，则跳过普通同名目录，避免重复转换。"""

    if path.suffix.lower() == ".fls":
        return False
    return path.with_name(path.name + ".fls").is_dir()


def scan_name(scan_path: Path) -> str:
    """根据 .fls 文件或展开目录生成稳定的输出文件名。"""

    return scan_path.stem if scan_path.suffix.lower() == ".fls" else scan_path.name


def discover_fls_inputs(input_path: Path) -> list[Path]:
    """从单个文件、单个 .fls 文件夹或批量目录中查找可转换的 FLS 输入。"""

    if input_path.is_file() or is_expanded_fls_dir(input_path):
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    scans: list[Path] = []

    for path in sorted(input_path.rglob("*")):
        if not is_expanded_fls_dir(path):
            continue
        if is_inside_fls_package(path, input_path):
            continue
        if skip_plain_duplicate_dir(path):
            continue
        scans.append(path)

    for path in sorted(input_path.rglob("*.fls")):
        if not path.is_file():
            continue
        if path.stat().st_size <= 0:
            continue
        if is_inside_fls_package(path, input_path):
            continue
        scans.append(path)

    return scans


def resolve_cloudcompare(explicit_path: str | None) -> Path:
    """从命令行参数、环境变量、系统 PATH 或默认位置查找 CloudCompare.exe。"""

    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("CLOUDCOMPARE_EXE")
    if env_path:
        candidates.append(env_path)

    path_match = shutil.which("CloudCompare.exe") or shutil.which("CloudCompare")
    if path_match:
        candidates.append(path_match)

    candidates.extend(DEFAULT_CLOUDCOMPARE_PATHS)

    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path

    raise FileNotFoundError(
        "未找到 CloudCompare.exe。请使用 --cloudcompare 指定 CloudCompare 路径。"
    )


def resolve_jobs(input_path: Path, output_path: Path) -> list[ConvertJob]:
    """将发现的 FLS 输入映射为对应的 PLY 输出任务。"""

    scans = discover_fls_inputs(input_path)
    if not scans:
        return []

    single_input = input_path.is_file() or is_expanded_fls_dir(input_path)
    if single_input and output_path.suffix.lower() == ".ply":
        return [ConvertJob(scans[0], output_path)]
    if not single_input and output_path.suffix.lower() == ".ply":
        raise ValueError("批量转换时输出路径必须是文件夹，不能是单个 .ply 文件")

    jobs: list[ConvertJob] = []
    used_names: set[str] = set()

    for scan in scans:
        name = f"{scan_name(scan)}.ply"
        lower = name.lower()
        if lower in used_names:
            stem = Path(name).stem
            suffix = Path(name).suffix
            index = 2
            while f"{stem}_{index}{suffix}".lower() in used_names:
                index += 1
            name = f"{stem}_{index}{suffix}"
            lower = name.lower()
        used_names.add(lower)
        jobs.append(ConvertJob(scan, output_path / name))

    return jobs


def read_ply_vertex_count(ply_path: Path) -> int:
    """读取 PLY 文件头中的点数，用于判断转换结果是否有效。"""

    with ply_path.open("rb") as handle:
        header_lines: list[str] = []
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("输出文件不是有效的 PLY 文件")

    for line in header_lines:
        match = re.match(r"^element\s+vertex\s+(\d+)\s*$", line)
        if match:
            return int(match.group(1))

    raise ValueError("PLY 文件头中没有 element vertex 点数信息")


def quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def build_command(cloudcompare: Path, scan: Path, output: Path) -> list[str]:
    """构造 CloudCompare 命令：打开 FLS 并导出为 PLY。"""

    return [
        str(cloudcompare),
        "-SILENT",
        "-NO_TIMESTAMP",
        "-AUTO_SAVE",
        "OFF",
        "-O",
        str(scan),
        "-C_EXPORT_FMT",
        "PLY",
        "-SAVE_CLOUDS",
        "FILE",
        str(output),
    ]


def summarize_cloudcompare_log(text: str) -> list[str]:
    """只保留对用户有用的 CloudCompare 日志行。"""

    patterns = (
        "Opening file",
        "Successfully connected",
        "contains",
        "Scan size",
        "loaded successfully",
        "Found one cloud",
        "Output export format",
        "saved successfully",
        "Error",
        "Failed",
        "Process",
    )
    lines = []
    for line in text.splitlines():
        if any(pattern in line for pattern in patterns):
            lines.append(line)
    return lines


def convert_one(
    job: ConvertJob,
    cloudcompare: Path,
    *,
    overwrite: bool,
    keep_empty: bool,
    timeout: int,
    dry_run: bool,
) -> ConvertResult:
    """执行一个转换任务，并清理无效或空点云输出。"""

    # 输出文件已存在时，先确认它是可读取的 PLY，再跳过转换。
    # 这样可以避免把之前损坏的文件误认为成功结果。
    if job.output.exists() and not overwrite:
        try:
            vertex_count = read_ply_vertex_count(job.output)
        except Exception as exc:
            return ConvertResult(job.scan, job.output, "FAILED", message=f"已有输出文件无效: {exc}")
        return ConvertResult(job.scan, job.output, "SKIPPED", vertex_count, "输出文件已存在")

    command = build_command(cloudcompare, job.scan, job.output)

    print(f"\n输入: {job.scan}")
    print(f"输出: {job.output}")
    print(f"命令: {quote_command(command)}")

    if dry_run:
        return ConvertResult(job.scan, job.output, "DRY_RUN", message="仅预览命令，未实际执行")

    job.output.parent.mkdir(parents=True, exist_ok=True)

    # 转换前删除旧输出，避免 CloudCompare 执行失败时误用旧的 PLY 文件。
    if job.output.exists():
        job.output.unlink()

    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if job.output.exists():
            job.output.unlink()
        return ConvertResult(job.scan, job.output, "FAILED", message=f"转换超时，超过 {timeout} 秒")

    log_text = completed.stdout or ""
    for line in summarize_cloudcompare_log(log_text):
        print(line)

    # CloudCompare 可能会失败并留下半截输出文件。
    # 这种文件不能交给后续点云算法使用，因此这里主动删除。
    if completed.returncode != 0:
        if job.output.exists():
            job.output.unlink()
        return ConvertResult(job.scan, job.output, "FAILED", message=f"CloudCompare 退出码 {completed.returncode}")

    if not job.output.exists():
        return ConvertResult(job.scan, job.output, "FAILED", message="CloudCompare 没有生成输出文件")

    try:
        vertex_count = read_ply_vertex_count(job.output)
    except Exception as exc:
        if job.output.exists():
            job.output.unlink()
        return ConvertResult(job.scan, job.output, "FAILED", message=str(exc))

    if vertex_count <= 0 and not keep_empty:
        job.output.unlink()
        return ConvertResult(job.scan, job.output, "FAILED", vertex_count, "PLY 点数为 0，已删除空输出")

    return ConvertResult(job.scan, job.output, "OK", vertex_count, "转换完成")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="借助 CloudCompare 将 FARO FLS 扫描文件转换为 PLY 点云文件。"
    )
    parser.add_argument("input", help="输入 .fls 扫描、展开的 .fls 文件夹，或包含多个扫描的目录。")
    parser.add_argument("output", help="单个扫描的输出 .ply 文件，或批量转换的输出文件夹。")
    parser.add_argument("--cloudcompare", help="CloudCompare.exe 路径。默认会从常见安装位置和系统 PATH 中查找。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 PLY 文件。")
    parser.add_argument("--keep-empty", action="store_true", help="保留点数为 0 的 PLY 文件。")
    parser.add_argument("--timeout", type=int, default=1800, help="单个扫描允许的转换秒数，默认 1800 秒。")
    parser.add_argument("--dry-run", action="store_true", help="只打印转换命令，不实际运行 CloudCompare。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口，供命令行执行和简单测试调用。"""

    args = build_parser().parse_args(argv)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cloudcompare = resolve_cloudcompare(args.cloudcompare)
    jobs = resolve_jobs(input_path, output_path)

    if not jobs:
        print(f"没有找到可转换的 FLS 输入: {input_path}", file=sys.stderr)
        return 1

    print(f"CloudCompare: {cloudcompare}")
    print(f"任务数: {len(jobs)}")

    results: list[ConvertResult] = []
    for job in jobs:
        result = convert_one(
            job,
            cloudcompare,
            overwrite=args.overwrite,
            keep_empty=args.keep_empty,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        results.append(result)
        count = "" if result.vertex_count is None else f" 点数={result.vertex_count}"
        print(f"结果: {result.status}{count} {result.message}")

    print("\n汇总:")
    for result in results:
        count = "-" if result.vertex_count is None else str(result.vertex_count)
        print(f"{result.status:8} 点数={count:>10} 输入={result.scan.name} 输出={result.output}")

    failed = [result for result in results if result.status == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
