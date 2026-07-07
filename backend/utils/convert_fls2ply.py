import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
from typing import List, Dict, Optional
import shutil

class CloudCompareFLSExporter:
    def __init__(self, cc_exe: Optional[str] = None):
        if cc_exe is None:
            # 默认路径，请根据实际安装位置修改
            self.cc_exe = r"D:\CloudCompare\CloudCompare.exe"
        else:
            self.cc_exe = cc_exe
        self._verify_installation()

    def _verify_installation(self):
        if not Path(self.cc_exe).exists():
            raise FileNotFoundError(
                f"找不到 CloudCompare: {self.cc_exe}\n"
                "请安装 CloudCompare 或指定正确路径"
            )

    def find_all_fls_files(self, root_dir: str) -> List[Dict]:
        """
        递归查找 root_dir 下所有 .fls 文件，返回列表，每个元素包含：
        - 'full_path': 完整路径
        - 'relative_path': 相对于 root_dir 的路径（不含文件名，用于构建输出目录）
        - 'filename': 文件名
        """
        root = Path(root_dir).resolve()
        fls_files = []
        for fls_path in root.rglob("*.fls"):
            if fls_path.is_file():
                rel_dir = fls_path.parent.relative_to(root)  # 相对目录
                fls_files.append({
                    'full_path': fls_path,
                    'relative_dir': rel_dir,
                    'filename': fls_path.name,
                })
        return fls_files

    def export_single(self, fls_path: Path, output_file: Path, export_format: str = "PLY") -> dict:
        """
        导出单个 FLS 文件到指定输出路径
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cc_exe,
            "-SILENT",
            "-NO_TIMESTAMP",
            "-O", str(fls_path),
            "-C_EXPORT_FMT", export_format,
            "-SAVE_CLOUDS", "FILE", str(output_file),
            "-CLEAR"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 增加到10分钟，大文件可能需要更久
                encoding='utf-8',
                errors='ignore'
            )
            success = output_file.exists() and output_file.stat().st_size > 0
            return {
                'input': str(fls_path),
                'output': str(output_file) if success else None,
                'success': success,
                'stdout': result.stdout[-500:] if result.stdout else "",
                'stderr': result.stderr[-500:] if result.stderr else "",
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'input': str(fls_path), 'success': False, 'error': '超时 (>600s)'}
        except Exception as e:
            return {'input': str(fls_path), 'success': False, 'error': str(e)}

    def batch_export_recursive(self, root_dir: str, output_root: str,
                               export_format: str = "PLY",
                               max_workers: int = 1) -> List[dict]:
        """
        递归转换 root_dir 下所有 .fls 文件，输出到 output_root 并保持目录结构
        """
        fls_items = self.find_all_fls_files(root_dir)
        if not fls_items:
            print(f"在 {root_dir} 下未找到任何 .fls 文件")
            return []

        print(f"找到 {len(fls_items)} 个 FLS 文件")
        results = []

        for idx, item in enumerate(fls_items, 1):
            fls_path = item['full_path']
            rel_dir = item['relative_dir']
            output_dir = Path(output_root) / rel_dir
            output_file = output_dir / f"{fls_path.stem}.{export_format.lower()}"
            print(f"\n[{idx}/{len(fls_items)}] {fls_path.relative_to(root_dir)}")
            result = self.export_single(fls_path, output_file, export_format)
            results.append(result)
            status = "✅" if result['success'] else "❌"
            print(f"    {status} {result.get('output', result.get('error', 'Unknown'))}")

        # 保存报告
        report_path = Path(output_root) / "cc_export_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n报告已保存: {report_path}")

        success_count = sum(1 for r in results if r['success'])
        print(f"\n完成: {success_count}/{len(results)} 成功")
        return results


# ============ 使用示例 ============
if __name__ == "__main__":
    # 输入根目录（包含多个可能含有 .fls 的子文件夹）
    INPUT_ROOT = r"D:\ElevationDetect\faro"
    # 输出根目录（将保持子文件夹结构）
    OUTPUT_ROOT = r"D:\P3D\data\ply"

    exporter = CloudCompareFLSExporter()
    # 递归转换所有 .fls，输出 PLY 格式
    results = exporter.batch_export_recursive(INPUT_ROOT, OUTPUT_ROOT, export_format="PLY")