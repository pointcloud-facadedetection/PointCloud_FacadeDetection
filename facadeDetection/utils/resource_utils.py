from __future__ import annotations
import sys
from pathlib import Path

def resource_path(*parts: str) -> Path:
    """
    获取资源文件的绝对路径，兼容开发环境和 PyInstaller 打包模式。
    在打包模式下，资源文件会被解压到 sys._MEIPASS 下的 facadeDetection 子目录。
    """
    if getattr(sys, '_MEIPASS', None):
        # PyInstaller 打包模式（onefile / onedir）
        base = Path(sys._MEIPASS) / 'facadeDetection'
    elif getattr(sys, 'frozen', False):
        # 兼容其他冻结模式（一般不会进入）
        base = Path(sys.executable).resolve().parent / 'facadeDetection'
    else:
        # 开发模式：当前文件位于 facadeDetection/utils/resource_utils.py
        base = Path(__file__).resolve().parents[1]  # 返回 facadeDetection 目录
    return base.joinpath(*parts)