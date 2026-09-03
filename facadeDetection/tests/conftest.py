"""pytest 公共配置：offscreen 环境、包路径、QApplication 单例（由 pytest-qt 的
qapp fixture 提供）。测试不得创建真实项目目录或数据库文件——涉及 Storage 的
被测代码一律在对应测试模块内打桩。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# 允许从 facadeDetection/ 以外目录运行 pytest 时也能解析包内裸导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
