"""pytest 公共配置：offscreen 环境、包路径、QApplication 单例（由 pytest-qt 的
qapp fixture 提供）、slow 用例开关。测试不得创建真实项目目录或数据库文件——涉及
Storage 的被测代码一律在对应测试模块内打桩。目录约定见同目录 README.md。
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# 允许从 facadeDetection/ 以外目录运行 pytest 时也能解析包内裸导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true',
                     help='同时运行标记为 slow 的大数据量用例（默认跳过）')


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: 为验证优化放大数据规模、单个要跑十几秒以上的用例；默认跳过，加 --slow 运行')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--slow'):
        return
    skip_slow = pytest.mark.skip(reason='大数据量用例，加 --slow 运行')
    for item in items:
        if item.get_closest_marker('slow') is not None:
            item.add_marker(skip_slow)
