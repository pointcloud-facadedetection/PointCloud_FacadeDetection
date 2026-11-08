from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def windows_memory_regions():
    if os.name != 'nt':
        return {}
    import ctypes

    class MemoryBasicInformation(ctypes.Structure):
        _fields_ = [
            ('BaseAddress', ctypes.c_void_p),
            ('AllocationBase', ctypes.c_void_p),
            ('AllocationProtect', ctypes.c_ulong),
            ('RegionSize', ctypes.c_size_t),
            ('State', ctypes.c_ulong),
            ('Protect', ctypes.c_ulong),
            ('Type', ctypes.c_ulong),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.VirtualQueryEx.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(MemoryBasicInformation), ctypes.c_size_t]
    kernel32.VirtualQueryEx.restype = ctypes.c_size_t
    process = kernel32.OpenProcess(0x0400, 0, os.getpid())
    labels = {0x1000000: 'image', 0x40000: 'mapped', 0x20000: 'private'}
    totals = {label: 0 for label in labels.values()}
    largest = []
    address = 0x10000
    info = MemoryBasicInformation()
    while address < (1 << 47):
        result = kernel32.VirtualQueryEx(
            process, address, ctypes.byref(info), ctypes.sizeof(info))
        if not result or not info.RegionSize:
            break
        if info.State == 0x1000:
            label = labels.get(info.Type, 'other')
            totals[label] = totals.get(label, 0) + int(info.RegionSize)
            largest.append((int(info.RegionSize), label, int(info.BaseAddress)))
        address += int(info.RegionSize)
    kernel32.CloseHandle(process)
    return {
        'committed_mib': {key: round(value / 1024**2, 2)
                          for key, value in totals.items()},
        'largest_regions': [
            {'mib': round(size / 1024**2, 2), 'type': label,
             'base': hex(base)}
            for size, label, base in sorted(largest, reverse=True)[:12]],
    }


def array_nbytes(value, seen):
    if isinstance(value, np.ndarray):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(array_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(array_nbytes(item, seen) for item in value)
    if hasattr(value, '__dict__'):
        return array_nbytes(vars(value), seen)
    return 0


def inventory(window):
    seen = set()
    controller = window.facade_quality_controller
    operation = window.project_operation_service
    categories = {
        'source_assets': window.pointcloud_service.source_assets,
        'datasets': window.pointcloud_service.datasets,
        'viewport_point_data': window.viewport.point_data,
        'quality_result_cache': controller.quality_result_cache,
        'quality_reports': controller.quality_reports,
        'all_facade_results': operation.all_facade_results,
        'review_png_cache': getattr(window, '_review_heat_png_cache', {}),
    }
    result = {
        name: round(array_nbytes(value, seen) / 1024**2, 2)
        for name, value in categories.items()
    }
    result['process_rss'] = round(psutil.Process().memory_info().rss / 1024**2, 2)
    result['children'] = [child.pid for child in psutil.Process().children(recursive=True)]
    return result


def run(project_id):
    from main import main
    state = {'ticks': 0, 'before': None}

    def window():
        return next((widget for widget in QApplication.allWidgets()
                     if widget.__class__.__name__ == 'MainWindow'), None)

    def finish(code=0):
        win = window()
        if win is not None:
            win.close()
        QApplication.instance().exit(code)

    def after_quality():
        state['after'] = inventory(window())
        print(json.dumps({'before': state['before'], 'after': state['after'],
                          'memory_regions': windows_memory_regions()},
                         ensure_ascii=False), flush=True)
        finish(0 if not state['after']['children'] else 2)

    def evaluate():
        win = window()
        facades = win.project_operation_service.last_facade_results or []
        facade = next((item for item in facades
                       if isinstance(item.get('quality_report'), dict)), None)
        if facade is None:
            print('no facade available', flush=True)
            finish(4)
            return
        facade = dict(facade)
        facade['quality_status'] = 'pending'
        facade['__suppress_quality_dialog'] = True
        state['before'] = inventory(win)
        win.facade_quality_controller.quality_completed.connect(
            lambda: QTimer.singleShot(10000, after_quality))
        win.facade_quality_controller.quality_failed.connect(
            lambda: finish(5))
        win.facade_quality_controller.evaluate_facade(facade)

    def ready():
        win = window()
        state['ticks'] += 1
        if win is None:
            QTimer.singleShot(500, ready)
            return
        if (win.current_project is not None and win.viewport.has_clouds()
                and win.project_operation_service.last_facade_results):
            QTimer.singleShot(1500, evaluate)
        elif state['ticks'] > 180:
            finish(3)
        else:
            QTimer.singleShot(500, ready)

    def start():
        win = window()
        if win is None:
            QTimer.singleShot(200, start)
            return
        project = win.project_overview_service.get_project(project_id)
        if project is None:
            finish(4)
            return
        win._activate_project(project)
        ready()

    original_exec = QApplication.exec

    def exec_with_watchdog():
        QTimer.singleShot(500, start)
        QTimer.singleShot(240000, lambda: finish(3))
        return original_exec()

    QApplication.exec = staticmethod(exec_with_watchdog)
    return main()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=os.getenv('FACD_PROBE_PROJECT'))
    args = parser.parse_args()
    raise SystemExit(run(args.project))
