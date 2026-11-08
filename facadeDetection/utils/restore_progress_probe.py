from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def run(project_id):
    from main import main

    state = {'last': None, 'values': [], 'finished': False, 'verified': False}

    def window():
        return next((widget for widget in QApplication.allWidgets()
                     if widget.__class__.__name__ == 'MainWindow'), None)

    def sample():
        win = window()
        if win is None:
            QTimer.singleShot(100, sample)
            return
        dialog = getattr(win, '_loading_dialog', None)
        if dialog is not None:
            bar = dialog._bar
            value = (bar.value(), bar.maximum())
            if value != state['last']:
                state['last'] = value
                state['values'].append(value)
                print(f'[restore-progress] value={value[0]} maximum={value[1]}', flush=True)
        if not state['verified']:
            QTimer.singleShot(100, sample)

    def verify():
        win = window()
        dialog = getattr(win, '_loading_dialog', None) if win is not None else None
        if dialog is not None and (
                not dialog._finished or dialog._bar.value() != 100):
            QTimer.singleShot(100, verify)
            return
        state['verified'] = True
        values = state['values']
        assert win is not None and win.viewport.has_clouds(), '站点视口尚未显示点云'
        assert values, '没有读取到进度条状态'
        assert all(percent >= 0 and maximum == 100
                   for percent, maximum in values), values
        assert [percent for percent, _ in values] == sorted(
            percent for percent, _ in values), values
        assert any(percent > 75 for percent, _ in values), values
        assert values[-1] == (100, 100), values
        print(f'[restore-progress] PASS {values}', flush=True)
        win.close()
        QApplication.instance().quit()

    def finished(success, _message):
        state['finished'] = True
        if success:
            QTimer.singleShot(1000, verify)
        else:
            QApplication.instance().exit(2)

    def open_project():
        win = window()
        if win is None:
            QTimer.singleShot(200, open_project)
            return
        project = win.project_overview_service.get_project(project_id)
        if project is None:
            available = [p.project_id for p in
                         win.project_overview_service.list_projects()]
            print(f'[restore-progress] project not found: {project_id}; '
                  f'available={available}', flush=True)
            win.close()
            QApplication.instance().exit(4)
            return
        win.lifecycle_controller.load_finished.connect(finished)
        win._open_project_card(project_id)
        sample()

    def fail():
        print(f'[restore-progress] timeout values={state["values"]}', flush=True)
        for widget in QApplication.topLevelWidgets():
            widget.close()
        QApplication.instance().exit(3)

    original_exec = QApplication.exec

    def exec_with_watchdog():
        QTimer.singleShot(500, open_project)
        QTimer.singleShot(180000, fail)
        return original_exec()

    QApplication.exec = staticmethod(exec_with_watchdog)
    return main()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        default=os.getenv('FACD_PROBE_PROJECT',
                          '5306bb8f-fd32-4431-a71b-97a4963063e4'))
    args = parser.parse_args()
    run(args.project)
