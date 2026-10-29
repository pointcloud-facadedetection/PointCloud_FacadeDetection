from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def run(project_id=None):
    from main import main

    state = {'stage': 0, 'ticks': 0}

    def window():
        app = QApplication.instance()
        return next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == 'MainWindow'), None)

    def mark(label):
        win = window()
        probe = getattr(win, '_viewport_probe', None) if win else None
        if probe is not None:
            probe.mark(label)
        print(f'[scenario] {label}', flush=True)

    def open_cached_project():
        win = window()
        if win is None:
            state['stage'] += 1
            if state['stage'] > 20:
                print('[scenario] MainWindow not found', flush=True)
                QApplication.instance().quit()
            else:
                QTimer.singleShot(200, open_cached_project)
            return
        projects = win.project_overview_service.list_projects()
        project = next((p for p in projects
                        if project_id is None or p.project_id == project_id), None)
        if project is None:
            print('[scenario] cached project not found', flush=True)
            QApplication.instance().quit()
            return
        mark(f'open.cached.{project.project_id}')
        probe = getattr(win, '_viewport_probe', None)
        if probe is not None:
            probe.attach_window(win)
        win._activate_project(project)
        QTimer.singleShot(500, enter_review_when_ready)

    def enter_review_when_ready():
        win = window()
        state['ticks'] += 1
        if win is None:
            QTimer.singleShot(500, enter_review_when_ready)
            return
        ready = (win.current_project is not None
                 and win.viewport.has_clouds()
                 and bool(win.project_operation_service.last_facade_results))
        if not ready:
            if state['ticks'] > 120:
                print('[scenario] project ready timeout', flush=True)
                QApplication.instance().quit()
            else:
                QTimer.singleShot(500, enter_review_when_ready)
            return
        mark('project.ready')
        win.set_current_page(2)
        QTimer.singleShot(1500, select_facade)

    def measured(label, callback):
        if os.getenv('FACD_PROFILE') != '1':
            callback()
            return
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            callback()
        finally:
            profiler.disable()
            path = Path(os.getenv(
                'FACD_VIEWPORT_PROBE_PATH', 'data/diagnostics/app.jsonl'))
            output = path.with_name(f'{path.stem}.{label}.prof')
            profiler.dump_stats(str(output))
            print(f'[scenario] profile={output}', flush=True)

    def select_facade():
        win = window()
        items = [win.list_facades.item(i)
                 for i in range(win.list_facades.count())]
        item = next((item for item in items
                     if (win._facade_from_item(item) or {}).get('quality_report')), None)
        if item is None and items:
            item = items[0]
        if item is None:
            print('[scenario] no facade available', flush=True)
            QApplication.instance().quit()
            return
        win.list_facades.setCurrentItem(item)
        mark('facade.select.start')
        measured('facade_select', lambda: win._on_facade_item_clicked(item))
        mark('facade.select.return')
        QTimer.singleShot(2500, toggle_measure)

    def toggle_measure():
        win = window()
        mark('measure.activate.start')
        measured('measure_activate', lambda: win.btn_review_measure.click())
        mark('measure.activate.return')
        QTimer.singleShot(1200, deactivate_measure)

    def deactivate_measure():
        win = window()
        win.btn_review_measure.click()
        mark('measure.deactivate')
        QTimer.singleShot(800, toggle_ruler)

    def toggle_ruler():
        win = window()
        mark('ruler.activate.start')
        measured('ruler_activate', lambda: win.btn_review_ruler.click())
        mark('ruler.activate.return')
        QTimer.singleShot(2500, finish)

    def finish():
        win = window()
        if win is not None:
            win.btn_review_ruler.click()
        mark('scenario.finish')
        if win is not None:
            win.close()
        QApplication.instance().quit()

    original_exec = QApplication.exec

    def exec_with_watchdog():
        QTimer.singleShot(500, open_cached_project)
        QTimer.singleShot(90000, finish)
        return original_exec()

    QApplication.exec = staticmethod(exec_with_watchdog)
    return main()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default=os.getenv('FACD_PROBE_PROJECT'))
    args = parser.parse_args()
    raise SystemExit(run(args.project))
