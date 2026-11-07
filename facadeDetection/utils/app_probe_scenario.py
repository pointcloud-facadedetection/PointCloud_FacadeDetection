from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget


def run(project_id=None):
    from main import main

    state = {'stage': 0, 'ticks': 0}

    def probe_exception(exc_type, value, traceback):
        sys.__excepthook__(exc_type, value, traceback)
        state['failed'] = True
        finish(3)

    sys.excepthook = probe_exception

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
        def has_heatmap(item):
            facade = win._facade_from_item(item) or {}
            quality = facade.get('quality_report')
            return bool(isinstance(quality, dict) and
                        win._review_heatmap_png_path(facade, quality))

        item = next((item for item in items if has_heatmap(item)), None)
        if item is None:
            item = next((item for item in items
                         if (win._facade_from_item(item) or {}).get('quality_report')), None)
        if item is None and items:
            item = items[0]
        if item is None:
            print('[scenario] no facade available', flush=True)
            QApplication.instance().quit()
            return
        state['cloud_buffers'] = {
            name: id(data['pos']) for name, data in win.viewport.point_data.items()}
        win.list_facades.setCurrentItem(item)
        mark('facade.select.start')
        measured('facade_select', lambda: win._on_facade_item_clicked(item))
        mark('facade.select.return')
        QTimer.singleShot(2500, verify_original_mode)

    def verify_original_mode():
        win = window()
        dimensions = win.viewport._dimension_actors
        labels = [actor for actor in dimensions if hasattr(actor, 'GetTextProperty')]
        expect(len(labels) == 2, 'dimension labels missing')
        expect('W ' in labels[0].GetInput() and 'H ' in labels[1].GetInput(),
               'dimension label values')
        expect('横向' in win.lbl_review_dimensions.text() and
               '纵向' in win.lbl_review_dimensions.text(), 'dimension summary')
        expect(state['cloud_buffers'] == {
            name: id(data['pos']) for name, data in win.viewport.point_data.items()},
            'dimension rendering reloaded point-cloud buffers')
        state['dimension_actors'] = dimensions
        QApplication.processEvents()
        screenshot = Path(os.environ['FACD_DATA_DIR']) / 'diagnostics' / 'facade_dimensions.png'
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        win.viewport.get_widget().grab().save(str(screenshot))
        print(f'[scenario] dimensions.screenshot={screenshot}', flush=True)
        state['heatmap_mode'] = win.review_heatmap_combo.currentData()
        index = win.review_heatmap_combo.findData(win.REVIEW_ORIGINAL_MODE)
        expect(index >= 0, 'original mode is missing')
        win.review_heatmap_combo.setCurrentIndex(index)
        QApplication.processEvents()
        actors = win.viewport._heatmap_quad_actors
        expect(bool(actors), 'original mode has no facade frame')
        expect(all(actor.GetTexture() is None for actor in actors),
               'original mode still has a textured actor')
        expect('仅显示立面外框' in win.lbl_review_heatmap_hint.text(),
               'original mode hint')
        mark('original.frame_only')
        win.review_heatmap_combo.setCurrentIndex(
            win.review_heatmap_combo.findData(state['heatmap_mode']))
        QTimer.singleShot(1000, restore_heatmap)

    def restore_heatmap():
        win = window()
        actors = win.viewport._heatmap_quad_actors
        expect(any(actor.GetTexture() is not None and actor.GetVisibility()
                   for actor in actors), 'heatmap was not restored')
        expect(win.viewport._dimension_actors == state['dimension_actors'],
               'dimension actors were rebuilt on heatmap switch')
        mark('original.heatmap_restored')
        toggle_measure()

    def expect(condition, message):
        if condition:
            return True
        if not state.get('failed'):
            state['failed'] = True
            print(f'[scenario] FAILED: {message}', flush=True)
            finish(2)
        return False

    def click_button(button):
        if os.getenv('FACD_QT_INPUT_PROBE') == '1':
            QTest.mouseClick(button, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,
                             button.rect().center())
        else:
            button.click()

    def report_input(label):
        win = window()
        mouse = QWidget.mouseGrabber()
        keyboard = QWidget.keyboardGrabber()
        modal = QApplication.activeModalWidget()
        print(f'[input] {label} mouse={mouse} keyboard={keyboard} modal={modal} '
              f'focus={QApplication.focusWidget()} '
              f'tool={win.viewport_tools_controller.active_tool} '
              f'result={win.lbl_review_tool_result.text()}', flush=True)

    def click_viewport(label, offset=0):
        win = window()
        widget = win.viewport.get_widget()
        pos = QPoint(widget.width() // 2 + offset, widget.height() // 2)
        geometry = win.viewport_tools_controller._plane_geometry
        if isinstance(geometry, dict):
            origin = geometry['origin']
            u_axis, v_axis = geometry['u_axis'], geometry['v_axis']
            u0, v0, u1, v1 = geometry['uv_bounds']
            center = [origin[i] + (u0 + u1) * 0.5 * u_axis[i]
                      + (v0 + v1) * 0.5 * v_axis[i] for i in range(3)]
            point = [center[i] + u_axis[i] * offset / 100 for i in range(3)]
            screen, valid = win.viewport.project_points([point])
            if valid[0]:
                pos = QPoint(round(screen[0][0]), round(screen[0][1]))
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, pos)
        report_input(label)

    def click_measure_points():
        click_viewport('measure.first')
        click_viewport('measure.second', 45)
        expect(len(window().viewport_tools_controller._measure_points) == 2,
               'two Qt measure points were not received')
        QTimer.singleShot(600, deactivate_measure)

    def click_ruler_points():
        controller = window().viewport_tools_controller
        if controller._ruler_base is None:
            state['ruler_wait'] = state.get('ruler_wait', 0) + 1
            if state['ruler_wait'] > 20:
                expect(False, 'ruler base preparation timeout')
                return
            QTimer.singleShot(500, click_ruler_points)
            return
        click_viewport('ruler.first')
        expect(controller._ruler_start is not None,
               'ruler start was not received inside the facade')
        click_viewport('ruler.direction', 50)
        QTimer.singleShot(1800, click_ruler_again)

    def click_ruler_again():
        controller = window().viewport_tools_controller
        if controller._placement_pending:
            state['placement_wait'] = state.get('placement_wait', 0) + 1
            if state['placement_wait'] > 20:
                expect(False, 'ruler placement timeout')
                return
            QTimer.singleShot(500, click_ruler_again)
            return
        click_viewport('ruler.again.first')
        expect(controller._ruler_start is not None,
               'second ruler start was not received')
        click_viewport('ruler.again.direction', 45)
        QTimer.singleShot(1800, verify_qt_pages)

    def verify_qt_pages():
        win = window()
        report_input('before.page.operation')
        click_button(win.page_buttons['project_operation'])
        QApplication.processEvents()
        if not expect(not win._workspace_in_review,
                      'Qt project-operation button did not respond'):
            return
        report_input('after.page.operation')
        QTimer.singleShot(3200, return_to_review)

    def return_to_review():
        win = window()
        report_input('before.page.review')
        click_button(win.page_buttons['inspection_review'])
        QApplication.processEvents()
        if not expect(win._workspace_in_review,
                      'Qt inspection-review button did not respond'):
            return
        report_input('after.page.review')
        QTimer.singleShot(500, finish)

    def toggle_measure():
        win = window()
        mark('measure.activate.start')
        measured('measure_activate', lambda: click_button(win.btn_review_measure))
        mark('measure.activate.return')
        QApplication.processEvents()
        button = win.btn_review_measure
        expect(button.isChecked() and button.property('toolActive'),
               'measure active state')
        color = button.grab().toImage().pixelColor(6, 6)
        expect(color.blue() > color.red() and color.blue() > 150,
               f'measure active color {color.name()}')
        if os.getenv('FACD_QT_INPUT_PROBE') == '1':
            QTimer.singleShot(400, click_measure_points)
        else:
            QTimer.singleShot(1200, deactivate_measure)

    def deactivate_measure():
        win = window()
        click_button(win.btn_review_measure)
        mark('measure.deactivate')
        expect(not win.btn_review_measure.property('toolActive'),
               'measure state was not restored')
        QTimer.singleShot(800, toggle_ruler)

    def toggle_ruler():
        win = window()
        mark('ruler.activate.start')
        measured('ruler_activate', lambda: click_button(win.btn_review_ruler))
        mark('ruler.activate.return')
        expect(win.btn_review_ruler.property('toolActive') and
               not win.btn_review_measure.property('toolActive'),
               'ruler/measure active states are not exclusive')
        if os.getenv('FACD_QT_INPUT_PROBE') == '1':
            QTimer.singleShot(500, click_ruler_points)
        else:
            QTimer.singleShot(2500, finish)

    def finish(code=0):
        if state.get('failed') and code == 0:
            return
        win = window()
        if win is not None and code == 0 and win.btn_review_ruler.isChecked():
            click_button(win.btn_review_ruler)
        if code == 0:
            expect(win is not None and not win.btn_review_ruler.property('toolActive'),
                   'ruler state was not restored')
        mark('scenario.finish')
        if win is not None:
            win.close()
        QApplication.instance().exit(code)

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
