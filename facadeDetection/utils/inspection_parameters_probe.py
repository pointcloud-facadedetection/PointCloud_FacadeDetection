from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def run(project_id):
    from main import main

    state = {'ticks': 0, 'dialog_ticks': 0, 'original': None, 'facade': None}

    def window():
        return next((w for w in QApplication.topLevelWidgets()
                     if w.__class__.__name__ == 'MainWindow'), None)

    def finish(code=0):
        for widget in QApplication.topLevelWidgets():
            widget.close()
        QApplication.instance().exit(code)

    def restore_and_finish(win):
        from services.inspection_parameters import InspectionParametersService
        service = InspectionParametersService
        project_uuid = win.current_project.project_id
        facade_id = int(state['facade']['id'])
        original_default, original_facade = state['original']
        if original_default:
            service.save(project_uuid, original_default)
        if original_facade:
            service.save(project_uuid, original_facade, facade_id)
        else:
            service.use_project_default(project_uuid, facade_id)
        print('[parameters-probe] restored original parameters', flush=True)
        finish()

    def verify(win, facade):
        from db.connection import project_session
        from models import Facade
        from services.inspection_parameters import InspectionParametersService
        from ui.dialogs.inspection_parameters_dialog import InspectionParametersDialog
        values = json.loads(win.current_project.inspection_params_json)
        expected = state['expected']
        assert values['ruler_length_m'] == expected, values
        assert '已单独设置' in win.lbl_inspection_params_source.text()
        with project_session(win.current_project.project_id) as session:
            saved = session.get(Facade, int(facade['id']))
            assert saved is not None
            assert json.loads(saved.inspection_params_json)['ruler_length_m'] == expected
        other = InspectionParametersDialog(win.current_project, facade_no=99, parent=win)
        assert other.cb_ruler_length.currentData() == expected
        other.close()
        fallback, source = InspectionParametersService.resolve(
            win.current_project.inspection_params_json, None)
        assert source == 'default' and fallback['ruler_length_m'] == expected
        print('[parameters-probe] PASS: operation UI, facade override, latest default, DB reload',
              flush=True)
        restore_and_finish(win)

    def edit_dialog(win, facade):
        dialog = QApplication.activeModalWidget()
        if dialog is None or not hasattr(dialog, 'inspection_values'):
            state['dialog_ticks'] += 1
            if state['dialog_ticks'] > 30:
                print('[parameters-probe] dialog did not open', flush=True)
                finish(2)
            else:
                QTimer.singleShot(200, lambda: edit_dialog(win, facade))
            return
        index = (dialog.cb_ruler_length.currentIndex() + 1) % dialog.cb_ruler_length.count()
        dialog.cb_ruler_length.setCurrentIndex(index)
        state['expected'] = dialog.cb_ruler_length.currentData()
        dialog._on_accept()
        QTimer.singleShot(300, lambda: verify(win, facade))

    def select_facade(win):
        if win.list_facades.count() == 0:
            print('[parameters-probe] no persisted facade', flush=True)
            finish(2)
            return
        item = win.list_facades.item(0)
        win.list_facades.setCurrentItem(item)
        win._on_facade_item_clicked(item)
        facade = win._facade_from_item(item)
        state['facade'] = facade
        original_default = json.loads(win.current_project.inspection_params_json or '{}')
        original_facade = json.loads(facade.get('inspection_params_json') or 'null')
        state['original'] = (original_default, original_facade)
        print('[parameters-probe] ' + win.lbl_inspection_params_source.text(), flush=True)
        QTimer.singleShot(500, lambda: edit_dialog(win, facade))
        win.btn_inspection_params.click()

    def open_project():
        win = window()
        state['ticks'] += 1
        if win is None:
            QTimer.singleShot(250, open_project)
            return
        if win.current_project is None and not state.get('activated'):
            project = win.project_overview_service.get_project(project_id)
            if project is None:
                print('[parameters-probe] cached project not found', flush=True)
                finish(2)
                return
            state['activated'] = True
            print('[parameters-probe] activate once', flush=True)
            win._activate_project(project)
            QTimer.singleShot(500, open_project)
            return
        if state['ticks'] % 20 == 0:
            print(f"[parameters-probe] waiting project={win.current_project is not None} "
                  f"facades={len(win.project_operation_service.last_facade_results or [])} "
                  f"listed={win.list_facades.count()}", flush=True)
        if win.list_facades.count() > 0:
            win.set_current_page(1)
            QTimer.singleShot(500, lambda: select_facade(win))
        elif state['ticks'] > 160:
            print('[parameters-probe] project load timeout', flush=True)
            finish(3)
        else:
            QTimer.singleShot(500, open_project)

    original_exec = QApplication.exec

    def exec_with_watchdog():
        QTimer.singleShot(500, open_project)
        QTimer.singleShot(100000, lambda: finish(4))
        return original_exec()

    QApplication.exec = staticmethod(exec_with_watchdog)
    return main()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run(os.environ['FACD_PROBE_PROJECT'])
