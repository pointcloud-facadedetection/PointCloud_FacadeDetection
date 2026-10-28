from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _cpp_pointer(obj):
    if obj is None:
        return None
    try:
        from shiboken6 import getCppPointer
        return int(getCppPointer(obj)[0])
    except Exception:
        return None


class ViewportProbe:
    def __init__(self, viewport, output_path):
        self.viewport = viewport
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open('w', encoding='utf-8')
        self._started = time.perf_counter()
        self._frame_started = []
        self._wrapped = []

    def start(self):
        viewport = self.viewport
        widget = viewport._widget
        for name in ('initializeGL', '_cleanup_context', 'resizeGL', 'paintGL'):
            self._wrap(widget, name, name)
        for name in ('_upload_cloud', 'add_cloud', 'update_cloud_points',
                     'update_cloud_color', 'show_heatmap_quad',
                     'enter_plane_pick_mode', 'show_measure_line',
                     'show_ruler_overlay', '_flush_render'):
            self._wrap(viewport, name, name)
        rw = widget.render_window
        rw.AddObserver('StartEvent', self._on_frame_start)
        rw.AddObserver('EndEvent', self._on_frame_end)
        context = widget.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._on_context_destroyed)
        self.mark('probe.start')

    def _wrap(self, obj, name, event):
        original = getattr(obj, name, None)
        if original is None:
            return

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - started) * 1000.0
                self._event(event, elapsed_ms=round(elapsed, 3),
                            **self._context_snapshot())

        setattr(obj, name, wrapped)
        self._wrapped.append((obj, name, original))

    def _event(self, kind, **details):
        if self._file.closed:
            return
        record = {
            't': round(time.perf_counter() - self._started, 6),
            'event': kind,
            **details,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._file.flush()

    def _on_frame_start(self, *_args):
        self._frame_started.append(time.perf_counter())
        self._event('frame.start', **self._context_snapshot())

    def _on_frame_end(self, *_args):
        started = self._frame_started.pop() if self._frame_started else None
        elapsed = ((time.perf_counter() - started) * 1000.0
                   if started is not None else None)
        self._event('frame.end', elapsed_ms=round(elapsed, 3) if elapsed else None,
                    **self._context_snapshot())

    def _on_context_destroyed(self):
        self._event('context.about_to_be_destroyed', **self._context_snapshot())

    def _context_snapshot(self):
        widget = self.viewport._widget
        try:
            context = widget.context()
        except Exception:
            context = None
        try:
            fbo = widget.defaultFramebufferObject()
        except Exception:
            fbo = None
        return {
            'context_ptr': _cpp_pointer(context),
            'vtk_context_ptr': _cpp_pointer(getattr(widget, '_ctx', None)),
            'fbo': fbo,
        }

    def _cloud_snapshot(self):
        clouds = []
        for name, data in self.viewport.point_data.items():
            poly = self.viewport._polys.get(name)
            actor = self.viewport._actors.get(name)
            upload_tick = None
            vbo_handle = None
            if actor is not None:
                try:
                    vbo = actor.GetMapper().GetVBOs().GetVBO('vertexMC')
                    upload_tick = int(vbo.GetUploadTime().GetMTime())
                    vbo_handle = int(vbo.GetHandle())
                except Exception:
                    pass
            clouds.append({
                'name': name,
                'points': len(data.get('pos', [])),
                'array_ptr': _cpp_pointer(data.get('pos')),
                'poly': poly.GetAddressAsString('') if poly is not None else None,
                'actor': actor.GetAddressAsString('') if actor is not None else None,
                'vbo_upload_tick': upload_tick,
                'vbo_handle': vbo_handle,
            })
        return clouds

    def mark(self, label):
        self._event('mark', label=label, **self._context_snapshot(),
                    clouds=self._cloud_snapshot())

    def close(self):
        for obj, name, original in self._wrapped:
            try:
                setattr(obj, name, original)
            except Exception:
                pass
        self._wrapped.clear()
        self._file.close()


def _spin(app, milliseconds):
    from PySide6.QtCore import QThread
    deadline = time.perf_counter() + milliseconds / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents()
        QThread.msleep(5)


def run_scenario(output_path, point_count=150000):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QSplitter, QWidget
    from view3d.pyvista_viewport import PyVistaViewport

    os.environ.pop('QT_QPA_PLATFORM', None)
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle('Viewport probe')
    window.resize(1100, 720)
    left = QSplitter(Qt.Horizontal)
    right = QSplitter(Qt.Horizontal)
    window.setCentralWidget(left)
    viewport = PyVistaViewport()
    panel = QWidget()
    layout = QHBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(viewport.get_widget())
    left.addWidget(panel)
    probe = ViewportProbe(viewport, output_path)
    probe.start()

    rng = np.random.default_rng(42)
    points = rng.random((point_count, 3), dtype=np.float32) * 4.0
    colors = np.broadcast_to(np.array([0.65, 0.65, 0.65], dtype=np.float32),
                             (point_count, 3)).copy()
    viewport.add_cloud('synthetic', points, colors)
    image = np.zeros((512, 512, 4), dtype=np.uint8)
    image[..., 0] = 30
    image[..., 1] = 100
    image[..., 2] = 220
    image[..., 3] = 180
    quad_args = dict(
        image=image, origin=[0, 0, 0], u_axis=[0, 1, 0], v_axis=[0, 0, 1],
        uv_bounds=(0, 0, 4, 3), normal=[1, 0, 0], cache_key=('synthetic', 'mode'))
    viewport.show_heatmap_quad(**quad_args)

    def scenario():
        probe.mark('baseline')
        viewport.enter_plane_pick_mode([1.0, 0, 0, 0], callback=None)
        _spin(app, 100)
        probe.mark('pick.activate')
        viewport.update_pick_markers(src_points=[[0, 1, 1]])
        _spin(app, 100)
        probe.mark('pick.first_marker')
        viewport.show_measure_line([0, 1, 1], [0, 2, 2])
        _spin(app, 100)
        probe.mark('measure.line')
        viewport.show_heatmap_quad(**quad_args)
        _spin(app, 100)
        probe.mark('heatmap.warm')
        panel.setParent(None)
        right.addWidget(panel)
        right.setParent(window)
        window.setCentralWidget(right)
        panel.show()
        _spin(app, 250)
        probe.mark('viewport.reparent')
        args = dict(quad_args)
        args['cache_key'] = ('synthetic', 'new-mode')
        viewport.show_heatmap_quad(**args)
        _spin(app, 150)
        probe.mark('heatmap.cold_new_key')
        summary_path = Path(output_path).with_suffix('.summary.json')
        summary_path.write_text(json.dumps({
            'context': probe._context_snapshot(),
            'clouds': probe._cloud_snapshot(),
            'output': str(output_path),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'probe_output={output_path}')
        print(f'probe_summary={summary_path}')
        probe.close()
        window.close()
        app.quit()

    window.show()
    QTimer.singleShot(500, scenario)
    return app.exec()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(
        Path(__file__).resolve().parents[2] / 'data' / 'diagnostics' / 'viewport_probe.jsonl'))
    parser.add_argument('--points', type=int, default=150000)
    args = parser.parse_args()
    raise SystemExit(run_scenario(args.output, args.points))
