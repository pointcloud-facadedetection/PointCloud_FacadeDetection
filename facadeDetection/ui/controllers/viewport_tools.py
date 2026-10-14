"""复核页视口工具编排：两点测距 + 交互靠尺。

拾取方式：屏幕点击 → 相机射线 → 与选中立面的 plane_model 数学求交
（viewport.enter_plane_pick_mode / ray_plane_pick），不依赖云点 proximity。
几何计算在 algorithms.facade.interactive_ruler，VTK 可视化在视口层
（show_measure_line / show_ruler_overlay，仅 PyVista 后端实现，
其它后端经 hasattr 退化为纯文本结果）。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from algorithms.facade import interactive_ruler


class _RulerSurfaceWorker(QThread):
    """后台构建靠尺 Surface（1.8M 立面点约 1.8s，同步会卡 UI）。

    generation 用于作废过期结果：cancel/invalidate 后 worker 仍跑完，
    但代数不匹配的结果直接丢弃。
    """

    done = Signal(int, object, int)      # generation, surface, facade_id
    failed = Signal(int, str)

    def __init__(self, generation, facade_id, facade, plane_model,
                 plane_origin, params, facade_service, cloud, parent=None):
        super().__init__(parent)
        self._generation = generation
        self._facade_id = facade_id
        self._facade = facade
        self._plane_model = plane_model
        self._plane_origin = plane_origin
        self._params = params
        self._facade_service = facade_service
        self._cloud = cloud

    def run(self):
        try:
            dataset = self._facade_service.get_dataset(self._cloud)
            points = interactive_ruler.extract_facade_points(
                self._facade, dataset.processed_raw_points,
                index=getattr(dataset, 'index', None))
            if points is None:
                self.failed.emit(self._generation,
                                 '选中立面点数不足，无法放置靠尺。')
                return
            p = self._params
            surface = interactive_ruler.build_interactive_surface(
                points, self._plane_model,
                plane_origin=self._plane_origin,
                ruler_dir=interactive_ruler.resolve_ruler_direction(
                    self._facade),
                select_band=float(p.get('select_band', 0.01)),
                hole_band=float(p.get('hole_band', 0.02)),
                bin_size=float(p.get('bin_size', 0.04)),
                top_q=float(p.get('top_q', 1.0)),
                sor=p.get('sor'))
            self.done.emit(self._generation, surface, self._facade_id)
        except Exception as exc:
            self.failed.emit(self._generation, str(exc))


class ViewportToolsController(QObject):
    status_message = Signal(str, int)      # 状态栏文本与超时（毫秒）
    warning_requested = Signal(str)        # QMessageBox.warning 正文
    measure_text = Signal(str)             # 复核页工具区结果文案
    tool_activated = Signal(str)           # 'measure' / 'ruler'
    tool_deactivated = Signal()            # 工具退出（页面复位按钮态）

    TOOL_MEASURE = 'measure'
    TOOL_RULER = 'ruler'

    def __init__(self, viewport, facade_service, cloud_provider,
                 facade_provider, ruler_params_provider, parent=None):
        super().__init__(parent)
        self._viewport = viewport
        self._facade_service = facade_service
        # cloud_provider() -> 当前点云名；facade_provider() -> 当前选中立面 dict
        # ruler_params_provider() -> 尺长/尺宽/限差等参数快照 dict
        self._cloud_provider = cloud_provider
        self._facade_provider = facade_provider
        self._ruler_params_provider = ruler_params_provider
        self._tool = None
        self._measure_points = []
        self._plane_model = None
        self._plane_geometry = None
        self._ruler_surface = None
        self._ruler_facade_id = None
        self._surface_gen = 0
        self._surface_pending = False
        self._surface_worker = None

    @property
    def active_tool(self):
        return self._tool

    # ------------------------------------------------------------------
    # 工具开关（由复核页 checkable 按钮驱动）
    # ------------------------------------------------------------------
    def toggle_measure(self, enabled):
        print(f'[PCFD] tools.toggle_measure({enabled})', flush=True)
        if enabled:
            self._activate(self.TOOL_MEASURE)
        elif self._tool == self.TOOL_MEASURE:
            self.cancel()

    def toggle_ruler(self, enabled):
        print(f'[PCFD] tools.toggle_ruler({enabled})', flush=True)
        if enabled:
            self._activate(self.TOOL_RULER)
        elif self._tool == self.TOOL_RULER:
            self.cancel()

    def _hint(self, message):
        """前置条件提示：工具区标签 + 状态栏，不用模态弹窗（避免遮挡冻结）。"""
        print(f'[PCFD] tools.hint: {message}', flush=True)
        self.measure_text.emit(message)
        self.status_message.emit(message, 6000)

    def _activate(self, tool):
        viewport = self._viewport
        if viewport is None or not hasattr(viewport, 'enter_plane_pick_mode'):
            self._hint('当前视口后端不支持平面拾取。')
            self.tool_deactivated.emit()
            return
        facade = self._facade_provider()
        print(f'[PCFD] tools._activate({tool}) facade='
              f'{None if not facade else facade.get("id")}', flush=True)
        if not facade:
            self._hint('请先在右侧立面列表中选中一个立面，再使用测距/靠尺。')
            self.tool_deactivated.emit()
            return
        if not self._prepare_plane(facade):
            self.tool_deactivated.emit()
            return
        self._clear_state()
        if tool == self.TOOL_RULER:
            self._start_ruler_surface_build(facade)
        self._tool = tool
        if tool == self.TOOL_MEASURE:
            viewport.enter_plane_pick_mode(self._plane_model,
                                           callback=self._on_measure_pick)
            self.measure_text.emit('测距：在立面上点击第一个点')
            self.status_message.emit('测距模式：点击立面上两个点测量距离，Esc 退出', 5000)
        else:
            viewport.enter_plane_pick_mode(self._plane_model,
                                           callback=self._on_ruler_pick)
            self.measure_text.emit('靠尺：在立面上点击放置位置')
            self.status_message.emit('靠尺模式：点击立面放置靠尺，Esc 退出', 5000)
        self.tool_activated.emit(tool)

    def _prepare_plane(self, facade):
        """从选中立面取拾取平面与四边面几何（质量结果优先）。"""
        quality = facade.get('quality_report')
        geometry = (quality or {}).get('facade_geometry') \
            if isinstance(quality, dict) else None
        plane = None
        if isinstance(geometry, dict):
            plane = np.asarray(geometry.get('plane_model') or [],
                               dtype=np.float64).reshape(-1)
        if plane is None or plane.size != 4:
            plane = np.asarray(facade.get('plane_model') or [],
                               dtype=np.float64).reshape(-1)
            geometry = None
        if plane.size != 4 or np.linalg.norm(plane[:3]) < 1e-12:
            self._hint('选中立面缺少平面参数，请先完成质量检测。')
            return False
        self._plane_model = plane
        self._plane_geometry = geometry
        return True

    def cancel(self):
        """Esc / 页面迁出：退出拾取模式并清空全部工具可视化。"""
        import traceback
        print('[PCFD] tools.cancel() tool=%s\n%s'
              % (self._tool, ''.join(traceback.format_stack(limit=6)[:-1])),
              flush=True)
        self._clear_state()
        viewport = self._viewport
        if viewport is not None and hasattr(viewport, 'exit_pick_mode'):
            try:
                viewport.exit_pick_mode()
            except Exception:
                pass
        self._surface_gen += 1        # 作废在跑的靠尺表面构建结果
        self._surface_pending = False
        if self._tool is not None:
            self._tool = None
            self.measure_text.emit('')
            self.tool_deactivated.emit()

    def clear_visuals(self):
        """【清除】按钮：清空标记与结果，保持当前工具模式可用。"""
        self._clear_state()
        if self._tool == self.TOOL_MEASURE:
            self.measure_text.emit('测距：在立面上点击第一个点')
        elif self._tool == self.TOOL_RULER:
            self.measure_text.emit('靠尺：在立面上点击放置位置')

    def invalidate(self):
        """选中立面/点云切换：清空靠尺缓存；工具激活中则以新立面重建。"""
        self._ruler_surface = None
        self._ruler_facade_id = None
        self._surface_gen += 1        # 作废在跑的构建结果
        self._surface_pending = False
        if self._tool is not None:
            tool = self._tool
            self._tool = None
            self._activate(tool)

    def _clear_state(self):
        self._measure_points = []
        viewport = self._viewport
        if viewport is not None and hasattr(viewport, 'clear_tool_visuals'):
            try:
                viewport.clear_tool_visuals()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 测距（平面交点）
    # ------------------------------------------------------------------
    def _on_measure_pick(self, picked):
        print(f'[PCFD] tools._on_measure_pick picked={picked is not None}',
              flush=True)
        if not picked:
            return
        point = np.asarray(picked.get('point'), dtype=np.float64)
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            point, self._plane_geometry)
        if inside is False:
            self.measure_text.emit('拾取点在立面范围外，请点击立面内部')
            return
        viewport = self._viewport
        if len(self._measure_points) >= 2:
            # 新一轮测量：清掉上一组连线
            self._measure_points = []
            if hasattr(viewport, 'clear_measure_visuals'):
                viewport.clear_measure_visuals()
        self._measure_points.append(point)
        if len(self._measure_points) == 1:
            viewport.update_pick_markers(src_points=[point])
            self.measure_text.emit('测距：点击第二个点')
            return
        p1, p2 = self._measure_points
        distance = float(np.linalg.norm(p2 - p1))
        viewport.clear_pick_markers()
        if hasattr(viewport, 'show_measure_line'):
            viewport.show_measure_line(p1, p2)
        text = f'距离：{distance:.3f} m（{distance * 1000:.1f} mm）'
        self.measure_text.emit(text)
        self.status_message.emit(text, 0)

    # ------------------------------------------------------------------
    # 交互靠尺（平面交点作为尺心，place_ruler 内部自行钳制）
    # ------------------------------------------------------------------
    def _start_ruler_surface_build(self, facade):
        """后台构建靠尺 Surface（百万级立面点约 1-2s，同步会卡 UI）。

        已有匹配缓存则直接复用；构建期间点击会得到"准备中"提示。
        """
        facade_id = int(facade.get('id', -1))
        if (self._ruler_surface is not None
                and self._ruler_facade_id == facade_id):
            return
        cloud = self._cloud_provider()
        if not cloud:
            self._hint('当前没有可用点云。')
            self.tool_deactivated.emit()
            return
        plane_origin = None
        if isinstance(self._plane_geometry, dict):
            plane_origin = self._plane_geometry.get('origin')
        if plane_origin is None:
            plane_origin = facade.get('center')
        params = dict(self._ruler_params_provider() or {})
        self._surface_gen += 1
        generation = self._surface_gen
        self._surface_pending = True
        worker = _RulerSurfaceWorker(
            generation, facade_id, facade, self._plane_model,
            plane_origin, params, self._facade_service, cloud, parent=self)
        worker.done.connect(self._on_surface_built)
        worker.failed.connect(self._on_surface_build_failed)
        worker.finished.connect(self._on_surface_thread_finished)
        self._surface_worker = worker
        self.measure_text.emit('靠尺：正在准备靠尺表面…')
        worker.start()

    def _on_surface_built(self, generation, surface, facade_id):
        if generation != self._surface_gen:
            return    # 已取消/换立面，丢弃过期结果
        self._ruler_surface = surface
        self._ruler_facade_id = facade_id
        self._surface_pending = False
        if self._tool == self.TOOL_RULER:
            self.measure_text.emit('靠尺：在立面上点击放置位置')

    def _on_surface_build_failed(self, generation, message):
        if generation != self._surface_gen:
            return
        self._surface_pending = False
        self._hint(f'靠尺表面构建失败：{message}')
        if self._tool == self.TOOL_RULER:
            self.cancel()

    def _on_surface_thread_finished(self):
        self._surface_worker = None

    def _on_ruler_pick(self, picked):
        if not picked:
            return
        if self._ruler_surface is None:
            if self._surface_pending:
                self.measure_text.emit('靠尺表面准备中，请稍候…')
            return
        params = dict(self._ruler_params_provider() or {})
        threshold_mm = params.get('threshold_mm')
        point = np.asarray(picked.get('point'), dtype=np.float64)
        _, _, inside = interactive_ruler.point_in_facade_bounds(
            point, self._plane_geometry)
        try:
            result = interactive_ruler.place_ruler(
                self._ruler_surface, point,
                ruler_length=float(params.get('ruler_length', 2.0)),
                ruler_width=float(params.get('ruler_width', 0.055)),
                threshold_mm=threshold_mm,
                max_hole_ratio=float(params.get('max_hole_ratio', 0.2)))
        except Exception as exc:
            self.status_message.emit(f'靠尺计算失败：{exc}', 5000)
            return
        if not result.get('ok'):
            self.status_message.emit(
                f"靠尺放置失败：{result.get('reason', '未知原因')}", 5000)
            return
        if hasattr(self._viewport, 'show_ruler_overlay'):
            self._viewport.show_ruler_overlay(
                result['render'], exceeded=result.get('exceeded', False))
        gap_mm = float(result.get('gap_mm', 0.0))
        if threshold_mm is not None:
            verdict = ('超限' if result.get('exceeded') else '合格')
            text = (f"靠尺 gap = {gap_mm:.1f} mm"
                    f"（限差 {float(threshold_mm):.0f} mm）：{verdict}")
        else:
            text = f'靠尺 gap = {gap_mm:.1f} mm'
        if inside is False:
            text += '；拾取点在立面范围外，尺心已钳制到立面边缘'
        if not result.get('coverage_valid', True):
            text += f"；空洞占比 {result.get('hole_ratio', 0.0) * 100:.0f}%，结果仅供参考"
        self.measure_text.emit(text)
        self.status_message.emit(text, 0)
