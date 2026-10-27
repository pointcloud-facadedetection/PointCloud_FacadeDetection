from PySide6.QtCore import Qt, QSize, QUrl, QThread, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

import os

import cv2
import numpy as np

from algorithms.facade import interactive_ruler
from services.heatmap_spec import heatmap_spec


class _HeatRectWorker(QThread):
    done = Signal(object, object)
    failed = Signal(object, str)

    def __init__(self, key, dataset, quality, image_hw, reconstruct, parent=None):
        super().__init__(parent)
        self._key = key
        self._dataset = dataset
        self._quality = quality
        self._image_hw = image_hw
        self._reconstruct = reconstruct

    def run(self):
        try:
            rect = self._reconstruct(
                self._dataset, self._quality, self._image_hw)
            self.done.emit(self._key, rect)
        except Exception as exc:
            self.failed.emit(self._key, str(exc))


class InspectionReviewPageMixin:
    """检测复核页 mixin：布局 + 视口迁移 + 复核立面列表。

    接收从【项目操作】页迁移过来的 viewport_panel（3D 视口）和
    right_dock（立面检测结果面板），形成两栏布局。
    """

    # 复核页热力贴图的 4 档模式（值与导出 PNG 文件名中的 mode 一致）
    # 全局平面两档（红蓝渐变热力）优先，默认选中第一档
    REVIEW_HEATMAP_MODES = (
        ('global_plane_flatness_area', '全局平整度'),
        ('global_plane_verticality_area', '全局垂直度'),
        ('ruler_flatness_area', '靠尺平整度'),
        ('ruler_verticality_area', '靠尺垂直度'),
    )

    def _create_inspection_review_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(page_title, page_key)

        self._review_camera_snapshot = None
        self._review_heatmap_mode = self.REVIEW_HEATMAP_MODES[0][0]
        body_layout.addWidget(self._create_review_tools_bar(), 0)

        # 检测复核页：两栏 splitter（视口 + 右侧面板）
        self.review_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.review_splitter.setObjectName('reviewSplitter')
        self.review_splitter.setChildrenCollapsible(False)
        self.review_splitter.setHandleWidth(1)

        # 左侧占位：视口迁移到本页时被摘出隐藏，迁回操作页时重新挂回
        self.review_viewport_placeholder = QWidget()
        self.review_viewport_placeholder.setObjectName('reviewViewportPlaceholder')
        pl = QVBoxLayout(self.review_viewport_placeholder)
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel('3D 检测工作台将在此处显示')
        lbl.setStyleSheet('font-size:14px; color:#64748B;')
        pl.addWidget(lbl)

        # 右侧占位：右侧面板迁移到本页时被摘出隐藏，迁回操作页时重新挂回
        self.review_right_placeholder = QWidget()
        self.review_right_placeholder.setObjectName('reviewRightPlaceholder')
        pl2 = QVBoxLayout(self.review_right_placeholder)
        pl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl2 = QLabel('立面检测结果将在此处显示')
        lbl2.setStyleSheet('font-size:14px; color:#64748B;')
        pl2.addWidget(lbl2)

        self.review_splitter.addWidget(self.review_viewport_placeholder)
        self.review_splitter.addWidget(self.review_right_placeholder)
        self.review_splitter.setStretchFactor(0, 1)
        self.review_splitter.setStretchFactor(1, 0)
        self.review_splitter.setSizes([1040, 300])

        body_layout.addWidget(self.review_splitter, 1)
        return page

    # ------------------------------------------------------------------
    # 视口工具区：测距 / 放靠尺 / 清除（编排见 ViewportToolsController）
    # ------------------------------------------------------------------
    def _create_review_tools_bar(self):
        bar = QFrame()
        bar.setObjectName('reviewToolsBar')
        bar.setStyleSheet("""
            #reviewToolsBar {
                background: #fafbfc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
        """)
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        title = QLabel('视口工具')
        title.setStyleSheet('font-weight:700; color:#1e293b; font-size:12px;')
        row.addWidget(title)

        self.btn_review_measure = QPushButton('测距')
        self.btn_review_measure.setObjectName('btn_review_measure')
        self.btn_review_measure.setProperty('buttonRole', 'secondary')
        self.btn_review_measure.setCheckable(True)
        self.btn_review_measure.setEnabled(False)
        self.btn_review_measure.setToolTip('在视口中依次点击两个点，测量其距离')
        self.btn_review_measure.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_measure.toggled.connect(self._on_review_measure_toggled)
        row.addWidget(self.btn_review_measure)

        self.btn_review_ruler = QPushButton('放靠尺')
        self.btn_review_ruler.setObjectName('btn_review_ruler')
        self.btn_review_ruler.setProperty('buttonRole', 'secondary')
        self.btn_review_ruler.setCheckable(True)
        self.btn_review_ruler.setEnabled(False)
        self.btn_review_ruler.setToolTip(
            '先在右侧列表选中立面；第一击定起点，第二击定方向（尺长 2m）')
        self.btn_review_ruler.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_ruler.toggled.connect(self._on_review_ruler_toggled)
        row.addWidget(self.btn_review_ruler)

        self.btn_review_clear = QPushButton('清除')
        self.btn_review_clear.setObjectName('btn_review_clear')
        self.btn_review_clear.setProperty('buttonRole', 'secondary')
        self.btn_review_clear.setToolTip('清除测距连线与靠尺标记')
        self.btn_review_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_clear.clicked.connect(self._on_review_clear_clicked)
        row.addWidget(self.btn_review_clear)

        self.btn_review_resize = QPushButton('调整边界')
        self.btn_review_resize.setObjectName('btn_review_resize')
        self.btn_review_resize.setProperty('buttonRole', 'secondary')
        self.btn_review_resize.setToolTip('拖动四边面边中点，松开后增量更新热力图')
        self.btn_review_resize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_resize.setEnabled(False)
        self.btn_review_resize.clicked.connect(self._on_review_resize_clicked)
        row.addWidget(self.btn_review_resize)

        mode_label = QLabel('热力图')
        mode_label.setStyleSheet('color:#475569; font-size:12px;')
        row.addWidget(mode_label)
        self.review_heatmap_combo = QComboBox()
        self.review_heatmap_combo.setObjectName('reviewHeatmapCombo')
        for mode, label in self.REVIEW_HEATMAP_MODES:
            self.review_heatmap_combo.addItem(label, mode)
        self.review_heatmap_combo.setToolTip('选中立面后切换热力贴图模式')
        self.review_heatmap_combo.currentIndexChanged.connect(
            self._on_review_heatmap_mode_changed)
        row.addWidget(self.review_heatmap_combo)

        self.btn_review_open_results = QPushButton('打开结果文件夹')
        self.btn_review_open_results.setObjectName('btn_review_open_results')
        self.btn_review_open_results.setProperty('buttonRole', 'secondary')
        self.btn_review_open_results.setToolTip('打开选中立面所在站点的结果目录')
        self.btn_review_open_results.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_review_open_results.clicked.connect(
            self._on_review_open_results_dir)
        row.addWidget(self.btn_review_open_results)

        self.lbl_review_heatmap_hint = QLabel('')
        self.lbl_review_heatmap_hint.setObjectName('lblReviewHeatmapHint')
        self.lbl_review_heatmap_hint.setStyleSheet(
            'color:#64748B; font-size:12px;')
        row.addWidget(self.lbl_review_heatmap_hint)

        self.lbl_review_tool_result = QLabel('')
        self.lbl_review_tool_result.setObjectName('lblReviewToolResult')
        self.lbl_review_tool_result.setStyleSheet(
            'color:#334155; font-size:12px;')
        row.addWidget(self.lbl_review_tool_result, 1)

        controller = self.viewport_tools_controller
        controller.measure_text.connect(self.lbl_review_tool_result.setText)
        controller.tool_activated.connect(self._on_review_tool_activated)
        controller.tool_deactivated.connect(self._on_review_tool_deactivated)

        escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.viewport_panel)
        escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(controller.cancel)
        return bar

    def _on_review_measure_toggled(self, checked):
        print(f'[PCFD] review.btn_measure_toggled({checked})', flush=True)
        self.viewport_tools_controller.toggle_measure(checked)

    def _on_review_ruler_toggled(self, checked):
        print(f'[PCFD] review.btn_ruler_toggled({checked})', flush=True)
        self.viewport_tools_controller.toggle_ruler(checked)

    def _on_review_clear_clicked(self):
        self.viewport_tools_controller.clear_visuals()

    def _on_review_resize_clicked(self):
        facade = self._selected_review_facade()
        quality = (facade or {}).get('quality_report') or {}
        geometry = quality.get('facade_geometry')
        viewport = getattr(self, 'viewport', None)
        if not isinstance(geometry, dict) or viewport is None \
                or not hasattr(viewport, 'enter_heatmap_resize'):
            self.statusBar().showMessage('当前立面没有可调整的四边面', 4000)
            return
        if self.facade_quality_controller.active_quality_worker is not None:
            self.statusBar().showMessage('质量评估进行中，请完成后再调整边界', 4000)
            return
        self.viewport_tools_controller.cancel()
        self._review_resize_facade_id = int(facade.get('id', -1))
        viewport.enter_heatmap_resize(
            geometry, on_complete=self._on_review_resize_complete)
        self._show_review_resize_confirm_card()
        self.lbl_review_heatmap_hint.setText('拖动边中点，调整后点击右下角确认')

    def _show_review_resize_confirm_card(self):
        from PySide6.QtCore import QPoint
        self._close_review_resize_confirm_card()
        card = QFrame(
            self,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        card.setObjectName('roiConfirmCard')
        card.setFixedSize(190, 96)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        label = QLabel('是否更新立面边界？')
        label.setStyleSheet('font-size:13px; font-weight:600; color:#0F172A;')
        layout.addWidget(label)
        row = QHBoxLayout()
        cancel = QPushButton('取消')
        cancel.setProperty('buttonRole', 'secondary')
        confirm = QPushButton('确认')
        confirm.setProperty('buttonRole', 'primary')
        cancel.clicked.connect(
            lambda: self._on_review_resize_confirm(card, False))
        confirm.clicked.connect(
            lambda: self._on_review_resize_confirm(card, True))
        row.addWidget(cancel)
        row.addWidget(confirm)
        layout.addLayout(row)
        panel = getattr(self, 'viewport_panel', None)
        if panel is not None:
            origin = panel.mapToGlobal(QPoint(0, 0))
            card.move(origin.x() + max(0, panel.width() - 202),
                      origin.y() + max(0, panel.height() - 108))
        else:
            card.move(100, 100)
        card.show()
        card.raise_()
        self._review_resize_confirm_card = card

    def _close_review_resize_confirm_card(self):
        card = getattr(self, '_review_resize_confirm_card', None)
        self._review_resize_confirm_card = None
        if card is not None:
            card.close()
            card.deleteLater()

    def _on_review_resize_confirm(self, card, accepted):
        self._close_review_resize_confirm_card()
        viewport = getattr(self, 'viewport', None)
        if viewport is None:
            return
        if accepted:
            viewport.confirm_heatmap_resize()
        else:
            viewport.exit_heatmap_resize()
            self._review_repaint_quad()

    def _on_review_resize_complete(self, uv_bounds):
        facade = self._selected_review_facade()
        if not facade or int(facade.get('id', -2)) != getattr(
                self, '_review_resize_facade_id', -1):
            return
        cloud = self.facade_quality_controller.active_cloud_name()
        if not cloud:
            return
        try:
            profile = getattr(self, '_inspection_profile', None)
            band = float(getattr(profile, 'select_band_m', 0.01))
            adjusted = self.project_operation_service.resize_facade(
                cloud, facade, uv_bounds, surface_band_m=band)
            adjusted['__suppress_quality_dialog'] = True
            self.lbl_review_heatmap_hint.setText('正在增量更新热力图…')
            self.btn_review_resize.setEnabled(False)
            self.facade_quality_controller.evaluate_facade(adjusted)
        except Exception as exc:
            self.statusBar().showMessage(f'边界调整失败：{exc}', 6000)
            self._review_repaint_quad()

    def _review_update_tool_buttons(self):
        """测距/放靠尺/边界调整依赖选中立面与空闲质量任务。"""
        enabled = (self._selected_review_facade() is not None
                   and self.facade_quality_controller.active_quality_worker is None)
        for name in ('btn_review_measure', 'btn_review_ruler', 'btn_review_resize'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(enabled)

    def _on_review_tool_activated(self, tool):
        buttons = {'measure': self.btn_review_measure,
                   'ruler': self.btn_review_ruler}
        for name, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(name == tool)
            button.blockSignals(False)

    def _on_review_tool_deactivated(self):
        for button in (self.btn_review_measure, self.btn_review_ruler):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

    def _selected_review_facade(self):
        """靠尺交互与热力贴图的立面来源：右侧立面列表当前选中行。"""
        if not hasattr(self, 'list_facades'):
            return None
        items = self.list_facades.selectedItems()
        if not items:
            return None
        facade = items[0].data(Qt.ItemDataRole.UserRole)
        print(f'[PCFD] review.selected_facade -> '
              f'{None if not facade else facade.get("id")}', flush=True)
        return facade

    # ------------------------------------------------------------------
    # 选中立面呈现：热力贴图 quad + 相机正视（仅复核页模式）
    # ------------------------------------------------------------------
    def _review_repaint_quad(self):
        facade = self._selected_review_facade()
        if facade is not None and getattr(self.render_service,
                                          'review_mode', False):
            self._review_present_facade(facade, focus_camera=False)

    def _review_present_facade(self, facade, focus_camera=True):
        """选中立面：检测时导出的热力 PNG 贴到四边面上 + 相机正视。"""
        viewport = getattr(self, 'viewport', None)
        hint_label = getattr(self, 'lbl_review_heatmap_hint', None)
        self._close_review_resize_confirm_card()
        if viewport is None or not hasattr(viewport, 'show_heatmap_quad'):
            return
        if getattr(viewport, '_quad_resize', None) is not None:
            viewport.exit_heatmap_resize()
        quality = facade.get('quality_report') if isinstance(facade, dict) \
            else None
        geometry = quality.get('facade_geometry') \
            if isinstance(quality, dict) else None
        normal = None
        if isinstance(geometry, dict):
            normal = geometry.get('normal')
            if normal is None:
                normal = np.asarray(geometry['plane_model'],
                                    dtype=float)[:3]
            normal = np.asarray(normal, dtype=np.float64)
            if focus_camera and hasattr(viewport, 'focus_plane'):
                if self._review_camera_snapshot is None:
                    self._review_camera_snapshot = viewport.snapshot_camera()
                bounds = np.asarray(geometry['uv_bounds'],
                                    dtype=np.float64).reshape(4)
                corners = np.asarray(geometry.get('corners_3d') or [],
                                     dtype=np.float64).reshape(-1, 3)
                center = (corners.mean(axis=0) if len(corners)
                          else np.asarray(geometry['origin'],
                                          dtype=np.float64))
                viewport.focus_plane(normal, center,
                                     geometry['v_axis'],
                                     (bounds[2] - bounds[0],
                                      bounds[3] - bounds[1]))

        hint = ''
        if not isinstance(quality, dict) or not isinstance(geometry, dict):
            viewport.clear_heatmap_quad()
            hint = '该立面暂无质量结果，无法生成热力图'
        else:
            rgba, token = self._review_heat_png(facade, quality)
            if rgba is None:
                # 无热力 PNG 也要给选中反馈：仅画四边面边缘高亮
                corners = np.asarray(geometry.get('corners_3d') or [],
                                     dtype=np.float64).reshape(-1, 3)
                if len(corners) >= 3 and hasattr(viewport,
                                                 'show_heatmap_frame'):
                    viewport.show_heatmap_frame(
                        corners,
                        cache_key=('frame', int(facade.get('id', -1))))
                hint = (f"{heatmap_spec(self._review_heatmap_mode)['title']}"
                        '：该立面此模式无热力图')
            else:
                uv_bounds = self._review_heat_uv_bounds(
                    facade, quality, rgba.shape[:2])
                # 同 key（立面+模式+mtime）重贴在视口层只切可见性，
                # 不再销毁重建 actor、不重传纹理
                viewport.show_heatmap_quad(
                    rgba, geometry['origin'], geometry['u_axis'],
                    geometry['v_axis'], uv_bounds, normal=normal,
                    cache_key=token,
                    geometry_changed=getattr(
                        self, '_review_heat_rect_geometry_dirty', False))
                self._review_heat_rect_geometry_dirty = False
                hint = heatmap_spec(self._review_heatmap_mode)['title']
        if hint_label is not None:
            hint_label.setText(hint)

    def _review_heat_png(self, facade, quality):
        """加载透明热力 PNG 为 RGBA；按路径+mtime 缓存，切模式/重选秒切。

        返回 (rgba, cache_token)；无产出返回 (None, None)。cache_token
        为 (立面 id, 模式, mtime)，供视口 actor 组缓存复用。
        """
        png_path = self._review_heatmap_png_path(facade, quality)
        if not png_path:
            return None, None
        try:
            mtime = os.path.getmtime(png_path)
            token = (int(facade.get('id', -1)),
                     str(self._review_heatmap_mode), mtime)
            cache = self.__dict__.setdefault('_review_heat_png_cache', {})
            cached = cache.get(png_path)
            if cached is not None and cached[0] == mtime:
                return cached[1], token
            image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                return None, None
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            cache[png_path] = (mtime, rgba)
            return rgba, token
        except Exception as exc:
            print(f'[PCFD] review.heat_png_failed: {exc!r}', flush=True)
            return None, None

    def _review_heatmap_png_path(self, facade, quality):
        """检测时导出的透明热力 PNG 路径（facade_NNN_{mode}_transparent.png）。"""
        context = quality.get('__export_context') or {}
        results_dir = context.get('results_dir')
        facade_no = int(context.get('facade_no')
                        or facade.get('display_no') or facade.get('id', 0))
        if not results_dir:
            project_uuid = getattr(self.current_project, 'project_id', None)
            if not project_uuid:
                return None
            results_dir = str(
                self.facade_quality_controller._quality_results_dir(
                    project_uuid, facade))
        folder = os.path.join(str(results_dir), f'facade_{facade_no:03d}')
        name = f'facade_{facade_no:03d}_{self._review_heatmap_mode}_transparent.png'
        path = os.path.join(folder, name)
        return path if os.path.isfile(path) else None

    def _review_heat_uv_bounds(self, facade, quality, image_hw):
        """缓存未命中时后台重建裁剪框，GUI 先使用几何边界。"""
        fallback = tuple(np.asarray(quality['facade_geometry']['uv_bounds'],
                                    dtype=np.float64).reshape(4))
        try:
            cloud = self.facade_quality_controller.active_cloud_name()
            dataset = self.facade_service.get_dataset(cloud)
            key = (int(facade.get('id', -1)),
                   str(getattr(dataset, 'revision', None)),
                   int(facade.get('edit_revision', 0) or 0),
                   str(self._review_heatmap_mode),
                   (int(image_hw[0]), int(image_hw[1])))
            self._review_heat_rect_current_key = key
            cache = self.__dict__.setdefault('_review_heat_rect_cache', {})
            if key in cache:
                rect = cache[key]
                return rect['uv_bounds'] if rect else fallback
            workers = self.__dict__.setdefault('_review_heat_rect_workers', {})
            if key not in workers:
                worker = _HeatRectWorker(
                    key, dataset, quality, image_hw,
                    self._reconstruct_heat_rect, parent=self)
                worker.done.connect(self._on_review_heat_rect_ready)
                worker.failed.connect(self._on_review_heat_rect_failed)
                worker.finished.connect(
                    lambda k=key, w=worker: self._on_review_heat_rect_finished(k, w))
                workers[key] = worker
                worker.start()
        except Exception as exc:
            print(f'[PCFD] review.heat_rect_failed: {exc!r}', flush=True)
        return fallback

    def _on_review_heat_rect_ready(self, key, rect):
        cache = self.__dict__.setdefault('_review_heat_rect_cache', {})
        cache[key] = rect
        if key == getattr(self, '_review_heat_rect_current_key', None):
            self._review_heat_rect_geometry_dirty = True
            self._review_repaint_quad()

    def _on_review_heat_rect_failed(self, key, message):
        cache = self.__dict__.setdefault('_review_heat_rect_cache', {})
        cache[key] = None
        print(f'[PCFD] review.heat_rect_failed: {message}', flush=True)

    def _on_review_heat_rect_finished(self, key, worker):
        workers = self.__dict__.setdefault('_review_heat_rect_workers', {})
        if workers.get(key) is worker:
            workers.pop(key, None)

    @staticmethod
    def _reconstruct_heat_rect(dataset, quality, image_hw):
        h, w = int(image_hw[0]), int(image_hw[1])

        def _dims_match(rect):
            if not rect:
                return False
            x0, y0, x1, y1 = rect['crop_xy']
            return (x1 - x0 + 1, y1 - y0 + 1) == (w, h)

        proxy = getattr(dataset, 'proxy_points', None)
        if proxy is not None and len(proxy):
            rect = interactive_ruler.transparent_png_uv_rect(proxy, quality)
            if _dims_match(rect):
                return rect
        # proxy 裁剪框与 PNG 尺寸不符（稀疏代理在 2% 覆盖率规则下可能收缩），
        # 用导出时同一份全量 raw 点复核
        rect = interactive_ruler.transparent_png_uv_rect(
            dataset.processed_raw_points, quality)
        if _dims_match(rect):
            return rect
        if rect:
            x0, y0, x1, y1 = rect['crop_xy']
            print(f'[PCFD] review.heat_rect_mismatch '
                  f'crop=({x1 - x0 + 1}x{y1 - y0 + 1}) png=({w}x{h})',
                  flush=True)
        return None

    def _review_clear_presentation(self):
        """清除选中/迁出复核页：摘除贴图与边缘高亮、恢复相机。"""
        self._review_heat_rect_current_key = None
        viewport = getattr(self, 'viewport', None)
        if viewport is not None and hasattr(viewport, 'clear_heatmap_quad'):
            viewport.clear_heatmap_quad()
        if self._review_camera_snapshot is not None and viewport is not None \
                and hasattr(viewport, 'restore_camera'):
            viewport.restore_camera(self._review_camera_snapshot)
        self._review_camera_snapshot = None
        hint_label = getattr(self, 'lbl_review_heatmap_hint', None)
        if hint_label is not None:
            hint_label.setText('')

    def _on_review_heatmap_mode_changed(self, index):
        mode = self.review_heatmap_combo.itemData(index)
        if mode:
            self._review_heatmap_mode = mode
        self._review_repaint_quad()

    def _on_review_open_results_dir(self):
        facade = self._selected_review_facade()
        if not facade:
            self.statusBar().showMessage('请先在右侧列表中选中一个立面', 3000)
            return
        project_uuid = getattr(self.current_project, 'project_id', None)
        if not project_uuid:
            self.statusBar().showMessage('当前没有打开的项目', 3000)
            return
        results_dir = self.facade_quality_controller._quality_results_dir(
            project_uuid, facade)
        if not results_dir.exists():
            self.statusBar().showMessage(
                f'结果目录不存在：{results_dir}', 5000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(results_dir)))


    # ------------------------------------------------------------------
    # 视口迁移：检测复核页 ↔ 项目操作页
    # ------------------------------------------------------------------
    def _migrate_workspace_to_review(self):
        """将 viewport_panel + right_dock 迁到检测复核页。

        占位 widget 常驻不销毁：迁入时把占位摘出、面板插入；迁出时反向。
        绝不用 QSplitter.replaceWidget——它会消费占位 widget，迁出后
        splitter 计数归零，二次进入时按索引替换必然越界。
        """
        if getattr(self, '_workspace_in_review', False):
            return
        self._operation_splitter_sizes = self.operation_splitter.sizes()
        self.review_viewport_placeholder.setParent(None)
        self.review_right_placeholder.setParent(None)
        # insertWidget 自动把面板从 operation_splitter 摘除并挂入本页
        self.review_splitter.insertWidget(0, self.viewport_panel)
        self.review_splitter.insertWidget(1, self.right_dock)
        self.review_splitter.setSizes([1040, 300])
        self._show_operation_placeholder()
        # 复核页呈现模式：点云恒灰 + 热力贴图；相机快照态随进入重置
        self._review_camera_snapshot = None
        self.render_service.set_review_mode(True)
        results = self.project_operation_service.last_facade_results or []
        self._show_facade_results_review(results)
        self._review_update_tool_buttons()
        controller = getattr(self, 'viewport_tools_controller', None)
        if controller is not None:
            controller.invalidate()
        self._workspace_in_review = True

    def _migrate_workspace_to_operation(self):
        """将 viewport_panel + right_dock 迁回项目操作页。"""
        if not getattr(self, '_workspace_in_review', False):
            return
        # 先关 review 模式再 cancel 工具：tool_deactivated 触发的
        # 贴图补画会因 review_mode 守卫短路，避免"先贴一次再摘"
        self.render_service.set_review_mode(False)
        # 视口迁出复核页前退出测距/靠尺工具，避免拾取回调挂在隐藏页面上
        controller = getattr(self, 'viewport_tools_controller', None)
        if controller is not None:
            controller.cancel()
        # 摘除热力贴图、恢复相机
        self._review_clear_presentation()
        self.operation_splitter.insertWidget(1, self.viewport_panel)
        self.operation_splitter.insertWidget(2, self.right_dock)
        # 占位 widget 挂回复核页，保持其默认引导界面可用
        self.review_splitter.insertWidget(0, self.review_viewport_placeholder)
        self.review_splitter.insertWidget(1, self.review_right_placeholder)
        self.review_splitter.setSizes([1040, 300])
        self.operation_splitter.setSizes(self._operation_splitter_sizes)
        self._hide_operation_placeholder()
        results = self.project_operation_service.last_facade_results or []
        self._show_facade_results(results)
        self._workspace_in_review = False

    def _show_operation_placeholder(self):
        """视口迁走后，操作页工作区显示引导提示。"""
        if not hasattr(self, '_operation_placeholder'):
            ph = QWidget()
            ph.setObjectName('operationPlaceholder')
            lay = QVBoxLayout(ph)
            lay.setAlignment(Qt.AlignCenter)
            lbl = QLabel('🔧 3D检测工作台已移至【检测复核】页面')
            lbl.setStyleSheet('font-size:16px; color:#64748B; font-weight:600;')
            lay.addWidget(lbl)
            btn = QPushButton('前往检测复核页')
            btn.setProperty('buttonRole', 'primary')
            btn.clicked.connect(lambda: self.set_current_page(2))
            lay.addWidget(btn, alignment=Qt.AlignCenter)
            self._operation_placeholder = ph
        self.operation_splitter.insertWidget(1, self._operation_placeholder)
        self.operation_splitter.setSizes([220, 1040, 0])

    def _hide_operation_placeholder(self):
        if hasattr(self, '_operation_placeholder'):
            self._operation_placeholder.setParent(None)

    # ------------------------------------------------------------------
    # 立面列表双版本
    # ------------------------------------------------------------------
    def _show_facade_results_review(self, results: list[dict]):
        """检测复核页：只显示 complete 立面，隐藏点数，【图片匹配】按钮。"""
        results = self.facade_quality_controller.process_facade_results(results)
        complete_results = [
            f for f in results
            if self._facade_review_status(f) == 'complete'
        ]
        count = len(complete_results)
        self.lbl_facade_summary.setText(f'已检测复核立面：{count}' if count else '未检测')
        if not complete_results:
            self.list_facades.clear()
            return
        self.list_facades.clear()
        for index, f in enumerate(complete_results, 1):
            display_no = int(f.get('display_no') or index)
            f['display_no'] = display_no
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setSizeHint(QSize(0, 40))
            self.list_facades.addItem(item)
            row = self._create_facade_list_row(f, display_no, review_mode=True)
            self.list_facades.setItemWidget(item, row)

    def _create_facade_list_row(self, f: dict, display_no: int, review_mode: bool):
        """创建立面列表行 widget。review_mode=True 时为检测复核模式。"""
        row = QWidget()
        row.setStyleSheet("""
            QWidget { background: transparent; }
            QLabel { font-size: 12px; color: #334155; }
        """)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 14, 6)
        row_layout.setSpacing(8)
        info = QLabel(f"立面{display_no}")
        info.setStyleSheet('font-size: 12px; color: #334155;')
        row_layout.addWidget(info, 1)
        color = self.render_facade.facade_color(f, display_no)
        swatch = QFrame()
        swatch.setFixedSize(18, 18)
        swatch.setStyleSheet(
            'background-color: rgb(%d,%d,%d); border: 1px solid #94a3b8; border-radius: 3px;' %
            tuple(int(max(0, min(1, x)) * 255) for x in color)
        )
        row_layout.addWidget(swatch)
        if f.get('__stale'):
            # 修订号失修（去噪后旧索引）：禁用操作，避免对错位结果继续评估
            info.setText(f"立面{display_no}（已失效）")
            info.setStyleSheet('font-size: 12px; color: #94A3B8;')
            action_button = QPushButton('已失效')
            action_button.setFixedWidth(84)
            action_button.setMinimumHeight(32)
            action_button.setEnabled(False)
        elif review_mode:
            action_button = QPushButton('图片匹配')
            action_button.setFixedWidth(84)
            action_button.setMinimumHeight(32)
            action_button.clicked.connect(
                lambda _=False, obj=f: self._on_facade_image_match(obj))
        else:
            status = self._facade_review_status(f)
            action_button = QPushButton('处理' if status == 'complete' else '标记处理')
            action_button.setFixedWidth(84)
            action_button.setMinimumHeight(32)
            action_button.setToolTip('点击确认立面状态；仅标记立面允许质量计算')
            action_button.clicked.connect(
                lambda _=False, obj=f, button=action_button:
                self._toggle_facade_review_status(obj, button))
        action_button.setStyleSheet("""
            QPushButton {
                font-size: 11px; padding: 2px 8px; border-radius: 4px;
                border: 1px solid #cbd5e1; background: #ffffff; color: #475569;
            }
            QPushButton:hover {
                background: #f1f5f9; border-color: #94a3b8;
            }
        """)
        row_layout.addWidget(action_button)
        row.setMaximumHeight(48)
        return row

    def _on_facade_image_match(self, facade: dict):
        """【图片匹配】按钮回调：委托给 PhotoMatchController 编排。"""
        self.photo_match_controller.start(facade)
