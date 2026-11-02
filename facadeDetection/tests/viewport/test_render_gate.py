"""渲染门控的实效验证：页面不可见时 GLFW 轮询与帧提交真实停止，恢复可见后补一帧。"""
from __future__ import annotations

import threading
import time

from view3d.open3d_adapter import Open3DAdapter


# ---------------------------------------------------------------------------
# 渲染门控：页面不可见时 GLFW 轮询与帧提交真实停止，恢复后补一帧
# ---------------------------------------------------------------------------
class _FakeVis:
    def __init__(self):
        self.poll_events_count = 0
        self.update_renderer_count = 0

    def poll_events(self):
        self.poll_events_count += 1

    def update_renderer(self):
        self.update_renderer_count += 1


class TestRenderGate:
    def _adapter(self):
        adapter = Open3DAdapter()
        adapter.vis = _FakeVis()
        adapter._owner_thread_id = threading.get_ident()
        return adapter

    def test_hidden_page_stops_polling_visible_page_resumes(self):
        adapter = self._adapter()
        adapter.set_render_enabled(False)
        adapter._last_event_poll_time = 0.0
        time.sleep(0.04)  # 超过 30Hz 轮询间隔，若未门控必然触发 poll_events
        t0 = time.perf_counter()
        for _ in range(10):
            assert adapter.poll() is False
        t_disabled = time.perf_counter() - t0
        assert adapter.vis.poll_events_count == 0      # GLFW 轮询真实停止
        assert adapter.vis.update_renderer_count == 0  # 帧提交真实停止

        adapter.set_render_enabled(True)  # 恢复时应挂起一次刷新
        assert adapter._render_pending is True
        adapter._last_event_poll_time = 0.0
        adapter._last_render_time = 0.0
        assert adapter.poll() is True
        assert adapter.vis.poll_events_count == 1
        assert adapter.vis.update_renderer_count == 1
        assert adapter._render_pending is False  # 恢复帧被真实提交而非丢弃

        # 节流未被破坏：30Hz 间隔内重复 poll 不再触发轮询
        assert adapter.poll() is False
        assert adapter.vis.poll_events_count == 1
        print(f'\n[perf] 10x gated poll = {t_disabled*1e3:.3f}ms, '
              f'glfw_calls_while_hidden=0')
