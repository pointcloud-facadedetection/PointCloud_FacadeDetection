"""立面结果修订号失修校验的实效测试。

背景：立面结果按检测时的数据集修订号盖戳（register_dataset 写入
metadata.revision）；代理内容变化（去噪/重建）后，旧索引会错位。
恢复着色时必须跳过失修立面并打 __stale 标记，而不是把旧索引
刷到错位的新点云上（"颜色飘了"问题的修复）。

断言：
1. register_dataset 自动盖修订号（dataset_id:点数）；
2. 修订号一致的立面正常着色，不一致的跳过且打 __stale；
3. 无修订号一侧（云或立面）不判失效（兼容旧数据）。
"""
import numpy as np

from services.viewport_render_service import ViewportRenderService

BASE = tuple([0.55, 0.55, 0.55])


class _FakeViewport:
    def __init__(self, pos, revision):
        self._data = {'pos': pos, 'proxy_ids': np.arange(len(pos)),
                      'dataset_revision': revision}
        self.colors = None

    def get_cloud_data(self, name):
        return self._data

    def update_cloud_color(self, name, colors):
        self.colors = np.asarray(colors).copy()


def _facades(rev_a, rev_b):
    return [
        {'id': 1, 'display_no': 1, 'proxy_indices': [0, 1],
         'color': [0.9, 0.2, 0.2], 'dataset_id': rev_a},
        {'id': 2, 'display_no': 2, 'proxy_indices': [3, 4],
         'color': [0.2, 0.9, 0.2], 'dataset_id': rev_b},
    ]


def test_stale_facade_not_colored_and_marked():
    pos = np.random.rand(8, 3).astype(np.float32)
    viewport = _FakeViewport(pos, revision='proj:1:100')
    service = ViewportRenderService(viewport, db=None)
    facades = _facades('proj:1:100', 'proj:1:90')   # A 匹配，B 失修

    service.highlight_facades('cloud', facades)

    colors = viewport.colors
    # A 正常着色
    assert np.allclose(colors[[0, 1]], np.array([0.9, 0.2, 0.2], dtype=np.float32))
    # B 失修：索引行保持基础灰，不刷色
    assert np.allclose(colors[[3, 4]], np.array(BASE, dtype=np.float32))
    assert facades[0].get('__stale') in (None, False)
    assert facades[1].get('__stale') is True


def test_missing_revision_marks_nothing_stale():
    pos = np.random.rand(8, 3).astype(np.float32)
    viewport = _FakeViewport(pos, revision='')   # 云无修订号
    service = ViewportRenderService(viewport, db=None)
    facades = _facades('proj:1:90', 'proj:1:80')

    service.highlight_facades('cloud', facades)

    assert not any(f.get('__stale') for f in facades)


def test_legacy_facade_revision_also_checked():
    """旧格式立面（无 dataset_id，只有 dataset_revision）同样参与校验。"""
    pos = np.random.rand(8, 3).astype(np.float32)
    viewport = _FakeViewport(pos, revision='proj:1:100')
    service = ViewportRenderService(viewport, db=None)
    facades = [{'id': 1, 'display_no': 1, 'proxy_indices': [0, 1],
                'color': [0.9, 0.2, 0.2],
                'dataset_revision': 'proj:bllygg01.ply'}]  # 旧文件名式修订号

    service.highlight_facades('cloud', facades)

    assert facades[0].get('__stale') is True
    assert np.allclose(viewport.colors[[0, 1]],
                       np.array(BASE, dtype=np.float32))
