"""图像摆正测试：读取 algorithms/data，输出到 algorithms/data/guizheng。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

# 允许从 facadeDetection 根目录或 algorithms/test 直接运行
_FACADE_ROOT = Path(__file__).resolve().parents[2]
if str(_FACADE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FACADE_ROOT))

from algorithms.View_aligned_photo_pointcloud_matching import (  # noqa: E402
    PHOTO_SUFFIXES,
    rectify_photo_perspective,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
_OUTPUT_DIR = _DATA_DIR / 'guizheng'


def _write_bgr_image(path: Path, bgr) -> None:
    """写入 BGR 图像，兼容中文路径。"""
    ok, buf = cv2.imencode(path.suffix or '.jpg', bgr)
    if not ok:
        raise ValueError(f'图像编码失败: {path}')
    path.write_bytes(buf.tobytes())


def run_rectify_test(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    target_max_dim: int = 2048,
) -> list[dict]:
    """对 input_dir 下所有照片做摆正，保存到 output_dir。"""
    input_dir = (input_dir or _DATA_DIR).resolve()
    output_dir = (output_dir or _OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f'输入目录不存在: {input_dir}')

    results = []
    photos = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES
    )
    if not photos:
        raise FileNotFoundError(f'未在 {input_dir} 找到可处理的照片')

    for photo_path in photos:
        print(f'[rectify] {photo_path.name} ...', flush=True)
        result = rectify_photo_perspective(
            photo_path,
            target_max_dim=target_max_dim,
        )
        out_path = output_dir / photo_path.name
        _write_bgr_image(out_path, result['rectified_bgr'])
        summary = {
            'input': str(photo_path),
            'output': str(out_path),
            'method': result.get('method'),
            'original_size': result.get('original_size'),
            'rectified_size': result.get('rectified_size'),
        }
        results.append(summary)
        print(
            f"  -> {out_path.name}  {summary['original_size']} -> "
            f"{summary['rectified_size']}  method={summary['method']}",
            flush=True,
        )

    print(f'\n完成：共处理 {len(results)} 张，输出目录 {output_dir}', flush=True)
    return results


if __name__ == '__main__':
    run_rectify_test()
