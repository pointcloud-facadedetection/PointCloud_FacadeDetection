"""验证 SuperPoint + LightGlue 照片↔正射图匹配。

用法:
    python -m facadeDetection.algorithms.auto_photo_pointcloud_matching.check_matcher \\
        --photo path/to/photo.jpg \\
        --render path/to/ortho.png \\
        --output match_vis.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from algorithms.auto_photo_pointcloud_matching.facade_2D_matcher import (
    FacadeFeatureMatcher,
    _as_rgb_uint8,
)


def _load_render(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"正射图不存在: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取正射图: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_matches(
    photo_bgr: np.ndarray,
    render_rgb: np.ndarray,
    kpts_photo: np.ndarray,
    kpts_render: np.ndarray,
    max_draw: int = 80,
) -> np.ndarray:
    """拼接照片与正射图，绘制匹配点。"""
    img_p = photo_bgr.copy()
    img_r = cv2.cvtColor(_as_rgb_uint8(render_rgb), cv2.COLOR_RGB2BGR)

    n = min(len(kpts_photo), max_draw)
    for i in range(n):
        pt_p = (int(kpts_photo[i, 0]), int(kpts_photo[i, 1]))
        pt_r = (int(kpts_render[i, 0]), int(kpts_render[i, 1]))
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.circle(img_p, pt_p, 4, color, -1)
        cv2.circle(img_r, pt_r, 4, color, -1)

    h_max = max(img_p.shape[0], img_r.shape[0])
    img_p_resized = cv2.resize(
        img_p,
        (int(img_p.shape[1] * h_max / img_p.shape[0]), h_max),
        interpolation=cv2.INTER_AREA,
    )
    img_r_resized = cv2.resize(
        img_r,
        (int(img_r.shape[1] * h_max / img_r.shape[0]), h_max),
        interpolation=cv2.INTER_AREA,
    )
    return np.hstack([img_p_resized, img_r_resized])


def check_2D_matcher(
    photo_path,
    render_img,
    *,
    resize_max=1024,
    max_keypoints=1024,
    min_confidence=None,
    output_path=None,
    show_window=True,
) -> dict:
    matcher = FacadeFeatureMatcher(max_num_keypoints=max_keypoints)
    result = matcher.match(
        photo_path,
        render_img,
        resize_max=resize_max,
        min_confidence=min_confidence,
    )

    print(f"推理设备: {result['device']}")
    print(f"成功找到匹配点对数: {result['match_count']}")
    if result['match_count'] > 0:
        print(
            f"置信度: min={result['scores'].min():.3f}, "
            f"mean={result['scores'].mean():.3f}, max={result['scores'].max():.3f}"
        )

    photo_bgr = cv2.imread(str(photo_path))
    if photo_bgr is None:
        raise IOError(f"无法读取照片: {photo_path}")

    if result['match_count'] > 0:
        combined = visualize_matches(
            photo_bgr,
            render_img,
            result['kpts_photo'],
            result['kpts_render'],
        )
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), combined)
            print(f"可视化已保存: {out}")
        if show_window:
            cv2.imshow("LightGlue Matches (photo | ortho)", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="验证照片与正射图 2D 匹配")
    parser.add_argument('--photo', required=True, help='现场照片路径')
    parser.add_argument('--render', required=True, help='正射图 PNG/JPG 路径')
    parser.add_argument('--output', default='match_vis.png', help='可视化输出路径')
    parser.add_argument('--resize-max', type=int, default=1024)
    parser.add_argument('--max-keypoints', type=int, default=1024)
    parser.add_argument('--min-confidence', type=float, default=None)
    parser.add_argument('--no-window', action='store_true', help='不弹出 OpenCV 窗口')
    args = parser.parse_args(argv)

    render_rgb = _load_render(Path(args.render))
    check_2D_matcher(
        args.photo,
        render_rgb,
        resize_max=args.resize_max,
        max_keypoints=args.max_keypoints,
        min_confidence=args.min_confidence,
        output_path=args.output,
        show_window=not args.no_window,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
