#!/usr/bin/env python3
"""
test_mapping.py

示例（PowerShell 用反引号续行）：
    python facadeDetection\\test_mapping.py `
        --pose  "D:\\PointCloud_FacadeDetection\\data\\projects\\cs-2\\cache\\bllygg01.fls\\pointclouds\\bllygg01.json" `
        --cloud "D:\\PointCloud_FacadeDetection\\data\\projects\\cs-2\\cache\\bllygg01.fls\\pointclouds\\bllygg01.ply" `
        --out   "D:\\PointCloud_FacadeDetection\\data\\cm_out" `
        --photo "D:\\PointCloud_FacadeDetection\\data\\projects\\cs-2\\cache\\photos\\bllygg01.ply.jpg"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 把项目根目录塞进 sys.path，方便直接跑
_THIS = Path(__file__).resolve()
for parent in _THIS.parents:
    if (parent / 'utils' / 'photo_matching.py').is_file():
        sys.path.insert(0, str(parent))
        break

from utils.photo_matching import (          # noqa: E402
    generate_cloud_mapping_image,
    estimate_photo_cloud_match,
)


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def parse_image_size(text: str) -> tuple[int, int]:
    w, h = text.lower().replace('*', 'x').split('x')
    return int(w), int(h)


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """读取点坐标及逐点 intensity；没有强度时返回 ``None``。"""
    ext = path.suffix.lower()
    if ext == '.npy':
        points = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
        return points, None
    if ext == '.npz':
        data = np.load(path)
        points = None
        for key in ('points', 'xyz', 'arr_0'):
            if key in data:
                points = np.asarray(data[key], dtype=np.float64).reshape(-1, 3)
                break
        if points is None:
            raise ValueError(f'{path}: npz 里找不到 points/xyz 数组')
        intensity = None
        for key in ('intensity', 'intensities', 'scalar_intensity', 'values'):
            if key in data:
                intensity = np.asarray(data[key], dtype=np.float32).reshape(-1)
                break
        if intensity is not None and len(intensity) != len(points):
            raise ValueError(f'{path}: intensity 数量与点数不一致')
        return points, intensity
    if ext in ('.ply', '.pcd'):
        import open3d as o3d
        try:
            cloud = o3d.t.io.read_point_cloud(str(path))
            points = cloud.point['positions'].numpy().astype(
                np.float64, copy=False
            )
            intensity = None
            names = [str(name) for name in cloud.point]
            for key in (
                'intensity',
                'intensities',
                'scalar_intensity',
                'Intensity',
            ):
                if key in names:
                    intensity = cloud.point[key].numpy().reshape(-1).astype(
                        np.float32
                    )
                    break
            if intensity is None and 'colors' in names:
                colors = cloud.point['colors'].numpy().astype(np.float32)
                if colors.ndim == 2 and colors.shape[1] >= 3:
                    intensity = (
                        0.299 * colors[:, 0]
                        + 0.587 * colors[:, 1]
                        + 0.114 * colors[:, 2]
                    )
        except (KeyError, RuntimeError, OSError, ValueError) as exc:
            print(f'[TEST] Tensor 点云读取失败（{exc}），使用 legacy reader')
            cloud = o3d.io.read_point_cloud(str(path))
            points = np.asarray(cloud.points, dtype=np.float64)
            intensity = None
            if cloud.has_colors():
                colors = np.asarray(cloud.colors, dtype=np.float32)
                intensity = (
                    0.299 * colors[:, 0]
                    + 0.587 * colors[:, 1]
                    + 0.114 * colors[:, 2]
                )
        if intensity is not None and len(intensity) != len(points):
            intensity = None
        return np.ascontiguousarray(points), intensity
    if ext == '.csv':
        data = np.loadtxt(path, delimiter=',', dtype=np.float64)
        data = np.atleast_2d(data)
        intensity = data[:, 3].astype(np.float32) if data.shape[1] >= 4 else None
        return data[:, :3], intensity
    if ext in ('.xyz', '.txt', '.pts'):
        data = np.loadtxt(path, dtype=np.float64)
        data = np.atleast_2d(data)
        intensity = data[:, 3].astype(np.float32) if data.shape[1] >= 4 else None
        return data[:, :3], intensity
    raise ValueError(f'不支持的点云格式：{ext}')


def save_rgb(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or '.png'
    ok, buf = cv2.imencode(suffix, bgr)
    if not ok:
        raise OSError(f'编码失败：{path}')
    buf.tofile(str(path))


# ----------------------------------------------------------------------
# 体素下采样（与 TwoDMatchingService._voxel_downsample 保持一致）
# ----------------------------------------------------------------------
def voxel_downsample(
    points: np.ndarray,
    values: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """体素下采样点坐标，并同步聚合每个点的 intensity。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if voxel_size <= 0 or len(points) == 0:
        return points, values
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        scale = offset = None
        if values is not None:
            intensity = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(intensity) != len(points):
                raise ValueError('intensity 数量与点数不一致')
            finite = np.isfinite(intensity)
            if finite.any():
                offset = float(np.min(intensity[finite]))
                peak = float(np.max(intensity[finite]))
                scale = peak - offset if peak > offset else 1.0
                gray = np.zeros(len(intensity), dtype=np.float64)
                gray[finite] = np.clip(
                    (intensity[finite] - offset) / scale, 0.0, 1.0
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    np.repeat(gray[:, None], 3, axis=1)
                )
        down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        down_points = np.asarray(down.points, dtype=np.float64)
        if scale is None or not down.has_colors():
            return down_points, None
        down_values = np.asarray(down.colors)[:, 0] * scale + offset
        return down_points, down_values.astype(np.float32)
    except Exception as exc:
        print(f'[TEST] Open3D 不可用（{exc}），改用 numpy 网格归并', flush=True)
        return _voxel_downsample_numpy(points, values, voxel_size)


def _voxel_downsample_numpy(
    points: np.ndarray,
    values: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """numpy 版体素下采样：每个体素保留离质心最近的点。"""
    if len(points) == 0:
        return points, values
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    # 用一维唯一编码，避免开 dict
    span = keys.max(axis=0) - keys.min(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    order = np.argsort(flat, kind='stable')
    flat_sorted = flat[order]
    # 分段边界
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], len(order))
    kept_indices = []
    pts_sorted = points[order]
    for s, e in zip(starts, ends):
        chunk = pts_sorted[s:e]
        if len(chunk) == 1:
            kept_indices.append(order[s])
            continue
        centroid = chunk.mean(axis=0)
        idx = int(np.argmin(np.sum((chunk - centroid) ** 2, axis=1)))
        kept_indices.append(order[s + idx])
    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    down_values = (
        None
        if values is None
        else np.asarray(values).reshape(-1)[kept_indices]
    )
    return np.asarray(points[kept_indices], dtype=np.float64), down_values


# ----------------------------------------------------------------------
# 诊断
# ----------------------------------------------------------------------
def print_match_summary(match) -> None:
    print('=' * 72)
    print(f'point_count={match.point_count}  '
          f'inlier_count={match.inlier_count}  '
          f'reprojection_rmse={match.reprojection_rmse_px:.2f}px')
    print(f'K =\n{np.round(match.camera_matrix, 2)}')
    print(f'R =\n{np.round(match.rotation_matrix, 4)}')
    print(f't = {np.round(match.translation_vector, 4)}')
    print(f'P_world_to_photo shape = {match.match_matrix.shape}')

    print('-' * 72)
    print('对应点（前 20 条）:')
    for i, c in enumerate(match.correspondences[:20], 1):
        vx, vy = c['view_point']
        px, py = c['image_point']
        ox, oy, oz = c['object_point']
        print(f'  #{i:2d} 视图({vx:8.3f},{vy:8.3f})  '
              f'照片({px:9.3f},{py:9.3f})  '
              f'world=({ox:7.3f},{oy:7.3f},{oz:7.3f})')

    if len(match.correspondences) >= 3:
        vp = np.array([c['view_point'] for c in match.correspondences])
        ip = np.array([c['image_point'] for c in match.correspondences])
        vp_span = vp.max(0) - vp.min(0)
        ip_span = ip.max(0) - ip.min(0)
        sx = ip_span[0] / max(vp_span[0], 1e-6)
        sy = ip_span[1] / max(vp_span[1], 1e-6)
        print('-' * 72)
        print(f'视图跨度 dx={vp_span[0]:.1f} dy={vp_span[1]:.1f}')
        print(f'照片跨度 dx={ip_span[0]:.1f} dy={ip_span[1]:.1f}')
        print(f'粗略缩放 sx={sx:.2f} sy={sy:.2f}')

        print('-' * 72)
        print('两两距离比例（视图/照片）—— 应基本一致，飘忽即有错配:')
        pairs = []
        for i in range(len(vp)):
            for j in range(i + 1, len(vp)):
                dv = np.linalg.norm(vp[i] - vp[j])
                di = np.linalg.norm(ip[i] - ip[j])
                ratio = dv / max(di, 1e-6)
                pairs.append((ratio, i + 1, j + 1))
        pairs.sort()
        for r, i, j in pairs[:12]:
            print(f'  ({i:2d},{j:2d})  dv/di = {r:.5f}')


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', required=True,
                        help='扫描仪位姿 JSON（含 transformToGlobal）')
    parser.add_argument('--cloud', required=True, help='3D 点云文件')
    parser.add_argument('--out', required=True, help='输出目录')
    parser.add_argument('--photo', default=None, help='可选原始照片')
    parser.add_argument('--image-size', default='1024x576',
                        help='映射图尺寸，宽x高，默认 1024x576')
    parser.add_argument('--no-crop', action='store_true',
                        help='不裁剪主体')
    parser.add_argument('--voxel', type=float, default=0.1,
                        help='体素下采样尺寸（米），默认 0.1')
    parser.add_argument('--no-downsample', action='store_true',
                        help='关闭体素下采样（默认开启）')
    args = parser.parse_args()

    pose_path = Path(args.pose).expanduser().resolve()
    cloud_path = Path(args.cloud).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[TEST] 位姿 : {pose_path}')
    print(f'[TEST] 点云 : {cloud_path}')
    print(f'[TEST] 输出 : {out_dir}')

    pose_json = json.loads(pose_path.read_text(encoding='utf-8-sig'))
    raw_points, raw_intensity = load_cloud(cloud_path)
    print(f'[TEST] 原始点数 {len(raw_points)}  '
          f'范围 {raw_points.min(0)} .. {raw_points.max(0)}')
    print(
        '[TEST] intensity '
        + (
            f'已读取，范围 {np.nanmin(raw_intensity):.3f}'
            f' .. {np.nanmax(raw_intensity):.3f}'
            if raw_intensity is not None
            else '未找到，将使用深度着色'
        )
    )

    # ---- 体素下采样 ----
    if args.no_downsample or args.voxel <= 0:
        points = raw_points
        intensity = raw_intensity
        print('[TEST] 跳过体素下采样')
    else:
        t0 = time.perf_counter()
        points, intensity = voxel_downsample(
            raw_points,
            raw_intensity,
            voxel_size=args.voxel,
        )
        dt = time.perf_counter() - t0
        keep = len(points) / max(len(raw_points), 1) * 100.0
        print(f'[TEST] 体素下采样 voxel={args.voxel:.3f}m  '
              f'{len(raw_points)} -> {len(points)} '
              f'({keep:.1f}%, {dt:.2f}s)')
        if len(points) < 100:
            print('[TEST] 警告：下采样后点数过少，可能影响自动取景')

    size = parse_image_size(args.image_size)
    t0 = time.perf_counter()
    mapping = generate_cloud_mapping_image(
        points,
        pose_json,
        colors=intensity,
        image_size=size,
        crop_subject=not args.no_crop,
    )
    print(f'[TEST] 映射图耗时 {time.perf_counter() - t0:.2f}s')

    map_path = out_dir / 'pointcloud_mapping.png'
    save_rgb(map_path, mapping.mapping_image)
    print(f'[TEST] 映射图保存：{map_path}  '
          f'shape={mapping.mapping_image.shape}')
    print(f'[TEST] 相机内参 K=\n{np.round(mapping.camera_matrix, 2)}')
    print(f'[TEST] 相机外参 E=\n{np.round(mapping.extrinsic, 4)}')

    # 顺手把深度图也导出一份
    depth = mapping.depth_image
    if depth.size:
        dvis = depth.astype(np.float32)
        finite = np.isfinite(dvis) & (dvis > 0)
        if finite.any():
            dmin, dmax = np.percentile(dvis[finite], (2, 98))
            dvis = np.clip((dvis - dmin) / max(dmax - dmin, 1e-6), 0, 1)
        dvis = (dvis * 255).astype(np.uint8)
        save_rgb(out_dir / 'pointcloud_mapping_depth.png',
                 cv2.applyColorMap(dvis, cv2.COLORMAP_TURBO))
        print(f'[TEST] 深度图保存：{out_dir / "pointcloud_mapping_depth.png"}')

    if not args.photo:
        return

    photo_path = Path(args.photo).expanduser().resolve()
    photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(photo_path)
    print(f'[TEST] 照片 {photo_path} shape={photo.shape}')

    try:
        t0 = time.perf_counter()
        match = estimate_photo_cloud_match(
            points,
            photo,
            mapping,
            mapping.camera_matrix,
            mapping.extrinsic,
            depth_image=mapping.depth_image,
            pixel_point_index=mapping.pixel_point_index,
        )
        print(f'[TEST] 匹配耗时 {time.perf_counter() - t0:.2f}s')
    except Exception as exc:
        print(f'[TEST] 匹配失败：{exc}')
        return

    print_match_summary(match)


if __name__ == '__main__':
    main()