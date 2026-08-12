# -*- coding: utf-8 -*-
"""
导出点云 -> 距离分层下采样，独立预处理脚本。

把 FARO 导出的 E57（单站或多站，自动读取测站位姿合并）转成检测用的 PLY：
按"到最近测站的距离"裁剪，并做距离分层下采样——近处过采样区用大体素狠采、
远处稀疏区少采/不采，把各距离段密度拉平（均匀体素会让总点数被近场主导，
检测里按总点数比例的阈值会把远处目标全部挡掉）。

用法：
  python scripts/e57_stratified_downsample.py 站1.e57 站2.e57 ... -o merged.ply
  python scripts/e57_stratified_downsample.py cloud.ply --scan-origin 0 0 0 -o out.ply

输出：
  <out>.ply           合并下采样点云（强度写入颜色通道，p2-p98拉伸）
  <out>.stations.json 测站坐标 / 分层参数 / 各文件点数（供检测端传 scan_origin）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d

from backend.core.geometry_utils import estimate_point_ranges

DEFAULT_SHELLS = ((8.0, 0.15), (16.0, 0.10), (28.0, 0.07),
                  (45.0, 0.05), (70.0, 0.04), (100.0, 0.035))


def parse_shells(text):
    """'8:0.15,16:0.10,...' -> ((8,0.15),(16,0.10),...)；升序校验"""
    shells = []
    for item in text.split(','):
        hi, vox = item.split(':')
        shells.append((float(hi), float(vox)))
    if shells != sorted(shells):
        raise ValueError('shells 必须按距离升序')
    return tuple(shells)


def read_e57_scans(path):
    """逐个yield (points_project_frame, intensity, station)"""
    import pye57
    e = pye57.E57(path)
    for si in range(e.scan_count):
        header = e.get_header(si)
        station = np.asarray(header.translation, dtype=float)
        data = e.read_scan(si, ignore_missing_fields=True, transform=True,
                           colors=False, intensity=True, row_column=False)
        pts = np.column_stack([data['cartesianX'], data['cartesianY'],
                               data['cartesianZ']])
        inten = data.get('intensity')
        if inten is None:
            inten = np.full(len(pts), 0.5)
        yield pts, np.asarray(inten, dtype=np.float64), station


def read_generic(path):
    """PLY/PCD/XYZ：单块点云，无位姿"""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        inten = np.asarray(pcd.colors)[:, 0]
    else:
        inten = np.full(len(pts), 0.5)
    return pts, inten


def stratify(pts, inten, ranges, shells, keep_tail=True):
    """按距离壳层分别体素下采样，返回 (pts_ds, inten_ds)"""
    parts_p, parts_i = [], []
    lo = 0.0
    bounds = list(shells) + ([(np.inf, None)] if keep_tail else [])
    for hi, vox in bounds:
        m = (ranges >= lo) & (ranges < hi)
        lo = hi
        if not m.any():
            continue
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[m])
        g = np.clip(inten[m], 0, None)
        sub.colors = o3d.utility.Vector3dVector(np.stack([g, g, g], axis=1))
        if vox:
            sub = sub.voxel_down_sample(float(vox))
        parts_p.append(np.asarray(sub.points))
        parts_i.append(np.asarray(sub.colors)[:, 0])
    if not parts_p:
        return np.zeros((0, 3)), np.zeros(0)
    return np.vstack(parts_p), np.concatenate(parts_i)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='E57（自动读位姿）或 PLY/PCD/XYZ')
    ap.add_argument('-o', '--output', required=True, help='输出 PLY 路径')
    ap.add_argument('--crop', type=float, default=120.0,
                    help='保留距最近测站多少米内的点（默认120）')
    ap.add_argument('--min-range', type=float, default=0.5,
                    help='剔除距测站过近的点（默认0.5m，三脚架/扫描仪自身）')
    ap.add_argument('--shells', type=parse_shells, default=DEFAULT_SHELLS,
                    help='分层参数 "上限米:体素米,..."，默认 '
                         '8:0.15,16:0.10,28:0.07,45:0.05,70:0.04,100:0.035')
    ap.add_argument('--scan-origin', type=float, nargs=3, action='append',
                    metavar=('X', 'Y', 'Z'),
                    help='非E57输入的测站坐标（可重复给出多站）')
    args = ap.parse_args()

    # 第一遍：收集所有测站（裁剪与分层都要用全体测站的最近距离）
    stations = []
    e57_inputs = [p for p in args.inputs if p.lower().endswith('.e57')]
    other_inputs = [p for p in args.inputs if not p.lower().endswith('.e57')]
    if e57_inputs:
        import pye57
        for path in e57_inputs:
            e = pye57.E57(path)
            for si in range(e.scan_count):
                stations.append(np.asarray(e.get_header(si).translation, dtype=float))
    if args.scan_origin:
        stations.extend(np.asarray(s, dtype=float) for s in args.scan_origin)
    if not stations:
        stations = [np.zeros(3)]
        print('[warn] 无测站信息，假定扫描仪位于坐标原点', flush=True)
    stations = np.asarray(stations)
    print(f'stations: {len(stations)}', flush=True)
    for s in stations:
        print('  ', np.round(s, 3), flush=True)

    parts_p, parts_i, sources = [], [], []

    def process(pts, inten, label):
        r = estimate_point_ranges(pts, stations)
        keep = (r >= args.min_range) & (r <= args.crop)
        pts, inten, r = pts[keep], inten[keep], r[keep]
        p_ds, i_ds = stratify(pts, inten, r, args.shells)
        parts_p.append(p_ds)
        parts_i.append(i_ds)
        sources.append({'source': label, 'cropped': int(len(pts)),
                        'downsampled': int(len(p_ds))})
        print(f'[{label}] crop {len(pts):,} -> {len(p_ds):,}', flush=True)

    for path in e57_inputs:
        for si, (pts, inten, _st) in enumerate(read_e57_scans(path)):
            process(pts, inten, f'{os.path.basename(path)}#{si}')
    for path in other_inputs:
        pts, inten = read_generic(path)
        process(pts, inten, os.path.basename(path))

    pts = np.vstack(parts_p)
    inten = np.concatenate(parts_i)
    # 强度 p2-p98 拉伸后写入颜色
    p2, p98 = np.percentile(inten, 2), np.percentile(inten, 98)
    g = np.clip((inten - p2) / max(p98 - p2, 1e-9), 0, 1)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(np.stack([g, g, g], axis=1))
    o3d.io.write_point_cloud(args.output, out)

    meta_path = os.path.splitext(args.output)[0] + '.stations.json'
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump({
            'stations': [[float(x) for x in s] for s in stations],
            'crop_range_m': args.crop,
            'shells': [[float(h), float(v)] for h, v in args.shells],
            'total_points': int(len(pts)),
            'sources': sources,
        }, fh, ensure_ascii=False, indent=2)

    print(f'written {args.output} ({len(pts):,} pts) + {os.path.basename(meta_path)}',
          flush=True)


if __name__ == '__main__':
    main()
