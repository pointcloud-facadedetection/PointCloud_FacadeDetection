# -*- coding: utf-8 -*-
"""
单站TLS模拟验证：固定阈值管线（旧） vs 距离自适应管线（新）

模拟一台位于原点、高1.5m的扫描仪（角步距0.10°，约1/10分辨率档）：
  - 近墙：正对扫描仪，距离约8m
  - 远墙：距离32~79m，入射角55~72°（斜视，点间距被拉大到0.1~0.4m）
  - 地面：水平面
噪声沿测距方向 σ(r) = 1.5mm + 0.4mm/m，另加0.8%粗差点。

对比完整使用链路（与前端操作一致：上传下采样 -> 去噪 -> 检测）：
  旧链路：radius半径滤波(默认参数) -> 固定0.2m半径法向 -> 固定阈值检测
  新链路：adaptive距离分壳滤波     -> 自适应法向      -> 距离自适应检测

指标（按每面真值墙统计）：
  - keep    : 去噪后该墙真值点保留率（radius滤波会把远处稀疏点当离群点删掉）
  - recall  : 检测阶段召回率（相对去噪后存活点）
  - e2e     : 端到端召回 = keep × recall
  - frags   : 该墙被拆成几个检出面（理想=1）
  - angle   : 最大匹配面的法向角误差（度）
  - offset  : 真值墙中心到拟合平面的距离（毫米）

运行：py -3.11 tests/validate_range_adaptive.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import open3d as o3d

from backend.services.facade_detection import detect_facades
from backend.core.geometry_utils import adaptive_outlier_mask

RNG = np.random.default_rng(42)
SCANNER = np.array([0.0, 0.0, 1.5])
VOXEL = 0.02

# 真值面：point_on_plane, normal(朝向扫描仪), u_axis, u_range(局部坐标), v_axis, v_range
WALLS = {
    'near_wall': {
        'p0': np.array([0.0, 8.0, 0.0]),
        'n': np.array([0.0, -1.0, 0.0]),
        # 右边界收到 x=2，避免遮挡右侧远墙的视线
        'u': np.array([1.0, 0.0, 0.0]), 'u_rng': (-12.0, 2.0),
        'v': np.array([0.0, 0.0, 1.0]), 'v_rng': (0.0, 12.0),
    },
    'far_wall': {
        'p0': np.array([25.0, 20.0, 0.0]),
        'n': np.array([-1.0, 0.0, 0.0]),
        'u': np.array([0.0, 1.0, 0.0]), 'u_rng': (0.0, 55.0),  # y: 20~75m
        'v': np.array([0.0, 0.0, 1.0]), 'v_rng': (0.0, 18.0),
    },
    'ground': {
        'p0': np.array([0.0, 0.0, 0.0]),
        'n': np.array([0.0, 0.0, 1.0]),
        'u': np.array([1.0, 0.0, 0.0]), 'u_rng': (-90.0, 90.0),
        'v': np.array([0.0, 1.0, 0.0]), 'v_rng': (-90.0, 90.0),
    },
}
WALL_IDS = list(WALLS.keys())


def simulate_scan(step_deg=0.10, max_range=90.0):
    az = np.deg2rad(np.arange(-90.0, 90.0, step_deg))
    el = np.deg2rad(np.arange(-5.0, 40.0, step_deg))
    AZ, EL = np.meshgrid(az, el)
    dirs = np.stack([
        np.cos(EL) * np.sin(AZ),
        np.cos(EL) * np.cos(AZ),
        np.sin(EL),
    ], axis=-1).reshape(-1, 3)

    n_rays = len(dirs)
    best_t = np.full(n_rays, np.inf)
    best_label = np.full(n_rays, -1, dtype=int)

    for wi, name in enumerate(WALL_IDS):
        w = WALLS[name]
        denom = dirs @ w['n']
        valid = np.abs(denom) > 1e-9
        t = np.full(n_rays, -1.0)
        t[valid] = ((w['p0'] - SCANNER) @ w['n']) / denom[valid]
        hit = SCANNER + dirs * t[:, None]
        lu = (hit - w['p0']) @ w['u']
        lv = (hit - w['p0']) @ w['v']
        ok = (
            (t > 0.5) & (t < max_range) &
            (lu >= w['u_rng'][0]) & (lu <= w['u_rng'][1]) &
            (lv >= w['v_rng'][0]) & (lv <= w['v_rng'][1])
        )
        closer = ok & (t < best_t)
        best_t[closer] = t[closer]
        best_label[closer] = wi

    mask = best_label >= 0
    t = best_t[mask]
    d = dirs[mask]

    # 测距噪声随距离增长 + 0.8% 粗差
    sigma = 0.0015 + 0.0004 * t
    t_noisy = t + RNG.normal(0.0, sigma)
    gross = RNG.random(len(t)) < 0.008
    n_gross = int(np.sum(gross))
    t_noisy[gross] += RNG.uniform(0.3, 2.5, n_gross) * RNG.choice([-1, 1], n_gross)

    return SCANNER + d * t_noisy[:, None]


def gt_labels(points, dist_tol=0.18, bound_margin=0.3):
    """按到真值面的距离与边界给点打真值标签，-1为粗差/未归属。"""
    n = len(points)
    best = np.full(n, -1, dtype=int)
    best_dist = np.full(n, dist_tol)
    for wi, name in enumerate(WALL_IDS):
        w = WALLS[name]
        dist = np.abs((points - w['p0']) @ w['n'])
        lu = (points - w['p0']) @ w['u']
        lv = (points - w['p0']) @ w['v']
        ok = (
            (dist < best_dist) &
            (lu >= w['u_rng'][0] - bound_margin) & (lu <= w['u_rng'][1] + bound_margin) &
            (lv >= w['v_rng'][0] - bound_margin) & (lv <= w['v_rng'][1] + bound_margin)
        )
        best[ok] = wi
        best_dist[ok] = dist[ok]
    return best


def evaluate(facades, labels, points):
    """把检出立面匹配到真值墙并统计指标。"""
    report = {}
    for wi, name in enumerate(WALL_IDS):
        gt_idx = np.where(labels == wi)[0]
        if len(gt_idx) == 0:
            continue
        gt_mask = labels == wi
        w = WALLS[name]

        matched = []
        for f in facades:
            inl = np.asarray(f['inlier_indices'], dtype=int)
            overlap = int(np.sum(gt_mask[inl]))
            angle = np.degrees(np.arccos(np.clip(
                abs(float(np.dot(f['normal'], w['n']))), 0.0, 1.0)))
            if overlap >= 50 and angle < 12.0:
                matched.append((overlap, angle, f))

        if not matched:
            report[name] = dict(recall=0.0, frags=0, angle=float('nan'),
                                offset_mm=float('nan'), gt_points=len(gt_idx))
            continue

        captured = np.zeros(len(points), dtype=bool)
        for overlap, angle, f in matched:
            inl = np.asarray(f['inlier_indices'], dtype=int)
            captured[inl[gt_mask[inl]]] = True

        matched.sort(key=lambda x: -x[0])
        _, best_angle, best_f = matched[0]
        gt_center = np.mean(points[gt_idx], axis=0)
        pm = np.asarray(best_f['plane_model'])
        offset = abs(float(gt_center @ pm[:3] + pm[3])) * 1000.0

        report[name] = dict(
            recall=float(np.sum(captured)) / len(gt_idx),
            frags=len(matched),
            angle=best_angle,
            offset_mm=offset,
            gt_points=len(gt_idx),
        )
    return report


def denoise_radius_legacy(pcd):
    """旧链路默认去噪：radius滤波，radius=voxel*2=0.04, min_neighbors=10"""
    clean, _ = pcd.remove_radius_outlier(nb_points=10, radius=VOXEL * 2)
    if len(clean.points) < 0.1 * len(pcd.points):
        clean = pcd
    return clean


def denoise_adaptive_new(pcd):
    """新链路去噪：距离分壳自适应"""
    keep = adaptive_outlier_mask(pcd, scan_origin=SCANNER, std_ratio=2.5)
    return pcd.select_by_index(np.where(keep)[0].tolist())


def run_chain(base_pcd, base_counts, adaptive):
    tag = '新·距离自适应' if adaptive else '旧·固定阈值'

    # 1) 去噪
    pcd1 = denoise_adaptive_new(base_pcd) if adaptive else denoise_radius_legacy(base_pcd)
    labels1 = gt_labels(np.asarray(pcd1.points))
    keep_rate = {}
    for wi, name in enumerate(WALL_IDS):
        n_before = base_counts.get(name, 0)
        keep_rate[name] = (int(np.sum(labels1 == wi)) / n_before) if n_before else 0.0

    # 2) 法向（旧链路复现固定0.2m半径；ensure_normals 检测到已有法向会直接复用）
    if not adaptive:
        pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))

    # 3) 检测
    t0 = time.time()
    result = detect_facades(
        pcd1, voxel_size=VOXEL, min_facade_area=5.0,
        range_adaptive=adaptive,
        scan_origin=SCANNER if adaptive else None,
    )
    elapsed = time.time() - t0

    pts = np.asarray(pcd1.points)
    rep = evaluate(result['facades'], labels1, pts)

    print(f"\n[{tag}] 去噪后 {len(pts):,} 点 | 检出 {len(result['facades'])} 个面 | 检测耗时 {elapsed:.1f}s")
    out = {}
    for name in WALL_IDS:
        r = rep.get(name)
        if r is None:
            keep = keep_rate.get(name, 0.0)
            out[name] = dict(keep=keep, recall=0.0, e2e=0.0, frags=0,
                             angle=float('nan'), offset_mm=float('nan'))
            print(f"  {name:10s} keep={keep*100:6.1f}%  [去噪后无存活点或未检出]")
            continue
        keep = keep_rate[name]
        e2e = keep * r['recall']
        out[name] = dict(keep=keep, recall=r['recall'], e2e=e2e, frags=r['frags'],
                         angle=r['angle'], offset_mm=r['offset_mm'])
        print(f"  {name:10s} keep={keep*100:6.1f}%  recall={r['recall']*100:6.1f}%  "
              f"e2e={e2e*100:6.1f}%  frags={r['frags']}  "
              f"angle={r['angle']:.4f}deg  offset={r['offset_mm']:.2f}mm")
    return out


def main():
    print('模拟单站扫描 ...')
    points = simulate_scan()
    print(f'  原始点数: {len(points):,}')

    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(points)
    base_pcd = base_pcd.voxel_down_sample(VOXEL)  # 与上传流程一致
    base_labels = gt_labels(np.asarray(base_pcd.points))
    base_counts = {name: int(np.sum(base_labels == wi))
                   for wi, name in enumerate(WALL_IDS)}
    print(f'  下采样后: {len(base_pcd.points):,} 点 | 各墙真值点: ' +
          ', '.join(f"{k}={v:,}" for k, v in base_counts.items()))

    old = run_chain(base_pcd, base_counts, adaptive=False)
    new = run_chain(base_pcd, base_counts, adaptive=True)

    print('\n==== 远墙端到端对比（去噪×检测）====')
    of, nf = old['far_wall'], new['far_wall']
    print(f"keep  : {of['keep']*100:6.1f}% -> {nf['keep']*100:6.1f}%")
    print(f"recall: {of['recall']*100:6.1f}% -> {nf['recall']*100:6.1f}%")
    print(f"e2e   : {of['e2e']*100:6.1f}% -> {nf['e2e']*100:6.1f}%")
    print(f"frags : {of['frags']} -> {nf['frags']}")
    print(f"angle : {of['angle']:.4f} -> {nf['angle']:.4f} deg")
    print(f"offset: {of['offset_mm']:.2f} -> {nf['offset_mm']:.2f} mm")


if __name__ == '__main__':
    main()
