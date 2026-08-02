import copy
import json
import os
import re
import time
import uuid as uuid_module
from datetime import datetime

import numpy as np
import open3d as o3d

from ..config import Config
from ..core.geometry_utils import (
    to_zup, compute_registration_error, pcd_to_json
)
from ..core.cache import get_cache


def _safe_stem(filename):
    base = os.path.basename(filename or 'cloud.ply')
    stem, _ = os.path.splitext(base)
    stem = re.sub(r'[^\w.\-]+', '_', stem, flags=re.UNICODE)
    return stem or 'cloud'


def _voxel_tag(voxel_size):
    return f"{float(voxel_size):.6g}".replace('.', 'p')


def _registration_result_id(src_filename, tgt_filename, voxel_size):
    return f"{_safe_stem(src_filename)}__{_safe_stem(tgt_filename)}_v{_voxel_tag(voxel_size)}"


def _registration_paths(result_id):
    os.makedirs(Config.REGISTRATION_FOLDER, exist_ok=True)
    json_path = os.path.join(Config.REGISTRATION_FOLDER, f"{result_id}.json")
    ply_path = os.path.join(Config.REGISTRATION_FOLDER, f"{result_id}.ply")
    return json_path, ply_path


def register_correspondences(src_uuid, tgt_uuid, pairs):
    """批量记录点对"""
    cache = get_cache()
    processed_pairs = []

    for pair in pairs:
        src_pt_y = np.array(pair['src_point'])
        tgt_pt_y = np.array(pair['tgt_point'])
        src_pt_z = to_zup(src_pt_y)
        tgt_pt_z = to_zup(tgt_pt_y)
        processed_pairs.append({
            'src': src_pt_z,
            'tgt': tgt_pt_z
        })

    cache.set_reg_pairs(src_uuid, tgt_uuid, processed_pairs)
    return len(processed_pairs)


def apply_registration(src_uuid, tgt_uuid):
    """应用粗配准 (SVD)"""
    cache = get_cache()
    pairs = cache.get_reg_pairs(src_uuid, tgt_uuid)

    if len(pairs) < 3:
        raise ValueError(f'至少需要3组对应点，当前只有{len(pairs)}组')

    # 提取点对
    src_pts = np.array([p['src'] for p in pairs])
    tgt_pts = np.array([p['tgt'] for p in pairs])

    # SVD求解
    src_mean = np.mean(src_pts, axis=0)
    tgt_mean = np.mean(tgt_pts, axis=0)
    src_centered = src_pts - src_mean
    tgt_centered = tgt_pts - tgt_mean
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    # 从 ORIGINAL_DOWNSAMPLED 出发应用变换
    src_original = cache.get_original(src_uuid)
    src_transformed = copy.deepcopy(src_original)
    src_transformed.transform(transform)

    # 更新当前显示点云和累积变换
    cache.set_display(src_uuid, src_transformed)
    cache.set_transform(src_uuid, transform)

    # 计算误差
    tgt_pcd = cache.get_display(tgt_uuid)
    error_info = compute_registration_error(src_original, tgt_pcd, transform, threshold=1.0)

    # 清空配准点对
    cache.clear_reg_pairs(src_uuid, tgt_uuid)

    return {
        'transformation': transform.tolist(),
        'rmse': error_info['rmse'],
        'overlap_ratio': error_info['overlap_ratio']
    }


def icp_refine(src_uuid, tgt_uuid, initial_transform, voxel_size=0.05):
    """ICP 精配准"""
    cache = get_cache()

    src_original = cache.get_original(src_uuid)
    tgt_pcd = cache.get_display(tgt_uuid)

    if src_original is None:
        raise ValueError('源点云原始备份未找到，请重新上传')
    if tgt_pcd is None:
        raise ValueError('目标点云未找到')

    # 验证变换矩阵
    current_stored = cache.get_transform(src_uuid)
    if not np.allclose(initial_transform, current_stored, atol=1e-5):
        print(f"[WARN] 前端传入的变换矩阵与后端存储不一致，以后端为准")
        initial_transform = current_stored

    print(f"[INFO] ICP精配准: 源={src_uuid}, 目标={tgt_uuid}")
    print(f"[INFO] 源原始点数: {len(src_original.points)}, 目标点数: {len(tgt_pcd.points)}")

    # 第1级：粗尺度 ICP
    voxel_coarse = max(voxel_size * 6, 0.3)
    src_coarse = src_original.voxel_down_sample(voxel_coarse)
    tgt_coarse = tgt_pcd.voxel_down_sample(voxel_coarse)
    src_coarse.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse * 2, max_nn=30))
    tgt_coarse.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse * 2, max_nn=30))

    reg_coarse = o3d.pipelines.registration.registration_icp(
        src_coarse, tgt_coarse,
        max_correspondence_distance=voxel_coarse * 8,
        init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    )
    print(f"[INFO] 粗ICP: fitness={reg_coarse.fitness:.4f}, RMSE={reg_coarse.inlier_rmse:.4f}")

    # 第2级：中尺度 ICP
    voxel_medium = max(voxel_size * 2, 0.1)
    src_medium = src_original.voxel_down_sample(voxel_medium)
    tgt_medium = tgt_pcd.voxel_down_sample(voxel_medium)
    src_medium.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_medium * 2, max_nn=30))
    tgt_medium.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_medium * 2, max_nn=30))

    reg_medium = o3d.pipelines.registration.registration_icp(
        src_medium, tgt_medium,
        max_correspondence_distance=voxel_medium * 6,
        init=reg_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    )
    print(f"[INFO] 中ICP: fitness={reg_medium.fitness:.4f}, RMSE={reg_medium.inlier_rmse:.4f}")

    # 第3级：精细 ICP
    src_fine = src_original
    tgt_fine = tgt_pcd
    if len(src_fine.points) > 2000000:
        src_fine = src_fine.voxel_down_sample(voxel_size)
        tgt_fine = tgt_fine.voxel_down_sample(voxel_size)
        print(f"[INFO] 精细ICP前下采样: {len(src_fine.points)} 点")

    src_fine.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    tgt_fine.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    current_transform = reg_medium.transformation
    thresholds = [voxel_size * 6, voxel_size * 3, voxel_size * 1.5]
    reg_fine = None

    for i, thresh in enumerate(thresholds):
        print(f"[INFO] 精细ICP第{i + 1}轮: 阈值={thresh:.4f}m")
        reg_fine = o3d.pipelines.registration.registration_icp(
            src_fine, tgt_fine,
            max_correspondence_distance=thresh,
            init=current_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        current_transform = reg_fine.transformation
        print(f"[INFO]  结果: fitness={reg_fine.fitness:.4f}, RMSE={reg_fine.inlier_rmse:.4f}")
        if reg_fine.fitness < 0.01:
            print("[WARN] 拟合度过低，提前终止")
            break

    final_transform = current_transform
    print(f"[INFO] ICP最终变换矩阵:\n{final_transform}")

    # 应用最终变换到原始点云
    src_transformed = copy.deepcopy(src_original)
    src_transformed.transform(final_transform)
    cache.set_display(src_uuid, src_transformed)
    cache.set_transform(src_uuid, final_transform)

    # 计算误差
    error_info = compute_registration_error(
        src_original, tgt_pcd, final_transform, threshold=voxel_size * 5
    )

    return {
        'transformation': final_transform.tolist(),
        'fitness': float(reg_fine.fitness),
        'inlier_rmse': float(reg_fine.inlier_rmse),
        'rmse': error_info['rmse'],
        'overlap_ratio': error_info['overlap_ratio']
    }


def merge_clouds(uuid1, uuid2):
    """合并两个点云"""
    cache = get_cache()
    pcd1 = copy.deepcopy(cache.get_display(uuid1))
    pcd2 = copy.deepcopy(cache.get_display(uuid2))

    if not pcd1.has_colors() or is_uniform_color(np.asarray(pcd1.colors)):
        pcd1.paint_uniform_color([1.0, 0.4, 0.4])
    if not pcd2.has_colors() or is_uniform_color(np.asarray(pcd2.colors)):
        pcd2.paint_uniform_color([0.4, 0.4, 1.0])

    merged = pcd1 + pcd2
    new_uuid = f"merged_{uuid_module.uuid4().hex[:8]}"

    cache.set_display(new_uuid, merged)
    cache.set_original(new_uuid, copy.deepcopy(merged))
    cache.set_transform(new_uuid, np.eye(4))

    meta1 = cache.get_meta(uuid1)
    meta2 = cache.get_meta(uuid2)
    cache.set_meta(new_uuid, {
        'filename': f'merged_{meta1["filename"]}_{meta2["filename"]}',
        'point_count': len(merged.points)
    })

    return new_uuid, merged


def is_uniform_color(colors, eps=1e-4):
    if len(colors) == 0:
        return True
    r, g, b = colors[0]
    for i in range(1, len(colors)):
        if abs(colors[i][0] - r) > eps or abs(colors[i][1] - g) > eps or abs(colors[i][2] - b) > eps:
            return False
    return True


def save_registration_result(src_uuid, tgt_uuid, transformation=None, voxel_size=0.05,
                             metrics=None, also_save_download=True):
    """保存配准变换 JSON + 合并后的点云 PLY，便于下次直接加载。"""
    cache = get_cache()
    src_meta = cache.get_meta(src_uuid) or {}
    tgt_meta = cache.get_meta(tgt_uuid) or {}
    if cache.get_display(src_uuid) is None or cache.get_display(tgt_uuid) is None:
        raise ValueError('源或目标点云未找到')

    transform = np.asarray(
        transformation if transformation is not None else cache.get_transform(src_uuid),
        dtype=float
    )
    if transform.shape != (4, 4):
        raise ValueError('变换矩阵必须是 4x4')

    src_filename = src_meta.get('filename', src_uuid)
    tgt_filename = tgt_meta.get('filename', tgt_uuid)
    result_id = _registration_result_id(src_filename, tgt_filename, voxel_size)
    json_path, ply_path = _registration_paths(result_id)

    # 确保源点云已应用当前变换后再合并
    current = cache.get_transform(src_uuid)
    if not np.allclose(current, transform, atol=1e-6):
        src_original = cache.get_original(src_uuid)
        if src_original is None:
            raise ValueError('源点云原始备份未找到，请重新上传后再保存')
        src_aligned = copy.deepcopy(src_original)
        src_aligned.transform(transform)
        cache.set_display(src_uuid, src_aligned)
        cache.set_transform(src_uuid, transform)

    merged_uuid, merged = merge_clouds(src_uuid, tgt_uuid)
    ok = o3d.io.write_point_cloud(ply_path, merged)
    if not ok:
        raise RuntimeError(f'写入配准点云失败: {ply_path}')

    payload = {
        'id': result_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'source_filename': src_filename,
        'target_filename': tgt_filename,
        'source_uuid': src_uuid,
        'target_uuid': tgt_uuid,
        'voxel_size': float(voxel_size),
        'transformation': transform.tolist(),
        'point_count': int(len(merged.points)),
        'metrics': metrics or {},
        'ply_file': os.path.basename(ply_path),
        'json_file': os.path.basename(json_path),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    download_name = None
    download_path = None
    if also_save_download:
        download_name = f"registered_{result_id}_{int(time.time())}.ply"
        download_path = os.path.join(Config.UPLOAD_FOLDER, download_name)
        o3d.io.write_point_cloud(download_path, merged)

    data = pcd_to_json(merged)
    data.update({
        'uuid': merged_uuid,
        'filename': f'registered_{src_filename}_{tgt_filename}',
        'point_count': len(merged.points),
        'result_id': result_id,
        'json_path': json_path,
        'ply_path': ply_path,
        'download_filename': download_name,
        'download_path': download_path,
        'transformation': transform.tolist(),
    })
    print(f"[INFO] 配准结果已保存: {json_path}")
    return data


def list_registration_results():
    """列出已保存的配准结果。"""
    os.makedirs(Config.REGISTRATION_FOLDER, exist_ok=True)
    results = []
    for name in sorted(os.listdir(Config.REGISTRATION_FOLDER)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(Config.REGISTRATION_FOLDER, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            result_id = info.get('id') or os.path.splitext(name)[0]
            _, ply_path = _registration_paths(result_id)
            results.append({
                'id': result_id,
                'source_filename': info.get('source_filename'),
                'target_filename': info.get('target_filename'),
                'voxel_size': info.get('voxel_size'),
                'point_count': info.get('point_count'),
                'created_at': info.get('created_at'),
                'has_ply': os.path.isfile(ply_path),
                'metrics': info.get('metrics', {}),
            })
        except Exception as e:
            print(f"[WARN] 读取配准结果失败 {path}: {e}")
    results.sort(key=lambda x: x.get('created_at') or '', reverse=True)
    return results


def load_registration_result(result_id):
    """直接加载已保存的合并配准点云，无需重新手动配准。"""
    if not result_id:
        raise ValueError('缺少配准结果 ID')
    json_path, ply_path = _registration_paths(result_id)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f'未找到配准结果: {result_id}')
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f'未找到配准点云文件: {os.path.basename(ply_path)}')

    with open(json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    start = time.time()
    merged = o3d.io.read_point_cloud(ply_path)
    if len(merged.points) == 0:
        raise ValueError('配准点云为空')
    if not merged.has_colors():
        merged.paint_uniform_color([0.7, 0.7, 0.7])

    cache = get_cache()
    new_uuid = f"registered_{uuid_module.uuid4().hex[:8]}"
    cache.set_display(new_uuid, merged)
    cache.set_original(new_uuid, copy.deepcopy(merged))
    cache.set_transform(new_uuid, np.eye(4))
    filename = (
        f"registered_{info.get('source_filename', 'src')}_"
        f"{info.get('target_filename', 'tgt')}"
    )
    cache.set_meta(new_uuid, {
        'filename': filename,
        'point_count': len(merged.points),
        'registration_result_id': result_id,
    })

    data = pcd_to_json(merged)
    data.update({
        'uuid': new_uuid,
        'filename': filename,
        'point_count': len(merged.points),
        'result_id': result_id,
        'transformation': info.get('transformation'),
        'source_filename': info.get('source_filename'),
        'target_filename': info.get('target_filename'),
        'voxel_size': info.get('voxel_size'),
        'metrics': info.get('metrics', {}),
        'load_time_s': round(time.time() - start, 3),
    })
    print(f"[INFO] 已加载配准结果 {result_id}: {len(merged.points):,} 点")
    return data


def apply_saved_registration(src_uuid, tgt_uuid, result_id=None, transformation=None):
    """将已保存的变换应用到当前源点云（不重新采点）。"""
    cache = get_cache()
    if cache.get_display(src_uuid) is None or cache.get_display(tgt_uuid) is None:
        raise ValueError('源或目标点云未找到')

    info = {}
    if transformation is None:
        if not result_id:
            raise ValueError('需要 result_id 或 transformation')
        json_path, _ = _registration_paths(result_id)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f'未找到配准结果: {result_id}')
        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        transformation = info.get('transformation')

    transform = np.asarray(transformation, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError('变换矩阵必须是 4x4')

    src_original = cache.get_original(src_uuid)
    if src_original is None:
        raise ValueError('源点云原始备份未找到，请重新上传')

    src_transformed = copy.deepcopy(src_original)
    src_transformed.transform(transform)
    cache.set_display(src_uuid, src_transformed)
    cache.set_transform(src_uuid, transform)

    tgt_pcd = cache.get_display(tgt_uuid)
    error_info = compute_registration_error(src_original, tgt_pcd, transform, threshold=1.0)

    data = pcd_to_json(src_transformed)
    data.update({
        'uuid': src_uuid,
        'filename': cache.get_meta(src_uuid)['filename'],
        'point_count': len(src_transformed.points),
        'transformation': transform.tolist(),
        'rmse': error_info['rmse'],
        'overlap_ratio': error_info['overlap_ratio'],
        'result_id': result_id or info.get('id'),
    })
    return data
