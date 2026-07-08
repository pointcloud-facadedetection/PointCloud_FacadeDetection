import copy
import numpy as np
import open3d as o3d

from ..core.geometry_utils import (
    to_zup, compute_registration_error
)
from ..core.cache import get_cache


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
    import uuid as uuid_module
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
