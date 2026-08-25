"""步骤 4：基于 cv2.solvePnPRansac 的相机内外参估计。"""

from __future__ import annotations

# OpenCV Python 接口由二进制扩展动态导出。
# pylint: disable=no-member

import cv2
import numpy as np

from facadeDetection.algorithms.photo_pointcloud_matching.pnp_solver import estimate_camera_matrix


def _default_distortion(dist_coeffs=None):
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    return np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)


def _is_nearly_coplanar(pts_3d, ratio_tol=0.02):
    pts = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        return False
    centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-9:
        return True
    return float(s[-1] / s[0]) < ratio_tol


def _pnp_flag_candidates(object_points):
    flags = []
    if _is_nearly_coplanar(object_points) and hasattr(cv2, 'SOLVEPNP_IPPE'):
        flags.append(cv2.SOLVEPNP_IPPE)
    for name in ('SOLVEPNP_SQPNP', 'SOLVEPNP_EPNP', 'SOLVEPNP_ITERATIVE'):
        if hasattr(cv2, name):
            flag = getattr(cv2, name)
            if flag not in flags:
                flags.append(flag)
    return flags or [cv2.SOLVEPNP_ITERATIVE]


def _compute_reprojection(object_points, image_points, rvec, tvec, camera_matrix, dist):
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rvec,
        tvec,
        camera_matrix,
        dist,
    )
    projected = projected.reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return projected, errors


class FacadePoseEstimator:
    """PnP RANSAC + LM 精化，估计相机外参 [R|T]。"""

    def __init__(self, reproj_error=5.0, confidence=0.999, refine_lm=True):
        self.reproj_error = float(reproj_error)
        self.confidence = float(confidence)
        self.refine_lm = bool(refine_lm)

    def estimate(
        self,
        pts_2d_photo,
        pts_3d_world,
        img_shape,
        K=None,
        dist_coeffs=None,
        horizontal_fov_deg=60.0,
    ):
        """
        :param pts_2d_photo: 2D 照片像素 (N,2)
        :param pts_3d_world: 3D 世界坐标 (N,3)，Z-up
        :param img_shape: (height, width) 或 (h, w, ...)
        :return: (success, R, T, pose_info, msg)
        """
        h, w = int(img_shape[0]), int(img_shape[1])
        pts_2d = np.ascontiguousarray(pts_2d_photo, dtype=np.float32).reshape(-1, 2)
        pts_3d = np.ascontiguousarray(pts_3d_world, dtype=np.float32).reshape(-1, 3)

        if len(pts_2d) != len(pts_3d):
            return False, None, None, None, '2D 与 3D 点数量不一致'
        if len(pts_2d) < 4:
            return False, None, None, None, f'匹配点过少 ({len(pts_2d)} < 4)，无法求解 PnP'

        dist = _default_distortion(dist_coeffs)
        if K is None:
            fov_candidates = sorted(set([
                float(horizontal_fov_deg),
                40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
            ]))
            intrinsic_candidates = [
                (estimate_camera_matrix(w, h, fov), f'fov_guess_{fov:.0f}deg')
                for fov in fov_candidates
            ]
        else:
            intrinsic_candidates = [
                (np.asarray(K, dtype=np.float64), 'provided')
            ]

        best = None
        best_score = None
        last_err = 'PnP RANSAC 解算失败'

        for candidate_k, intrinsic_method in intrinsic_candidates:
            for flag in _pnp_flag_candidates(pts_3d):
                try:
                    ok, rv, tv, inl = cv2.solvePnPRansac(
                        objectPoints=pts_3d,
                        imagePoints=pts_2d,
                        cameraMatrix=candidate_k,
                        distCoeffs=dist,
                        reprojectionError=self.reproj_error,
                        confidence=self.confidence,
                        flags=flag,
                    )
                except cv2.error as exc:
                    last_err = str(exc)
                    continue
                if not ok or inl is None or len(inl) < 4:
                    last_err = f'PnP RANSAC 内点不足 (flag={flag})'
                    continue

                candidate_inliers = inl.flatten().astype(int)
                if self.refine_lm and len(candidate_inliers) >= 6:
                    try:
                        rv, tv = cv2.solvePnPRefineLM(
                            objectPoints=pts_3d[candidate_inliers],
                            imagePoints=pts_2d[candidate_inliers],
                            cameraMatrix=candidate_k,
                            distCoeffs=dist,
                            rvec=rv,
                            tvec=tv,
                        )
                    except cv2.error:
                        pass

                candidate_r, _ = cv2.Rodrigues(rv)
                camera_points = (
                    candidate_r @ pts_3d[candidate_inliers].T
                    + tv.reshape(3, 1)
                ).T
                positive_ratio = float(np.mean(camera_points[:, 2] > 0))
                if positive_ratio < 0.9:
                    continue
                projected, errors = _compute_reprojection(
                    pts_3d, pts_2d, rv, tv, candidate_k, dist
                )
                inlier_errors = errors[candidate_inliers]
                rmse = float(np.sqrt(np.mean(np.square(inlier_errors))))
                score = (len(candidate_inliers), -rmse)
                if best_score is None or score > best_score:
                    best_score = score
                    best = {
                        'K': candidate_k,
                        'intrinsic_method': intrinsic_method,
                        'rvec': rv,
                        'tvec': tv,
                        'inliers': candidate_inliers,
                        'R': candidate_r,
                        'projected': projected,
                        'errors': errors,
                        'positive_depth_ratio': positive_ratio,
                        'flag': flag,
                    }

        if best is None:
            hint = (
                f'共 {len(pts_2d)} 对 2D-3D 对应点，RANSAC 未找到 ≥4 个一致内点。'
                '常见原因：GlueStick 线段端点 3D 不准、相机 FOV 猜测偏差、'
                '或正射图过窄导致 2D 匹配质量差。可尝试「矫正照片视角」或调大 horizontal_fov_deg。'
            )
            return False, None, None, None, f'{last_err}；{hint}'

        K = best['K']
        rvec = best['rvec']
        tvec = best['tvec']
        used_flag = best['flag']
        inlier_indices = best['inliers']
        inlier_pts_2d = pts_2d[inlier_indices]
        inlier_pts_3d = pts_3d[inlier_indices]
        R = best['R']
        projected = best['projected']
        errors = best['errors']
        inlier_errors = errors[inlier_indices]

        pose_info = {
            'R': R,
            'T': tvec,
            'rvec': rvec,
            'K': K,
            'distortion_coefficients': dist,
            'inlier_indices': inlier_indices,
            'inliers_count': int(len(inlier_indices)),
            'total_count': int(len(pts_2d)),
            'inlier_ratio': float(len(inlier_indices) / len(pts_2d)),
            'inlier_pts_2d': inlier_pts_2d,
            'inlier_pts_3d': inlier_pts_3d,
            'projected_points': projected,
            'reprojection_errors_px': errors,
            'inlier_reprojection_errors_px': inlier_errors,
            'pnp_flag': int(used_flag) if used_flag is not None else None,
            'reproj_error_threshold_px': self.reproj_error,
            'intrinsic_method': best['intrinsic_method'],
            'positive_depth_ratio': best['positive_depth_ratio'],
        }

        msg = (
            f'解算成功！有效内点 {len(inlier_indices)} / {len(pts_2d)} '
            f'(内点率 {pose_info["inlier_ratio"] * 100:.1f}%)'
        )
        return True, R, tvec, pose_info, msg


def estimate_camera_pose_ransac(
    object_points,
    image_points,
    image_width,
    image_height,
    camera_matrix=None,
    distortion_coefficients=None,
    horizontal_fov_deg=60.0,
    reproj_error=5.0,
    confidence=0.999,
):
    """
    函数式入口：2D-3D 对应点 → API 兼容位姿字典。

    与手动匹配 `solve_camera_pose` 返回字段对齐，供前端/后端复用。
    """
    estimator = FacadePoseEstimator(
        reproj_error=reproj_error,
        confidence=confidence,
    )
    success, rotation, tvec, info, msg = estimator.estimate(
        image_points,
        object_points,
        (int(image_height), int(image_width)),
        K=camera_matrix,
        dist_coeffs=distortion_coefficients,
        horizontal_fov_deg=float(horizontal_fov_deg),
    )
    if not success:
        raise ValueError(msg)

    K = info['K']
    dist = info['distortion_coefficients']
    rvec = info['rvec']
    extrinsic = np.hstack([rotation, tvec.reshape(3, 1)])
    inlier_indices = info['inlier_indices']
    inlier_errors = info['inlier_reprojection_errors_px']

    return {
        'camera_matrix': K.tolist(),
        'distortion_coefficients': dist.reshape(-1).tolist(),
        'intrinsic_method': info.get('intrinsic_method', 'provided'),
        'intrinsic_warning': None,
        'calibration_rms_px': float(np.sqrt(np.mean(np.square(inlier_errors)))),
        'pnp_method': 'solvePnPRansac',
        'pnp_mode': 'ransac+refineLM' if estimator.refine_lm else 'ransac',
        'reprojection_threshold_px': float(reproj_error),
        'rotation_matrix': rotation.tolist(),
        'rotation_vector': rvec.reshape(-1).tolist(),
        'translation_vector': tvec.reshape(-1).tolist(),
        'extrinsic_matrix': extrinsic.tolist(),
        'projection_matrix': (K @ extrinsic).tolist(),
        'camera_center_world': (-rotation.T @ tvec.reshape(3)).tolist(),
        'inlier_indices': inlier_indices.tolist(),
        'inlier_count': int(info['inliers_count']),
        'point_count': int(info['total_count']),
        'inlier_ratio': float(info['inlier_ratio']),
        'positive_depth_ratio': float(info.get('positive_depth_ratio', 1.0)),
        'reprojection_rmse_px': float(np.sqrt(np.mean(np.square(inlier_errors)))),
        'reprojection_mean_px': float(np.mean(inlier_errors)),
        'reprojection_max_px': float(np.max(inlier_errors)),
        'projected_points': info['projected_points'].tolist(),
        'coordinate_convention': 'X_camera = R * X_world + T',
        'pnp_message': msg,
    }


__all__ = ['FacadePoseEstimator', 'estimate_camera_pose_ransac']
