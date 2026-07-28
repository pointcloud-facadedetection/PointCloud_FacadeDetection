"""根据手动标注的 2D-3D 点对估计相机内参与外参。"""

# OpenCV 的 Python API 由二进制扩展动态导出，Pylint 无法静态识别成员。
# pylint: disable=no-member

import math

import cv2
import numpy as np


def estimate_camera_matrix(image_width, image_height, horizontal_fov_deg=60.0):
    """按图像尺寸和假定水平视场角生成内参初值。"""
    width = float(image_width)
    height = float(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须大于 0")
    if not 10.0 <= float(horizontal_fov_deg) <= 150.0:
        raise ValueError("水平视场角应在 10° 到 150° 之间")

    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_points(object_points, image_points, minimum=4):
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("3D 点坐标必须是 N×3 数组")
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("2D 像素坐标必须是 N×2 数组")
    if len(object_points) != len(image_points):
        raise ValueError("2D 点与 3D 点数量不一致")
    if len(object_points) < minimum:
        raise ValueError(f"至少需要 {minimum} 对 2D-3D 匹配点")
    if not np.all(np.isfinite(object_points)) or not np.all(np.isfinite(image_points)):
        raise ValueError("匹配点中包含无效数值")
    return object_points, image_points


def _clamp_focal(focal, width, height, fallback):
    min_focal = max(width, height) * 0.15
    max_focal = max(width, height) * 10.0
    if not np.isfinite(focal) or not min_focal <= focal <= max_focal:
        return float(fallback)
    return float(focal)


def estimate_camera_matrix_from_correspondences(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg=60.0,
):
    """从单张照片的手动点对优化焦距，主点固定为图像中心。"""
    object_points, image_points = _validate_points(
        object_points, image_points, minimum=6
    )
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    initial_matrix = estimate_camera_matrix(width, height, horizontal_fov_deg)
    distortion = np.zeros((5, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [object_points.astype(np.float32)],
        [image_points.astype(np.float32)],
        (width, height),
        initial_matrix.copy(),
        distortion,
        flags=flags,
    )

    focal = _clamp_focal(
        float(camera_matrix[0, 0]),
        width,
        height,
        initial_matrix[0, 0],
    )
    camera_matrix[0, 0] = focal
    camera_matrix[1, 1] = focal

    centered = object_points - np.mean(object_points, axis=0)
    point_rank = int(np.linalg.matrix_rank(centered))
    warning = None
    if point_rank < 3:
        warning = (
            "3D 匹配点近似共面；内参仅作粗略估计，"
            "后续将尝试多种 PnP 策略求解外参。"
        )

    return {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion,
        "calibration_rms_px": float(rms),
        "method": "single_view_calibrateCamera_fixed_principal_aspect",
        "point_rank": point_rank,
        "warning": warning,
    }


def estimate_camera_matrix_with_radial_distortion(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg=60.0,
):
    """在固定主点/宽高比下估计焦距与 k1，改善广角/仰拍照片边缘区域。"""
    object_points, image_points = _validate_points(
        object_points, image_points, minimum=8
    )
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    initial_matrix = estimate_camera_matrix(width, height, horizontal_fov_deg)
    distortion = np.zeros((5, 1), dtype=np.float64)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [object_points.astype(np.float32)],
        [image_points.astype(np.float32)],
        (width, height),
        initial_matrix.copy(),
        distortion,
        flags=flags,
    )

    focal = _clamp_focal(
        float(camera_matrix[0, 0]),
        width,
        height,
        initial_matrix[0, 0],
    )
    camera_matrix[0, 0] = focal
    camera_matrix[1, 1] = focal
    k1 = float(distortion[0, 0])
    if not np.isfinite(k1) or abs(k1) > 1.5:
        k1 = 0.0
        distortion[0, 0] = 0.0

    return {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion,
        "calibration_rms_px": float(rms),
        "method": "single_view_calibrateCamera_k1",
        "point_rank": int(
            np.linalg.matrix_rank(object_points - np.mean(object_points, axis=0))
        ),
        "warning": None,
        "k1": k1,
    }


def _pnp_method_candidates():
    methods = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        methods.append(("IPPE", cv2.SOLVEPNP_IPPE))
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        methods.append(("SQPNP", cv2.SOLVEPNP_SQPNP))
    methods.extend([
        ("EPNP", cv2.SOLVEPNP_EPNP),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ])
    return methods


def _collect_intrinsic_candidates(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg,
    camera_matrix=None,
):
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    candidates = []

    zero_dist = np.zeros((5, 1), dtype=np.float64)

    if camera_matrix is not None:
        candidates.append((
            np.asarray(camera_matrix, dtype=np.float64),
            "provided",
            None,
            None,
            zero_dist,
        ))
        return candidates

    if len(object_points) >= 6:
        try:
            calibration = estimate_camera_matrix_from_correspondences(
                object_points,
                image_points,
                image_width,
                image_height,
                horizontal_fov_deg,
            )
            candidates.append((
                calibration["camera_matrix"],
                calibration["method"],
                calibration.get("warning"),
                calibration.get("calibration_rms_px"),
                calibration["distortion_coefficients"],
            ))
        except (ValueError, cv2.error):
            pass

    if len(object_points) >= 8:
        try:
            calibration = estimate_camera_matrix_with_radial_distortion(
                object_points,
                image_points,
                image_width,
                image_height,
                horizontal_fov_deg,
            )
            k1 = calibration.get("k1", 0.0)
            warning = None
            if abs(k1) > 0.05:
                warning = (
                    f"检测到径向畸变 k1={k1:.4f}；"
                    "已用于改善照片边缘/远距离区域对齐。"
                )
            candidates.append((
                calibration["camera_matrix"],
                calibration["method"],
                warning,
                calibration.get("calibration_rms_px"),
                calibration["distortion_coefficients"],
            ))
        except (ValueError, cv2.error):
            pass

    fov_guesses = sorted(set([
        float(horizontal_fov_deg),
        40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0,
    ]))
    for fov in fov_guesses:
        candidates.append((
            estimate_camera_matrix(width, height, fov),
            f"fov_guess_{fov:.0f}deg",
            "内参来自假定视场角，适用于手动标注粗略匹配。",
            None,
            zero_dist.copy(),
        ))

    return candidates


def _try_pnp_ransac(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    reproj_err,
    flag,
    iterations,
    confidence,
):
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            iterationsCount=int(iterations),
            reprojectionError=float(reproj_err),
            confidence=float(confidence),
            flags=int(flag),
        )
        if success and rvec is not None and tvec is not None:
            return rvec, tvec, inliers
    except cv2.error:
        pass
    return None, None, None


def _try_pnp_direct(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    flag,
):
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            flags=int(flag),
        )
        if success and rvec is not None and tvec is not None:
            return rvec, tvec, np.arange(len(object_points), dtype=int).reshape(-1, 1)
    except cv2.error:
        pass
    return None, None, None


def _evaluate_pose(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    distortion,
    inliers,
):
    inlier_indices = (
        inliers.reshape(-1).astype(int)
        if inliers is not None
        else np.arange(len(object_points), dtype=int)
    )
    if len(inlier_indices) < 3:
        return None

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    inlier_errors = errors[inlier_indices]
    inlier_rmse = float(np.sqrt(np.mean(np.square(inlier_errors))))
    overall_rmse = float(np.sqrt(np.mean(np.square(errors))))

    return {
        "rvec": rvec,
        "tvec": tvec,
        "inlier_indices": inlier_indices,
        "inlier_count": int(len(inlier_indices)),
        "inlier_rmse": inlier_rmse,
        "overall_rmse": overall_rmse,
        "errors": errors,
        "projected": projected,
    }


def _search_best_pose(
    object_points,
    image_points,
    intrinsic_candidates,
    distortion,
    reproj_thresholds,
    iterations,
    confidence,
    use_ransac=True,
):
    best = None
    best_meta = None
    best_score = None

    attempt = _try_pnp_ransac if use_ransac else _try_pnp_direct
    for (
        camera_matrix,
        intrinsic_method,
        intrinsic_warning,
        calibration_rms,
        candidate_distortion,
    ) in intrinsic_candidates:
        trial_distortion = np.asarray(candidate_distortion, dtype=np.float64).reshape(-1, 1)
        for method_name, flag in _pnp_method_candidates():
            thresholds = reproj_thresholds if use_ransac else [None]
            for reproj in thresholds:
                if use_ransac:
                    rvec, tvec, inliers = attempt(
                        object_points,
                        image_points,
                        camera_matrix,
                        trial_distortion,
                        reproj,
                        flag,
                        iterations,
                        confidence,
                    )
                else:
                    rvec, tvec, inliers = attempt(
                        object_points,
                        image_points,
                        camera_matrix,
                        trial_distortion,
                        flag,
                    )
                if rvec is None:
                    continue

                evaluated = _evaluate_pose(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    trial_distortion,
                    inliers,
                )
                if evaluated is None:
                    continue

                score = (
                    evaluated["inlier_count"],
                    -evaluated["inlier_rmse"],
                )
                if best_score is None or score > best_score:
                    best = evaluated
                    best_score = score
                    best_meta = {
                        "camera_matrix": camera_matrix,
                        "distortion_coefficients": trial_distortion,
                        "intrinsic_method": intrinsic_method,
                        "intrinsic_warning": intrinsic_warning,
                        "calibration_rms_px": calibration_rms,
                        "pnp_method": method_name,
                        "pnp_mode": "ransac" if use_ransac else "direct",
                        "reprojection_threshold_px": reproj,
                    }

    return best, best_meta


def _log_camera_pose(result):
    print("[INFO] ========== 2D-3D 相机位姿估计结果 ==========")
    print(f"[INFO] 内参方法: {result['intrinsic_method']}")
    print(
        f"[INFO] PnP 策略: {result.get('pnp_method', '?')} "
        f"({result.get('pnp_mode', '?')}, "
        f"阈值 {result.get('reprojection_threshold_px', '?')} px)"
    )
    k = result["camera_matrix"]
    print(
        "[INFO] 相机内参 K:\n"
        f"       [{k[0][0]:.4f}, {k[0][1]:.4f}, {k[0][2]:.4f}]\n"
        f"       [{k[1][0]:.4f}, {k[1][1]:.4f}, {k[1][2]:.4f}]\n"
        f"       [{k[2][0]:.4f}, {k[2][1]:.4f}, {k[2][2]:.4f}]"
    )
    print(f"[INFO] 畸变系数: {result['distortion_coefficients']}")
    r = result["rotation_matrix"]
    print(
        "[INFO] 旋转矩阵 R:\n"
        f"       [{r[0][0]:.6f}, {r[0][1]:.6f}, {r[0][2]:.6f}]\n"
        f"       [{r[1][0]:.6f}, {r[1][1]:.6f}, {r[1][2]:.6f}]\n"
        f"       [{r[2][0]:.6f}, {r[2][1]:.6f}, {r[2][2]:.6f}]"
    )
    t = result["translation_vector"]
    print(f"[INFO] 平移向量 T: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
    c = result["camera_center_world"]
    print(f"[INFO] 相机中心(世界坐标): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")
    print(
        f"[INFO] 重投影 RMSE: {result['reprojection_rmse_px']:.2f} px | "
        f"内点: {result['inlier_count']}/{result['point_count']}"
    )
    if result.get("intrinsic_warning"):
        print(f"[INFO] 提示: {result['intrinsic_warning']}")
    print("[INFO] ==========================================")


def solve_camera_pose(
    object_points,
    image_points,
    image_width,
    image_height,
    camera_matrix=None,
    distortion_coefficients=None,
    horizontal_fov_deg=60.0,
    reprojection_error_px=20.0,
    confidence=0.95,
    iterations=500,
):
    """多策略鲁棒估计 K 与外参 [R|T]，允许手动标注存在一定误差。"""
    object_points, image_points = _validate_points(object_points, image_points)

    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if distortion_coefficients is None
        else np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1, 1)
    )

    intrinsic_candidates = _collect_intrinsic_candidates(
        object_points,
        image_points,
        image_width,
        image_height,
        horizontal_fov_deg,
        camera_matrix=camera_matrix,
    )

    reproj_thresholds = sorted(set([
        float(reprojection_error_px),
        12.0, 20.0, 30.0, 45.0, 60.0, 80.0,
    ]))

    best, best_meta = _search_best_pose(
        object_points,
        image_points,
        intrinsic_candidates,
        distortion,
        reproj_thresholds,
        iterations,
        confidence,
        use_ransac=True,
    )
    if best is None:
        best, best_meta = _search_best_pose(
            object_points,
            image_points,
            intrinsic_candidates,
            distortion,
            reproj_thresholds,
            iterations,
            confidence,
            use_ransac=False,
        )

    if best is None or best_meta is None:
        raise ValueError(
            "PnP 求解失败，请检查 2D/3D 对应关系；"
            "可尝试增加分布更均匀的匹配点"
        )

    rotation, _ = cv2.Rodrigues(best["rvec"])
    extrinsic = np.hstack([rotation, best["tvec"].reshape(3, 1)])
    inlier_indices = best["inlier_indices"]
    inlier_errors = best["errors"][inlier_indices]
    best_distortion = best_meta["distortion_coefficients"]

    result = {
        "camera_matrix": best_meta["camera_matrix"].tolist(),
        "distortion_coefficients": best_distortion.reshape(-1).tolist(),
        "intrinsic_method": best_meta["intrinsic_method"],
        "intrinsic_warning": best_meta["intrinsic_warning"],
        "calibration_rms_px": best_meta["calibration_rms_px"],
        "pnp_method": best_meta["pnp_method"],
        "pnp_mode": best_meta["pnp_mode"],
        "reprojection_threshold_px": best_meta["reprojection_threshold_px"],
        "rotation_matrix": rotation.tolist(),
        "rotation_vector": best["rvec"].reshape(-1).tolist(),
        "translation_vector": best["tvec"].reshape(-1).tolist(),
        "extrinsic_matrix": extrinsic.tolist(),
        "projection_matrix": (best_meta["camera_matrix"] @ extrinsic).tolist(),
        "camera_center_world": (-rotation.T @ best["tvec"].reshape(3)).tolist(),
        "inlier_indices": inlier_indices.tolist(),
        "inlier_count": best["inlier_count"],
        "point_count": int(len(object_points)),
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(inlier_errors)))),
        "reprojection_mean_px": float(np.mean(inlier_errors)),
        "reprojection_max_px": float(np.max(inlier_errors)),
        "projected_points": best["projected"].tolist(),
        "coordinate_convention": "X_camera = R * X_world + T",
    }
    _log_camera_pose(result)
    return result


def refine_camera_pose_fixed_intrinsics(
    object_points,
    image_points,
    camera_matrix,
    rotation_matrix,
    translation_vector,
    distortion_coefficients=None,
):
    """固定内参，利用 3D-2D 角点对外参做微调（用于 2D 拖拽后位姿修正）。"""
    object_points, image_points = _validate_points(object_points, image_points, minimum=4)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if distortion_coefficients is None
        else np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1, 1)
    )

    rvec, _ = cv2.Rodrigues(rotation)
    tvec = translation.reshape(3, 1)
    success, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        camera_matrix,
        distortion,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("外参微调失败，请检查角点对应关系")

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points.astype(np.float64),
                image_points.astype(np.float64),
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        object_points.astype(np.float64),
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    extrinsic = np.hstack([rotation, tvec.reshape(3, 1)])

    return {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rotation_matrix": rotation.tolist(),
        "rotation_vector": rvec.reshape(-1).tolist(),
        "translation_vector": tvec.reshape(-1).tolist(),
        "extrinsic_matrix": extrinsic.tolist(),
        "projection_matrix": (camera_matrix @ extrinsic).tolist(),
        "camera_center_world": (-rotation.T @ tvec.reshape(3)).tolist(),
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_max_px": float(np.max(errors)),
        "projected_points": projected.tolist(),
        "point_count": int(len(object_points)),
        "inlier_count": int(len(object_points)),
        "intrinsic_method": "fixed_from_prior",
        "pnp_method": "ITERATIVE_refine",
        "pnp_mode": "fixed_intrinsic_refine",
        "coordinate_convention": "X_camera = R * X_world + T",
    }
