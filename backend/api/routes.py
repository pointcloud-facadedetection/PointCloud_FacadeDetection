"""
Flask API路由模块
"""
import os
import time
import uuid
import traceback
import numpy as np
import open3d as o3d
import copy
from flask import request, jsonify, render_template, send_from_directory

from ..core.cache import get_cache
from ..core.geometry_utils import pcd_to_json, to_zup, ensure_normals, plane_axes
from ..config import Config
from ..services.file import process_ply, save_cloud, denoise_cloud, compute_normals, get_bounding_box
from ..services.facade_detection import detect_facades
from ..services.segmentation import segment_selected_region
from ..services.registration import (
    register_correspondences, apply_registration, icp_refine, merge_clouds,
    save_registration_result, list_registration_results,
    load_registration_result, apply_saved_registration
)
from ..services.facade_result_store import (
    build_facade_detection_response,
    labels_from_facades,
    save_facade_detection,
    list_facade_detections,
    load_facade_detection,
)
from ..core.quality_assessment import compute_facade_quality
from facadeDetection.algorithms.photo_pointcloud_matching import (
    solve_camera_pose, create_photo_facade_heatmap,
    build_region_highlight_colors, resolve_photo_detection_region,
    project_world_points_to_image, snap_point_to_facade_plane,
    refine_camera_pose_fixed_intrinsics,
)


def register_routes(app):
    """注册所有API路由"""

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        """上传PLY文件"""
        files = request.files.getlist('files')
        voxel_size = float(request.form.get('voxel_size', 0.02))
        results = []
        cache = get_cache()

        for file in files:
            if not file.filename.lower().endswith('.ply'):
                continue

            unique_name = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, unique_name)
            file.save(filepath)

            try:
                display_pcd = process_ply(
                    filepath, voxel_size, original_filename=file.filename
                )

                # 同时设置 DISPLAY 和 ORIGINAL_DOWNSAMPLED
                cache.set_display(unique_name, display_pcd)
                cache.set_original(unique_name, copy.deepcopy(display_pcd))
                cache.set_transform(unique_name, np.eye(4))
                cache.set_meta(unique_name, {
                    'filename': file.filename,
                    'point_count': len(display_pcd.points),
                    'source_size': os.path.getsize(filepath),
                    'voxel_size': voxel_size,
                })

                data = pcd_to_json(display_pcd)
                data['point_count'] = len(display_pcd.points)
                data['filename'] = file.filename
                data['uuid'] = unique_name
                results.append(data)

            except Exception as e:
                results.append({'error': str(e), 'filename': file.filename})

        return jsonify(results)

    @app.route('/denoise', methods=['POST'])
    def denoise():
        """去噪处理"""
        body = request.get_json()
        uuid_name = body.get('uuid')
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404

        try:
            voxel_size = float(body.get('voxel_size', 0.02))
            method = body.get('method', 'radius')
            kwargs = {
                'radius': body.get('radius'),
                'min_neighbors': body.get('min_neighbors'),
                'nb_neighbors': body.get('nb_neighbors'),
                'std_ratio': body.get('std_ratio')
            }

            clean_pcd = denoise_cloud(uuid_name, voxel_size, method,
                                      **{k: v for k, v in kwargs.items() if v is not None})

            data = pcd_to_json(clean_pcd)
            data['uuid'] = uuid_name
            data['filename'] = cache.get_meta(uuid_name)['filename']
            data['point_count'] = len(clean_pcd.points)

            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/compute_normals', methods=['POST'])
    def compute_normals_api():
        body = request.get_json()
        uuid_name = body.get('uuid')
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404

        try:
            voxel_size = float(body.get('voxel_size', Config.DEFAULT_VOXEL_SIZE))
            pcd_work = compute_normals(uuid_name, voxel_size)

            data = pcd_to_json(pcd_work)
            data['uuid'] = uuid_name
            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/boundingbox', methods=['POST'])
    def boundingbox():
        body = request.get_json()
        uuid_name = body.get('uuid')

        try:
            result = get_bounding_box(uuid_name)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/register_correspondences', methods=['POST'])
    def register_correspondences_api():
        body = request.get_json()
        src_uuid = body['src_uuid']
        tgt_uuid = body['tgt_uuid']
        pairs = body.get('pairs', [])

        count = register_correspondences(src_uuid, tgt_uuid, pairs)
        return jsonify({'status': 'ok', 'count': count})

    @app.route('/apply_registration', methods=['POST'])
    def apply_registration_api():
        body = request.get_json()
        src_uuid = body['src_uuid']
        tgt_uuid = body['tgt_uuid']
        cache = get_cache()

        try:
            result = apply_registration(src_uuid, tgt_uuid)

            src_transformed = cache.get_display(src_uuid)
            data = pcd_to_json(src_transformed)
            data['uuid'] = src_uuid
            data['filename'] = cache.get_meta(src_uuid)['filename']
            data['transformation'] = result['transformation']
            data['rmse'] = result['rmse']
            data['overlap_ratio'] = result['overlap_ratio']
            data['point_count'] = len(src_transformed.points)

            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/icp_refine', methods=['POST'])
    def icp_refine_api():
        body = request.get_json()
        src_uuid = body['src_uuid']
        tgt_uuid = body['tgt_uuid']
        initial_transform = np.array(body.get('initial_transform', np.eye(4).tolist()))
        voxel_size = float(body.get('voxel_size', 0.05))
        cache = get_cache()

        try:
            result = icp_refine(src_uuid, tgt_uuid, initial_transform, voxel_size)

            src_transformed = cache.get_display(src_uuid)
            data = pcd_to_json(src_transformed)
            data['uuid'] = src_uuid
            data['filename'] = cache.get_meta(src_uuid)['filename']
            data['transformation'] = result['transformation']
            data['fitness'] = result['fitness']
            data['inlier_rmse'] = result['inlier_rmse']
            data['rmse'] = result['rmse']
            data['overlap_ratio'] = result['overlap_ratio']
            data['point_count'] = len(src_transformed.points)

            return jsonify(data)

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/merge_clouds', methods=['POST'])
    def merge_clouds_api():
        body = request.get_json()
        uuid1 = body.get('uuid1')
        uuid2 = body.get('uuid2')
        cache = get_cache()

        if cache.get_display(uuid1) is None or cache.get_display(uuid2) is None:
            return jsonify({'error': '点云未找到'}), 404

        try:
            new_uuid, merged = merge_clouds(uuid1, uuid2)

            data = pcd_to_json(merged)
            data['uuid'] = new_uuid
            data['filename'] = cache.get_meta(new_uuid)['filename']
            data['point_count'] = len(merged.points)

            return jsonify(data)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/save_registered', methods=['POST'])
    def save_registered():
        body = request.get_json()
        uuid_name = body.get('uuid')
        filename = body.get('filename', 'registered.ply')

        try:
            save_path, point_count = save_cloud(uuid_name, filename)
            return jsonify({
                'status': 'ok',
                'path': save_path,
                'points': point_count
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/save_registration_result', methods=['POST'])
    def save_registration_result_api():
        """保存配准变换与合并点云，供下次直接加载。"""
        body = request.get_json() or {}
        src_uuid = body.get('src_uuid')
        tgt_uuid = body.get('tgt_uuid')
        if not src_uuid or not tgt_uuid:
            return jsonify({'error': '需要 src_uuid 与 tgt_uuid'}), 400

        try:
            data = save_registration_result(
                src_uuid=src_uuid,
                tgt_uuid=tgt_uuid,
                transformation=body.get('transformation'),
                voxel_size=float(body.get('voxel_size', Config.DEFAULT_VOXEL_SIZE)),
                metrics=body.get('metrics'),
                also_save_download=bool(body.get('also_save_download', True)),
            )
            data['status'] = 'ok'
            return jsonify(data)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/list_registration_results', methods=['GET', 'POST'])
    def list_registration_results_api():
        try:
            return jsonify({
                'status': 'ok',
                'results': list_registration_results()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/load_registration_result', methods=['POST'])
    def load_registration_result_api():
        """直接加载已保存的合并配准点云。"""
        body = request.get_json() or {}
        result_id = body.get('result_id')
        if not result_id:
            return jsonify({'error': '缺少 result_id'}), 400
        try:
            data = load_registration_result(result_id)
            data['status'] = 'ok'
            return jsonify(data)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/apply_saved_registration', methods=['POST'])
    def apply_saved_registration_api():
        """对已上传的源/目标点云应用保存的变换矩阵。"""
        body = request.get_json() or {}
        src_uuid = body.get('src_uuid')
        tgt_uuid = body.get('tgt_uuid')
        if not src_uuid or not tgt_uuid:
            return jsonify({'error': '需要 src_uuid 与 tgt_uuid'}), 400
        try:
            data = apply_saved_registration(
                src_uuid=src_uuid,
                tgt_uuid=tgt_uuid,
                result_id=body.get('result_id'),
                transformation=body.get('transformation'),
            )
            data['status'] = 'ok'
            return jsonify(data)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/detect_facades', methods=['POST'])
    def detect_facades_api():
        body = request.get_json()
        uuid_name = body.get('uuid')
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404

        voxel_size = float(body.get('voxel_size', 0.02))
        min_area = float(body.get('min_area', 20.0))

        try:
            pcd = cache.get_display(uuid_name)
            pcd = ensure_normals(pcd, voxel_size, inplace=False, force=False, orient=False)
            cache.set_display(uuid_name, pcd)

            result = detect_facades(pcd, voxel_size=voxel_size, min_facade_area=min_area)
            facades = result['facades']

            points = np.asarray(pcd.points)
            original_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.7
            point_labels = labels_from_facades(facades, len(points))

            return jsonify(build_facade_detection_response(
                uuid_name, facades, point_labels, original_colors
            ) | {
                'facade_count': len(facades),
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/save_facade_detection', methods=['POST'])
    def save_facade_detection_api():
        body = request.get_json() or {}
        uuid_name = body.get('uuid')
        if not uuid_name:
            return jsonify({'error': '缺少点云 uuid'}), 400
        try:
            data = save_facade_detection(
                uuid_name,
                voxel_size=float(body.get('voxel_size', Config.DEFAULT_VOXEL_SIZE)),
                min_area=float(body.get('min_area', 5.0)),
            )
            return jsonify({'status': 'ok', **data})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/list_facade_detections', methods=['GET', 'POST'])
    def list_facade_detections_api():
        try:
            return jsonify({
                'status': 'ok',
                'results': list_facade_detections(),
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/load_facade_detection', methods=['POST'])
    def load_facade_detection_api():
        body = request.get_json() or {}
        uuid_name = body.get('uuid')
        result_id = body.get('result_id')
        if not uuid_name:
            return jsonify({'error': '缺少点云 uuid'}), 400
        try:
            data = load_facade_detection(result_id, uuid_name)
            return jsonify(data)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/segment_selection', methods=['POST'])
    def segment_selection_api():
        body = request.get_json()
        uuid_name = body.get('uuid')
        selected_indices = body.get('selected_indices', [])
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404
        if not selected_indices:
            return jsonify({'error': '未收到框选点索引'}), 400

        voxel_size = float(body.get('voxel_size', Config.DEFAULT_VOXEL_SIZE))

        try:
            pcd = cache.get_display(uuid_name)
            result = segment_selected_region(
                pcd,
                selected_indices=selected_indices,
                voxel_size=voxel_size,
                distance_threshold=body.get('distance_threshold'),
                normal_angle_deg=float(body.get('normal_angle_deg', 18.0)),
                dbscan_eps=body.get('dbscan_eps'),
                min_segment_points=int(body.get('min_segment_points', 80)),
                max_segments=int(body.get('max_segments', 30))
            )

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
            base_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.7

            point_labels = np.asarray(result['point_labels'], dtype=int)
            colors = base_colors.copy() * 0.25
            segment_by_id = {seg['id']: seg for seg in result['segments']}

            for i, label in enumerate(point_labels):
                if label >= 0 and label in segment_by_id:
                    colors[i] = np.asarray(segment_by_id[label]['color'], dtype=float)

            cache.set_segment_cache(uuid_name, {
                'segments': result['segments'],
                'point_labels': point_labels.tolist(),
                'base_colors': base_colors.tolist()
            })

            return jsonify({
                'status': 'ok',
                'uuid': uuid_name,
                'filename': cache.get_meta(uuid_name)['filename'],
                'segment_count': len(result['segments']),
                'segments': result['segments'],
                'point_labels': point_labels.tolist(),
                'positions': points.astype(np.float32).flatten().tolist(),
                'colors': colors.astype(np.float32).flatten().tolist(),
                'normals': normals.astype(np.float32).flatten().tolist(),
                'message': result.get('message', '')
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/get_facade_quality', methods=['POST'])
    def get_facade_quality_api():
        body = request.get_json()
        uuid_name = body.get('uuid')
        facade_id = body.get('facade_id')
        grid_size = body.get('grid_size', Config.QUALITY_GRID_SIZE)
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404

        if cache.get_facade_cache(uuid_name) is None:
            return jsonify({'error': '请先执行立面检测'}), 400

        facades_info = cache.get_facade_cache(uuid_name)['facades']
        target_facade = None
        for f in facades_info:
            if f['id'] == facade_id:
                target_facade = f
                break

        if target_facade is None:
            return jsonify({'error': f'平面ID {facade_id} 不存在'}), 404

        try:
            pcd = cache.get_display(uuid_name)
            points = np.asarray(pcd.points)

            plane_model = np.asarray(target_facade.get('plane_model', [0, 0, 1, 0]), dtype=float)
            normal = plane_model[:3]
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            d = float(plane_model[3])
            center = np.asarray(target_facade.get('center', np.mean(points, axis=0)), dtype=float)

            bbox = target_facade.get('bbox_2d') or {}
            u_axis = np.asarray(bbox.get('u_axis', []), dtype=float)
            v_axis = np.asarray(bbox.get('v_axis', []), dtype=float)
            if u_axis.size != 3 or v_axis.size != 3:
                u_axis, v_axis = plane_axes(normal, target_facade.get('type'))

            inlier_indices = np.asarray(target_facade.get('inlier_indices', []), dtype=int)
            inlier_indices = inlier_indices[(inlier_indices >= 0) & (inlier_indices < len(points))]
            if len(inlier_indices) > 0:
                facade_points = points[inlier_indices]
            else:
                local_u = np.dot(points - center, u_axis)
                local_v = np.dot(points - center, v_axis)
                plane_dist = np.abs(points @ normal + d)

                distance_threshold = float(body.get('distance_threshold', max(Config.DEFAULT_VOXEL_SIZE * 2.0, 0.08)))
                bbox_padding = float(body.get('bbox_padding', 0.5))

                u_min = float(bbox.get('u_min', np.min(local_u))) - bbox_padding
                u_max = float(bbox.get('u_max', np.max(local_u))) + bbox_padding
                v_min = float(bbox.get('v_min', np.min(local_v))) - bbox_padding
                v_max = float(bbox.get('v_max', np.max(local_v))) + bbox_padding

                mask = (
                        (plane_dist <= distance_threshold) &
                        (local_u >= u_min) & (local_u <= u_max) &
                        (local_v >= v_min) & (local_v <= v_max)
                )
                facade_points = points[mask]

            if len(facade_points) == 0:
                return jsonify({'error': '未提取到立面点'}), 400

            facade_pcd = o3d.geometry.PointCloud()
            facade_pcd.points = o3d.utility.Vector3dVector(facade_points)

            flatness_limit = float(body.get('flatness_limit', 0.004))
            verticality_limit_mm = float(body.get('verticality_limit_mm', 4.0))
            ruler_size = float(body.get('ruler_size', Config.RULER_SIZE))
            ruler_step = float(body.get('ruler_step', Config.RULER_STEP))

            start_time = time.time()
            quality = compute_facade_quality(
                target_facade,
                facade_pcd,
                grid_size=grid_size,
                flatness_limit=flatness_limit,
                verticality_limit_mm=verticality_limit_mm,
                ruler_size=ruler_size,
                ruler_step=ruler_step
            )
            compute_time = time.time() - start_time

            quality['source_point_count'] = len(points)
            quality['selected_point_count'] = len(facade_points)
            quality['compute_time_ms'] = int(compute_time * 1000)
            quality['note'] = '基于下采样点云计算，精度低于稠密点云'

            return jsonify({
                'status': 'ok',
                'facade_id': facade_id,
                'quality': quality
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/highlight_facade', methods=['POST'])
    def highlight_facade():
        body = request.get_json()
        uuid_name = body.get('uuid')
        facade_id = body.get('facade_id')
        cache = get_cache()

        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404
        if cache.get_facade_cache(uuid_name) is None:
            return jsonify({'error': '请先执行立面检测'}), 400

        try:
            cache_data = cache.get_facade_cache(uuid_name)
            point_labels = np.asarray(cache_data['point_labels'], dtype=int)
            facades = cache_data.get('facades', [])
            facade_by_id = {f['id']: f for f in facades}

            # 获取基础颜色（
            base_colors = np.asarray(cache_data.get('base_colors', []), dtype=float)
            if base_colors.shape[0] != len(point_labels):
                # 回退：从display pcd读取
                pcd = cache.get_display(uuid_name)
                base_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(point_labels), 3)) * 0.7

            type_dim_colors = Config.FACADE_DIM_COLORS
            highlight_color = np.array(Config.HIGHLIGHT_COLOR, dtype=float)
            colors = base_colors.copy() * 0.22  # 默认：所有点变暗

            # 非选中立面：使用类型暗色
            other_mask = (point_labels >= 0) & (point_labels != facade_id)
            other_labels = point_labels[other_mask]
            if len(other_labels) > 0:
                # 批量映射类型颜色
                type_colors = np.array([
                    type_dim_colors.get(facade_by_id.get(lid, {}).get('type'), np.array([0.18, 0.18, 0.18]))
                    for lid in other_labels
                ], dtype=float)
                colors[other_mask] = type_colors

            # 选中立面：高亮色
            highlight_mask = point_labels == facade_id
            colors[highlight_mask] = highlight_color

            return jsonify({
                'status': 'ok',
                'colors': colors.astype(np.float32).flatten().tolist(),
                'highlighted_facade_id': facade_id,
                'uuid': uuid_name,
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/match_photo_pointcloud', methods=['POST'])
    def match_photo_pointcloud():
        """根据手动 2D-3D 匹配点估计相机外参。"""
        body = request.get_json() or {}
        pairs = body.get('pairs', [])
        image_width = body.get('image_width')
        image_height = body.get('image_height')

        if len(pairs) < 6:
            return jsonify({'error': '估计相机内参至少需要 6 对完整的 2D-3D 匹配点'}), 400
        if not image_width or not image_height:
            return jsonify({'error': '缺少照片原始宽高'}), 400

        try:
            image_points = []
            object_points = []
            for pair in pairs:
                image_point = pair.get('image_point')
                cloud_point_yup = pair.get('cloud_point')
                if image_point is None or cloud_point_yup is None:
                    continue
                image_points.append(image_point)
                # Three.js 使用 Y-up；PnP 统一使用后端/Open3D 的 Z-up 坐标。
                object_points.append(to_zup(np.asarray(cloud_point_yup, dtype=float)))

            if len(object_points) < 6:
                return jsonify({'error': '有效匹配点少于 6 对，无法估计相机内参'}), 400

            camera_matrix = body.get('camera_matrix')
            distortion = body.get('distortion_coefficients')
            result = solve_camera_pose(
                object_points=object_points,
                image_points=image_points,
                image_width=float(image_width),
                image_height=float(image_height),
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion,
                horizontal_fov_deg=float(body.get('horizontal_fov_deg', 60.0)),
                reprojection_error_px=float(body.get('reprojection_error_px', 20.0)),
            )
            return jsonify({'status': 'ok', 'pose': result})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/project_world_points', methods=['POST'])
    def project_world_points_api():
        """将 3D 世界坐标点投影到照片像素坐标。"""
        body = request.get_json() or {}
        pose = body.get('pose') or {}
        points_yup = body.get('points', [])

        required = ('camera_matrix', 'rotation_matrix', 'translation_vector')
        if any(pose.get(key) is None for key in required):
            return jsonify({'error': '相机内参或外参不完整'}), 400
        if not isinstance(points_yup, list) or len(points_yup) == 0:
            return jsonify({'error': '缺少待投影的 3D 点'}), 400

        try:
            object_points = [
                to_zup(np.asarray(point, dtype=float))
                for point in points_yup
            ]
            result = project_world_points_to_image(
                object_points,
                np.asarray(pose['camera_matrix'], dtype=float),
                np.asarray(pose['rotation_matrix'], dtype=float),
                np.asarray(pose['translation_vector'], dtype=float),
                distortion_coefficients=pose.get('distortion_coefficients'),
            )
            return jsonify({'status': 'ok', **result})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/snap_point_to_facade_plane', methods=['POST'])
    def snap_point_to_facade_plane_api():
        body = request.get_json() or {}
        uuid_name = body.get('uuid')
        point_yup = body.get('point')
        cache = get_cache()
        if cache.get_display(uuid_name) is None:
            return jsonify({'error': '点云未找到'}), 404
        facade_cache = cache.get_facade_cache(uuid_name)
        if facade_cache is None or not facade_cache.get('facades'):
            return jsonify({'error': '请先对当前点云执行建筑立面检测'}), 400
        if point_yup is None:
            return jsonify({'error': '缺少 3D 点坐标'}), 400
        try:
            point_zup = to_zup(np.asarray(point_yup, dtype=float))
            result = snap_point_to_facade_plane(
                point_zup,
                facade_cache['facades'],
                facade_id=body.get('facade_id'),
                max_plane_dist=float(body.get('max_plane_dist', 4.0)),
            )
            snapped_yup = [
                float(result['snapped_point'][0]),
                float(result['snapped_point'][2]),
                float(-result['snapped_point'][1]),
            ]
            return jsonify({
                'status': 'ok',
                'snapped_point_yup': snapped_yup,
                'snapped_point_zup': result['snapped_point'],
                'facade_id': result['facade_id'],
                'plane_distance_m': result['plane_distance_m'],
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/refine_camera_pose_corners', methods=['POST'])
    def refine_camera_pose_corners_api():
        body = request.get_json() or {}
        pose = body.get('pose') or {}
        points_yup = body.get('points_3d') or body.get('points')
        image_points = body.get('image_points')

        required = ('camera_matrix', 'rotation_matrix', 'translation_vector')
        if any(pose.get(key) is None for key in required):
            return jsonify({'error': '相机内参或外参不完整'}), 400
        if not isinstance(points_yup, list) or len(points_yup) != 4:
            return jsonify({'error': '需要恰好 4 个 3D 角点'}), 400
        if not isinstance(image_points, list) or len(image_points) != 4:
            return jsonify({'error': '需要恰好 4 个 2D 角点'}), 400

        try:
            object_points = np.asarray([
                to_zup(np.asarray(point, dtype=float))
                for point in points_yup
            ], dtype=float)
            image_points = np.asarray(image_points, dtype=float)
            result = refine_camera_pose_fixed_intrinsics(
                object_points,
                image_points,
                np.asarray(pose['camera_matrix'], dtype=float),
                np.asarray(pose['rotation_matrix'], dtype=float),
                np.asarray(pose['translation_vector'], dtype=float),
                distortion_coefficients=pose.get('distortion_coefficients'),
            )
            return jsonify({'status': 'ok', 'pose': result})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/photo_highlight_region', methods=['POST'])
    def photo_highlight_region():
        """高亮照片检测四边形对应的 3D 点云区域。"""
        body = request.get_json() or {}
        uuid_name = body.get('uuid')
        pose = body.get('pose') or {}
        cache = get_cache()

        pcd = cache.get_display(uuid_name)
        if pcd is None:
            return jsonify({'error': '点云未找到'}), 404
        facade_cache = cache.get_facade_cache(uuid_name)
        if facade_cache is None or not facade_cache.get('facades'):
            return jsonify({'error': '请先对当前点云执行建筑立面检测'}), 400

        required = ('camera_matrix', 'rotation_matrix', 'translation_vector')
        if any(pose.get(key) is None for key in required):
            return jsonify({'error': '相机内参或外参不完整，请先估算相机内外参数'}), 400
        image_quadrilateral = body.get('image_quadrilateral')
        if not isinstance(image_quadrilateral, list) or len(image_quadrilateral) != 4:
            return jsonify({'error': '请在照片中标注恰好 4 个检测区域顶点'}), 400

        try:
            points = np.asarray(pcd.points)
            corners_3d = None
            raw_corners = body.get('corners_3d')
            if isinstance(raw_corners, list) and len(raw_corners) == 4:
                corners_3d = np.asarray([
                    to_zup(np.asarray(corner, dtype=float))
                    for corner in raw_corners
                ], dtype=float)
            region = resolve_photo_detection_region(
                points=points,
                facades=facade_cache['facades'],
                camera_matrix=np.asarray(pose['camera_matrix'], dtype=float),
                rotation=np.asarray(pose['rotation_matrix'], dtype=float),
                translation=np.asarray(pose['translation_vector'], dtype=float),
                image_quadrilateral=image_quadrilateral,
                distortion_coefficients=pose.get('distortion_coefficients'),
                corners_3d=corners_3d,
            )
            base_colors = np.asarray(
                facade_cache.get('base_colors', []), dtype=float
            )
            if base_colors.shape[0] != len(points):
                base_colors = (
                    np.asarray(pcd.colors)
                    if pcd.has_colors()
                    else np.ones((len(points), 3)) * 0.7
                )
            colors = build_region_highlight_colors(
                len(points),
                region['indices'],
                base_colors,
                Config.HIGHLIGHT_COLOR,
            )
            corners = region['corners_3d']
            print("[INFO] ========== 2D 顶点对应 3D 坐标 (Z-up) ==========")
            for index, corner in enumerate(corners, start=1):
                print(
                    f"[INFO] V{index}: "
                    f"[{corner[0]:.4f}, {corner[1]:.4f}, {corner[2]:.4f}]"
                )
            print(
                f"[INFO] 立面 #{int(region['facade']['id']) + 1} · "
                f"框内点数 {len(region['indices'])}"
            )
            print("[INFO] ================================================")
            return jsonify({
                'status': 'ok',
                'facade_id': int(region['facade']['id']),
                'highlight_count': int(len(region['indices'])),
                'facade_area_m2': region['area'],
                'corners_3d': corners.tolist(),
                'colors': colors.astype(np.float32).flatten().tolist(),
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/photo_facade_heatmap', methods=['POST'])
    def photo_facade_heatmap():
        """根据已求相机位姿选择照片内最大立面并生成网格热力图。"""
        body = request.get_json() or {}
        uuid_name = body.get('uuid')
        pose = body.get('pose') or {}
        cache = get_cache()

        pcd = cache.get_display(uuid_name)
        if pcd is None:
            return jsonify({'error': '点云未找到'}), 404
        facade_cache = cache.get_facade_cache(uuid_name)
        if facade_cache is None or not facade_cache.get('facades'):
            return jsonify({'error': '请先对当前点云执行建筑立面检测'}), 400

        required = ('camera_matrix', 'rotation_matrix', 'translation_vector')
        if any(pose.get(key) is None for key in required):
            return jsonify({'error': '相机内参或外参不完整，请先执行 2D-3D 匹配'}), 400
        image_quadrilateral = body.get('image_quadrilateral')
        if not isinstance(image_quadrilateral, list) or len(image_quadrilateral) != 4:
            return jsonify({'error': '请在照片中标注恰好 4 个立面顶点'}), 400

        try:
            corners_3d = None
            raw_corners = body.get('corners_3d')
            if isinstance(raw_corners, list) and len(raw_corners) == 4:
                corners_3d = [
                    to_zup(np.asarray(corner, dtype=float)).tolist()
                    for corner in raw_corners
                ]
            result = create_photo_facade_heatmap(
                points=np.asarray(pcd.points),
                facades=facade_cache['facades'],
                camera_matrix=pose['camera_matrix'],
                rotation=pose['rotation_matrix'],
                translation=pose['translation_vector'],
                image_width=float(body.get('image_width')),
                image_height=float(body.get('image_height')),
                grid_size=float(body.get('grid_size', 0.5)),
                image_quadrilateral=image_quadrilateral,
                distortion_coefficients=pose.get('distortion_coefficients'),
                corners_3d=corners_3d,
            )
            return jsonify({'status': 'ok', **result})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

    @app.route('/upload_photo', methods=['POST'])
    def upload_photo():
        """上传 2D 照片（jpg/png/bmp/webp）"""
        file = request.files.get('file') or request.files.get('photo')
        if file is None or not file.filename:
            return jsonify({'error': '未收到照片文件'}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in Config.ALLOWED_PHOTO_EXTENSIONS:
            return jsonify({
                'error': f'不支持的图片格式: {ext or "(无扩展名)"}，请上传 jpg/png/bmp/webp'
            }), 400

        os.makedirs(Config.PHOTO_FOLDER, exist_ok=True)
        photo_id = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(Config.PHOTO_FOLDER, photo_id)
        try:
            file.save(save_path)
            return jsonify({
                'status': 'ok',
                'photo_id': photo_id,
                'filename': file.filename,
                'url': f'/photo/{photo_id}',
                'size': os.path.getsize(save_path),
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/photo/<path:filename>')
    def serve_photo(filename):
        """预览已上传的 2D 照片"""
        try:
            return send_from_directory(Config.PHOTO_FOLDER, filename)
        except FileNotFoundError:
            return jsonify({'error': '照片不存在'}), 404

    @app.route('/download/<filename>')
    def download_file(filename):
        try:
            return send_from_directory(Config.UPLOAD_FOLDER, filename, as_attachment=True)
        except FileNotFoundError:
            return jsonify({'error': '文件不存在'}), 404