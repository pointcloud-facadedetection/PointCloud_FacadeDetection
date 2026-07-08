import os
import copy
import json
import struct
import numpy as np
import open3d as o3d
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- 全局缓存 ----------
CACHED_CLOUDS = {}  # uuid -> o3d.geometry.PointCloud
ORIGINAL_DOWNSAMPLED = {}   # uuid -> 原始下采样点云
CLOUD_META = {}  # uuid -> {'filename': str}
REG_PAIRS = {}  # src_uuid -> { tgt_uuid -> [{'src': ndarray, 'tgt': ndarray}, ...] }
DEFAULT_VOXEL_SIZE = 0.05


def process_ply(filepath, voxel_size):
    """加载并预处理PLY文件"""
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError("点云为空")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def pcd_to_binary(pcd, **meta):
    """
    将Open3D点云序列化为二进制响应（Z-up坐标系）。

    布局（小端）:
        [uint32 头长度][JSON头，空格补齐到4字节对齐]
        [positions: N*3 float32]
        [normals: N*3 float32]（仅当 has_normals 时存在）
        [colors: N*3 uint8]
    JSON头包含 point_count / has_normals 以及调用方传入的元数据
    （uuid、filename、transformation、rmse 等）。
    """
    points = np.ascontiguousarray(np.asarray(pcd.points), dtype=np.float32)
    n = len(points)
    has_normals = pcd.has_normals()

    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    else:
        colors = np.full((n, 3), 178, dtype=np.uint8)  # 默认灰色 0.7

    header = {'point_count': n, 'has_normals': bool(has_normals)}
    header.update(meta)
    header_bytes = json.dumps(header).encode('utf-8')
    # 补齐到4字节对齐，前端可直接在buffer上建立Float32Array视图
    pad = (-len(header_bytes)) % 4
    header_bytes += b' ' * pad

    parts = [struct.pack('<I', len(header_bytes)), header_bytes, points.tobytes()]
    if has_normals:
        normals = np.ascontiguousarray(np.asarray(pcd.normals), dtype=np.float32)
        parts.append(normals.tobytes())
    parts.append(np.ascontiguousarray(colors).tobytes())

    return Response(b''.join(parts), mimetype='application/octet-stream')


# ---------- 坐标转换 ----------
def to_zup(p_yup):
    """前端 Y-up (x,y,z) -> 后端 Z-up (x, -z, y)"""
    return np.array([p_yup[0], -p_yup[2], p_yup[1]])


def to_yup(p_zup):
    """后端 Z-up (x,y,z) -> 前端 Y-up (x, z, -y)"""
    return np.array([p_zup[0], p_zup[2], -p_zup[1]])


def transform_to_yup(positions_z, normals_z=None):
    """批量将Z-up坐标转换为Y-up"""
    pos_z = np.array(positions_z).reshape(-1, 3)
    pos_y = np.array([to_yup(p) for p in pos_z]).flatten().tolist()

    result = {'positions': pos_y}

    if normals_z is not None and len(normals_z) > 0:
        norm_z = np.array(normals_z).reshape(-1, 3)
        # 法向量是方向向量，使用相同的旋转
        norm_y = np.array([to_yup(p) for p in norm_z]).flatten().tolist()
        result['normals'] = norm_y

    return result


# ---------- 配准评估 ----------
def compute_registration_error(src_pcd, tgt_pcd, transformation, threshold=1.0):
    """计算配准后的RMSE和重叠率"""
    # 使用深拷贝避免修改原始点云
    src_transformed = copy.deepcopy(src_pcd)
    src_transformed.transform(transformation)

    # 计算点到点距离
    dists = src_transformed.compute_point_cloud_distance(tgt_pcd)
    dists = np.asarray(dists)

    # RMSE（只考虑距离小于阈值的点）
    inlier_dists = dists[dists < threshold]
    rmse = np.sqrt(np.mean(inlier_dists ** 2)) if len(inlier_dists) > 0 else float('inf')

    # 重叠率
    overlap_ratio = len(inlier_dists) / len(dists) if len(dists) > 0 else 0

    return {
        'rmse': float(rmse),
        'overlap_ratio': float(overlap_ratio),
        'inlier_count': int(len(inlier_dists)),
        'total_count': int(len(dists))
    }


# ---------- 路由 ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """单文件上传，返回二进制点云（前端逐个文件请求）"""
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': '缺少文件'}), 400
    if not file.filename.lower().endswith('.ply'):
        return jsonify({'error': '仅支持PLY文件', 'filename': file.filename}), 400

    voxel_size = float(request.form.get('voxel_size', 0.05))
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    try:
        pcd = process_ply(filepath, voxel_size)
        ORIGINAL_DOWNSAMPLED[unique_name] = copy.deepcopy(pcd)
        CACHED_CLOUDS[unique_name] = pcd
        CLOUD_META[unique_name] = {'filename': file.filename}
        return pcd_to_binary(pcd, uuid=unique_name, filename=file.filename)
    except Exception as e:
        return jsonify({'error': str(e), 'filename': file.filename}), 500


@app.route('/denoise', methods=['POST'])
def denoise():
    body = request.get_json()
    uuid_name = body.get('uuid')
    if uuid_name not in CACHED_CLOUDS:
        return jsonify({'error': '点云未缓存'}), 404

    pcd = CACHED_CLOUDS[uuid_name]
    voxel_size = body.get('voxel_size', 0.05)
    method = body.get('method', 'radius')          # 'radius' 或 'statistical'
    radius = body.get('radius', voxel_size * 2)    # 半径滤波器半径
    min_neighbors = body.get('min_neighbors', 10)  # 半径内最少邻居数
    nb_neighbors = body.get('nb_neighbors', 20)    # 统计滤波器邻居数
    std_ratio = body.get('std_ratio', 2.0)         # 统计滤波器标准差倍数

    try:
        # 1. 先进行体素下采样（大幅减少点数）
        if voxel_size > 0:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        else:
            pcd_down = pcd

        # 2. 去噪处理
        if method == 'radius':
            # 半径滤波：剔除半径内邻居数少于 min_neighbors 的点
            pcd_clean, _ = pcd_down.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        else:  # statistical
            pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # 如果去噪后点数过少，则回退到原始下采样点云
        if len(pcd_clean.points) < len(pcd_down.points) * 0.1:
            pcd_clean = pcd_down

        # 3. 更新缓存
        ORIGINAL_DOWNSAMPLED[uuid_name] = copy.deepcopy(pcd_clean)
        CACHED_CLOUDS[uuid_name] = pcd_clean
        return pcd_to_binary(pcd_clean, uuid=uuid_name,
                             filename=CLOUD_META[uuid_name]['filename'])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compute_normals', methods=['POST'])
def compute_normals():
    body = request.get_json()
    uuid_name = body.get('uuid')
    if not uuid_name or uuid_name not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    voxel_size = body.get('voxel_size', DEFAULT_VOXEL_SIZE)

    try:
        pcd = CACHED_CLOUDS[uuid_name]
        # 复制一份避免修改原始数据
        pcd_work = copy.deepcopy(pcd)
        if voxel_size > 0:
            pcd_work = pcd_work.voxel_down_sample(voxel_size)

        pcd_work.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        pcd_work.orient_normals_towards_camera_location([0, 0, 0])

        # 更新缓存
        CACHED_CLOUDS[uuid_name] = pcd_work

        return pcd_to_binary(pcd_work, uuid=uuid_name,
                             filename=CLOUD_META[uuid_name]['filename'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/boundingbox', methods=['POST'])
def boundingbox():
    body = request.get_json()
    uuid_name = body.get('uuid')
    if not uuid_name or uuid_name not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    try:
        pcd = CACHED_CLOUDS[uuid_name]
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound

        return jsonify({
            'center': center.tolist(),
            'size': size.tolist(),
            'min': min_bound.tolist(),
            'max': max_bound.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 配准：批量记录点对 ----------
@app.route('/register_correspondences', methods=['POST'])
def register_correspondences():
    """
    批量记录对应点对
    前端发送: {
        src_uuid: str,
        tgt_uuid: str,
        pairs: [{src_point: [x,y,z], tgt_point: [x,y,z]}, ...]  // Y-up
    }
    """
    body = request.get_json()
    src_uuid = body['src_uuid']
    tgt_uuid = body['tgt_uuid']
    pairs = body.get('pairs', [])

    # 初始化存储
    if src_uuid not in REG_PAIRS:
        REG_PAIRS[src_uuid] = {}
    if tgt_uuid not in REG_PAIRS[src_uuid]:
        REG_PAIRS[src_uuid][tgt_uuid] = []

    # 清空旧数据，存储新数据
    REG_PAIRS[src_uuid][tgt_uuid] = []

    for pair in pairs:
        src_pt_y = np.array(pair['src_point'])
        tgt_pt_y = np.array(pair['tgt_point'])

        # 转为 Z-up 存储
        src_pt_z = to_zup(src_pt_y)
        tgt_pt_z = to_zup(tgt_pt_y)

        REG_PAIRS[src_uuid][tgt_uuid].append({
            'src': src_pt_z,
            'tgt': tgt_pt_z
        })

    count = len(REG_PAIRS[src_uuid][tgt_uuid])
    return jsonify({'status': 'ok', 'count': count})


# ---------- 配准：执行粗配准 (SVD) ----------
@app.route('/apply_registration', methods=['POST'])
def apply_registration():
    """
    基于对应点执行SVD配准
    返回变换后的点云 + 变换矩阵 + RMSE
    """
    body = request.get_json()
    src_uuid = body['src_uuid']
    tgt_uuid = body['tgt_uuid']

    if src_uuid not in REG_PAIRS or tgt_uuid not in REG_PAIRS[src_uuid]:
        return jsonify({'error': '未找到对应点对，请先采集并记录对应点'}), 400

    pairs = REG_PAIRS[src_uuid][tgt_uuid]
    if len(pairs) < 3:
        return jsonify({'error': f'至少需要3组对应点，当前只有{len(pairs)}组'}), 400

    # 提取点对
    src_pts = np.array([p['src'] for p in pairs])  # Nx3 (Z-up)
    tgt_pts = np.array([p['tgt'] for p in pairs])  # Nx3 (Z-up)

    # ---- SVD 求解刚性变换 R, t ----
    src_mean = np.mean(src_pts, axis=0)
    tgt_mean = np.mean(tgt_pts, axis=0)

    # 中心化
    src_centered = src_pts - src_mean
    tgt_centered = tgt_pts - tgt_mean

    # 计算协方差矩阵
    H = src_centered.T @ tgt_centered

    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射情况（保证旋转矩阵行列式为+1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean

    # 构建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    # 从原始下采样点云复制并变换
    src_original = ORIGINAL_DOWNSAMPLED[src_uuid]   # 原始未变换
    src_pcd = copy.deepcopy(src_original)
    src_pcd.transform(transform)
    CACHED_CLOUDS[src_uuid] = src_pcd

    # 计算误差（使用原始点云评估）
    tgt_pcd = CACHED_CLOUDS[tgt_uuid]
    error_info = compute_registration_error(src_original, tgt_pcd, transform, threshold=1.0)

    # 清空已使用的配准点对
    if src_uuid in REG_PAIRS and tgt_uuid in REG_PAIRS[src_uuid]:
        del REG_PAIRS[src_uuid][tgt_uuid]

    return pcd_to_binary(
        src_pcd,
        uuid=src_uuid,
        filename=CLOUD_META[src_uuid]['filename'],
        transformation=transform.tolist(),
        rmse=error_info['rmse'],
        overlap_ratio=error_info['overlap_ratio'],
    )


# ---------- ICP 精配准（多尺度 + 自适应阈值） ----------
@app.route('/icp_refine', methods=['POST'])
def icp_refine():
    body = request.get_json()
    src_uuid = body['src_uuid']
    tgt_uuid = body['tgt_uuid']

    if src_uuid not in ORIGINAL_DOWNSAMPLED or tgt_uuid not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    # 从原始副本获取未变换的源点云
    src_original = ORIGINAL_DOWNSAMPLED[src_uuid]
    tgt_pcd = CACHED_CLOUDS[tgt_uuid]

    # 粗配准矩阵（前端传入）
    initial_transform = np.array(body.get('initial_transform', np.eye(4).tolist()))
    voxel_size = body.get('voxel_size', 0.05)

    try:
        # 自适应阈值多尺度 Point-to-Plane ICP
        # 1. 粗尺度
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

        # 2. 中尺度
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

        # 3. 精细尺度
        src_fine = src_original  # 原始下采样点云
        tgt_fine = tgt_pcd
        # 若点数过大，再下采样一次
        if len(src_fine.points) > 2000000:
            src_fine = src_fine.voxel_down_sample(voxel_size)
            tgt_fine = tgt_fine.voxel_down_sample(voxel_size)

        src_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        tgt_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        # 从粗配准结果或中尺度结果开始
        current_transform = reg_medium.transformation

        # 自适应阈值：从较大值逐步减小到精细阈值
        thresholds = [voxel_size * 6, voxel_size * 3, voxel_size * 1.5]
        for thresh in thresholds:
            reg_fine = o3d.pipelines.registration.registration_icp(
                src_fine, tgt_fine,
                max_correspondence_distance=thresh,
                init=current_transform,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            current_transform = reg_fine.transformation
            # 若 fitness 太低，可能已收敛，提前跳出
            if reg_fine.fitness < 0.01:
                break

        final_transform = current_transform

        # 应用最终变换到原始点云（精配准结果）
        src_transformed = copy.deepcopy(src_original)
        src_transformed.transform(final_transform)
        CACHED_CLOUDS[src_uuid] = src_transformed

        # 计算误差
        error_info = compute_registration_error(
            src_original, tgt_pcd, final_transform, threshold=voxel_size * 5
        )

        return pcd_to_binary(
            src_transformed,
            uuid=src_uuid,
            filename=CLOUD_META[src_uuid]['filename'],
            transformation=np.asarray(final_transform).tolist(),
            fitness=float(reg_fine.fitness),
            inlier_rmse=float(reg_fine.inlier_rmse),
            rmse=error_info['rmse'],
            overlap_ratio=error_info['overlap_ratio'],
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- 新增：点云叠加可视化 ----------
@app.route('/merge_clouds', methods=['POST'])
def merge_clouds():
    """将两个点云合并为一个，用于可视化配准结果"""
    body = request.get_json()
    uuid1 = body.get('uuid1')
    uuid2 = body.get('uuid2')

    if uuid1 not in CACHED_CLOUDS or uuid2 not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    pcd1 = copy.deepcopy(CACHED_CLOUDS[uuid1])  # 修复：使用深拷贝
    pcd2 = copy.deepcopy(CACHED_CLOUDS[uuid2])  # 修复：使用深拷贝

    # 给两个点云不同颜色以便区分
    colors1 = np.asarray(pcd1.colors)
    if len(colors1) == 0 or np.all(colors1 == colors1[0]):
        pcd1.paint_uniform_color([1.0, 0.4, 0.4])  # 红色

    colors2 = np.asarray(pcd2.colors)
    if len(colors2) == 0 or np.all(colors2 == colors2[0]):
        pcd2.paint_uniform_color([0.4, 0.4, 1.0])  # 蓝色

    merged = pcd1 + pcd2

    new_uuid = f"merged_{uuid.uuid4().hex[:8]}"
    CACHED_CLOUDS[new_uuid] = merged
    CLOUD_META[new_uuid] = {'filename': f'merged_{CLOUD_META[uuid1]["filename"]}_{CLOUD_META[uuid2]["filename"]}'}

    # 默认只返回元数据（前端保存流程不需要点云数据本身）
    if body.get('return_data', False):
        return pcd_to_binary(merged, uuid=new_uuid, filename=CLOUD_META[new_uuid]['filename'])

    return jsonify({
        'uuid': new_uuid,
        'filename': CLOUD_META[new_uuid]['filename'],
        'point_count': len(merged.points),
    })


# ---------- 新增：保存配准结果 ----------
@app.route('/save_registered', methods=['POST'])
def save_registered():
    """保存配准后的点云为PLY文件"""
    body = request.get_json()
    uuid_name = body.get('uuid')
    filename = body.get('filename', 'registered.ply')

    if uuid_name not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    try:
        pcd = CACHED_CLOUDS[uuid_name]
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        o3d.io.write_point_cloud(save_path, pcd)
        return jsonify({'status': 'ok', 'path': save_path, 'points': len(pcd.points)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """提供文件下载功能"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': '文件不存在'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)