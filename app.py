import os
import copy
import numpy as np
import open3d as o3d
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- 全局缓存 ----------
CACHED_CLOUDS = {}  # uuid -> o3d.geometry.PointCloud
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


def pcd_to_json(pcd):
    """将Open3D点云转换为JSON格式（Z-up坐标系）"""
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.ones_like(points) * 0.7
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(points)
    return {
        'positions': points.flatten().tolist(),
        'colors': colors.flatten().tolist(),
        'normals': normals.flatten().tolist()
    }


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
    """支持多文件上传"""
    files = request.files.getlist('files')
    voxel_size = float(request.form.get('voxel_size', 0.05))
    results = []

    for file in files:
        if not file.filename.lower().endswith('.ply'):
            continue
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        try:
            pcd = process_ply(filepath, voxel_size)
            CACHED_CLOUDS[unique_name] = pcd
            CLOUD_META[unique_name] = {'filename': file.filename}
            data = pcd_to_json(pcd)
            data['filename'] = file.filename
            data['uuid'] = unique_name
            results.append(data)
        except Exception as e:
            results.append({'error': str(e), 'filename': file.filename})

    return jsonify(results)


@app.route('/denoise', methods=['POST'])
def denoise():
    body = request.get_json()
    uuid_name = body.get('uuid')
    if uuid_name not in CACHED_CLOUDS:
        return jsonify({'error': '点云未缓存'}), 404

    pcd = CACHED_CLOUDS[uuid_name]
    voxel_size = body.get('voxel_size', 0.05)

    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        CACHED_CLOUDS[uuid_name] = pcd
        data = pcd_to_json(pcd)
        data['uuid'] = uuid_name
        data['filename'] = CLOUD_META[uuid_name]['filename']
        return jsonify(data)
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

        data = pcd_to_json(pcd_work)
        data['uuid'] = uuid_name
        return jsonify(data)
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

    # ---- 变换源点云 ----
    src_pcd = copy.deepcopy(CACHED_CLOUDS[src_uuid])
    src_pcd.transform(transform)
    CACHED_CLOUDS[src_uuid] = src_pcd

    # ---- 计算配准误差 ----
    tgt_pcd = CACHED_CLOUDS[tgt_uuid]

    # 从原始文件读取未变换的源点云来计算误差
    original_filepath = os.path.join(UPLOAD_FOLDER, src_uuid)
    if os.path.exists(original_filepath):
        original_src = o3d.io.read_point_cloud(original_filepath)
        error_info = compute_registration_error(original_src, tgt_pcd, transform, threshold=1.0)
    else:
        # 如果原始文件不存在，使用缓存中的点云（但注意它可能已经被变换过）
        # 这里需要重新读取原始数据
        error_info = {'rmse': 0.0, 'overlap_ratio': 0.0, 'inlier_count': 0, 'total_count': 0}

    # ---- 返回前端----
    data = pcd_to_json(src_pcd)  # Z-up 格式
    data['uuid'] = src_uuid
    data['filename'] = CLOUD_META[src_uuid]['filename']
    data['transformation'] = transform.tolist()
    data['rmse'] = error_info['rmse']
    data['overlap_ratio'] = error_info['overlap_ratio']

    # 清空已使用的配准点对
    if src_uuid in REG_PAIRS and tgt_uuid in REG_PAIRS[src_uuid]:
        del REG_PAIRS[src_uuid][tgt_uuid]

    return jsonify(data)


# ---------- 配准：ICP 精配准 ----------
@app.route('/icp_refine', methods=['POST'])
def icp_refine():
    """
    ICP精配准，使用原始源点云 + 粗配准变换作为初始值
    采用 Point-to-Plane ICP，多尺度策略
    """
    body = request.get_json()
    src_uuid = body['src_uuid']
    tgt_uuid = body['tgt_uuid']

    if src_uuid not in CACHED_CLOUDS or tgt_uuid not in CACHED_CLOUDS:
        return jsonify({'error': '点云未找到'}), 404

    # 从文件加载原始源点云（未变换）
    original_filepath = os.path.join(UPLOAD_FOLDER, src_uuid)
    if not os.path.exists(original_filepath):
        return jsonify({'error': '原始源文件丢失，请重新上传'}), 400

    src_original = o3d.io.read_point_cloud(original_filepath)
    if len(src_original.points) == 0:
        return jsonify({'error': '原始点云为空'}), 400

    tgt_pcd = CACHED_CLOUDS[tgt_uuid]

    # 获取粗配准变换（如果未提供，则使用单位矩阵）
    initial_transform = np.array(body.get('initial_transform', np.eye(4).tolist()))
    voxel_size = body.get('voxel_size', 0.05)

    try:
        # ---- 多尺度 ICP（Point-to-Plane） ----
        # 阶段1：粗尺度 (voxel_coarse = max(0.3, voxel_size*4) )
        voxel_coarse = max(voxel_size * 4, 0.3)
        src_coarse = src_original.voxel_down_sample(voxel_coarse)
        tgt_coarse = tgt_pcd.voxel_down_sample(voxel_coarse)

        # 计算法向量（Point-to-Plane 需要）
        src_coarse.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2, max_nn=30))
        tgt_coarse.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_coarse*2, max_nn=30))

        reg_coarse = o3d.pipelines.registration.registration_icp(
            src_coarse, tgt_coarse,
            max_correspondence_distance=voxel_coarse * 5,  # 增大搜索半径
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
        )

        # 阶段2：中等尺度
        voxel_medium = max(voxel_size * 2, 0.2)
        src_medium = src_original.voxel_down_sample(voxel_medium)
        tgt_medium = tgt_pcd.voxel_down_sample(voxel_medium)
        src_medium.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_medium*2, max_nn=30))
        tgt_medium.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_medium*2, max_nn=30))

        reg_medium = o3d.pipelines.registration.registration_icp(
            src_medium, tgt_medium,
            max_correspondence_distance=voxel_medium * 5,
            init=reg_coarse.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
        )

        # 阶段3：精细尺度（使用原始点云或精细下采样）
        if len(src_original.points) > 1000000:
            voxel_fine = voxel_size
            src_fine = src_original.voxel_down_sample(voxel_fine)
            tgt_fine = tgt_pcd.voxel_down_sample(voxel_fine)
        else:
            src_fine = src_original
            tgt_fine = tgt_pcd

        src_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_fine*2, max_nn=30))
        tgt_fine.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_fine*2, max_nn=30))

        reg_fine = o3d.pipelines.registration.registration_icp(
            src_fine, tgt_fine,
            max_correspondence_distance=voxel_fine * 5,
            init=reg_medium.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120)
        )

        final_transform = reg_fine.transformation

        # ---- 应用最终变换到原始源点云 ----
        src_transformed = copy.deepcopy(src_original)
        src_transformed.transform(final_transform)
        CACHED_CLOUDS[src_uuid] = src_transformed

        # ---- 计算误差 ----
        error_info = compute_registration_error(
            src_original, tgt_pcd, final_transform, threshold=voxel_size * 5
        )

        # ---- 返回前端 (Z-up) ----
        data = pcd_to_json(src_transformed)
        data['uuid'] = src_uuid
        data['filename'] = CLOUD_META[src_uuid]['filename']
        data['transformation'] = final_transform.tolist()
        data['fitness'] = float(reg_fine.fitness)
        data['inlier_rmse'] = float(reg_fine.inlier_rmse)
        data['rmse'] = error_info['rmse']
        data['overlap_ratio'] = error_info['overlap_ratio']

        return jsonify(data)

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

    data = pcd_to_json(merged)
    new_uuid = f"merged_{uuid.uuid4().hex[:8]}"
    CACHED_CLOUDS[new_uuid] = merged
    CLOUD_META[new_uuid] = {'filename': f'merged_{CLOUD_META[uuid1]["filename"]}_{CLOUD_META[uuid2]["filename"]}'}

    data['uuid'] = new_uuid
    data['filename'] = CLOUD_META[new_uuid]['filename']

    return jsonify(data)


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
    app.run(host='0.0.0.0', port=5000, debug=True)