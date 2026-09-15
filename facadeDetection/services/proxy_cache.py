"""站点加载链路的两层磁盘缓存。

原始点缓存：Open3D 解包 PLY 的结果（float32 数组）以裸 .npy 三件套落盘
（points.npy / colors.npy / 指纹 json），重开项目用 np.load(mmap_mode='r')
建立只读映射，既不整读也不解析，RSS 只计实际触碰的页，OS 可回收。

代理缓存：非去噪站点的代理重建（read_dist + estimate_elevation_angles +
stratified_proxy_build）对同一资产结果完全确定；把 CSR 映射、代表行与代理
数组本体（proxy_points/proxy_colors）持久化后，重开项目直接使用缓存的代理
数组，既不重建也不从 raw memmap 采集（代表点散布全文件，fancy 采集会把
整个映射换入内存）。无 proxy 数组字段的旧缓存回退按代表行采集。

缓存以资产指纹为唯一有效性凭证：指纹不一致、文件缺失或损坏一律未命中，
绝不使用过期缓存。读写失败只允许警告，不得影响主流程。
"""
import json
from pathlib import Path

import numpy as np

from config.storage import Storage

# raw 缓存格式版本：布局或语义变更时递增，旧版本一律未命中回退重建
RAW_CACHE_FORMAT = 2


def raw_cache_paths(project_uuid, station_id):
    """原始点缓存三件套路径：points.npy / colors.npy / 指纹 json。"""
    base = (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'raw' / str(station_id))
    return (base.with_suffix('.points.npy'),
            base.with_suffix('.colors.npy'),
            base.with_suffix('.json'))


def _legacy_raw_cache_path(project_uuid, station_id) -> Path:
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'raw' / f'{station_id}.npz')


def proxy_cache_path(project_uuid, station_id) -> Path:
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'proxy' / f'{station_id}.npz')


def denoise_sidecar_path(project_uuid, station_id) -> Path:
    """去噪状态大数组的二进制 sidecar（DB JSON 只存标量元数据）。"""
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'denoise' / f'{station_id}.npz')


def _fingerprint_fields(fingerprint_key):
    fp_path, fp_sha, fp_size = fingerprint_key
    return (str(fp_path), '' if fp_sha is None else str(fp_sha),
            -1 if fp_size is None else int(fp_size))


def _save_npz(path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp')
    np.savez(tmp,  # 实际写入 <stem>.tmp.npz，完成后原子替换正式文件
             **arrays)
    Path(f'{tmp}.npz').replace(path)


def _save_npy(path, array) -> None:
    """裸 .npy 落盘（tmp + 原子替换）；裸格式才能被 np.load(mmap_mode) 映射。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp')
    with open(tmp, 'wb') as stream:
        np.save(stream, array)
    tmp.replace(path)


def _save_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    tmp.replace(path)


def _fingerprint_matches(data, fingerprint_key) -> bool:
    fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
    return (str(data['fingerprint_path']) == fp_path and
            str(data['fingerprint_sha']) == fp_sha and
            int(data['fingerprint_size']) == fp_size)


def _remove_legacy_raw_cache(project_uuid, station_id) -> None:
    """旧格式 raw npz 的一次性迁移清理；best-effort。"""
    try:
        _legacy_raw_cache_path(project_uuid, station_id).unlink(missing_ok=True)
    except OSError:
        pass


def save_raw_cache(project_uuid, station_id, fingerprint_key, *, points,
                   colors) -> bool:
    """写入原始点缓存三件套（.npy 不压缩）；失败仅警告并返回 False。

    指纹 json 最后落盘，读端先验 json 再映射数组：写入中途崩溃只会导致
    未命中重建，绝不会读到半个缓存。sidecar 同时记录 colors_in_range：
    三件套是本进程写入的已校验数据，读端凭标记跳过 register_source_asset
    的值域扫描（对 memmap 的 min/max 整扫会把全部颜色页换入内存）。
    """
    try:
        fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
        points_path, colors_path, meta_path = raw_cache_paths(
            project_uuid, station_id)
        colors_array = (np.empty((0, 3), dtype=np.float32) if colors is None
                        else np.asarray(colors, dtype=np.float32).reshape(-1, 3))
        colors_in_range = bool(
            colors_array.size == 0 or
            (float(colors_array.min()) >= 0.0 and
             float(colors_array.max()) <= 1.0))
        _save_npy(points_path,
                  np.asarray(points, dtype=np.float32).reshape(-1, 3))
        # 无颜色站点用空数组作标记，读回时还原为 None
        _save_npy(colors_path, colors_array)
        _save_json(meta_path, {
            'format': RAW_CACHE_FORMAT,
            'fingerprint_path': fp_path,
            'fingerprint_sha': fp_sha,
            'fingerprint_size': fp_size,
            'colors_in_range': colors_in_range,
        })
        _remove_legacy_raw_cache(project_uuid, station_id)
        return True
    except Exception as exc:
        print(f'[PCFD] raw_cache.save_failed station={station_id} '
              f'reason={exc}', flush=True)
        return False


def load_raw_cache(project_uuid, station_id, fingerprint_key):
    """读取原始点缓存；指纹不符/缺失/损坏一律返回 None。

    命中时返回 np.load(mmap_mode='r') 的只读映射，不整读数据；
    形状校验只看 header，不触碰负载页。
    """
    try:
        points_path, colors_path, meta_path = raw_cache_paths(
            project_uuid, station_id)
        _remove_legacy_raw_cache(project_uuid, station_id)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if int(meta.get('format', -1)) != RAW_CACHE_FORMAT:
            return None
        fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
        if (str(meta.get('fingerprint_path')) != fp_path or
                str(meta.get('fingerprint_sha')) != fp_sha or
                int(meta.get('fingerprint_size', -2)) != fp_size):
            return None
        if not points_path.exists() or not colors_path.exists():
            return None
        points = np.load(points_path, mmap_mode='r')
        colors = np.load(colors_path, mmap_mode='r')
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            return None
        if colors.size == 0:
            colors = None
        elif colors.shape != points.shape:
            return None
        elif meta.get('colors_in_range'):
            # 附带在 memmap 对象上的进程内标记（ndarray 语义不变）：
            # register_source_asset 据此跳过值域扫描，保持映射不换页。
            # 旧缓存无此字段，保持现有扫描行为。
            colors._pcfd_colors_in_range = True
        return points, colors
    except Exception:
        return None


def save_proxy_cache(project_uuid, station_id, fingerprint_key, *, offsets,
                     indices, ranges, scan_origins, distance_source,
                     representative_ids, proxy_points=None,
                     proxy_colors=None, proxy_normals=None) -> bool:
    """写入代理缓存（np.savez 不压缩）；失败仅警告并返回 False。

    proxy_points/proxy_colors 是重建产出的代理数组本体；持久化后重开项目
    直接使用它们，不从 raw memmap 按代表行采集（采集会把整个映射换页）。
    proxy_normals 是检测首次估计的代理法向（float64 原值），随同一指纹
    持久化后下次检测/重开直接复用。
    """
    try:
        fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
        arrays = {}
        if proxy_points is not None:
            arrays['proxy_points'] = np.asarray(
                proxy_points, dtype=np.float32).reshape(-1, 3)
            # 无颜色代理用空数组作标记，读回时还原为 None
            arrays['proxy_colors'] = (
                np.empty((0, 3), dtype=np.float32) if proxy_colors is None
                else np.asarray(proxy_colors, dtype=np.float32).reshape(-1, 3))
        if proxy_normals is not None:
            arrays['proxy_normals'] = np.asarray(
                proxy_normals, dtype=np.float64).reshape(-1, 3)
        _save_npz(
            proxy_cache_path(project_uuid, station_id),
            offsets=np.asarray(offsets, dtype=np.int64),
            indices=np.asarray(indices, dtype=np.int32),
            # CSR 组内按 lexsort 排列，代表点不一定是组首；
            # 显式持久化代表行才能逐点复现重建结果
            representative_ids=np.asarray(representative_ids, dtype=np.int64),
            ranges=np.asarray(ranges, dtype=np.float32),
            scan_origins=np.asarray(scan_origins, dtype=np.float32),
            distance_source=np.array(str(distance_source)),
            fingerprint_path=np.array(fp_path),
            fingerprint_sha=np.array(fp_sha),
            fingerprint_size=np.array(fp_size, dtype=np.int64),
            **arrays)
        return True
    except Exception as exc:
        print(f'[PCFD] proxy_cache.save_failed station={station_id} '
              f'reason={exc}', flush=True)
        return False


def load_proxy_cache(project_uuid, station_id, fingerprint_key,
                     source_count=None):
    """读取代理缓存；指纹不符/缺失/损坏一律返回 None。

    返回的 proxy_points/proxy_colors 是缓存的代理数组本体；旧缓存没有
    这两个字段（或形状与 CSR 不一致）时为 None，调用方回退按代表行采集。
    proxy_normals 同理：无字段或长度与代理数不一致时为 None，调用方按
    现行逻辑估计法向。
    """
    try:
        path = proxy_cache_path(project_uuid, station_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if not _fingerprint_matches(data, fingerprint_key):
                return None
            offsets = np.asarray(data['offsets'], dtype=np.int64)
            indices = np.asarray(data['indices'], dtype=np.int32)
            representative_ids = np.asarray(data['representative_ids'],
                                            dtype=np.int64)
            ranges = np.asarray(data['ranges'], dtype=np.float32)
            scan_origins = np.asarray(data['scan_origins'], dtype=np.float32)
            distance_source = str(data['distance_source'])
            proxy_points = None
            proxy_colors = None
            if 'proxy_points' in data.files:
                candidate = np.asarray(data['proxy_points'], dtype=np.float32)
                colors = np.asarray(data['proxy_colors'], dtype=np.float32)
                if candidate.ndim == 2 and candidate.shape[1] == 3 and \
                        len(candidate) == len(offsets) - 1:
                    proxy_points = candidate
                    if colors.size == 0:
                        proxy_colors = None
                    elif colors.shape == candidate.shape:
                        proxy_colors = colors
                    else:
                        # 颜色形状损坏：整份代理回退采集，不混用半份缓存
                        proxy_points = None
            proxy_normals = None
            if 'proxy_normals' in data.files:
                cand_normals = np.asarray(data['proxy_normals'],
                                          dtype=np.float64)
                if cand_normals.ndim == 2 and cand_normals.shape[1] == 3 and \
                        len(cand_normals) == len(offsets) - 1:
                    proxy_normals = cand_normals
        # CSR 结构完整性：损坏文件绝不进入主流程
        if (len(offsets) < 2 or offsets[0] != 0 or
                np.any(np.diff(offsets) <= 0) or
                len(indices) != int(offsets[-1]) or
                len(representative_ids) != len(offsets) - 1 or
                len(ranges) != len(offsets) - 1):
            return None
        if source_count is not None:
            if len(indices) and (indices.min() < 0 or
                                 indices.max() >= source_count):
                return None
            if len(representative_ids) and \
                    (representative_ids.min() < 0 or
                     representative_ids.max() >= source_count):
                return None
        return {'offsets': offsets, 'indices': indices,
                'representative_ids': representative_ids,
                'ranges': ranges, 'scan_origins': scan_origins,
                'distance_source': distance_source,
                'proxy_points': proxy_points, 'proxy_colors': proxy_colors,
                'proxy_normals': proxy_normals}
    except Exception:
        return None


def save_proxy_normals(project_uuid, station_id, fingerprint_key,
                       normals) -> bool:
    """向既有 proxy 缓存追加代理法向（检测首次估计后持久化）。

    代理法向对同一资产指纹是确定的；仅当缓存存在、指纹一致且法向条数与
    CSR 代理数一致时才原子重写 npz（去噪子集的法向长度不符，绝不写入）。
    任何失败仅警告并返回 False，不影响主流程。
    """
    try:
        path = proxy_cache_path(project_uuid, station_id)
        if not path.exists():
            return False
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        with np.load(path, allow_pickle=False) as data:
            if not _fingerprint_matches(data, fingerprint_key):
                return False
            # 长度对不上 CSR 代理数的法向不是这份缓存的法向，拒绝混入
            if len(normals) != len(data['offsets']) - 1:
                return False
            arrays = {name: data[name] for name in data.files}
        arrays['proxy_normals'] = normals
        _save_npz(path, **arrays)
        return True
    except Exception as exc:
        print(f'[PCFD] proxy_cache.normals_save_failed station={station_id} '
              f'reason={exc}', flush=True)
        return False


def delete_station_cache(project_uuid, station_id) -> None:
    """站点删除时清理两层缓存；连同空目录一起移除，全部 best-effort。"""
    points_path, colors_path, meta_path = raw_cache_paths(
        project_uuid, station_id)
    for path in (points_path, colors_path, meta_path,
                 _legacy_raw_cache_path(project_uuid, station_id),
                 proxy_cache_path(project_uuid, station_id),
                 denoise_sidecar_path(project_uuid, station_id)):
        try:
            path.unlink(missing_ok=True)
            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
