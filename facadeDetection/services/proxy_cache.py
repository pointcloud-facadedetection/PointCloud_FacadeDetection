"""站点加载链路的两层磁盘缓存。

原始点缓存：Open3D 解包 PLY 的结果（float32 数组）落盘后，重开项目用
np.load 直接读回，跳过每站约 3.5s 的 PLY 解包。

代理缓存：非去噪站点的代理重建（read_dist + estimate_elevation_angles +
stratified_proxy_build）对同一资产结果完全确定；把 CSR 映射与代表行持久化后，
重开项目可以直接从源点云采集代理点，跳过每站约 12.6s 的体素分组计算。

缓存以资产指纹为唯一有效性凭证：指纹不一致、文件缺失或损坏一律未命中，
绝不使用过期缓存。读写失败只允许警告，不得影响主流程。
"""
from pathlib import Path

import numpy as np

from config.storage import Storage


def raw_cache_path(project_uuid, station_id) -> Path:
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'raw' / f'{station_id}.npz')


def proxy_cache_path(project_uuid, station_id) -> Path:
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'proxy' / f'{station_id}.npz')


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


def _fingerprint_matches(data, fingerprint_key) -> bool:
    fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
    return (str(data['fingerprint_path']) == fp_path and
            str(data['fingerprint_sha']) == fp_sha and
            int(data['fingerprint_size']) == fp_size)


def save_raw_cache(project_uuid, station_id, fingerprint_key, *, points,
                   colors) -> bool:
    """写入原始点缓存（np.savez 不压缩）；失败仅警告并返回 False。"""
    try:
        fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
        _save_npz(
            raw_cache_path(project_uuid, station_id),
            points=np.asarray(points, dtype=np.float32).reshape(-1, 3),
            # 无颜色站点用空数组作标记，读回时还原为 None
            colors=(np.empty((0, 3), dtype=np.float32) if colors is None
                    else np.asarray(colors, dtype=np.float32).reshape(-1, 3)),
            fingerprint_path=np.array(fp_path),
            fingerprint_sha=np.array(fp_sha),
            fingerprint_size=np.array(fp_size, dtype=np.int64))
        return True
    except Exception as exc:
        print(f'[PCFD] raw_cache.save_failed station={station_id} '
              f'reason={exc}', flush=True)
        return False


def load_raw_cache(project_uuid, station_id, fingerprint_key):
    """读取原始点缓存；指纹不符/缺失/损坏一律返回 None。"""
    try:
        path = raw_cache_path(project_uuid, station_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if not _fingerprint_matches(data, fingerprint_key):
                return None
            points = np.asarray(data['points'], dtype=np.float32)
            colors = np.asarray(data['colors'], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            return None
        if colors.size == 0:
            colors = None
        elif colors.shape != points.shape:
            return None
        return points, colors
    except Exception:
        return None


def save_proxy_cache(project_uuid, station_id, fingerprint_key, *, offsets,
                     indices, ranges, scan_origins, distance_source,
                     representative_ids) -> bool:
    """写入代理缓存（np.savez 不压缩）；失败仅警告并返回 False。"""
    try:
        fp_path, fp_sha, fp_size = _fingerprint_fields(fingerprint_key)
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
            fingerprint_size=np.array(fp_size, dtype=np.int64))
        return True
    except Exception as exc:
        print(f'[PCFD] proxy_cache.save_failed station={station_id} '
              f'reason={exc}', flush=True)
        return False


def load_proxy_cache(project_uuid, station_id, fingerprint_key,
                     source_count=None):
    """读取代理缓存；指纹不符/缺失/损坏一律返回 None。"""
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
                'distance_source': distance_source}
    except Exception:
        return None


def delete_station_cache(project_uuid, station_id) -> None:
    """站点删除时清理两层缓存；连同空目录一起移除，全部 best-effort。"""
    for path in (raw_cache_path(project_uuid, station_id),
                 proxy_cache_path(project_uuid, station_id)):
        try:
            path.unlink(missing_ok=True)
            parent = path.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
