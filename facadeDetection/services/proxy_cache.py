"""分层代理（stratified proxy）的磁盘缓存。

非去噪站点的代理重建（read_dist + estimate_elevation_angles +
stratified_proxy_build）对同一资产结果完全确定；把 CSR 映射持久化后，
重开项目可以直接从源点云采集代理点，跳过每站约 12.6s 的体素分组计算。

缓存以资产指纹为唯一有效性凭证：指纹不一致、文件缺失或损坏一律未命中，
绝不使用过期缓存。读写失败只允许警告，不得影响主流程。
"""
from pathlib import Path

import numpy as np

from config.storage import Storage


def proxy_cache_path(project_uuid, station_id) -> Path:
    return (Storage.project_root(project_uuid) / Storage.CACHE_DIRNAME
            / 'proxy' / f'{station_id}.npz')


def save_proxy_cache(project_uuid, station_id, fingerprint_key, *, offsets,
                     indices, ranges, scan_origins, distance_source) -> bool:
    """写入代理缓存（np.savez 不压缩）；失败仅警告并返回 False。"""
    try:
        path = proxy_cache_path(project_uuid, station_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fp_path, fp_sha, fp_size = fingerprint_key
        tmp = path.with_name(f'{path.stem}.tmp')
        np.savez(tmp,  # 实际写入 <stem>.tmp.npz，完成后原子替换正式文件
                 offsets=np.asarray(offsets, dtype=np.int64),
                 indices=np.asarray(indices, dtype=np.int32),
                 ranges=np.asarray(ranges, dtype=np.float32),
                 scan_origins=np.asarray(scan_origins, dtype=np.float32),
                 distance_source=np.array(str(distance_source)),
                 fingerprint_path=np.array(str(fp_path)),
                 fingerprint_sha=np.array(
                     '' if fp_sha is None else str(fp_sha)),
                 fingerprint_size=np.array(
                     -1 if fp_size is None else int(fp_size), dtype=np.int64))
        Path(f'{tmp}.npz').replace(path)
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
        fp_path, fp_sha, fp_size = fingerprint_key
        with np.load(path, allow_pickle=False) as data:
            if (str(data['fingerprint_path']) != str(fp_path) or
                    str(data['fingerprint_sha']) !=
                    ('' if fp_sha is None else str(fp_sha)) or
                    int(data['fingerprint_size']) !=
                    (-1 if fp_size is None else int(fp_size))):
                return None
            offsets = np.asarray(data['offsets'], dtype=np.int64)
            indices = np.asarray(data['indices'], dtype=np.int32)
            ranges = np.asarray(data['ranges'], dtype=np.float32)
            scan_origins = np.asarray(data['scan_origins'], dtype=np.float32)
            distance_source = str(data['distance_source'])
        # CSR 结构完整性：损坏文件绝不进入主流程
        if (len(offsets) < 2 or offsets[0] != 0 or
                np.any(np.diff(offsets) <= 0) or
                len(indices) != int(offsets[-1]) or
                len(ranges) != len(offsets) - 1):
            return None
        if (source_count is not None and len(indices) and
                (indices.min() < 0 or indices.max() >= source_count)):
            return None
        return {'offsets': offsets, 'indices': indices, 'ranges': ranges,
                'scan_origins': scan_origins,
                'distance_source': distance_source}
    except Exception:
        return None


def delete_proxy_cache(project_uuid, station_id) -> None:
    """站点删除时清理缓存；连同空目录一起移除，全部 best-effort。"""
    try:
        path = proxy_cache_path(project_uuid, station_id)
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass
