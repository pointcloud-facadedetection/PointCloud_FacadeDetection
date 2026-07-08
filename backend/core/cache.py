import numpy as np

class CloudCache:
    """点云数据缓存管理器"""

    def __init__(self):
        # 当前显示/处理用点云
        self.display = {}           # uuid -> o3d.geometry.PointCloud
        # 原始下采样点云备份
        self.original_downsampled = {}  # uuid -> o3d.geometry.PointCloud
        # 当前累积变换矩阵
        self.current_transform = {}     # uuid -> 4x4 ndarray
        # 元数据
        self.cloud_meta = {}            # uuid -> {'filename': str, 'point_count': int}
        # 配准对应点缓存
        self.reg_pairs = {}             # src_uuid -> { tgt_uuid -> [{'src': ndarray, 'tgt': ndarray}, ...] }
        # 立面检测缓存
        self.facade_cache = {}          # uuid -> {'facades': [...], 'point_labels': [...], ...}
        # 框选细粒度分割缓存
        self.segment_cache = {}         # uuid -> {'segments': [...], 'point_labels': [...], 'base_colors': [...]}

    def get_display(self, uuid):
        return self.display.get(uuid)

    def set_display(self, uuid, pcd):
        self.display[uuid] = pcd

    def get_original(self, uuid):
        return self.original_downsampled.get(uuid)

    def set_original(self, uuid, pcd):
        self.original_downsampled[uuid] = pcd

    def get_transform(self, uuid):
        return self.current_transform.get(uuid, np.eye(4))

    def set_transform(self, uuid, transform):
        self.current_transform[uuid] = transform

    def get_meta(self, uuid):
        return self.cloud_meta.get(uuid)

    def set_meta(self, uuid, meta):
        self.cloud_meta[uuid] = meta

    def get_facade_cache(self, uuid):
        return self.facade_cache.get(uuid)

    def set_facade_cache(self, uuid, cache):
        self.facade_cache[uuid] = cache

    def get_segment_cache(self, uuid):
        return self.segment_cache.get(uuid)

    def set_segment_cache(self, uuid, cache):
        self.segment_cache[uuid] = cache

    def get_reg_pairs(self, src_uuid, tgt_uuid):
        return self.reg_pairs.get(src_uuid, {}).get(tgt_uuid, [])

    def set_reg_pairs(self, src_uuid, tgt_uuid, pairs):
        if src_uuid not in self.reg_pairs:
            self.reg_pairs[src_uuid] = {}
        self.reg_pairs[src_uuid][tgt_uuid] = pairs

    def clear_reg_pairs(self, src_uuid, tgt_uuid):
        if src_uuid in self.reg_pairs and tgt_uuid in self.reg_pairs[src_uuid]:
            del self.reg_pairs[src_uuid][tgt_uuid]

    def remove_cloud(self, uuid):
        """移除点云的所有缓存"""
        for cache in [self.display, self.original_downsampled, self.current_transform,
                      self.cloud_meta, self.facade_cache, self.segment_cache]:
            cache.pop(uuid, None)
        self.reg_pairs.pop(uuid, None)

# 全局单例
_cache = CloudCache()

def get_cache():
    return _cache

