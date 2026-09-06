"""PLY 二进制负载的 memmap 免解析快读。

只接受 format binary_little_endian、单 element vertex、属性为连续定长记录的
PLY：按属性表构造记录布局后 np.memmap 负载区，xyz 取 stride 视图拷贝成
(N,3) float32，uchar rgb 归一化 /255 成 (N,3) float32。任何不符合（ASCII、
大端、list 属性、多 element、未知类型、负载截断）都返回 None，调用方回退
o3d.io.read_point_cloud，行为与旧路径完全一致。
"""
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': '<i2', 'int16': '<i2', 'ushort': '<u2', 'uint16': '<u2',
    'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
}


def _parse_header(stream):
    """解析 PLY 头；返回 (顶点数, [(名称, dtype, 偏移)], 记录长, 负载偏移) 或 None。"""
    if stream.readline().strip() != b'ply':
        return None
    vertex_count = None
    properties = []
    offset = 0
    saw_end = False
    for line in stream:
        tokens = line.split()
        if not tokens:
            continue
        keyword = tokens[0]
        if keyword == b'format':
            if tokens[1:3] != [b'binary_little_endian', b'1.0']:
                return None
        elif keyword == b'element':
            # 单 element vertex：出现第二个 element 或非 vertex 一律回退
            if vertex_count is not None or tokens[1] != b'vertex':
                return None
            vertex_count = int(tokens[2])
        elif keyword == b'property':
            if vertex_count is None or tokens[1] == b'list':
                return None
            dtype = _PLY_TYPES.get(tokens[1].decode('ascii', 'replace'))
            if dtype is None:
                return None
            properties.append((tokens[2].decode('ascii', 'replace'),
                               dtype, offset))
            offset += np.dtype(dtype).itemsize
        elif keyword == b'end_header':
            saw_end = True
            break
        # comment / obj_info 等头行直接跳过
    if not saw_end or vertex_count is None or vertex_count <= 0 or not properties:
        return None
    return vertex_count, properties, offset, stream.tell()


def read_ply_fast(path):
    """快读 PLY；返回 (points float32 (N,3), colors float32 (N,3) 或 None)。

    不满足快读条件或文件损坏时返回 None，调用方必须回退 Open3D 读取。
    """
    try:
        if Path(str(path)).suffix.lower() != '.ply':
            return None
        with open(path, 'rb') as stream:
            parsed = _parse_header(stream)
            if parsed is None:
                return None
            count, properties, record_size, data_offset = parsed
        if Path(path).stat().st_size - data_offset < count * record_size:
            return None
        by_name = {name: (dtype, off) for name, dtype, off in properties}
        xyz = [by_name.get(axis) for axis in ('x', 'y', 'z')]
        if any(item is None for item in xyz):
            return None
        raw = np.memmap(path, dtype=np.uint8, mode='r',
                        offset=data_offset, shape=(count * record_size,))
        records = raw.reshape(count, record_size)
        (x_dtype, x_off), (y_dtype, y_off), (z_dtype, z_off) = xyz
        itemsize = np.dtype(x_dtype).itemsize
        if (x_dtype == y_dtype == z_dtype and
                y_off == x_off + itemsize and z_off == y_off + itemsize):
            # xyz 连续同型：一次 stride 视图 + 连续化拷贝
            points = np.ascontiguousarray(
                records[:, x_off:x_off + 3 * itemsize].view(x_dtype),
                dtype=np.float32)
        else:
            points = np.empty((count, 3), dtype=np.float32)
            for axis, (dtype, off) in enumerate(xyz):
                width = np.dtype(dtype).itemsize
                points[:, axis] = records[:, off:off + width].view(dtype)[:, 0]
        colors = None
        rgb = [by_name.get(name) for name in ('red', 'green', 'blue')]
        if all(item is not None for item in rgb):
            if any(dtype != 'u1' for dtype, _ in rgb):
                # 非 uchar 颜色的归一化语义由 Open3D 决定，不做猜测
                return None
            # 逐列写入 C 序数组：直接花式取列会产生 F 序结果，
            # 下游 ascontiguousarray 会因此整份拷贝
            colors = np.empty((count, 3), dtype=np.float32)
            for axis, (_, off) in enumerate(rgb):
                colors[:, axis] = records[:, off]
            colors /= np.float32(255.0)
        return points, colors
    except Exception:
        return None
