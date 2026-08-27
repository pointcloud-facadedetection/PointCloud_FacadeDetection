"""读取 algorithms/data/bllygg01.json，在终端输出扫描仪位姿。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DATA_JSON = Path(__file__).resolve().parents[1] / 'data' / 'bllygg01.json'


def load_scanner_pose(json_path: Path | None = None) -> dict:
    path = (json_path or _DATA_JSON).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到扫描位姿 JSON: {path}')

    with path.open('r', encoding='utf-8') as f:
        scan_data = json.load(f)

    transform_to_global = np.asarray(scan_data['transformToGlobal'], dtype=np.float64)
    if transform_to_global.shape != (4, 4):
        raise ValueError(f'transformToGlobal 应为 4x4，实际 {transform_to_global.shape}')

    camera_extrinsic = np.linalg.inv(transform_to_global)
    rotation = transform_to_global[:3, :3]
    translation = transform_to_global[:3, 3]

    return {
        'json_path': str(path),
        'scan_name': scan_data.get('scanName'),
        'scanner_type': scan_data.get('scannerType'),
        'transform_to_global': transform_to_global,
        'camera_extrinsic': camera_extrinsic,
        'rotation': rotation,
        'translation': translation,
    }


def print_scanner_pose(pose: dict) -> None:
    np.set_printoptions(precision=8, suppress=True)
    print(f'JSON: {pose["json_path"]}')
    print(f'扫描名: {pose["scan_name"]}')
    print(f'扫描仪型号: {pose["scanner_type"]}')
    print()
    print('扫描仪位姿矩阵 transformToGlobal（扫描仪 -> 全局）:')
    print(pose['transform_to_global'])
    print()
    print('旋转 R:')
    print(pose['rotation'])
    print()
    print('平移 T（全局坐标下的扫描仪位置）:')
    print(pose['translation'])
    print()
    print('相机外参 Extrinsic = inv(transformToGlobal)（全局 -> 相机）:')
    print(pose['camera_extrinsic'])


if __name__ == '__main__':
    print_scanner_pose(load_scanner_pose())
