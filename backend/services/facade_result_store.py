"""立面检测结果落盘缓存与加载。"""

import hashlib
import json
import os
import time
from datetime import datetime

import numpy as np
import open3d as o3d

from ..config import Config
from ..core.cache import get_cache
from ..core.geometry_utils import pcd_to_json


def _facade_folder():
    folder = os.path.join(Config.CACHE_FOLDER, "facade_detection")
    os.makedirs(folder, exist_ok=True)
    return folder


def _facade_result_id(filename, point_count, voxel_size, min_area):
    raw = f"{filename}|{point_count}|{voxel_size:.4f}|{min_area:.2f}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in filename)
    safe = safe[:40] or "cloud"
    return f"facade_{safe}_{digest}"


def _facade_paths(result_id):
    folder = _facade_folder()
    return (
        os.path.join(folder, f"{result_id}.json"),
        os.path.join(folder, f"{result_id}.npz"),
    )


def build_facade_display_colors(facades, point_labels, original_colors):
    """根据立面标签生成可视化颜色。"""
    point_labels = np.asarray(point_labels, dtype=int)
    original_colors = np.asarray(original_colors, dtype=float).reshape(-1, 3)
    colors = original_colors.astype(float, copy=True) * 0.35
    facade_by_id = {int(f["id"]): f for f in facades}
    for fid, facade in facade_by_id.items():
        base = np.asarray(
            Config.FACADE_TYPE_COLORS.get(facade["type"], [0.6, 0.6, 0.6]),
            dtype=float,
        )
        colors[point_labels == fid] = base
    return colors


def labels_from_facades(facades, num_points):
    point_labels = np.full(num_points, -1, dtype=int)
    for facade in facades:
        idx = np.asarray(facade.get("inlier_indices", []), dtype=int)
        idx = idx[(idx >= 0) & (idx < num_points)]
        point_labels[idx] = int(facade["id"])
    return point_labels


def build_facade_detection_response(uuid_name, facades, point_labels, base_colors):
    """构建立面检测/加载 API 返回体，并写入内存缓存。"""
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError("点云未找到")

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
    point_labels = np.asarray(point_labels, dtype=int)
    base_colors = np.asarray(base_colors, dtype=float).reshape(-1, 3)

    if len(point_labels) != len(points):
        raise ValueError(
            f"立面标签数量 ({len(point_labels)}) 与点云点数 ({len(points)}) 不一致"
        )
    if len(base_colors) != len(points):
        raise ValueError(
            f"原始颜色数量 ({len(base_colors)}) 与点云点数 ({len(points)}) 不一致"
        )

    colors = build_facade_display_colors(facades, point_labels, base_colors)
    cache.set_facade_cache(uuid_name, {
        "facades": facades,
        "point_labels": point_labels.tolist(),
        "base_colors": base_colors.tolist(),
    })

    meta = cache.get_meta(uuid_name) or {}
    return {
        "status": "ok",
        "facade_count": len(facades),
        "facades": facades,
        "point_labels": point_labels.tolist(),
        "positions": points.astype(np.float32).flatten().tolist(),
        "colors": colors.astype(np.float32).flatten().tolist(),
        "normals": normals.astype(np.float32).flatten().tolist(),
        "uuid": uuid_name,
        "filename": meta.get("filename", uuid_name),
    }


def save_facade_detection(uuid_name, voxel_size=0.05, min_area=5.0):
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError("点云未找到")

    facade_cache = cache.get_facade_cache(uuid_name)
    if facade_cache is None or not facade_cache.get("facades"):
        raise ValueError("当前点云尚未完成立面检测，请先执行检测")

    meta = cache.get_meta(uuid_name) or {}
    filename = meta.get("filename", uuid_name)
    point_count = int(len(pcd.points))
    facades = facade_cache["facades"]
    point_labels = np.asarray(facade_cache.get("point_labels", []), dtype=int)
    base_colors = np.asarray(facade_cache.get("base_colors", []), dtype=float)

    if len(point_labels) != point_count:
        point_labels = labels_from_facades(facades, point_count)
    if len(base_colors) != point_count:
        base_colors = (
            np.asarray(pcd.colors)
            if pcd.has_colors()
            else np.ones((point_count, 3), dtype=float) * 0.7
        )

    result_id = _facade_result_id(filename, point_count, voxel_size, min_area)
    json_path, npz_path = _facade_paths(result_id)

    payload = {
        "id": result_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_filename": filename,
        "source_uuid": uuid_name,
        "point_count": point_count,
        "voxel_size": float(voxel_size),
        "min_area": float(min_area),
        "facade_count": len(facades),
        "facades": facades,
        "json_file": os.path.basename(json_path),
        "npz_file": os.path.basename(npz_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    np.savez_compressed(
        npz_path,
        point_labels=point_labels.astype(np.int32),
        base_colors=base_colors.astype(np.float32),
    )
    print(f"[INFO] 立面检测结果已保存: {json_path}")
    return {
        "result_id": result_id,
        "json_path": json_path,
        "npz_path": npz_path,
        "facade_count": len(facades),
        "point_count": point_count,
        "source_filename": filename,
    }


def list_facade_detections():
    folder = _facade_folder()
    results = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            result_id = info.get("id") or os.path.splitext(name)[0]
            _, npz_path = _facade_paths(result_id)
            results.append({
                "id": result_id,
                "source_filename": info.get("source_filename"),
                "point_count": info.get("point_count"),
                "facade_count": info.get("facade_count"),
                "voxel_size": info.get("voxel_size"),
                "min_area": info.get("min_area"),
                "created_at": info.get("created_at"),
                "has_npz": os.path.isfile(npz_path),
            })
        except Exception as exc:
            print(f"[WARN] 读取立面检测结果失败 {path}: {exc}")
    results.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return results


def load_facade_detection(result_id, uuid_name):
    if not result_id:
        raise ValueError("缺少立面检测结果 ID")
    json_path, npz_path = _facade_paths(result_id)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"未找到立面检测结果: {result_id}")

    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError("点云未找到，请先加载对应点云")

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    point_count = int(info.get("point_count", 0))
    if len(pcd.points) != point_count:
        raise ValueError(
            f"当前点云点数 ({len(pcd.points)}) 与保存结果 ({point_count}) 不一致，"
            "请加载同名/同内容的点云后再试"
        )

    facades = info.get("facades", [])
    if not facades:
        raise ValueError("保存的立面结果为空")

    if os.path.isfile(npz_path):
        arrays = np.load(npz_path)
        point_labels = arrays["point_labels"]
        base_colors = arrays["base_colors"]
    else:
        point_labels = labels_from_facades(facades, point_count)
        base_colors = (
            np.asarray(pcd.colors)
            if pcd.has_colors()
            else np.ones((point_count, 3), dtype=float) * 0.7
        )

    start = time.time()
    response = build_facade_detection_response(
        uuid_name, facades, point_labels, base_colors
    )
    response["result_id"] = result_id
    response["loaded_from_cache"] = True
    response["load_elapsed_s"] = round(time.time() - start, 3)
    response["source_filename"] = info.get("source_filename")
    print(
        f"[INFO] 立面检测结果已加载: {result_id} · "
        f"{len(facades)} 个立面 · {point_count} 点"
    )
    return response
