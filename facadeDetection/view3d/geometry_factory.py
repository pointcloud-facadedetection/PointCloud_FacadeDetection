import math
import numpy as np
import open3d as o3d


def make_point_cloud(positions, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(positions, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def make_bbox(min_bound, max_bound, color=(0.3, 0.8, 1.0)):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(min_bound, dtype=np.float64),
        np.asarray(max_bound, dtype=np.float64),
    )
    bbox.color = list(color)
    return bbox


def make_normals(points, normals, length=0.5, max_lines=8000):
    pos = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if len(pos) == 0 or len(normals) != len(pos):
        return None

    step = max(1, int(math.ceil(len(pos) / max_lines)))
    p = pos[::step]
    n = normals[::step]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.vstack([p, p + n * float(length)]))
    lines.lines = o3d.utility.Vector2iVector([[i, i + len(p)] for i in range(len(p))])
    lines.colors = o3d.utility.Vector3dVector(np.tile([[0.0, 1.0, 0.55]], (len(p), 1)))
    return lines


def make_sphere(point, color, radius):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=max(radius, 1e-4), resolution=12)
    mesh.translate(np.asarray(point, dtype=np.float64))
    mesh.paint_uniform_color(color)
    return mesh


def make_axes_cross(point, color, size):
    origin = np.asarray(point, dtype=np.float64).reshape(3)
    arm = float(max(size, 1e-4))
    pts = np.vstack(
        [
            origin,
            origin + [arm, 0.0, 0.0],
            origin,
            origin - [arm, 0.0, 0.0],
            origin,
            origin + [0.0, arm, 0.0],
            origin,
            origin - [0.0, arm, 0.0],
            origin,
            origin + [0.0, 0.0, arm],
            origin,
            origin - [0.0, 0.0, arm],
        ]
    )
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pts)
    lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, 12, 2)])
    lines.colors = o3d.utility.Vector3dVector(np.tile([list(color)], (6, 1)))
    return lines


def make_pair_lines(src_points, tgt_points):
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_points, dtype=np.float64).reshape(-1, 3)
    pair_count = min(len(src), len(tgt))
    if not pair_count:
        return None

    lines = o3d.geometry.LineSet()
    pts = np.vstack([src[:pair_count], tgt[:pair_count]])
    lines.points = o3d.utility.Vector3dVector(pts)
    lines.lines = o3d.utility.Vector2iVector([[i, i + pair_count] for i in range(pair_count)])
    lines.colors = o3d.utility.Vector3dVector(np.tile([[1.0, 1.0, 1.0]], (pair_count, 1)))
    return lines
