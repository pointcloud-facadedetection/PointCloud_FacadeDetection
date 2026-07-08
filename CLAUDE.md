# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A web-based point cloud visualization and registration system (大规模点云可视化与配准系统) for building facade detection from FARO laser scans. The pipeline: FARO `.fls` scans → PLY conversion (via CloudCompare) → web app for interactive visualization, denoising, and multi-scan registration.

## Running

```bash
python app.py          # Flask dev server on http://localhost:5000 (override with PORT env var)
```

Note: on macOS, port 5000 is often taken by ControlCenter (AirPlay) — use `PORT=5050 python app.py`.

Dependencies (no requirements.txt exists): `flask`, `flask-cors`, `open3d`, `numpy`.

FLS → PLY conversion is Windows-only and requires CloudCompare installed:

```bash
python flstoply.py <input.fls|dir> <output.ply|dir> --overwrite   # CLI tool, auto-locates CloudCompare.exe
```

`convert_fls2ply.py` is an older batch-conversion variant with hardcoded `D:\` paths.

There are no tests or linters.

## Architecture

Two-file application: `app.py` (Flask + Open3D backend) and `templates/index.html` (~1900-line single-file frontend with inline CSS/JS, Three.js 0.160 loaded from CDN via import map). The frontend calls the backend same-origin (`API_BASE = ''`), since Flask serves the page itself.

### Binary point cloud protocol

Endpoints that return cloud geometry (`/upload`, `/denoise`, `/compute_normals`, `/apply_registration`, `/icp_refine`, `/merge_clouds` with `return_data: true`) respond with `application/octet-stream`, not JSON. Layout (little-endian):

```
[uint32 header length][JSON header, space-padded to 4-byte alignment]
[positions: N×3 float32][normals: N×3 float32, only if has_normals][colors: N×3 uint8]
```

The JSON header carries `point_count`, `has_normals`, plus metadata (uuid, filename, transformation, rmse, …). Serialization: `pcd_to_binary()` in app.py; parsing: `parseBinaryCloud()` / `fetchCloud()` in index.html. Errors still return JSON with a 4xx/5xx status — `fetchCloud` dispatches on Content-Type. `/upload` takes ONE file per request (field name `file`); the frontend loops over selected files. `/merge_clouds` returns JSON metadata only unless `return_data: true` is passed (the save flow never needs the merged geometry client-side).

### Coordinate system convention (critical)

- **Backend (Open3D): Z-up.** All cached point clouds and transforms are Z-up.
- **Frontend (Three.js): Y-up.** Conversion is `(x, y, z)_zup ↔ (x, z, -y)_yup`.
- Conversion helpers exist on both sides: `to_zup`/`to_yup` in app.py, `convertZupToYup*` in index.html. `pcd_to_json` returns raw Z-up arrays; the frontend converts on load. Correspondence points picked in the browser are converted to Z-up before being stored via `/register_correspondences`. Any new endpoint returning geometry must keep this convention consistent.

### Backend state (in-memory, per-process)

All state lives in module-level dicts keyed by a uuid-prefixed filename — nothing persists across restarts:

- `CACHED_CLOUDS`: current working cloud (may be transformed by registration)
- `ORIGINAL_DOWNSAMPLED`: untransformed copy, used as the source for every registration so transforms don't compound
- `CLOUD_META`: original filename
- `REG_PAIRS`: correspondence point pairs, `src_uuid → tgt_uuid → [{src, tgt}]` (Z-up), cleared after use

Uploaded and saved PLY files go to `uploads/`.

### Registration pipeline

1. User picks ≥3 correspondence point pairs in the browser (raycast picking in registration mode) → `POST /register_correspondences`
2. `POST /apply_registration` — coarse alignment via SVD/Kabsch on the point pairs; returns transformed cloud + 4×4 matrix + RMSE/overlap
3. `POST /icp_refine` — fine alignment: multi-scale point-to-plane ICP (coarse → medium → fine voxel sizes, then a decreasing-threshold loop), initialized with the coarse transform passed from the frontend
4. `POST /merge_clouds` (visual overlay, colors clouds red/blue) and `POST /save_registered` / `GET /download/<filename>` for export

Both registration steps transform `ORIGINAL_DOWNSAMPLED[src]` (not the current cached cloud) and overwrite `CACHED_CLOUDS[src]` with the result.

### Other endpoints

`/upload` (multi-file PLY, voxel downsampling), `/denoise` (radius or statistical outlier removal, falls back if >90% of points removed), `/compute_normals`, `/boundingbox`.
