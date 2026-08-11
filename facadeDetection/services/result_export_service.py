"""Atomic export of facade projection and sparse defect artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import uuid
from algorithms.facade.projection import rasterize_facade


class ResultExportService:
    def start_run(self, results_dir, project_name='', interval_size_m=20.0):
        run = Path(results_dir) / f'run_{uuid.uuid4().hex[:12]}'
        run.mkdir(parents=True, exist_ok=True)
        return run, {'project_name': project_name, 'interval_size_m': interval_size_m, 'facades': []}

    @staticmethod
    def save_manifest(run_dir, manifest):
        path = Path(run_dir) / 'manifest.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)
        return path

    def export_facade(self, results_dir, facade_id, points, colors, quality,
                      pixel_size=0.01):
        root = Path(results_dir) / f'facade_{int(facade_id):03d}'
        root.mkdir(parents=True, exist_ok=True)
        indices = np.asarray(quality.get('__global_indices') or [], dtype=int)
        pts = np.asarray(points, dtype=float)[indices]
        rgb = np.asarray(colors, dtype=float)[indices] if colors is not None else None
        signed = np.asarray(quality.get('signed_gap') or [], dtype=float)
        if len(signed) != len(indices):
            raise ValueError('signed_gap must use facade-local index space')
        raster = rasterize_facade(pts, rgb, quality['overall']['plane_model'], signed,
                                  quality.get('flatness_limit', 0.004), pixel_size)
        base = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2BGR)
        overlay = cv2.cvtColor(raster['overlay_rgba'], cv2.COLOR_RGBA2BGRA)
        composite = cv2.cvtColor(raster['base_rgb'], cv2.COLOR_RGB2RGBA)
        composite[overlay[:, :, 3] > 0] = overlay[overlay[:, :, 3] > 0]
        cv2.imwrite(str(root/'projection_rgb.png'), base); cv2.imwrite(str(root/'defect_heatmap_rgba.png'), overlay); cv2.imwrite(str(root/'defect_overlay.png'), cv2.cvtColor(composite, cv2.COLOR_RGBA2BGRA))
        np.savez_compressed(root/'heatmap_grid.npz', count=raster['count'], defect_mask=raster['defect_mask'], uv=raster['uv'])
        meta={'facade_id':int(facade_id),'pixel_size':raster['pixel_size'],'vmin':raster['vmin'],'vmax':raster['vmax'],'cmap':'turbo','flatness_limit':quality.get('flatness_limit'),'plane_model':quality['overall']['plane_model']}
        meta['interval_size_m'] = quality.get('interval_size_m', quality.get('grid_size', 20.0))
        meta['window_size_m'] = quality.get('window_size_m', quality.get('ruler_size', 2.0))
        meta['step_size_m'] = quality.get('step_size_m', quality.get('ruler_step', 0.05))
        (root/'quality.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'directory': str(root), **{k: str(root/f) for k,f in {'projection':'projection_rgb.png','heatmap':'defect_heatmap_rgba.png','overlay':'defect_overlay.png','grid':'heatmap_grid.npz','json':'quality.json'}.items()}}