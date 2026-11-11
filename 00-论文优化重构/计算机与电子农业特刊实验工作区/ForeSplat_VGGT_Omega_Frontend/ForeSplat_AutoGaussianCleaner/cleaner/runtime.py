"""Scene-independent camera loading, including the validated half-pixel interface."""
import argparse
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image


def setup(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    from utils.cuda_preload import preload_cuda_ext_deps
    preload_cuda_ext_deps()
    from scene.gaussian_model import GaussianModel
    import gaussian_renderer
    return GaussianModel, gaussian_renderer


def camera(K, E, width, height, name, uid):
    from utils.graphics_utils import getProjectionMatrix
    c = argparse.Namespace(uid=uid, image_name=name, image_width=width, image_height=height,
        Fx=float(K[0, 0]), Fy=float(K[1, 1]), Cx=float(K[0, 2]), Cy=float(K[1, 2]))
    c.FoVx = 2 * math.atan(width / (2 * c.Fx))
    c.FoVy = 2 * math.atan(height / (2 * c.Fy))
    c.R = torch.as_tensor(E[:3, :3].T.copy(), dtype=torch.float32, device='cuda')
    c.T = torch.as_tensor(E[:3, 3].copy(), dtype=torch.float32, device='cuda')
    c.world_view_transform = torch.as_tensor(E.T.copy(), dtype=torch.float32, device='cuda')
    P = getProjectionMatrix(.01, 100., c.FoVx, c.FoVy).T.cuda()
    P[2, 0] = (2 * c.Cx + 1) / width - 1
    P[2, 1] = (2 * c.Cy + 1) / height - 1
    P[0, 0], P[1, 1] = 2 * c.Fx / width, 2 * c.Fy / height
    c.projection_matrix = P
    c.full_proj_transform = c.world_view_transform @ P
    c.camera_center = c.world_view_transform.inverse()[3, :3]
    return c


class Scene:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.dataset = Path(spec['dataset'])
        self.Model, self.renderer = setup(spec['gggs_repo'])
        self.pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.background = torch.zeros(3, device='cuda')
        self.views = self._views()
        self.view_cache = {}
        self.geometry_frames = {}
        if spec.get('geometry_cameras'):
            meta = json.loads(Path(spec['geometry_cameras']).read_text())
            self.alignment = np.asarray(meta.get('scene_alignment_matrix', np.eye(4)))
            self.geometry_frames = {Path(r['image_name']).stem: r for r in meta['frames']}
        else:
            self.alignment = np.eye(4)
        if not self.views:
            raise ValueError('No cameras found')

    def _views(self):
        from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary, read_intrinsics_text, read_extrinsics_text
        sparse = self.dataset / 'sparse/0'
        if (sparse / 'cameras.bin').exists():
            intr = read_intrinsics_binary(str(sparse / 'cameras.bin'))
            ext = read_extrinsics_binary(str(sparse / 'images.bin'))
        else:
            intr = read_intrinsics_text(str(sparse / 'cameras.txt'))
            ext = read_extrinsics_text(str(sparse / 'images.txt'))
        result = []
        for uid, e in enumerate(sorted(ext.values(), key=lambda x: x.name)):
            i = intr[e.camera_id]
            if i.model == 'PINHOLE':
                fx, fy, cx, cy = i.params
            elif i.model == 'SIMPLE_PINHOLE':
                fx, cx, cy = i.params
                fy = fx
            else:
                raise ValueError(f'Undistorted PINHOLE required, got {i.model}')
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
            E = np.eye(4)
            E[:3, :3], E[:3, 3] = e.qvec2rotmat(), e.tvec
            factor = min(1., self.config['evaluation_max_side'] / max(i.width, i.height))
            w, h = round(i.width * factor), round(i.height * factor)
            A = np.array([[w/i.width, 0, (w/i.width-1)/2], [0, h/i.height, (h/i.height-1)/2], [0, 0, 1.]])
            K = A @ K
            name = Path(e.name).stem
            rgb = self.dataset / 'images' / Path(e.name).name
            mask = self.dataset / 'masks' / (name + '.png')
            if not rgb.exists() or not mask.exists():
                raise FileNotFoundError(f'Missing RGB/Mask for {name}')
            result.append(dict(name=name, uid=uid, K=K, E=E, width=w, height=h, rgb=rgb, mask=mask))
        return result

    def evidence_indices(self):
        # Farthest-point sampling on optical directions and normalized camera locations.
        n = len(self.views)
        if self.config['evidence_views'] <= 0 or self.config['evidence_views'] >= n:
            return list(range(n))
        centers = np.array([np.linalg.inv(v['E'])[:3, 3] for v in self.views])
        directions = np.array([v['E'][2, :3] for v in self.views])
        extent = max(np.linalg.norm(np.ptp(centers, axis=0)), 1e-9)
        features = np.column_stack([centers/extent, directions])
        chosen = [0]
        dist = np.full(n, np.inf)
        for _ in range(self.config['evidence_views']-1):
            dist = np.minimum(dist, ((features-features[chosen[-1]])**2).sum(1))
            dist[chosen] = -1
            chosen.append(int(np.argmax(dist)))
        return sorted(chosen)

    def get_view(self, index):
        v = self.views[index]
        c = camera(v['K'], v['E'], v['width'], v['height'], v['name'], v['uid'])
        if index in self.view_cache:
            return (c, *self.view_cache[index])
        with Image.open(v['rgb']) as im:
            rgb = np.asarray(im.convert('RGB').resize((v['width'], v['height']), Image.Resampling.BILINEAR)).astype(np.float32)/255
        with Image.open(v['mask']) as im:
            mask = np.asarray(im.convert('L').resize((v['width'], v['height']), Image.Resampling.NEAREST)) > 127
        depth, confidence = self._depth(v)
        self.view_cache[index] = (rgb, mask, depth, confidence)
        return c, rgb, mask, depth, confidence

    def _depth(self, v):
        r = self.geometry_frames.get(v['name'])
        if r is None:
            return None, None
        base = Path(self.spec['geometry_dir'])
        stem = r.get('stem', v['name'])
        dp, cp = base/f'depth_{stem}.npy', base/f'confidence_{stem}.npy'
        if not dp.exists() or not cp.exists():
            raise FileNotFoundError(f'Depth/Confidence missing: {stem}')
        d, conf = np.load(dp).squeeze(), np.load(cp).squeeze()
        A = v['K'] @ np.linalg.inv(np.asarray(r['intrinsic_model']))
        size = (v['width'], v['height'])
        depth = cv2.warpPerspective(d.astype(np.float32), A, size, flags=cv2.INTER_NEAREST)
        confidence = cv2.warpPerspective(conf.astype(np.float32), A, size, flags=cv2.INTER_NEAREST)
        # Verify external bundle and GGGS poses are in the same frame before using depth.
        raw = np.eye(4)
        raw[:3, :] = np.asarray(r['extrinsic_model_w2c'])[:3, :]
        aligned = raw @ np.linalg.inv(self.alignment)
        if not np.allclose(aligned, v['E'], atol=2e-4, rtol=2e-4):
            raise ValueError(f'External depth/GGGS pose mismatch: {v["name"]}; refusing geometry certification')
        return depth, confidence

    def load_model(self, path):
        from .io import read_ply
        rows = read_ply(path)
        nrest = sum(k.startswith('f_rest_') for k in rows.dtype.names)
        sh = round(math.sqrt(nrest / 3 + 1) - 1)
        sg = sum(k.startswith('sg_sharpness_') for k in rows.dtype.names)
        if sg:
            raise ValueError('Equivalent SuperSplat export currently requires sg_degree=0')
        model = self.Model(sh, sg)
        model.load_ply(str(path))
        for key, value in vars(model).items():
            if isinstance(value, torch.Tensor):
                value.requires_grad_(False)
        return model, rows

    def render(self, c, model):
        return self.renderer.render(c, model, self.pipe, self.background, 0., require_depth=True)
