import sys
import os
import torch
import torch.nn as nn
import numpy as np
from .model import AMB3R

_THIRDPARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty')
sys.path.append(_THIRDPARTY)

def load_model(model_name, ckpt_path=None):
    if model_name == 'amb3r':
        model = AMB3R()
        if ckpt_path is not None:
            model.load_weights(ckpt_path)

    elif model_name == 'da3':
        model = DA3(ckpt_path=ckpt_path) if ckpt_path is not None else DA3()

    elif model_name == 'omega':
        model = VGGT_Omega(ckpt_path=ckpt_path)

    elif model_name == 'dvlt':
        model = DVLTWrapper(ckpt_path=ckpt_path)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


class DA3(nn.Module):
    def __init__(self, device='cuda', ckpt_path="depth-anything/DA3NESTED-GIANT-LARGE"):
        super().__init__()

        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.geometry import unproject_depth

        self.model = DepthAnything3.from_pretrained(ckpt_path).to(device).eval()


        self.device = device
        self.name = 'da3'
    

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        def select_by_index(l, idx):
            """Select an element from a list by an index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(idx, int):
                ret = l[idx]
            else:
                indices = idx
                ret = []
                for batch_idx, idx in enumerate(indices):
                    ret.append(l[idx][batch_idx])

                if isinstance(ret[0], np.ndarray):
                    ret = np.stack(ret, 0)
                else:
                    ret = torch.stack(ret, 0)

            return ret

        def exclude_index(l, exclude_idx):
            """Selects all element from a list, excluding a specific index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(exclude_idx, int):
                ret = [ele for idx, ele in enumerate(l) if idx != exclude_idx]
            else:
                exclude_indices = exclude_idx
                ret = []
                for batch_idx, exclude_idx in enumerate(exclude_indices):
                    ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != exclude_idx])

                transposed = list(zip(*ret))
                if isinstance(transposed[0][0], np.ndarray):
                    ret = [np.stack(ele, 0) for ele in transposed]
                else:
                    ret = [torch.stack(ele, 0) for ele in transposed]

            return ret
    
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source
        images = np.stack(images, 1) / 255.0  # Normalize images to [0, 1]
        images = images * 2.0 - 1.0  # Normalize to [-1, 1]

        images = torch.from_numpy(images).float()
        
        # Ensure channel dimension is at position 1: (B, C, H, W) or (B, N, C, H, W)
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3) if images.dim() == 5 else images.permute(0, 3, 1, 2)


        frames_ = {
            'images': images.to(self.device),
            'keyview_idx': keyview_idx
        }

        frames = {
            'frames': frames_,
        }

        return frames

    
    def _normalize_images(self, imgs):
        """Normalize images from [-1, 1] to ImageNet-normalized range."""
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 1, 3, 1, 1)
        return (imgs - mean) / std

    def forward(self, frames):
        # input should be (-1, 1)
        imgs = frames['images']  # (B, N, 3, H, W)
        B, nimgs, three, H, W = imgs.shape
        imgs = self._normalize_images(imgs)

        raw_output = self.model.forward(imgs, None, None, [], False, 
                                        use_ray_pose=False, ref_view_strategy='first')

        # Convert raw output to prediction
        prediction = self.model._convert_to_prediction(raw_output)

        # Align prediction to extrinsics
        prediction = self.model._align_to_input_extrinsics_intrinsics(
            None, None, prediction, True
        )

        # Build c2w from w2c (N, 3, 4) -> (N, 4, 4) -> invert
        w2c_34 = torch.from_numpy(prediction.extrinsics).to(imgs.device)  # (N, 3, 4)
        N = w2c_34.shape[0]
        bottom = torch.tensor([0, 0, 0, 1], dtype=w2c_34.dtype, device=imgs.device).view(1, 1, 4).expand(N, -1, -1)
        w2c = torch.cat([w2c_34, bottom], dim=1)  # (N, 4, 4)
        c2w = torch.inverse(w2c)  # (N, 4, 4)

        # Build depth and intrinsics tensors
        intrinsics_t = torch.from_numpy(prediction.intrinsics).unsqueeze(0).to(imgs.device)  # (1, N, 3, 3)
        depth_t = torch.from_numpy(prediction.depth).unsqueeze(0).unsqueeze(-1).to(imgs.device)  # (1, N, H, W, 1)

        # Unproject depth to world points
        from depth_anything_3.utils.geometry import unproject_depth
        world_points = unproject_depth(depth_t, intrinsics_t, c2w.unsqueeze(0))  # (1, N, H, W, 3)
        conf = torch.from_numpy(prediction.conf).unsqueeze(0).to(imgs.device)  # (1, N, H, W)

        return {
            'world_points': world_points.float(),        # (1, N, H, W, 3)
            'world_points_conf': conf.float(),            # (1, N, H, W)
            'depth': depth_t.squeeze(-1).float(),         # (1, N, H, W)
            'pose': c2w.unsqueeze(0).float(),             # (1, N, 4, 4)
        }
    
    
    def output_adapter(self, model_output):
        aux = {}

        # model_output is the dict returned by forward()
        depth = model_output['depth'][:, 0].cpu().numpy()  # (1, H, W)

        pred = {
            'depth': depth[None]  # (1, 1, H, W)
        }

        return pred, aux
    

    def run_amb3r_benchmark(self, frames):
        return self.forward(frames)

    
    @torch.inference_mode()
    def run_amb3r_vo(self, frames, cfg, keyframe_memory):
        return self.forward(frames)


class VGGT_Omega(nn.Module):
    def __init__(self, device='cuda', ckpt_path=None, image_resolution=512):
        super().__init__()

        vggt_omega_path = os.path.join(_THIRDPARTY, 'vggt-omega')
        if vggt_omega_path not in sys.path:
            sys.path.insert(0, vggt_omega_path)

        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri

        self.model = VGGTOmega().eval().to(device)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self.device = device
        self.name = 'vggt_omega'

    def _unproject_depth(self, depth, extrinsic, intrinsic):
        # depth:     (B, N, H, W)
        # extrinsic: (B, N, 3, 4)  w2c
        # intrinsic: (B, N, 3, 3)
        B, N, H, W = depth.shape
        dev = depth.device

        y, x = torch.meshgrid(
            torch.arange(H, device=dev, dtype=depth.dtype),
            torch.arange(W, device=dev, dtype=depth.dtype),
            indexing='ij',
        )

        fx = intrinsic[:, :, 0, 0, None, None]
        fy = intrinsic[:, :, 1, 1, None, None]
        cx = intrinsic[:, :, 0, 2, None, None]
        cy = intrinsic[:, :, 1, 2, None, None]

        cam_pts = torch.stack([
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ], dim=-1)  # (B, N, H, W, 3)

        R_T = extrinsic[:, :, :3, :3].transpose(-1, -2)          # (B, N, 3, 3)
        t   = extrinsic[:, :, :3,  3][:, :, None, None, :]       # (B, N, 1, 1, 3)

        return torch.einsum('bnij,bnhwj->bnhwi', R_T, cam_pts - t)  # (B, N, H, W, 3)

    def forward(self, frames):
        images = frames['images'].to(self.device)   # (B, T, C, H, W)  in [-1, 1]
        images = (images + 1.0) / 2.0               # -> [0, 1] for VGGT-Omega

        with torch.inference_mode():
            predictions = self.model(images)

        extrinsic, intrinsic = self._pose_encoding_to_extri_intri(
            predictions['pose_enc'],
            predictions['images'].shape[-2:],
        )
        # extrinsic: (B, N, 3, 4) w2c;  intrinsic: (B, N, 3, 3)

        depth = predictions['depth'][..., 0]        # (B, N, H, W)
        depth_conf = predictions['depth_conf']      # (B, N, H, W)

        world_points = self._unproject_depth(depth, extrinsic, intrinsic)  # (B, N, H, W, 3)

        # Build c2w (4x4) by inverting w2c
        B, N = extrinsic.shape[:2]
        bottom = torch.tensor([[0, 0, 0, 1]], dtype=extrinsic.dtype, device=extrinsic.device)
        bottom = bottom.view(1, 1, 1, 4).expand(B, N, -1, -1)
        w2c = torch.cat([extrinsic, bottom], dim=2)     # (B, N, 4, 4)
        c2w = torch.linalg.inv(w2c)                     # (B, N, 4, 4)

        return {
            'world_points': world_points.float(),           # (B, N, H, W, 3)
            'world_points_conf': depth_conf.float(),        # (B, N, H, W)
            'depth': depth.float(),                         # (B, N, H, W)
            'pose': c2w.float(),                            # (B, N, 4, 4)
            'pts3d_by_unprojection': world_points.float(),  # (B, N, H, W, 3)
        }

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        def select_by_index(l, idx):
            if isinstance(idx, int):
                return l[idx]
            ret = []
            for batch_idx, i in enumerate(idx):
                ret.append(l[i][batch_idx])
            return np.stack(ret, 0) if isinstance(ret[0], np.ndarray) else torch.stack(ret, 0)

        def exclude_index(l, exclude_idx):
            if isinstance(exclude_idx, int):
                return [ele for idx, ele in enumerate(l) if idx != exclude_idx]
            ret = []
            for batch_idx, ei in enumerate(exclude_idx):
                ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != ei])
            transposed = list(zip(*ret))
            return [np.stack(e, 0) for e in transposed] if isinstance(transposed[0][0], np.ndarray) \
                else [torch.stack(e, 0) for e in transposed]

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        imgs = np.stack([image_key] + images_source, axis=1) / 255.0  # (B, N, H, W, 3) in [0,1]
        imgs = (imgs * 2.0 - 1.0).astype(np.float32)                  # -> [-1, 1]
        imgs = torch.from_numpy(imgs)
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)

        return {'frames': {'images': imgs.to(self.device), 'keyview_idx': keyview_idx}}

    def output_adapter(self, model_output):
        depth = model_output['depth'][:, 0].cpu().numpy()  # (B, H, W), keyview is index 0
        return {'depth': depth[None]}, {}                  # (1, B, H, W) → (1,1,H,W) when B=1

    def run_amb3r_benchmark(self, frames):
        return self.forward(frames)

    @torch.inference_mode()
    def run_amb3r_vo(self, frames, cfg, keyframe_memory):
        return self.forward(frames)

    @torch.inference_mode()
    def extract_amb3r_sfm_features(self, views):
        images = views['images'].to(self.device)  # (B, N, 3, H, W) in [-1, 1]
        images = (images + 1.0) / 2.0             # -> [0, 1]
        B, N, C, H, W = images.shape
        agg = self.model.aggregator
        imgs_flat = images.view(B * N, C, H, W)
        imgs_flat = (imgs_flat - agg._resnet_mean.view(1, 3, 1, 1)) / agg._resnet_std.view(1, 3, 1, 1)
        patch_tokens = agg.patch_embed(imgs_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens['x_norm_patchtokens']
        # (B*N, num_patches, C) -> mean over patches -> (B*N, C)
        return patch_tokens.mean(dim=1).float()

    @torch.inference_mode()
    def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None, benchmark_conf0=None):
        return self.forward(frames)


class DVLTWrapper(nn.Module):
    def __init__(self, device='cuda', ckpt_path=None, img_size=504):
        super().__init__()

        try:
            from dvlt.model.dvlt.model import DVLT
            from dvlt.common.constants import DataField
        except ImportError:
            raise ImportError("DVLT not found. Please ensure dvlt is installed in thirdparty/dvlt")

        dvlt_wrapper = DVLT(img_size=img_size)
        self.model = dvlt_wrapper.model  # the actual DVLTModel (nn.Module)
        if ckpt_path is not None and ckpt_path != 'None':
            with open(ckpt_path, 'rb') as f:
                magic = f.read(8)
            if magic[:1] == b'\xd8' or ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                state_dict = torch.load(ckpt_path, map_location='cpu')
                if 'model' in state_dict and isinstance(state_dict['model'], dict):
                    state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=True)

        self.model.eval().to(device)
        self.device = device
        self.name = 'dvlt'
        self.img_size = img_size
        self.DataField = DataField

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        def select_by_index(l, idx):
            if isinstance(idx, int):
                return l[idx]
            ret = []
            for batch_idx, i in enumerate(idx):
                ret.append(l[i][batch_idx])
            return np.stack(ret, 0) if isinstance(ret[0], np.ndarray) else torch.stack(ret, 0)

        def exclude_index(l, exclude_idx):
            if isinstance(exclude_idx, int):
                return [ele for idx, ele in enumerate(l) if idx != exclude_idx]
            ret = []
            for batch_idx, ei in enumerate(exclude_idx):
                ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != ei])
            transposed = list(zip(*ret))
            return [np.stack(e, 0) for e in transposed] if isinstance(transposed[0][0], np.ndarray) \
                else [torch.stack(e, 0) for e in transposed]

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        # Stack and normalize to [-1, 1] (benchmark convention)
        imgs = np.stack([image_key] + images_source, axis=1) / 255.0  # (B, N, H, W, 3) in [0,1]
        imgs = (imgs * 2.0 - 1.0).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)

        # Key must match forward(self, batch) parameter name so model(**sample) works
        return {'batch': {'images': imgs.to(self.device)}}

    def _run_model(self, images):
        """Run DVLTModel following _postprocess_predictions. images: (B, S, C, H, W) in [-1, 1]."""
        from dvlt.common.rays import rays_to_pose
        from dvlt.common.geometry import depth_to_world_coords_points

        # DVLTModel._encode_images expects [0, 1]
        images_01 = (images + 1.0) / 2.0
        H, W = images_01.shape[-2:]

        predictions = self.model.forward_inference(images_01)

        depth = predictions['depth'].squeeze(-1)            # (B, S, H, W)
        depth_conf = predictions['depth_conf']              # (B, S, H, W)

        # Pose fitting: uniform weights (use_depth_conf_for_pose=False by default)
        rays = predictions['rays'].float()                  # (B, S, H, W, 6)
        pose_conf = torch.ones_like(depth_conf).float()
        with torch.autocast(device_type='cuda', enabled=False):
            extrinsics_c2w, intrinsics = rays_to_pose(
                rays, pose_conf, H, W, self.model.patch_size
            )
        # extrinsics_c2w: (B, S, 4, 4) c2w,  intrinsics: (B, S, 3, 3)

        # World points via proper unprojection through fitted camera (not ray composition)
        with torch.autocast(device_type='cuda', enabled=False):
            world_points, _, _ = depth_to_world_coords_points(
                depth.float(), extrinsics_c2w, intrinsics.float()
            )
        # world_points: (B, S, H, W, 3)

        return {
            'world_points': world_points.float(),           # (B, S, H, W, 3)
            'pts3d_by_unprojection': world_points.float(),  # (B, S, H, W, 3)
            'world_points_conf': depth_conf.float(),        # (B, S, H, W)
            'depth': depth.float(),                         # (B, S, H, W)
            'pose': extrinsics_c2w.float(),                 # (B, S, 4, 4)
        }

    def forward(self, batch):
        images = batch.get('images') if 'images' in batch else batch.get(self.DataField.IMAGES)
        return self._run_model(images.to(self.device))

    def output_adapter(self, model_output):
        depth = model_output['depth'][:, 0].cpu().numpy()  # (B, H, W)
        return {'depth': depth[None]}, {}

    def run_amb3r_benchmark(self, frames):
        return self.forward(frames)

    @torch.inference_mode()
    def run_amb3r_vo(self, frames, cfg, keyframe_memory):
        return self.forward(frames)

    @torch.inference_mode()
    def extract_amb3r_sfm_features(self, views):
        images = views['images'].to(self.device)            # (B, N, 3, H, W) in [-1, 1]
        images_01 = (images + 1.0) / 2.0                   # -> [0, 1] as DVLTModel expects
        # _encode_images handles ImageNet normalization internally
        patch_tokens = self.model._encode_images(images_01) # (B*N, num_patches, C)
        return patch_tokens.mean(dim=1).float()

    @torch.inference_mode()
    def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None, benchmark_conf0=None):
        return self.forward(frames)
