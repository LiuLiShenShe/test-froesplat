"""Fast3R local geometry for the unmodified official AMB3R SfM pipeline.

Each call creates a new first-view reference frame. Only AMB3R's official
registration/memory code connects these frames. No adapter point averaging.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'fast3r'))


class Fast3RForAMB3RSfM(nn.Module):
    """Public confidence shape is B,T,H,W,1; PipelineBridge handles source ABI.

    Global descriptors are mean-pooled original CroCo encoder patch tokens.
    The official clustering code performs whitening/normalization itself.
    """
    def __init__(self, weight_path=None, head_parallel_views=2, encoder_chunk_size=8):
        super().__init__()
        from fast3r.models.fast3r import Fast3R
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
        self.model = Fast3R.from_pretrained(str(weight_path or ROOT / 'weights' / 'fast3r'))
        self.model.set_max_parallel_views_for_head(head_parallel_views)
        MultiViewDUSt3RLitModule.load_for_inference(self.model).eval()
        self.model.eval()
        # Official pipeline passes this value as autocast device TYPE.
        self.device = 'cuda'
        self.name = 'fast3r'
        self.metric_scale = False
        self.encoder_chunk_size = encoder_chunk_size
        self.last_geometry_validation = None
        self.last_pose_status = None

    def _check_images(self, frames):
        images = frames['images']
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError('frames[images] must have shape (B,T,3,H,W)')
        if images.min() < -1 or images.max() > 1:
            raise ValueError('Fast3R input must be in [-1,1]')
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError('Pad images and masks to multiples of 16 before the adapter')
        return images

    @torch.inference_mode()
    def extract_amb3r_sfm_features(self, frames):
        images = self._check_images(frames)
        flat = images.flatten(0, 1)
        descriptors = []
        for start in range(0, len(flat), self.encoder_chunk_size):
            chunk = flat[start:start + self.encoder_chunk_size]
            shapes = torch.tensor(chunk.shape[-2:], device=chunk.device).repeat(len(chunk), 1)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=chunk.is_cuda):
                tokens, _ = self.model.encoder(chunk, shapes)
            descriptors.append(tokens.float().mean(dim=1))
        # Official SfM currently uses B=1 and requires (T,D), not (1,T,D).
        return torch.cat(descriptors, dim=0)

    @torch.inference_mode()
    def run_amb3r_sfm(self, frames, cfg=None, keyframe_memory=None, benchmark_conf0=None):
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
        from frontends.fast3r.run import observe_pnp, validate_geometry
        images = self._check_images(frames)
        b, t, _, h, w = images.shape
        if b != 1:
            raise ValueError('The official AMB3R SfM pipeline supports one scene per run (B=1)')
        views = [dict(img=images[:, i], true_shape=torch.tensor([[h, w]]),
                      idx=i, instance=str(i)) for i in range(t)]
        result = inference(views, self.model, images.device, dtype=torch.bfloat16,
                           verbose=False, profiling=False)
        preds = result['preds']
        # Observer records real failures; official identity fallback is unchanged.
        with observe_pnp(preds) as pose_status:
            poses, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
                preds, niter_PnP=100,
                focal_length_estimation_method='first_view_from_global_head')
        poses = np.asarray(poses[0], dtype=np.float32)
        focals = np.asarray(focals[0], dtype=np.float32)
        k = np.repeat(np.eye(3, dtype=np.float32)[None], t, axis=0)
        k[:, 0, 0] = k[:, 1, 1] = focals
        k[:, 0, 2] = w / 2
        k[:, 1, 2] = h / 2
        # An independent numerical check uses inversion and pixel reprojection.
        masks = frames.get('masks')
        if masks is None:
            masks = np.ones((t, h, w), dtype=bool)
        elif hasattr(masks, 'detach'):
            masks = masks.detach().cpu().numpy()
        self.last_geometry_validation = validate_geometry(preds, poses, focals, masks, pose_status)
        incomplete = [i for i, record in enumerate(pose_status)
                      if record['status'] in ('not_called', 'started')]
        self.last_geometry_validation['pnp_observation'] = dict(
            expected_views=t, recorded_views=len(pose_status),
            actual_success_count=sum(record['status'] == 'success' for record in pose_status),
            actual_failure_count=sum(record['status'] == 'failed_identity_fallback' for record in pose_status),
            instrumentation_incomplete_indices=incomplete,
            interpretation='not_called/started indicate incomplete observation, not solver failure')
        self.last_pose_status = pose_status
        confidence = torch.stack([p['conf'] for p in preds], dim=1).float()
        if not torch.isfinite(confidence).all() or (confidence < 1).any():
            raise ValueError('Unexpected Fast3R confidence outside finite [1,+inf); no remapping applied')
        return dict(
            world_points=torch.stack([p['pts3d_in_other_view'] for p in preds], dim=1).float(),
            world_points_conf=confidence.unsqueeze(-1),
            pose=torch.from_numpy(poses).unsqueeze(0),
            intrinsic=torch.from_numpy(k).unsqueeze(0),
            focal=torch.from_numpy(focals).unsqueeze(0),
            local_points=torch.stack([p['pts3d_local'] for p in preds], dim=1).float(),
            local_confidence=torch.stack([p['conf_local'] for p in preds], dim=1).float(),
            benchmark_conf0=confidence[0].mean(dim=(1, 2)),
        )


class PipelineBridge(nn.Module):
    """Adapt README's trailing singleton confidence dimension to actual source.

    Neither confidence values nor geometry change. This is necessary because
    official SfMemory allocates T,H,W and reduces dimensions (1,2) throughout.
    """
    def __init__(self, model, descriptor_fp32=False):
        super().__init__()
        self.model = model
        self.device = model.device
        self.name = model.name
        self.metric_scale = getattr(model, 'metric_scale', False)
        self.descriptor_fp32 = descriptor_fp32
        self.last_descriptor_original_dtype = None

    def extract_amb3r_sfm_features(self, frames):
        result = self.model.extract_amb3r_sfm_features(frames)
        self.last_descriptor_original_dtype = str(result.dtype)
        # BF16 -> FP32 preserves every existing feature value exactly. Only
        # downstream official covariance/whitening arithmetic gains precision.
        return result.float() if self.descriptor_fp32 else result

    def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None, benchmark_conf0=None):
        result = self.model.run_amb3r_sfm(frames, cfg, keyframe_memory, benchmark_conf0)
        if result is None:
            return None
        result = dict(result)
        conf = result['world_points_conf']
        if conf.ndim == 5 and conf.shape[-1] == 1:
            result['world_points_conf'] = conf.squeeze(-1)
        if result['world_points_conf'].ndim != 4:
            raise ValueError('Official AMB3R SfM needs confidence shape B,T,H,W')
        return result
