#!/usr/bin/env python3
"""Read-only historical checkpoint rendering; writes solely into a new audit run.

GPU1 only. No optimizer, training, model frontend inference or checkpoint writes.
GGGS Scene writes multi_view.json even when loading, so its model_path is an
audit-owned directory with a read-only symlink to the original trained PLY.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

WORKSPACE = Path('/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区')
OMEGA = WORKSPACE / 'ForeSplat_VGGT_Omega_Frontend'
STANDARD = WORKSPACE / 'ForeSplat_Standard_VGGT_Frontend'
STD = STANDARD / 'experiments/KongQueZhuYu/standard_vggt_comparison'
AUDIT = OMEGA / 'experiments/KongQueZhuYu/root_cause_audit'
SOURCE_IDS = [0, 19, 38, 57, 76, 95, 114, 133, 152, 170, 189, 209]


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def json_write(path, value):
    def portable(x):
        if isinstance(x, np.ndarray): return portable(x.tolist())
        if isinstance(x, np.generic): return portable(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {k: portable(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [portable(v) for v in x]
        if isinstance(x, Path): return str(x)
        return x
    Path(path).write_text(json.dumps(portable(value), ensure_ascii=False, indent=2, allow_nan=False))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def camera_record(cam):
    width, height = int(cam.image_width), int(cam.image_height)
    record = dict(name=cam.image_name, width=width, height=height,
                  source_id=int(Path(cam.image_name).stem.split('_')[-1]),
                  FoVx=float(cam.FoVx), FoVy=float(cam.FoVy),
                  fov_derived_fx=width/(2*math.tan(cam.FoVx/2)),
                  fov_derived_fy=height/(2*math.tan(cam.FoVy/2)),
                  gt_tensor_shape=list(cam.original_image.shape))
    for key in ('Fx', 'Fy', 'Cx', 'Cy'):
        record[key] = float(getattr(cam, key)) if hasattr(cam, key) else None
    for key in ('R', 'T', 'world_view_transform', 'projection_matrix',
                'full_proj_transform', 'camera_center'):
        record[key] = numpy(getattr(cam, key))
    return record


def controlled_effective_opacity_subset(gaussians, threshold, output):
    """Retain exact rows of every Gaussian parameter; never use an optimizer."""
    count=int(gaussians.get_xyz.shape[0])
    with torch.no_grad():
        raw=gaussians.get_opacity.detach().reshape(-1)
        effective=gaussians.get_opacity_with_3D_filter.detach().reshape(-1)
        if not torch.isfinite(effective).all():raise ValueError('Nonfinite effective opacity; preserve source and stop control')
        keep=effective>=threshold
        retained=int(keep.sum().item())
        if not retained:raise ValueError('Control would remove all Gaussians')
        summary=dict(threshold=float(threshold),before_count=count,after_count=retained,
            removed_count=count-retained,removed_fraction=(count-retained)/count,
            raw_opacity_ge_threshold_effective_below_count=int(((raw>=threshold)&~keep).sum().item()),
            effective_formula='native get_opacity_with_3D_filter: sigmoid(raw_opacity) * prod(scale/hypot(scale,filter_3D))',
            selected_fields=[],optimizer_used=False,remaining_gaussian_parameters_bit_identical=True)
        np.savez_compressed(output/'effective_opacity_selection.npz',keep_mask=numpy(keep),
            retained_source_indices=numpy(torch.nonzero(keep,as_tuple=False).reshape(-1)))
        fields=('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation',
                '_sg_axis','_sg_sharpness','_sg_color','filter_3D')
        for name in fields:
            value=getattr(gaussians,name)
            if int(value.shape[0])!=count:raise ValueError(f'Unexpected per-Gaussian field shape: {name} {value.shape}')
            selected=value.detach()[keep].contiguous()
            tensor=torch.nn.Parameter(selected,requires_grad=False) if isinstance(value,torch.nn.Parameter) else selected
            setattr(gaussians,name,tensor)
            if not torch.equal(tensor,selected):raise RuntimeError(f'Control changed retained values: {name}')
            selected_cpu=numpy(selected)
            summary['selected_fields'].append(dict(name=name,shape=list(selected.shape),dtype=str(selected.dtype),
                retained_values_sha256=hashlib.sha256(selected_cpu.tobytes()).hexdigest()))
        if int(gaussians.get_xyz.shape[0])!=retained:raise RuntimeError('Control cardinality mismatch')
    json_write(output/'effective_opacity_subset_control.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('E1', 'E3', 'E5', 'E6'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-ids', type=int, nargs='+', default=SOURCE_IDS)
    parser.add_argument('--effective-opacity-threshold',type=float,
        help='Explicit isolated control: remove Gaussian rows with native 3D-filter effective opacity below threshold')
    args = parser.parse_args()
    mode = args.experiment
    if mode in ('E1', 'E3'):
        dataset_path = OMEGA / 'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
        original_model = OMEGA / 'experiments/KongQueZhuYu' / ('E1_GGGS_Baseline' if mode=='E1' else 'E3_GGGS_Foreground') / 'checkpoints'
    else:
        dataset_path = STD / 'Standard_VGGT_Dataset'
        original_model = STD / ('E5_Standard_VGGT_GGGS_Baseline' if mode=='E5' else 'E6_2DGS_Standard_VGGT') / 'checkpoints'
    repo = OMEGA / 'gggs_community' if mode != 'E6' else Path('/data/fj/F2DMAS/2d-gaussian-splatting-main')
    out = args.output or AUDIT / 'render_error_analysis' / 'checkpoint_rerender_v1' / mode
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Preserve existing audit: {out}')
    if args.effective_opacity_threshold is not None and (mode not in ('E1','E5') or args.output is None):
        raise ValueError('Effective-opacity control requires explicit new output and E1/E5 checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    runtime_model = out / 'runtime_checkpoint'
    linked = runtime_model / 'point_cloud/iteration_30000'
    linked.mkdir(parents=True)
    checkpoint = original_model / ('final_gaussians/point_cloud.ply' if mode=='E3' else 'point_cloud/iteration_30000/point_cloud.ply')
    (linked / 'point_cloud.ply').symlink_to(checkpoint)
    sys.path.insert(0, str(repo))
    if mode != 'E6':
        from utils.cuda_preload import preload_cuda_ext_deps
        preload_cuda_ext_deps()
    import gaussian_renderer as native
    from scene import Scene, GaussianModel
    namespace = argparse.Namespace(sh_degree=3, sg_degree=0, source_path=str(dataset_path),
        model_path=str(runtime_model), images='images', dataset='', resolution=2,
        white_background=False, data_device='cpu', eval=False,
        use_decoupled_appearance=0, disable_filter3D=False, kernel_size=0.0,
        multi_view_num=8, multi_view_max_angle=30, multi_view_min_dis=.01,
        multi_view_max_dis=1.5)
    pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                                  debug=False, depth_ratio=0.0)
    torch.set_num_threads(8)
    started = time.perf_counter()
    original_guard = {}
    for name in ('multi_view.json', 'cameras.json', 'cfg_args'):
        f = original_model/name
        if f.exists(): original_guard[name] = sha256(f)
    gaussians = GaussianModel(3, 0) if mode != 'E6' else GaussianModel(3)
    scene = Scene(namespace, gaussians, load_iteration=30000, shuffle=False)
    if mode=='E3' and int(gaussians.get_xyz.shape[0])!=1036619:
        raise ValueError(f'Wrong E3 post-prune checkpoint count: {gaussians.get_xyz.shape[0]}')
    control=None
    if args.effective_opacity_threshold is not None:
        control=controlled_effective_opacity_subset(gaussians,args.effective_opacity_threshold,out)
    cameras = scene.getTrainCameras()
    records = [camera_record(cam) for cam in cameras]
    json_write(out/'camera_runtime.json', dict(experiment=mode, camera_count=len(records),
                resolution_divisor=2, data_device='cpu (GT placement only; camera matrices and renderer use GPU1)',
                records=records))
    np.savez_compressed(out/'camera_runtime.npz',
        names=np.asarray([r['name'] for r in records]), source_ids=np.asarray([r['source_id'] for r in records]),
        width=np.asarray([r['width'] for r in records]), height=np.asarray([r['height'] for r in records]),
        **{key:np.asarray([r[key] if r[key] is not None else np.nan for r in records]) for key in ('R','T','FoVx','FoVy',
            'Fx','Fy','Cx','Cy','world_view_transform','projection_matrix','full_proj_transform','camera_center')})
    name_to_camera = {int(Path(cam.image_name).stem.split('_')[-1]):cam for cam in cameras}
    missing = sorted(set(args.source_ids)-set(name_to_camera))
    if missing: raise ValueError(f'Missing common source IDs: {missing}')
    context = {}
    if mode == 'E6':
        OriginalRasterizer = native.GaussianRasterizer
        class ObservedRasterizer(OriginalRasterizer):
            def __call__(self, *rargs, **kwargs):
                result = super().__call__(*rargs, **kwargs)
                context['allmap'] = result[2].detach()
                return result
        native.GaussianRasterizer = ObservedRasterizer
    background = torch.zeros(3, device='cuda', dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    views = out/'views'; views.mkdir()
    view_records = []
    for source_id in args.source_ids:
        camera = name_to_camera[source_id]
        begin = time.perf_counter()
        with torch.no_grad(): package = native.render(camera, gaussians, pipeline, background, kernel_size=0.0) if mode!='E6' else native.render(camera, gaussians, pipeline, background)
        torch.cuda.synchronize()
        positive_radii=numpy(package['radii'][package['radii']>0]).astype(np.float64)
        radius_summary=dict(total_gaussians=int(gaussians.get_xyz.shape[0]),
            positive_radii_count=int(len(positive_radii)),positive_radii_fraction=len(positive_radii)/int(gaussians.get_xyz.shape[0]),
            visibility_filter_count=int(package['visibility_filter'].sum().item()),
            units='actual native rasterizer screen-space radius in pixels',
            quantiles=np.quantile(positive_radii,[.5,.9,.99]).tolist() if len(positive_radii) else None,
            maximum=float(positive_radii.max()) if len(positive_radii) else None,
            positive_radii_gt20_fraction=float((positive_radii>20).mean()) if len(positive_radii) else None)
        raw = numpy(package['render']).transpose(1,2,0).astype(np.float32)
        rgb = np.clip(raw, 0, 1)
        gt = numpy(camera.original_image[:3]).transpose(1,2,0).astype(np.float32)
        alpha = numpy(package['mask'] if mode!='E6' else package['rend_alpha']).squeeze().astype(np.float32)
        native_normal = numpy(package['normal'] if mode!='E6' else package['rend_normal']).transpose(1,2,0).astype(np.float32)
        normal_camera = native_normal if mode!='E6' else native_normal @ numpy(camera.world_view_transform)[:3,:3]
        depth = numpy(package['median_depth']).squeeze() if mode!='E6' else numpy(context['allmap'][5]).squeeze()
        mask_path = dataset_path/'masks'/f'crop_{source_id:04d}.png'
        mask = np.asarray(Image.open(mask_path).convert('L').resize((camera.image_width,camera.image_height), Image.Resampling.NEAREST))>127
        directory = views/f'crop_{source_id:04d}'; directory.mkdir()
        values = dict(render_raw=raw, render_rgb=rgb, gt_rgb=gt, foreground_mask=mask,
                      alpha=alpha, depth_median=depth.astype(np.float32), native_normal=native_normal,
                      normal_camera=normal_camera.astype(np.float32))
        if mode=='E6': values['depth_expected_baseline'] = numpy(package['surf_depth']).squeeze().astype(np.float32)
        np.savez_compressed(directory/'raw_render_outputs.npz', **values)
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(directory/'render.png')
        Image.fromarray(np.round(gt*255).astype(np.uint8)).save(directory/'gt.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(directory/'foreground_mask.png')
        Image.fromarray(np.round(np.clip(alpha,0,1)*255).astype(np.uint8)).save(directory/'alpha.png')
        row = dict(source_id=source_id, name=camera.image_name, image_size_wh=[camera.image_width,camera.image_height],
                   render_seconds=time.perf_counter()-begin, gaussian_count=int(gaussians.get_xyz.shape[0]),
                   native_normal_convention='camera' if mode!='E6' else 'world (renderer conversion from camera allmap)',
                   depth_convention='actual raster median depth; arbitrary scene units',
                   alpha_source='native renderer opacity accumulation',
                   gt_rgb_tensor_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
                   gt_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                   render_rgb_tensor_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        row['native_screen_radius_statistics']=radius_summary
        json_write(directory/'render_provenance.json', row);view_records.append(row)
        print(f'{mode} source{source_id} {camera.image_width}x{camera.image_height} actual checkpoint render saved', flush=True)
        del package, raw, rgb, gt, alpha, native_normal, normal_camera, depth, values
        context.clear()
    for name, digest in original_guard.items():
        if sha256(original_model/name)!=digest: raise RuntimeError(f'Unexpected historical modification: {name}')
    report = dict(experiment=mode, status='complete', checkpoint=str(checkpoint),
                  checkpoint_sha256=sha256(checkpoint), repo=str(repo), renderer_file=str(Path(native.__file__)),
                  renderer_sha256=sha256(native.__file__), python=sys.executable, torch=torch.__version__,
                  cuda=torch.version.cuda, physical_gpu=1, resolution_divisor=2,
                  rendered_views=len(view_records), source_ids=args.source_ids, gaussian_count=int(gaussians.get_xyz.shape[0]),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds=time.perf_counter()-started, camera_count=len(cameras),
                  native_rgb_background=[0,0,0], rgb_for_metrics='float32 clamp of actual native render, no PNG requantization or exposure',
                  historical_outputs_changed=False, original_guard=original_guard,
                  depth_ratio_baseline=0.0 if mode=='E6' else None,
                  two_dgs_readonly_allmap_observer=mode=='E6', view_records=view_records)
    report['checkpoint_kind']='final post-prune 1036619 PLY' if mode=='E3' else 'saved iteration30000 PLY'
    report['peak_memory_scope']='after Scene load, actual render and raw export; checkpoint-load transient peak not instrumented'
    report['wall_time_scope']='Scene load, historical guard hashing, native render, CPU raw/PNG export, final checkpoint hashing'
    if control is not None:
        report['point_subset_control']=control
        report['scope']='existing checkpoint isolated Gaussian deletion, no training; baseline comparison must use same source IDs'
        report['peak_memory_scope']='after Gaussian deletion subset, actual render and raw export; checkpoint-load/selection transient peaks not instrumented'
        report['wall_time_scope']='Scene load, historical guard hashing, Gaussian subset selection/parameter SHA, native render, CPU raw/PNG export, final source checkpoint hashing'
    json_write(out/'render_run.json', report)
    print(f'COMPLETE {mode}: {len(view_records)} render views, {len(cameras)} runtime cameras', flush=True)


if __name__=='__main__': main()
