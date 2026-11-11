import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.io import bake_viewer, save_ply, read_ply
from ForeSplat_AutoGaussianCleaner.cleaner.automatic_selection import select
from ForeSplat_AutoGaussianCleaner.cleaner.artifact_scoring import classify


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT/'ForeSplat_AutoGaussianCleaner/configs/default.yaml').read_text())


def synthetic_rows():
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    fields += [f'f_rest_{i}' for i in range(45)]
    fields += ['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+['filter_3D']
    rows = np.zeros(2, dtype=[(k, 'f4') for k in fields])
    rows['z'] = [2., 2.2]
    rows['rot_0'] = 1.
    rows['opacity'] = [3., 1.]
    rows['filter_3D'] = .005
    for i in range(3):
        rows[f'scale_{i}'] = np.log(.22)
    return rows


def test_native_schema_and_viewer_equivalent(tmp_path):
    from scipy.special import expit
    rows = synthetic_rows()
    save_ply(tmp_path/'native.ply', rows)
    assert np.array_equal(rows, read_ply(tmp_path/'native.ply'))
    baked, report = bake_viewer(rows)
    sigma = np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype(float))
    eff = np.hypot(sigma, rows['filter_3D'][:, None])
    assert np.allclose(np.exp(np.column_stack([baked[f'scale_{i}'] for i in range(3)])), eff, rtol=1e-6)
    assert np.allclose(expit(baked['opacity']), expit(rows['opacity'])*(sigma/eff).prod(1), rtol=1e-6)
    assert np.array_equal(rows['f_dc_0'], baked['f_dc_0'])


def test_sparse_supported_thin_structure_survives():
    # A low-density Gaussian with actual foreground/contour support is mixed, not deletable.
    e = dict(fg=np.array([.1, 0.]), bg=np.array([.01, .1]), boundary=np.array([.08, 0.]),
        thin=np.array([.07, 0.]), visible_views=np.array([4, 4]), fg_views=np.array([4, 0]),
        geometry_views=np.array([2, 0]), occlusion_views=np.array([0, 0]))
    result = classify(e, CONFIG)
    assert 0 not in result['prune']
    assert 1 in result['prune']


def test_background_dominant_gaussian_with_contour_support_is_mixed():
    e = dict(fg=np.array([.001]), bg=np.array([.4]), boundary=np.array([.001]), thin=np.array([.0005]),
        visible_views=np.array([4]), fg_views=np.array([3]), geometry_views=np.array([2]), occlusion_views=np.array([0]))
    result = classify(e, CONFIG)
    assert result['labels'][0] == 3
    assert len(result['prune']) == 0


def test_depth_supported_interior_foreground_cannot_be_whole_point_invalid():
    e = dict(fg=np.array([.001]), bg=np.array([.4]), boundary=np.array([0.]), thin=np.array([0.]),
        visible_views=np.array([4]), fg_views=np.array([3]), geometry_views=np.array([2]), occlusion_views=np.array([0]))
    result = classify(e, CONFIG)
    assert result['labels'][0] == 3
    assert len(result['prune']) == 0


def test_cumulative_transparency_is_rejected():
    mean = dict(fg_alpha=.99, fg_mse=.02, boundary_f1=.90, thin_coverage=.99,
        fg_coverage=.99, boundary_coverage=.99, skeleton_coverage=.99, depth_changed_fraction=0.,
        depth_relative_p95=0., normal_angle_p95_deg=0., new_depth_invalid_fraction=0.,
        fg_ssim=.9, fg_masked_lpips=.1, bg_alpha_mass_per_fg=.4)
    reference = dict(mean=mean, views=[])
    candidate = dict(mean={**mean, 'fg_alpha': .987, 'bg_alpha_mass_per_fg': .1}, views=[])
    accepted, gain, reasons = select(candidate, reference, reference, CONFIG)
    assert not accepted and gain > .5 and any('fg_alpha' in r for r in reasons)


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='CUDA required')
def test_occlusion_reattribution_and_dark_color_independence(tmp_path):
    import torch
    from ForeSplat_AutoGaussianCleaner.cleaner.runtime import setup, camera
    from ForeSplat_AutoGaussianCleaner.cleaner.contribution_analysis import contribution
    repo = ROOT/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/camera_interface/gggs_corrected'
    Model, renderer = setup(repo)
    rows = synthetic_rows()
    save_ply(tmp_path/'two.ply', rows)
    model = Model(3, 0); model.load_ply(str(tmp_path/'two.ply'))
    K = np.array([[100., 0, 31.5], [0, 100., 31.5], [0, 0, 1.]])
    c = camera(K, np.eye(4), 64, 64, 'test', 0)
    fg = np.zeros((64, 64), bool); fg[20:44, 20:44] = True
    weights = np.stack([fg, ~fg, fg])
    first, _, error = contribution(c, model, weights)
    # Changing SH to dark or purple must not alter color-independent alpha attribution.
    with torch.no_grad():
        model._features_dc.fill_(-1.5)
    dark, _, _ = contribution(c, model, weights)
    assert np.allclose(first, dark, rtol=1e-5)
    save_ply(tmp_path/'back.ply', rows[1:])
    back = Model(3, 0); back.load_ply(str(tmp_path/'back.ply'))
    exposed, _, _ = contribution(c, back, weights)
    assert exposed[0, 0] > first[1, 0]*1.5
    assert error.max() < .002


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='CUDA required')
def test_complete_native_pipeline_with_refinement_and_exports(tmp_path):
    import copy
    import torch
    from PIL import Image
    from ForeSplat_AutoGaussianCleaner.cleaner.runtime import setup, camera
    from ForeSplat_AutoGaussianCleaner.cleaner.iterative_cleanup import run
    repo = ROOT/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/camera_interface/gggs_corrected'
    Model, renderer = setup(repo)
    fgrows = synthetic_rows()
    rows = np.concatenate([fgrows, fgrows[:1]])
    rows['x'][2] = .55
    for i in range(3):
        rows[f'scale_{i}'][2] = np.log(.025)
    save_ply(tmp_path/'native.ply', rows)
    save_ply(tmp_path/'foreground.ply', fgrows)
    model = Model(3, 0); model.load_ply(str(tmp_path/'native.ply'))
    fgmodel = Model(3, 0); fgmodel.load_ply(str(tmp_path/'foreground.ply'))
    K = np.array([[100., 0, 31.5], [0, 100., 31.5], [0, 0, 1.]])
    c = camera(K, np.eye(4), 64, 64, 'view', 0)
    from argparse import Namespace
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    with torch.no_grad():
        rendered = renderer.render(c, model, pipe, torch.zeros(3, device='cuda'), 0.)
        fore = renderer.render(c, fgmodel, pipe, torch.zeros(3, device='cuda'), 0.)
    rgb = rendered['render'].permute(1, 2, 0).cpu().numpy()
    mask = fore['mask'].squeeze().cpu().numpy() > .5
    dataset = tmp_path/'dataset'
    for sub in ['images', 'masks', 'sparse/0']:
        (dataset/sub).mkdir(parents=True)
    for name in ['view1', 'view2']:
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype('uint8')).save(dataset/'images'/f'{name}.png')
        Image.fromarray(mask.astype('uint8')*255).save(dataset/'masks'/f'{name}.png')
    (dataset/'sparse/0/cameras.txt').write_text('1 PINHOLE 64 64 100 100 31.5 31.5\n')
    (dataset/'sparse/0/images.txt').write_text('1 1 0 0 0 0 0 0 1 view1.png\n\n2 1 0 0 0 0 0 0 1 view2.png\n\n')
    config = copy.deepcopy(CONFIG)
    config.update(evaluation_max_side=64, evidence_views=0, max_rounds=1, geometry=False,
        refinement=True, refine_steps=1, refine_validation_interval=1)
    result = run(dict(plant='synthetic', dataset=str(dataset), model=str(tmp_path/'native.ply'), gggs_repo=str(repo)),
        config, tmp_path/'output')
    assert result['cleaned_gaussians'] <= 3
    assert result['foreground_preservation'] >= .998
    assert (tmp_path/'output/COMPLETE.json').exists()
    assert json.loads((tmp_path/'output/viewer_baking.json').read_text())['parity_passed']
    assert (tmp_path/'output/removed_gaussians.ply').exists()
    assert any(item['action'] == 'refine' for item in json.loads((tmp_path/'output/edit_manifest.json').read_text())['edits'])
