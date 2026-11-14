"""Behavioral safety checks for history restoration and residual-only editing."""
import numpy as np
from scipy.special import expit
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import historical_candidates, camera_indices
from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import covariance_proposal, filtered_covariance_proposal, count_changes, quality_reasons
from ForeSplat_AutoGaussianCleaner.cleaner.historical.damage_protection import protected_members
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import measure
from ForeSplat_AutoGaussianCleaner.evaluation.free_view_evaluation import free_framing


def native_rows(n=4):
    names = ['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]
    rows = np.zeros(n, dtype=[(k,'f4') for k in names])
    rows['rot_0'] = 1
    for i, value in enumerate([.02,.03,.08]):
        rows[f'scale_{i}'] = np.log(value)
    rows['opacity'] = -2
    rows['filter_3D'] = .01
    return rows


def effective_alpha(rows):
    s = np.exp(np.column_stack([rows[f'scale_{k}'] for k in range(3)]).astype(float))
    return expit(rows['opacity'].astype(float))*(s/np.hypot(s, rows['filter_3D'][:,None])).prod(1)


def test_only_selected_covariance_changes_and_effective_opacity_is_preserved():
    rows = native_rows()
    out, changed = covariance_proposal(rows, [1,3], {}, None, .75)
    assert np.array_equal(changed, [1,3])
    assert np.array_equal(rows[[0,2]], out[[0,2]])
    for field in ['x','y','z','filter_3D','rot_0','rot_1','rot_2','rot_3','f_dc_0']:
        assert np.array_equal(rows[field], out[field])
    np.testing.assert_allclose(effective_alpha(rows), effective_alpha(out), rtol=2e-6)
    assert count_changes(rows, np.arange(4), out, np.arange(4)) == 2


def test_original_foreground_envelope_survives_reindexing_and_dark_leaves():
    # Local row0 is original7, a true foreground contributor whose current
    # observed FG is small through changed occlusion; row1 is background.
    ids = np.array([7,11,12])
    original_fg = np.zeros(13)
    original_fg[7] = 100
    e = dict(fg=np.array([0.,0.,0.]), bg=np.array([2.,200.,.01]),
             free_bg=np.zeros(3), bg_views=np.array([3,3,0]))
    remove, why = historical_candidates(e, original_fg, ids, 2)
    assert not remove[0]
    assert remove[1]
    assert not remove[2]
    assert why['foreground_envelope']


def test_history_reattribution_uses_round1_then_round2_threshold():
    fg = np.array([0.,2.,10.,0.])
    e = dict(fg=fg, bg=np.array([100.,100.,.001,.01]), free_bg=np.zeros(4), bg_views=np.array([4,4,0,0]))
    r1, _ = historical_candidates(e, fg, np.arange(4), 0)
    r2, _ = historical_candidates(e, fg, np.arange(4), 1)
    assert r1[0] and not r1[1]
    assert r2[0] and r2[1]


def test_camera_sampling_does_not_mutate_global_configuration():
    class S:
        config = {'evidence_views':24}
        def evidence_indices(self):
            return list(range(self.config['evidence_views']))
    scene = S()
    assert len(camera_indices(scene,64)) == 64
    assert scene.config['evidence_views'] == 24


def test_landscape_free_view_preserves_vertical_rays_and_exposes_outer_splats():
    view=dict(K=np.array([[800.,0,269.5],[0,800.,479.5],[0,0,1.]]),width=540,height=960)
    wide=free_framing(view,dict(evaluation_max_side=960,free_view_aspect=4/3))
    assert (wide['width'],wide['height'])==(960,720)
    assert np.isclose(wide['height']/wide['K'][1,1],view['height']/view['K'][1,1])
    # A detached splat at a lateral ray x/z=.6 lies outside the old portrait
    # image but inside the editor-matched landscape image at the same depth.
    point=np.array([.6,0,1.]);old=view['K']@point;new=wide['K']@point
    assert old[0]>view['width']
    assert 0<new[0]<wide['width']
    np.testing.assert_array_equal(view['K'],[[800,0,269.5],[0,800,479.5],[0,0,1]])


def test_changed_model_invalidates_contribution_cache(tmp_path):
    rows = native_rows()
    ids = np.arange(len(rows))
    identity = dict(model_sha256='old-model', source_sha256='implementation', config={'version':1})
    result = measure(None, None, rows, ids, [], [], tmp_path, identity)
    result['fg'][:] = 42
    np.savez(tmp_path/'contribution.npz', **result)
    reused = measure(None, None, rows, ids, [], [], tmp_path, identity)
    assert np.all(reused['fg'] == 42)
    changed = measure(None, None, rows, ids, [], [], tmp_path, {**identity,'model_sha256':'edited-model'})
    assert np.all(changed['fg'] == 0)


def test_infeasible_filtered_opacity_compensation_keeps_native_row_exact():
    rows = native_rows()
    rows['opacity'] = 10
    rows['filter_3D'] = 1
    out, changed = covariance_proposal(rows, [1], {}, None, .75)
    assert len(changed) == 0
    assert np.array_equal(out, rows)


def test_saturated_mixed_splat_can_change_actual_covariance_without_losing_center_opacity():
    rows=native_rows();rows['opacity']=10;rows['filter_3D']=.05
    out,changed=filtered_covariance_proposal(rows,[1],None,.5,'longest')
    assert np.array_equal(changed,[1])
    assert np.array_equal(rows[[0,2,3]],out[[0,2,3]])
    np.testing.assert_allclose(effective_alpha(rows),effective_alpha(out),rtol=2e-6)
    old_sigma=np.hypot(np.exp([rows[f'scale_{k}'][1] for k in range(3)]),rows['filter_3D'][1])
    new_sigma=np.exp([out[f'scale_{k}'][1] for k in range(3)])
    np.testing.assert_allclose(new_sigma,old_sigma*[1,1,.5],rtol=2e-6)
    for key in ['x','y','z','rot_0','rot_1','rot_2','rot_3','f_dc_0']:
        assert np.array_equal(rows[key],out[key])


def test_damage_protection_retains_dark_contributors_even_when_the_group_improves_elsewhere():
    # Independent background rows0/3 have no measured damage. Row1 damages
    # dark RGB, row2 loses supported depth coverage; large global RGB gains
    # must not cancel either local foreground protection.
    evidence=np.array([[0,0,100],[.1,0,1000],[0,.2,1000],[0,0,0]],float)
    protect=protected_members(evidence,[0,1,2,3],1.)
    assert np.array_equal(protect,[False,True,True,False])


def test_supported_quality_preserves_raw_thin_guard_and_photo_damage_guard():
    import copy
    import yaml
    from pathlib import Path
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/historical_two_stage.yaml').read_text())
    values=dict(fg_psnr=20.,fg_alpha=1.,fg_coverage=1.,boundary_coverage=1.,thin_coverage=1.,
        skeleton_coverage=1.,dark_foreground_alpha=1.,dark_foreground_mse=.01,
        boundary_rgb_mse=.01,thin_rgb_mse=.01,depth_changed_fraction=0.,normal_angle_p95_deg=0.,new_depth_invalid_fraction=0.,
        edge_photo_worse_alpha_loss=0.,depth_supported_alpha_loss=0.)
    values.update({'supported_'+k:values[k] for k in ['fg_alpha','fg_coverage','boundary_coverage','thin_coverage',
        'skeleton_coverage','dark_foreground_alpha','dark_foreground_mse']})
    initial=dict(mean=values,views=[dict(values,index=0)])
    tail=copy.deepcopy(initial)
    tail['mean']['boundary_coverage']=.95
    assert quality_reasons(initial,tail,config)==[]
    leaf=copy.deepcopy(tail)
    leaf['mean']['thin_coverage']=.9
    assert 'raw_thin_coverage' in quality_reasons(initial,leaf,config)
    dark=copy.deepcopy(tail)
    dark['mean']['supported_dark_foreground_mse']=.02
    assert 'dark_foreground_rgb' in quality_reasons(initial,dark,config)
    damaged=copy.deepcopy(tail)
    damaged['mean']['edge_photo_worse_alpha_loss']=.01
    assert 'edge_photo_worse_alpha_loss' in quality_reasons(initial,damaged,config)


def test_historical_stage1_tradeoff_stays_within_fg_budget_and_preserves_structure():
    import copy,yaml
    from pathlib import Path
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/historical_two_stage.yaml').read_text())
    values=dict(fg_psnr=20.,fg_alpha=1.,fg_coverage=1.,boundary_coverage=1.,thin_coverage=1.,
        skeleton_coverage=1.,dark_foreground_alpha=1.,dark_foreground_mse=.01,boundary_rgb_mse=.01,
        thin_rgb_mse=.01,depth_changed_fraction=0.,normal_angle_p95_deg=0.,new_depth_invalid_fraction=0.,
        edge_photo_worse_alpha_loss=0.,depth_supported_alpha_loss=0.)
    initial=dict(mean=values,views=[dict(values,index=0)])
    improved=copy.deepcopy(initial);improved['mean'].update(fg_psnr=20.2,dark_foreground_mse=.015)
    assert quality_reasons(initial,improved,dict(config,active_stage=1))==[]
    assert 'dark_foreground_rgb' in quality_reasons(initial,improved,dict(config,active_stage=2))
    worsening=copy.deepcopy(improved);worsening['mean']['fg_psnr']=19.95
    assert quality_reasons(initial,worsening,dict(config,active_stage=1))==[]
    severe=copy.deepcopy(worsening);severe['mean']['fg_psnr']=19.5
    assert 'foreground_psnr' in quality_reasons(initial,severe,dict(config,active_stage=1))
    missing_leaf=copy.deepcopy(improved);missing_leaf['mean']['thin_coverage']=.9
    assert 'thin_coverage' in quality_reasons(initial,missing_leaf,dict(config,active_stage=1))


def test_boundary_support_ignores_exterior_tail_but_rejects_lost_real_edge(monkeypatch):
    from types import SimpleNamespace
    from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
    monkeypatch.setattr(Evaluator,'photometry_cached',lambda *args: ({},None))
    mask=np.zeros((10,10),bool);mask[3:7,3:7]=True
    boundary=np.ones_like(mask);support=mask.copy()
    target=np.zeros((10,10,3));target[mask]=[.1,.5,.2]
    before=target.copy();before[~mask]=.1
    alpha=np.ones(mask.shape)
    ev=object.__new__(HistoricalEvaluator)
    ev.scene=SimpleNamespace(config={'historical_supported_quality':True,'foreground_support_minimum_fraction':.5},
        original_foreground_support={0:(support,support)})
    ev.reference={0:{'alpha':alpha.copy()}};ev.reference_rgb={0:before}
    r={'boundary':boundary,'thin':np.zeros_like(mask),'skeleton':mask}
    after_alpha=alpha.copy();after_alpha[~mask]=0
    after=target.copy()
    photo,_=ev.photometry_cached(0,after,target,mask,r,after_alpha)
    assert photo['supported_boundary_coverage']==1
    assert photo['qualified_boundary_pixels']==mask.sum()
    assert photo['edge_photo_worse_alpha_loss']==0
    after[3,3]=0;after_alpha[3,3]=0
    damaged,_=ev.photometry_cached(0,after,target,mask,r,after_alpha)
    assert damaged['supported_boundary_coverage']<1
    assert damaged['edge_photo_worse_alpha_loss']>.002
    assert damaged['depth_supported_alpha_loss']>.002
