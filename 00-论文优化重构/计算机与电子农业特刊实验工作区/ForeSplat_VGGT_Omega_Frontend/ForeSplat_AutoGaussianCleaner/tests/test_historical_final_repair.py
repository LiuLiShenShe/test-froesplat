"""Final repair restores actual harmed original rows and preserves all others."""
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.historical.final_repair import edited_members,restore_members,repair_view_indices


def test_restoration_handles_sparse_global_lineage_and_covariance_edits():
    rows=np.zeros(5,dtype=[('x','f4'),('scale_0','f4')]);rows['x']=np.arange(5);rows['scale_0']=1
    ids=np.array([3,10,20,25,90]);candidate=rows[[0,2,4]].copy();candidate['scale_0'][1]=.5
    assert np.array_equal(edited_members(rows,ids,candidate,ids[[0,2,4]]),[1,2,3])
    restored,new_ids=restore_members(rows,ids,candidate,ids[[0,2,4]],[1,2])
    assert np.array_equal(new_ids,[3,10,20,90])
    np.testing.assert_array_equal(restored,rows[[0,1,2,4]])
    # An unselected local edit survives when another deleted row is restored.
    partial,partial_ids=restore_members(rows,ids,candidate,ids[[0,2,4]],[1])
    assert partial['scale_0'][np.flatnonzero(partial_ids==20)[0]]==.5
    np.testing.assert_array_equal(rows['scale_0'],np.ones(5))


def test_failure_views_are_selected_from_measured_harm_not_fixed_camera_ids():
    before=[];after=[]
    for i in range(100):
        a=dict(index=i,thin_coverage=1.,skeleton_coverage=1.,fg_psnr=20.)
        b=dict(a);b['thin_coverage']-=.1 if i in [42,87] else 0.
        before.append(a);after.append(b)
    selected=repair_view_indices(dict(views=before),dict(views=after),24)
    assert len(selected)==24 and {42,87}<=set(selected)


def test_raw_guard_exemption_requires_measured_unsupported_nonworsening_loss():
    import copy,yaml
    from pathlib import Path
    from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import quality_reasons
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/historical_two_stage.yaml').read_text())
    v=dict(fg_psnr=20.,fg_alpha=1.,fg_coverage=1.,boundary_coverage=1.,thin_coverage=1.,skeleton_coverage=1.,
        dark_foreground_alpha=1.,dark_foreground_mse=.01,boundary_rgb_mse=.01,thin_rgb_mse=.01,
        depth_changed_fraction=0.,normal_angle_p95_deg=0.,new_depth_invalid_fraction=0.,
        edge_photo_worse_alpha_loss=0.,depth_supported_alpha_loss=0.)
    v.update({'supported_'+k:v[k] for k in ['fg_alpha','fg_coverage','boundary_coverage','thin_coverage','skeleton_coverage','dark_foreground_alpha','dark_foreground_mse']})
    before=dict(mean=v,views=[dict(v,index=0)]);after=copy.deepcopy(before)
    after['mean']['thin_coverage']=.99
    assert 'raw_thin_coverage' in quality_reasons(before,after,config)
    after['mean']['unsupported_nonworsening_thin_loss']=.009
    assert 'raw_thin_coverage' not in quality_reasons(before,after,config)
    after['mean']['supported_thin_coverage']=.99
    assert 'thin_coverage' in quality_reasons(before,after,config)


def test_visible_dark_leaf_and_depth_supported_edge_never_receive_tail_exemption(monkeypatch):
    from types import SimpleNamespace
    from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
    monkeypatch.setattr(Evaluator,'photometry_cached',lambda *args: ({},None))
    mask=np.ones((2,3),bool);thin=mask.copy();support=np.zeros_like(mask);support[0,0]=True
    geometry=np.zeros_like(mask);geometry[0,1]=True
    target=np.full((2,3,3),.1);before_rgb=target.copy();before_rgb[1,2]=0.
    after=before_rgb.copy();after[0,0]=0.;after[0,1]=0.;after[1,0]=0.
    alpha=np.ones((2,3));alpha[[0,0,1,1],[0,1,0,2]]=0
    ev=object.__new__(HistoricalEvaluator);ev.scene=SimpleNamespace(config=dict(historical_supported_quality=True,
        allow_unsupported_nonworsening_tails=True,foreground_support_minimum_fraction=.5),
        original_foreground_support={0:(support,geometry)})
    ev.reference={0:{'alpha':np.ones((2,3))}};ev.reference_rgb={0:before_rgb}
    photo,_=ev.photometry_cached(0,after,target,mask,dict(boundary=thin,thin=thin,skeleton=thin),alpha)
    # Only the alpha-only black tail[1,2] qualifies. The supported dark leaf,
    # depth-supported edge and an unsupported but RGB-damaged pixel do not.
    assert np.isclose(photo['unsupported_nonworsening_thin_loss'],1/6)
