import unittest
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features,rules
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.grouping import groups
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import contour_metrics,choose,potential_dark_tradeoff
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.depth_recovery import potential_tradeoff


class AdaptiveChecks(unittest.TestCase):
    def rows(self,n=32):
        names=['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'f_dc_{i}' for i in range(3)]
        r=np.zeros(n,dtype=[(k,'f4') for k in names]);r['x']=np.linspace(0,1,n);r['opacity']=2
        for i in range(3):r[f'scale_{i}']=-5
        return r

    def test_third_axis_used_in_historical_v3(self):
        r=self.rows();r['scale_2'][0]=0
        mask,_,_=rules(features(r),3,False)
        self.assertTrue(mask[0])

    def test_normalized_rules_invariant_to_world_rescaling(self):
        r=self.rows();r['filter_3D']=.004
        scaled=r.copy()
        for k in ['x','y','z','filter_3D']:scaled[k]*=100
        for i in range(3):scaled[f'scale_{i}']+=np.log(100)
        a,_,_=rules(features(r),3,True);b,_,_=rules(features(scaled),3,True)
        np.testing.assert_array_equal(a,b)

    def test_effective_alpha_keeps_determinant(self):
        r=self.rows();r['filter_3D']=.02
        f=features(r,knn=False)
        self.assertTrue(np.all(f['effective_alpha']<f['raw_alpha']))
        np.testing.assert_allclose(f['effective']**2,f['sigma']**2+.02**2,rtol=1e-6)

    def test_distant_cloud_does_not_change_plant_local_contour(self):
        m=np.zeros((128,128),bool);m[40:90,40:90]=True
        a=m.astype(float);b=a.copy();b[5:20,5:20]=1
        x=contour_metrics(a,a,m,2);y=contour_metrics(b,a,m,2)
        self.assertEqual(x['fg_local_boundary_f1'],y['fg_local_boundary_f1'])
        self.assertEqual(y['fg_alpha_loss_pixels'],0)

    def test_groups_separate_risk_and_disconnected_regions(self):
        f=dict(xyz=np.array([[0,0,0],[.01,0,0],[1,0,0],[1.01,0,0]]),extent=1,spacing=np.ones(4)*.001)
        e=dict(fg=np.array([0,0,.1,.1]),bg=np.ones(4))
        g=groups(np.arange(4),f,e,dict(group_relative_cell=.05,group_max_gaussians=10))
        self.assertEqual(sorted(len(x) for x in g),[2,2])

    def test_incomplete_metrics_cannot_certify_even_large_background_gain(self):
        current=dict(mean={'bg_alpha_mass_per_fg':1.},views=[{'bg_alpha_mass_per_fg':1.}])
        candidate=dict(mean={'bg_alpha_mass_per_fg':.5},views=[{'bg_alpha_mass_per_fg':.5}],
            metrics_complete=False,preflight_rejection_reasons=['view_x:dark_leaf_mse'])
        accepted,gain,_,reasons=choose(candidate,current,current,
            dict(group_minimum_global_gain=1e-5,group_minimum_view_gain=.0002))
        self.assertFalse(accepted)
        self.assertEqual(gain,.5)
        self.assertEqual(reasons,['view_x:dark_leaf_mse'])

    def test_dark_tradeoff_never_defers_depth_or_larger_color_damage(self):
        before=[dict(name='view_x',dark_foreground_mse=.003)]
        small=[dict(name='view_x',dark_foreground_mse=(np.sqrt(.003)+.002)**2)]
        large=[dict(name='view_x',dark_foreground_mse=(np.sqrt(.003)+.01)**2)]
        cfg=dict(structure_constrained_dark_tradeoff=True,dark_rmse_increase_cap=1/255)
        self.assertTrue(potential_dark_tradeoff(['view_x:dark_leaf_mse'],small,before,before,cfg))
        self.assertFalse(potential_dark_tradeoff(['view_x:depth_changed_fraction,dark_leaf_mse'],small,before,before,cfg))
        self.assertFalse(potential_dark_tradeoff(['view_x:dark_leaf_mse'],large,before,before,cfg))

    def test_depth_recovery_requires_support_and_keeps_normal_thin_hole_guards(self):
        cfg=dict(certify_depth_recovery=True,depth_recovery_metric_tolerance=1e-6,
            depth_recovery_minimum_support=.8,depth_recovery_minimum_toward=.95,
            depth_recovery_maximum_harmful_fraction=.001,depth_recovery_maximum_edge_harmful_fraction=.0005)
        old=dict(name='v',external_depth_mae=.05,external_depth_p95=.2)
        good=dict(name='v',external_depth_mae=.04,external_depth_p95=.15,recovery_changed_pixels=100,
            recovery_supported_fraction=.9,recovery_toward_fraction=.99,recovery_harmful_fraction=0.,
            recovery_edge_harmful_fraction=0.,recovery_changed_mae_before=.3,recovery_changed_mae_after=.1)
        self.assertTrue(potential_tradeoff(['v:depth_changed_fraction'],[good],[old],[old],cfg))
        for harm in ['normal_angle_p95_deg','thin_coverage','depth_holes','dark_leaf_alpha']:
            self.assertFalse(potential_tradeoff(['v:depth_changed_fraction,'+harm],[good],[old],[old],cfg))
        unsupported={**good,'recovery_supported_fraction':.3}
        self.assertFalse(potential_tradeoff(['v:depth_changed_fraction'],[unsupported],[old],[old],cfg))
        worse={**good,'external_depth_mae':.06}
        self.assertFalse(potential_tradeoff(['v:depth_changed_fraction'],[worse],[old],[old],cfg))


if __name__=='__main__':unittest.main()
