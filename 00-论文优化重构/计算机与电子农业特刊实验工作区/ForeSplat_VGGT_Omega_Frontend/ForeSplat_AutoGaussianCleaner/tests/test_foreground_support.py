import unittest
from pathlib import Path
import numpy as np
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.foreground_support import support_masks,support_metrics,project_record,STRUCTURAL_FIELDS
from ForeSplat_AutoGaussianCleaner.execution_provenance import resolve_variant
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import background_selection_score


class ForegroundSupportChecks(unittest.TestCase):
    def fixture(self):
        mask=np.ones((5,5),bool);rgb=np.ones((5,5,3),np.float32)*.2
        normal=np.zeros((5,5,3),np.float32);normal[:,:,2]=1
        before=dict(rgb=rgb.copy(),alpha=np.ones((5,5),np.float32),depth=np.ones((5,5),np.float32),normal=normal)
        cfg=dict(foreground_support_mode='photo_relative_or_depth',foreground_support_relative_error=.25,
            foreground_support_minimum_fraction=.5,foreground_geometry_minimum_fraction=.1,unsupported_local_rmse_cap=1/255,
            quality={'depth_relative_p95':.03})
        r=dict(boundary=mask.copy(),thin=mask.copy(),skeleton=mask.copy())
        return before,rgb,mask,r,cfg

    def test_black_placeholder_is_reported_as_excluded_not_preserved_leaf(self):
        b,target,mask,r,cfg=self.fixture();b['rgb'][2,2]=0
        support,geo=support_masks(b,target,mask,None,None,cfg)
        self.assertFalse(support[2,2])
        after={k:v.copy() for k,v in b.items()};after['alpha'][2,2]=0
        metrics=support_metrics(after,b,b['rgb'],target,mask,r,support,geo,cfg)
        self.assertEqual(metrics['unsupported_new_alpha_holes'],1)
        self.assertEqual(metrics['supported_new_alpha_holes'],0)
        self.assertEqual(metrics['unsupported_photo_worse_pixels'],0)

    def test_small_excluded_dark_leaf_cannot_hide_in_full_image_mean(self):
        b,target,mask,r,cfg=self.fixture();target[2,2]=.03;b['rgb'][2,2]=.01
        support,geo=support_masks(b,target,mask,None,None,cfg);self.assertFalse(support[2,2])
        after={k:v.copy() for k,v in b.items()};after['rgb'][2,2]=0;after['alpha'][2,2]=0
        metrics=support_metrics(after,b,b['rgb'],target,mask,r,support,geo,cfg)
        self.assertEqual(metrics['unsupported_photo_worse_pixels'],1)
        self.assertGreater(metrics['unsupported_photo_max_rmse_increase'],1/255)

    def test_empty_supported_thin_region_falls_back_to_raw_structure(self):
        b,target,mask,r,cfg=self.fixture();b['rgb'][2,2]=0
        support,geo=support_masks(b,target,mask,None,None,cfg)
        r['thin']=np.zeros_like(mask);r['thin'][2,2]=True
        after={k:v.copy() for k,v in b.items()};after['alpha'][2,2]=0
        metrics=support_metrics(after,b,b['rgb'],target,mask,r,support,geo,cfg)
        self.assertTrue(metrics['thin_support_fallback'])
        self.assertEqual(metrics['photo_depth_thin_support_pixels'],0)
        self.assertEqual(metrics['guard_thin_support_pixels'],1)
        self.assertEqual(metrics['supported_thin_coverage'],0)

    def test_wholly_untrusted_plant_uses_conservative_full_mask_fallback(self):
        b,target,mask,r,cfg=self.fixture();b['rgb'][:]=0
        support,_=support_masks(b,target,mask,None,None,cfg)
        np.testing.assert_array_equal(support,mask)

    def test_support_projection_keeps_full_photographic_loss_and_requires_measurement(self):
        _,_,_,_,cfg=self.fixture()
        record={key:1. for key in STRUCTURAL_FIELDS};record.update({'supported_'+key:.9 for key in STRUCTURAL_FIELDS});record['fg_mse']=.2
        projected=project_record(record,cfg)
        self.assertEqual(projected['fg_mse'],.2);self.assertEqual(record['fg_alpha'],1.)
        missing=record.copy();del missing['supported_thin_coverage']
        with self.assertRaises(ValueError):project_record(missing,cfg)

    def test_real_development_configs_resolve_user_production_interface(self):
        folder=Path(__file__).resolve().parents[1]/'configs'
        for name in ['adaptive_gradient.yaml','adaptive_supported.yaml']:
            cfg=yaml.safe_load((folder/name).read_text())
            self.assertEqual(resolve_variant('C5',cfg),'C5-C')
            self.assertEqual(resolve_variant('C5-D',cfg),'C5-D')

    def test_zero_or_tiny_free_background_cannot_swamp_real_training_improvement(self):
        for initial_free in [0.,1e-18]:
            before=dict(mean={'bg_alpha_mass_per_fg':1.},free_background={'alpha_mass':initial_free})
            after=dict(mean={'bg_alpha_mass_per_fg':.5},free_background={'alpha_mass':.001})
            score,training=background_selection_score(after,before)
            self.assertEqual(training,.5)
            self.assertGreater(score,0.)
            self.assertLessEqual(score,1.)


if __name__=='__main__':unittest.main()
