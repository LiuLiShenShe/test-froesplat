from __future__ import annotations
import copy
import json
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from protocol import assert_pair,config_invariant,recording_value
from metric_core import foreground_metrics,background_geometry_metrics,resize_reference,load_roi
from recompute_metrics import assert_shared_hongzhang_alias


class ProtocolTests(unittest.TestCase):
    def test_observer_optimizer_metadata_numpy_scalars_are_serializable(self):
        import numpy as np
        original={'groups':[{'lr':np.float32(.02),'betas':(.9,.999),'eps':1e-15}], 'extent':np.float64(1.2)}
        encoded=json.dumps(recording_value(original),sort_keys=True)
        decoded=json.loads(encoded)
        self.assertAlmostEqual(decoded['groups'][0]['lr'],.02)
        self.assertIsInstance(original['groups'][0]['lr'],np.float32)

    def test_predeclared_exact_matrix_and_same_complete_parameters(self):
        plan=read_json(STAGE/'configs/EXPERIMENT_PLAN.json')
        self.assertEqual([tuple(r) for r in plan['training_order']],list(TRAIN_ORDER))
        configs=[read_json(r['path']) for r in plan['configs']]
        for r in plan['configs']:self.assertEqual(sha256(r['path']),r['sha256'])
        for cfg in configs:self.assertEqual(config_invariant(cfg),config_invariant(configs[0]))
        self.assertEqual({c['seed'] for c in configs},{1,2})
        self.assertEqual({c['optimization']['iterations'] for c in configs},{30000})
        bad=copy.deepcopy(configs[0]);bad['optimization']['lambda_normal']+=.001
        self.assertNotEqual(config_invariant(bad),config_invariant(configs[0]))

    def test_pair_gate_rejects_changed_camera_or_points_or_rng(self):
        base={k:'same' for k in ('initial_rng_sha256','ordered_camera_sha256','initial_ply_sha256',
                                'initial_parameter_sha256','post_scene_rng_sha256',
                                'pre_optimizer_rng_sha256','post_optimizer_rng_sha256')}
        assert_pair(base,base)
        for key in base:
            altered={**base,key:'changed'}
            with self.assertRaises(AssertionError):assert_pair(base,altered)

    def test_real_rng_verification_not_synthetic_claim(self):
        report=read_json(STAGE/'reports/SEED_CONTROL_VERIFICATION.json')
        self.assertEqual(report['status'],'PASS');self.assertTrue(all(report['checks'].values()))
        first=read_json(STAGE/'reports/RNG_stage1a.json')
        second=read_json(STAGE/'reports/RNG_stage2.json')
        for k in ('python','numpy','torch_cpu','torch_cuda_0'):
            self.assertNotEqual(first['initial_rng_sha256'][k],second['initial_rng_sha256'][k])
        self.assertEqual(first['initial_rng_sha256'],read_json(STAGE/'reports/RNG_stage1b.json')['initial_rng_sha256'])

    def test_no_historical_source_or_stage_changes(self):
        # Exact historical-stage protection is also rehashed by the full acceptance script.
        checked=read_json(STAGE/'reports/PRETRAINING_INPUT_RECHECK.json')
        self.assertEqual(checked['status'],'PASS');self.assertEqual(checked['failed'],[])
        self.assertEqual(checked['production_sources'],57)


class MetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import_paths()
        import torch
        cls.torch=torch

    def evaluate_fg(self,rgb,reference,alpha,roi):
        t=self.torch
        return foreground_metrics(rgb,reference,alpha,roi,
                                  lambda x,y:1-(x-y).square().mean(),
                                  lambda x,y,normalize=True:(x-y).abs().mean())

    def test_foreground_denominator_ignores_added_outside_black_pixels(self):
        t=self.torch
        roi=t.tensor([[True,False],[False,False]])
        rgb=t.zeros(3,2,2);reference=t.zeros_like(rgb);rgb[:,0,0]=.25
        result=self.evaluate_fg(rgb,reference,t.ones(2,2),roi)
        roi_large=t.zeros(20,20,dtype=t.bool);roi_large[0,0]=True
        rgb_large=t.zeros(3,20,20);rgb_large[:,0,0]=.25
        larger=self.evaluate_fg(rgb_large,t.zeros_like(rgb_large),t.ones(20,20),roi_large)
        self.assertAlmostEqual(result['foreground_MSE'],.25**2)
        self.assertEqual(result['foreground_MSE'],larger['foreground_MSE'])
        self.assertEqual(result['foreground_PSNR'],larger['foreground_PSNR'])
        self.assertEqual(result['foreground_channel_samples'],3)
        self.assertNotEqual(result['fixed_mask_composited_SSIM'],larger['fixed_mask_composited_SSIM'])

    def test_coverage_threshold_and_empty_roi_are_explicit(self):
        t=self.torch;rgb=t.zeros(3,2,2);roi=t.ones(2,2,dtype=t.bool)
        result=self.evaluate_fg(rgb,rgb,t.tensor([[0.,.5],[.5001,1.]]),roi)
        self.assertEqual(result['foreground_coverage_proxy'],.5)
        self.assertEqual(result['missing_alpha_fraction'],.5)
        self.assertFalse(result['ROI_is_independent_GT'])
        with self.assertRaises(ValueError):self.evaluate_fg(rgb,rgb,t.ones(2,2),~roi)

    def test_geometry_support_excludes_invalid_depth_and_normals(self):
        t=self.torch;roi=t.ones(2,2,dtype=t.bool);rgb=t.zeros(3,2,2);alpha=t.ones(2,2)
        depth=t.tensor([[1.,2.],[float('nan'),0.]])
        normal=t.zeros(3,2,2);normal[2]=1
        result=background_geometry_metrics(rgb,alpha,roi,depth,normal,normal)
        self.assertEqual(result['depth_valid_ROI_pixels'],2)
        self.assertEqual(result['depth_p50'],1.5)
        self.assertAlmostEqual(result['normal_consistency_cosine_proxy'],1.)
        self.assertEqual(result['geometry_GT'],'NONE')
        self.assertFalse(result['components_are_confirmed_floaters'])

    def test_real_frozen_roi_is_binary_and_reference_resize_matches_E1(self):
        import torch
        rows=read_csv(E1/'HELDOUT_PER_VIEW_METRICS.csv')
        from protocol import tensor_sha
        for sample in SCENES:
            row=next(r for r in rows if r['sample']==sample and r['group']=='B_R0')
            proxy=next(r for r in read_csv(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['sample']==sample and r['group']=='B_R0' and Path(r['registered_name']).stem==row['loader_image_name'])
            width,height=int(row['render_width']),int(row['render_height'])
            self.assertEqual(tensor_sha(resize_reference(row['reference_path'],width,height)),row['reference_resized_tensor_sha256'])
            mask=load_roi(proxy['mask_path'],width,height)
            self.assertEqual(mask.dtype,torch.bool);self.assertGreater(mask.sum().item(),0)
            self.assertEqual(mask.sum().item(),int(row['R0_proxy_area_px']))

    def test_shared_hongzhang_alias_requires_exact_pair_and_equal_metrics(self):
        common={'sample':'HongZhang','seed':'0','loader_image_name':'0244',
                'float_render_path':'/immutable/HongZhang/0244.pt','float_render_sha256':'render-sha',
                'model_checkpoint_sha256':'checkpoint-sha','reference_resized_tensor_sha256':'ref-sha',
                'ROI_resized_tensor_sha256':'roi-sha','PSNR':'5.1','SSIM':'0.7','LPIPS':'0.2'}
        left={**common,'group':'B_R0','shared_model_alias':'HongZhang/C_R1/seed0'}
        right={**common,'group':'C_R1','shared_model_alias':''}
        fg={('HongZhang',group,'0','0244'):{**common,'group':group,
             'foreground_PSNR':'8.2','shared_model_alias':alias}
            for group,alias in [('B_R0','HongZhang/C_R1/seed0'),('C_R1','')]}
        bg={('HongZhang',group,'0','0244'):{**common,'group':group,
             'outside_ROI_RGB_energy':'0.01','shared_model_alias':alias}
            for group,alias in [('B_R0','HongZhang/C_R1/seed0'),('C_R1','')]}
        assert_shared_hongzhang_alias(left,right,fg,bg)
        changed={**right,'PSNR':'5.2'}
        with self.assertRaises(AssertionError):
            assert_shared_hongzhang_alias(left,changed,fg,bg)
        wrong_alias={**left,'shared_model_alias':'unexpected'}
        with self.assertRaises(AssertionError):
            assert_shared_hongzhang_alias(wrong_alias,right,fg,bg)

if __name__=='__main__':unittest.main()
