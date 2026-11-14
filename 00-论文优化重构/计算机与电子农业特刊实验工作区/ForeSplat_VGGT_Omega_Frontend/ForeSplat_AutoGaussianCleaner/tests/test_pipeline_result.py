"""Pipeline consumers must reject partial exports and truthful-zero violations."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from ForeSplat_AutoGaussianCleaner.integration import read_cleaning_result


class PipelineResultChecks(unittest.TestCase):
    def setUp(self):
        self.camera_inventory=patch('ForeSplat_AutoGaussianCleaner.integration.scene_camera_names',return_value=['view1','view2'])
        self.camera_inventory.start();self.addCleanup(self.camera_inventory.stop)

    def fixture(self,folder,modified=False):
        path=Path(folder);source=path/'original.ply';source.write_bytes(b'untouched original')
        scene=dict(plant='fixture',model=str(source));scene_path=path/'scene.json'
        scene_path.write_text(json.dumps(scene));output=path/'result';output.mkdir()
        names=['cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply','removed_gaussians.ply',
               'removed_gaussians_supersplat.ply','original_gaussian_ids.npy','removed_original_gaussian_ids.npy','viewer_baking.json']
        for name in names:(output/name).write_bytes(b'complete artifact '+name.encode())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        marker=dict(scene=scene,variant='C5-C',config={'fixed':'configuration'},input_sha256=digest(source),
            native_sha256=digest(output/'cleaned_gaussians.ply'),artifact_hashes={name:digest(output/name) for name in names})
        report=dict(method='C5-C',config=marker['config'],input_sha256=marker['input_sha256'],
            removed_gaussians=int(modified),refined_gaussians=0,artifact_reduction=.1 if modified else 0.,
            outcome='measured_improvement_with_residuals' if modified else 'no_accepted_modification',seconds=1.,
            initial={'views':[{'name':'view1'},{'name':'view2'}]},
            final={'views':[{'name':'view1'},{'name':'view2'}],'metrics_complete':True})
        self.write(output,marker,report)
        return scene_path,output,marker,report

    def write(self,output,marker,report):
        (output/'COMPLETE.json').write_text(json.dumps(marker))
        (output/'cleanup_metrics.json').write_text(json.dumps(report))

    def test_zero_modification_remains_an_explicit_zero_pipeline_outcome(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,_,_=self.fixture(folder)
            result=read_cleaning_result(scene,output)
            self.assertFalse(result['modified'])
            self.assertEqual(result['status'],'no_accepted_modification')

    def test_real_edit_with_measured_gain_is_passed_to_downstream_native_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,_,_=self.fixture(folder,True)
            result=read_cleaning_result(scene,output)
            self.assertTrue(result['modified'])
            self.assertEqual(result['downstream_gaussian_model'],str(output/'cleaned_gaussians.ply'))

    def test_claimed_edit_cannot_export_identical_original_native_model(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder,True)
            original=Path(json.loads(scene.read_text())['model'])
            (output/'cleaned_gaussians.ply').write_bytes(original.read_bytes())
            marker['native_sha256']=marker['input_sha256']
            marker['artifact_hashes']['cleaned_gaussians.ply']=marker['input_sha256']
            self.write(output,marker,report)
            with self.assertRaisesRegex(ValueError,'identical original native model'):
                read_cleaning_result(scene,output)

    def test_partial_matched_view_metrics_cannot_enter_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder,True)
            report['final']['views']=report['final']['views'][:1]
            self.write(output,marker,report)
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)

    def test_matching_truncated_lists_cannot_claim_full_scene_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder,True)
            for key in ['initial','final']:report[key]['views']=report[key]['views'][:1]
            self.write(output,marker,report)
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)

    def test_initial_incomplete_flag_cannot_enter_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder,True)
            report['initial']['metrics_complete']=False
            self.write(output,marker,report)
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)

    def test_missing_removed_id_export_cannot_enter_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,_,_=self.fixture(folder)
            (output/'removed_original_gaussian_ids.npy').unlink()
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)

    def test_zero_model_cannot_be_labeled_measured_cleanup_success(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder)
            report['outcome']='measured_improvement_with_residuals'
            self.write(output,marker,report)
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)

    def test_wrong_execution_report_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,output,marker,report=self.fixture(folder,True)
            report['config']={'different':'configuration'}
            self.write(output,marker,report)
            with self.assertRaises(ValueError):read_cleaning_result(scene,output)


if __name__=='__main__':unittest.main()
