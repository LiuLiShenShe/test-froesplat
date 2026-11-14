import json
import tempfile
import unittest
from pathlib import Path
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.execution_provenance import completed_identity,scene_input_fingerprint,ARTIFACT_FILES,invalidate_output,assert_output_separate,resolve_variant
from ForeSplat_AutoGaussianCleaner.auto_clean_batch import summarize
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import require_final_quality,background_selection_score


class ProvenanceChecks(unittest.TestCase):
    def test_output_cannot_archive_or_overwrite_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spec={'model':str(p/'checkpoint/cleaned_gaussians.ply')}
            with self.assertRaises(ValueError):assert_output_separate(p/'checkpoint',spec)
            with self.assertRaises(ValueError):assert_output_separate(p,spec)
            assert_output_separate(p/'comparison/result',spec)

    def test_production_alias_uses_selected_grouped_variant(self):
        self.assertEqual(resolve_variant('C5',{'production_variant':'C5-C'}),'C5-C')
        with self.assertRaises(ValueError):resolve_variant('C5',{'production_variant':'C5-A'})
        self.assertEqual(resolve_variant('C5-A',{}),'C5-A')
    def test_resume_cannot_reuse_another_ablation_or_corrupted_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'original.ply';source.write_bytes(b'original')
            native=p/'cleaned_gaussians.ply';native.write_bytes(b'cleaned')
            for name in ARTIFACT_FILES:
                if not (p/name).exists():(p/name).write_bytes(b'artifact')
            config={'method_version':'same'};spec={'model':str(source)}
            write_json(p/'COMPLETE.json',dict(config=config,scene=spec,variant='C5-C',input_sha256=sha(source),native_sha256=sha(native),artifact_hashes={name:sha(p/name) for name in ARTIFACT_FILES}))
            self.assertTrue(completed_identity(p,config,spec,'C5-C'))
            self.assertFalse(completed_identity(p,config,spec,'C5-D'))
            (p/'cleaned_gaussians_supersplat.ply').write_bytes(b'corrupted viewer')
            self.assertFalse(completed_identity(p,config,spec,'C5-C'))
            native.write_bytes(b'corrupted')
            self.assertFalse(completed_identity(p,config,spec,'C5-C'))

    def test_explicit_invalidation_removes_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/'COMPLETE.json',{'valid_before_verification':True})
            invalidate_output(p,'Missing depth file during post-run verification')
            self.assertFalse((p/'COMPLETE.json').exists())
            self.assertTrue((p/'INVALIDATED_COMPLETE.json').exists())
            self.assertFalse(json.loads((p/'INVALIDATED.json').read_text())['certified'])

    def test_missing_completed_report_is_excluded_from_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'original.ply';source.write_bytes(b'original')
            scene=p/'scene.json';write_json(scene,dict(model=str(source)))
            write_json(p/'development/plant/COMPLETE.json',{})
            summarize({'plants':[dict(plant='plant',group='development',scene=str(scene))]},p,{})
            self.assertEqual(json.loads((p/'GENERALIZATION_STATUS.json').read_text())['completed'],0)
            self.assertIn('invalid_completed_output',(p/'CROSS_PLANT_GGGS_CLEANING.csv').read_text())

    def test_in_place_mask_update_changes_scene_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ['images','masks','sparse/0']:(p/name).mkdir(parents=True)
            (p/'images/view.png').write_bytes(b'rgb')
            mask=p/'masks/view.png';mask.write_bytes(b'mask-a')
            (p/'sparse/0/cameras.txt').write_bytes(b'camera')
            (p/'sparse/0/images.txt').write_bytes(b'poses')
            spec={'dataset':str(p)};before=scene_input_fingerprint(spec)
            mask.write_bytes(b'mask-b')
            self.assertNotEqual(before,scene_input_fingerprint(spec))

    def test_failed_child_cannot_promote_an_old_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'original.ply';source.write_bytes(b'original')
            scene=p/'scene.json';write_json(scene,dict(model=str(source)))
            dest=p/'development/plant';write_json(dest/'COMPLETE.json',{})
            manifest={'plants':[dict(plant='plant',group='development',scene=str(scene))]}
            summarize(manifest,p,{'plant':{'status':'failed'}})
            status=json.loads((p/'GENERALIZATION_STATUS.json').read_text())
            self.assertEqual(status['completed'],0)
            self.assertEqual(status['modified'],0)
            self.assertIn('failed',(p/'CROSS_PLANT_GGGS_CLEANING.csv').read_text())

    def test_worsened_edited_final_model_cannot_be_published(self):
        initial=dict(mean={'bg_alpha_mass_per_fg':1.},free_background={'alpha_mass':1.})
        worse=dict(mean={'bg_alpha_mass_per_fg':1.01},free_background={'alpha_mass':1.01})
        self.assertLess(background_selection_score(worse,initial)[0],0)
        with self.assertRaises(ValueError):
            require_final_quality(False,['insufficient measured group benefit'],True)
        require_final_quality(False,['insufficient measured group benefit'],False)
        with self.assertRaises(ValueError):
            require_final_quality(False,['thin_coverage'],False)


if __name__=='__main__':unittest.main()
