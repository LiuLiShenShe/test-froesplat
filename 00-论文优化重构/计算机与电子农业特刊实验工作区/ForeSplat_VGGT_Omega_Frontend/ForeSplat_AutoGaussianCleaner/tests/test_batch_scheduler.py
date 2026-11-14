"""Concurrent scenes must execute once, respect the bound, and propagate failure."""
import importlib.util
import threading
import time
import unittest
import json
import tempfile
from pathlib import Path
from ForeSplat_AutoGaussianCleaner import auto_clean_batch as production


def scheduler_module():
    if hasattr(production,'execute_jobs'):return production
    path=Path(__file__).resolve().parents[1]/'experiments/GGGS_Cleaner_Comparison/development_variants/v11/ForeSplat_AutoGaussianCleaner/auto_clean_batch.py'
    spec=importlib.util.spec_from_file_location('c11_batch_scheduler',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def scheduler():return scheduler_module().execute_jobs


class BatchSchedulerChecks(unittest.TestCase):
    def test_each_scene_executes_once_with_at_most_two_active_workers(self):
        lock=threading.Lock();ready=threading.Event()
        state={'active':0,'peak':0,'seen':[]}
        def worker(scene):
            with lock:
                state['active']+=1;state['peak']=max(state['peak'],state['active'])
                if state['active']==2:ready.set()
            if not ready.wait(2):raise RuntimeError('Two requested workers never overlapped')
            time.sleep(.005)
            with lock:state['seen'].append(scene);state['active']-=1
        scheduler()(list(range(8)),worker,2)
        self.assertEqual(sorted(state['seen']),list(range(8)))
        self.assertEqual(state['peak'],2)
        self.assertEqual(state['active'],0)

    def test_default_execution_preserves_single_scene_order(self):
        seen=[];scheduler()(list(range(5)),seen.append)
        self.assertEqual(seen,list(range(5)))

    def test_scene_failure_is_visible_to_batch_caller(self):
        def worker(scene):
            if scene==2:raise ValueError('Actual child failure')
        with self.assertRaisesRegex(ValueError,'Actual child failure'):
            scheduler()(list(range(5)),worker,2)

    def report_fixture(self,folder):
        root=Path(folder);source=root/'original.ply';source.write_bytes(b'original')
        scene=root/'scene.json';scene.write_text(json.dumps({'model':str(source)}))
        manifest={'plants':[{'plant':'plant','group':'development','scene':str(scene)}]}
        output=root/'development/plant';output.mkdir(parents=True)
        return root,output,manifest

    def test_pending_old_completion_is_not_counted_before_child_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root,output,manifest=self.report_fixture(folder)
            (output/'COMPLETE.json').write_text('{}')
            status={};scheduler_module().summarize(manifest,root,status)
            self.assertEqual(json.loads((root/'GENERALIZATION_STATUS.json').read_text())['completed'],0)
            self.assertNotIn('plant',status)

    def test_bad_successful_child_certificate_updates_authoritative_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root,output,manifest=self.report_fixture(folder)
            (output/'COMPLETE.json').write_text('{}')
            status={'plant':{'status':'complete'}}
            scheduler_module().summarize(manifest,root,status)
            self.assertEqual(status['plant']['status'],'invalid_completed_output')
            self.assertEqual(json.loads((root/'GENERALIZATION_STATUS.json').read_text())['completed'],0)

    def test_disappearing_completion_after_success_cannot_exit_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_,manifest=self.report_fixture(folder)
            status={'plant':{'status':'complete'}}
            scheduler_module().summarize(manifest,root,status)
            self.assertEqual(status['plant']['status'],'invalid_completed_output')


if __name__=='__main__':unittest.main()
