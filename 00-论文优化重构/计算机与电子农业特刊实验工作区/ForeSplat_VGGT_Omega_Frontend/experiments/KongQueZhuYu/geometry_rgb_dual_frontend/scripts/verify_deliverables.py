"""Verify actual completed artifacts and record their identities without inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = [i for i in range(210) if i not in (44, 206)]
NEW = {'E7':'E7_Omega_Original_Interface', 'E8':'E8_Omega_Corrected_Interface',
       'E9':'E9_Standard_Original_Interface', 'E10':'E10_Standard_Corrected_Interface'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def main():
    rows = list(csv.DictReader((ROOT/'shared_manifest.csv').open()))
    assert [int(r['source_index']) for r in rows] == IDS
    inv = json.loads((ROOT/'configs/input_invariants.json').read_text())
    assert inv['source_ids'] == IDS and all(r['foreground_changed_pixels'] == 0 for r in inv['frames'])
    files = []
    for r in rows:
        for key in ['raw_rgb_path', 'geometry_rgb_path', 'training_rgb_path', 'mask_path']: assert Path(r[key]).is_file()
        assert sha(Path(r['training_rgb_path'])) == r['rgb_sha256_historical']
        assert sha(Path(r['mask_path'])) == r['mask_sha256_historical']
    for folder, shape in [('VGGT_Omega_Original_RGB', (688,384)), ('Standard_VGGT_Original_RGB', (518,518))]:
        d = ROOT/folder
        cameras = json.loads((d/'cameras.json').read_text())['frames']
        assert len(cameras) == 208
        assert json.loads((d/'metadata.json').read_text())['all_208_new_geometry_inference_executed']
        for sid in IDS:
            stem = f'crop_{sid:04d}'
            for key, target in [('depth',shape),('confidence',shape),('world_points',(*shape,3)),('valid_foreground',shape),('model_mask',shape)]:
                a = np.load(d/'per_frame'/f'{key}_{stem}.npy', mmap_mode='r'); assert a.shape == target
            a = np.load(d/'per_frame'/f'foreground_world_points_{stem}.npy', mmap_mode='r'); assert a.ndim == 2 and a.shape[1] == 3
        for name in ['foreground.ply','unfiltered.ply']: assert (d/name).stat().st_size > 1024
        old = d/'COLMAP_Original'; new = d/'COLMAP_Corrected'
        assert (old/'sparse/0/points3D.ply').resolve() == (new/'sparse/0/points3D.ply').resolve()
        for sid in IDS:
            for sub in ['images','masks']:
                a = old/sub/f'crop_{sid:04d}.png'; b = new/sub/f'crop_{sid:04d}.png'
                assert a.resolve() == b.resolve()
    parameters = {}
    for eid, folder in NEW.items():
        d = ROOT/folder
        p = json.loads((d/'actual_training_parameters.json').read_text()); parameters[eid] = p
        done = json.loads((d/'densification_events.complete.json').read_text())
        assert done['status'] == 'complete' and done['physical_gpu'] == 1
        assert p['opt']['iterations'] == 30000 and p['seed'] == 0 and p['CUDA_VISIBLE_DEVICES'] == '1'
        assert not p['eo_enabled']
        assert not p['fg']['use_foreground_rgb_loss'] and not p['fg']['use_mask_loss'] and not p['fg']['use_bg_opacity_loss']
        assert p['fg']['pruning_mode'] == 'none' and p['dataset']['use_decoupled_appearance'] == 0
        events = [json.loads(v) for v in (d/'densification_events.jsonl').read_text().splitlines()]
        assert len(events) == 144 and events[-1]['iteration'] == 14900
        metrics = list(csv.DictReader((d/'train_metrics.csv').open()))
        assert int(metrics[-1]['iteration']) == 30000
        for name in ['point_cloud/iteration_30000/point_cloud.ply','chkpnt30000.pth','final_gaussians/gaussians.npz']:
            f = d/'checkpoints'/name; assert f.stat().st_size > 1024
            files.append(dict(experiment=eid,path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
    for p in parameters.values():
        assert p['opt'] == parameters['E8']['opt'] and p['pipe'] == parameters['E8']['pipe']
    per = {}; evaluator_hashes=set(); audit_metric_hashes=set()
    for eid in ['E1','E5','E7','E8','E9','E10']:
        d = ROOT/'render_comparison'/eid
        report = json.loads((d/'render_complete.json').read_text()); assert report['summary']['views'] == 208
        evaluator_hashes.add(report['evaluator_script_sha256']); audit_metric_hashes.add(report['audit_metric_source_sha256'])
        table = list(csv.DictReader((d/'per_view_metrics.csv').open()))
        assert sorted(int(r['source_id']) for r in table) == IDS
        per[eid] = {int(r['source_id']): r for r in table}
    for sid in IDS:
        for key in ['gt_sha256','mask_sha256']: assert len({p[sid][key] for p in per.values()}) == 1
    assert len(evaluator_hashes) == 1 and len(audit_metric_hashes) == 1
    visual = ROOT/'render_comparison/visualizations'
    for index in range(0,208,26):
        sid = IDS[index]
        for typ in ['whole_plant','leaf_native','leaf_edge_native320','foreground_error','normal','alpha','depth_per_extent']:
            assert (visual/f'source_{sid:04d}_{typ}.png').is_file()
    for name in ['frontend_comparison.csv','reconstruction_comparison.csv','gaussian_geometry_comparison.csv','FINAL_ANALYSIS.md']:
        assert (ROOT/name).stat().st_size > 64
    review = json.loads((ROOT/'render_comparison/compact_review/visual_review.json').read_text())
    assert review['status'] == 'complete' and review['direct_model_vision']
    assert not review['external_vision_api_used'] and len(review['directly_viewed_files']) == 6
    for item in review['directly_viewed_files']:
        assert sha(ROOT/item['path']) == item['sha256']
    decisions = json.loads((ROOT/'configs/final_decisions.json').read_text())
    assert decisions['status'] == 'complete'
    assert sorted(int(k) for k in decisions['answers']) == list(range(1,10))
    report = (ROOT/'FINAL_ANALYSIS.md').read_text()
    assert '等待直接查看' not in report and '仍需结合直接查看' not in report
    for number in range(1,10): assert f'\n{number}. **' in report
    for name in ['six_model_view_distributions.pdf','geometry_and_interface_paired_gains.pdf','statistical_protocol.json']:
        assert (ROOT/'render_comparison/figures'/name).is_file()
    result = dict(status='passed',verified_unix=time.time(),physical_gpu=1,dependency_manager='uv',
        frame_ids=IDS,actual_four_30000_checkpoints=files,unchanged_training_rgb_and_masks=True,
        identical_pair_initialization=True,six_times_208_native_evaluation=True,exact_gt_and_mask_hash_parity=True,
        eight_E6_visuals=True,direct_compact_visual_review_complete=True,nine_final_decisions_complete=True,
        scope='Artifact completion/provenance validation, saved direct visual review and final nine decisions; judgments in FINAL_ANALYSIS.md')
    (ROOT/'configs/deliverable_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('verified: four actual 30K runs, 208 shared inputs, six complete common208 evaluations and eight-view evidence')

if __name__ == '__main__': main()
