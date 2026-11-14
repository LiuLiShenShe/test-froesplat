#!/usr/bin/env python3
"""Serial GGGS-only development comparisons, then frozen all-plant execution."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
sys.path.insert(0, str(PROJECT))
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json, sha
from viewer_review_validity import validated_review
from historical_batch_scheduler import execute_pipeline

DEVELOPMENT = ['KongQueZhuYu','ChangShouHua1','XiangPiShu1','WangWenCao1']


def augmented_scene(path, output):
    scene = json.loads(Path(path).read_text())
    baseline = scene.get('baseline_models', {}).get('R5')
    if baseline:
        task = Path(baseline).parent.parent
        protocol = task.parent.parent/'comparison/shared_free_views.json'
        silhouettes = task/'diagnosis/free_contribution_views'
        if protocol.exists() and silhouettes.exists():
            scene.update(historical_free_protocol=str(protocol), historical_free_silhouettes=str(silhouettes))
    destination = output/'scene_configs'/f'{scene["plant"]}.json'
    write_json(destination, scene)
    return destination


def summarize(root, manifest):
    models, rows, statuses = {}, [], []
    for item in manifest:
        plant, out = item['plant'], Path(item['output'])
        marker, metrics = out/'COMPLETE.json', out/'cleanup_metrics.json'
        if not marker.exists() or not metrics.exists():
            status = ('failed' if (out/'FAILED.json').exists() else
                'incomplete' if (out/'execution.json').exists() else item.get('status','pending'))
            statuses.append(dict(plant=plant, status=status, output=str(out)))
            continue
        result = json.loads(metrics.read_text())
        completed = json.loads(marker.read_text())
        if not completed.get('source_and_input_verification'):
            statuses.append(dict(plant=plant, status='unverified_completion', output=str(out)))
            continue
        models[plant] = result['models']
        statuses.append(dict(plant=plant, status='complete', output=str(out)))
        s1, s2 = result['stage1'], result['stage2']
        comparisons = [('B0_original',s1['initial'],s1['input_gaussians'],0),
                       ('B2_stage1',s1['final'],s1['final_gaussians'],s1['seconds']),
                       ('B1_historical' if result['models']['historical'] else 'stage2_input',s2['initial'],s2['input_gaussians'],0),
                       ('B3_stage2' if result['models']['historical'] else 'stage1_plus_stage2',s2['final'],s2['final_gaussians'],s2['seconds'])]
        for label, values, n, seconds in comparisons:
            row = dict(plant=plant, model=label, gaussians=n, seconds=seconds, observed_views=len(values['views']),
                resolution=values['resolution_max_side'], **values['mean'])
            row.update({'automatic_free_'+k:v for k,v in values['free_background'].items() if isinstance(v,(int,float))})
            rows.append(row)
    write_json(root/'model_index/models.json', models)
    write_json(root/'batch_status.json', statuses)
    if rows:
        path = root/'metrics/comparison.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    report = ['# ForeSplat Two-Stage Gaussian Artifact Suppression', '',
        f'Completed plants: {len(models)}/{len(manifest)}. All figures below must be read from actual completed outputs.', '',
        '| Plant | Status | Output |', '|---|---|---|',
        *[f'| {s["plant"]} | {s["status"]} | {s["output"]} |' for s in statuses], '',
        'Automatic free-view scores use independent masked depth projections. Historical eight-view scores are in separate visualizations/historical_free_metrics.json files.',
        'Interactive SuperSplat review and any unmeasured speedup/generalization claim remain pending.', '',
        'Sources: local GGGS rendering, per-round contribution and edit evidence. Bocha search unavailable (HTTP403 quota).']
    (root/'BATCH_PROGRESS.md').write_text('\n'.join(report)+'\n')
    return models


def development_gate(manifest):
    results = {}
    for item in manifest:
        if item['plant'] not in DEVELOPMENT:
            continue
        out = Path(item['output'])
        if not (out/'COMPLETE.json').exists():
            return False, f'{item["plant"]} has no completed experiment'
        marker=json.loads((out/'COMPLETE.json').read_text())
        if not marker.get('source_and_input_verification'):
            return False, f'{item["plant"]} has no verified source/input completion'
        results[item['plant']] = json.loads((out/'cleanup_metrics.json').read_text())
    k = results.get('KongQueZhuYu')
    if k is None:
        return False, 'Historical four-way development comparison missing'
    if k['stage2']['score'] <= 0 or k['stage2']['rolled_back']:
        return False, 'B3 has not demonstrated a foreground-preserving improvement over B1'
    # Stage1 should recover most historical suppression, not simply match PSNR.
    b2 = k['stage1']['final']['free_background']
    b1 = k['stage2']['initial']['free_background']
    if b2['black_pixels'] > b1['black_pixels']*1.10+10:
        return False, 'Generalized Stage1 has not reproduced historical black-artifact suppression'
    others = [r for p,r in results.items() if p != 'KongQueZhuYu']
    if len(others) < 2 or sum(r['stage1']['score'] > 0 for r in others) < 2:
        return False, 'Two additional morphologies have not demonstrated Stage1 improvement'
    reviews = {}
    for item in manifest:
        if item['plant'] not in DEVELOPMENT:continue
        review=validated_review(item['output'])
        if not review:
            return False, f'{item["plant"]} actual SuperSplat review is not recorded'
        reviews[item['plant']] = review
    # The requested development comparison is Kong plus at least two other
    # morphologies. A fourth, failed case must stay visible in the frozen
    # evaluation; running heldout measurements does not accept it visually.
    if not reviews.get('KongQueZhuYu',{}).get('eligible_for_batch',False):
        return False, 'KongQueZhuYu actual SuperSplat improvement is not accepted'
    if sum(p!='KongQueZhuYu' and r.get('eligible_for_batch',False)
           for p,r in reviews.items()) < 2:
        return False, 'Two additional morphologies lack actual SuperSplat improvement'
    failures = [p for p,r in reviews.items() if not r.get('eligible_for_batch',False)]
    return True, ('Frozen heldout evaluation permitted after Kong and two additional '
        f'morphologies improve; known visual failures remain failures: {failures}')


def freeze_heldout_protocol(root, config, manifest):
    """Resume heldout work only under the original frozen implementation."""
    import yaml
    parameters = yaml.safe_load(Path(config).read_text())
    sources = sorted(list((BASE/'cleaner').rglob('*.py'))
        + list((BASE/'evaluation').glob('*.py'))
        + [BASE/'auto_clean_gaussians.py', BASE/'execution_provenance.py'])
    identity = dict(config_sha256=sha(config),
        sources={str(f.relative_to(BASE)):sha(f) for f in sources},
        heldout=[i['plant'] for i in manifest if i['plant'] not in DEVELOPMENT])
    path = root/'FROZEN_HELDOUT_PROTOCOL.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(key) != value for key,value in identity.items()):
            raise RuntimeError('Frozen heldout protocol differs from requested configuration/source/inventory; use a separate experiment')
        return old
    development = {}
    for item in manifest:
        if item['plant'] not in DEVELOPMENT:continue
        out = Path(item['output'])
        complete = json.loads((out/'COMPLETE.json').read_text())
        if complete['config'].get('implementation_sources') != identity['sources']:
            raise RuntimeError(f'{item["plant"]} was measured using a different implementation')
        if any(complete['config'].get(key) != value for key,value in parameters.items()):
            raise RuntimeError(f'{item["plant"]} was measured using different cleaning parameters')
        development[item['plant']] = dict(complete_sha256=sha(out/'COMPLETE.json'),
            direct_review_sha256=sha(out/'comparison/official_supersplat/DIRECT_IMAGE_REVIEW.json'))
    reviews = {i['plant']:validated_review(i['output']) for i in manifest if i['plant'] in DEVELOPMENT}
    write_json(path, dict(**identity, development=development, frozen_at=time.time(),
        purpose='Heldout evaluation of a fixed method, not universal visual acceptance',
        known_visual_failures=[p for p,r in reviews.items() if not r or not r.get('eligible_for_batch',False)]))
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=BASE/'configs/plant_manifest.json')
    p.add_argument('--output',type=Path,default=BASE/'experiments/Historical_TwoStage_Cleaner')
    p.add_argument('--config',type=Path,default=BASE/'configs/historical_two_stage.yaml')
    p.add_argument('--development-only',action='store_true')
    p.add_argument('--editor-source',type=Path,help='Locally validated SuperSplat source/build for automatic actual-model captures')
    p.add_argument('--serial-postprocessing',action='store_true',
        help='Wait for CPU retirement/comparison/capture before starting the next GPU scene')
    p.add_argument('--plants',nargs='+',help='Optional explicit inventory subset; never changes cleaning parameters')
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (BASE/'experiments/.historical_gpu1.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A historical two-stage serial batch already owns GPU1')
    write_json(root/'BATCH_DRIVER_IDENTITY.json',dict(started=time.time(),
        batch_pid=os.getpid(),physical_gpu=1,
        overlap_cpu_delivery=not args.serial_postprocessing,
        maximum_gpu_children=1,maximum_delivery_workers=1,
        sources={name:sha(BASE/name) for name in ['run_historical_two_stage.py',
            'historical_batch_scheduler.py','compact_historical_outputs.py']},
        cleaning_protocol='Unchanged sources/config recorded in FROZEN_HELDOUT_PROTOCOL.json'))
    inventory = json.loads(args.manifest.read_text())['plants']
    if args.plants:
        unknown = set(args.plants)-{i['plant'] for i in inventory}
        if unknown:
            p.error(f'Unknown plants: {sorted(unknown)}')
        inventory = [i for i in inventory if i['plant'] in args.plants]
    inventory.sort(key=lambda i:(DEVELOPMENT.index(i['plant']) if i['plant'] in DEVELOPMENT else len(DEVELOPMENT),i['plant']))
    manifest = []
    for item in inventory:
        plant = item['plant']
        sub = plant if plant == 'KongQueZhuYu' else ('cross_plant/'+plant if plant in DEVELOPMENT else 'all_19_plants/'+plant)
        scene_path = augmented_scene(item['scene'], root)
        scene = json.loads(scene_path.read_text())
        manifest.append(dict(plant=plant, scene=str(scene_path), output=str(root/sub),
            status='pending' if Path(scene['model']).exists() else 'missing_native_model'))
    write_json(root/'run_manifest.json',dict(plants=manifest, config=str(args.config.resolve()),config_sha256=sha(args.config),
        physical_gpu=1, retraining=False, mesh_edits=False, created=time.time()))
    summarize(root,manifest)
    frozen = False
    activity = {}
    activity_lock = threading.Lock()
    summary_lock = threading.Lock()

    def set_activity(slot, value):
        with activity_lock:
            if value is None:
                activity.pop(slot, None)
            else:
                activity[slot] = dict(**value, updated=time.time())
            if activity:
                write_json(root/'ACTIVE_JOB.json',dict(batch_pid=os.getpid(),
                    gpu=activity.get('gpu'),postprocessing=activity.get('postprocessing'),
                    updated=time.time()))
            else:
                (root/'ACTIVE_JOB.json').unlink(missing_ok=True)

    def publish():
        with summary_lock:
            summarize(root,manifest)

    env = dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    def jobs():
        nonlocal frozen
        for item in manifest:
            plant = item['plant']
            if plant in DEVELOPMENT:
                if item['status'] != 'missing_native_model':
                    yield item
                continue
            if args.development_only:
                break
            if not frozen:
                passed, reason = development_gate(manifest)
                write_json(root/'DEVELOPMENT_GATE.json',dict(passed=passed,reason=reason))
                if not passed:
                    print('Development evidence requires further work:',reason,flush=True)
                    break
                freeze_heldout_protocol(root,args.config,manifest)
                frozen = True
            if item['status'] != 'missing_native_model':
                yield item

    def clean(item):
        plant = item['plant']
        command = [shutil_uv(),'run','--no-project','--python',str(PROJECT/'.venv/bin/python3'),'python','-u',
            str(BASE/'auto_clean_gaussians.py'),'--scene',item['scene'],'--strategy','historical_two_stage',
            '--output',item['output'],'--config',str(args.config.resolve()),'--resume']
        logfile = root/'logs'/f'{plant}.log'
        logfile.parent.mkdir(parents=True,exist_ok=True)
        started=time.time()
        print('Cleaning/validating:',plant,flush=True)
        try:
            with logfile.open('a') as log:
                process = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
                set_activity('gpu',dict(plant=plant,phase='cleaning',pid=process.pid,
                    command=command,log=str(logfile),started=started))
                code = process.wait()
        finally:
            set_activity('gpu',None)
        item['status'] = 'complete' if code == 0 else 'failed'
        write_json(root/'LAST_JOB.json',dict(plant=plant,returncode=code,
            phase='cleaning',wall_seconds=time.time()-started))
        publish()
        if code:
            raise RuntimeError(f'{plant} failed; inspect {logfile}')

    def postprocess(item):
        plant=item['plant']
        started=time.time()
        def phase(name):
            print('Postprocessing:',plant,name,flush=True)
            set_activity('postprocessing',dict(plant=plant,phase=name,started=started))
        try:
            phase('compacting_trials')
            from compact_historical_outputs import compact
            compact(Path(item['output']))
            phase('assembling_comparison')
            subprocess.run([sys.executable,str(BASE/'collect_historical_comparison.py'),
                '--output',item['output']],check=True)
            if args.editor_source and not validated_review(item['output']):
                phase('capturing_actual_editor')
                labels=['B0_original','B2_stage1','B3_stage2']
                if (Path(item['output'])/'comparison/models/B1_historical_best_supersplat.ply').exists():
                    labels.insert(1,'B1_historical_best')
                subprocess.run([sys.executable,str(BASE/'review_historical_supersplat.py'),
                    '--experiment',item['output'],'--editor-source',str(args.editor_source.resolve()),
                    '--models',*labels],check=True)
                subprocess.run([sys.executable,str(BASE/'make_historical_review_contact.py'),
                    '--experiment',item['output']],check=True)
            write_json(Path(item['output'])/'POSTPROCESS_COMPLETE.json',dict(plant=plant,
                wall_seconds=time.time()-started,completed=time.time(),
                direct_review_recorded=bool(validated_review(item['output']))))
            (Path(item['output'])/'POSTPROCESS_FAILED.json').unlink(missing_ok=True)
        except Exception as error:
            write_json(Path(item['output'])/'POSTPROCESS_FAILED.json',dict(plant=plant,
                error=repr(error),failed=time.time()))
            raise
        finally:
            set_activity('postprocessing',None)
        publish()

    try:
        execute_pipeline(jobs(),clean,postprocess,overlap=not args.serial_postprocessing)
        (root/'BATCH_FAILED.json').unlink(missing_ok=True)
    except Exception as error:
        write_json(root/'BATCH_FAILED.json',dict(error=repr(error),failed=time.time()))
        raise
    finally:
        publish()
        subprocess.run([sys.executable,str(BASE/'report_historical_two_stage.py'),'--root',str(root)],check=True)


def shutil_uv():
    import shutil
    uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('uv is required')
    return uv


if __name__ == '__main__':
    main()
