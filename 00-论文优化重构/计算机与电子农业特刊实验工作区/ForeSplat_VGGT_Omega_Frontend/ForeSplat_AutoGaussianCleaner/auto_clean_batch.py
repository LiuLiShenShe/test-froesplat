#!/usr/bin/env python3
"""Restartable C5 batch, immutable protocol, explicit development/test boundary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
import subprocess
import time
import csv
import fcntl
import yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.execution_provenance import gaussian_count,native_renderer_fingerprint,completed_identity,scene_input_fingerprint,invalidate_output,resolve_variant


def decision_protocol(protocol):
    return {key:value for key,value in protocol.items() if key!='manifest_sha256'}


def validate_development_decision(manifest,protocol):
    decision=json.loads(Path(manifest['development_decision']).read_text())
    if decision.get('development_protocol')!=decision_protocol(protocol):
        raise ValueError('Development decision does not match this frozen method/source/config/camera protocol')
    expected={j['plant'] for j in manifest['plants'] if j.get('group')=='development'}
    results=decision.get('development_results',[])
    if {r['plant'] for r in results}!=expected or not expected:
        raise ValueError('Development decision must identify every real development result')
    variant=resolve_variant(protocol['method'],protocol['configuration'])
    for result in results:
        dest=Path(result['output'])
        marker=json.loads((dest/'COMPLETE.json').read_text())
        report=json.loads((dest/'cleanup_metrics.json').read_text())
        actual=marker.get('variant',report['method'])
        job=next(j for j in manifest['plants'] if j['plant']==result['plant'])
        spec=json.loads(Path(job['scene']).read_text())
        configuration=marker['config']
        expected_sources={k:v for k,v in protocol['sources'].items() if k.startswith(('cleaner/','evaluation/')) or k in ['auto_clean_gaussians.py','execution_provenance.py']}
        valid=(actual==variant and completed_identity(dest,configuration,spec,variant) and report['plant']==result['plant'] and marker['scene']==spec
            and marker['native_sha256']==sha(dest/'cleaned_gaussians.ply')
            and result.get('input_sha256')==marker['input_sha256']==sha(spec['model'])
            and configuration.get('implementation_sources')==expected_sources
            and configuration.get('scene_input_fingerprint')==protocol['scene_contents'][result['plant']]
            and all(configuration.get(k)==v for k,v in protocol['configuration'].items())
            and all(protocol['native_renderer_sources'].get(k)==v for k,v in configuration.get('native_renderer_fingerprint',{}).items())
            and bool(configuration.get('native_renderer_fingerprint')))
        if not valid:
            raise ValueError(f'Invalid development result: {result["plant"]}')
    return decision


def model_ready(job,spec):
    model=Path(spec['model'])
    if not model.exists():return False
    training=Path(job['training_dir']) if job.get('training_dir') else None
    if training is not None and model.is_relative_to(training):
        managed=any((training/name).exists() for name in ['TRAINING_PROTOCOL.json','ADAPTIVE_TRAINING_PROTOCOL.json','actual_training_parameters.json'])
        if managed and not (training/'densification_events.complete.json').exists():return False
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--method',default='C5',choices=['C5','C5-C','C5-D'])
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--group',choices=['development','independent_test','all'],default='all')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-for-models',action='store_true',help='Observe separately running locked GGGS preparation; no frontend inference')
    p.add_argument('--preparation-status',type=Path,help='Training status JSON; stop waiting on an actual preparation failure')
    a=p.parse_args();base=Path(__file__).resolve().parent
    manifest=json.loads(a.manifest.read_text());out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    has_test=any(j.get('group')=='independent_test' for j in manifest['plants'])
    if has_test and a.group!='development' and not manifest.get('development_decision'):
        raise ValueError('Independent C5 execution requires a recorded C0–C5/development decision first')
    if manifest.get('development_decision') and not Path(manifest['development_decision']).exists():
        raise FileNotFoundError('Recorded development decision missing')
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=list((base/'cleaner').rglob('*.py'))+list((base/'evaluation').glob('*.py'))+[base/'auto_clean_gaussians.py',base/'execution_provenance.py',Path(__file__).resolve()]
    repositories={json.loads(Path(j['scene']).read_text())['gggs_repo'] for j in manifest['plants']}
    scene_contents={j['plant']:scene_input_fingerprint(json.loads(Path(j['scene']).read_text())) for j in manifest['plants']}
    def current_protocol():
        configuration=yaml.safe_load((base/'configs/default.yaml').read_text())
        configuration.update(yaml.safe_load((a.config or base/'configs/adaptive.yaml').read_text()))
        return dict(sources={str(f.relative_to(base)):sha(f) for f in sorted(sources)},
            configuration=configuration,
            config_sha256=sha(a.config or base/'configs/adaptive.yaml'),base_config_sha256=sha(base/'configs/default.yaml'),
            manifest_sha256=sha(a.manifest),method=a.method,
            native_renderer_sources=native_renderer_fingerprint(repositories),
            scene_contents=scene_contents,
            scenes_sha256={j['plant']:sha(j['scene']) for j in manifest['plants']})
    protocol=current_protocol()
    if has_test and a.group!='development':
        validate_development_decision(manifest,protocol)
        # Bind independent plants across output roots, rather than permitting a
        # new directory to silently reset their untouched-test status.
        ledger=a.manifest.resolve().parent/'INDEPENDENT_TEST_PROTOCOL.json'
        independent_lock=ledger.with_suffix('.lock').open('a')
        fcntl.flock(independent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        independent=dict(protocol=decision_protocol(protocol),plants=sorted(j['plant'] for j in manifest['plants'] if j.get('group')=='independent_test'))
        if ledger.exists() and json.loads(ledger.read_text())!=independent:
            raise ValueError('These independent plants already belong to another frozen test protocol; disclose any subsequent run as development')
        write_json(ledger,independent)
    freeze=out/'METHOD_FREEZE.json'
    if freeze.exists() and json.loads(freeze.read_text())!=protocol:
        raise ValueError('Frozen batch source/config/manifest changed; a new experiment and test disclosure are required')
    write_json(freeze,protocol)
    for j in manifest['plants']:
        j.setdefault('group','production')
    jobs=[j for j in manifest['plants'] if a.group=='all' or j['group']==a.group]
    jobs.sort(key=lambda j:(j['group']!='development',gaussian_count(json.loads(Path(j['scene']).read_text())['model']),j['plant']))
    write_json(out/'BATCH_ORDER.json',dict(plants=[j['plant'] for j in jobs],
        policy='Development first, then ascending available original native Gaussian count; missing models follow deterministically; no cleaning thresholds depend on this ordering'))
    status={}
    for j in jobs:
        if protocol!=current_protocol():
            raise RuntimeError('Frozen source/config/manifest/scene changed during batch')
        spec=json.loads(Path(j['scene']).read_text());dest=out/j['group']/j['plant']
        while a.wait_for_models and not model_ready(j,spec):
            status[j['plant']]=dict(status='waiting_for_gggs_checkpoint',source=spec['model'])
            write_json(out/'batch_status.json',status);summarize(manifest,out,status)
            if a.preparation_status and a.preparation_status.exists():
                preparation=json.loads(a.preparation_status.read_text()).get(j['plant'],{})
                if preparation.get('status')=='failed':
                    status[j['plant']]=dict(status='missing_gggs_checkpoint',source=spec['model'],reason=preparation.get('reason','Training failed'))
                    break
            time.sleep(10)
        if not model_ready(j,spec):
            status[j['plant']]=dict(status='missing_gggs_checkpoint',source=spec['model']);continue
        if protocol!=current_protocol():
            raise RuntimeError('Frozen protocol changed while waiting for the GGGS checkpoint')
        if scene_input_fingerprint(spec)!=scene_contents[j['plant']]:
            raise RuntimeError(f'Frozen scene contents changed before cleaning: {j["plant"]}')
        original_sha=sha(spec['model'])
        cmd=['uv','run','--no-project','--python',sys.executable,'python','-u',str(base/'auto_clean_gaussians.py'),
            '--scene',j['scene'],'--method',a.method,'--output',str(dest)]
        if a.config:cmd+=['--config',str(a.config)]
        if a.resume:cmd+=['--resume']
        log=out/'logs'/f"{j['plant']}.log";log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            f.write(json.dumps(dict(command=cmd,started=time.time()))+'\n');f.flush()
            status[j['plant']]=dict(status='running',log=str(log));write_json(out/'batch_status.json',status)
            code=subprocess.call(cmd,cwd=base.parent,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
        status[j['plant']]=dict(status='complete' if code==0 else 'failed',code=code,output=str(dest),log=str(log))
        try:
            intact=(scene_input_fingerprint(spec)==scene_contents[j['plant']] and protocol==current_protocol() and sha(spec['model'])==original_sha)
        except (OSError,ValueError,KeyError):
            intact=False
        if not intact:
            reason=f'Frozen scene contents/source protocol changed during cleaning: {j["plant"]}'
            status[j['plant']]['status']='invalidated'
            invalidate_output(dest,reason)
            write_json(out/'batch_status.json',status);summarize(manifest,out,status)
            raise RuntimeError(reason)
        if code==0 and not (dest/'COMPLETE.json').exists():
            status[j['plant']]['status']='failed'
            status[j['plant']]['reason']='Child exited successfully without a certified completion marker'
        write_json(out/'batch_status.json',status);summarize(manifest,out,status)
    write_json(out/'batch_status.json',status);summarize(manifest,out,status)
    if any(s['status'] in ['failed','missing_gggs_checkpoint'] for s in status.values()):raise SystemExit(1)


def summarize(manifest,out,status):
    rows=[]
    frozen=json.loads((out/'METHOD_FREEZE.json').read_text()) if (out/'METHOD_FREEZE.json').exists() else None
    for j in manifest['plants']:
        dest=out/j['group']/j['plant'];spec=json.loads(Path(j['scene']).read_text())
        r=dict(plant=j['plant'],group=j['group'],status=status.get(j['plant'],{}).get('status','pending'),
            source_ply=spec['model'],model_available=model_ready(j,spec),original_gaussians='',cleaned_gaussians='',removed_gaussians='',refined_gaussians='',
            background_reduction='',free_background_reduction='',fg_psnr_delta='',fg_alpha_delta='',boundary_f1_delta='',fg_local_boundary_f1_delta='',thin_coverage_delta='',
            depth_changed_fraction='',normal_p95_deg='',seconds='',peak_allocated_mib='',sampled_process_peak_mib='',gpu_sampling_seconds='',
            supported_fg_alpha_delta='',supported_thin_coverage_delta='',supported_depth_changed_fraction='',unsupported_photo_worse_pixels='',unsupported_alpha_holes='',
            native_ply='',supersplat_ply='',removed_ply='',outcome='',
            training_wall_seconds='',training_peak_allocated_gib='',input_frontend=spec.get('frontend','provided'))
        if j.get('training_dir'):
            training=Path(j['training_dir'])/'densification_events.complete.json'
            if training.exists():
                t=json.loads(training.read_text());r.update(training_wall_seconds=t['wall_seconds'],training_peak_allocated_gib=t.get('peak_allocated_gib',''))
        marker=dest/'COMPLETE.json'
        if marker.exists() and r['status'] not in ['failed','missing_gggs_checkpoint','running','waiting_for_gggs_checkpoint','invalidated'] and not (dest/'INVALIDATED.json').exists():
            try:
                m=json.loads((dest/'cleanup_metrics.json').read_text())
                valid=completed_identity(dest,m['config'],spec,m['method'])
                if frozen is not None:
                    requested=resolve_variant(frozen['method'],frozen['configuration'])
                    valid &= m['method']==requested
                    valid &= all(m['config'].get(k)==v for k,v in frozen['configuration'].items())
                    valid &= all(frozen['sources'].get(k)==v for k,v in m['config'].get('implementation_sources',{}).items())
            except (OSError,ValueError,KeyError):
                valid=False
            if not valid:
                r['status']='invalid_completed_output';rows.append(r);continue
            ini,fin=m['initial']['mean'],m['final']['mean']
            r.update({k:m[k] for k in ['original_gaussians','cleaned_gaussians','removed_gaussians','refined_gaussians','seconds','peak_allocated_mib','native_ply','supersplat_ply','removed_ply','outcome']})
            r['status']='complete'
            r.update(background_reduction=m['artifact_reduction'],fg_psnr_delta=fin['fg_psnr']-ini['fg_psnr'],
                free_background_reduction=m.get('free_artifact_reduction',''),fg_alpha_delta=fin['fg_alpha']-ini['fg_alpha'],
                fg_local_boundary_f1_delta=fin['fg_local_boundary_f1']-ini['fg_local_boundary_f1'],
                boundary_f1_delta=fin['boundary_f1']-ini['boundary_f1'],thin_coverage_delta=fin['thin_coverage']-ini['thin_coverage'],
                depth_changed_fraction=fin['depth_changed_fraction'],normal_p95_deg=fin['normal_angle_p95_deg'])
            gpu=m.get('gpu_memory',{})
            r.update(sampled_process_peak_mib=gpu.get('sampled_process_peak_mib',''),gpu_sampling_seconds=gpu.get('sampling_interval_seconds',''))
            if 'supported_fg_alpha' in fin:
                r.update(supported_fg_alpha_delta=fin['supported_fg_alpha']-ini['supported_fg_alpha'],
                    supported_thin_coverage_delta=fin['supported_thin_coverage']-ini['supported_thin_coverage'],
                    supported_depth_changed_fraction=fin['supported_depth_changed_fraction'],
                    unsupported_photo_worse_pixels=sum(v['unsupported_photo_worse_pixels'] for v in m['final']['views']),
                    unsupported_alpha_holes=sum(v['unsupported_new_alpha_holes'] for v in m['final']['views']))
        rows.append(r)
    with (out/'CROSS_PLANT_GGGS_CLEANING.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    completed=[r for r in rows if r['native_ply']];tests=[r for r in completed if r['group']=='independent_test']
    write_json(out/'GENERALIZATION_STATUS.json',dict(completed=len(completed),expected=len(rows),
        independent_test_completed=len(tests),modified=sum(r['removed_gaussians']>0 or r['refined_gaussians']>0 for r in completed),
        zero_modification=sum(r['removed_gaussians']==0 and r['refined_gaussians']==0 for r in completed),
        warning='Pending/missing rows have no invented metrics; only frozen independent tests support generalization'))


if __name__=='__main__':main()
