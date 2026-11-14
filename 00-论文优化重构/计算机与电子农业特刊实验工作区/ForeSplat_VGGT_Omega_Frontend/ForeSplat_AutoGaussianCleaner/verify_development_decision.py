#!/usr/bin/env python3
"""Read-only pre/post-test binding of the selected reviewed development models.

The frozen batch's source checks remain unchanged. This certificate supplements
them with the decision's exact native, report, review and summary identities.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.execution_provenance import completed_identity
from ForeSplat_AutoGaussianCleaner.integration import read_cleaning_result
from ForeSplat_AutoGaussianCleaner.summarize_cross_plant_comparison import paired_clouds,verified_visual_review
from ForeSplat_AutoGaussianCleaner.prepare_development_decision import require_matched_main_identity


def read(path):
    return json.loads(Path(path).read_text())


def require_recorded_identity(record,marker,report):
    if (any(record[k]!=marker[k] for k in ['input_sha256','native_sha256'])
            or report['config']!=marker['config']):
        raise ValueError('The selected reviewed development model was replaced or its report configuration changed')


def verify(manifest_path,source_root,config_path,output):
    manifest=read(manifest_path);decision_path=Path(manifest['development_decision']);decision=read(decision_path)
    protocol=decision['development_protocol']
    if (len(manifest['plants'])!=len(decision['plant_groups'])
            or {j['plant']:j['group'] for j in manifest['plants']}!=decision['plant_groups']):
        raise ValueError('The development/independent plant split changed after the decision')
    expected={j['plant'] for j in manifest['plants'] if j['group']=='development'}
    records=decision['development_results']
    if len(records)!=len(expected) or {r['plant'] for r in records}!=expected:
        raise ValueError('Decision does not identify every development model exactly once')
    for rel,digest in protocol['sources'].items():
        if sha(source_root/rel)!=digest:raise ValueError('Promoted source differs from frozen development: '+rel)
    if any(sha(j['scene'])!=protocol['scenes_sha256'][j['plant']] for j in manifest['plants']):
        raise ValueError('A recorded native scene configuration changed')
    if sha(config_path)!=protocol['config_sha256'] or sha(source_root/'configs/default.yaml')!=protocol['base_config_sha256']:
        raise ValueError('Promoted configuration differs from frozen development')
    for key in ['decision_notes','development_summary']:
        if sha(decision[key])!=decision[key+'_sha256']:raise ValueError('Recorded decision evidence changed: '+key)
    verified=[]
    for record in records:
        job=next(j for j in manifest['plants'] if j['plant']==record['plant'])
        path=Path(record['output']);marker=read(path/'COMPLETE.json');report=read(path/'cleanup_metrics.json')
        require_recorded_identity(record,marker,report)
        if sha(path/'cleanup_metrics.json')!=record['cleanup_report_sha256'] or sha(record['visual_review'])!=record['visual_review_sha256']:
            raise ValueError('The reviewed report or visual observations changed: '+record['plant'])
        result=read_cleaning_result(job['scene'],path)
        verified_visual_review(path,result);paired_clouds(path,report,marker['native_sha256'])
        verified.append(dict(plant=record['plant'],input_sha256=result['input_sha256'],native_sha256=result['native_sha256'],
            cleanup_report_sha256=record['cleanup_report_sha256'],visual_review_sha256=record['visual_review_sha256']))
    main_scene=read(next(j['scene'] for j in manifest['plants'] if j['plant']=='KongQueZhuYu'))
    for record in decision['matched_main_ablations']:
        path=Path(record['output']);marker=read(path/'COMPLETE.json');report=read(path/'cleanup_metrics.json')
        require_recorded_identity(record,marker,report)
        if not completed_identity(path,report['config'],main_scene,record['method']):raise ValueError('Matched main ablation changed')
        if (sha(path/'cleanup_metrics.json')!=record['cleanup_report_sha256']
                or sha(path/'COMMON_COMPARISON_EVALUATION_COMPLETE.json')!=record['common_evaluation_sha256']
                or sha(record['direct_visual_review'])!=record['direct_visual_review_sha256']):
            raise ValueError('Matched ablation evidence changed')
        require_matched_main_identity(report,protocol,read(path/'COMMON_COMPARISON_EVALUATION_COMPLETE.json'),path,marker)
        verified_visual_review(path,marker);paired_clouds(path,report,marker['native_sha256'])
    result=dict(decision=str(decision_path.resolve()),decision_sha256=sha(decision_path),
        manifest=str(manifest_path.resolve()),manifest_sha256=sha(manifest_path),development_models=verified,
        source_root=str(source_root.resolve()),config_sha256=sha(config_path),verifier_sha256=sha(Path(__file__)),
        status='Selected reviewed development native models, reports, visual evidence, ablations and promoted sources verified',
        scope='Read-only integrity certificate; no independent models rendered or cleaner parameters changed')
    write_json(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.manifest,args.source_root,args.config,args.output)
    print(args.output.resolve())


if __name__=='__main__':main()
