#!/usr/bin/env python3
"""Verify real stage exports, whole-view evaluation and edit lineage."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,sha,write_json
from ForeSplat_AutoGaussianCleaner.integration import read_cleaning_result
from viewer_review_validity import validated_review


def audit(output,scene_path):
    output=Path(output).resolve()
    pipeline=read_cleaning_result(scene_path,output)
    result=json.loads((output/'cleanup_metrics.json').read_text())
    original=read_ply(result['scene']['model'])
    stages={}
    for key,folder in [('stage1','stage1_generalized'),('stage2','stage2_residual')]:
        directory=output/folder
        marker=json.loads((directory/'COMPLETE.json').read_text())
        assert sha(directory/'cleanup_metrics.json')==marker['metrics_sha256'],'Stage metrics changed'
        assert sha(directory/'cleaned_gaussians.ply')==marker['native_sha256'],'Stage native changed'
        rows=read_ply(directory/'cleaned_gaussians.ply')
        ids=np.load(directory/'original_gaussian_ids.npy')
        assert len(rows)==len(ids)==result[key]['final_gaussians']
        assert len(np.unique(ids))==len(ids) and np.all(np.diff(ids)>0)
        assert np.all(ids>=0) and np.all(ids<len(original))
        allowed={'opacity','filter_3D','scale_0','scale_1','scale_2'} if key=='stage2' else set()
        changed=set()
        refined=np.zeros(len(rows),bool)
        for name in rows.dtype.names:
            mismatch=rows[name]!=original[name][ids]
            if mismatch.any():changed.add(name);refined|=mismatch
        assert changed<=allowed, f'Unexpected native fields changed: {changed-allowed}'
        assert int(refined.sum())==result[key]['refined_gaussians']
        stages[key]=dict(rows=len(rows),unique_original_ids=len(ids),changed_fields=sorted(changed),
            refined_rows=int(refined.sum()),whole_views=len(result[key]['final']['views']),
            native_sha256=marker['native_sha256'],reported_rollback=result[key]['rolled_back'])
        del rows,ids
    review=validated_review(output)
    selected=json.loads((output/'VISUAL_SELECTION.json').read_text()) if (output/'VISUAL_SELECTION.json').exists() else None
    if selected:
        assert review and selected['selected_model']==review['recommended_model']
        for model in selected['models'].values():
            assert sha(model['path'])==model['sha256']
    record=dict(plant=result['plant'],passed=True,pipeline_result=pipeline,stage_lineage=stages,
        verified_direct_browser_review=bool(review),verified_selected_model=selected['selected_model'] if selected else None,
        evidence_sha256={name:sha(output/name) for name in ['COMPLETE.json','cleanup_metrics.json',
            'comparison/official_supersplat/REVIEW.json','comparison/official_supersplat/DIRECT_IMAGE_REVIEW.json',
            'VISUAL_SELECTION.json'] if (output/name).is_file()},
        source='Actual native/viewer artifacts, hashes, original row lineage, full camera inventory and browser capture provenance')
    write_json(output/'DELIVERY_AUDIT.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.root/'run_manifest.json').read_text())['plants'];records=[]
    for item in manifest:
        output=Path(item['output'])
        if not (output/'COMPLETE.json').exists():continue
        try:record=audit(output,item['scene'])
        except Exception as error:record=dict(plant=item['plant'],passed=False,error=repr(error))
        records.append(record);print(record['plant'],record['passed'],record.get('error',''),flush=True)
    write_json(a.root/'metrics/delivery_audits.json',dict(plants=records,all_completed_valid=all(r['passed'] for r in records)))
    if any(not r['passed'] for r in records):raise SystemExit(1)


if __name__=='__main__':main()
