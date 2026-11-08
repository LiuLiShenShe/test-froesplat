"""Strict24 background-only factor; GPU run only after70/110 serial jobs.

Foreground pixels always equal RAP; only mask-exterior RGB comes from02-FFT.
This is background restoration, not untouched original RGB. Original files are
read-only. Preparation writes new PNGs, asserts exact roundtrip pixel identity,
and saves source hashes. All model calls reuse the count-control original APIs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

CONTROL=Path(__file__).resolve().parent;AUDIT=CONTROL.parents[1]
DRIVER=AUDIT/'controlled_fixes/view_count_control_v1/run_view_count_control.py'
spec=importlib.util.spec_from_file_location('count_control_shared',DRIVER)
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
IDS=[0,1,19,20,38,39,57,58,76,77,95,96,114,115,133,134,152,153,170,171,189,190,208,209]
RAP=Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu');ORIGINAL=Path('/data/fj/F2DMAS/02-FFT/KongQueZhuYu')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prepare():
    if (CONTROL/'prepared_sources.json').exists():
        print('Reuse completed exact pixel-asserted preparation; no repeated JPEG/RAP scan',flush=True)
        return json.loads((CONTROL/'prepared_sources.json').read_text())
    raw_by_id={r['source_index']:r for r in common.source_rows()}
    baseline=[];restored=[];checks=[]
    images_dir=CONTROL/'prepared_images/background_restored';images_dir.mkdir(parents=True,exist_ok=True)
    try:
        for source in IDS:
            row=dict(raw_by_id[source]);name=row['image_name']
            rgb_path=RAP/name;mask_path=RAP/f'mask_{source:04d}.png';orig_path=ORIGINAL/f'{source:04d}.jpg'
            if not rgb_path.exists() or not mask_path.exists() or not orig_path.exists():raise FileNotFoundError(f'Fixed source{source}: exact original/RAP/Mask must exist')
            with Image.open(rgb_path) as im:rgb=np.array(im.convert('RGB'))
            with Image.open(mask_path) as im:mask_raw=np.array(im.convert('L'))
            with Image.open(orig_path) as im:original=np.array(im.convert('RGB'))
            if rgb.shape!=original.shape or mask_raw.shape!=rgb.shape[:2]:raise ValueError(f'Fixed source{source}: no resize allowed to fix mismatched original/RAP/Mask size')
            if not set(np.unique(mask_raw)).issubset({0,255}):raise ValueError('Require original binary RAP mask')
            mask=mask_raw>=128;composite=np.where(mask[...,None],rgb,original)
            out=images_dir/name;Image.fromarray(composite).save(out)
            with Image.open(out) as im:roundtrip=np.array(im.convert('RGB'))
            if not np.array_equal(roundtrip[mask],rgb[mask]):raise AssertionError('Foreground RAP pixels changed')
            if not np.array_equal(roundtrip[~mask],original[~mask]):raise AssertionError('Restored outside does not equal original')
            changed=np.any(roundtrip!=rgb,axis=-1)
            if np.any(changed&mask):raise AssertionError('Condition difference extends into foreground')
            source_info=dict(rap_path=str(rgb_path),original_jpeg_path=str(orig_path),mask_path=str(mask_path),
                rap_file_sha256=digest(rgb_path),original_jpeg_sha256=digest(orig_path),mask_file_sha256=digest(mask_path),
                restored_file_sha256=digest(out),shape_hwc=list(rgb.shape),foreground_pixels=int(mask.sum()),background_pixels=int((~mask).sum()),
                changed_foreground_pixels=int((changed&mask).sum()),changed_background_pixels=int((changed&~mask).sum()),
                png_roundtrip_foreground_exact=True,png_roundtrip_background_exact=True,
                original_vs_RAP_inside_MAE=float(np.abs(original[mask].astype(np.int16)-rgb[mask].astype(np.int16)).mean()))
            checks.append(dict(source_index=source,image_name=name,**source_info))
            for cond,impath,sha in [('RAP_baseline',rgb_path,source_info['rap_file_sha256']),('background_restored',out,source_info['restored_file_sha256'])]:
                item=dict(row,image_path=str(impath),mask_path=str(mask_path),condition=cond,rgb_sha256_actual=sha,mask_sha256_actual=source_info['mask_file_sha256'],
                    original_background_path=str(orig_path),hash_scope='Actual condition RGB/shared RAP Mask SHA measured in CPU preparation; inside restoredMask exact RAP, outside exact original02-FFT RGB.')
                (baseline if cond=='RAP_baseline' else restored).append(item)
            print('CPUprepared exact source',source,'BGchanged',source_info['changed_background_pixels'],flush=True)
        data=dict(schema='foresplat.background_factor.preparation.v1',source_ids=IDS,baseline=baseline,restored=restored,
            pixel_assertions=checks,condition_definition='Only outside original binary RAP Mask replaced by matched02-FFT RGB; foreground remains exact RAP. Not unmodified original RGB.',
            all_foreground_differences_zero=True,original_files_modified=False,GPU_inference_executed=False)
        common.write_json(CONTROL/'prepared_sources.json',data)
        return data
    except Exception as exc:
        common.write_json(CONTROL/'preparation_failed.json',dict(status='failed',stage='CPU_background_preparation',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),fallback_original_substitution=False))
        raise

def selected(model,data):
    info=common.model_metadata(model)
    info.update(control_factor='background restoration outside Mask at fixed24',source_ids=IDS,
        background_condition_definition=data['condition_definition'],preparation_source_sha256=digest(CONTROL/'prepared_sources.json'))
    common.write_json(CONTROL/f'{model}_model_metadata.json',info)
    items={}
    for cond,rows in [('RAP_baseline',data['baseline']),('background_restored',data['restored'])]:
        out=CONTROL/f'{model}_{cond}_24'
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete':raise ValueError('Refuse completed-condition overwrite '+str(out))
        common.manifest(out,rows)
        common.write_json(out/'run_parameters.json',dict(**info,input_views=24,condition=cond,only_experimental_factor='outside-Mask background RGB',
            source_order=[r['image_name'] for r in rows],Mask_identical_between_conditions=True,original_resolution_identical=True,
            core_network_and_weights_unchanged=True,pixel_assertions_file=str(CONTROL/'prepared_sources.json')))
        common.write_json(out/'status.json',dict(status='prepared',model=model,input_views=24,condition=cond,GPU_inference_executed=False))
        items[cond]=(out,rows)
    return info,items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['Omega','Standard']);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run and not args.model:parser.error('--model required for allocated serial GPU job')
    if args.run and args.prepare_only:parser.error('Choose prepare-only or run')
    data=prepare()
    models=[args.model] if args.model else ['Omega','Standard']
    for model in models:
        info,items=selected(model,data)
        print(model,'24baseline/restored prepared',flush=True)
        if args.run:common.run(model,info,items)

if __name__=='__main__':main()
