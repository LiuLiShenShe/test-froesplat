#!/usr/bin/env python3
"""Assemble actual B0/B1/B2/B3 native outputs without mixing view protocols."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply, save_ply, bake_viewer, write_json, sha


def alias(source,target):
    source,target=Path(source).resolve(),Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink() and target.resolve()==source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source)


def contact(models, protocol, background, target):
    # These are real existing rasterizations, never synthesized plant imagery.
    available=[]
    for label,folder in models:
        files=sorted((folder/protocol).glob(f'*_{background}.png'))
        if not files:
            return None
        available.append((label,{p.name:p for p in files}))
    common=sorted(set.intersection(*(set(p) for _,p in available)))
    if not common:
        return None
    thumb=(320,300)
    canvas=Image.new('RGB',(len(available)*thumb[0],len(common)*(thumb[1]+28)), 'white')
    draw=ImageDraw.Draw(canvas)
    for y,name in enumerate(common):
        for x,(label,files) in enumerate(available):
            with Image.open(files[name]) as source:
                im=source.convert('RGB')
                im.thumbnail(thumb)
                left=x*thumb[0]+(thumb[0]-im.width)//2
                top=y*(thumb[1]+28)+28+(thumb[1]-im.height)//2
                canvas.paste(im,(left,top))
            draw.text((x*thumb[0]+5,y*(thumb[1]+28)+5),f'{label}: {name}',fill='black')
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target)
    return str(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'experiments/Historical_TwoStage_Cleaner/KongQueZhuYu')
    a=p.parse_args()
    output=a.output.resolve()
    marker=json.loads((output/'COMPLETE.json').read_text())
    report=json.loads((output/'cleanup_metrics.json').read_text())
    if not marker.get('source_and_input_verification') or sha(output/'cleaned_gaussians.ply')!=marker['native_sha256']:
        raise ValueError('Require an actually complete and verified two-stage model')
    spec=report['scene']
    model_dir=output/'comparison/models'
    original=Path(spec['model'])
    alias(original,model_dir/'B0_original_native.ply')
    b0_viewer=model_dir/'B0_original_supersplat.ply'
    if not b0_viewer.exists():
        source_rows=read_ply(original)
        viewer,baking=bake_viewer(source_rows)
        save_ply(b0_viewer,viewer)
        write_json(model_dir/'B0_viewer_baking.json',dict(**baking,input_sha256=sha(original),viewer_sha256=sha(b0_viewer)))
    for label,folder in [('B2_stage1','stage1_generalized'),('B3_stage2','stage2_residual')]:
        for source,suffix in [('cleaned_gaussians.ply','native'),('cleaned_gaussians_supersplat.ply','supersplat')]:
            alias(output/folder/source,model_dir/f'{label}_{suffix}.ply')
    historical=report['models'].get('historical')
    if historical:
        historical=Path(historical)
        alias(historical,model_dir/'B1_historical_best_native.ply')
        alias(historical.with_name(historical.stem+'_supersplat.ply'),model_dir/'B1_historical_best_supersplat.ply')
        round5=historical.parent.parent/'additional_rounds/round_05'
        alias(round5/'gaussians.ply',model_dir/'R5_with_marker_native.ply')
        alias(round5/'gaussians_supersplat.ply',model_dir/'R5_with_marker_supersplat.ply')
    render_sets=[('B0 original',output/'stage1_generalized/visualizations_input'),
                 ('B1 historical' if historical else 'Stage2 input',output/'stage2_residual/visualizations_input'),
                 ('B2 Stage1',output/'stage1_generalized/visualizations'),
                 ('B3 Stage2',output/'stage2_residual/visualizations')]
    images=[]
    for protocol in ['free','historical_free']:
        for background in ['white','black']:
            path=contact(render_sets,protocol,background,output/'comparison'/f'{protocol}_{background}_comparison.png')
            if path:
                images.append(path)
    metrics=[]
    for label,folder in render_sets:
        for protocol,filename in [('historical_fixed8','historical_free_metrics.json'),('automatic_depth_trajectory','automatic_free_views.json')]:
            path=folder/filename
            if path.exists():
                row=json.loads(path.read_text())
                metrics.append(dict(model=label,protocol=protocol,source=str(path),values=row))
    write_json(output/'comparison/view_protocols.json',metrics)
    paths={p.stem:str(p.resolve()) for p in sorted(model_dir.glob('*.ply'))}
    write_json(output/'comparison/model_index.json',dict(models=paths,comparison_images=images,
        interactive_supersplat_review='pending_until_actual_editor_captures_are_inspected',
        provenance='Only real model files and existing native rasterizations; fixed historical and automatic camera protocols remain separate'))
    print(json.dumps(dict(models=paths,comparison_images=images),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
