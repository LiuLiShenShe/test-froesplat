"""Read historical experiment metadata and PLY headers; no training or GPU work.

Large point/parameter payloads are not scanned here. RGB byte/pixel comparisons
use only three explicit source views; all dataset files receive inode/header
checks. Historical defaults are distinguished from an actually saved cfg.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

AUDIT = Path(__file__).resolve().parents[1]
OMEGA = AUDIT.parents[2]
STANDARD = OMEGA.parent / 'ForeSplat_Standard_VGGT_Frontend'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ply_header(path):
    path=Path(path)
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline()
            if not line or f.tell()>65536: raise ValueError('invalid PLY header: '+str(path))
            lines.append(line.decode('ascii').strip())
            if lines[-1]=='end_header': break
        offset=f.tell()
    fields=[x.split()[2] for x in lines if x.startswith('property ') and not x.startswith('property list')]
    n=int(next(x for x in lines if x.startswith('element vertex ')).split()[-1])
    rest=sum(x.startswith('f_rest_') for x in fields)
    return dict(path=str(path),vertices=n,header_bytes=offset,file_bytes=path.stat().st_size,
                fields=fields,f_rest_count=rest,sh_capacity_degree=int(round(np.sqrt(rest/3+1)-1)) if rest else 0,
                format=lines[1],stat_inode=path.stat().st_ino)

def saved_cfg(path):
    text=Path(path).read_text()
    call=ast.parse(text,mode='eval').body
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Namespace':
        raise ValueError('not an argparse Namespace')
    return {k.arg:ast.literal_eval(k.value) for k in call.keywords}

def literal_defaults(path, class_name):
    tree=ast.parse(Path(path).read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name)
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    result={}
    for s in init.body:
        if isinstance(s,ast.Assign) and len(s.targets)==1 and isinstance(s.targets[0],ast.Attribute):
            try: result[s.targets[0].attr.lstrip('_')]=ast.literal_eval(s.value)
            except (ValueError,TypeError): pass
    return result

def camera_summary(path):
    rows=json.loads(Path(path).read_text())
    centers=np.array([r['position'] for r in rows],np.float64)
    radius=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
    dims=sorted({(int(r['width']),int(r['height'])) for r in rows})
    return dict(count=len(rows),stored_resolution_wh=dims,training_resolution_wh_resolution2=[(round(w/2),round(h/2)) for w,h in dims],
                recomputed_camera_radius_float64=radius,clone_split_size_cutoff_default=radius*.01,
                first_camera_name=rows[0].get('img_name'),source='saved camera JSON positions; tiny precision differences from float32 reader are possible')

def dataset_headers(path):
    path=Path(path)
    images=sorted((path/'images').glob('*.png'))
    masks=sorted((path/'masks').glob('*.png'))
    dims=set(); modes=set();mask_dims=set();mask_modes=set()
    for p in images:
        with Image.open(p) as im: dims.add(im.size); modes.add(im.mode)
    for p in masks:
        with Image.open(p) as im:mask_dims.add(im.size);mask_modes.add(im.mode)
    return dict(path=str(path),image_count=len(images),mask_count=len(masks),original_size_wh=sorted(dims),image_modes=sorted(modes),
                mask_size_wh=sorted(mask_dims),mask_modes=sorted(mask_modes),mask_and_rgb_header_size_equal=mask_dims==dims,
                names_match_masks=[p.name for p in images]==[p.name for p in masks])

def image_chain(omega_ds, standard_ds):
    paths=sorted((omega_ds/'images').glob('*.png'))
    same=sum((standard_ds/'images'/p.name).stat().st_ino==p.stat().st_ino and (standard_ds/'images'/p.name).stat().st_dev==p.stat().st_dev for p in paths)
    samples=[]
    for i in (0,len(paths)//2,len(paths)-1):
        p=paths[i]; other=standard_ds/'images'/p.name
        rap=Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu')/p.name
        with Image.open(p) as im: rgb=np.array(im.convert('RGB'))
        with Image.open(other) as im: standard=np.array(im.convert('RGB'))
        with Image.open(rap) as im: original=np.array(im.convert('RGB'))
        mask_path=omega_ds/'masks'/p.name
        with Image.open(mask_path) as im: mask=np.array(im.convert('L'))>=128
        bg=rgb[~mask]
        samples.append(dict(name=p.name,dataset_sha256=digest(p),standard_dataset_sha256=digest(other),rap_sha256=digest(rap),
                            decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            standard_pixel_equal=bool(np.array_equal(rgb,standard)),rap_pixel_equal=bool(np.array_equal(rgb,original)),
                            rap_byte_equal=p.read_bytes()==rap.read_bytes(),background_zero_rgb_fraction=float(np.mean(np.all(bg==0,axis=1))) if len(bg) else None,
                            background_mean_rgb=bg.mean(0).tolist() if len(bg) else None,
                            foreground_pixel_fraction=float(mask.mean()),source_rap=str(rap)))
    return dict(all208_standard_images_share_omega_inodes=same==len(paths),shared_inode_count=same,sampled_rgb_chain=samples,
                scope='all files inode/header checked; RAP RGB byte/pixel content checked only on first/middle/last, not a full208 RAP payload rehash')

def summarize_csv(path):
    rows=list(csv.DictReader(Path(path).open()))
    counts=[int(r['num_gaussians']) for r in rows]
    indexed={int(r['iteration']):r for r in rows}
    samples={str(i):indexed[i] for i in (100,500,600,700,1000,3000,6000,7000,7100,10000,14000,15000,18000,18100,21000,21100,24000,24100,27000,27100,30000) if i in indexed}
    return dict(path=str(path),rows=len(rows),intervals=sorted({int(rows[i+1]['iteration'])-int(rows[i]['iteration']) for i in range(len(rows)-1)}),
                initial_logged_gaussians=counts[0],last_logged_gaussians=counts[-1],min_logged_gaussians=min(counts),max_logged_gaussians=max(counts),
                sampled_rows=samples,measurement_stage='CSV is written before this iteration densification/mask pruning/optimizer step; component losses are instantaneous, not EMA')

def npz_metadata(path):
    with zipfile.ZipFile(path) as archive: keys=[p.filename[:-4] for p in archive.infolist()]
    with np.load(path,allow_pickle=False) as a:
        values={k:np.asarray(a[k]).item() for k in ('num_gaussians','training_seconds','peak_vram_mib') if k in keys}
    return dict(keys=keys,small_values=values,gradient_accumulators_present=any(k in keys for k in ('xyz_gradient_accum','xyz_gradient_accum_abs','denom')),
                contains_optimizer_state=False,scope='ZIP member index and scalar arrays only, no large parameter payload scan')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default=str(AUDIT/'gaussian_analysis/experiment_provenance.json')); args=parser.parse_args()
    names=['E1_GGGS_Baseline','E2_GGGS_EO','E3_GGGS_Foreground','E4_ForeSplat_GGGS']
    exps=[(f'E{i+1}',OMEGA/'experiments/KongQueZhuYu'/name) for i,name in enumerate(names)]
    base=STANDARD/'experiments/KongQueZhuYu/standard_vggt_comparison'
    exps += [('E5',base/'E5_Standard_VGGT_GGGS_Baseline'),('E6',base/'E6_2DGS_Standard_VGGT')]
    omega_ds=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'; standard_ds=base/'Standard_VGGT_Dataset'
    defaults=OMEGA/'gggs_community/arguments/__init__.py'
    rows=[]
    for name,path in exps:
        plies=[ply_header(p) for p in sorted((path/'checkpoints').rglob('*.ply')) if p.name=='point_cloud.ply' or p.name=='input.ply']
        log=path/('train_30000.log' if name=='E5' else 'train.log')
        log_text=log.read_text(errors='replace')
        firstline=log_text.splitlines()[0]
        npz=path/'checkpoints/final_gaussians/gaussians.npz'
        csvpath=path/'metrics/train_metrics.csv'; cfg=path/'checkpoints/cfg_args'
        rows.append(dict(id=name,path=str(path),training_script='GGGS train_foresplat.py' if name!='E6' else '2DGS train.py',
                         log_path=str(log),log_first_line=firstline,log_sha256=digest(log),log_normalized_text_sha256=hashlib.sha256(log_text.encode()).hexdigest(),
                         camera_summary=camera_summary(path/'checkpoints/cameras.json'),ply_checkpoints=plies,
                         csv=summarize_csv(csvpath) if csvpath.exists() else dict(present=False,note='No exact training component CSV; only stdout EMA/counts can be recovered'),
                         saved_cfg=saved_cfg(cfg) if cfg.exists() else None,npz=npz_metadata(npz) if npz.exists() else None,
                         optimizer_checkpoints=[str(p) for p in path.rglob('chkpnt*.pth')],
                         gradients_for_historical_densification_available=False if name!='E6' else 'E6 checkpoint exists; a different algorithm, not E5 GGGS accumulators'))
    output=dict(schema='foresplat.root_cause.provenance.v1',experiments=rows,datasets=[dataset_headers(omega_ds),dataset_headers(standard_ds)],
                dataset_image_chain=image_chain(omega_ds,standard_ds),current_source_defaults=dict(ModelParams=literal_defaults(defaults,'ModelParams'),OptimizationParams=literal_defaults(defaults,'OptimizationParams')),
                defaults_provenance='Current tracked official GGGS commit23b418c2; E1-E5 lack saved parsed cfg_args, runner/actual logs and outputs supply separate evidence',
                versions_scope='Environment separately captured as current audit environment, not assumed historical training environment',
                repository_limit='Omega adapter and 2DGS tree inherit enclosing F2DMAS git; enclosing f1a... is not their official model commit',
                no_training=True,no_gpu=True)
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(p)
    for row in rows:
        print(row['id'],row['camera_summary'],row['log_first_line'],[(Path(p['path']).parent.name,p['vertices'],p['sh_capacity_degree']) for p in row['ply_checkpoints']])

if __name__=='__main__': main()
