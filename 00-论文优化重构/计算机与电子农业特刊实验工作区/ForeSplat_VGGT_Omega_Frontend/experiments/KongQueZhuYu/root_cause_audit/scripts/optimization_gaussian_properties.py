"""One sequential CPU pass per final PLY, with a compact reusable cache.

Only saved scalar properties are decoded; no renderer, alignment, training or
frustum projection is performed. DC brightness is a proxy, not the actual SH3
view-dependent rendered color. Normalized scale uses each own camera extent.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

AUDIT=Path(__file__).resolve().parents[1]
COLUMNS=['x','y','z','dc_rgb_0','dc_rgb_1','dc_rgb_2','raw_alpha','sigma_0','sigma_1','sigma_2','filter_3D','filtered_alpha']
Q=[0,.01,.1,.5,.9,.99,1]

def header(path):
    raw=[]
    with path.open('rb') as f:
        while True:
            line=f.readline();raw.append(line)
            if not line or f.tell()>65536:raise ValueError('invalid PLY header')
            if line.strip()==b'end_header':break
        offset=f.tell()
    lines=b''.join(raw).decode('ascii').splitlines()
    if lines[1]!='format binary_little_endian 1.0':raise ValueError('Unsupported PLY format')
    props=[line.split() for line in lines if line.startswith('property ')]
    if any(p[1]!='float' for p in props):raise ValueError('Expected scalar float PLY properties')
    fields=[p[2] for p in props];dtype=np.dtype([(k,'<f4') for k in fields])
    n=int(next(line.split()[-1] for line in lines if line.startswith('element vertex ')))
    if path.stat().st_size!=offset+n*dtype.itemsize:raise ValueError('PLY length does not match vertex payload')
    return n,offset,dtype,hashlib.sha256(b''.join(raw)).hexdigest()

def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    good=np.isfinite(values)
    return {'finite_count':int(good.sum()),'nonfinite_count':int((~good).sum()),
            'quantiles':{str(q):float(v) for q,v in zip(Q,np.quantile(values[good],Q))} if good.any() else {}}

def analyze(row,chunk):
    ident=row['id'];paths=row['ply_checkpoints']
    entry=next(p for p in paths if ('final_gaussians' in p['path']) or (ident=='E6' and 'iteration_30000' in p['path']))
    path=Path(entry['path']);n,offset,dtype,hs=header(path)
    extent=row['camera_summary']['recomputed_camera_radius_float64']
    outdir=AUDIT/'gaussian_analysis';cache_path=outdir/f'{ident}_final_scalar_cache.npy'
    meta_path=cache_path.with_suffix('.json')
    source=dict(path=str(path),file_bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,header_sha256=hs,vertex_count=n)
    started=time.time()
    if not cache_path.exists():
        cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=(n,len(COLUMNS)))
        checksum=hashlib.sha256()
        with path.open('rb') as f:
            f.seek(offset)
            for start in range(0,n,chunk):
                count=min(chunk,n-start);raw=np.fromfile(f,dtype=dtype,count=count)
                if len(raw)!=count:raise ValueError('Short PLY payload')
                out=np.zeros((count,len(COLUMNS)),np.float32)
                out[:,:3]=np.stack([raw[k] for k in ('x','y','z')],axis=1)
                out[:,3:6]=np.clip(.5+.28209479177387814*np.stack([raw[f'f_dc_{i}'] for i in range(3)],axis=1),0,1)
                out[:,6]=1/(1+np.exp(-np.clip(raw['opacity'].astype(np.float64),-700,700)))
                scale_names=[k for k in dtype.names if k.startswith('scale_')]
                scales=np.exp(np.stack([raw[k] for k in scale_names],axis=1).astype(np.float64))
                out[:,7:7+len(scale_names)]=scales
                if len(scale_names)==2:out[:,9]=np.nan # 2DGS tangent scales: third sigma absent, not zero
                if 'filter_3D' in dtype.names:
                    filtering=raw['filter_3D'].astype(np.float64);out[:,10]=filtering
                    correction=np.prod(np.sqrt(1+(filtering[:,None]/scales)**2),axis=1)
                    out[:,11]=out[:,6]/correction
                else:
                    out[:,10:12]=np.nan # No GGGS 3D filter in E6
                cache[start:start+count]=out;checksum.update(out.tobytes())
                print(ident,'decoded',start+count,'/',n,flush=True)
        cache.flush();del cache
        meta_path.write_text(json.dumps(dict(source=source,columns=COLUMNS,selected_scalar_cache_sha256=checksum.hexdigest(),
                checksum_scope='SHA256 of newly decoded selected N×12 scalar cache payload, NOT the entire source PLY',
                no_raw_prediction_changes=True,scale_dimensions=2 if ident=='E6' else 3),indent=2)+'\n')
    else:
        previous=json.loads(meta_path.read_text())
        if previous['source']!=source:raise ValueError('Refuse stale cache; source provenance changed')
        print(ident,'reuse own previously completed selected scalar cache',flush=True)
    c=np.load(cache_path,mmap_mode='r');alpha=np.asarray(c[:,6],np.float64)
    dims=2 if ident=='E6' else 3
    sigma=np.max(c[:,7:7+dims],axis=1).astype(np.float64)
    bright=c[:,3:6].mean(axis=1).astype(np.float64)
    metrics=dict(id=ident,source=source,cache=str(cache_path),cache_columns=COLUMNS,extent=extent,
        scale_dimensions=dims,sh3_capacity=True,
        dc_color_interpretation='clamped SH DC-only RGB proxy; full SH3 directional color and composited alpha require actual renderer',
        raw_alpha=distribution(alpha),max_sigma=distribution(sigma),max_sigma_over_extent=distribution(sigma/extent),
        dc_brightness=distribution(bright),raw_alpha_below_prune_005=int((alpha<.05).sum()),
        raw_alpha_ge05_dc_brightness_lt003=int(((alpha>=.5)&(bright<.03)).sum()),
        raw_alpha_ge09_dc_brightness_lt003=int(((alpha>=.9)&(bright<.03)).sum()),
        max_sigma_over_extent_gt001=int((sigma/extent>.01).sum()),max_sigma_over_extent_gt01=int((sigma/extent>.1).sum()),
        near_black_large_scale_raw_alpha_ge05=int(((alpha>=.5)&(bright<.03)&(sigma/extent>.01)).sum()),
        numerical_notes='Saved params only; no historical densification gradient state inferred; extent normalizes own scene gauge.',
        processing_seconds=time.time()-started)
    if ident!='E6':
        eff=np.asarray(c[:,11],np.float64);filtering=np.asarray(c[:,10],np.float64)
        metrics.update(filtered_alpha=distribution(eff),filter_3D_over_extent=distribution(filtering/extent),
            filter_to_max_sigma=distribution(filtering/sigma),
            raw_ge005_filtered_lt005=int(((alpha>=.05)&(eff<.05)).sum()),
            filtered_alpha_lt005=int((eff<.05).sum()),
            filtered_alpha_ge05_dc_brightness_lt003=int(((eff>=.5)&(bright<.03)).sum()),
            filtered_alpha_ge09_dc_brightness_lt003=int(((eff>=.9)&(bright<.03)).sum()),
            gg_gs_filter_rule='effective_alpha=raw_alpha/prod(sqrt(sigma_i^2+filter_3D^2)/sigma_i); same official GaussianModel determinant correction')
    return metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiments',nargs='+',default=['E1','E2','E3','E4','E5','E6']);parser.add_argument('--chunk',type=int,default=200000)
    args=parser.parse_args();provenance=json.loads((AUDIT/'gaussian_analysis/experiment_provenance.json').read_text())
    rows=[r for r in provenance['experiments'] if r['id'] in args.experiments]
    out=AUDIT/'gaussian_analysis/final_gaussian_properties.json'
    previous=json.loads(out.read_text()) if out.exists() else dict(schema='gggs.final_gaussian_properties.v1',experiments=[],no_gpu=True,no_training=True)
    by_id={r['id']:r for r in previous['experiments']}
    for row in rows:
        result=analyze(row,args.chunk);by_id[row['id']]=result
        previous['experiments']=[by_id[k] for k in sorted(by_id)]
        out.write_text(json.dumps(previous,indent=2,allow_nan=False)+'\n')
        print(row['id'],'complete',json.dumps({k:result[k] for k in ['raw_alpha_below_prune_005','raw_alpha_ge05_dc_brightness_lt003','max_sigma_over_extent_gt001']}),flush=True)

if __name__=='__main__':main()
