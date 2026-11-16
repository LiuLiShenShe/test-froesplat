"""Adapt native three-axis schemas to the existing frozen C1_raw rule only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import hashlib
import io
import json
import sys
import time
from runtime import OUT, CLEANER, Meter, write
import numpy as np
from plyfile import PlyData,PlyElement

parser=argparse.ArgumentParser();parser.add_argument('method',choices=['3DGS','GGGS','gsplat','MCMC_3DGS'])
args=parser.parse_args();output=OUT/'cleaning'/args.method;output.mkdir(parents=True,exist_ok=True)
source=json.loads((OUT/'gaussian'/args.method/'complete.json').read_text())['native_ply']
sys.path.insert(0,str(CLEANER.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features,rules
meter=Meter();wall=time.perf_counter();data=PlyData.read(io.BytesIO(meter.read(source)),mmap=False);rows=data['vertex'].data
assert all(f'scale_{i}' in rows.dtype.names for i in range(3))
adapted=rows
if 'filter_3D' not in rows.dtype.names:
    # The other native representations have no GGGS low-pass filter. A zero
    # filter is only an in-memory feature adapter; native exported fields stay
    # byte-identical and the raw C1 rule uses their original scales/opacities.
    adapted=np.empty(len(rows),dtype=rows.dtype.descr+[('filter_3D','<f4')])
    for key in rows.dtype.names:adapted[key]=rows[key]
    adapted['filter_3D']=0
feature=features(adapted,knn=False)
removed,bits,params=rules(feature,version=2,normalized=False)
retained=rows[~removed]
assert 0<len(retained)<=len(rows)
buffer=io.BytesIO();PlyData([PlyElement.describe(retained,'vertex')],text=False,byte_order='<').write(buffer)
meter.save(output/'cleaned_gaussians.ply',buffer.getbuffer())
buffer=io.BytesIO();np.save(buffer,np.flatnonzero(~removed));meter.save(output/'retained_indices.npy',buffer.getbuffer())
write(output/'complete.json',dict(status='complete',method=args.method,input_ply=source,input_gaussians=len(rows),
    retained_gaussians=len(retained),removed_gaussians=int(removed.sum()),parameters=params,
    algorithm='Unchanged existing legacy.features(knn=False), legacy.rules(version=2,normalized=False)',
    rule_sha256=hashlib.sha256((CLEANER/'cleaner/adaptive/legacy.py').read_bytes()).hexdigest(),
    native_fields=list(rows.dtype.names),native_fields_preserved=True,zero_filter_adapter=args.method!='GGGS',
    note='Original native model retained. This candidate must pass same-camera damage review before mesh selection.',
    wall_s=time.perf_counter()-wall,io=meter.stats()))
