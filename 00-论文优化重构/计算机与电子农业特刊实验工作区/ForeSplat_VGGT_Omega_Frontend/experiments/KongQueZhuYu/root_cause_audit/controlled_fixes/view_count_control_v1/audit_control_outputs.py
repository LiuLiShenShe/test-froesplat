"""CPU persisted-export audit; NPY headers/bytes and small camera values only."""
import csv
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def npy_header(path):
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Unsupported NPY header version')
        expected=f.tell()+math.prod(shape)*dtype.itemsize
    return shape,dtype,expected,path.stat().st_size

def audit(path):
    issues=[]
    for filename in ['status.json','metadata.json','run_parameters.json','manifest.csv','cameras.json','cameras.npz','pixel_transforms.json','RUN_REPORT.md']:
        if not (path/filename).is_file():issues.append('missing:'+filename)
    if issues:return dict(path=str(path),passed=False,issues=issues)
    status=json.loads((path/'status.json').read_text());metadata=json.loads((path/'metadata.json').read_text())
    params=json.loads((path/'run_parameters.json').read_text());camera_data=json.loads((path/'cameras.json').read_text())
    rows=list(csv.DictReader((path/'manifest.csv').open()));frames=camera_data['frames'];n=len(rows)
    if status['status']!='complete' or metadata['status']!='complete':issues.append('actual_status_not_complete')
    if not all(x==n for x in [status['input_views'],metadata['input_views'],params['input_views'],len(frames)]):issues.append('frame_count_mismatch')
    names=[r['image_name'] for r in rows]
    if names!=[r['image_name'] for r in frames] or names!=metadata['source_order']:issues.append('source_order_mismatch')
    if [int(r['source_index']) for r in rows]!=sorted(int(r['source_index']) for r in rows):issues.append('source_order_not_monotonic')
    if metadata['seed']!=0 or params['seed']!=0:issues.append('seed_not0')
    if metadata['CUDA_VISIBLE_DEVICES']!='1':issues.append('physical_GPU_constraint')
    if metadata['point_head_executed'] is not False or metadata['training'] is not False:issues.append('model_support_or_training_metadata')
    if len(metadata['checkpoint_sha256'])!=64 or not metadata['source_file_sha256']:issues.append('provenance_missing')
    if metadata['input_tensor_dtype']!='torch.float32' or metadata['saved_model_rgb_dtype']!='float32':issues.append('input_dtype_mismatch')
    npy_count=0
    for r in frames:
        stem=Path(r['image_name']).stem;h,w=r['model_height'],r['model_width']
        specs=[('depth',(h,w),'float32'),('confidence',(h,w),'float32'),('model_rgb_float32',(3,h,w),'float32'),('model_mask',(h,w),'uint8')]
        for prefix,shape,dtype in specs:
            p=path/'per_frame'/f'{prefix}_{stem}.npy'
            if not p.is_file():issues.append('missing:'+p.name);continue
            actual,dt,expected,size=npy_header(p);npy_count+=1
            if actual!=shape or dt!=np.dtype(dtype) or expected!=size:issues.append('header_or_payload_length:'+p.name)
    with np.load(path/'cameras.npz',allow_pickle=False) as arrays:
        e=arrays['extrinsic_model_w2c'];c=arrays['c2w'];k=arrays['intrinsic_model'];a=arrays['pixel_transform'];ac=arrays['pixel_transform_pixel_center']
        for key,value,expected in [('E',e,(n,4,4)),('C2W',c,(n,4,4)),('K',k,(n,3,3)),('A',a,(n,3,3)),('Acorr',ac,(n,3,3))]:
            if value.shape!=expected or not np.isfinite(value).all():issues.append('camera_shape_or_nonfinite:'+key)
        inv_error=float(np.max(np.abs(e@c-np.eye(4))))
        ortho=float(np.max(np.abs(e[:,:3,:3]@e[:,:3,:3].transpose(0,2,1)-np.eye(3))))
        if inv_error>1e-10:issues.append('camera_inverse_convention_failure')
        if ortho>1e-4:issues.append('camera_rotation_nonorthonormal')
        if not np.allclose(e,np.array([r['extrinsic_model_w2c'] for r in frames]),atol=1e-12,rtol=0):issues.append('NPZ_JSON_E_mismatch')
        if not np.allclose(k,np.array([r['intrinsic_model'] for r in frames]),atol=1e-12,rtol=0):issues.append('NPZ_JSON_K_mismatch')
    result=dict(path=str(path),passed=not issues,issues=issues,frames=n,npy_headers_checked=npy_count,camera_inverse_max_error=inv_error,
        rotation_orthonormal_max_error=ortho,
        evidence_scope='Actual complete file counts/NPY headers/dtypes/exact payload length; small camera values and source/provenance. No full depth/RGB payload checksum recomputation; actual-value geometric accuracy evaluated separately with fixed independent RGB matches.')
    write(path/'control_export_audit.json',result);print(path.name,result['passed'],n,npy_count,issues,flush=True);return result

def pair_inputs(first,second,kind):
    ma=json.loads((first/'metadata.json').read_text());mb=json.loads((second/'metadata.json').read_text())
    aa={r['image_name']:r for r in json.loads((first/'cameras.json').read_text())['frames']}
    bb={r['image_name']:r for r in json.loads((second/'cameras.json').read_text())['frames']}
    names=sorted(set(aa)&set(bb));fields=['pixel_transform','pixel_transform_pixel_center','mask_sha256_model','model_width','model_height']
    if kind=='view_count':fields+=['input_float32_sha256']
    mismatches={field:[name for name in names if aa[name][field]!=bb[name][field]] for field in fields}
    for field in ['seed','checkpoint_sha256','source_file_sha256','input_tensor_dtype','saved_model_rgb_dtype']:
        if ma[field]!=mb[field]:mismatches[field]=['metadata']
    return dict(first=str(first),second=str(second),factor=kind,shared_count=len(names),shared_sources=names,mismatches=mismatches,passed=not any(mismatches.values()),
        scope='View count:38 same evaluated sources have exact saved actual input/Mask SHA and same geometry. Background:24 same sources, Mask/A/dtype/weights/seed unchanged; source-space RGB pixel difference assertions independently in prepared_sources.json.')

def main():
    count_paths=[HERE/name for name in ['Omega_70','Omega_110','Standard_70','Standard_110']]
    bg=ROOT/'background_control_v1';bg_paths=[bg/name for name in ['Omega_RAP_baseline_24','Omega_background_restored_24','Standard_RAP_baseline_24','Standard_background_restored_24']]
    results=[audit(p) for p in count_paths+bg_paths]
    pairs=[pair_inputs(count_paths[0],count_paths[1],'view_count'),pair_inputs(count_paths[2],count_paths[3],'view_count'),
           pair_inputs(bg_paths[0],bg_paths[1],'background'),pair_inputs(bg_paths[2],bg_paths[3],'background')]
    write(HERE/'all_control_export_audit.json',dict(status='passed' if all(r['passed'] for r in results+pairs) else 'failed',exports=results,pair_input_invariants=pairs,
        historical208_rerun=False,training_executed=False,GPU_used_by_auditor=False))
    print('All8 complete export+pairedinput audit',all(r['passed'] for r in results+pairs),flush=True)

if __name__=='__main__':main()
