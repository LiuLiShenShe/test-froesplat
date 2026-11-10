"""Read-only chunked validation of the actually saved failed quadratic PLY."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    dest=ROOT/'failure_exports/20261109T084120Z_1535250'
    path=dest/'native_failed_state.ply';props=[];header=b'';count=None
    with path.open('rb') as stream:
        for _ in range(300):
            raw=stream.readline();header+=raw;tokens=raw.decode('ascii').strip().split()
            if tokens[:2]==['format','binary_little_endian']:pass
            elif tokens[:2]==['element','vertex']:count=int(tokens[2])
            elif tokens[:1]==['property']:
                assert tokens[1]=='float',tokens
                props.append(tokens[2])
            elif tokens==['end_header']:break
        else:raise AssertionError('Missing native header end')
    assert count==17228608 and all(f'scale_{i}' in props for i in range(6))
    assert len(header)+count*len(props)*4==path.stat().st_size
    initial=path.stat();data=np.memmap(path,dtype='<f4',mode='r',offset=len(header),shape=(count,len(props)))
    digest=hashlib.sha256(header);counts=np.zeros(len(props),dtype=np.int64);bad_rows=[]
    scales=[props.index(f'scale_{i}') for i in range(6)]
    decoded_bad=raw_scale_bad=0;started=time.time()
    for start in range(0,count,65536):
        block=np.asarray(data[start:start+65536]);digest.update(block.tobytes())
        finite=np.isfinite(block);counts+=(~finite).sum(axis=0)
        bad_rows.extend((np.flatnonzero(~finite.all(axis=1))+start).tolist())
        raw_scales=block[:,scales]
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            decoded=np.exp(raw_scales[:,:3])*np.tanh(raw_scales[:,3:])
        decoded_bad+=int((~np.isfinite(decoded).all(axis=1)).sum())
        raw_scale_bad+=int((~np.isfinite(raw_scales).all(axis=1)).sum())
    assert (initial.st_size,initial.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Original PLY changed during audit'
    np.save(dest/'native_ply_nonfinite_primitive_indices.npy',np.asarray(bad_rows,dtype=np.int64))
    result=dict(status='native_PLY_fully_read_with_nonfinite_parameters_retained' if bad_rows else 'native_PLY_fully_read_all_raw_parameters_finite',
        native_ply=str(path.resolve()),primitives=count,file_size_bytes=initial.st_size,SHA256=digest.hexdigest(),
        raw_nonfinite_primitives=len(bad_rows),raw_nonfinite_fraction=len(bad_rows)/count,
        nonfinite_values_by_native_property={name:int(n) for name,n in zip(props,counts) if n},
        raw_nonfinite_scale_primitives=raw_scale_bad,decoded_nonfinite_signed_scale_primitives=decoded_bad,
        decoding='CPU float32 exp(scale_0..2)*tanh(scale_3..5); native quadratic signed scales, third is curvature; not bitwise CUDA validation',
        elapsed_seconds=time.time()-started,CPU_only=True,original_native_parameters_unchanged=True,
        companion_PTH='Unreadable original failed export; native PLY is a separate live-model export, not derived from a readable checkpoint',
        scope='Failed base-preset transient13301 state only; not formal30K performance or evidence about current Mip training')
    (dest/'PLY_NUMERICAL_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],count,'nonfinite primitives',len(bad_rows),flush=True)

if __name__=='__main__':main()
