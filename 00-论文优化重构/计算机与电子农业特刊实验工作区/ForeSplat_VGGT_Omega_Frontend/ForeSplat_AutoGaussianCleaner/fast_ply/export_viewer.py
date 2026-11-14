"""Optional streaming SuperSplat export; never called by fast_clean_gggs.py."""
import argparse
from pathlib import Path
import os
import numpy as np
from scipy.special import expit,logit
from .io import Meter,read_header,chunks
from .features import validate


def export(source,output,io_limit_mib=20):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or output.exists():raise FileExistsError('Choose a separate new output path')
    meter=Meter(io_limit_mib);partial=output.with_name(output.name+'.partial')
    output.parent.mkdir(parents=True,exist_ok=True)
    if partial.exists():raise FileExistsError(partial)
    try:
        with source.open('rb',buffering=0) as stream:
            header=read_header(stream,meter)
            if any(k.startswith(('sg_color_','sg_axis_','sg_sharpness_')) for k in header.dtype.names):
                raise ValueError('SG lobes have no standard SuperSplat equivalent')
            with partial.open('xb',buffering=0) as dest:
                meter.write(dest,header.output(header.count))
                for start,rows in chunks(stream,header,meter,16384):
                    validate(rows)
                    sigma=np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype('f8'))
                    effective=np.hypot(sigma,rows['filter_3D'].astype('f8')[:,None])
                    alpha=expit(rows['opacity'].astype('f8'))*(sigma/effective).prod(1)
                    for i in range(3):rows[f'scale_{i}']=np.log(effective[:,i])
                    rows['opacity']=logit(np.clip(alpha,np.finfo('f4').tiny,1-1e-7));rows['filter_3D']=0
                    meter.write(dest,rows)
                os.fsync(dest.fileno())
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True);raise
    return dict(read_bytes=meter.read_bytes,write_bytes=meter.write_bytes,throttle_wait_s=meter.wait_s)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--io-limit-mib',default=20,type=float)
    a=p.parse_args()
    if not np.isfinite(a.io_limit_mib) or a.io_limit_mib<0:raise ValueError('Invalid rate')
    print(export(a.input,a.output,a.io_limit_mib))

if __name__=='__main__':main()
