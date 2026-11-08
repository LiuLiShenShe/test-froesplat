"""Supplement source-mask provenance without rereading all RGB payloads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from provenance_collect import dataset_headers

AUDIT=Path(__file__).resolve().parents[1]
def main():
    path=AUDIT/'gaussian_analysis/experiment_provenance.json';data=json.loads(path.read_text())
    data['datasets']=[dataset_headers(r['path']) for r in data['datasets']]
    omega=Path(data['datasets'][0]['path']);standard=Path(data['datasets'][1]['path'])
    rap=Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu');masks=sorted((omega/'masks').glob('*.png'))
    shared=sum(p.stat().st_ino==(standard/'masks'/p.name).stat().st_ino and p.stat().st_dev==(standard/'masks'/p.name).stat().st_dev for p in masks)
    samples=[]
    for i in (0,len(masks)//2,len(masks)-1):
        p=masks[i];original=rap/p.name.replace('crop_','mask_')
        with Image.open(p) as im:a=np.array(im.convert('L'))
        with Image.open(original) as im:b=np.array(im.convert('L'))
        samples.append(dict(name=p.name,source_rap=str(original),dataset_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            rap_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),pixels_equal=bool(np.array_equal(a,b)),binary_unique_values=np.unique(a).tolist()))
    data['dataset_mask_chain']=dict(all208_standard_masks_share_omega_inodes=shared==len(masks),shared_inode_count=shared,samples=samples,
        scope='All208 dataset mask headers and Ω↔Standard inodes checked; RAP source mask payload equality checked on fixed3 frames.')
    for row in data['experiments']:
        raw=Path(row['log_path']).read_bytes()
        row['log_sha256']=hashlib.sha256(raw).hexdigest()
        row['log_normalized_text_sha256']=hashlib.sha256(Path(row['log_path']).read_text(errors='replace').encode()).hexdigest()
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(data['dataset_mask_chain'],indent=2));print(data['datasets'])

if __name__=='__main__':main()
