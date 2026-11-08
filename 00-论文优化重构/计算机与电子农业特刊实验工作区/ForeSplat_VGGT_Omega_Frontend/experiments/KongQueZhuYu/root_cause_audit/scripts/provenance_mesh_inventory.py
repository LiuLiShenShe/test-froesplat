"""Locate actual E5/E6 surfaces, read headers only, never extract a new mesh."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
STD=ROOT.parents[2].parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'

def main():
    rows=[]
    for name in ['E5_Standard_VGGT_GGGS_Baseline','E6_2DGS_Standard_VGGT']:
        surfaces=[];gaussian_or_pointcloud=[]
        for path in sorted((STD/name).rglob('*.ply')):
            header=[]
            with path.open('rb') as f:
                while True:
                    line=f.readline();header.append(line)
                    if not line or line.strip()==b'end_header':break
                    if sum(map(len,header))>65536:raise ValueError('Unexpected PLY header')
            fields={}
            for line in header:
                parts=line.split()
                if parts[:1]==[b'element']:fields[parts[1].decode()]=int(parts[2])
            item=dict(path=str(path),file_bytes=path.stat().st_size,elements=fields,
                      header_sha256=hashlib.sha256(b''.join(header)).hexdigest(),payload_read=False)
            if fields.get('face',0)>0:surfaces.append(item)
            else:gaussian_or_pointcloud.append(str(path))
        rows.append(dict(experiment=name,surface_meshes=surfaces,non_surface_PLY_files=gaussian_or_pointcloud,
                         status='existing_surface_located' if surfaces else 'no_existing_surface_mesh_found'))
    old=STD/'E6_2DGS_Standard_VGGT/checkpoints/mesh_metrics.json'
    result=dict(inventory=rows,existing_E6_mesh_metrics=json.loads(old.read_text()),
                scope='Read-only location and original headers. Vertex/face counts are not geometry accuracy, scan registration or measured metric scale.',
                new_meshing=False,Gaussian_training=False)
    out=ROOT/'gaussian_analysis/existing_mesh_inventory.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(experiment=r['experiment'],status=r['status'],meshes=[p['elements'] for p in r['surface_meshes']]) for r in rows]))

if __name__=='__main__':main()
