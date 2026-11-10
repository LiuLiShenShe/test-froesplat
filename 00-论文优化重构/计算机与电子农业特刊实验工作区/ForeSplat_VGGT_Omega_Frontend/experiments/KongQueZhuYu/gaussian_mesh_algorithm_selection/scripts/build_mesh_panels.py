"""Actual gray, normal, color, wireframe and identical crop contact sheets."""
import json
import csv
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/visualizations/mesh_comparison'
DEST.mkdir(exist_ok=True)
REPS=[0,26,53,79,105,131,157,183]

def main():
    meshrows=list(csv.DictReader((OUT/'comparison/ALL_MESH_EXPERIMENTS.csv').open()))
    selected={r['post_mesh_path']:r['mesh_id'] for r in meshrows if r['mesh_stage']=='post' and r['post_mesh_path']}
    for r in meshrows:
        if r['mesh_stage']=='post' and r.get('textured_mesh_path'):
            selected[r['textured_mesh_path']]=r['mesh_id']+'_native_textured'
    verified_color={}
    for folder in [OUT/'P1_GGGS_Tetrahedra',OUT/'P2_GGGS_Cleaning/Tetrahedra',OUT/'E12_GGGS_EO/Tetrahedra']:
        marker=folder/'recon_post_vertex_color.colour.json'
        if not marker.is_file():continue
        color=json.loads(marker.read_text())
        if color.get('status')=='complete' and color.get('geometry_unchanged'):
            assert color['geometry_sha256_before']==color['geometry_sha256_after']
            assert Path(color['output']).is_file() and Path(color['source_mesh']).is_file()
            verified_color[color['output']]=color
    chosen={}
    for marker in (OUT/'comparison/visualizations/mesh').glob('*/complete.json'):
        info=json.loads(marker.read_text())
        if info.get('status')!='complete':continue
        if info['normal_space']!='camera' or info['background']!=.25:continue
        actual=info['source_mesh'];canonical=actual;priority=1
        if actual in verified_color:
            canonical=verified_color[actual]['source_mesh'];priority=2
        if canonical not in selected:continue
        if canonical not in chosen or chosen[canonical][0]<priority:
            chosen[canonical]=(priority,(selected[canonical],marker.parent,actual))
    bundles=[entry[1] for entry in chosen.values()]
    bundles.sort()
    assert bundles,'No completed actual mesh view bundles'
    truth={r['source_id']:r for r in json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())}
    edge={r['source_id']:r['bounds_native'] for r in json.loads((OLD/'comparisons/visualizations/leaf_boundary_roi_protocol.json').read_text())['records']}
    protocol=[]
    for sid in REPS:
        with np.load(truth[sid]['path']) as z:fg=z['foreground_mask'];gt=z['gt_rgb']
        missing=np.zeros(fg.shape,bool)
        for name,folder,path in bundles:
            with np.load(folder/f'fixed_{sid:04d}/geometry_channels.npz') as z:pred=z['silhouette']
            labels,count=ndimage.label(fg&~pred)
            rim=fg&~ndimage.binary_erosion(fg,iterations=3)
            touch=np.unique(labels[rim]);sizes=np.bincount(labels.ravel())
            keep=np.flatnonzero(sizes>=9);keep=keep[(keep>0)&~np.isin(keep,touch)]
            missing|=np.isin(labels,keep)
        labels,count=ndimage.label(missing);sizes=np.bincount(labels.ravel());sizes[0]=0
        if count:
            yy,xx=np.nonzero(labels==sizes.argmax());cx=int(np.mean(xx));cy=int(np.mean(yy));reason='Largest connected union of internal missing projections across all available post routes; same ROI for every method'
        else:
            bounds=edge[sid];cx=(bounds[0]+bounds[2])//2;cy=(bounds[1]+bounds[3])//2;reason='No internal missing-projection union; GT-only edge ROI fallback, not a claimed hole'
        x1=max(0,min(fg.shape[1]-320,cx-160));y1=max(0,min(fg.shape[0]-320,cy-160))
        rois={'internal_missing_union':[x1,y1,x1+320,y1+320],'leaf_edge':edge[sid]}
        for label in ['gray','normal_camera','wireframe','vertex_color','textured']:
            images=[]
            for name,folder,path in bundles:
                file=folder/f'fixed_{sid:04d}'/(label+'.png')
                if file.exists():images.append((name,Image.open(file).convert('RGB'),path))
            if not images:continue
            for region,bounds in [('full',None),*rois.items()]:
                width,height=(220,392) if bounds is None else (320,320)
                canvas=Image.new('RGB',(width*len(images),height+28),'#202020');draw=ImageDraw.Draw(canvas)
                for col,(name,im,path) in enumerate(images):
                    crop=im if bounds is None else im.crop(bounds)
                    canvas.paste(crop.resize((width,height),Image.Resampling.LANCZOS),(col*width,28));draw.text((col*width+4,5),name,fill='white')
                    if bounds is not None:
                        target=DEST/name/f'{sid:04d}';target.mkdir(parents=True,exist_ok=True);crop.save(target/f'{region}_{label}.png')
                canvas.save(DEST/f'{sid:04d}_{region}_{label}.png')
        protocol.append(dict(source_id=sid,ROIs=rois,internal_missing_selection=reason))
    (DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='actual_available_mesh_panels',views=REPS,
        methods=[r[0] for r in bundles],sources=[r[2] for r in bundles],records=protocol,
        tetra_vertex_color_provenance=list(verified_color.values()),
        color='Only actual vertex colors, not UV texture substitutes. Native textured OBJ full views separately preserved.',
        wireframe='Actual projected frontmost triangle edges; not image edge detection',
        geometric_comparison='Same native1080x1920 pixel regions, camera-space normals, two-sided headlight, background0.25. Omega retains its own W2C for identical source RGB.',
        hole_policy='Inside RAP missing projections excluding its exterior3px boundary and true mask holes. Union ROI can depend on available methods and is recalculated when more methods complete; not a physical3D-hole label.'),indent=2))
    print('Actual equal-window Mesh panels',len(bundles),flush=True)

if __name__=='__main__':main()
