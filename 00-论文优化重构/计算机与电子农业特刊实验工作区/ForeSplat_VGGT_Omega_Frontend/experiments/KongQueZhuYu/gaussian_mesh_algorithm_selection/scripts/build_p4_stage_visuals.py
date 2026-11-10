"""Display actual native E11 and P4 raw/post meshes in the same GT-only ROI."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

OUT=Path(__file__).resolve().parents[1]
DEST=OUT/'comparison/visualizations/P4_stage_visuals'
DEST.mkdir(exist_ok=True)
gaussian=json.loads((OUT/'comparison/visualizations/gaussian/PROTOCOL.json').read_text())
refs={r['source_id']:r for r in gaussian['records'] if r['method']=='E11'}
meshes={}
for marker in (OUT/'comparison/visualizations/mesh').glob('*/complete.json'):
    item=json.loads(marker.read_text())
    if item.get('status')=='complete' and item.get('normal_space')=='camera':
        meshes[item['source_mesh']]=marker.parent
records=[]
for sid in [0,26,53,79,105,131,157,183]:
    ref=refs[sid]
    with np.load(ref['source_npz']) as z:
        gt=Image.fromarray((np.clip(z['gt_rgb'],0,1)*255).astype(np.uint8))
    columns=[('GT',None),('E11 native2DGS',None)]
    for depth in ['expected','median']:
        for stage in ['raw','post']:
            actual=(OUT/f'P4_2DGS_Depth_Mesh/{depth}_mask/{stage}.ply').resolve(strict=True)
            columns.append((f'{depth} {stage}',meshes[str(actual)]/f'fixed_{sid:04d}'))
    for region in ['leaf_texture','leaf_tip_edge','leaf_center_occlusion']:
        bounds=ref['ROIs'][region];size=(bounds[2]-bounds[0],bounds[3]-bounds[1])
        canvas=Image.new('RGB',(size[0]*len(columns),size[1]*3+90),'#202020')
        draw=ImageDraw.Draw(canvas)
        for row,channel in enumerate(['RGB','Normal_camera','Depth']):
            for col,(label,folder) in enumerate(columns):
                y=row*(size[1]+30)
                if col==0:im=gt
                elif col==1:im=Image.open(Path(ref['output'])/(channel+'.png')).convert('RGB')
                else:
                    mesh_channel={'RGB':'gray','Normal_camera':'normal_camera','Depth':'vertex_color'}[channel]
                    im=Image.open(folder/(mesh_channel+'.png')).convert('RGB')
                note=channel if col<2 else {'RGB':'gray geometry','Normal_camera':'camera normal','Depth':'native vertex color'}[channel]
                draw.text((col*size[0]+3,y+4),label+' / '+note,fill='white')
                canvas.paste(im.crop(bounds),(col*size[0],y+30))
        output=DEST/f'{sid:04d}_{region}.png';canvas.save(output)
        records.append(dict(source_id=sid,ROI=region,bounds_native=bounds,source_gaussian_npz=ref['source_npz'],actual_mesh_folders=[str(c[1]) for c in columns if c[1]],output=str(output.resolve())))
(DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='complete',records=records,
    purpose='Separate pre-existing native2DGS rendering/depth artifacts, TSDF raw geometry, and post-removal effects. Actual surfaces only; no new model or image generation.',
    display='Six identical native-pixel GT-only regions per row. First two columns RGB/normal/depth; mesh columns gray/camera-normal/vertex-color. Vertex colors are not UV textures.',
    caveat='A visually black nativeRGB region is not automatically a missing surface. Gray and native depth must be checked independently; positive depth can still be wrong.'),indent=2)+'\n')
print('Actual E11/P4 identical-window stage visuals',len(records),flush=True)
