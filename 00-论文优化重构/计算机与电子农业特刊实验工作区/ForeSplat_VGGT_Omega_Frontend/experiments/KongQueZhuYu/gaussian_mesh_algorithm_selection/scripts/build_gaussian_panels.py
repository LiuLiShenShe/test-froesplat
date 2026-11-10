"""Actual frozen-view images and GT-only ROIs; equal camera/normal/depth display."""
import os
os.environ.update(MPLBACKEND='Agg',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';DUAL=OUT.parent/'geometry_rgb_dual_frontend';DEST=OUT/'comparison/visualizations/gaussian';DEST.mkdir(parents=True,exist_ok=True)
REPS=[0,26,53,79,105,131,157,183]
NORMAL_POLICY='unit_camera_undefined_visible_magenta_v1'
previous=json.loads((DEST/'PROTOCOL.json').read_text()) if (DEST/'PROTOCOL.json').is_file() else {}
refresh_normal=previous.get('normal_display_policy')!=NORMAL_POLICY
roots=[('E10',OLD/'surface_inputs/e10_gggs_surface','camera',False),('P2_Cleaned',OUT/'P2_GGGS_Cleaning/surface_inputs','camera',False),('E11',OLD/'surface_inputs/e11_2dgs_surface','world',False),('E12_raw',OUT/'E12_GGGS_EO/surface_inputs','camera',False),('E12_corrected',OUT/'E12_GGGS_EO/surface_inputs','camera',False),('E13_QGS',OUT/'E13_QGS/surface_inputs','camera',False),('E14_2D_30K',OUT/'E14_2D_SuGaR/stage_renders/final_30000','world',False),('E14_Refined',OUT/'E14_2D_SuGaR/stage_renders/refined','world',True)]
texture=json.loads((DUAL/'render_comparison/visualizations/GT_texture_crop_protocol.json').read_text());texture={r['source_id']:r['bounds_native'] for r in texture['records']}
edge=json.loads((OLD/'comparisons/visualizations/leaf_boundary_roi_protocol.json').read_text());edge={r['source_id']:r['bounds_native'] for r in edge['records']}
def im(arr):return Image.fromarray((np.nan_to_num(np.clip(arr,0,1))*255).round().astype('uint8'))
def panel(items,path,size):
    canvas=Image.new('RGB',(size[0]*len(items),size[1]+25),'#202020');draw=ImageDraw.Draw(canvas)
    for i,(name,arr) in enumerate(items):
        canvas.paste(im(arr).resize(size,Image.Resampling.LANCZOS),(size[0]*i,25));draw.text((size[0]*i+4,4),name,fill='white')
    canvas.save(path)
records=[];available=[]
for name,root,ncoord,aux in roots:
    if (root/'render_complete.json').is_file() and (root/'render_index.json').is_file():available.append((name,root,ncoord,aux,{r['source_id']:r for r in json.loads((root/'render_index.json').read_text())}))
for sid in REPS:
    entries=[];truth=None;fg=None;roi=None
    for name,root,ncoord,aux,index in available:
        r=index[sid]
        with np.load(r['path']) as z:
            rgb=z['render_corrected'] if name=='E12_corrected' else z['render_raw'];alpha=z['alpha'];depth=z['depth_expected_center_aux'] if aux else z['depth_surface'];normal=z['normal_smallest_axis_aux'] if aux else z['native_normal'];gt=z['gt_rgb'];mask=z['foreground_mask']
        if truth is None:
            truth=gt;fg=mask;yy,xx=np.nonzero(fg);x1,x2=xx.min(),xx.max();y1,y2=yy.min(),yy.max();cx=(x1+x2)//2
            # GT/mask-only common interior and pot windows; preserve old texture/edge ROIs.
            center_y=int(y1+.4*(y2-y1));pot_y=int(y1+.80*(y2-y1))
            roi={'leaf_texture':texture[sid],'leaf_tip_edge':edge[sid],'leaf_center_occlusion':[max(0,cx-160),max(0,center_y-160),min(1080,cx+160),min(1920,center_y+160)],'pot_rim':[max(0,cx-160),max(0,pot_y-160),min(1080,cx+160),min(1920,pot_y+160)]}
        assert np.array_equal(gt,truth) and np.array_equal(mask,fg)
        E=np.asarray(r['camera']['world_view_transform']).T
        nc=normal if ncoord=='camera' else normal@E[:3,:3].T
        nl=np.linalg.norm(nc,axis=-1,keepdims=True)
        undefined=(~np.isfinite(nc).all(-1))|(nl[...,0]<1e-8)
        nc/=np.maximum(nl,1e-8)
        normalrgb=(nc+1)/2;normalrgb[alpha<.01]=0
        normalrgb[undefined&(alpha>=.01)]=[1,0,1]
        # Shared world-gauge display range, no per-model min/max normalization.
        depthrgb=colormaps['viridis'](np.clip((depth-.15)/(1.57-.15),0,1))[...,:3];depthrgb[(alpha<.01)|~np.isfinite(depth)|(depth<=0)]=0
        arrays={'RGB':rgb,'Alpha':np.repeat(alpha[...,None],3,-1),'Depth_aux' if aux else 'Depth':depthrgb,'Normal_axis_aux' if aux else 'Normal_camera':normalrgb}
        folder=DEST/name/f'{sid:04d}';folder.mkdir(parents=True,exist_ok=True)
        for key,arr in arrays.items():
            if not (folder/f'{key}.png').exists() or (refresh_normal and key.startswith('Normal')):im(arr).save(folder/f'{key}.png')
            for kind,(x1,y1,x2,y2) in roi.items():
                if not (folder/f'{kind}_{key}.png').exists() or (refresh_normal and key.startswith('Normal')):im(arr[y1:y2,x1:x2]).save(folder/f'{kind}_{key}.png')
        entries.append((name,arrays))
        records.append(dict(method=name,source_id=sid,source_npz=str(Path(r['path']).resolve()),output=str(folder.resolve()),normal_source=ncoord,auxiliary_depth_normal=aux,visible_undefined_normal_pixels=int((undefined&(alpha>=.01)).sum()),GT_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),ROIs=roi))
    if entries:
        for key in ['RGB','Alpha','Depth','Normal_camera']:
            items=([('GT_03SAM_train',truth)] if key=='RGB' else [])+[(name+('_AUX_centerZ' if key=='Depth' and 'Depth_aux' in arrays else '_AUX_axis' if key=='Normal_camera' and 'Normal_axis_aux' in arrays else ''),arrays[key if key in arrays else ('Depth_aux' if key=='Depth' else 'Normal_axis_aux')]) for name,arrays in entries]
            panel(items,DEST/f'source_{sid:04d}_{key}.png',(270,480))
            for kind,(x1,y1,x2,y2) in roi.items():panel([(name,arr[y1:y2,x1:x2]) for name,arr in items],DEST/f'source_{sid:04d}_{kind}_{key}.png',(320,320))
(DEST/'PROTOCOL.json').write_text(json.dumps(dict(status='active_actual_panels',source_ids=REPS,records=records,normal_display_policy=NORMAL_POLICY,normal_display='Unit camera-space normals; all world normals rotated by perview exact W2C; visible undefined/zero/nonfinite normals are magenta and counted; refined auxiliary axis moments separately labelled',depth_display='Same .15–1.57 uncalibrated VGGT gauge; native conditional depth types retained, cannot compare auxiliary3D moment as native2Dsurface depth',RGB='Actual native unbounded floats clipped only for display; raw/corrected EO separate',ROI='Historical GT-only texture and boundary windows; center/pot from same original RAP bbox, not optimized using model results',display_resize='Only overview contact sheets LANCZOS downsample; individual fullviews/crops saved at original pixels',ground_truth='No scan accuracy claim'),ensure_ascii=False,indent=2,default=int))
print('Actual Gaussian panels',len(records),flush=True)
