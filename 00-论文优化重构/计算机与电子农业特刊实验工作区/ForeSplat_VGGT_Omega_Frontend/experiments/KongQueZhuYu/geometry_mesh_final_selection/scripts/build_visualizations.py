"""Scientific comparisons from real arrays, bounded montages for visual review."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from surface_protocol import ROOT,DUAL,write

REPS=[0,26,53,79,105,131,157,183]
OUT=ROOT/'comparisons/visualizations'
def uint8(rgb):return (np.clip(rgb,0,1)*255).round().astype(np.uint8)
def tile(array,label,size=None):
    im=Image.fromarray(array)
    if size:im=im.resize(size,Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(im.width,im.height+24),'white');canvas.paste(im,(0,24));ImageDraw.Draw(canvas).text((5,5),label,fill='black');return canvas
def montage(rows,path):
    w=sum(t.width for t in rows[0]);h=sum(max(t.height for t in row) for row in rows);canvas=Image.new('RGB',(w,h),'white');y=0
    for row in rows:
        x=0
        for t in row:canvas.paste(t,(x,y));x+=t.width
        y+=max(t.height for t in row)
    assert w*h<=1200000,(w,h,path)
    canvas.save(path)
def get(exp,sid):
    prefix=f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface'
    return dict(np.load(ROOT/'surface_inputs'/prefix/'rendered'/f'crop_{sid:04d}.npz'))
def local(rgb,box,side=192):
    x0,y0,x1,y1=box;cx=(x0+x1)//2;cy=(y0+y1)//2
    return rgb[cy-side//2:cy+side//2,cx-side//2:cx+side//2]
def error_image(pred,gt):
    err=np.sqrt(np.mean((pred-gt)**2,axis=-1));colour=np.zeros((*err.shape,3));colour[...,0]=np.minimum(err/.3,1);return uint8(colour)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    roi=json.loads((DUAL/'render_comparison/visualizations/GT_texture_crop_protocol.json').read_text())['records'];roi={r['source_id']:r['bounds_native'] for r in roi}
    edges=json.loads((OUT/'leaf_boundary_roi_protocol.json').read_text())['records'];edges={r['source_id']:r['bounds_native'] for r in edges}
    photos={sid:{exp:get(exp,sid) for exp in ['E8','E10','E11']} for sid in REPS}
    # Fix the depth colour origin using original frontend observations, before
    # Gaussian/mesh optimization. Never re-centre each competing output.
    depth_centres={}
    for sid in REPS:
        depth_centres[sid]={}
        for name,folder in [('omega','VGGT_Omega_Original_RGB'),('standard','Standard_VGGT_Original_RGB')]:
            base=DUAL/folder/'per_frame';depth=np.load(base/f'depth_crop_{sid:04d}.npy');valid=np.load(base/f'valid_foreground_crop_{sid:04d}.npy')
            depth_centres[sid][name]=float(np.median(depth[valid&np.isfinite(depth)&(depth>0)]))
    for group in range(2):
        sids=REPS[group*4:group*4+4];overview=[];leaf=[];edge=[];errors=[];alpha=[];photo_depth=[];photo_normal=[]
        for sid in sids:
            gt=photos[sid]['E10']['gt_rgb'];overview.append([tile(uint8(gt),f'{sid}: GT',(180,320))]+[tile(uint8(photos[sid][e]['render_raw']),e,(180,320)) for e in ['E8','E10','E11']])
            leaf.append([tile(uint8(local(gt,roi[sid])),f'{sid}: GT')]+[tile(uint8(local(photos[sid][e]['render_raw'],roi[sid])),e) for e in ['E8','E10','E11']])
            edge.append([tile(uint8(local(gt,edges[sid])),f'{sid}: GT')]+[tile(uint8(local(photos[sid][e]['render_raw'],edges[sid])),e) for e in ['E8','E10','E11']])
            errors.append([tile(uint8(local(gt,roi[sid])),f'{sid}: GT')]+[tile(error_image(local(photos[sid][e]['render_raw'].clip(0,1),roi[sid]),local(gt,roi[sid])),e) for e in ['E8','E10','E11']])
            alpha.append([tile(np.repeat(photos[sid]['E10']['foreground_mask'][...,None],3,axis=2).astype(np.uint8)*255,f'{sid}: RAP',(180,320))]+[tile(uint8(np.repeat(photos[sid][e]['alpha'][...,None],3,axis=2)),e,(180,320)) for e in ['E8','E10','E11']])
            dr=[tile(uint8(gt),f'{sid}: GT',(180,320))];nr=[tile(uint8(gt),f'{sid}: GT',(180,320))]
            for exp in ['E8','E10','E11']:
                data=photos[sid][exp];depth=data['depth_surface'];valid=np.isfinite(depth)&(depth>0);fg=data['foreground_mask'];name='omega' if exp=='E8' else 'standard'
                scale=json.loads((ROOT/'configs'/f'{name}_spatial_scale.json').read_text());centre=depth_centres[sid][name]
                z=np.clip((depth-centre)/scale['diagonal']+.5,0,1);colour=np.stack([z,1-abs(2*z-1),1-z],-1);colour[~valid]=0;dr.append(tile(uint8(colour),exp,(180,320)))
                normal=data['native_normal'].copy()
                if exp=='E11':
                    cams=json.loads((ROOT/'surface_inputs/e11_2dgs_surface/camera_runtime.json').read_text())['records'];c=next(c for c in cams if c['source_id']==sid)
                    normal=normal@np.asarray(c['world_view_transform']).T[:3,:3].T
                length=np.linalg.norm(normal,axis=-1);normal/=np.maximum(length[...,None],1e-8);display=(normal+1)/2;display[~valid|(length<1e-8)]=0
                nr.append(tile(uint8(display),exp,(180,320)))
            photo_depth.append(dr);photo_normal.append(nr)
        for name,rows in [('photo_whole',overview),('photo_leaf',leaf),('photo_edge',edge),('photo_error',errors),('photo_alpha',alpha),('photo_depth',photo_depth),('photo_normal',photo_normal)]:montage(rows,OUT/f'{name}_{group+1}.png')
    methods=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF']
    for stage in ['raw','post']:
        for group in range(4):
            rows=[];whole=[];depths=[];normals=[];silhouette=[];edge_rgb=[];edge_depth=[];edge_normal=[];edge_silhouette=[]
            for sid in REPS[group*2:group*2+2]:
                gt=photos[sid]['E10']['gt_rgb'];row=[tile(uint8(local(gt,roi[sid])),f'{sid}: GT')];wr=[tile(uint8(gt),f'{sid}: GT',(180,320))];dr=[tile(uint8(gt),f'{sid}: GT',(180,320))];nr=[tile(uint8(gt),f'{sid}: GT',(180,320))]
                sr=[tile(np.repeat(photos[sid]['E10']['foreground_mask'][...,None],3,axis=2).astype(np.uint8)*255,f'{sid}: RAP',(180,320))]
                er=[tile(uint8(local(gt,edges[sid])),f'{sid}: GT')];ed=[tile(uint8(local(gt,edges[sid])),f'{sid}: GT RGB')];en=[tile(uint8(local(gt,edges[sid])),f'{sid}: GT RGB')]
                es=[tile(np.repeat(local(photos[sid]['E10']['foreground_mask'],edges[sid])[...,None],3,axis=2).astype(np.uint8)*255,f'{sid}: RAP')]
                for method in methods:
                    folder=ROOT/'meshes'/method;d=dict(np.load(folder/'evaluation/views'/f'{stage}_{sid:04d}.npz'));rgb=d['mesh_rgb'];label=method.split('_')[0]
                    row.append(tile(uint8(local(rgb,roi[sid])),label));wr.append(tile(uint8(rgb),label,(180,320)))
                    depth=d['mesh_depth'];valid=depth>0;scale=json.loads((folder/'complete.json').read_text())['scale'];fg=d['foreground_mask']
                    centre=depth_centres[sid]['omega' if method.startswith('M1') else 'standard'];z=np.clip((depth-centre)/scale['diagonal']+.5,0,1)
                    colour=np.stack([z,1-abs(2*z-1),1-z],-1);colour[~valid]=0;dr.append(tile(uint8(colour),label,(180,320)))
                    inp='omega_direct' if method.startswith('M1') else 'e11_2dgs_surface' if method.startswith('M5') else 'e10_gggs_surface'
                    cams=json.loads((ROOT/'surface_inputs'/inp/'camera_runtime.json').read_text())['records'];c=next(c for c in cams if int(c.get('source_id',c['name'].split('_')[-1]))==sid)
                    normal=d['mesh_normal_world']@np.asarray(c['world_view_transform']).T[:3,:3].T;normal=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8);n=(normal+1)/2;n[~valid]=0;nr.append(tile(uint8(n),label,(180,320)))
                    pred=d['mesh_silhouette'];display=np.zeros((*fg.shape,3),dtype=np.uint8);display[pred&fg]=[210,210,210];display[fg&~pred]=[255,0,0];display[pred&~fg]=[0,90,255];sr.append(tile(display,label,(180,320)))
                    er.append(tile(uint8(local(rgb,edges[sid])),label));ed.append(tile(uint8(local(colour,edges[sid])),label));en.append(tile(uint8(local(n,edges[sid])),label));es.append(tile(local(display,edges[sid]),label))
                rows.append(row);whole.append(wr);depths.append(dr);normals.append(nr);silhouette.append(sr)
                edge_rgb.append(er);edge_depth.append(ed);edge_normal.append(en);edge_silhouette.append(es)
            for name,data in [('mesh_leaf',rows),('mesh_whole',whole),('mesh_depth',depths),('mesh_normal',normals),('mesh_silhouette',silhouette),('mesh_edge',edge_rgb),('mesh_edge_depth',edge_depth),('mesh_edge_normal',edge_normal),('mesh_edge_silhouette',edge_silhouette)]:montage(data,OUT/f'{name}_{stage}_{group+1}.png')
    # Same frontend direct/Gaussian control and the 2DGS/NKSR supplementary control.
    for stage in ['raw','post']:
        for group in range(4):
            sids=REPS[group*2:group*2+2];rows=[]
            for sid in sids:
                row=[tile(uint8(local(photos[sid]['E10']['gt_rgb'],roi[sid])),f'{sid}: GT')]
                for method in ['M1_Omega_NKSR','E8_GGGS_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR'] + (['E11_2DGS_NKSR'] if (ROOT/'meshes/supplementary/E11_2DGS_NKSR/complete.json').exists() else []):
                    folder=ROOT/'meshes'/method if method.startswith('M') else ROOT/'meshes/supplementary'/method
                    with np.load(folder/'evaluation/views'/f'{stage}_{sid:04d}.npz') as data:rgb=data['mesh_rgb']
                    row.append(tile(uint8(local(rgb,roi[sid])),method.replace('_NKSR','')))
                rows.append(row)
            montage(rows,OUT/f'mesh_same_frontend_{stage}_{group+1}.png')
    write(OUT/'protocol.json',dict(source_ids=REPS,photo_columns=['GT','E8','E10','E11'],mesh_columns=['GT',*methods],
        maximum_review_image_pixels=1200000,local_pixels='Leaf texture: centre192 of fixed legacy GT-only320 windows; outer edges: native192 from recorded common GT/RAP boundary protocol. No resize',overview='180x320 nearest only for display, all metrics native1080x1920',
        mesh_depth='Native camera-axis depth, common centre fixed by original frontend valid-foreground depth median for each frame; common frontend diagonal and display[-.5,.5], no per-output re-centring',
        depth_display_centres=depth_centres,
        mesh_normal='World normals transformed to each own camera coordinates, same RGB convention',raw_and_post=True,original_arrays_preserved=True))
    print('Real native RGB/leaf/edge/error/Alpha and raw/post Mesh RGB/depth/normal/silhouette bounded montages saved')

if __name__=='__main__':main()
