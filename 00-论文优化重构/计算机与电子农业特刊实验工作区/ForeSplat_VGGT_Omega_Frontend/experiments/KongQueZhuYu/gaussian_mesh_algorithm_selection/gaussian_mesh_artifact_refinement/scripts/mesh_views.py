from common import *
import open3d as o3d
from PIL import Image,ImageDraw
from evaluate_meshes import cast
from scipy import ndimage
def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors);comp=np.asarray(mesh.cluster_connected_triangles()[0]);scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());reps=[0,26,53,79,105,131,157,183];regions=json.loads((SEL/'comparison/visualizations/gaussian/PROTOCOL.json').read_text())['records'];records=[]
    def png(p,img):Image.fromarray((img.clip(0,1)*255).round().astype(np.uint8)).save(p,compress_level=1)
    for item in index:
        sid=item['source_id']
        if sid not in reps:continue
        c=item['camera'];depth,norm,rgb=cast(scene,c,v,f,col);pred=np.isfinite(depth)&(depth>0);E=np.asarray(c['world_view_transform']).T;C=np.linalg.inv(E);nc=norm@E[:3,:3].T
        yy,xx=np.mgrid[:c['height'],:c['width']];dirs=np.stack([(xx-c['Cx'])/c['Fx'],(yy-c['Cy'])/c['Fy'],np.ones(xx.shape)],-1)@C[:3,:3].T;rays=np.concatenate([np.broadcast_to(C[:3,3],dirs.shape),dirs],-1).astype(np.float32);hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8);tid=hit['primitive_ids'].numpy();uv=hit['primitive_uvs'].numpy();valid=tid<len(f)
        gray=np.repeat((.2+.8*np.abs(nc[:,:,2]))[...,None],3,2);gray[~pred]=0;normal=nc*.5+.5;normal[~pred]=0
        wire=gray.copy();edge=valid&((uv.min(2)<.04)|(uv.sum(2)>.96));wire[edge]=[.1,.1,.1]
        overlay=gray*.5;ids=np.where(valid,tid,0);cc=comp[ids];overlay[valid]=np.stack([((cc[valid]*73)%255)/255,((cc[valid]*151+40)%255)/255,((cc[valid]*199+80)%255)/255],1)
        with np.load(item['path']) as z:fg=z['foreground_mask'].copy();gt=z['gt_rgb'].copy()
        missing=fg&~pred;wrongblack=fg&pred&(rgb@[.2126,.7152,.0722]<.12)&(gt@[.2126,.7152,.0722]>.25)
        d=a.out/f'{sid:04d}';d.mkdir(exist_ok=True);ims=[('gray',gray),('normal',normal),('wireframe',wire),('color',rgb),('components',overlay)]
        for name,img in ims:png(d/f'{name}.png',img)
        save_npz_fast(d/'raycast.npz',depth=np.where(pred,depth,0),triangle_ids=tid,component_ids=np.where(valid,cc,-1),missing=missing,wrong_black_surface=wrongblack)
        if sid==157:bounds=[836,299,1028,491]
        else:bounds=next(r['ROIs']['leaf_texture'] for r in regions if r['source_id']==sid)
        x1,y1,x2,y2=bounds;panel=Image.new('RGB',(6*320,350),'white');draw=ImageDraw.Draw(panel)
        for j,(name,img) in enumerate([('GT',gt),*ims]):
            crop=Image.fromarray((img[y1:y2,x1:x2].clip(0,1)*255).round().astype(np.uint8)).resize((320,320));panel.paste(crop,(j*320,30));draw.text((j*320+5,5),name,fill='black')
        panel.save(d/'local_panel.png');records.append(dict(source_id=sid,bounds=bounds,missing_pixels=int(missing[y1:y2,x1:x2].sum()),wrong_black_surface_pixels=int(wrongblack[y1:y2,x1:x2].sum()),extra_pixels=int((pred&~fg&(ndimage.distance_transform_edt(~fg)>3))[y1:y2,x1:x2].sum())))
        print('mesh views',a.out.name,sid,records[-1],flush=True)
    write(a.out/'views_complete.json',dict(mesh=str(a.mesh.resolve()),sha256=sha(a.mesh),views=8,records=records,wireframe='Raycast barycentric edge display, not geometry smoothing',normal='native world normal converted to camera coordinates',black_classification='Missing geometry and covered dark color errors are counted separately; original GT darkness excluded from color-error candidate'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
