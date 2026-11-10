from common import *
from artifact_evidence import contribution
from plyfile import PlyData,PlyElement
import torch
def main():
    d=ROOT/'task0_gaussian_cleaning/diagnosis';GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(P2));torch.set_num_threads(8);old=np.load(SEL/'P2_GGGS_Cleaning/cleaning_evidence.npz');ids=np.flatnonzero(~old['removed']);data=PlyData.read(str(P2))['vertex'].data;xyz=np.column_stack([data[k] for k in ['x','y','z']])
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());cams,refs=cameras();regions=[('left_cloud','free_0000_-15',[0,200,380,900]),('upper_cloud','free_0053_-15',[0,0,1080,800]),('outer_leaf_strip','free_0157_+15',[0,850,1080,1250]),('leaf_center_crossing',0,[398,596,718,916]),('pot_marker_floating','free_0000_+15',[450,1180,950,1700]),('source157_tip',157,[836,299,1028,491])];records=[]
    for name,view,bounds in regions:
        if isinstance(view,str):r=next(r for r in protocol['views'] if r['name']==view);c=free_camera(r)
        else:c=cams[next(i for i,r in enumerate(refs) if r['source_id']==view)]
        x1,y1,x2,y2=bounds;roi=np.zeros((c.image_height,c.image_width),bool);roi[y1:y2,x1:x2]=True
        g,radii,err=contribution(c,model,np.stack([roi,roi,roi]));w=g[:,0];order=np.argsort(-w);cumulative=np.cumsum(w[order]);sel=order[:np.searchsorted(cumulative,.95*cumulative[-1])+1];np.save(d/(name+'_dominant_original_ids.npy'),ids[sel]);points=xyz[sel]
        records.append(dict(name=name,camera=view,roi=bounds,p2_original_ids=ids[sel].tolist(),native_alpha_contribution=float(w.sum()),bbox_min=points.min(0).tolist(),bbox_max=points.max(0).tolist(),description='95% actual T*alpha contribution in fixed visually selected region. Central-leaf and pot regions contain valid geometry: this is localization, not blanket deletion authorization.'))
        PlyData([PlyElement.describe(data[sel].copy(),'vertex')],text=False,byte_order='<').write(str(d/(name+'_contributors.ply')))
        print('regional evidence',name,len(sel),flush=True)
    write(d/'artifact_regions.json',dict(regions=records,source=str(P2),source_sha256=sha(P2),actual_native_rasterizer=True))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
