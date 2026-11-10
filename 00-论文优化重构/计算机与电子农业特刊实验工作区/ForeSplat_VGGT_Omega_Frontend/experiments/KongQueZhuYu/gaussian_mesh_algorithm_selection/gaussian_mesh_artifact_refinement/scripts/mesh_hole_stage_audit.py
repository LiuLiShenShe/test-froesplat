from common import *
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
def one(sid):
    route=ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh';pairs=json.loads((route/'stereo/camera_data.json').read_text())['pairs'];i=next(i for i,p in enumerate(pairs) if p['left']['source_id']==sid);d=route/'stereo'/f'{i:03}';fg=np.load(d/'left_mask.npy');sd=np.load(d/'out_DLNR_Middlebury/depth.npy');occ=np.load(d/'out_DLNR_Middlebury/occlusion_mask.npy');support=fg&occ&(sd>=.1511057476)&(sd<=1.5692457557)
    with np.load(route/'visualizations'/f'{sid:04d}'/'raycast.npz') as z:missing=z['missing'].copy();wrong=z['wrong_black_surface'].copy()
    rec=next(r for r in json.loads((route/'visualizations/views_complete.json').read_text())['records'] if r['source_id']==sid);x1,y1,x2,y2=rec['bounds'];roi=np.zeros_like(fg);roi[y1:y2,x1:x2]=True;select=missing&roi;internal=select&ndimage.binary_erosion(fg,iterations=3)
    return dict(source_id=sid,roi=rec['bounds'],mesh_missing_pixels=int(select.sum()),interior_mesh_missing_pixels=int(internal.sum()),missing_with_valid_source_stereo=int((select&support).sum()),missing_without_valid_source_stereo=int((select&~support).sum()),interior_missing_with_valid_stereo=int((internal&support).sum()),wrong_black_covered_surface_pixels=int((roi&wrong).sum()))
def main():
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,[0,26,53,79,105,131,157,183]))
    write(ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/LOCAL_HOLE_STAGE_AUDIT.json',dict(records=rows,interpretation='Mesh-uncovered pixels with valid source stereo can arise from conflicting other views, finite voxel integration or different surfaces. This does not prove those stereo depths correct. Missing projections and covered black color are distinct; no arbitrary hole fill was performed.'))
    print(rows,flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
