from common import *
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save(path,v):PlyData([PlyElement.describe(v.copy(),'vertex')],text=False,byte_order='<').write(str(path))
def main():
    d=ROOT/'task0_gaussian_cleaning/diagnosis';best=ROOT/'task0_gaussian_cleaning/best_cleaned'
    data=PlyData.read(str(SOURCE))['vertex'].data;xyz=np.column_stack([data[k] for k in ['x','y','z']])
    evidence=dict(np.load(d/'actual_contribution_evidence.npz'));geom=dict(np.load(d/'geometric_auxiliary_evidence.npz'))
    ids=np.load(best/'original_gaussian_ids.npy');v=PlyData.read(str(best/'cleaned_gaussians.ply'))['vertex'].data
    assert data[ids].dtype==v.dtype and data[ids].tobytes()==v.tobytes()
    write(best/'NATIVE_LINEAGE_VERIFICATION.json',dict(status='passed',count=len(ids),all_native_fields_bitwise_equal_original=True,fields=list(v.dtype.names),sha256=sha(best/'cleaned_gaussians.ply'),original_sha256=sha(SOURCE)))
    remaining=np.load(best/'remaining_dependencies.npz') if (best/'remaining_dependencies.npz').exists() else None
    # Candidate models preserve native fields and original ID maps, without certifying all as junk.
    residual=json.loads((best/'remaining_dependencies.json').read_text());candidate_ids=np.unique(np.asarray(residual['top100_original_ids']+[j for r in residual['per_view'] for j in r['top_remaining_original_ids']],np.int64))
    save(d/'candidate_remaining_gaussians.ply',data[candidate_ids]);np.save(d/'candidate_remaining_original_ids.npy',candidate_ids)
    normal_ids=np.setdiff1d(ids,candidate_ids);save(d/'retained_gaussians.ply',data[normal_ids]);np.save(d/'retained_original_ids.npy',normal_ids)
    # Conservative supplementary spatial crop, separate from the G2 input.
    keptxyz=xyz[ids];pot=(keptxyz[:,1]>-.015)&(np.linalg.norm(keptxyz[:,[0,2]]-[.09015,1.04532],axis=1)<.205)
    save(best/'plant_only_geometry_v2.ply',v[~pot]);np.save(best/'plant_only_v2_original_gaussian_ids.npy',ids[~pot])
    write(best/'PLANT_ONLY_V2.json',dict(count=int((~pot).sum()),excluded=int(pot.sum()),rule='Y>-.015 and radiusXZ around [.09015,1.04532]<.205; actual original P3 pot region plus conservative margin',status='Exploratory geometry crop; overlap with leaf bases/root structures remains uncertain. Native Gaussian footprint can still cross semantic boundary.',source_sha256=sha(best/'cleaned_gaussians.ply')))
    labels=np.zeros(len(data),np.uint8);labels[np.load(best/'removed_original_gaussian_ids.npy')]=2;labels[candidate_ids]=1
    colors=['#3388bb','#d9a600','#c44949'];fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,(u,w) in zip(axes,[(0,1),(2,1)]):
        for k,name in enumerate(['Retained','Residual candidates','Removed']):
            ix=np.flatnonzero(labels==k); ix=ix[::max(1,len(ix)//18000)]
            ax.scatter(xyz[ix,u],-xyz[ix,w],s=1 if k!=1 else 4,c=colors[k],label=name,rasterized=True,alpha=.4)
        ax.set_xlabel(['X','Y','Z'][u]+' (native model units)');ax.set_ylabel('-Y');ax.set_aspect('equal');ax.legend(markerscale=4)
    fig.suptitle('Actual original Gaussian centers: removal and residual attribution');fig.tight_layout();fig.savefig(d/'gaussian_spatial_distribution.png',dpi=150);plt.close(fig)
    # Radius graph of poorly foreground-supported centers is a proxy, never a plant semantic certificate.
    fg=evidence['fg_contribution']/208;bg=evidence['bg_contribution']/208+evidence['free_bg_contribution']/8;harm=bg/(fg+bg+1e-10)
    candidate=(harm>.95)&(bg>.05)
    p2_ids=np.flatnonzero(~np.load(SEL/'P2_GGGS_Cleaning/cleaning_evidence.npz')['removed'])
    records=[]
    for name,active in [('E10',np.arange(len(data))),('P2',p2_ids),('Best',ids)]:
        points=xyz[active[candidate[active]]];pairs=cKDTree(points).query_pairs(.015,output_type='ndarray')
        if len(pairs):a=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(points),len(points)))
        else:a=coo_matrix((len(points),len(points)))
        n,l=connected_components(a,directed=False);sizes=np.bincount(l);volume=0.;large=0
        for k in np.flatnonzero(sizes>=10):
            p=points[l==k];volume+=float(np.prod(np.ptp(p,axis=0)));large+=1
        records.append(dict(model=name,candidate_points=len(points),radius_graph_components=int(n),components_ge10_points=large,isolated_centers=int((sizes==1).sum()),summed_cluster_aabb_volume=volume))
    write(d/'SPATIAL_COMPONENT_METRICS.json',dict(radius=.015,records=records,scope='Original measured background harm>.95,bgweight>.05; not all connected components are proven floaters. AABB volume is model-coordinate bounding volume, not material volume.'))
    print('spatial deliverables complete',flush=True)
if __name__=='__main__':main()
