"""Separate nearby layer counts from parallel front/back spacing; draw actual rays."""
import csv,json,hashlib,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT/'comparison/same_ray_mesh_layers'
VIS=OUT/'comparison/visualizations/same_ray_layers'

def read(path):return list(csv.DictReader(path.open()))
def writecsv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def hits_at(data,i,epsilon):
    lo,hi=map(int,data['ray_splits'][i:i+2]);t=data['t_hit'];normal=data['triangle_normals_world'];direction=data['rays_world'][i,3:].astype(np.float64)
    length=np.linalg.norm(direction);order=np.arange(lo,hi);order=order[np.isfinite(t[order])&(t[order]>0)]
    order=order[np.argsort(t[order],kind='stable')];keep=[]
    for k in order:
        if not keep or (float(t[k])-float(t[keep[-1]]))*length>epsilon:keep.append(int(k))
    return keep,length,direction/length

def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());scale=json.loads((OUT.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
    rows=[];routes={};pair_data={};queries=read(OUT/'comparison/reflection_analysis/RGB_QUERY_MANIFEST.csv');raw=read(OUT/'comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv')
    bright=np.array([r['raw_context_candidate'].lower()=='true' for r in raw]);assert bright.sum()==134
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or not (folder/'complete.json').is_file():continue
        meta=json.loads((folder/'complete.json').read_text());summary=read(folder/'summary.csv')[0];key=(summary['mesh_id'],summary['mesh_stage']);routes[key]=folder
        with np.load(folder/'actual_intersections.npz') as stored:
            data={key:stored[key] for key in ['ray_splits','t_hit','triangle_normals_world','rays_world','query_ids']}
            n=len(data['query_ids']);gap=np.full(n,np.nan);plane=np.full(n,np.nan);dot=np.full(n,np.nan);parallel=np.zeros(n,bool)
            for i in range(n):
                keep,length,direction=hits_at(data,i,protocol['merge_world_epsilon'])
                if len(keep)<2:continue
                left,right=keep[:2]
                if data['t_hit'][right]>protocol['maximum_camera_Z']:continue
                n0,n1=data['triangle_normals_world'][[left,right]]
                gap[i]=(float(data['t_hit'][right])-float(data['t_hit'][left]))*length
                dot[i]=np.dot(n0,n1);incidence=abs(np.dot(n0,direction))
                parallel[i]=dot[i]<=-.9 and incidence>=.1
                if parallel[i]:plane[i]=gap[i]*incidence
            np.savez_compressed(folder/'first_pair_spacing.npz',query_ids=data['query_ids'],first_two_gap_world=gap,first_two_normal_dot=dot,parallel_opposite_pair=parallel,parallel_plane_normal_gap_world=plane)
            assert np.array_equal(data['query_ids'],np.arange(len(queries))),folder
            if key[1]=='post':
                pair_data[key[0]]=dict(query_ids=data['query_ids'],parallel=parallel,plane=plane,mesh=meta['mesh'],folder=folder)
        for group,choose in [('all_fixed_queries',np.ones(n,bool)),('original_RGB_brightness',bright)]:
            valid=choose&parallel;values=plane[valid]
            rows.append(dict(mesh_id=key[0],mesh_stage=key[1],evaluated_mesh_path=meta['mesh'],group=group,queries=int(choose.sum()),
                opposite_parallel_first_pair_queries=int(valid.sum()),opposite_parallel_pair_fraction_all_queries=float(valid.sum()/max(choose.sum(),1)),
                parallel_plane_gap_D_median=float(np.median(values)/scale['diagonal']) if len(values) else '',
                parallel_plane_gap_voxels_median=float(np.median(values)/scale['tsdf_voxel']) if len(values) else '',
                parallel_plane_gap_voxels_p10=float(np.quantile(values,.1)/scale['tsdf_voxel']) if len(values) else '',
                parallel_plane_gap_voxels_p90=float(np.quantile(values,.9)/scale['tsdf_voxel']) if len(values) else '',
                scope='Conditional signed opposite parallel first-face pair, |dot(n0,n1)|>=.9 and incidence>=.1. Normal-projected spacing is not physical leaf thickness or scan error; another leaf can form the pair. Coverage/cohort sizes must accompany spacing and layer counts.'))
    writecsv(ROOT/'FIRST_PAIR_SPACING_METRICS.csv',rows)
    preferred=['M4_E10_GGGS_TSDF','P1','P2_TSDF','P2_Tetrahedra','P3_shared','P4_median_mask_alpha','E12_TSDF','E12_Tetrahedra','E13_QGS','E14_initial','E14_refined']
    paired=[];paired_root=ROOT/'matched_first_pairs';paired_root.mkdir(exist_ok=True)
    for reference,candidate in itertools.combinations([name for name in preferred if name in pair_data],2):
        left,right=pair_data[reference],pair_data[candidate]
        assert np.array_equal(left['query_ids'],right['query_ids'])
        joint=left['parallel']&right['parallel']
        difference=(right['plane']-left['plane'])/scale['tsdf_voxel']
        cache=paired_root/(reference+'__'+candidate+'.npz')
        np.savez_compressed(cache,query_ids=left['query_ids'],both_opposite_parallel=joint,
            reference_plane_normal_gap_world=left['plane'],candidate_plane_normal_gap_world=right['plane'],
            candidate_minus_reference_gap_voxels=difference,original_RGB_brightness=bright)
        for group,choose in [('all_fixed_queries',np.ones(len(queries),bool)),('original_RGB_brightness',bright)]:
            valid=choose&joint;values=difference[valid]
            paired.append(dict(reference_mesh_id=reference,candidate_mesh_id=candidate,mesh_stage='post',group=group,
                reference_mesh_path=left['mesh'],candidate_mesh_path=right['mesh'],queries=int(choose.sum()),
                joint_pair_queries=int(valid.sum()),joint_pair_fraction_all_queries=float(valid.sum()/max(choose.sum(),1)),
                reference_eligible_queries=int((choose&left['parallel']).sum()),candidate_eligible_queries=int((choose&right['parallel']).sum()),
                reference_joint_gap_voxels_median=float(np.median(left['plane'][valid])/scale['tsdf_voxel']) if valid.any() else '',
                candidate_joint_gap_voxels_median=float(np.median(right['plane'][valid])/scale['tsdf_voxel']) if valid.any() else '',
                paired_candidate_minus_reference_gap_voxels_median=float(np.median(values)) if len(values) else '',
                paired_candidate_minus_reference_gap_voxels_p10=float(np.quantile(values,.1)) if len(values) else '',
                paired_candidate_minus_reference_gap_voxels_p90=float(np.quantile(values,.9)) if len(values) else '',
                paired_candidate_gap_greater_fraction=float(np.mean(values>0)) if len(values) else '',
                actual_pair_arrays=str(cache.resolve()),query_manifest_SHA256=protocol['query_SHA256'],source_views=protocol['source_views'],
                scope='Paired on the same frozen source-ray IDs, eligible in both methods. First opposite parallel surfaces can be different anatomical leaves; the common cohort is not physical leaf thickness, scan accuracy or all-surface coverage.'))
    if paired:writecsv(ROOT/'MATCHED_FIRST_PAIR_SPACING.csv',paired)
    (ROOT/'FIRST_PAIR_PROTOCOL.json').write_text(json.dumps(dict(status='actual_conditional_and_joint_pair_metrics',
        query_manifest_SHA256=protocol['query_SHA256'],queries=len(queries),source_views=protocol['source_views'],
        mesh_stage='post for joint comparisons; all actual stages retained individually',
        opposite_normal_dot_maximum=-.9,minimum_absolute_ray_normal_dot=.1,
        pair_definition='First two distinct positive mesh intersections within the shared camera-Z limit; normal-projected world spacing uses the first signed triangle normal.',
        joint_definition='Same query IDs satisfy the pair criterion in both methods; report pair eligibility and joint coverage rather than comparing independent conditional medians.',
        physical_thickness_claim=False,raw_mesh_files_unchanged=True),indent=2)+'\n')
    assert ('P1','post') in routes and ('M4_E10_GGGS_TSDF','post') in routes
    VIS.mkdir(exist_ok=True);selection=VIS/'SELECTED_RAY_IDS.json'
    if selection.is_file():selected=json.loads(selection.read_text())['records']
    else:
        with np.load(routes[('P1','post')]/'actual_intersections.npz') as p1,np.load(routes[('M4_E10_GGGS_TSDF','post')]/'actual_intersections.npz') as m4:
            candidate=(p1['near_front_layers']>=3)&(m4['near_front_layers']<=1)&(m4['distinct_layers']>0)
            selected=[]
            for sid in [0,26,53,79,105,131,157,183]:
                indices=np.array([i for i,r in enumerate(queries) if int(r['source_id'])==sid],dtype=int)
                if not len(indices):continue
                eligible=indices[candidate[indices]];i=int(eligible[0] if len(eligible) else indices[0])
                selected.append(dict(query_id=i,source_id=sid,x=float(queries[i]['x']),y=float(queries[i]['y']),selected_layer_difference_example=bool(len(eligible))))
        selection.write_text(json.dumps(dict(status='frozen_actual_query_examples',records=selected,
            criterion='First existing fixed RGB match at each fixed view with P1>=3 near-front layers and M4<=1; fallback first existing match. Purpose-selected examples; rates use all fixed queries, not these crops.',query_manifest_SHA256=protocol['query_SHA256']),indent=2)+'\n')
    for selected_ray in selected:
        i=selected_ray['query_id'];sid=selected_ray['source_id'];record=raw[i];x,y=selected_ray['x'],selected_ray['y']
        with np.load(routes[('M4_E10_GGGS_TSDF','post')]/'actual_intersections.npz') as data:
            keep,length,_=hits_at(data,i,protocol['merge_world_epsilon']);reference=float(data['t_hit'][keep[0]])
        shown=[(method,routes[(method,'post')]) for method in preferred if (method,'post') in routes]
        fig=plt.figure(figsize=(14,7));grid=fig.add_gridspec(2,2,width_ratios=[1,2.3]);left=max(0,min(1080-192,int(round(x))-96));top=max(0,min(1920-192,int(round(y))-96))
        for j,(label,path) in enumerate([('Original02FFT RGB',Path(record['source_raw_rgb_path'])),('03SAM training RGB',Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu')/f'crop_{sid:04d}.png')]):
            assert path.is_file(),path
            image=Image.open(path).convert('RGB').resize((1080,1920));crop=image.crop((left,top,left+192,top+192));ax=fig.add_subplot(grid[j,0]);ax.imshow(crop);ax.scatter([x-left],[y-top],s=65,facecolors='none',edgecolors='#f500b4',linewidths=1.5);ax.set_title(label);ax.axis('off')
        ax=fig.add_subplot(grid[:,1])
        for row,(method,folder) in enumerate(shown):
            with np.load(folder/'actual_intersections.npz') as data:
                keep,length,direction=hits_at(data,i,protocol['merge_world_epsilon'])
                z=(data['t_hit'][keep]-reference)*length/scale['tsdf_voxel'];normals=data['triangle_normals_world'][keep];colors=normals@direction;valid=(z>=-20)&(z<=25)
                ax.scatter(z[valid],np.full(int(valid.sum()),row),c=colors[valid],cmap='coolwarm',vmin=-1,vmax=1,s=48,edgecolors='black',linewidths=.3)
        ax.set_yticks(range(len(shown)),[r[0] for r in shown]);ax.invert_yaxis();ax.axvline(0,color='#555555',linestyle='--',linewidth=.8);ax.set_xlim(-20,25);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('World distance along the same ray relative to M4 first hit / shared voxel')
        ax.set_title('Actual distinct mesh intersections; blue/red = signed face-normal dot ray')
        fig.suptitle(f'Fixed source{sid}, query{i}, pixel({x:.2f},{y:.2f})',fontsize=13)
        fig.text(.02,.015,'Same corrected camera and original mesh files. A pair can be legitimate leaf front/back or an occluded leaf; several nearby layers do not certify an error.\nPurpose-selected illustration; frequency and spacing use the full24375 fixed RGB queries from69 source cameras;195 source/target photographs.',fontsize=8)
        fig.tight_layout(rect=[0,.075,1,.94]);fig.savefig(VIS/f'source_{sid:04d}_query_{i}_actual_layers.png',dpi=150);plt.close(fig)
    (VIS/'PROTOCOL.json').write_text(json.dumps(dict(status='complete_available_actual_layer_examples',selected_queries=selected,actual_post_routes=[k[0] for k in routes if k[1]=='post'],layer_protocol=str((ROOT/'PROTOCOL.json').resolve()),physical_thickness_claim=False),indent=2)+'\n')
    print('Actual pair spacing and ray figures',len(routes),len(selected),flush=True)

if __name__=='__main__':main()
