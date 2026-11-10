"""Describe actual matched-view changes; one plant/seed supplies no independent run inference."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
DEST=OUT/'comparison/paired_view_changes'
PHOTO={'full_psnr':1,'fg_mask_psnr':1,'boundary_band5_psnr':1,'inner_fg_edge5_psnr':1,
       'ssim':1,'fg_mask_ssim_map':1,'lpips_alex':-1,'fg_gradient_ratio_abs_error':-1,
       'background_alpha_mean':-1,'foreground_alpha_mean':0,'fg_gradient_retention':0}
MESH={'silhouette_iou':1,'foreground_coverage':1,'foreground_precision':1,
      'background_geometry_fraction':-1,'boundary_fscore_2px':1,'boundary_chamfer_px':-1,
      'hole_count':-1,'hole_area_px':-1,'hole_area_fg_ratio':-1,'open_edge_missing_area_px':-1}

def read(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []

def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def group(rows,keys):
    result=defaultdict(dict)
    for row in rows:
        key=tuple(row[k] for k in keys);sid=int(row['source_id'])
        assert sid not in result[key],('Duplicated source view',key,sid)
        result[key][sid]=row
    return result

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    paths=[OUT/'comparison/rendering_metrics.csv',OUT/'comparison/mesh_metrics.csv']
    photos=group(read(paths[0]),['experiment','stage'])
    meshes=group([r for r in read(paths[1]) if r['reference']=='RAP predicted mask'],['method','stage'])
    expected=set(photos[('E10','raw')]);assert len(expected)==208
    def native_photo_key(method):
        keys=[key for key in photos if key[0]==method]
        assert len(keys)<=1,('Explicit native stage required',method,keys)
        return keys[0] if keys else None
    pairs=[]
    for candidate in sorted(photos):
        if candidate!=('E10','raw'):
            controlled='EO_only_E10_training_baseline' if candidate==('E12','corrected') or candidate==('E12','raw') else 'post_training_cleaning_same_E10_checkpoint' if candidate[0]=='P2' else 'matched_evaluation_historical_conditions_or_native_strategy_differ'
            pairs.append(('Gaussian',('E10','raw'),candidate,controlled))
        if candidate[0] in ['E13','E14','E14_initial','E14_cluster_before','E14_cluster_after','E14_refined']:
            pairs.append(('Gaussian',('E11','raw'),candidate,'same_input_native_method_and_extra_prior_or_stage_differ'))
    before=native_photo_key('E14_cluster_before');after=native_photo_key('E14_cluster_after')
    if before and after:pairs.append(('Gaussian',before,after,'exact_same_iteration_DBSCAN_intervention'))
    for candidate in sorted(meshes):
        if candidate[1]=='post' and candidate!=('M4_E10_GGGS_TSDF','post'):
            pairs.append(('Mesh',('M4_E10_GGGS_TSDF','post'),candidate,'own_Omega_camera_gauge_same_RGB_ids_and_RAP_reference' if candidate[0]=='M1_Omega_NKSR' else 'common_post_rule_and_RAP_reference_native_source_or_extractor_differ'))
        if candidate[1]=='post':
            before=(candidate[0],'binary') if (candidate[0],'binary') in meshes else (candidate[0],'raw')
            if before in meshes:pairs.append(('Mesh',before,candidate,'same_surface_component_only_postprocessing'))
        if candidate[1]=='binary' and (candidate[0],'raw') in meshes:
            pairs.append(('Mesh',(candidate[0],'raw'),candidate,'same_marching_tetrahedra_binary_search_refinement'))
    summaries=[];detail=[];excluded=[];series={}
    for family,base,candidate,condition in pairs:
        groups=photos if family=='Gaussian' else meshes
        if base not in groups or candidate not in groups or set(groups[base])!=expected or set(groups[candidate])!=expected:
            excluded.append(dict(family=family,baseline=base,candidate=candidate,reason='Requires all actual208 identical source IDs'));continue
        if family=='Gaussian':
            differences=[sid for sid in sorted(expected) if any(groups[base][sid].get(k)!=groups[candidate][sid].get(k) for k in ['width','height','gt_sha256','mask_sha256'])]
            assert not differences,('Photometric evaluation reference differs',base,candidate,differences)
        metrics=PHOTO if family=='Gaussian' else MESH
        for metric,direction in metrics.items():
            values=[];base_values=[];candidate_values=[];ids=[];invalid=[]
            for sid in sorted(expected):
                a=groups[base][sid];b=groups[candidate][sid]
                try:x=float(a[metric]);y=float(b[metric])
                except (KeyError,ValueError):invalid.append(sid);continue
                if not math.isfinite(x) or not math.isfinite(y):invalid.append(sid);continue
                change=y-x;values.append(change);base_values.append(x);candidate_values.append(y);ids.append(sid)
                detail.append(dict(family=family,baseline_method=base[0],baseline_stage=base[1],candidate_method=candidate[0],candidate_stage=candidate[1],metric=metric,source_id=sid,baseline_value=x,candidate_value=y,change_candidate_minus_baseline=change,direction=direction,comparison_condition=condition,baseline_metric_source=a.get('metric_source',''),candidate_metric_source=b.get('metric_source','')))
            if not values:continue
            arr=np.asarray(values);a=np.asarray(base_values);b=np.asarray(candidate_values)
            summaries.append(dict(family=family,baseline_method=base[0],baseline_stage=base[1],candidate_method=candidate[0],candidate_stage=candidate[1],metric=metric,actual_paired_views=len(arr),invalid_views=len(invalid),direction=direction,baseline_mean=float(a.mean()),candidate_mean=float(b.mean()),mean_change=float(arr.mean()),view_change_SD=float(arr.std(ddof=1)),change_p10=float(np.quantile(arr,.1)),change_median=float(np.median(arr)),change_p90=float(np.quantile(arr,.9)),change_min=float(arr.min()),change_max=float(arr.max()),improved_view_fraction=float((direction*arr>1e-12).mean()) if direction else '',tied_view_fraction=float((np.abs(arr)<=1e-12).mean()),comparison_condition=condition))
            series[(family,base,candidate,metric)]=(ids,arr)
    write(DEST/'PAIRED_VIEW_METRIC_CHANGES.csv',detail)
    write(DEST/'PAIRED_VIEW_CHANGE_SUMMARY.csv',summaries)
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#000000','#999999']
    plot_specs=[('Gaussian',('E10','raw'),[('P2','raw'),('E11','raw'),('E12','corrected')]+[key for method in ['E13','E14','E14_refined'] if (key:=native_photo_key(method))],['fg_mask_psnr','lpips_alex']),
                ('Mesh',('M4_E10_GGGS_TSDF','post'),[('P1','post'),('P2_TSDF','post'),('P3_shared','post'),('P4_median_mask_alpha','post'),('E12_TSDF','post'),('E13_QGS','post'),('E14_initial','post'),('E14_refined','post')],['silhouette_iou','hole_area_fg_ratio'])]
    figures=[]
    for family,base,candidates,metrics in plot_specs:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        plotted=[]
        for index,candidate in enumerate(candidates):
            available=False
            for axis,metric in zip(axes,metrics):
                result=series.get((family,base,candidate,metric))
                if result is None:continue
                ids,values=result;sorted_values=np.sort(values)
                axis.step(sorted_values,np.arange(1,len(values)+1)/len(values),where='post',label=candidate[0]+(' corrected' if candidate[1]=='corrected' else ''),color=colors[index])
                available=True
            if available:plotted.append(candidate)
        labels={'fg_mask_psnr':'Foreground PSNR change (dB; positive improves)',
                'lpips_alex':'Full-image Alex LPIPS change (negative improves)',
                'silhouette_iou':'RAP silhouette IoU change (positive improves)',
                'hole_area_fg_ratio':'Enclosed missing / RAP area change (negative reduces missing)'}
        for axis,metric in zip(axes,metrics):
            axis.axvline(0,color='.5',lw=.8,ls='--');axis.set_xlabel(labels[metric]);axis.set_ylabel('Cumulative fraction of actual paired source views');axis.set_ylim(0,1.01);axis.grid(alpha=.2)
        axes[0].legend(fontsize=8,loc='best');fig.suptitle(f'{family}: matched 208 source views, baseline {base[0]} {base[1]}\nCandidate minus baseline; one plant/seed, descriptive ECDF, no independent-run inference',fontsize=10)
        fig.tight_layout()
        for suffix in ['png','pdf']:
            path=DEST/f'{family.lower()}_paired_view_ECDF.{suffix}';fig.savefig(path,dpi=160);figures.append(str(path.resolve()))
        plt.close(fig)
    result=dict(status='complete_available',summary_rows=len(summaries),paired_metric_rows=len(detail),
        expected_source_ids=sorted(expected),one_plant='KongQueZhuYu',native_training_seed=0,
        statistical_scope='Descriptive within-sequence view SD/quantiles/fractions; views are correlated, not independent plants/runs. No p-value, standard error or confidence interval is inferred.',
        selection_scope='Per-metric improved fractions are not an automatic algorithm ranking. Coverage/hole reductions may coexist with overfilling; native alpha and gradient retention are not physical accuracy.',
        depth_scope='Per-method Gaussian-to-Mesh depth discrepancies are omitted: they use different model references, not common scan ground truth.',
        exclusions=excluded,figures=figures,
        input_files=[dict(path=str(p.resolve()),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DEST/'PROTOCOL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Actual matched-view summaries',len(summaries),'metric pairs',len(detail),'excluded comparisons',len(excluded),flush=True)

if __name__=='__main__':main()
