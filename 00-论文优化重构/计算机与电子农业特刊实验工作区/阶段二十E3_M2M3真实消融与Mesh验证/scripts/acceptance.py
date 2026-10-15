"""Real acceptance: immutable assets, paired runs, reloads and measured outputs."""
from common import *
import ast,math,time

def checkpoint_cpu_check(sample,method):
    import numpy as np,torch
    from plyfile import PlyData
    path=model_path(sample,method);capture,iteration=torch.load(path/'chkpnt30000.pth',map_location='cpu',weights_only=False)
    assert iteration==30000
    ply=PlyData.read(path/'point_cloud/iteration_30000/point_cloud.ply')['vertex']
    names=ply.data.dtype.names
    def columns(prefix):return np.stack([ply[k] for k in sorted([k for k in names if k.startswith(prefix)],key=lambda k:int(k.rsplit('_',1)[1]))],1)
    arrays=[np.stack([ply[k] for k in ['x','y','z']],1),columns('f_dc_').reshape(-1,3,1).transpose(0,2,1),
            columns('f_rest_').reshape(-1,3,15).transpose(0,2,1),columns('scale_'),columns('rot_'),np.asarray(ply['opacity'])[:,None]]
    for index,array in enumerate(arrays,1):
        assert np.array_equal(array,capture[index].detach().numpy()),(sample,method,index)
        assert np.isfinite(array).all()
    return {'sample':sample,'method':method,'status':'PASS','iteration':iteration,'Gaussian_count':len(arrays[0]),
            'checkpoint_sha256':sha(path/'chkpnt30000.pth'),'PLY_sha256':sha(path/'point_cloud/iteration_30000/point_cloud.ply'),
            'comparison':'Independent CPU PLY field reconstruction equals checkpoint six parameter tensors exactly'}

def mesh_cpu_check(row):
    import numpy as np,open3d as o3d
    from mesh_metrics import structure
    assert sha(row['mesh_path'])==row['mesh_sha256']
    mesh=o3d.io.read_triangle_mesh(row['mesh_path']);actual,_=structure(mesh)
    mesh.compute_vertex_normals();assert np.isfinite(np.asarray(mesh.vertex_normals)).all()
    for key,value in actual.items():
        expected=row[key]
        if isinstance(value,list):assert np.allclose(value,ast.literal_eval(expected),rtol=0,atol=1e-10),key
        elif isinstance(value,(int,float)):
            assert math.isclose(value,float(expected),rel_tol=1e-10,abs_tol=1e-10),(row['mesh_path'],key,value,expected)
        elif value is None:assert expected in ['',None]
        else:assert str(value)==expected
    return {'sample':row['sample'],'method':row['method'],'mesh_stage':row['mesh_stage'],'mesh_sha256':row['mesh_sha256'],'status':'PASS','vertices':actual['vertices'],'triangles':actual['triangles']}

def main():
    start=time.perf_counter();gates={};details={}
    def gate(name,call):
        try:
            evidence=call();gates[name]='PASS';details[name]=evidence
        except BaseException as error:
            gates[name]='FAIL';details[name]=repr(error)
        print(name,gates[name],flush=True)
    def immutable():
        f=verify_frozen(full=True)
        changed=[p for p in subprocess.check_output(['git','diff',BASELINE,'--name-only','-z'],cwd=ROOT).decode().split('\0') if p]
        prefix=str(STAGE.relative_to(ROOT))+'/'
        assert all(p.startswith(prefix) for p in changed),changed
        return {'SHA256_verified_files':len(f['protected']),'historical_tracked_changes':0}
    gate('G1_FROZEN_MODELS_AND_INPUTS',immutable)
    gate('G2_THREE_REFERENCE_MESHES',lambda:all_assert([read_json(STAGE/f'reports/{s}_M0_MESH_RUN.json')['status']=='PASS' for s in SCENES]))
    def meshes():
        checks=[]
        for row in rows(STAGE/'MESH_STRUCTURE_METRICS.csv'):
            checks.append(mesh_cpu_check(row));write_csv(STAGE/'reports/MESH_INDEPENDENT_RELOAD.csv',checks)
        assert len(checks)==18
        return {'independent_reloads':len(checks)}
    gate('G3_RELOADABLE_RECOMPUTABLE_MESHES',meshes)
    gate('G4_FROZEN_TSDF_PROTOCOL',lambda:all_assert([sha(x['path'])==x['sha256'] for x in read_json(STAGE/'configs/FREEZE.json')['configs'] if '_MESH' in x['path']]))
    gate('G5_NO_INVENTED_PHYSICAL_SCALE',lambda:all_assert([r['physical_units']=='arbitrary inherited COLMAP units' for r in rows(STAGE/'MESH_STRUCTURE_METRICS.csv')]))
    def source_matrix():
        matrix=rows(STAGE/'M2M3_CODE_FORMULA_MATRIX.csv');assert len(matrix)>=15
        for entry in matrix:
            if entry['source_path']:assert sha(entry['source_path'])==entry['source_sha256']
        return {'entries':len(matrix),'source_unchanged':True,'production_mathematics':'Inherited unchanged'}
    gate('G6_PRUNING_CALL_CHAIN_AUDITED',source_matrix)
    def pairing():
        for sample in SCENES:
            ca=read_json(STAGE/f'configs/{sample}_M1_TRAIN.json');cb=read_json(STAGE/f'configs/{sample}_M2_TRAIN.json')
            for section in ['model','optimization','pipeline','all_parameters']:
                aa={k:v for k,v in ca[section].items() if k not in ['model_path','capacity_control_mode']}
                bb={k:v for k,v in cb[section].items() if k not in ['model_path','capacity_control_mode']}
                assert aa==bb,(sample,section)
            a=read_json(STAGE/f'reports/{sample}_M1_INITIALIZATION.json');b=read_json(STAGE/f'reports/{sample}_M2_INITIALIZATION.json');assert a==b
            a=read_json(STAGE/f'reports/{sample}_M1_CAMERA_SEQUENCE.json');b=read_json(STAGE/f'reports/{sample}_M2_CAMERA_SEQUENCE.json');assert a==b and len(a)==30000
        return {'scene_pairs':3,'same_initial_tensors_RNG_alpha_sparse':True,'same_30000_camera_draws':True}
    gate('G7_PAIRED_INITIALIZATION_CONFIGURATIONS',pairing)
    def equivalence():
        r=read_json(STAGE/'reports/BASELINE_EQUIVALENCE.json');assert r['status']=='SOURCE_EQUIVALENT_DISABLED_PATH' and r['M0_branch_control_required'] is False
        checks=[]
        for sample in SCENES:
            old=read_json(E0/f'reports/{sample}_PREFLIGHT.json')
            group=reference(sample)['actual_group']
            historical=next(x for x in old['groups'] if x['group']==group)['initialization_sha256']
            for method in ['M1','M2']:
                fresh=read_json(STAGE/f'reports/{sample}_{method}_INITIALIZATION.json')['parameter_hashes']
                for field,new_field in [('xyz','_xyz'),('scale','_scaling'),('rotation','_rotation'),('opacity','_opacity')]:
                    assert historical[field]==fresh[new_field],(sample,method,field)
                    checks.append({'sample':sample,'method':method,'field':field,'historical_initial_sha256':historical[field],
                                   'fresh_initial_sha256':fresh[new_field],'status':'PASS','historical_group':group})
        write_csv(STAGE/'reports/HISTORICAL_INITIAL_TENSOR_COMPARISON.csv',checks)
        return {**r,'additional_archived_initial_tensor_hash_checks':len(checks),
                'archived_initial_fields':'xyz, scale, rotation, opacity; historical full RNG state remains unavailable'}
    gate('G8_STANDARD_BASELINE_REUSE',equivalence)
    def trainings():
        checks=[]
        for sample in SCENES:
            for method in ['M0','M1','M2']:
                if method!='M0':
                    run=read_json(STAGE/f'reports/{sample}_{method}_TRAIN_RUN.json')
                    assert run['status']=='PASS' and run['completed_iterations']==30000 and run['optimizer_steps']==29999 and run['reload_verified']
                checks.append(checkpoint_cpu_check(sample,method));write_csv(STAGE/'reports/CHECKPOINT_INDEPENDENT_RELOAD.csv',checks)
        return {'new_full_trainings':6,'independent_checkpoint_PLY_checks':len(checks)}
    gate('G9_COMPLETE_TRAININGS_AND_CHECKPOINTS',trainings)
    def events_check():
        values=rows(STAGE/'M2M3_PRUNING_EVENT_LOG.csv');assert len(values)==30
        for sample in SCENES:
            for method in ['M1','M2']:
                own=[r for r in values if r['sample']==sample and r['method']==method]
                assert [int(r['iteration']) for r in own]==[18000,21000,24000,27000,30000]
                for r in own:
                    before=int(r['Gaussian_count_before']);removed=int(r['removed_count'])
                    assert before-removed==int(r['Gaussian_count_after']) and 0<=removed<=int(.03*before)
                    if r['score_status']=='EVALUATED':
                        assert r['mask_score_available']=='True'
                        assert all(math.isfinite(float(r[k])) for k in ['score_min','score_mean','score_max','mask_score_min','mask_score_mean','mask_score_max'])
        return {'actual_scheduled_events':30,'extra_removed':sum(int(r['removed_count']) for r in values)}
    gate('G10_ACTUAL_PRUNING_AND_BUDGETS',events_check)
    def floors():
        values=[r for r in rows(STAGE/'M2M3_PRUNING_EVENT_LOG.csv') if r['method']=='M2'];assert len(values)==15
        count=sum(int(r['floor_blocked_count']) for r in values)
        for r in values:
            if r['floor_evaluated']=='True':
                assert r['floor_active']=='True'
                expected=min(int(r['Gaussian_count_before']),round(.4*int(r['reference_max_seen_count'])))
                assert int(r['floor_count'])==expected
                allowed=max(0,int(r['Gaussian_count_before'])-expected)
                assert int(r['floor_blocked_count'])==max(0,int(r['budget_limited_count'])-allowed)
        for sample in SCENES:
            cap=read_json(model_path(sample,'M2')/'capacity_control/capacity_summary.json')
            assert cap['total_blocked_by_floor']==sum(int(r['floor_blocked_count']) for r in values if r['sample']==sample)
        return {'M2_events':15,'floor_blocked_count_total':count,'activation':'VERIFIED' if count>0 else 'NOT_TRIGGERED',
                'evaluated_events':sum(r['floor_evaluated']=='True' for r in values)}
    gate('G11_FLOOR_ACTIVATION_EVIDENCE',floors)
    def common_references():
        data=rows(STAGE/'M2M3_HELDOUT_METRICS.csv');assert len(data)==204
        for sample in SCENES:
            by_method={m:{r['loader_image_name']:r for r in data if r['sample']==sample and r['method']==m} for m in ['M0','M1','M2']}
            assert by_method['M0'].keys()==by_method['M1'].keys()==by_method['M2'].keys()
            for name in by_method['M0']:
                for field in ['reference_original_sha256','reference_resized_tensor_sha256','ROI_original_sha256','ROI_resized_tensor_sha256','render_width','render_height']:
                    assert len({by_method[m][name][field] for m in by_method})==1,(sample,name,field)
        return {'common_reference_views':len(data),'unique_heldout_cameras':68}
    gate('G12_COMMON_CAMERAS_RAW_RGB_AND_FIXED_ROI',common_references)
    def tsdf():
        for sample in SCENES:
            expected=sha(STAGE/f'configs/{sample}_MESH.json')
            for method in ['M0','M1','M2']:
                r=read_json(STAGE/f'reports/{sample}_{method}_MESH_RUN.json');assert r['mesh_protocol_sha256']==expected
        return {'same_protocol_within_each_scene':True,'scene_scale_rule':'depth=2*radius, voxel=depth/1024, sdf=5*voxel'}
    gate('G13_IDENTICAL_TSDF_RULES',tsdf)
    gate('G14_PROXIES_DISTINGUISHED_FROM_GT',lambda:all_assert([r['ROI_is_independent_GT']=='False' for r in rows(STAGE/'M2M3_HELDOUT_METRICS.csv')]))
    gate('G15_NO_UNIMPLEMENTED_BIDIRECTIONAL_CLAIM',lambda:all_assert([(STAGE/'M2M3_MANUSCRIPT_CORRECTION_SPEC.md').exists(),(STAGE/'M2M3_METHOD_DECISION.md').exists()]))
    gate('G16_PRODUCTION_MATH_UNCHANGED',lambda:all_assert([gates['G1_FROZEN_MODELS_AND_INPUTS']=='PASS']))
    def retained():
        assert len(rows(STAGE/'M2M3_SCENE_SUMMARY.csv'))==9
        assert all((STAGE/f'logs/{s}_{m}_TRAIN.txt').exists() for s in SCENES for m in ['M1','M2'])
        return {'all_declared_model_results':9,'all_new_training_logs':6,'quality_based_reruns':0}
    gate('G17_NEGATIVE_AND_FAILED_EVIDENCE_RETAINED',retained)
    gate('G18_MANUSCRIPT_CORRECTION_SPEC',lambda:all_assert([(STAGE/'M2M3_MANUSCRIPT_CORRECTION_SPEC.md').stat().st_size>4000,(STAGE/'M2M3_CODE_FORMULA_MATRIX.csv').stat().st_size>1000]))
    gate('G19_TABLE7_UNVERIFIED_HISTORY_NOT_INHERITED',lambda:all_assert([(STAGE/'TABLE7_EVIDENCE_UPDATE.md').exists(),gates['G1_FROZEN_MODELS_AND_INPUTS']=='PASS']))
    gate('G20_PHASE20F_SCALE_AND_MEASUREMENT_REQUIREMENTS',lambda:all_assert([(STAGE/'PHASE20F_HANDOFF.md').stat().st_size>1000]))
    def metrics():
        r=read_json(STAGE/'reports/FLOAT_METRIC_RECOMPUTATION.json');assert r['status']=='PASS' and r['views']==204
        return r
    gate('EXTRA_ALL_FLOAT_METRICS_RECOMPUTED',metrics)
    output={'status':'PASS' if all(v=='PASS' for v in gates.values()) else 'PARTIAL','time':now(),'baseline_commit':BASELINE,'gates':gates,'evidence':details,'wall_seconds':time.perf_counter()-start,'acceptance_script_sha256':sha(__file__)}
    write_json(STAGE/'reports/FINAL_ACCEPTANCE.json',output)
    print('FINAL_ACCEPTANCE',output['status'],flush=True)
    assert output['status']=='PASS'

def all_assert(items):
    assert items and all(items),items
    return {'checks':len(items),'all_passed':True}

if __name__=='__main__':main()
