"""Execute selected, reviewed models only; every GPU stage owns the same lock."""
import fcntl,json,subprocess
from runtime import OUT,ROOT,FRONT,PYTHONS,write
from pipeline import stage


def main():
    selection=json.loads((OUT/'summary/selected_models.json').read_text())
    assert selection['review_complete']
    assert 1<=len(selection['mesh_model_keys'])<=2
    assert (OUT/'summary/cleaning_decision.md').exists()
    nksr_python=FRONT/'experiments/KongQueZhuYu/geometry_mesh_final_selection/.venv-nksr/bin/python'
    with (OUT/'summary/gpu1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for key in selection['mesh_model_keys']:
            method=key.removesuffix('_Cleaned');cleaned=key.endswith('_Cleaned')
            surface=OUT/'mesh'/f'{key}_surface'
            args=[method,'--surface']+(['--cleaned'] if cleaned else [])
            stage('surface_'+key,PYTHONS[method],'evaluate.py',args,surface/'complete.json')
            for algorithm in ['NKSR','TSDF','Poisson']:
                python=nksr_python if algorithm=='NKSR' else PYTHONS['2DGS' if algorithm=='TSDF' else 'GGGS']
                folder=OUT/'mesh'/algorithm/key
                stage('mesh_'+algorithm+'_'+key,python,'mesh_extract.py',[key,algorithm],folder/'complete.json')
                stage('mesh_eval_'+algorithm+'_'+key,PYTHONS['GGGS'],'mesh_evaluate.py',[key,algorithm],folder/'evaluation/complete.json')
                stage('mesh_view_'+algorithm+'_'+key,PYTHONS['GGGS'],'viewer.py',[key,'--mesh',algorithm],folder/'post_3d.html')
            if method=='GGGS':
                folder=OUT/'mesh/native'/key
                stage('mesh_native_'+key,PYTHONS['GGGS'],'native_tetra.py',['--cleaned'] if cleaned else [],folder/'complete.json')
                stage('mesh_eval_native_'+key,PYTHONS['GGGS'],'mesh_evaluate.py',[key,'native'],folder/'evaluation/complete.json')
                stage('mesh_view_native_'+key,PYTHONS['GGGS'],'viewer.py',[key,'--mesh','native'],folder/'post_3d.html')
        subprocess.run([str(PYTHONS['GGGS']),str(OUT/'scripts/report.py')],cwd=ROOT,check=True)
        write(OUT/'summary/progress.json',dict(status='mesh_review_ready',selected=selection['mesh_model_keys'],requires='Inspect actual mesh projections and geometry, write final recommendation, run delivery validation'))


if __name__=='__main__':main()
