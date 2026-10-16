"""Record pinned official implementation source, dependencies and code anchors."""
from common import *

PUP='https://github.com/j-alex-hanson/gaussian-splatting-pup'
LIGHT='https://github.com/VITA-Group/LightGaussian'

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

def main():
    data=[];index=[]
    for name,url in [('gaussian-splatting-pup',PUP),('LightGaussian',LIGHT)]:
        repo=UPSTREAM/name;commit=git(repo,'rev-parse','HEAD')
        status=git(repo,'status','--porcelain');assert not status,status
        submodules=[]
        for line in git(repo,'ls-tree','-r','HEAD').splitlines():
            info,path=line.split('\t')
            if info.startswith('160000'):
                revision=info.split()[2]
                actual=git(repo/path,'rev-parse','HEAD') if (repo/path/'.git').exists() else 'NOT_INITIALIZED'
                if actual!='NOT_INITIALIZED':assert actual==revision
                submodules.append({'path':path,'pinned_gitlink':revision,'checkout':actual})
        license=repo/'LICENSE.md'
        data.append({'name':name,'url':url,'commit':commit,'checkout_path':str(repo),'worktree_clean':True,
            'commit_date':git(repo,'show','-s','--format=%cI','HEAD'),
            'license':'Gaussian-Splatting License (Inria/MPII), non-commercial research and evaluation; not an MIT license',
            'license_sha256':sha(license),'license_path':str(license),'environment_yaml':(repo/'environment.yml').read_text(),
            'environment_sha256':sha(repo/'environment.yml'),'submodules':submodules,'executed_or_built_upstream_CUDA':False})
        wanted=[repo/'environment.yml',license,repo/'.gitmodules']
        wanted += [f for f in repo.rglob('*') if f.is_file() and '.git' not in f.parts and
                   f.suffix in ['.py','.cu','.cuh','.cpp','.h','.md'] and
                   (f.name in ['prune.py','prune_finetune.py','fisher_pool_xyz_scaling.py','gaussian_model.py','rasterize_points.cu','LICENSE.md']
                    or f.name=='__init__.py' and any(t in f.parts for t in ['fisher_renderer','gaussian_renderer','diff_surfel_rasterization','diff_gaussian_rasterization','rasterization_and_pup_fisher'])
                    or f.name in ['forward.cu','backward.cu','rasterizer_impl.cu','rasterizer.h'])]
        for f in sorted(set(wanted)):
            rel=f.relative_to(repo)
            parent=repo;origin=url;rev=commit;within=rel
            for s in submodules:
                if Path(s['path']) in rel.parents:
                    parent=repo/s['path'];rev=s['pinned_gitlink'];within=f.relative_to(parent)
                    origin=git(parent,'remote','get-url','origin').removesuffix('.git')
                    break
            index.append({'repository':name,'relative_path':str(rel),'sha256':sha(f),'bytes':f.stat().st_size,
                          'lines':len(f.read_text(errors='replace').splitlines()),'commit':rev,
                          'source_url':origin+'/blob/'+rev+'/'+str(within)})
    write_json(STAGE/'reports/UPSTREAM_AUDIT.json',{'recorded_at':now(),'repositories':data})
    write_csv(STAGE/'configs/UPSTREAM_CODE_INDEX.csv',index)
    # Existing local implementation is equally traceable.
    local=[]
    paths=['scene/gaussian_model.py','scene/__init__.py','gaussian_renderer/__init__.py','utils/general_utils.py','utils/mesh_utils.py',
           'submodules/diff-surfel-rasterization/diff_surfel_rasterization/__init__.py',
           'submodules/diff-surfel-rasterization/rasterize_points.cu',
           'submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu',
           'submodules/diff-surfel-rasterization/cuda_rasterizer/backward.cu']
    for rel in paths:
        path=GS/rel
        local.append({'relative_path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size,
                      'baseline_commit':BASELINE,'source_url':'https://github.com/LiuLiShenShe/test-froesplat/blob/'+BASELINE+'/'+str(path.relative_to(ROOT))})
    write_csv(STAGE/'configs/NATIVE_CODE_INDEX.csv',local)
    print('UPSTREAM_AUDIT_RECORDED',[(r['name'],r['commit']) for r in data],len(index),flush=True)

if __name__=='__main__':main()
