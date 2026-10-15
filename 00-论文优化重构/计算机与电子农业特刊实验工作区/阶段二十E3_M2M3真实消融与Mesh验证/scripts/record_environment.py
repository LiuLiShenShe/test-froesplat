"""CPU-only environment inventory; does not create a CUDA context."""
from common import *
import importlib.metadata,platform,shutil

def main():
    versions={}
    for n in ['torch','torchvision','open3d','numpy','scipy','lpips','Pillow','matplotlib','trimesh','plyfile','opencv-python']:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]='NOT_INSTALLED_AS_DISTRIBUTION'
    commands={}
    for name,command in [('gpu',['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv']),('nvcc',['nvcc','--version']),('system',['uname','-a'])]:
        r=subprocess.run(command,capture_output=True,text=True) if shutil.which(command[0]) else None
        commands[name]={'command':command,'stdout':r.stdout if r else 'NOT_IN_PATH','returncode':r.returncode if r else None}
    extensions=[]
    for base in [GS/'submodules/diff-surfel-rasterization',GS/'submodules/simple-knn']:
        for path in base.rglob('*.so'):extensions.append({'path':str(path),'sha256':sha(path)})
    write_json(STAGE/'reports/ENVIRONMENT.json',{'timestamp':now(),'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),
               'packages':versions,'commands':commands,'compiled_extensions':extensions,'CUDA_VISIBLE_DEVICES':'1','physical_GPU_UUID':GPU_UUID,
               'thread_limits':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},'working_directory':str(ROOT)})

if __name__=='__main__':main()
