"""Native GGGS inference with dimensionful search/near lengths in cm.

The native SAMPLE_RANGE is0.4 model units. A metric Gaussian must use0.4*s
centimetres for identical rays. This isolated build changes only those two
length constants; original training code/extension remain untouched.
"""
from batch_common import *
import shutil,subprocess,fcntl

def prepare():
 source=GGGS/'submodules/diff-gaussian-rasterization';dest=OUT/'runtime/metric_rasterizer';marker=dest/'build_complete.json';dest.parent.mkdir(parents=True,exist_ok=True);lock=(dest.parent/'metric_renderer_build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 if marker.exists():return dest
 dest.mkdir(exist_ok=True)
 for name in ['cuda_rasterizer','diff_gaussian_rasterization']:
  shutil.copytree(source/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.so','__pycache__','*.pyc'))
 (dest/'third_party').mkdir(exist_ok=True)
 if not(dest/'third_party/glm').exists():link(dest/'third_party/glm',source/'third_party/glm')
 for name in ['setup.py','ext.cpp','rasterize_points.h','rasterize_points.cu','LICENSE.md']:
  if (source/name).exists():shutil.copyfile(source/name,dest/name)
 forward=dest/'cuda_rasterizer/render_forward.cu';text=forward.read_text();anchor='template <int SH_C>'
 # A symbol local to this translation unit needs no cross-CUDA device link.
 include='\n__device__ __constant__ float foresplat_length_unit_scale = 1.0f;\nvoid foresplat_set_length_unit_scale(float value) {\n  cudaError_t result = cudaMemcpyToSymbol(foresplat_length_unit_scale, &value, sizeof(float));\n  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));\n}\n'
 position=text.index('\n',text.index('#include'))
 # Insert after the last include so cuda runtime and stdexcept are declared.
 lines=text.splitlines();last=max(i for i,line in enumerate(lines) if line.startswith('#include'));lines.insert(last+1,'#include <stdexcept>\n'+include);text='\n'.join(lines)+'\n'
 assert text.count('mDepthinit - SAMPLE_RANGE')==1 and text.count('mDepthinit + SAMPLE_RANGE')==1
 text=text.replace('mDepthinit - SAMPLE_RANGE','mDepthinit - SAMPLE_RANGE * foresplat_length_unit_scale').replace('mDepthinit + SAMPLE_RANGE','mDepthinit + SAMPLE_RANGE * foresplat_length_unit_scale').replace('prefiltered, p_view))','prefiltered, p_view, 0.2f * foresplat_length_unit_scale))');forward.write_text(text)
 auxiliary=dest/'cuda_rasterizer/auxiliary.h';text=auxiliary.read_text();assert text.count('float3& p_view)')==1;text=text.replace('float3& p_view)','float3& p_view, float near_plane = 0.2f)').replace('p_view.z <= 0.2f','p_view.z <= near_plane');auxiliary.write_text(text)
 binding=dest/'ext.cpp';text=binding.read_text().replace('PYBIND11_MODULE(', 'void foresplat_set_length_unit_scale(float value);\n\nPYBIND11_MODULE(').replace('  m.def("rasterize_gaussians",','  m.def("set_length_unit_scale", &foresplat_set_length_unit_scale);\n  m.def("rasterize_gaussians",');binding.write_text(text)
 env=dict(os.environ,CUDA_HOME='/usr/local/cuda-12.6',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='2',CUDA_VISIBLE_DEVICES='1');log=dest/'build.log'
 with log.open('w') as stream:subprocess.run([str(UV),'run','--no-project','--python',str(GPU_PYTHON),'setup.py','build_ext','--inplace'],cwd=dest,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 write(marker,dict(status='complete',source=str(source),source_hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['ext.cpp','cuda_rasterizer/render_forward.cu','cuda_rasterizer/auxiliary.h']},changed_files=['ext.cpp','cuda_rasterizer/render_forward.cu','cuda_rasterizer/auxiliary.h'],changes=['Expose a process-local inference length-unit scale, default1','SAMPLE_RANGE0.4 multiplied by length-unit factor','Render frustum near0.2 multiplied by length-unit factor'],training_extension_modified=False,model_parameters_modified=False,algorithm_changed=False,build_log=str(log)));return dest

def activate(factor):
 if not np.isfinite(factor) or factor<=0:raise ValueError('Positive length-unit factor required')
 path=prepare();sys.path.insert(0,str(path))
 import diff_gaussian_rasterization as raster
 assert Path(raster.__file__).resolve().is_relative_to(path.resolve())
 raster._C.set_length_unit_scale(float(factor));return dict(scale=float(factor),native_depth_search_range=.4,metric_depth_search_range=.4*float(factor),near_plane=.2*float(factor),source=str(path),algorithm='Existing nativeGGGS median-depth; only dimensionful constants expressed in same unit as input Gaussian/cameras')

if __name__=='__main__':print(prepare())
