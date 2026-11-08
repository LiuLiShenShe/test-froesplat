"""Isolated GGGS camera/pixel repair; optimizer and losses are unchanged."""
import difflib, hashlib, json, re, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parents[2]/'gggs_community'
DEST=ROOT/'camera_interface/gggs_corrected';EXT=DEST/'corrected_rasterizer'

def replace(path,old,new,count=None):
    text=path.read_text();n=text.count(old)
    assert n>0 and (count is None or n==count),(path,old,n)
    path.write_text(text.replace(old,new))

def region_edit(path,start,end,callback):
    s=path.read_text();a=s.index(start);b=s.index(end,a) if end else len(s)
    path.write_text(s[:a]+callback(s[a:b])+s[b:])

def matrix_signature(s):
    # This region contains only the render/sample declaration, device kernel,
    # or matching wrapper. Preserve all existing arithmetic and argument order.
    s,n=re.subn(r'((?:const )?float focal_y,)',r'\1 const float* ray_view, const float* ray_proj,',s)
    assert n>0
    return s

def matrix_calls(s):
    return s.replace('focal_x, focal_y,','focal_x, focal_y, ray_view, ray_proj,')

def main():
    if DEST.exists():raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    for d in ['scene','utils','arguments','gaussian_renderer','models','lpipsPyTorch']:
        shutil.copytree(SOURCE/d,DEST/d,ignore=shutil.ignore_patterns('__pycache__'))
    for f in ['train_foresplat.py','LICENSE.md']:shutil.copy2(SOURCE/f,DEST/f)
    shutil.copytree(SOURCE/'submodules/diff-gaussian-rasterization',EXT,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__','*.so','.git'))
    # Python extension is loaded from this private source directory only.
    replace(DEST/'gaussian_renderer/__init__.py','from diff_gaussian_rasterization import',"import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]/'corrected_rasterizer'))\nfrom diff_gaussian_rasterization import",1)
    replace(DEST/'scene/dataset_readers.py','    mask: np.array = None','    mask: np.array = None\n    cx: float = None\n    cy: float = None',1)
    replace(DEST/'scene/dataset_readers.py','image_path=image_path, image_name=image_name, width=width, height=height)',
        'image_path=image_path, image_name=image_name, width=width, height=height, cx=float(intr.params[-2]), cy=float(intr.params[-1]))',1)
    replace(DEST/'utils/camera_utils.py','image_name=cam_info.image_name, uid=id, data_device=args.data_device)',
        'image_name=cam_info.image_name, uid=id, data_device=args.data_device, intrinsics=(\n'
        '                      fov2focal(cam_info.FovX, orig_w)*resolution[0]/orig_w,\n'
        '                      fov2focal(cam_info.FovY, orig_h)*resolution[1]/orig_h,\n'
        '                      (cam_info.cx+0.5)*resolution[0]/orig_w-0.5,\n'
        '                      (cam_info.cy+0.5)*resolution[1]/orig_h-0.5))',1)
    replace(DEST/'scene/cameras.py','scale=1.0, data_device = "cuda",','scale=1.0, data_device = "cuda", intrinsics=None,',1)
    replace(DEST/'scene/cameras.py','        self.Cy = float(self.image_height - 1) / 2',
        '        self.Cy = float(self.image_height - 1) / 2\n        if intrinsics is not None:\n            self.Fx,self.Fy,self.Cx,self.Cy=map(float,intrinsics)',1)
    replace(DEST/'scene/cameras.py','        self.full_proj_transform = (self.world_view_transform.unsqueeze',
        '        # Integer pixel centers, matching CUDA ndc2Pix.\n'
        '        self.projection_matrix[2,0]=(2*self.Cx+1)/self.image_width-1\n'
        '        self.projection_matrix[2,1]=(2*self.Cy+1)/self.image_height-1\n'
        '        self.projection_matrix[0,0]=2*self.Fx/self.image_width\n'
        '        self.projection_matrix[1,1]=2*self.Fy/self.image_height\n'
        '        self.full_proj_transform = (self.world_view_transform.unsqueeze',1)
    cu=EXT/'cuda_rasterizer'
    # The public Python/C++ ABI is unchanged. The already supplied W2C and
    # full projection matrices reach each CUDA ray kernel. Projecting the
    # camera optical-axis DIRECTION removes world translation and recovers K.
    helper='''
// Recover the integer-center principal point from supplied W2C/full P.
// Matrices have the existing GGGS column-major storage convention.
__device__ __forceinline__ float2 foresplatPrincipal(const float* view, const float* proj, int W, int H) {
    const float x=view[2], y=view[6], z=view[10];
    const float invw=1.f/(proj[3]*x+proj[7]*y+proj[11]*z);
    return {ndc2Pix((proj[0]*x+proj[4]*y+proj[8]*z)*invw,W),
            ndc2Pix((proj[1]*x+proj[5]*y+proj[9]*z)*invw,H)};
}
'''
    path=cu/'auxiliary.h';s=path.read_text();last=s.rfind('#endif');path.write_text(s[:last]+helper+s[last:])
    for stem,namespace in [('render_forward','FORWARD'),('render_backward','BACKWARD')]:
        path=cu/(stem+'.cu')
        region_edit(path,'    renderCUDA(', 'void '+namespace+('::render' if namespace=='FORWARD' else '::preprocess'),matrix_signature)
        region_edit(path,'void '+namespace+'::render','void '+namespace+'::preprocess' if namespace=='FORWARD' else None,lambda s:matrix_calls(matrix_signature(s)))
        region_edit(cu/(stem+'.h'),'void render(',');',matrix_signature)
    for stem,namespace in [('sample_forward','FORWARD'),('sample_backward','BACKWARD')]:
        path=cu/(stem+'.cu')
        # The only sampled-point reconstruction with centered rays is depth.
        a=path.read_text().index('sampleDepthCUDA(')
        s=path.read_text();b=s.index('void '+namespace+'::preprocess_points',a)
        path.write_text(s[:a]+matrix_signature(s[a:b])+s[b:])
        region_edit(path,'void '+namespace+'::sampleDepth(',None,lambda s:matrix_calls(matrix_signature(s)))
        region_edit(cu/(stem+'.h'),'void sampleDepth(',');',matrix_signature)
    for stem in ['render_forward','render_backward','sample_forward','sample_backward']:
        path=cu/(stem+'.cu')
        replace(path,'static_cast<float>(W - 1) / 2.f','foresplatPrincipal(ray_view,ray_proj,W,H).x')
        replace(path,'static_cast<float>(H - 1) / 2.f','foresplatPrincipal(ray_view,ray_proj,W,H).y')
    path=cu/'rasterizer_impl.cu';s=path.read_text()
    # Only internal ray-using call sites acquire matrix arguments.
    for name in ['FORWARD::render','BACKWARD::render','FORWARD::sampleDepth','BACKWARD::sampleDepth']:
        pattern=re.escape(name)+r'\((.*?)\),'
        def call(m):
            text=m.group(0);assert 'focal_x, focal_y,' in text
            return text.replace('focal_x, focal_y,','focal_x, focal_y, viewmatrix, projmatrix,')
        s,n=re.subn(pattern,call,s,flags=re.S);assert n>0,(name,n)
    path.write_text(s)
    changed=[]
    for file in DEST.rglob('*'):
        if not file.is_file():continue
        rel=file.relative_to(DEST)
        old=SOURCE/rel if rel.parts[0]!='corrected_rasterizer' else SOURCE/'submodules/diff-gaussian-rasterization'/Path(*rel.parts[1:])
        if old.exists() and old.read_bytes()!=file.read_bytes():
            changed.append(str(rel));diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),file.read_text().splitlines(True),fromfile=str(old),tofile=str(file)))
            target=ROOT/'camera_interface/diffs'/Path(str(rel)+'.diff');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(diff)
    (ROOT/'camera_interface/repair_provenance.json').write_text(json.dumps(dict(source=str(SOURCE),source_commit='23b418c2f0f19b30b251acad6fe13445dae968c7',corrected=str(DEST),changed_files=changed,
        unchanged_optimizer=hashlib.sha256((SOURCE/'scene/gaussian_model.py').read_bytes()).hexdigest()==hashlib.sha256((DEST/'scene/gaussian_model.py').read_bytes()).hexdigest(),
        python_cpp_public_abi_unchanged=True,repair='Source K -> actual PIL half-pixel resize -> Camera Fx/Fy/Cx/Cy -> asymmetric P -> forward/backward render and sample-depth rays; NCC already consumes Camera K'),indent=2))
    print('isolated corrected interface created',DEST,flush=True)
if __name__=='__main__':main()
