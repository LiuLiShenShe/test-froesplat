"""Prepare an isolated official QGS source with consistent integer pixel centers."""
from pathlib import Path
import shutil,json,hashlib,difflib
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'official'
ADAPTED=ROOT/'adapted'
def replace(relative,old,new,count=-1):
 p=ADAPTED/relative;t=p.read_text();assert old in t,(relative,old[:70]);p.write_text(t.replace(old,new,count))
def main():
 assert not ADAPTED.exists(),"Adapter already exists; inspect instead of overwriting."
 shutil.copytree(OFFICIAL,ADAPTED,ignore=shutil.ignore_patterns('.git','assets','__pycache__','build','*.egg-info'))
 replace('scene/cameras.py','self.cam_intr = torch.tensor(cam_info.cam_intr * cam_info.downsample,dtype=torch.float32,device="cpu")',
 '''# Canonical integer pixel centers; exact image resize including half-pixel.
        sx, sy = resolution[0] / cam_info.width, resolution[1] / cam_info.height
        intr = np.asarray(cam_info.cam_intr, dtype=np.float64).copy()
        intr[0] *= sx
        intr[1] *= sy
        intr[2] = (intr[2] + 0.5) * sx - 0.5
        intr[3] = (intr[3] + 0.5) * sy - 0.5
        self.cam_intr = torch.tensor(intr, dtype=torch.float32, device="cpu")''')
 replace('scene/cameras.py','        self.full_proj_transform =',
 '''        p = self.projection_matrix.T.clone()
        p[0,0] = 2 * self.Fx / self.image_width
        p[1,1] = 2 * self.Fy / self.image_height
        p[0,2] = (2 * self.Cx + 1) / self.image_width - 1
        p[1,2] = (2 * self.Cy + 1) / self.image_height - 1
        self.projection_matrix = p.T.contiguous()
        self.full_proj_transform =''',count=1)
 replace('scene/dataset_readers.py','dtype=np.float32','dtype=np.float64')
 for i,k in enumerate(['Fx','Fy','Cx','Cy']):
  replace('scene/cameras.py',f'self.{k} = self.cam_intr[{i}]',f'self.{k} = float(intr[{i}])')
 p=ADAPTED/'scene/cameras.py';t=p.read_text();start=t.index('    def get_rays(');end=t.index('\nclass MiniCam:',start)
 t=t[:start]+'''    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(torch.arange(W,device="cuda").float(),
                               torch.arange(H,device="cuda").float(),indexing="xy")
        K = self.get_k(scale)
        return torch.stack([(ix-K[0,2])/K[0,0],(iy-K[1,2])/K[1,1],torch.ones_like(ix)],-1)
    def get_k(self, scale=1.0):
        cx = (self.Cx + 0.5)/scale - 0.5
        cy = (self.Cy + 0.5)/scale - 0.5
        return torch.tensor([[self.Fx/scale,0,cx],[0,self.Fy/scale,cy],[0,0,1]],device="cuda")
    def get_inv_k(self, scale=1.0):
        return torch.linalg.inv(self.get_k(scale))
''' + t[end:];p.write_text(t)
 replace('utils/depth_utils.py',".float()+0.5",".float()")
 for relative in ['submodules/diff-quadratic-rasterization/cuda_rasterizer/forward.cu',
                  'submodules/diff-quadratic-rasterization/cuda_rasterizer/backward.cu',
                  'submodules/diff-quadratic-rasterization/cuda_rasterizer/stopthepop_QGS/resorted_render.cuh']:
  replace(relative,'(float)pix.x + 0.5f','(float)pix.x')
  replace(relative,'(float)pix.y + 0.5f','(float)pix.y')
 replace('train.py','torch.arange(W, device=device).float() + 0.5, torch.arange(H, device=device).float() + 0.5',
                    'torch.arange(W, device=device).float(), torch.arange(H, device=device).float()')
 replace('train.py','pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale',
                    '(pixels.reshape(-1, 1, 2) + 0.5) / viewpoint_cam.ncc_scale - 0.5')
 for variable in ['pixels_patch','grid']:
  for axis,dim in [(0,'W'),(1,'H')]:
   replace('train.py',f'2 * {variable}[:, :, {axis}] / {dim} - 1.0',
                        f'2 * ({variable}[:, :, {axis}] + 0.5) / {dim} - 1.0')
 replace('scene/gaussian_model.py','], -1).float()/scale','], -1).float()\n        pts_projections = (pts_projections + 0.5)/scale - 0.5')
 replace('scene/gaussian_model.py','pts_projections[..., 0] /= (W / 2)','pts_projections[..., 0] = (pts_projections[..., 0] + 0.5) / (W / 2)')
 replace('scene/gaussian_model.py','pts_projections[..., 1] /= (H / 2)','pts_projections[..., 1] = (pts_projections[..., 1] + 0.5) / (H / 2)')
 p=ADAPTED/'submodules/diff-quadratic-rasterization/cuda_rasterizer/rasterizer_impl.h'
 p.write_text('#include <cstdint>\n'+p.read_text())
 changed=[]
 for p in ADAPTED.rglob('*'):
  if not p.is_file():continue
  before=OFFICIAL/p.relative_to(ADAPTED)
  if before.read_bytes()==p.read_bytes():continue
  r=str(p.relative_to(ADAPTED));old=before.read_text();new=p.read_text()
  diffs=ROOT/'patches';diffs.mkdir(exist_ok=True)
  (diffs/(r.replace('/','__')+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='official/'+r,tofile='adapted/'+r)))
  changed.append(dict(file=r,official_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),adapted_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (ROOT/'adapter_provenance.json').write_text(json.dumps(dict(official_repository='https://github.com/will-zzy/QGS',official_commit='74d05c945e99fcaef7afe5a8831903be71ad9b55',pixel_convention='integer array coordinates, canonical resized principal point, normalized grid align_corners=False uses pixel+.5',changes=changed,primitive_preserved='Signed six-channel scaling; quadratic intersection/cone bounding box/reordering/loss/optimizer/densification equations unchanged',cli_bug='Official __main__ training call commented out. Driver calls official training() directly; no replacement implementation.'),indent=2)+'\n')
 print('QGS isolated camera and pixel adapter ready',len(changed),'files')
if __name__=='__main__':main()

