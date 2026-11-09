"""Isolate the E6 numerical implementation; repair only pixel/camera interfaces."""
from pathlib import Path
import shutil, json, hashlib, difflib

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/fj/F2DMAS/2d-gaussian-splatting-main')
NEW = ROOT/'camera_interface/2dgs_corrected'

def replace(path, old, new):
    text = path.read_text()
    assert old in text, (path, old[:90])
    path.write_text(text.replace(old, new, 1))

def main():
    NEW.mkdir(exist_ok=True)
    for name in ['scene','utils','arguments','gaussian_renderer','lpipsPyTorch','submodules']:
        if not (NEW/name).exists():
            shutil.copytree(OLD/name,NEW/name,ignore=shutil.ignore_patterns('__pycache__','build','*.egg-info'))
    for name in ['train.py','render.py','LICENSE.md','README.md']:
        if not (NEW/name).exists(): shutil.copy2(OLD/name,NEW/name)
    ready=ROOT/'camera_interface/2dgs_patch_complete.json'
    if ready.exists():return
    # Official training loop, original E6 GaussianModel, optimizer and CUDA kernels.
    shutil.copy2(ROOT/'source_reference/2DGS_official_train.py',NEW/'train.py')
    reader=NEW/'scene/dataset_readers.py'
    replace(reader,'    height: int\n','    height: int\n    cx: float = None\n    cy: float = None\n')
    replace(reader,'image_path=image_path, image_name=image_name, width=width, height=height)',
        'image_path=image_path, image_name=image_name, width=width, height=height,\n                              cx=float(intr.params[1] if intr.model == "SIMPLE_PINHOLE" else intr.params[2]),\n                              cy=float(intr.params[2] if intr.model == "SIMPLE_PINHOLE" else intr.params[3]))')
    camera=NEW/'scene/cameras.py'
    replace(camera,'scale=1.0, data_device = "cuda"','scale=1.0, data_device = "cuda", fx=None, fy=None, cx=None, cy=None')
    replace(camera,'        self.full_proj_transform =',
        '        self.Fx = float(fx if fx is not None else self.image_width/(2*np.tan(FoVx/2)))\n'
        '        self.Fy = float(fy if fy is not None else self.image_height/(2*np.tan(FoVy/2)))\n'
        '        self.Cx = float(cx if cx is not None else (self.image_width-1)/2)\n'
        '        self.Cy = float(cy if cy is not None else (self.image_height-1)/2)\n'
        '        p = self.projection_matrix.T.clone()\n'
        '        p[0,0] = 2*self.Fx/self.image_width\n'
        '        p[1,1] = 2*self.Fy/self.image_height\n'
        '        p[0,2] = (2*self.Cx+1)/self.image_width-1\n'
        '        p[1,2] = (2*self.Cy+1)/self.image_height-1\n'
        '        self.projection_matrix = p.T.contiguous()\n'
        '        self.full_proj_transform =')
    text=camera.read_text();camera.write_text(text[:text.index('class MiniCam:')]+'class MiniCam:\n    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):\n        self.image_width = width\n        self.image_height = height\n        self.FoVy = fovy\n        self.FoVx = fovx\n        self.znear = znear\n        self.zfar = zfar\n        self.world_view_transform = world_view_transform\n        self.full_proj_transform = full_proj_transform\n        # Recover the supplied noncentral projection instead of inventing K.\n        self.projection_matrix = torch.linalg.solve(world_view_transform, full_proj_transform)\n        p = self.projection_matrix.T\n        self.Fx = float(p[0,0] * width / 2)\n        self.Fy = float(p[1,1] * height / 2)\n        self.Cx = float((p[0,2]+1) * width / 2 - 0.5)\n        self.Cy = float((p[1,2]+1) * height / 2 - 0.5)\n        self.camera_center = torch.inverse(world_view_transform)[3,:3]\n')
    loader=NEW/'utils/camera_utils.py'
    replace(loader,'image_name=cam_info.image_name, uid=id, data_device=args.data_device)',
        'image_name=cam_info.image_name, uid=id, data_device=args.data_device,\n'
        '                  fx=fov2focal(cam_info.FovX,orig_w)*resolution[0]/orig_w,\n'
        '                  fy=fov2focal(cam_info.FovY,orig_h)*resolution[1]/orig_h,\n'
        '                  cx=((cam_info.cx if cam_info.cx is not None else (orig_w-1)/2)+0.5)*resolution[0]/orig_w-0.5,\n'
        '                  cy=((cam_info.cy if cam_info.cy is not None else (orig_h-1)/2)+0.5)*resolution[1]/orig_h-0.5)')
    points=NEW/'utils/point_utils.py'
    text=points.read_text();a=text.index('    ndc2pix =');b=text.index('    grid_x, grid_y',a)
    text=text[:a]+'''    # Use the same integer-pixel-center K as the renderer and TSDF cameras.
    fx = getattr(view, 'Fx', W/(2*math.tan(view.FoVx/2)))
    fy = getattr(view, 'Fy', H/(2*math.tan(view.FoVy/2)))
    cx = getattr(view, 'Cx', (W-1)/2)
    cy = getattr(view, 'Cy', (H-1)/2)
    intrins = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],device='cuda',dtype=torch.float32)

'''+text[b:];points.write_text(text)
    renderer=NEW/'gaussian_renderer/__init__.py'
    replace(renderer,"            'surf_depth': surf_depth,", "            'surf_depth': surf_depth,\n            'depth_median': render_depth_median,\n            'depth_expected': render_depth_expected,")
    changed=[];diffdir=ROOT/'camera_interface/diffs';diffdir.mkdir(exist_ok=True)
    for name in ['train.py','scene/dataset_readers.py','scene/cameras.py','utils/camera_utils.py','utils/point_utils.py','gaussian_renderer/__init__.py']:
        old=(OLD/name).read_text();new=(NEW/name).read_text()
        diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='E6/'+name,tofile='E11/'+name))
        (diffdir/(name.replace('/','__')+'.diff')).write_text(diff)
        changed.append({'path':name,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()})
    ready.write_text(json.dumps({'source':str(OLD),'isolated':str(NEW),'changed':changed,
        'cuda_numerical_changes':False,'cuda_note':'Both surfel forward and backward use full projective splat-to-pixel homography with W-1/H-1 pixel centers, preserving noncentral P; validated before training.',
        'optimizer_gaussian_model_unchanged':True,'official_training_loop':str(ROOT/'source_reference/2DGS_official_train.py')},indent=2)+'\n')
    print('independent E11 camera/pixel patch prepared; original E6 files unchanged')

if __name__=='__main__':main()
