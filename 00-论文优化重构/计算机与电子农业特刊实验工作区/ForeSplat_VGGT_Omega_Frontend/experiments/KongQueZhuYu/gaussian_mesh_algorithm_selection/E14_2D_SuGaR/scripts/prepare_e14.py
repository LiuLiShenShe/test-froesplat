"""Prepare official 2D-SuGaR for frozen ForeSplat cameras without changing its primitives/losses."""
from pathlib import Path
import json, shutil, difflib, hashlib, csv, subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[3]
OFF=ROOT/'official'
DUAL=ROOT.parent.parent/'geometry_rgb_dual_frontend'
DATA=ROOT/'input_colmap'
READY=ROOT/'preparation_complete.json'
changed=[]
def replace(rel,old,new):
    p=OFF/rel
    text=p.read_text()
    if old not in text: raise RuntimeError((rel,old[:120]))
    text=text.replace(old,new,1)
    p.write_text(text)
    if rel not in changed:changed.append(rel)
def main():
    if READY.exists():return
    original={}
    names=['gaussian_splatting_2d/scene/dataset_readers.py','gaussian_splatting_2d/scene/cameras.py','gaussian_splatting_2d/utils/camera_utils.py','gaussian_splatting_2d/utils/point_utils.py','gaussian_splatting_2d/gaussian_renderer/__init__.py','gaussian_splatting_2d/scene/point_generator.py','sugar_scene/cameras.py']
    for n in names:original[n]=(OFF/n).read_text()
    (DATA/'sparse/0').mkdir(parents=True,exist_ok=True)
    (DATA/'masks').mkdir(exist_ok=True)
    ref=DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
    if not (DATA/'images').exists():(DATA/'images').symlink_to(ref/'images',target_is_directory=True)
    for p in (ref/'sparse/0').iterdir():
        if p.is_file() and p.name!='points3D.ply' and not (DATA/'sparse/0'/p.name).exists():
            (DATA/'sparse/0'/p.name).symlink_to(p.resolve())
    rows=list(csv.DictReader((DUAL/'shared_manifest.csv').open()))
    assert len(rows)==208
    for r in rows:
        dest=DATA/'masks'/r['image_name']
        if not dest.exists():dest.symlink_to(Path(r['mask_path']).resolve())
    reader='gaussian_splatting_2d/scene/dataset_readers.py'
    replace(reader,'    world_normal_map: np.array = None\n','    world_normal_map: np.array = None\n    cx: float = None\n    cy: float = None\n    mask_path: str = None\n')
    replace(reader,'depth_map=depth_map, normal_map=normal_map, world_normal_map=world_normal_map)',
        'depth_map=depth_map, normal_map=normal_map, world_normal_map=world_normal_map,\n'
        '                              cx=float(intr.params[1] if intr.model=="SIMPLE_PINHOLE" else intr.params[2]),\n'
        '                              cy=float(intr.params[2] if intr.model=="SIMPLE_PINHOLE" else intr.params[3]),\n'
        '                              mask_path=os.path.join(os.path.dirname(images_folder),"masks",image_name+".png"))')
    # Preserve corrected pixel-centre K when loading/resizing training images.
    cam='gaussian_splatting_2d/scene/cameras.py'
    replace(cam,'world_normal_map=None, data_device = "cuda"','world_normal_map=None, data_device = "cuda", fx=None, fy=None, cx=None, cy=None')
    replacement='''        self.Fx = float(fx if fx is not None else self.image_width/(2*np.tan(FoVx/2)))
        self.Fy = float(fy if fy is not None else self.image_height/(2*np.tan(FoVy/2)))
        self.Cx = float(cx if cx is not None else (self.image_width-1)/2)
        self.Cy = float(cy if cy is not None else (self.image_height-1)/2)
        p = self.projection_matrix.T.clone()
        p[0,0] = 2*self.Fx/self.image_width
        p[1,1] = 2*self.Fy/self.image_height
        p[0,2] = (2*self.Cx+1)/self.image_width-1
        p[1,2] = (2*self.Cy+1)/self.image_height-1
        self.projection_matrix = p.T.contiguous()
        self.full_proj_transform ='''
    replace(cam,'        self.full_proj_transform =',replacement)
    loader='gaussian_splatting_2d/utils/camera_utils.py'
    replace(loader,'uid=id, data_device=args.data_device)',
        'uid=id, data_device=args.data_device,\n'
        '                  fx=fov2focal(cam_info.FovX,orig_w)*resolution[0]/orig_w,\n'
        '                  fy=fov2focal(cam_info.FovY,orig_h)*resolution[1]/orig_h,\n'
        '                  cx=(cam_info.cx+0.5)*resolution[0]/orig_w-0.5,\n'
        '                  cy=(cam_info.cy+0.5)*resolution[1]/orig_h-0.5)')
    replace(loader,"        'fy' : fov2focal(camera.FovY, camera.height),","        'cx' : camera.cx,\n        'cy' : camera.cy,\n        'fy' : fov2focal(camera.FovY, camera.height),")
    points='gaussian_splatting_2d/utils/point_utils.py'
    text=(OFF/points).read_text();a=text.index('    ndc2pix =');b=text.index('    grid_x, grid_y',a)
    text=text[:a]+'''    fx,fy,cx,cy = view.Fx,view.Fy,view.Cx,view.Cy
    intrins=torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],device='cuda',dtype=torch.float32)

'''+text[b:];(OFF/points).write_text(text);changed.append(points)
    replace('gaussian_splatting_2d/gaussian_renderer/__init__.py',"'surf_depth': surf_depth,","'surf_depth': surf_depth,\n            'depth_expected': render_depth_expected,\n            'depth_median': render_depth_median,")
    gen='gaussian_splatting_2d/scene/point_generator.py'
    replace(gen,'import numpy as np','import numpy as np\nfrom PIL import Image')
    replace(gen,'        xs = xs[valid_mask]','''        # Same native sampling distribution, restricted to the existing RAP foreground.
        if cam_info.mask_path and os.path.exists(cam_info.mask_path):
            fg=np.asarray(Image.open(cam_info.mask_path).convert('L'))>127
            candidates=np.flatnonzero(valid_mask)
            valid_mask[candidates] &= fg[ys[candidates],xs[candidates]]
        xs = xs[valid_mask]''')
    # SuGaR loads camera JSON separately: preserve all per-camera intrinsics as well.
    sugar='sugar_scene/cameras.py'
    replace(sugar,"with open(gs_output_path + 'cameras.json') as f:","with open(os.path.join(gs_output_path, 'cameras.json')) as f:")
    replace(sugar,'image_height=image_height, image_width=image_width,)',
        'image_height=image_height, image_width=image_width,\n'
        '            fx=fx/downscale_factor, fy=fy/downscale_factor,\n'
        '            cx=(camera_transform["cx"]+0.5)/downscale_factor-0.5,\n'
        '            cy=(camera_transform["cy"]+0.5)/downscale_factor-0.5,)')
    replace(sugar,'image_height=None, image_width=None,','image_height=None, image_width=None, fx=None, fy=None, cx=None, cy=None,')
    replace(sugar,'        self.full_proj_transform =',replacement)
    replace(sugar,'    cx = image_width / 2.  # torch.zeros_like(fx).to(device)\n    cy = image_height / 2.  # torch.zeros_like(fy).to(device)',
        '    cx = torch.tensor([c.Cx+0.5 for c in gs_cameras],device=device)\n'
        '    cy = torch.tensor([c.Cy+0.5 for c in gs_cameras],device=device)')
    p=OFF/sugar;text=p.read_text();a=text.index('    image_size = torch.Tensor(',text.index('def convert_camera_from_gs_to_pytorch3d'));b=text.index('    # Extrinsics',a)
    text=text[:a]+'''    image_size=torch.stack([image_width,image_height],dim=1).float()
    scale=image_size.min(dim=1,keepdim=True)[0]/2
    p0_pytorch3d=-(torch.stack([cx,cy],dim=1)-image_size/2)/scale
    focal_pytorch3d=torch.stack([fx,fy],dim=1)/scale
    K=_get_sfm_calibration_matrix(N,device,focal_pytorch3d,p0_pytorch3d,orthographic=False)

'''+text[b:]
    text=text.replace('        self.cx = self.width / 2.  # torch.zeros_like(fx).to(device)\n        self.cy = self.height / 2.  # torch.zeros_like(fy).to(device)',
        '        self.cx = torch.tensor([c.Cx for c in gs_cameras],device=device)\n        self.cy = torch.tensor([c.Cy for c in gs_cameras],device=device)')
    p.write_text(text)
    diffdir=ROOT/'patches';diffdir.mkdir(exist_ok=True)
    details=[]
    for n in changed:
        new=(OFF/n).read_text()
        diff=''.join(difflib.unified_diff(original[n].splitlines(True),new.splitlines(True),fromfile='official/'+n,tofile='foresplat/'+n))
        (diffdir/(n.replace('/','__')+'.diff')).write_text(diff)
        details.append(dict(path=n,before_sha256=hashlib.sha256(original[n].encode()).hexdigest(),after_sha256=hashlib.sha256(new.encode()).hexdigest()))
    data=dict(status='prepared',official_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OFF,text=True).strip(),source=str(OFF),input=str(DATA),frames=208,seed=0,
        modified_sources=details,prior='Metric3Dv2 ViT-giant2; native depth+normal prior, actual intrinsics replace DTU hardcoding; per-image median track depth scale aligns to existing VGGT gauge',
        initial_pointcloud='Original VGGT COLMAP1,252,383 track points plus official generated points filtered by unchanged RAP mask; all loss/densification/clustering native',
        rasterizer='Official 2DGS projective homography preserves noncentral projection without numerical kernel changes',
        training='30K, depth_ratio1, lambda_dist1000, lambda_normal.05, lambda_normal_prior.1, DBSCAN largest-cluster prune at7000; refinement15K on all208',
        read_only_history=True)
    READY.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

