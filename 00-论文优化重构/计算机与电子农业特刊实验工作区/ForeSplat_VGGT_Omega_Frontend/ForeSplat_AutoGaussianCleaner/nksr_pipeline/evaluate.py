"""Four matched views using existing GGGS rendering and mesh raycast metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import gc
import json
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont

from .common import (ROOT, Stage, camera_record, get_camera, guard_memory,
    load_gaussians, load_image, make_scene, read_mesh, save_image, stage_args, write)
from evaluate_meshes import cast, metrics, structure


def run(args):
    out = args.output/'visualizations'
    if (out/'complete.json').exists(): return
    spec = json.loads(args.scene.read_text())
    mesh_report = json.loads((args.output/'mesh/mesh_report.json').read_text())
    diagonal = mesh_report['scale']['diagonal']
    torch.set_num_threads(8)
    with Stage(out, gpu=True) as stage:
        meter = stage.meter
        scene = make_scene(spec, max_side=960)
        old = ROOT/f'experiments/C1_C2_all19/{spec["plant"]}/comparison/paired_quality.json'
        indices = json.loads(old.read_text())['observed_indices']
        camera_records, masks, depths, originals, gggs = [], [], [], [], []
        model,count = load_gaussians(scene,args.input_ply,meter)
        for order,index in enumerate(indices):
            view = scene.views[index];camera = get_camera(scene,index)
            camera_records.append(camera_record(camera))
            target = load_image(view['rgb'],(view['width'],view['height']),meter)
            mask = load_image(view['mask'],(view['width'],view['height']),meter,mask=True)
            with torch.no_grad(): rendered = scene.render(camera,model)
            rgb = rendered['render'].cpu().numpy().transpose(1,2,0)
            alpha = rendered['mask'].cpu().numpy().squeeze()
            depth = rendered['median_depth'].cpu().numpy().squeeze().copy()
            composite = rgb+(1-alpha[...,None])*.5
            save_image(out/f'reference_{order:02}.png',target,meter)
            save_image(out/f'C1_GGGS_{order:02}.png',composite,meter)
            originals.append(target);gggs.append(composite);masks.append(mask);depths.append(depth)
            del rendered,camera
        del model,scene
        torch.cuda.empty_cache();gc.collect()
        results,structures,panels = {},{},{}
        for name in ['raw','post']:
            mesh = read_mesh(args.output/f'mesh/{name}.ply',meter)
            structures[name] = structure(mesh,'C1_NKSR',name,diagonal)
            vertices,faces,colours = np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.asarray(mesh.vertex_colors)
            raycast = o3d.t.geometry.RaycastingScene(nthreads=8)
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            raycast.add_triangles(tensor_mesh)
            rows=[];images=[]
            for order,camera in enumerate(camera_records):
                depth,normal,rgb = cast(raycast,camera,vertices,faces,colours)
                occupied = np.isfinite(depth)&(depth>0)
                image = np.where(occupied[...,None],rgb,.5)
                save_image(out/f'NKSR_{name}_{order:02}.png',image,meter)
                # Face-normal lighting is supplementary evidence for surfaces;
                # the main comparison keeps the recorded observation colours.
                shade = .3+.7*np.abs(normal@np.array([.25,.4,.8817596]))
                shape = np.where(occupied[...,None],np.repeat(shade[...,None],3,axis=2),.5)
                save_image(out/f'NKSR_{name}_geometry_{order:02}.png',shape,meter)
                images.append(image)
                rows.append(dict(index=indices[order],name=camera['name'],
                    **metrics(occupied,masks[order],depth,depths[order],diagonal)))
                print('mesh view',spec['plant'],name,camera['name'],'IoU',rows[-1]['silhouette_iou'],flush=True)
            results[name]=rows;panels[name]=images
            del raycast,tensor_mesh,mesh,vertices,faces,colours
            gc.collect();guard_memory()
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
        width,height=300,580
        canvas=Image.new('RGB',(4*width,4*height),'white');draw=ImageDraw.Draw(canvas)
        for row,(label,images) in enumerate([('Reference RGB',originals),('C1 GGGS',gggs),('NKSR Raw',panels['raw']),('NKSR Post',panels['post'])]):
            for col,array in enumerate(images):
                image=Image.fromarray(np.uint8(np.clip(array,0,1)*255));image.thumbnail((width,height-45))
                canvas.paste(image,(col*width+(width-image.width)//2,row*height+45))
                draw.text((col*width+4,row*height+5),label,fill='black',font=font)
                draw.text((col*width+4,row*height+25),camera_records[col]['name'],fill='black',font=font)
        canvas.save(out/'mesh_comparison.png')
        write(out/'metrics.json',dict(plant=spec['plant'],indices=indices,cameras=camera_records,structures=structures,views=results,
            resolution='Same cameras, corrected half-pixel intrinsics and max side 960 for every row',
            main_render='Observation vertex colours by historical nearest-surface-point transfer, fixed gray 0.5 background; no exposure changes',
            metric_reference='Existing RAP foreground includes pot; rendered C1 median depth is a comparison, not a ground-truth scan',
            limitations='Four observed views; no organ labels. Raycast is opaque first-hit geometry, GGGS uses alpha compositing.'))
    mesh_report['structure']=structures
    write(args.output/'mesh/mesh_report.json',mesh_report)
    write(out/'complete.json',dict(status='complete',plant=spec['plant'],views=4))


if __name__=='__main__':
    run(stage_args(__doc__))
    # Match the historical evaluator: CUDA has synchronized, the monitor is
    # closed and all artifacts have been published. Avoid cold-library
    # Open3D/CUDA interpreter teardown on this existing mixed runtime.
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
