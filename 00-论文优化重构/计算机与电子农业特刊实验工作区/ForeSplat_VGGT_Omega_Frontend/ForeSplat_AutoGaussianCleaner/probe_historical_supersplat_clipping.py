#!/usr/bin/env python3
"""Diagnose official viewer pose/SH/antialias differences without editing any PLY."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json
import sys
import threading
import time
from urllib.parse import urlsplit
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.audit_supersplat_browser import vertex_count
from playwright.sync_api import sync_playwright


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--editor-source',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['B1_historical_best','B3_stage2','B2_stage1','B0_original'])
    a=p.parse_args();root=a.experiment.resolve();source=a.editor_source.resolve()
    view_protocol=json.loads((root/'stage2_residual/visualizations/automatic_free_views.json').read_text())
    definitions=[v for v in view_protocol['definitions'] if v['name'] in ['top','orbit_03']]
    output=root/'comparison/official_clip_probe';output.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(source/'dist')))
    threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_port
    results=dict(editor=str(source),editor_version=json.loads((source/'package.json').read_text())['version'],
        actual_official_editor=True,models=[],view_protocol=view_protocol.get('protocol','automatic_depth_trajectory_v1'),
        model_coordinate_transform='Official PLY importer applies Z180; camera position and target transformed by diag(-1,-1,1)',
        blocked_requests=[],errors=[],started=time.time(),visual_review='pending_direct_image_inspection')
    previous=output/'REVIEW.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old.get('view_protocol')==results['view_protocol'] and old.get('model_coordinate_transform')==results['model_coordinate_transform']:
            results['models']=[m for m in old.get('models',[]) if m['label'] not in a.models
                and Path(m['ply']).is_file() and sha(m['ply'])==m['sha256']]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
                args=['--use-angle=swiftshader','--use-vulkan=swiftshader','--enable-unsafe-webgpu',
                    '--enable-unsafe-swiftshader','--enable-features=Vulkan','--disable-accelerated-video-decode'])
            context=browser.new_context(viewport={'width':1440,'height':1080},device_scale_factor=1,service_workers='block')
            def route(r):
                u=urlsplit(r.request.url)
                if u.hostname=='127.0.0.1' and u.port==port and r.request.method in ['GET','HEAD']:r.continue_()
                else:results['blocked_requests'].append(r.request.url);r.abort()
            context.route('**/*',route)
            context.add_init_script("""(() => {
                Object.defineProperty(window,'showOpenFilePicker',{value:undefined,configurable:true});
                window.audit_adapters=[];
                const request=navigator.gpu.requestAdapter.bind(navigator.gpu);
                navigator.gpu.requestAdapter=async(options={})=>{
                    const adapter=await request({...options,forceFallbackAdapter:true});
                    window.audit_adapters.push(adapter?.info?.architecture);return adapter;
                };
            })();""")
            page=context.new_page();page.on('pageerror',lambda e:results['errors'].append(str(e)))
            page.goto(f'http://127.0.0.1:{port}/',wait_until='domcontentloaded',timeout=60000)
            page.wait_for_function('window.scene && document.querySelector("#file-selector")',timeout=60000)
            results['adapters']=page.evaluate('window.audit_adapters')
            if not any(s and ('swiftshader' in s.lower() or 'software' in s.lower()) for s in results['adapters']):
                raise ValueError('Only verified CPU software WebGPU permitted for browser review')
            for label in a.models:
                path=root/'comparison/models'/f'{label}_supersplat.ply'
                expected=vertex_count(path);count=page.locator('#status-bar .status-bar-stat-value').first
                page.evaluate('window.scene.events.fire("scene.clear")')
                page.wait_for_timeout(300)
                page.locator('#file-selector').set_input_files(str(path),timeout=60000)
                page.wait_for_function("expected=>Number(document.querySelector('#status-bar .status-bar-stat-value')?.textContent.replace(/[^0-9]/g,''))===expected",arg=expected,timeout=240000)
                page.wait_for_timeout(1200)
                page.evaluate("""() => {
                    const s=window.scene;
                    s.events.fire('grid.setVisible',false);
                    s.events.fire('camera.setBound',false);
                    s.events.fire('select.none');
                    const splats=s.elements.filter(e=>e.entity && e.asset);
                    window.review_rotations=splats.map(e=>({name:e.name,rotation:e.entity.getLocalEulerAngles().toString()}));
                }""")
                record=dict(label=label,ply=str(path.resolve()),sha256=sha(path),count=expected,
                    displayed=count.inner_text(),import_rotation=page.evaluate('window.review_rotations'),captures=[])
                for v in definitions:
                    E=np.asarray(v['E']);K=np.asarray(v['K']);center=np.linalg.inv(E)[:3,3]
                    target=center+E[2,:3]
                    # Official editor auto-imports PLY with a180° Z rotation.
                    tr=np.array([-1.,-1.,1.]);center*=tr;target*=tr
                    # Preserve the source vertical FoV despite the editor's
                    # different viewport aspect and horizontal-FoV convention.
                    viewport=page.evaluate("({aspect:window.scene.camera.camera.aspectRatio,horizontal:window.scene.camera.camera.horizontalFov})")
                    tangent=v['height']/(2*K[1,1])
                    fov=float(np.degrees(2*np.arctan(tangent*(viewport['aspect'] if viewport['horizontal'] else 1))))
                    for name,aa,exact,sh in [('clip_001','off',True,3),('clip_02','off',True,3)]:
                        value=1
                        from scipy.spatial.transform import Rotation
                        orientation=np.diag(tr)@E[:3,:3].T@np.diag([1.,-1.,-1.])
                        quaternion=Rotation.from_matrix(orientation).as_quat().tolist()
                        page.evaluate("""r=>{
                            const s=window.scene,c=s.camera,V=c.position.constructor;
                            const C=s.events.invoke('bgClr').constructor;
                            s.events.fire('setBgClr',new C(r.background,r.background,r.background,1));
                            c.fov=r.fov;c.setPose(new V(...r.center),new V(...r.target),0);
                            s.events.fire('view.setAntiAlias',r.aa);s.events.fire('view.setBands',r.sh);
                            if(r.exact){const Q=c.mainCamera.getRotation().constructor;c.setPoseOverride({position:new V(...r.center),rotation:new Q(...r.quaternion),fov:r.fov,near:r.near,far:100});}else{c.setPoseOverride(null);}
                            s.forceRender=true;
                        }""",dict(center=center.tolist(),target=target.tolist(),fov=fov,background=value,aa=aa,exact=exact,sh=sh,quaternion=quaternion,near=.2 if name=='clip_02' else .01))
                        page.wait_for_timeout(1100)
                        dest=output/label/f'{v["name"]}_{name}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                        page.screenshot(path=str(dest))
                        record['captures'].append(dict(view=v['name'],background=name,path=str(dest),
                            actual_pose=page.evaluate('window.scene.events.invoke("camera.getPose")'),anti_alias=aa,exact_orientation=exact,sh_bands=sh))
                results['models'].append(record)
                write_json(output/'REVIEW.json',results)
                print('Official SuperSplat captured:',label,expected,flush=True)
            results['status']='actual_models_matched_camera_captures_complete'
            browser.close()
    except Exception as e:
        results.update(status='failed',error=repr(e));raise
    finally:
        server.shutdown();server.server_close();results['seconds']=time.time()-results['started']
        write_json(output/'REVIEW.json',results)


if __name__=='__main__':main()
