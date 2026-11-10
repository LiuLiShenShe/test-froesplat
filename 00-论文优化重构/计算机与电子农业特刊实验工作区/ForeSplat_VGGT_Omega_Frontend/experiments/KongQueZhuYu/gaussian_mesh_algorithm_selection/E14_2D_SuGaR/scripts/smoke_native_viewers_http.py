"""Sequential real native viewer HTTP smoke on actual E11 surfels and actual refined3D Gaussian; close GPU services before completion."""
import os,sys,json,time,signal,subprocess,socket,base64,io,traceback
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import URLError,HTTPError
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PY=ROOT/'.venv/bin/python';VIEWER=ROOT/'scripts/view_native_web.py'
OUT=ROOT/'native_viewer_http_smoke';HIST=ROOT.parent.parent/'geometry_mesh_final_selection'
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def available_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def listen_open(port):
    with socket.socket() as s:s.settimeout(.3);return s.connect_ex(('127.0.0.1',port))==0
def stop_owned_service(process,port):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
    # The entire dedicated process group is ours; uv may have already exited before its Python child.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    deadline=time.time()+15
    while listen_open(port) and time.time()<deadline:time.sleep(.2)
    return dict(returncode=process.returncode,listener_closed=not listen_open(port),process_reaped=process.poll() is not None)
def smoke(tag,command_tail,model_path,representation):
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True);port=available_port()
    command=['uv','run','--no-project','--python',str(PY),'python',str(VIEWER),*command_tail,'--port',str(port)]
    event=dict(tag=tag,actual_model_path=str(model_path.resolve(strict=True)),representation=representation,command=command,physical_gpu=1,started_unix=time.time())
    write(folder/'actual_command.json',event)
    started=time.time();process=None;record=event.copy();record['status']='failed'
    with (folder/'server.log').open('a') as logfile:
        try:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['PYTHONUNBUFFERED']='1';env['MPLBACKEND']='Agg'
            process=subprocess.Popen(command,cwd=ROOT/'official',env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.time()+300;page=None
            while time.time()<deadline:
                if process.poll() is not None:raise RuntimeError(f'Native service exited before HTTP ready: {process.returncode}')
                try:
                    with urlopen(f'http://127.0.0.1:{port}/',timeout=2) as response:page=response.read();code=response.status
                    if code==200 and page:break
                except (URLError,TimeoutError,ConnectionError):time.sleep(.5)
            if page is None:raise TimeoutError('Native service did not become HTTP ready within300s')
            (folder/'served_page.html').write_bytes(page);views=[]
            for yaw in [0,180]:
                view=folder/f'yaw_{yaw:03d}';view.mkdir(exist_ok=True)
                request_json=json.dumps(dict(yaw=yaw,pitch=0)).encode()
                request=Request(f'http://127.0.0.1:{port}/render',data=request_json,headers={'Content-Type':'application/json'},method='POST')
                with urlopen(request,timeout=120) as response:payload=response.read();http_status=response.status
                assert http_status==200
                (view/'actual_HTTP_response.json').write_bytes(payload);data=json.loads(payload)
                assert Path(data['model']).resolve()==model_path.resolve()
                sizes={};paths={}
                for channel in ['rgb','alpha','depth','normal']:
                    uri=data[channel];assert uri.startswith('data:image/png;base64,')
                    binary=base64.b64decode(uri.split(',',1)[1],validate=True)
                    img=Image.open(io.BytesIO(binary));assert img.size==(1080,1920),img.size
                    filename=channel+('_aux' if representation=='native_refined3DGS' and channel in ['depth','normal'] else '')+'.png'
                    dest=view/filename;dest.write_bytes(binary);sizes[channel]=list(img.size);paths[channel]=str(dest.resolve())
                alpha=np.asarray(Image.open(paths['alpha']).convert('L'),dtype=np.float32)/255
                metadata=dict(status='complete',HTTP_status=http_status,requested_yaw_worldY_degrees=yaw,pitch_degrees=0,origin='Shared foreground bbox origin; camera-only rotation, unchanged model',model=str(model_path.resolve()),representation=representation,response_images=paths,image_sizes=sizes,alpha_png_mean=float(alpha.mean()),alpha_png_nonzero_fraction=float((alpha>0).mean()),depth_label=data['depth_label'],normal_label=data['normal_label'],native_depth_available=representation=='native2DGS',native_normal_available=representation=='native2DGS',refined_aux='Native override-color alpha-normalized center-Z and smallest-scale axis; no native3D surfaceDepth/Normal is claimed' if representation=='native_refined3DGS' else None,metrics='HTTP/display smoke only; PNG values are not used as reconstruction-quality metrics',viewer_note=data['viewer_note'])
                write(view/'response_semantics.json',metadata);views.append(metadata)
                print('Real native viewer HTTP smoke',tag,yaw,paths['rgb'],flush=True)
            record.update(status='complete',HTTP_views=views,elapsed_seconds=time.time()-started)
        except Exception as exc:record.update(reason=str(exc),traceback=traceback.format_exc())
        finally:
            if process is not None:
                record['service_shutdown']=stop_owned_service(process,port)
                if not all(record['service_shutdown'][key] for key in ['listener_closed','process_reaped']):
                    record.update(status='failed',reason='Owned native HTTP/GPU service did not completely stop')
    write(folder/'complete.json',record);return record
def main():
    marker=ROOT/'native_viewer_http_smoke_complete.json'
    if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':return
    assert (ROOT/'refinement_complete.json').exists()
    info=json.loads((ROOT/'refinement_complete.json').read_text());assert info['status']=='complete'
    e11=HIST/'E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
    refined=Path(info['gaussian_ply']);assert e11.is_file() and refined.is_file()
    OUT.mkdir(exist_ok=True);started=time.time();results=[]
    for tag,tail,path,kind in [
        ('E11_actual_external2DGS',['--ply',str(e11),'--label','E11 actual native2DGS'],e11,'native2DGS'),
        ('E14_actual_refined3DGS',['--stage','refined','--label','E14 actual refined3D Gaussian'],refined,'native_refined3DGS')]:
        # This call closes/reaps its service before returning, so the next native GPU renderer is sequential.
        results.append(smoke(tag,tail,path,kind))
    complete=all(result['status']=='complete' for result in results)
    write(marker,dict(status='complete' if complete else 'partial',results=results,elapsed_seconds=time.time()-started,sequential_GPU_services=True,all_services_stopped=all(r.get('service_shutdown',{}).get('listener_closed',False) and r.get('service_shutdown',{}).get('process_reaped',False) for r in results),future_history_queue_may_start_only_after_this_record_is_complete=True))
    if not complete:raise RuntimeError('One or more actual native viewer HTTP smokes failed; actual service logs/results retained')
if __name__=='__main__':main()

