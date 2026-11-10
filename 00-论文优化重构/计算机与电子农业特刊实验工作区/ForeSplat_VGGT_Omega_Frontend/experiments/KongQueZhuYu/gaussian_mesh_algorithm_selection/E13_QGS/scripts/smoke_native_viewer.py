"""Exercise actual native QGS HTTP rendering, then release GPU before next stage."""
from pathlib import Path
import json,time,subprocess,os,io
from urllib.request import urlopen
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'native_viewer/smoke';out.mkdir(parents=True,exist_ok=True)
 done=out/'complete.json';failure=out/'failure.json'
 if done.exists():return
 assert (ROOT/'training_complete.json').is_file()
 command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/native_qgs_viewer.py'),'--iteration','30000','--port','0','--width','270']
 started=time.time();records=[];launch=None;state=None;p=None;status='failed';inspection_record=None
 log=(out/'server.log').open('a')
 try:
  # The two extra top/oblique inspection cameras use the parent's exact protocol.
  # Render them sequentially before starting the native HTTP server; the original
  # eight free-camera comparison outputs and their completion record stay intact.
  inspection_protocol=ROOT.parent/'comparison/shared_inspection_views.json'
  inspection_done=ROOT/'inspection_views/complete.json'
  if inspection_protocol.is_file() and not inspection_done.is_file():
   inspection_started=time.time()
   inspection_command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/render_qgs_free.py'),
    '--protocol',str(inspection_protocol.resolve()),'--out',str(ROOT/'inspection_views'),'--skip-fixed-windows']
   with (out/'inspection.log').open('a') as inspection_log:
    subprocess.run(inspection_command,cwd=ROOT/'adapted',stdout=inspection_log,stderr=subprocess.STDOUT,env=os.environ,check=True)
   assert inspection_done.is_file(),'Actual native top/oblique inspection rendering did not complete'
   inspection_record=dict(command=inspection_command,completion=str(inspection_done.resolve()),elapsed_seconds=time.time()-inspection_started)
  readiness_started=time.time()
  p=subprocess.Popen(command,cwd=ROOT/'adapted',stdout=log,stderr=subprocess.STDOUT,env=os.environ)
  launch_file=ROOT/'native_viewer/server_launch.json'
  while time.time()-readiness_started<180:
   if p.poll() is not None:raise RuntimeError(f'Native viewer exited {p.returncode} before readiness')
   if launch_file.exists():
    candidate=json.loads(launch_file.read_text())
    if candidate.get('pid')==p.pid and candidate.get('status')=='running':launch=candidate;break
   time.sleep(1)
  assert launch is not None,'Actual native viewer readiness timeout'
  base=launch['URL']
  state=json.loads(urlopen(base+'/api/state',timeout=30).read())
  assert state['iteration']==30000 and state['scale_channels']==6 and state['quadratic_primitives']>0
  assert Path(state['native_model']).resolve()==(ROOT/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').resolve()
  (out/'actual_model_state.json').write_text(json.dumps(state,indent=2)+'\n')
  cases=[('rgb','mode=rgb&source=0&yaw=0&pitch=0&zoom=1&flip180=0'),
         ('alpha','mode=alpha&source=0&yaw=0&pitch=0&zoom=1&flip180=0'),
         ('normal','mode=normal&source=0&yaw=0&pitch=0&zoom=1&flip180=0'),
         ('depth','mode=depth&source=0&yaw=0&pitch=0&zoom=1&flip180=0'),
         ('gray','mode=gray&source=0&yaw=0&pitch=0&zoom=1&flip180=0'),
         ('orbit','mode=rgb&source=0&yaw=15&pitch=10&zoom=1&flip180=0'),
         ('zoom','mode=rgb&source=0&yaw=15&pitch=10&zoom=1.25&flip180=0'),
         ('rotate180','mode=rgb&source=0&yaw=0&pitch=0&zoom=1&flip180=1')]
  images={}
  for name,query in cases:
   response=urlopen(base+'/api/render?'+query,timeout=90)
   assert response.status==200 and response.headers['Content-Type']=='image/png'
   payload=response.read();path=out/f'{name}.png';path.write_bytes(payload)
   im=Image.open(io.BytesIO(payload));assert im.size==(270,480)
   arr=np.asarray(im).copy();assert arr.size and np.ptp(arr)>0,(name,'empty/uniform rendered image')
   images[name]=arr
   render_state=json.loads(urlopen(base+'/api/state',timeout=30).read())
   records.append(dict(mode=name,query=query,path=str(path.resolve()),bytes=len(payload),width=270,height=480,
     rendered_pixel_min=int(arr.min()),rendered_pixel_max=int(arr.max()),actual_native_render_validation=render_state.get('last_render')))
  assert images['alpha'].max()>0
  assert np.mean(abs(images['rgb'].astype(float)-images['orbit']))>1e-3
  assert np.mean(abs(images['orbit'].astype(float)-images['zoom']))>1e-3
  assert np.mean(abs(images['rgb'].astype(float)-images['rotate180']))>1e-3
  status='passed'
 except Exception as e:
  failure.write_text(json.dumps(dict(status='failed',reason=type(e).__name__+': '+str(e),command=command,actual_model_state=state,
   completed_responses=records,inspection=inspection_record,server_log=str((out/'server.log').resolve())),indent=2)+'\n')
  raise
 finally:
  if p is not None and p.poll() is None:
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  log.close()
  if launch is not None:
   launch.update(status='stopped',reason='Native model HTTP smoke finished; GPU released before next experiment',stopped_time=time.time(),
    test_result=status,no_server_left_running=True)
   (ROOT/'native_viewer/server_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 if status=='passed':
  done.write_text(json.dumps(dict(status='complete',native_HTTP_rendering_verified=True,native_model=state['native_model'],
    primitive_count=state['quadratic_primitives'],iteration=30000,viewer_command=command,views=records,elapsed_seconds=time.time()-started,
    inspection=inspection_record,native_HTTP_elapsed_seconds=time.time()-readiness_started,
    actual_server_terminated=True,no_competing_training=True,normal_coordinates='normalized native camera-direction channel',
    rotate180='inverse-camera view transform about shared plant center worldX; no geometry modification',physical_gpu=1),indent=2)+'\n')
  print('Actual native QGS HTTP/render/orbit/zoom/180 viewer smoke passed; server terminated',flush=True)
if __name__=='__main__':main()

