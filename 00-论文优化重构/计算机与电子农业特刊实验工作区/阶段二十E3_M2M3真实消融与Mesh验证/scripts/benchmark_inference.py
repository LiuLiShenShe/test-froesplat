"""Same-GPU native-render timing on identical frozen held-out cameras."""
from common import *
from argparse import Namespace
import gc,time,statistics

def main():
    gpu_idle();verify_frozen();import_paths()
    import torch
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    values=[];summary=[];start=time.perf_counter()
    with torch.no_grad():
        for sample in SCENES:
            c=config(sample);args=Namespace(**{**c['model'],'model_path':str(model_path(sample,'M0'))})
            initial=GaussianModel(3);scene=Scene(args,initial,load_iteration=30000,shuffle=False)
            cams=sorted(scene.getTestCameras(),key=lambda x:x.image_name)
            expected={r['loader_image_name'] for r in rows(STAGE/f'metrics/{sample}_M0_FULL.csv')}
            assert {x.image_name for x in cams}==expected
            del scene,initial;gc.collect();torch.cuda.empty_cache()
            for method in ['M0','M1','M2']:
                path=model_path(sample,method)/'point_cloud/iteration_30000/point_cloud.ply'
                expected_sha=reference(sample)['PLY_sha256'] if method=='M0' else read_json(STAGE/f'reports/{sample}_{method}_TRAIN_RUN.json')['PLY_sha256']
                assert sha(path)==expected_sha
                g=GaussianModel(3);g.load_ply(str(path));background=torch.zeros(3,device='cuda')
                for _ in range(3):
                    package=render(cams[0],g,Namespace(**c['pipeline']),background);torch.cuda.synchronize();del package
                torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();own=[]
                for repeat in [1,2]:
                    for cam in cams:
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();mark=time.perf_counter();a.record()
                        package=render(cam,g,Namespace(**c['pipeline']),background)
                        b.record();torch.cuda.synchronize();wall=time.perf_counter()-mark
                        assert torch.isfinite(package['render']).all()
                        row={'sample':sample,'method':method,'seed':0,'repeat':repeat,'camera':cam.image_name,'width':cam.image_width,'height':cam.image_height,
                             'CUDA_event_ms':a.elapsed_time(b),'synchronized_wall_seconds':wall,'Gaussian_count':len(g.get_xyz),'GPU_UUID':GPU_UUID,'PLY_sha256':expected_sha}
                        own.append(row);values.append(row);del package
                summary.append({'sample':sample,'method':method,'seed':0,'measured_views':len(cams),'passes':2,'warmup_same_first_camera':3,
                                'CUDA_event_mean_ms':statistics.mean(r['CUDA_event_ms'] for r in own),'CUDA_event_median_ms':statistics.median(r['CUDA_event_ms'] for r in own),
                                'synchronized_wall_mean_seconds':statistics.mean(r['synchronized_wall_seconds'] for r in own),
                                'base_CUDA_allocated_bytes':base,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
                                'Gaussian_count':len(g.get_xyz),'GPU_UUID':GPU_UUID,'PLY_sha256':expected_sha,
                                'scope':'Native renderer RGB/alpha/depth/normal package; excludes model loading, GT metrics and PNG/float serialization',
                                'limitation':'Same physical GPU, fixed M0/M1/M2 order, one session; shared host load and thermal variation uncontrolled; descriptive pilot, no robust speed claim'})
                write_csv(STAGE/'M2M3_RENDER_TIMING_PER_VIEW.csv',values);write_csv(STAGE/'M2M3_RENDER_TIMING_SUMMARY.csv',summary)
                print('BENCHMARKED',sample,method,summary[-1]['CUDA_event_mean_ms'],flush=True)
                del g,background;gc.collect();torch.cuda.empty_cache()
            del cams;gc.collect();torch.cuda.empty_cache()
    write_json(STAGE/'reports/RENDER_TIMING.json',{'status':'PASS','rows':len(values),'models':len(summary),'GPU_UUID':GPU_UUID,'wall_seconds':time.perf_counter()-start,
                 'script_sha256':sha(__file__),'procedure':'Three warmups per method, two full frozen-heldout passes, synchronized CUDA events plus wall time'})
if __name__=='__main__':main()
