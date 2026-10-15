"""Recompute every reported common-reference metric from retained real floats."""
from common import *
import math,time

def main():
    gpu_idle();verify_frozen();import_paths()
    import torch,lpips
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    core=metric_module();perceptual=lpips.LPIPS(net='vgg',version='0.1',verbose=False).eval().cuda()
    full=rows(STAGE/'M2M3_HELDOUT_METRICS.csv')
    identity=lambda r:(r['sample'],r['method'],r['loader_image_name'])
    fg={identity(r):r for r in rows(STAGE/'M2M3_FOREGROUND_PROXY_METRICS.csv')}
    bg={identity(r):r for r in rows(STAGE/'M2M3_BACKGROUND_GEOMETRY_PROXY_METRICS.csv')}
    result=[];started=time.perf_counter()
    with torch.no_grad():
        for r in full:
            assert sha(r['float_render_path'])==r['float_render_sha256']
            assert sha(r['reference_path'])==r['reference_original_sha256'] and sha(r['ROI_path'])==r['ROI_original_sha256']
            w,h=int(r['render_width']),int(r['render_height'])
            ref_cpu=core.resize_reference(r['reference_path'],w,h);roi_cpu=core.load_roi(r['ROI_path'],w,h)
            assert tensor_sha(ref_cpu)==r['reference_resized_tensor_sha256']
            assert tensor_sha(roi_cpu)==r['ROI_resized_tensor_sha256']
            data=torch.load(r['float_render_path'],map_location='cuda',weights_only=True)
            assert all(torch.isfinite(t).all() for t in data.values() if isinstance(t,torch.Tensor))
            rgb=data['rgb'];alpha=data['alpha'];ref=ref_cpu.cuda();roi=roi_cpu.cuda()
            values={'PSNR':psnr(rgb[None],ref[None]).mean().item(),'SSIM':ssim(rgb,ref).item(),
                    'LPIPS':perceptual(rgb[None],ref[None],normalize=True).item()}
            foreground=core.foreground_metrics(rgb,ref,alpha,roi,ssim,perceptual)
            background=core.background_geometry_metrics(rgb,alpha,roi,data['depth'],data['normal'],data['surface_normal'])
            coherent=core.background_geometry_metrics(rgb,data['geometry_replay_alpha'],roi,data['depth'],data['normal'],data['surface_normal'])
            for key in coherent:
                if key.startswith(('depth_','normal_')):background[key]=coherent[key]
            differences={};passed=True
            for family,computed,expected in [('FULL',values,r),('FG',foreground,fg[identity(r)]),('BG',background,bg[identity(r)])]:
                for key,value in computed.items():
                    if value is None or isinstance(value,(str,bool)):continue
                    assert expected.get(key) not in [None,'','None'],(identity(r),family,key)
                    error=abs(float(value)-float(expected[key]));differences[f'{family}_{key}_absolute_error']=error
                    # Same device/version should reproduce closely; permit parallel-reduction roundoff only.
                    tolerance=2e-5+2e-6*abs(float(expected[key]))
                    if key.endswith('pixels') or key in ['foreground_channel_samples','foreground_pixels','outside_alpha_components_ge16px']:tolerance=0
                    passed &= error<=tolerance
            result.append({'sample':r['sample'],'method':r['method'],'camera':r['loader_image_name'],
                           'float_render_sha256':r['float_render_sha256'],'status':'PASS' if passed else 'FAIL',**differences})
            write_csv(STAGE/'reports/FLOAT_METRIC_RECOMPUTATION.csv',result)
            assert passed,(identity(r),differences)
            print('RECOMPUTED',*identity(r),flush=True)
    write_json(STAGE/'reports/FLOAT_METRIC_RECOMPUTATION.json',{'status':'PASS','views':len(result),'wall_seconds':time.perf_counter()-started,
               'families':['full raw RGB','fixed R0 foreground proxy','background/normal/depth proxy'],
               'tolerance':'Absolute2e-5 + relative2e-6, exact integer pixel/component counts','GPU_UUID':GPU_UUID,
               'ROI_is_independent_GT':False,'source_sha256':sha(__file__)})

if __name__=='__main__':main()
