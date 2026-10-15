"""Recompute reported image/proxy metrics from every immutable float tensor."""
from __future__ import annotations
import math
import time
from common import *
from protocol import assert_gpu,tensor_sha
from metric_core import resize_reference,load_roi,foreground_metrics,background_geometry_metrics


def assert_shared_hongzhang_alias(left,right,foreground_rows,background_rows):
    rows=(left,right)
    identities={(row['sample'],row['group'],row['seed']) for row in rows}
    assert identities=={('HongZhang','B_R0','0'),('HongZhang','C_R1','0')}
    by_group={row['group']:row for row in rows}
    assert by_group['B_R0']['shared_model_alias']=='HongZhang/C_R1/seed0'
    assert by_group['C_R1']['shared_model_alias']==''
    assert left['loader_image_name']==right['loader_image_name']

    ignored={'group','shared_model_alias'}
    assert ({key:value for key,value in left.items() if key not in ignored}
            =={key:value for key,value in right.items() if key not in ignored})
    for table in (foreground_rows,background_rows):
        left_key=(left['sample'],left['group'],left['seed'],left['loader_image_name'])
        right_key=(right['sample'],right['group'],right['seed'],right['loader_image_name'])
        a,b=table[left_key],table[right_key]
        assert ({key:value for key,value in a.items() if key not in ignored}
                =={key:value for key,value in b.items() if key not in ignored})


def main():
    import_paths();assert_gpu()
    import torch
    import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    full=read_csv(STAGE/'MULTISEED_PER_VIEW_METRICS.csv')
    fg={(r['sample'],r['group'],r['seed'],r['loader_image_name']):r for r in read_csv(STAGE/'FOREGROUND_PROXY_METRICS.csv')}
    bg={(r['sample'],r['group'],r['seed'],r['loader_image_name']):r for r in read_csv(STAGE/'BACKGROUND_GEOMETRY_PROXY_METRICS.csv')}
    perceptual=lpips.LPIPS(net='vgg',version='0.1',verbose=False).eval().cuda()
    checked=[];failures=[];seen={};start=time.perf_counter()
    with torch.no_grad():
        for row in full:
            key=(row['sample'],row['group'],row['seed'],row['loader_image_name'])
            path=Path(row['float_render_path'])
            if str(path) in seen:
                original=seen[str(path)]
                assert_shared_hongzhang_alias(original,row,fg,bg)
                continue
            seen[str(path)]=row
            assert sha256(path)==row['float_render_sha256']
            assert sha256(row['reference_path'])==row['reference_original_sha256']
            assert sha256(row['ROI_path'])==row['ROI_original_sha256']
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            width,height=int(row['render_width']),int(row['render_height'])
            reference_cpu=resize_reference(row['reference_path'],width,height)
            roi_cpu=load_roi(row['ROI_path'],width,height)
            assert tensor_sha(reference_cpu)==row['reference_resized_tensor_sha256']
            assert tensor_sha(roi_cpu)==row['ROI_resized_tensor_sha256']
            reference=reference_cpu.cuda();roi=roi_cpu.cuda()
            rgb=tensors['rgb'].cuda();alpha=tensors['alpha'].cuda()
            depth=tensors['depth'].cuda();normal=tensors['normal'].cuda();surface_normal=tensors['surface_normal'].cuda()
            primary={'PSNR':psnr(rgb[None],reference[None]).mean().item(),'SSIM':ssim(rgb,reference).item(),
                     'LPIPS':perceptual(rgb[None],reference[None],normalize=True).item()}
            foreground=foreground_metrics(rgb,reference,alpha,roi,ssim,perceptual)
            background=background_geometry_metrics(rgb,alpha,roi,depth,normal,surface_normal)
            if int(row['seed'])==0:
                coherent=background_geometry_metrics(rgb,tensors['geometry_replay_alpha'].cuda(),roi,depth,normal,surface_normal)
                for field in coherent:
                    if field.startswith(('depth_','normal_')):background[field]=coherent[field]
            comparisons=0;max_abs=0.;view_fail=[]
            for kind,actual,reported in [('FULL',primary,row),('FOREGROUND',foreground,fg[key]),('BACKGROUND_GEOMETRY',background,bg[key])]:
                for metric,value in actual.items():
                    if value is None:
                        if reported.get(metric,'')!='':view_fail.append({'kind':kind,'metric':metric,'reason':'undefined must be blank'})
                        continue
                    if isinstance(value,(str,bool)):
                        if str(value)!=reported.get(metric):view_fail.append({'kind':kind,'metric':metric,'actual':str(value),'reported':reported.get(metric)})
                        continue
                    expected=float(reported[metric]);comparisons+=1
                    error=abs(float(value)-expected);max_abs=max(max_abs,error)
                    if not math.isclose(float(value),expected,rel_tol=1e-6,abs_tol=1e-6):
                        view_fail.append({'kind':kind,'metric':metric,'actual':float(value),'reported':expected,'abs_error':error})
            checked.append({'sample':row['sample'],'group':row['group'],'seed':row['seed'],'loader_image_name':row['loader_image_name'],
                            'status':'FAIL' if view_fail else 'PASS','metrics_compared':comparisons,'max_abs_error':max_abs,
                            'float_render_path':str(path),'float_render_sha256':row['float_render_sha256'],
                            'reference_tensor_sha256':row['reference_resized_tensor_sha256'],'ROI_tensor_sha256':row['ROI_resized_tensor_sha256']})
            failures.extend({'identity':key,**failure} for failure in view_fail)
            print(f'RECOMPUTED {key} {checked[-1]["status"]} max_abs_error={max_abs:.9g}',flush=True)
            del tensors,reference,roi,rgb,alpha,depth,normal,surface_normal
    write_csv(STAGE/'reports/FLOAT_RENDER_METRIC_RECOMPUTATION.csv',checked)
    result={'status':'PASS' if not failures else 'FAIL','timestamp':now(),'unique_float_renders':len(checked),'reported_rows_including_shared_alias':len(full),
            'numeric_metric_comparisons':sum(r['metrics_compared'] for r in checked),'max_abs_error':max(r['max_abs_error'] for r in checked),
            'failures':failures,'wall_seconds':time.perf_counter()-start,'GPU_UUID':GPU_UUID,
            'reference_and_ROI_hashes_verified':True,'raw_png_screenshots_used':False,'script_sha256':sha256(Path(__file__))}
    write_json(STAGE/'reports/FLOAT_RENDER_METRIC_RECOMPUTATION.json',result)
    assert result['status']=='PASS',failures[:3]
    print('FLOAT_METRIC_RECOMPUTATION_PASS',len(checked),flush=True)

if __name__=='__main__':main()
