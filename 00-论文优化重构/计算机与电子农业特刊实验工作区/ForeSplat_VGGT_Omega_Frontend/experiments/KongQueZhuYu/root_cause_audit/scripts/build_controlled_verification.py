"""Join measured control metrics; empty Gaussian fields mean not measured."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['control','model','scope','views','gaussian_before','gaussian_after',
        'PSNR_before','PSNR_after','FG_PSNR_before','FG_PSNR_after','Boundary_PSNR_before','Boundary_PSNR_after',
        'SSIM_before','SSIM_after','LPIPS_before','LPIPS_after','FG_gradient_before','FG_gradient_after',
        'Sampson_shared_px_before','Sampson_shared_px_after','Depth_RGB_shared_px_before','Depth_RGB_shared_px_after',
        'Depth_support_5pct_before','Depth_support_5pct_after','source','limitation']

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    rows=[]
    perceptual={r['experiment']:r for r in read_csv(ROOT/'controlled_fixes/prune_effective_render/perceptual_cpu_supplement/summary_perceptual_metrics.csv')}
    for model in ['E1','E5']:
        relative=f'controlled_fixes/prune_effective_render/{model}/evaluation/comparison.json'
        data=json.loads((ROOT/relative).read_text());before,after=data['summaries'];p=perceptual[model]
        r=dict(control='native_effective_alpha_below_0.05_deletion',model=model,scope='fixed existing Gaussian checkpoint',views=3,
               gaussian_before=data['retained_parameter_selection']['before_count'],gaussian_after=data['retained_parameter_selection']['after_count'],
               SSIM_before=p['ssim_baseline_reused'],SSIM_after=p['ssim_effective_pruned'],
               LPIPS_before=p['lpips_alex_baseline_reused'],LPIPS_after=p['lpips_alex_effective_pruned'],
               source=relative+'; controlled_fixes/prune_effective_render/perceptual_cpu_supplement/summary_perceptual_metrics.csv',
               limitation='Same three source0/95/189 cameras/RGB/Mask/retained parameters; no optimizer/retraining. LPIPS baseline GPU versus after CPU tiny numerical variation possible.')
        for label,key in [('PSNR','full_psnr'),('FG_PSNR','fg_mask_psnr'),('Boundary_PSNR','boundary_band5_psnr'),('FG_gradient','foreground_gradient_retention')]:
            r[label+'_before']=before[key];r[label+'_after']=after[key]
        rows.append(r)
    for folder,control,suffixes in [('background_control_v1','geometry_RGB_mask_exterior_restore',('RAP_baseline_24','background_restored_24')),
                                  ('view_count_control_v1','input_view_count_70_to_110',('70','110'))]:
        relative=f'controlled_fixes/{folder}/evaluation/control_summary.csv'
        table={r['condition']:r for r in read_csv(ROOT/relative) if r['control']=='mask_finite' and r['common_rank_offset']=='1'}
        for model in ['Omega','Standard']:
            before=table[model+'_'+suffixes[0]];after=table[model+'_'+suffixes[1]]
            r=dict(control=control,model=model,scope='geometry-only forward control, no Gaussian training',
                   views=24 if folder.startswith('background') else '70->110,38shared',source=relative,
                   limitation='Gaussian/PSNR/SSIM/LPIPS/gradient fields empty: no corrected Gaussian exists. Same source pairs/matches; shared384x688 pixels.24background or current70/110 only.')
            for label,key in [('Sampson_shared_px','sampson_pair_p50_px'),('Depth_RGB_shared_px','depth_to_matched_RGB_pair_p50_px'),('Depth_support_5pct','depth_support_5pct_all_queries')]:
                r[label+'_before']=before[key];r[label+'_after']=after[key]
            rows.append(r)
    with (ROOT/'controlled_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    print(f'Joined {len(rows)} measured control rows; no missing metrics filled with zero')

if __name__=='__main__':main()
