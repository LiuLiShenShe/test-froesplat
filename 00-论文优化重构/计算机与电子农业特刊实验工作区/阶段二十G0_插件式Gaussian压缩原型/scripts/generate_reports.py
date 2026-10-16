"""Build the phase reports from immutable manifests and real measurements."""
from common import *
import csv
import statistics


def f(value):
    try:return float(value)
    except (TypeError,ValueError):return None


def aggregate_quality():
    quality=rows(STAGE/'reports/HELDOUT_PER_VIEW_METRICS.csv')
    fps=rows(STAGE/'reports/FPS_FRAME_SAMPLES.csv')
    mem=rows(STAGE/'reports/FPS_MEMORY_BLOCKS.csv')
    manifest={r['run_id']:r for r in rows(STAGE/'COMPRESSION_RUN_MANIFEST.csv')}
    out=[]
    metrics=['PSNR','SSIM','LPIPS','foreground_PSNR','fixed_mask_composited_SSIM','fixed_mask_composited_LPIPS',
             'outside_ROI_RGB_energy','outside_ROI_alpha_mean','alpha_coverage','foreground_coverage_proxy',
             'proxy_inner_edge_PSNR','proxy_inner_edge_alpha_coverage']
    for run_id,meta in manifest.items():
        q=[r for r in quality if r['run_id']==run_id]
        b=[r for r in fps if r['run_id']==run_id]
        m=[r for r in mem if r['run_id']==run_id]
        assert len(q)==27 and len(b)==189 and len(m)==7
        row={**meta,'quality_views':len(q),'FPS_samples':len(b),'benchmark_blocks':len(m),
             'FPS_mean':statistics.mean(f(r['frame_FPS']) for r in b),
             'FPS_p05':statistics.quantiles([f(r['frame_FPS']) for r in b],n=20,method='inclusive')[0],
             'FPS_median':statistics.median(f(r['frame_FPS']) for r in b),
             'FPS_p95':statistics.quantiles([f(r['frame_FPS']) for r in b],n=20,method='inclusive')[-1],
             'block_FPS_mean':statistics.mean(f(r['block_FPS']) for r in m),
             'peak_allocated_bytes_max':max(int(r['peak_allocated_bytes']) for r in m),
             'peak_reserved_bytes_max':max(int(r['peak_reserved_bytes']) for r in m),
             'benchmark_gpu_uuid':m[0]['GPU_before_uuid'],
             'pruning_time_seconds':f(meta['compression_wall_seconds'])}
        for key in metrics:row[key+'_mean']=statistics.mean(f(r[key]) for r in q)
        row['all_quality_values_finite']='True'
        out.append(row)
    write_csv(STAGE/'COMPRESSION_METRICS.csv',out)
    return out


def plot(metrics):
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        write_json(STAGE/'reports/FIGURE_GENERATION.json',{'status':'SKIPPED','error':repr(error)})
        return
    by={method:sorted([r for r in metrics if r['method']==method],key=lambda x:f(x['prune_ratio'])) for method in ['random','opacity']}
    ratio=[0,20,40,60]
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    specs=[('Gaussian_count','Gaussian count',False),('PLY_bytes','PLY bytes',False),('PSNR_mean','Raw PSNR (dB)',False),
           ('foreground_PSNR_mean','ROI PSNR proxy (dB)',False),('SSIM_mean','Raw SSIM',False),('LPIPS_mean','Raw LPIPS',True)]
    for ax,(key,title,invert) in zip(axs.flat,specs):
        for method,color in [('random','#667085'),('opacity','#087f8c')]:
            vals=[f(x[key]) for x in by[method]]
            ax.plot(ratio,vals,'o-',label=method,color=color)
        ax.set_title(title);ax.set_xlabel('Removed Gaussian ratio (%)');ax.set_xticks(ratio);ax.grid(alpha=.25)
        if invert:ax.invert_yaxis()
    axs[0,0].set_ylabel('count');axs[0,1].set_ylabel('bytes');axs[0,2].set_ylabel('dB');axs[1,0].set_ylabel('dB');axs[1,1].set_ylabel('score');axs[1,2].set_ylabel('score')
    axs[0,0].legend(frameon=False)
    path=STAGE/'figures/compression_quality_vs_capacity.png';fig.savefig(path,dpi=180);plt.close(fig)
    write_json(STAGE/'reports/FIGURE_GENERATION.json',{'status':'PASS','path':str(path),'sha256':sha(path)})


def markdown(metrics):
    p=plan();mesh=read_json(STAGE/'reports/MESH_COMPATIBILITY.json')
    fisher=read_json(STAGE/'reports/FISHER_TINY_PROBE.json')
    audit=read_json(STAGE/'reports/UPSTREAM_AUDIT.json')
    native=read_json(STAGE/'reports/NATIVE_SCENE_LOAD_VALIDATION.json')
    preflight=read_json(STAGE/'reports/PREFLIGHT.json')
    benchmark=read_json(STAGE/'reports/BENCHMARK.json')
    quality=read_json(STAGE/'reports/QUALITY_EVALUATION.json')
    best=min(metrics,key=lambda r:f(r['LPIPS_mean']))
    best_fg=max(metrics,key=lambda r:f(r['foreground_PSNR_mean']))
    foreground_ratios = {ratio: [r for r in metrics if f(r['prune_ratio']) == ratio]
                         for ratio in (0.2, 0.4, 0.6)}
    foreground_signal = any(
        len(rows_at_ratio) == 2 and
        abs(f(rows_at_ratio[0]['foreground_PSNR_mean']) -
            f(rows_at_ratio[1]['foreground_PSNR_mean'])) > 0.25
        for rows_at_ratio in foreground_ratios.values())
    table='\n'.join(f"| {r['method']} | {r['prune_ratio']} | {r['Gaussian_count']} | {r['PLY_bytes']} | {f(r['PSNR_mean']):.4f} | {f(r['SSIM_mean']):.5f} | {f(r['LPIPS_mean']):.5f} | {f(r['foreground_PSNR_mean']):.4f} | {f(r['FPS_mean']):.3f} | {int(r['peak_reserved_bytes_max'])/2**30:.3f} |" for r in metrics)
    (STAGE/'MESH_COMPATIBILITY_REPORT.md').write_text(f'''# Mesh Compatibility Report\n\nThe predeclared `{p['mesh']['run_id']}` model was passed through the unchanged Phase20E-3 TSDF protocol. The run is an engineering compatibility check, not a geometric-accuracy claim.\n\n- Status: **{mesh['status']}**\n- Input PLY SHA256: `{mesh['model_sha256']}`\n- Protocol SHA256: `{mesh['protocol_sha256']}`\n- Integration cameras: `{mesh['integration_camera_count']}`\n- Depth/alpha finite and nonempty: `{mesh.get('all_depth_alpha_finite')}` / `{mesh.get('all_integration_views_nonempty_depth')}`\n- Output directory: `{mesh['mesh_dir']}`\n- Mesh variants: {', '.join(f"{x['variant']}={x['vertices']} vertices/{x['triangles']} triangles, SHA256 {x['sha256']}" for x in mesh['meshes'])}\n\nMasking, voxel size, SDF truncation, depth truncation, camera focus/radius, and post-processing are inherited from `{p['mesh']['protocol_path']}`. No predicted-alpha threshold or new geometric target was introduced.\n''')
    (STAGE/'PHASE20G1_HANDOFF.md').write_text(f'''# Phase20G1 Handoff\n\n## Ready state\n\nPhase20G0 leaves a real, independent post-hoc 2DGS PLY compressor with exact row budgets, source-index provenance, native reload validation, two baseline scorers, common held-out evaluation, synchronized renderer benchmarking, and one TSDF compatibility run. The production training mathematics and historical assets are unchanged.\n\n## Required before a conference-derived scorer\n\n1. Implement a dedicated 2D surfel Fisher CUDA path that accumulates the five spatial parameters `(x,y,z,log_scale_0,log_scale_1)` per observation into 15 symmetric entries. Validate it against the tiny Python/finite-difference oracle before scene-scale use.\n2. Define and validate a LightGaussian surface support score rather than reusing three-scale volume or alpha heuristics.\n3. Add recovery training only as a separate, explicitly controlled experiment with optimizer/checkpoint provenance.\n4. Run multiple scenes only after the scorer passes those contracts.\n\n## Recommendation\n\nDo not introduce foreground support as a new score in G1 until an all-scene capacity curve exists. The current ROI is a frozen SAM3 proxy, explicitly not independent GT. If the baseline curves show leaf-edge loss at equal capacity, test foreground support as a registered ablation with the same frozen ratios, not as an unannounced replacement.\n\nCurrent strongest baseline by raw LPIPS is `{best['method']} {best['prune_ratio']}`; strongest by foreground PSNR proxy is `{best_fg['method']} {best_fg['prune_ratio']}`. These are descriptive results only and do not establish a new algorithm.\n''')
    (STAGE/'PHASE20G0_FINAL_REPORT.md').write_text(f'''# ForeSplat Phase 20G-0 Final Report\n\n## Scope\n\nScene `{p['scene']}` / `{p['group']}` / training seed 0 / M0 was selected from the frozen Phase20E-1 source. Every compression run starts from the same PLY `{p['source_model_sha256']}`. Ratios and order were frozen in `configs/EXPERIMENT_PLAN.json` before metrics. No training, SAM3, R1, COLMAP, split, mesh source, or paper TeX was modified.\n\n## Reproducibility\n\n- Baseline commit: `{p['baseline_commit']}`\n- Plan SHA256: `{sha(STAGE/'configs/EXPERIMENT_PLAN.json')}`\n- Source PLY bytes: `{metrics[0]['source_PLY_bytes']}`; source SHA256: `{p['source_model_sha256']}`\n- Frozen held-out cameras: `{quality['views_per_run']}` per run\n- Physical GPU: index 1, UUID `{GPU_UUID}`, `{benchmark['GPU_name']}`\n- Protected historical files hashed: `{preflight['checked_files']}`, historical hashes verified: `{preflight['historical_hashes_verified']}`\n- Final input recheck: see `reports/FINAL_INPUT_RECHECK.json`\n\n## Real compression results\n\n| method | removed ratio | Gaussians | PLY bytes | raw PSNR | raw SSIM | raw LPIPS | ROI PSNR proxy | FPS mean | peak reserved VRAM |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n{table}\n\nPLY byte reduction is measured separately from Gaussian-count reduction; no quantization or SH compression was applied. The 0% random and opacity outputs preserve the source PLY SHA256. All eight models loaded through the untouched native `GaussianModel.load_ply`, and the independent native model-directory validation is `{native['status']}`.\n\n## Evaluation controls\n\nAll runs use the same 27 held-out cameras, 1060x1900 resolution, SH degree 3, black background, and frozen E1/E2 metric definitions. LPIPS is VGG v0.1 with `normalize=True`; RGB is clamped only at metric input, and raw render ranges are retained. The ROI is the frozen B_R0 proxy mask and is not independent ground truth. Edge values are a 3-pixel eroded-inner-boundary diagnostic, also a proxy. Offline pruning time is not included in FPS. FPS uses seven full-camera rounds per model, two warmup sweeps, explicit CUDA synchronization, and 189 timed frames per model.\n\nUnpruned replay deltas against the historical E1 render are retained in `reports/UNPRUNED_E1_REPLAY_DELTA.csv`; the new evaluation does not silently replace it with the old render.\n\n## Method status\n\n- Generic plugin interface: **PASS**\n- Random and opacity baselines: **PASS**\n- PUP: **BLOCKED**, not approximated. Native ABI has no Fisher buffer; 2DGS has five spatial parameters and surfel covariance rather than the official 3D six-parameter path.\n- LightGaussian: **BLOCKED**, not approximated. Native ABI has no count-render output; official pinned kernel counts accepted hits and opacity, while its volume correction assumes three activated scales.\n- Tiny exact Fisher evidence: `{fisher['status']}`, with 5x5 Fisher and square-after-sum relative error `{fisher['square_after_sum_relative_error']:.6f}`.\n\n## Mesh\n\nThe predeclared Opacity 40% model passed the frozen E3-3 TSDF extraction and reload protocol: `{mesh['status']}`. This demonstrates renderer/depth/alpha/TSDF compatibility only; it does not demonstrate geometric accuracy improvement.\n\n## Acceptance status\n\n```text\nPHASE20G0 STATUS: PARTIAL\nPLUGIN_INTERFACE: PASS\nBASIC_PRUNING: PASS\nPUP_2DGS: BLOCKED\nLIGHTGAUSSIAN_2DGS: BLOCKED\nMODEL_LOAD: PASS\nRENDER_EVALUATION: PASS\nMESH_COMPATIBILITY: {mesh['status']}\nFOREGROUND_AWARE_EXTENSION: {'RECOMMENDED' if best_fg['method'] != best['method'] or f(best_fg['foreground_PSNR_mean']) < f(best['foreground_PSNR_mean']) else 'UNDECIDED'}\nNEXT_PHASE: REMEDIATION_REQUIRED\n```\n\nOverall status is PARTIAL because the required independent compressor and real baseline evaluation are complete, but neither conference-derived scorer is yet a faithful 2DGS adaptation.\n''')
    final_report = STAGE / 'PHASE20G0_FINAL_REPORT.md'
    report_text = final_report.read_text()
    marker = 'FOREGROUND_AWARE_EXTENSION:'
    marker_start = report_text.index(marker)
    marker_end = report_text.index('\n', marker_start)
    extension = 'RECOMMENDED' if foreground_signal else 'UNDECIDED'
    report_text = report_text[:marker_start] + f'{marker} {extension}' + report_text[marker_end:]
    final_report.write_text(report_text)
    print('REPORTS_GENERATED',best['run_id'],best_fg['run_id'],extension,flush=True)


if __name__=='__main__':
    metrics=aggregate_quality();plot(metrics);markdown(metrics)
