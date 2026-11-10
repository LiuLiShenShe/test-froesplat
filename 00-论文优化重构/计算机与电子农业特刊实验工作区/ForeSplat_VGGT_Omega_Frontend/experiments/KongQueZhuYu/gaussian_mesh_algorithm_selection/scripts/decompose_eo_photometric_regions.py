"""Partition existing E10/P2/E12 photometric scores using actual frozen masks."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from queue_io import write_status

OUT = Path(__file__).resolve().parents[1]
DEST = OUT / 'comparison/EO_region_decomposition'
OLD = OUT.parent / 'geometry_mesh_final_selection'
DUAL = OUT.parent / 'geometry_rgb_dual_frontend'

def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def save(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    DEST.mkdir(exist_ok=True)
    index_path = OLD / 'surface_inputs/e10_gggs_surface/render_index.json'
    index = {int(r['source_id']): r for r in json.loads(index_path.read_text())}
    assert set(index) == set(range(210)) - {44, 206}
    counts = {}
    for sid, record in index.items():
        with np.load(record['path']) as data:
            mask = data['foreground_mask']
        assert mask.dtype == bool and mask.shape == (1920, 1080)
        assert hashlib.sha256(mask.tobytes()).hexdigest() == record['mask_sha256']
        counts[sid] = (int(mask.sum()), int(mask[3:-3, 3:-3].sum()), mask.size, (1920-6)*(1080-6))
    exposure = OUT / 'E12_GGGS_EO/exposure_params/exposure_matrices.npz'
    # This object camera-name array is the trusted actual E12 export.
    with np.load(exposure, allow_pickle=True) as data:
        matrices = {str(name): matrix for name, matrix in zip(data['camera_names'], data['matrices'])}
    assert len(matrices) == 208
    sources = [('E10', DUAL/'render_comparison/E10/per_view_metrics.csv'),
               ('P2', OUT/'P2_GGGS_Cleaning/surface_inputs/per_view_metrics.csv'),
               ('E12', OUT/'E12_GGGS_EO/surface_inputs/per_view_metrics.csv')]
    records, source_hashes = [], []
    for method, path in sources:
        rows = read(path)
        source_hashes.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        stages = {r.get('stage', 'raw') for r in rows}
        for stage in sorted(stages):
            selected = [r for r in rows if r.get('stage', 'raw') == stage]
            assert len(selected) == 208 and {int(r['source_id']) for r in selected} == set(index)
            for row in selected:
                sid = int(row['source_id']); reference = index[sid]
                assert (row['gt_sha256'], row['mask_sha256']) == (reference['gt_sha256'], reference['mask_sha256'])
                fg_pixels, fg_valid, all_pixels, all_valid = counts[sid]
                weight = fg_valid / all_valid
                full_ssim = float(row['ssim']); fg_ssim = float(row['fg_mask_ssim_map'])
                fg_contribution = weight * fg_ssim
                bg_contribution = full_ssim - fg_contribution
                bg_ssim = bg_contribution / (1-weight)
                full_mse = 10**(-float(row['full_psnr'])/10)
                fg_mse = 10**(-float(row['fg_mask_psnr'])/10)
                bg_mse = (all_pixels*full_mse - fg_pixels*fg_mse)/(all_pixels-fg_pixels)
                assert bg_mse >= -1e-8 and -1.00001 <= bg_ssim <= 1.00001
                bg_mse = max(bg_mse, 0)
                offset = matrices[reference['camera']['name']][:, 3] if method == 'E12' else None
                records.append(dict(method=method, stage=stage, source_id=sid,
                    foreground_pixels=fg_pixels, background_pixels=all_pixels-fg_pixels,
                    SSIM_valid_foreground_pixels=fg_valid, SSIM_valid_background_pixels=all_valid-fg_valid,
                    SSIM_foreground_weight=weight, full_ssim=full_ssim, foreground_ssim=fg_ssim,
                    derived_background_ssim=bg_ssim, foreground_ssim_full_image_contribution=fg_contribution,
                    background_ssim_full_image_contribution=bg_contribution,
                    full_psnr=float(row['full_psnr']), foreground_psnr=float(row['fg_mask_psnr']),
                    derived_background_mse=bg_mse, derived_background_psnr=-10*np.log10(bg_mse) if bg_mse else '',
                    EO_offset_R=float(offset[0]) if offset is not None else '',
                    EO_offset_G=float(offset[1]) if offset is not None else '',
                    EO_offset_B=float(offset[2]) if offset is not None else '',
                    EO_affine_black_clipped_mean=float(np.clip(offset, 0, 1).mean()) if offset is not None else '',
                    metric_source=str(path.resolve()), actual_mask_archive=reference['path']))
    save(DEST/'per_view.csv', records)
    groups = {}
    for row in records:
        groups.setdefault((row['method'], row['stage']), []).append(row)
    metrics = ['SSIM_foreground_weight', 'full_ssim', 'foreground_ssim', 'derived_background_ssim',
               'foreground_ssim_full_image_contribution', 'background_ssim_full_image_contribution',
               'full_psnr', 'foreground_psnr', 'derived_background_psnr']
    summaries = [dict(method=method, stage=stage, views=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in metrics})
                 for (method, stage), rows in groups.items()]
    save(DEST/'summary.csv', summaries)
    pairs = [('E10', 'raw', 'P2', 'raw'), ('E10', 'raw', 'E12', 'raw'),
             ('E10', 'raw', 'E12', 'corrected'), ('E12', 'raw', 'E12', 'corrected')]
    differences = []
    for a, astage, b, bstage in pairs:
        first = {r['source_id']:r for r in groups[(a, astage)]}
        second = {r['source_id']:r for r in groups[(b, bstage)]}
        for sid in sorted(index):
            differences.append(dict(before=a+'_'+astage, after=b+'_'+bstage, source_id=sid,
                full_ssim_delta=second[sid]['full_ssim']-first[sid]['full_ssim'],
                foreground_contribution_delta=second[sid]['foreground_ssim_full_image_contribution']-first[sid]['foreground_ssim_full_image_contribution'],
                background_contribution_delta=second[sid]['background_ssim_full_image_contribution']-first[sid]['background_ssim_full_image_contribution']))
    save(DEST/'paired_SSIM_contributions.csv', differences)
    write_status(DEST/'PROTOCOL.json', dict(status='complete', actual_views=208, conditions=list(groups), source_metrics=source_hashes,
        actual_mask_index=str(index_path.resolve()), exposure_parameters=str(exposure.resolve()), exposure_sha256=hashlib.sha256(exposure.read_bytes()).hexdigest(),
        SSIM_definition='Stored full-image SSIMwin7/samplecov score is the mean across RGB channels and3px-rim-excluded pixels; foreground map mean uses the same region. Background mean is derived from these scalars and actual mask counts.',
        MSE_definition='Background MSE is derived from stored full/FG PSNR and actual full-resolution mask counts; floating-score rounding is retained. This does not recompute/render images.',
        delta_definition='Per-view full SSIM delta is decomposed into changes of the foreground/background contributions; means of contributions are not unweighted region means.',
        limits='This partitions existing photometric fit, not true geometric accuracy or a causal test of each exposure matrix component. EO affine mapping of an exactly black input is a mathematical parameter diagnostic, not actual measured background RGB.',
        native_models_and_rendered_arrays_unchanged=True, CPU_only=True))
    print('Actual EO photometric region decomposition', len(records), 'per-view rows', summaries, flush=True)

if __name__ == '__main__':
    main()
