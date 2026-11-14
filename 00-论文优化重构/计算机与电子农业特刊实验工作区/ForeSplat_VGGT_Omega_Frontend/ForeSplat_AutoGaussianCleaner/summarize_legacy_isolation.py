#!/usr/bin/env python3
"""Quantify native original-row identity and the added rules' real rendered support."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,write_json,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args();root=args.comparison.resolve();base=root/'baselines'
    reference=json.loads((base/'C0/cleanup_metrics.json').read_text())
    original=read_ply(reference['input_model']);assert sha(reference['input_model'])==reference['input_sha256']
    evidence_path=root/'C5_Fused_GGGS/development_v8_1/C5-B/rounds/01/evidence.npz'
    with np.load(evidence_path) as z:
        assert np.array_equal(z['original_ids'],np.arange(len(original)))
        evidence={key:z[key].copy() for key in ['fg','bg','boundary','thin']}
    records=[]
    for name in ['C1_raw','C1_scale_only','C1_adaptive','C2_raw','C2_normalized_tier_only','C2_normalized_knn_only','C2_adaptive']:
        out=base/name;report=json.loads((out/'cleanup_metrics.json').read_text());marker=json.loads((out/'COMPLETE.json').read_text())
        ids=np.load(out/'original_gaussian_ids.npy');rows=read_ply(out/'cleaned_gaussians.ply')
        assert marker['native_sha256']==sha(out/'cleaned_gaussians.ply')
        assert np.array_equal(rows,original[ids]),'A legacy control changed native retained fields'
        removed=np.ones(len(original),bool);removed[ids]=False
        measurements=report['metrics']['mean']
        rec=dict(method=name,removed=int(removed.sum()),original_rows_retained_exactly=True,
            retained_native_field_count=len(original.dtype.names),native_sha256=marker['native_sha256'],
            fg_psnr=measurements['fg_psnr'],fg_alpha=measurements['fg_alpha'],thin_coverage=measurements['thin_coverage'],
            dark_foreground_mse=measurements['dark_foreground_mse'],background_alpha_mass_per_fg=measurements['bg_alpha_mass_per_fg'],
            removed_training_fg_contribution_fraction=float(evidence['fg'][removed].sum()/evidence['fg'].sum()),
            removed_training_bg_contribution_fraction=float(evidence['bg'][removed].sum()/evidence['bg'].sum()),
            seconds=report['seconds'],timing_scope=report.get('timing_scope','Standalone rule discovery, export and shared actual native evaluation'),
            native_ply=report['native_ply'])
        reason_file=out/'deletion_reasons.npz'
        if reason_file.exists():
            with np.load(reason_file) as z:
                if 'reason_bits' in z:
                    bits=z['reason_bits'];knn=bits==(1<<7)
                    rec.update(exclusive_knn_candidates=int(knn.sum()),
                        exclusive_knn_fg_contribution=float(evidence['fg'][knn].sum()),
                        exclusive_knn_bg_contribution=float(evidence['bg'][knn].sum()))
        records.append(rec);print(name,rec['removed'],rec['fg_psnr'],flush=True)
    write_json(root/'comparisons/LEGACY_FACTOR_ISOLATION.json',dict(input_sha256=reference['input_sha256'],
        original_gaussians=len(original),records=records,original_contribution_cache=str(evidence_path),
        limitation='Contribution shares are original-model native T*alpha attribution, not botanical ground-truth deletion labels. Every control uses unchanged full native GGGS fields; their quality loss is measured with the same 208 photographs/cameras/masks.'))

if __name__=='__main__':main()
