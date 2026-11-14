#!/usr/bin/env python3
"""Export the selected real model only after verified direct browser review."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from viewer_review_validity import validated_review


def finalize(output):
    output=Path(output).resolve();review=validated_review(output)
    if review is None:
        raise ValueError('A verified direct-image review is required for final selection')
    label=review['recommended_model'];models=output/'comparison/models'
    if label not in review['reviewed_models']:
        raise ValueError('Selected model was not directly reviewed')
    paths={}
    for suffix,name in [('native','selected_cleaned_gaussians.ply'),('supersplat','selected_supersplat.ply')]:
        source=models/f'{label}_{suffix}.ply'
        if not source.is_file():raise FileNotFoundError(source)
        target=output/name
        if target.is_symlink():target.unlink()
        elif target.exists():raise FileExistsError(target)
        target.symlink_to(source.resolve())
        paths[suffix]=dict(path=str(target),source=str(source.resolve()),sha256=sha(source))
    result=dict(selected_model=label,selection_reason=review['selection_reason'],models=paths,
        visual_acceptance=review.get('visual_acceptance'),
        selection_classification=('failed_visual_case_best_available' if review.get('visual_acceptance') is False
            else 'measured_improvement_with_recorded_limitations'),
        renderer_build_sha256=review['renderer_build_sha256'],
        direct_review=str(output/'comparison/official_supersplat/DIRECT_IMAGE_REVIEW.json'),
        remaining_artifact_severity=review.get('remaining_artifact_severity','see direct review'))
    write_json(output/'VISUAL_SELECTION.json',result)
    capture_record=output/'comparison/official_supersplat/REVIEW.json'
    captures=json.loads(capture_record.read_text())
    captures.update(visual_review='direct_image_inspection_completed',
        direct_review=result['direct_review'],reviewed_at=review.get('reviewed_at'))
    write_json(capture_record,captures)
    index_path=output/'comparison/model_index.json'
    if index_path.exists():
        index=json.loads(index_path.read_text())
        index.update(interactive_supersplat_review='verified_direct_image_inspection_completed',
            direct_review=result['direct_review'],selected_model=label)
        write_json(index_path,index)
    report=output/'VISUAL_SELECTION.md'
    report.write_text('\n'.join([f'# {output.name}: selected {label}','',review['selection_reason'],'',
        f'Native GGGS: {paths["native"]["path"]}',f'SuperSplat: {paths["supersplat"]["path"]}','',
        *review.get('observations',[]),'',
        'Source: actual local GGGS outputs and directly inspected browser screenshots; known-opacity-compatible renderer.'])+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(finalize(a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
