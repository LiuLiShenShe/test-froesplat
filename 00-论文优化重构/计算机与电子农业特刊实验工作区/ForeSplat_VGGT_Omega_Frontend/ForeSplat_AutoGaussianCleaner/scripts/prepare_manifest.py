#!/usr/bin/env python3
"""Inventory existing 19 bundles; prepare training links without any new inference."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    root = a.root.resolve()
    module = root/'ForeSplat_AutoGaussianCleaner'
    dual = root/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend'
    repo = dual/'camera_interface/gggs_corrected'
    history = root/'experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection'
    dev = {'KongQueZhuYu', 'XiangPiShu1', 'WangWenCao1', 'ChangShouHua1'}
    rows, scenes = [], []
    for bundle in sorted((root/'runs/full_sequence_batch').glob('*/GGGS_Dataset')):
        plant = bundle.parent.name
        output = module/'experiments/model_preparation'/plant
        dataset = output/'dataset'
        dataset.mkdir(parents=True, exist_ok=True)
        for name in ['images', 'masks', 'sparse', 'per_frame', 'metadata']:
            link = dataset/name
            if not link.exists():
                link.symlink_to(bundle/name, target_is_directory=True)
        model = output/'checkpoints/point_cloud/iteration_30000/point_cloud.ply'
        geometry_cameras, geometry_dir = bundle/'per_frame/cameras.json', bundle/'per_frame'
        baseline = {}
        if plant == 'KongQueZhuYu':
            dataset = dual/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
            model = dual/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
            geometry_cameras = dual/'Standard_VGGT_Original_RGB/cameras.json'
            geometry_dir = dual/'Standard_VGGT_Original_RGB/per_frame'
            baseline = {'P2': str(history/'P2_GGGS_Cleaning/cleaned_gaussians.ply'),
                'R5': str(history/'gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/best_cleaned/cleaned_gaussians.ply')}
        scene = dict(plant=plant, dataset=str(dataset), model=str(model), gggs_repo=str(repo),
            geometry_cameras=str(geometry_cameras), geometry_dir=str(geometry_dir),
            raw_rgb_dir=str(Path('/data/fj/F2DMAS/02-FFT')/plant), baseline_models=baseline,
            frontend='historical_standard' if plant == 'KongQueZhuYu' else 'existing_omega_bundle')
        file = module/'configs/scenes'/f'{plant}.json'
        write_json(file, scene)
        group = 'development' if plant in dev else 'heldout_validation'
        scenes.append(dict(plant=plant, group=group, scene=str(file), training_dir=str(output), existing_model=model.exists()))
        rows.append(dict(Plant=plant, Group=group, GGGSCheckpoint=model.exists(), GeometryBundle=True,
            Cameras=len(json.loads(geometry_cameras.read_text())['frames']),
            RGB=len(list((dataset/'images').glob('*.png'))), Mask=len(list((dataset/'masks').glob('*.png'))),
            Depth=len(list(geometry_dir.glob('depth_*.npy'))), Confidence=len(list(geometry_dir.glob('confidence_*.npy'))),
            Model=str(model), Bundle=str(bundle), Frontend=scene['frontend']))
    if len(scenes) != 19:
        raise ValueError(f'Expected 19 bundles, found {len(scenes)}')
    manifest = module/'configs/plant_manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if [(x['plant'], x['group']) for x in old['plants']] != [(x['plant'], x['group']) for x in scenes]:
            raise ValueError('Refusing to change frozen split')
    write_json(manifest, dict(plants=scenes, split_frozen_before_test=True,
        split_rationale='Development includes historical broad/variegated plant plus broad foliage, small dark foliage and overlapping foliage; heldout never used for tuning. Names choose groups, never cleaning rules.',
        model_preparation='Existing front-end bundles -> locked corrected GGGS, 30K, sh3/sg0, resolution2, seed0. No new inference. Historical Kong uses Standard; new plants use Omega so frontend differs and is disclosed.'))
    report = module/'reports/DATA_INVENTORY.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(module/'configs/input_inventory.json', dict(rows=rows, already_trained=sum(r['GGGSCheckpoint'] for r in rows)))
    print('19 bundles; GGGS ready:', sum(r['GGGSCheckpoint'] for r in rows))


if __name__ == '__main__':
    main()
