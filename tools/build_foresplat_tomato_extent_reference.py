"""Extract same-definition height and canopy spans from the metric cohort's official labels."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, distance

from pair_tomatopgt_compounds import REF, graph_reference, read_csv, write_csv


EVIDENCE = Path(__file__).resolve().parents[1] / 'ForeSplat_Paper_Evidence_Package'
COHORT = EVIDENCE / '07_PHENOTYPING_EVALUATION/Tomato_Metric_19'


def main():
    output = []
    scans = read_csv(COHORT / 'Tomato_Metric_19_Reconstruction.csv')
    assert len({r['sample_id'] for r in scans}) == len(scans) == 19
    for scan in scans:
        sid = scan['sample_id']
        data = REF / sid / 'annotations'
        graph = next((data / 'skeletons').glob('*_annotated_ext_graph.json'), None)
        labels = next((data / 'semantic').glob('*_annotated_ext.txt'))
        row = dict(sample_id=sid, annotated_cloud=str(labels), cloudgraph=str(graph or ''),
                   status='no_official_metric_transform', reference_height_cm='', reference_canopy_cm='',
                   reference_plant_points='', graph_anchor_median_residual_cm='')
        if graph:
            meta = json.loads(graph.read_text())['meta']['transform']
            rotation = np.asarray(meta['rotate'])
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            assert 0 < meta['scale_div'] and np.isclose(np.linalg.det(rotation), 1)
            raw = np.loadtxt(labels, usecols=(0, 1, 2, 6, 7))
            cls, ids = raw[:, 3:].astype(int).T
            groups = {(c, i): np.flatnonzero((cls == c) & (ids == i))
                      for c in (1, 3) for i in np.unique(ids[cls == c]) if i >= 0}
            xyz, _, _, residual = graph_reference(graph, raw[:, :3], groups)
            plant = xyz[np.isin(cls, [0, 1, 2, 3, 6, 7, 8, 9])]
            assert len(plant) > 100 and np.isfinite(plant).all()
            height = float(np.ptp(plant[:, 2]))
            assert residual is not None and residual < max(.22, .032 * height), (sid, residual)
            xy = plant[:, :2]
            canopy = float(distance.pdist(xy[ConvexHull(xy).vertices]).max())
            assert height > 0 and canopy > 0
            row.update(status='derived_from_official_labeled_cloud', reference_height_cm=height,
                       reference_canopy_cm=canopy, reference_plant_points=len(plant),
                       graph_anchor_median_residual_cm=residual)
        output.append(row)
        print(sid, row['status'], flush=True)
    target = COHORT / 'Whole_plant_pointcloud_reference.csv'
    write_csv(target, output)


if __name__ == '__main__':
    main()
