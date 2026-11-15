"""Run the existing guarded phenotype primitives on raw/post mesh geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import csv
import gc
import json
import sys
import numpy as np

from .common import (PHENOTYPE, PacedFile, Stage, guard_memory, read_mesh,
    stage_args, write)
sys.path.insert(0,str(PHENOTYPE/'scripts'))
from phenotype_core import (MeasurementBlocked, canopy_spans, height_along_axis,
    metric_length, pca_extents, post_vertices_subset_of_raw)


def geometric_proxy(points):
    center,directions,extents = pca_extents(points)
    projection = (points-center)@directions[:,0]
    base = center+float(projection.min())*directions[:,0]
    height = height_along_axis(points,base,directions[:,0])
    spans = canopy_spans(points,directions[:,0],directions[:,1],directions[:,2])
    return dict(bbox_min=points.min(0).tolist(),bbox_max=points.max(0).tolist(),
        bbox_xyz_extents=np.ptp(points,axis=0).tolist(),pca_center=center.tolist(),
        pca_directions_columns=directions.tolist(),pca_axis_extents=extents.tolist(),
        exploratory_height_pca_units=height,exploratory_horizontal_pca_spans_units=list(spans[:2]),
        exploratory_max_horizontal_pca_span_units=spans[2],
        proxy_base_point=base.tolist(),proxy_axis='PCA major axis, not verified gravity',
        formal_plant_height_units=None,formal_canopy_width_units=None,physical_trait_claim=False)


def run(args):
    out=args.output/'phenotype'
    if (out/'complete.json').exists(): return
    spec=json.loads(args.scene.read_text())
    report=json.loads((args.output/'mesh/mesh_report.json').read_text())
    records={};rows=[]
    with Stage(out) as stage:
        meter=stage.meter
        raw_vertices=None
        for name in ['raw','post']:
            mesh=read_mesh(args.output/f'mesh/{name}.ply',meter)
            vertices=np.asarray(mesh.vertices)
            record=geometric_proxy(vertices)
            record.update(vertices=len(vertices),faces=len(mesh.triangles),
                surface_area_model_units2=report['structure'][name]['surface_area'],
                scope='Entire reconstructed foreground mesh, includes pot, reference objects and possible artifacts; no validated plant-only region')
            if name=='raw': raw_vertices=vertices.copy()
            else: record['post_vertices_exact_subset_of_raw']=post_vertices_subset_of_raw(raw_vertices,vertices)
            try:
                metric_length(record['exploratory_height_pca_units'],None,None)
            except MeasurementBlocked as blocked:
                record['physical_scale_status']=str(blocked)
            record.update(physical_plant_height_cm=None,physical_canopy_width_cm=None,
                formal_trait_status='PLANT_ONLY_REGION_VERTICAL_AXIS_AND_BASE_NOT_AVAILABLE',
                leaf_status='LEAF_ID_NOT_AVAILABLE',leaf_length_units=None,leaf_width_units=None,leaf_area_units2=None,
                leaf_missing=['Leaf instance ID','leaf base/tip landmarks','leaf boundary or face membership','declared leaf measurement directions'])
            records[name]=record
            rows.append(dict(plant=spec['plant'],stage=name,vertices=record['vertices'],faces=record['faces'],
                bbox_x_units=record['bbox_xyz_extents'][0],bbox_y_units=record['bbox_xyz_extents'][1],bbox_z_units=record['bbox_xyz_extents'][2],
                pca_extent1_units=record['pca_axis_extents'][0],pca_extent2_units=record['pca_axis_extents'][1],pca_extent3_units=record['pca_axis_extents'][2],
                geometric_height_proxy_units=record['exploratory_height_pca_units'],horizontal_span1_proxy_units=record['exploratory_horizontal_pca_spans_units'][0],
                horizontal_span2_proxy_units=record['exploratory_horizontal_pca_spans_units'][1],
                surface_area_model_units2=record['surface_area_model_units2'],formal_plant_height_units='',formal_canopy_width_units='',
                physical_height_cm='',physical_canopy_width_cm='',physical_scale_status=record['physical_scale_status'],
                formal_trait_status=record['formal_trait_status'],leaf_status=record['leaf_status'],physical_trait_claim=False,
                scope='WHOLE_MESH_INCLUDES_POT_AND_REFERENCE_OBJECTS;PCA_AXES_ARE_NOT_GRAVITY'))
            del mesh,vertices;gc.collect();guard_memory()
        del raw_vertices
        with PacedFile(args.output/'surface/nksr_input.npz','rb',meter) as stream:
            with np.load(stream,allow_pickle=False) as archive:
                points=archive['xyz']
        candidate=geometric_proxy(points)
        candidate.update(points=len(points),scope='Existing 2D foreground and alpha/depth surface candidates include pot/reference objects; not plant-only',
            formal_status='PLANT_ONLY_REGION_VERTICAL_AXIS_AND_BASE_NOT_AVAILABLE')
        changes=dict(bbox_xyz_extents=(np.asarray(records['post']['bbox_xyz_extents'])-records['raw']['bbox_xyz_extents']).tolist(),
            pca_axis_extents=(np.asarray(records['post']['pca_axis_extents'])-records['raw']['pca_axis_extents']).tolist(),
            surface_area_model_units2=records['post']['surface_area_model_units2']-records['raw']['surface_area_model_units2'])
        context=dict(scene=str(args.scene.resolve()),coordinate_frame='Existing corrected exported Omega/GGGS world gauge; arbitrary units',
            plant_only_region='No validated 3D plant-only region; current reconstruction masks include pot and blue reference objects',vertical_axis='NOT_AVAILABLE; camera poses do not establish gravity',
            plant_base='NOT_AVAILABLE',physical_scale='PHYSICAL_SCALE_NOT_AVAILABLE',
            leaf_instances='LEAF_ID_NOT_AVAILABLE',
            evidence_checked=[str(args.scene.resolve()),spec['dataset']+'/metadata/reconstruction_metadata.json',
                str(PHENOTYPE/'PHYSICAL_SCALE_PROVENANCE.csv'),str(PHENOTYPE/'GRAVITY_AND_REFERENCE_FRAME.csv'),
                str(PHENOTYPE/'LEAF_MEASURABILITY_REPORT.csv'),str(PHENOTYPE/'HISTORICAL_PHENOTYPE_MATCHING.csv'),
                str(PHENOTYPE.parent/'阶段二十B_历史实验与证据修复/reports/SCALE_RECOVERY_AUDIT.md'),
                str(PHENOTYPE.parent/'阶段二十A_方法融合审计与算法锁定准备/PLANT_POT_SCALE_CONSISTENCY.md')],
            historical_workbook='Existing audit has no specimen/acquisition-linked length anchor or leaf IDs. Historical virtual values are not new mesh measurements.',
            reference_object='Blue calibration-object class is documented in existing annotation guidance, but a verified physical size and corresponding 3D landmarks are not archived',
            needed_for_centimeters=['Known real-world length and two corresponding reconstructed endpoints',
                'same specimen and acquisition linkage','plant-only region','vertical/reference plane and plant base'],
            no_scale_inference_from_pot_or_bbox=True)
        write(out/'measurement_context.json',context)
        write(out/'geometry.json',dict(plant=spec['plant'],stages=records,surface_candidate_proxy=candidate,
            post_minus_raw=changes,measurement_context=context,
            functions_reused=['pca_extents','height_along_axis','canopy_spans','metric_length','post_vertices_subset_of_raw']))
        with (out/'phenotype.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'complete.json',dict(status='complete',plant=spec['plant'],physical_scale='PHYSICAL_SCALE_NOT_AVAILABLE'))


if __name__=='__main__':
    run(stage_args(__doc__))
