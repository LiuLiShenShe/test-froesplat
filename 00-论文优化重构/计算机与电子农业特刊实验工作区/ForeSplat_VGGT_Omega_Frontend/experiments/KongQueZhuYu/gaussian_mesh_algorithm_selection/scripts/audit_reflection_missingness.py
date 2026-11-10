"""Same RGB-query denominators and finite sample counts for every native method."""
from pathlib import Path
import csv,math
from queue_io import write_status
OUT=Path(__file__).resolve().parents[1];ROOT=OUT/'comparison/reflection_analysis'

def finite(value):
 try:return math.isfinite(float(value))
 except (ValueError,TypeError):return False

def main():
 manifest=list(csv.DictReader((ROOT/'RGB_QUERY_MANIFEST.csv').open()))
 expected={int(r['query_id']) for r in manifest}
 records=[];model_sources=[]
 metrics=['RGB_match_reprojection_px','matched_relative_depth_error','normal_crossview_abs_angle_deg',
          'local_normal_jump3px_deg','nearest_primitive_anisotropy']
 for path in sorted(ROOT.glob('*_spatial_correspondence.csv')):
  rows=list(csv.DictReader(path.open()))
  assert len(rows)==len(expected) and {int(r['query_id']) for r in rows}==expected,('RGB-query denominator changed',path)
  method=rows[0]['method'];model_sources.append(str(path.resolve()))
  for candidate in [True,False]:
   selected=[r for r in rows if r['brightness_candidate'].lower()==str(candidate).lower()]
   valid=[r for r in selected if r['valid'].lower()=='true']
   record=dict(method=method,brightness_candidate=candidate,queries=len(selected),native_depth_alpha_valid_queries=len(valid),
    invalid_queries=len(selected)-len(valid),valid_fraction=len(valid)/max(len(selected),1))
   for metric in metrics:
    count=sum(finite(r.get(metric,'')) for r in valid)
    record[metric+'_finite_queries']=count
    record[metric+'_nonfinite_or_absent_valid_queries']=len(valid)-count
   record['source_world_coordinates_finite_queries']=sum(all(finite(r.get(k,'')) for k in ['source_world_x','source_world_y','source_world_z']) for r in selected)
   records.append(record)
 if records:
  with (ROOT/'ALL_METHOD_QUERY_VALIDITY.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 write_status(ROOT/'MISSINGNESS_AUDIT.json',dict(status='complete_available',manifest_queries=len(expected),
  methods=sorted({r['method'] for r in records}),same_image_only_query_ids_for_every_method=True,
  sources=model_sources,records=records,
  interpretation='Depth/alpha validity and finite metric sample counts are separate. Nonfinite/missing normals or primitive diagnostics cannot secretly improve a reported median by shrinking its denominator. No independent scan or certified specular reflection.'))
 print('Actual RGB query validity/finite denominators audited',len(model_sources),'methods',flush=True)

if __name__=='__main__':main()
