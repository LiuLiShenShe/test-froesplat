"""Read-only inventory: distinguish actual COLMAP from existing Omega models."""
from batch_common import *
from batch_prepare import locate,mask_source
from batch_measure import csvwrite

def main():
 records=[];loader=colmap_io()
 for row in read(MANIFEST)['plants']:
  ctx=Context(row['plant']);scene=read(row['scene']);source=locate(ctx);rgb=sorted(ctx.raw.glob('*.jpg'));annotations=[]
  for file in sorted(ctx.annotation.glob('*.json')):
   record=read(file)
   if any(s['label']=='blue_cube' for s in record.get('shapes',[])):annotations.append(str(file))
  sparse=source[0]/'sparse/0' if source else None;views=loader.read_extrinsics_binary(sparse/'images.bin') if sparse else {};observations=sum(int((v.point3D_ids>=0).sum()) for v in views.values());mesh_roots=[ROOT/'04-COLMAP'/ctx.plant,Path(row['training_dir']),ROOT/'experiments/COLMAP_Phenotyping'/ctx.plant]
  record=dict(plant=ctx.plant,group=row['group'],rgb_dir=str(ctx.raw),rgb_count=len(rgb),existing_manifest_gaussian=scene.get('model'),existing_gaussian_exists=Path(scene.get('model','/nonexistent')).exists(),existing_manifest_frontend=scene.get('frontend'),manifest_Gaussian_eligible_for_reuse=ctx.plant=='KongQueZhuYu',true_colmap_source=str(source[0]) if source else None,distorted_source=str(source[1]) if source else None,true_colmap_registered_views=len(views),valid_feature_track_observations=observations,blue_cube_historical_annotations=annotations,valid_archived_RAP_masks=sum(mask_source(ctx,file.stem) is not None for file in rgb),existing_meshes=[str(file) for root in mesh_roots for file in root.glob('**/*mesh*.ply')],existing_phenotype_tables=[str(file) for root in mesh_roots for file in root.glob('**/*measurement*.csv')])
  records.append(record);print('INVENTORY',ctx.plant,len(rgb),len(views),observations,flush=True)
 write(OUT/'summary/input_inventory.json',dict(plants=records,policy='Nonzero triangulated feature tracks and paired native distorted/undistorted SfM. Manifest Omega/VGGT models are ineligible. No source mutation.',manifest=signature(MANIFEST)))
 csvwrite(OUT/'summary/input_inventory.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()} for row in records])

if __name__=='__main__':main()
