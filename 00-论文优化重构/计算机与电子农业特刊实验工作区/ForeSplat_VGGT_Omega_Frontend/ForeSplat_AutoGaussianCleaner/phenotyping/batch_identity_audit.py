"""Read-only preflight of historical COLMAP RGB byte identity across all19."""
from batch_common import *
from batch_prepare import locate,bind_registered_rgb
from batch_measure import csvwrite

def main():
 records=[];loader=colmap_io();meter=Meter()
 for record in read(MANIFEST)['plants']:
  ctx=Context(record['plant']);ctx.meter=meter;source=locate(ctx)
  if source is None:records.append(dict(plant=ctx.plant,status='REAL_COLMAP_RERUN_REQUIRED',registered=0,mapped=0));continue
  directory,raw=source;views=loader.read_extrinsics_binary(directory/'sparse/0/images.bin');bound=bind_registered_rgb(ctx,directory,views);matches=[dict(registered_name=r[0].name,current_fft_name=r[1],proof=r[5]) for r in bound];matched={r[0].id for r in bound};missing=[v.name for v in views.values() if v.id not in matched];renamed=sum(r[1]!=r[3] for r in bound)
  row=dict(plant=ctx.plant,group=record['group'],status='VERIFIED' if len(matches)>=max(10,len(views)*.9) else 'INPUT_IDENTITY_MISMATCH_REQUIRES_NATIVE_SFM',registered=len(views),mapped=len(matches),renamed=renamed,source=str(directory),missing=missing,matches=matches);records.append(row);print('IDENTITY',ctx.plant,row['status'],len(matches),len(views),'renamed',renamed,flush=True);write(OUT/'summary/input_identity_audit.json',dict(plants=records,io=meter.stats(),complete=False))
 write(OUT/'summary/input_identity_audit.json',dict(plants=records,io=meter.stats(),complete=True));csvwrite(OUT/'summary/input_identity_audit.csv',[{k:v for k,v in row.items() if k not in ['matches','missing']} for row in records])

if __name__=='__main__':main()
