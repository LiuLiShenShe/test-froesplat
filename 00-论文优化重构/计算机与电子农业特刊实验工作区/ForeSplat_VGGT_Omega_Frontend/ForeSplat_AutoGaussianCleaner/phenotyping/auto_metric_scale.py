"""Uniform GGGS/cm conversion; never modify source axes or scale twice."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
from plyfile import PlyData

LENGTH_FIELDS={'x','y','z','filter_3D','filter_3d'}
LOG_FIELDS={'scale_0','scale_1','scale_2'}

def scale_vertices(vertices,factor):
 if not np.isfinite(factor) or factor<=0:raise ValueError('Positive physical scale required')
 result=vertices.copy()
 for name in vertices.dtype.names:
  if name in LENGTH_FIELDS:result[name]*=factor
  elif name in LOG_FIELDS:result[name]+=np.log(factor)
 return result

def apply(ctx):
 calibration=read(ctx.out/'calibration/scale_calibration.json');source=ctx.out/'gaussian/original_gaussians.ply';result=ctx.out/'gaussian/metric_gaussians.ply';marker=ctx.out/'gaussian/metric_conversion.json'
 if not has_metric_scale(calibration):ctx.status('metric','uncalibrated',reason=calibration.get('status'));return False
 factor=float(calibration['scale_cm_per_unit']);key=dict(source=signature(source),scale_cm_per_unit=factor,output_units='cm',applied_scale_count=1)
 if marker.exists():
  old=read(marker)
  if all(old[k]==v for k,v in key.items()) and result.exists():return True
  raise ValueError('Existing metric conversion has different input; refusing repeated scaling')
 source_payload=ctx.meter.read(source);header,original=scalar_ply(source_payload);converted=scale_vertices(original,factor);payload=header+converted.tobytes();artifact(result,payload,ctx.meter)
 cameras=read(ctx.common/'cameras.json');views=[]
 for originalcam in cameras['views']:
  c=dict(originalcam);c['center']=(np.array(c['center'])*factor).tolist();c['tvec']=(np.array(c['tvec'])*factor).tolist();views.append(c)
 cameras.update(views=views,coordinate_frame='Same COLMAP axes, centimetres',length_unit='cm',source_units='COLMAP arbitrary units',applied_scale_count=1,scale_cm_per_unit=factor)
 write(ctx.out/'gaussian/metric_cameras.json',cameras)
 changed=[n for n in original.dtype.names if n in LENGTH_FIELDS|LOG_FIELDS];unchanged=[n for n in original.dtype.names if n not in changed]
 assert all(np.array_equal(original[n],converted[n],equal_nan=True) for n in unchanged)
 write(marker,dict(**key,changed_fields=changed,unchanged_fields=unchanged,gaussians=len(original),source_sha256=hashlib.sha256(source_payload).hexdigest(),output_sha256=hashlib.sha256(payload).hexdigest(),camera_rule='center_cm=s*center, tvec_cm=s*tvec, same rotation and intrinsics',subsequent_units='Render metric Gaussian with metric camera; depth, scene extent,voxel and SDF truncation are cm. Phenotypes multiply by1.',io=ctx.meter.stats()));ctx.status('metric','complete',scale_cm_per_unit=factor,gaussians=len(original));return True

if __name__=='__main__':apply(Context(sys.argv[1]))
