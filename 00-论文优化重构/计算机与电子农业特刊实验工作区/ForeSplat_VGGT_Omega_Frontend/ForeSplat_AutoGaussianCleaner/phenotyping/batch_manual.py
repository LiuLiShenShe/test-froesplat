"""Explicit per-leaf manual pairs; no guessed workbook row matching."""
from batch_common import *
from batch_measure import csvread,csvwrite
TRAITS=['leaf_length_cm','leaf_width_cm','leaf_area_cm2','plant_height_cm','canopy_span_a_cm','canopy_span_b_cm','canopy_width_cm']
def main(ctx):
 auto={r['leaf_id']:r for r in csvread(ctx.out/'phenotype/leaf_measurements.csv')};plant=csvread(ctx.out/'phenotype/plant_measurements.csv')[0];manual=csvread(ctx.out/'phenotype/manual_reference.csv');pairs=[];seen={};conflicts=set();excluded=[]
 for ref in manual:
  if ref.get('plant_id')!=ctx.plant or ref.get('pair_confirmed','').lower() not in ['true','1','yes'] or not all(ref.get(k,'').strip() for k in ['capture_session','measured_at','measurement_tool','operator']):excluded.append(dict(leaf_id=ref.get('leaf_id'),reason='Unconfirmed sameplant/sameleaf/samecapture or missing protocol'));continue
  for trait in TRAITS:
   source=auto.get(ref.get('leaf_id'),{}) if trait.startswith('leaf_') else plant;key=(trait,ref.get('leaf_id') if trait.startswith('leaf_') else ctx.plant)
   if key in conflicts:continue
   try:a=float(source.get(trait,''));b=float(ref.get(trait,''))
   except (ValueError,TypeError):continue
   if not np.isfinite(a+b) or a<=0 or b<=0:continue
   if key in seen:
    previous=seen[key]
    if not np.isclose(previous['manual'],b,rtol=0,atol=1e-9) or previous['capture_session']!=ref['capture_session']:
     pairs.remove(previous);conflicts.add(key);excluded.append(dict(leaf_id=ref.get('leaf_id'),trait=trait,reason='Conflicting duplicate manual reference or capture_session; all pairs for this identity/trait withheld'))
    continue
   pair=dict(plant_id=ctx.plant,leaf_id=ref.get('leaf_id') if trait.startswith('leaf_') else '',trait=trait,automatic=a,manual=b,error=a-b,relative_error=(a-b)/b,capture_session=ref['capture_session'],measured_at=ref['measured_at'],measurement_tool=ref['measurement_tool'],operator=ref['operator'],automatic_geometry_status=source.get('leaf_measurement_status',source.get('measurement_status')));seen[key]=pair;pairs.append(pair)
 errors=[]
 for trait in TRAITS:
  pp=[p for p in pairs if p['trait']==trait];a=np.array([p['automatic'] for p in pp]);b=np.array([p['manual'] for p in pp]);e=a-b;n=len(pp);ss=float(np.sum((b-b.mean())**2)) if n else 0
  errors.append(dict(plant_id=ctx.plant,trait=trait,n_pairs=n,MAE=float(np.mean(abs(e))) if n else '',RMSE=float(np.sqrt(np.mean(e**2))) if n else '',Bias=float(np.mean(e)) if n else '',relative_error=float(np.mean(abs(e)/b)) if n else '',R2=float(1-np.sum(e**2)/ss) if n>=2 and ss>0 else '',status='PROVISIONAL_GEOMETRY_LIMITS_APPLY' if n else 'NO_VERIFIED_REAL_MANUAL_PAIRS'))
 csvwrite(ctx.out/'phenotype/phenotype_errors.csv',errors);csvwrite(ctx.out/'phenotype/paired_measurements.csv',pairs,fields=None if pairs else ['plant_id','leaf_id','trait','automatic','manual','error','relative_error','capture_session','measured_at','measurement_tool','operator']);write(ctx.out/'phenotype/manual_pairing_audit.json',dict(verified_pairs=len(pairs),excluded=excluded,historical_unmatched_values_used=False,rule='Explicit samecapture, stableLeafID, operator,tool and date required'))

if __name__=='__main__':main(Context(sys.argv[1]))
