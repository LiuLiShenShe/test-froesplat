"""Errors only for explicitly verified same-specimen/session/LeafID pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from common import *
from measure_phenotypes import csvwrite
import csv

TRAITS=['leaf_length_cm','leaf_width_cm','leaf_area_cm2','plant_height_cm','canopy_span_a_cm','canopy_span_b_cm']
def rows(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def positive(value):
 try:
  number=float(value);return number if np.isfinite(number) and number>0 else None
 except (TypeError,ValueError):return None
def main():
 auto=rows(OUT/'phenotype/leaf_measurements.csv');plant=rows(OUT/'phenotype/plant_measurements.csv')[0];lookup={r['leaf_id']:r for r in auto};manual=rows(OUT/'phenotype/manual_reference.csv');paired=[];audit=[];seen={}
 for ref in manual:
  accepted=ref.get('plant_id')==plant['plant_id'] and ref.get('pair_confirmed','').lower() in ['true','1','yes'] and all(ref.get(k,'').strip() for k in ['capture_session','measured_at','measurement_tool','operator'])
  if not accepted:
   audit.append(dict(leaf_id=ref.get('leaf_id'),status='Unconfirmed identity/session or missing protocol metadata; excluded'));continue
  if ref.get('leaf_id') not in lookup:
   audit.append(dict(leaf_id=ref.get('leaf_id'),status='Unknown LeafID; excluded'));continue
  for trait in TRAITS:
   source=lookup[ref['leaf_id']] if trait.startswith('leaf_') else plant
   predicted=positive(source.get(trait));measured=positive(ref.get(trait))
   if predicted is None or measured is None:continue
   key=(trait,ref['leaf_id'] if trait.startswith('leaf_') else plant['plant_id'])
   if key in seen:
    if not np.isclose(seen[key],measured):raise ValueError('Conflicting repeated manual reference: '+str(key))
    continue
   seen[key]=measured
   paired.append(dict(plant_id=plant['plant_id'],leaf_id=ref['leaf_id'] if trait.startswith('leaf_') else '',trait=trait,automatic_value=predicted,manual_value=measured,error=predicted-measured,relative_error=(predicted-measured)/measured,capture_session=ref['capture_session'],measured_at=ref['measured_at'],measurement_tool=ref['measurement_tool'],operator=ref['operator'],automatic_status=source.get('leaf_measurement_status',source.get('measurement_status')),formal_whole_leaf_valid=source.get('formal_whole_leaf_valid','Not applicable')))
 summary=[]
 for trait in TRAITS:
  pairs=[r for r in paired if r['trait']==trait];a=np.array([r['automatic_value'] for r in pairs]);b=np.array([r['manual_value'] for r in pairs]);e=a-b;n=len(pairs);ss=float(np.sum((b-b.mean())**2)) if n else 0
  summary.append(dict(trait=trait,n_pairs=n,MAE=float(np.mean(abs(e))) if n else '',RMSE=float(np.sqrt(np.mean(e**2))) if n else '',Bias=float(np.mean(e)) if n else '',relative_error=float(np.mean(abs(e)/b)) if n else '',R2=float(1-np.sum(e**2)/ss) if n>=3 and ss>0 else '',status='PROVISIONAL_AUTOMATIC_GEOMETRY_LIMITS_APPLY' if n else 'NO_REAL_PAIRED_MANUAL_REFERENCE'))
 csvwrite(OUT/'phenotype/phenotype_errors.csv',summary)
 if paired:csvwrite(OUT/'phenotype/paired_measurements.csv',paired)
 else:
  # Header-only file: no fabricated paired samples or zero-valued errors.
  with (OUT/'phenotype/paired_measurements.csv').open('w') as f:csv.writer(f).writerow(['plant_id','leaf_id','trait','automatic_value','manual_value','error','relative_error','capture_session','measured_at','measurement_tool','operator','automatic_status'])
 write(OUT/'phenotype/pairing_validation.json',dict(n_pairs=len(paired),rows=audit,rule='Explicit pair_confirmed plus plantID, stableLeafID, capture_session, measured_at, operator and tool required. Historical ordinal/virtual values excluded.',error_convention='automatic minus manual; relative_error summary is mean absolute relative error',formal_accuracy_limit='Current extracted leaves are partial; any future paired errors retain provisional automatic geometry status'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for ax,trait in zip(axes.ravel(),TRAITS):
  p=[r for r in paired if r['trait']==trait];ax.set_title(trait,fontsize=10)
  if p:
   x=np.array([r['manual_value'] for r in p]);y=np.array([r['automatic_value'] for r in p]);ax.scatter(x,y);lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color='gray',linestyle='--');ax.set_xlabel('Manual');ax.set_ylabel('Automatic (provisional)')
  else:ax.text(.5,.5,'n = 0\nNo verified manual pairs',ha='center',va='center',transform=ax.transAxes);ax.set_xticks([]);ax.set_yticks([])
 fig.suptitle('Paired validation: no historical virtual references');fig.tight_layout();fig.savefig(OUT/'phenotype/paired_scatter.png',dpi=150);plt.close(fig)
 print('REAL_MANUAL_PAIRS',len(paired),flush=True)

if __name__=='__main__':main()
