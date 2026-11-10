"""Plot observed native training trajectories and audited numerical quality."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'native_stability'

def rows(path):
 return list(csv.DictReader(path.open())) if path.is_file() else []

def main():
 DEST.mkdir(exist_ok=True)
 active=rows(ROOT/'train_metrics.csv');failed=rows(ROOT/'recovery/attempt_01_train_metrics.csv')
 parameters=json.loads((ROOT/'actual_training_parameters.json').read_text());preset=parameters.get('native_preset','base')
 resumed=bool(parameters.get('resume_checkpoint'));config=parameters['config']
 active_label='Retained prefix + resumed native trajectory' if resumed else 'Fresh native seed0 '+preset+' trajectory'
 assert active
 iterations=[int(r['iteration']) for r in active]
 assert iterations==sorted(set(iterations)), 'Active recorded trajectory must have unique increasing iterations.'
 audits=[]
 initial=ROOT/'partial_checkpoint_provenance_validation.json'
 if initial.is_file():
  for a in json.loads(initial.read_text()).get('records',[]):
   audits.append(dict(iteration=a['iteration'],primitives=a['primitives'],nonfinite_primitives=a['native_nonfinite_primitives'],nonfinite_fraction=a['native_nonfinite_fraction'],audit_path=str(initial.resolve()),source_checkpoint=a['checkpoint_path'],observed_sha256=a['checkpoint_sha256'],stage='fixed native checkpoint'))
 legacy_observation=ROOT/'recovery/native_nonfinite_checkpoint_8000.json'
 if legacy_observation.is_file():
  a=json.loads(legacy_observation.read_text())
  audits.append(dict(iteration=a['iteration'],primitives=a['primitive_count'],nonfinite_primitives=a['nonfinite_primitives'],nonfinite_fraction=a['nonfinite_fraction'],audit_path=str(legacy_observation.resolve()),source_checkpoint=a['actual_checkpoint'],observed_sha256=a['observed_checkpoint_sha256'],stage='observed rolling checkpoint'))
 for path in sorted((ROOT/'recovery').glob('*_checkpoint_validation_*.json')):
  a=json.loads(path.read_text())
  audits.append(dict(iteration=a['observed_iteration'],primitives=a['primitives'],nonfinite_primitives=a['nonfinite_native_primitives'],nonfinite_fraction=a['nonfinite_native_fraction'],audit_path=str(path.resolve()),source_checkpoint=a['checkpoint'],observed_sha256=a['observed_sha256'],stage='observed rolling checkpoint' if path.name.startswith('rolling') else 'fixed native checkpoint'))
 unique={r['iteration']:r for r in audits};audits=[unique[k] for k in sorted(unique)]
 if audits:
  with (DEST/'checkpoint_numeric_quality.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
 fig,axes=plt.subplots(2,2,figsize=(13.5,8.5),layout='constrained')
 for ax,key,label in [(axes[0,0],'quadratic_primitives','Native quadratic primitive count'),(axes[0,1],'image_loss','Native photometric image loss'),(axes[1,0],'dist_loss','Weighted native distance loss')]:
  for dataset,color,style,name in [(failed,'#777777','--','Initial attempt; stopped by OOM'),(active,'#1769aa','-',active_label)]:
   if not dataset:continue
   x=np.array([int(r['iteration']) for r in dataset]);y=np.array([float(r[key]) for r in dataset])
   valid=np.isfinite(y)&(y>0)
   ax.plot(x[valid],y[valid],color=color,linestyle=style,linewidth=1.2,label=name)
  ax.set_title(label);ax.set_yscale('log');ax.set_xlabel('Iteration');ax.grid(alpha=.2)
  if key=='dist_loss':
   ax.axvline(config['optimizer']['dist_from_iter'],color='#888888',linewidth=.8,linestyle=':')
  ax.set_xlim(0,max(iterations[-1],1000)+300)
 axes[0,0].legend(fontsize=8)
 ax=axes[1,1]
 if audits:
  ax.plot([a['iteration'] for a in audits],[a['nonfinite_primitives'] for a in audits],'o-',color='#a22323')
  for a in audits:ax.annotate(str(a['nonfinite_primitives']),(a['iteration'],a['nonfinite_primitives']),xytext=(3,6),textcoords='offset points',fontsize=8)
 ax.set_title('Nonfinite primitives in actual audited checkpoints')
 ax.set_yscale('symlog',linthresh=1);ax.set_xlabel('Saved iteration');ax.set_ylabel('Native primitives with any nonfinite parameter');ax.grid(alpha=.2)
 stage='30K native training recorded' if iterations[-1]==30000 else f'Partial native training: latest sampled iteration {iterations[-1]:,} / 30,000'
 fig.suptitle('E13 QGS '+preset+' configuration: actual numerical stability\n'+stage,fontsize=14)
 fig.text(.02,-.025,'Only sampled losses and actual saved states are shown. Missing checkpoint intervals are not certified finite.\nOriginal parameters and NaN payloads are retained; provenance success does not imply stable geometry.',fontsize=9)
 image=DEST/'native_training_stability.png';fig.savefig(image,dpi=170,bbox_inches='tight');plt.close(fig)
 events=[]
 for prev,curr in zip(active,active[1:]):
  ratio=float(curr['quadratic_primitives'])/float(prev['quadratic_primitives'])
  if ratio<.75 or ratio>1.5:
   events.append(dict(previous_sample=int(prev['iteration']),next_sample=int(curr['iteration']),previous_primitives=int(prev['quadratic_primitives']),next_primitives=int(curr['quadratic_primitives']),net_ratio=ratio))
 result=dict(status='complete_observed_partial' if iterations[-1]<30000 else 'complete_observed_30K',latest_recorded_iteration=iterations[-1],native_target_iterations=30000,
  active_metrics_path=str((ROOT/'train_metrics.csv').resolve()),discarded_attempt_metrics_path=str((ROOT/'recovery/attempt_01_train_metrics.csv').resolve()) if (ROOT/'recovery/attempt_01_train_metrics.csv').is_file() else '',native_preset=preset,
  checkpoint_audits=audits,large_sampled_population_changes=events,actual_visualization=str(image.resolve()),
  parameter_defects_retained=True,CPU_only=True,ordinary_Gaussian_covariance_interpretation=False,
  sampling='Training loss/counts are passive samples every100 iterations; snapshot audits observe existing saved checkpoints only, never all optimization steps.',
  scope='Actual official '+preset+' preset; other-preset attempts remain archived. No comparison to independent scan truth.',
  reset_and_pruning='Native opacity resets every3000 iterations, followed by100-step densify/prune with opacity threshold0.05. Logged counts precede that iteration pruning. Large sampled drops are net native optimization changes, not P2-style Gaussian Cleaning.',
  causality='Actual '+('resumed' if resumed else 'fresh seed0')+' '+preset+' run. Nonfinite parameter records do not isolate a CUDA derivative, input, allocator or optimization cause; archived base attempts are separate.')
 (DEST/'PROTOCOL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 content='# E13 原生训练稳定性：真实保存状态\n\n'+stage+'。这是一份执行中数值诊断，不是正式终态几何排序。\n\n'
 content+='![实际原生训练与Checkpoint数值状态](native_training_stability.png)\n\n'
 content+='|保存迭代|原生基元数|非有限基元|比例|真实核验记录|\n|---|---|---|---|---|\n'
 for a in audits:content+=f'|{a["iteration"]}|{a["primitives"]:,}|{a["nonfinite_primitives"]:,}|{a["nonfinite_fraction"]:.6%}|[{Path(a["audit_path"]).name}]({a["audit_path"]})|\n'
 content+='\n所有原生参数及NaN payload原样保留。Rolling路径会更新，表中SHA和计数只描述记录时实际观察到的状态；未另行保存该迭代的完整模型时，不把核验JSON冒称可下载Checkpoint。未核验间隔不能认定参数全部有限。\n\n'
 content+='原生opacity每3000步重置，下一次100步增密/删除使用0.05阈值；日志数量在当次增密/删除前采样。相邻采样数量骤降属于原生优化的净变化，不能称为另一个Gaussian Cleaning收益。距离项为已经乘lambda_dist的实际日志值，图中竖线使用实际距离项起始配置，非逐步完整峰值记录。\n\n'
 content+=f"当前实际{preset}配置：depth_ratio={config['pipeline']['depth_ratio']}、lambda_dist={config['optimizer']['lambda_dist']}、densify_grad_threshold={config['optimizer']['densify_grad_threshold']}。base失败状态另行归档，不与当前曲线合并；其他预设未训练时不推广结论。恢复标志={resumed}，原生捕获未保存RNG；不能单独把NaN归因于分配器。\n\n"
 content+='来源：[固定QGS训练与增密源码](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、实际train_metrics.csv及逐文件CPU核验。博查检索HTTP403不可用；不以训练数据补充实测结果。\n'
 (DEST/'NUMERICAL_STABILITY.md').write_text(content)
 print('Actual native stability plot',iterations[-1],'audited states',len(audits),flush=True)

if __name__=='__main__':main()
