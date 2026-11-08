"""Prepared observation wrapper; not executed as part of this CPU audit.

Future authorized replay: uv run --python OMEGA/.venv/bin/python this.py
--events NEW/events.jsonl -- <original train_foresplat.py arguments>.
Do not use --smoke: it disables the mechanism under investigation. Original
GaussianModel methods, RNG draws and returned counts are preserved. Observation
adds deterministic reductions and synchronization, hence changes wall time.
An early 1000-step replay cannot resolve E5 growth at 7000-15000.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

AUDIT=Path(__file__).resolve().parents[1]
GGGS=AUDIT.parents[2]/'gggs_community'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--events',required=True)
    parser.add_argument('training_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    argv=args.training_args
    if argv and argv[0]=='--':argv=argv[1:]
    if '--smoke' in argv:parser.error('--smoke disables densification; unsuitable for this diagnostic')
    if '--model_path' not in argv and '-m' not in argv:parser.error('Explicit new --model_path is required')
    idx=argv.index('--model_path') if '--model_path' in argv else argv.index('-m')
    model_path=Path(argv[idx+1]).resolve()
    if model_path.exists() and any(model_path.iterdir()):parser.error('Refuse nonempty model output; preserve historical experiments')
    event_path=Path(args.events).resolve()
    if event_path.exists():parser.error('Refuse overwrite of existing event file')
    event_path.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(GGGS))
    spec=importlib.util.spec_from_file_location('foresplat_replay_train',GGGS/'train_foresplat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import torch
    model_cls=module.GaussianModel
    original=model_cls.densify_and_prune
    state={'event':0}
    def instrumented(self,max_grad,min_opacity,extent,max_screen_size):
        state['event']+=1
        grads=self.xyz_gradient_accum/self.denom;grads[grads.isnan()]=0.
        absolute=self.xyz_gradient_accum_abs/self.denom;absolute[absolute.isnan()]=0.
        normal=torch.norm(grads,dim=-1)>=max_grad
        ratio=normal.float().mean();q=torch.quantile(absolute.reshape(-1),1-ratio)
        zero=(absolute.reshape(-1)==0)
        abs_selected=absolute.reshape(-1)>=q
        large=self.get_scaling.max(dim=1).values>self.percent_dense*extent
        row={'event_index':state['event'],'inferred_iteration_default_schedule':500+100*state['event'],
             'schedule_note':'Iteration inference valid only with unchanged original from500/interval100; event_index is authoritative.',
             'before':len(self._xyz),'ratio':float(ratio),'Q':float(q),'Q_is_zero':bool(q==0),
             'denom_zero':int((self.denom==0).sum()),'absolute_gradient_zero':int(zero.sum()),
             'zero_selected_by_abs_branch':int((zero&abs_selected).sum()),
             'normal_selected':int(normal.sum()),'large_scale_selected_by_normal':int((normal&large).sum()),
             'small_scale_clone_selected':int((normal&~large).sum()),'abs_branch_selected':int(abs_selected.sum()),
             'raw_opacity_below_prune_before':int((self.get_opacity<min_opacity).sum()),
             'max_grad':max_grad,'min_opacity':min_opacity,'extent':float(extent),'caller_max_screen_size':max_screen_size}
        result=original(self,max_grad,min_opacity,extent,max_screen_size)
        row.update(after=len(self._xyz),clone_added=int(result[0]),split_net_added=int(result[1]),raw_opacity_removed=int(result[2]))
        with event_path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        return result
    model_cls.densify_and_prune=instrumented
    provenance={'argv':argv,'source_sha256':hashlib.sha256((GGGS/'scene/gaussian_model.py').read_bytes()).hexdigest(),
                'torch_version':torch.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
                'observer_only':True,'random_calls_added':0,'events_output':str(event_path)}
    event_path.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return module.main(argv)

if __name__=='__main__':raise SystemExit(main())
