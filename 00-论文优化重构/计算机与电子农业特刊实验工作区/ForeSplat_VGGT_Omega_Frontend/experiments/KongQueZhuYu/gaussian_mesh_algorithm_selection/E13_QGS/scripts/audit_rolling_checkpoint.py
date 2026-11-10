"""CPU-only observation of the real, atomically replaced native recovery state."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import torch

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(8)
    checkpoint=ROOT/'recovery/latest_native_state.pth'
    metadata=ROOT/'recovery/latest_native_state.provenance.json'
    before=json.loads(metadata.read_text())
    state,iteration=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert len(state)==12 and iteration==before['iteration']
    n=len(state[1]);assert n>0 and state[4].shape==(n,6)
    invalid=torch.zeros(n,dtype=torch.bool);fields={}
    for name,index in [('xyz',1),('SH_dc',2),('SH_rest',3),('raw_scale6',4),('raw_quaternion4',5),('raw_opacity',6)]:
        array=state[index].detach();assert len(array)==n
        bad=~torch.isfinite(array).reshape(n,-1).all(1);invalid|=bad
        fields[name]=dict(shape=list(array.shape),nonfinite_rows=int(bad.sum()),
                          nonfinite_values=int((~torch.isfinite(array)).sum()))
    optimizer={}
    for group in state[10]['param_groups']:
        native_state=state[10]['state'].get(group['params'][0],{})
        optimizer[group.get('name',str(group['params'][0]))]={
            key:int((~torch.isfinite(value)).sum()) for key,value in native_state.items()
            if isinstance(value,torch.Tensor)}
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    assert json.loads(metadata.read_text())==before,'Rolling checkpoint changed during CPU audit; retry a stable snapshot'
    result=dict(status='observed_native_partial',iteration=iteration,primitive_count=n,
                nonfinite_primitives=int(invalid.sum()),nonfinite_fraction=float(invalid.sum())/n,
                raw_fields=fields,optimizer_nonfinite_values=optimizer,
                checkpoint_path_at_observation=str(checkpoint.resolve()),
                file_sha256_at_observation=digest.hexdigest(),file_size_at_observation=checkpoint.stat().st_size,
                observed_utc=datetime.now(timezone.utc).isoformat(),
                path_role='Rolling recovery path; future checkpoints replace it. No separately retained intermediate artifact is claimed.',
                native_model_unfiltered=True,terminal_result_not_claimed=True,CPU_only=True)
    target=ROOT/f'recovery/native_numerical_observation_{iteration}.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['status','iteration','primitive_count','nonfinite_primitives','nonfinite_fraction']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
