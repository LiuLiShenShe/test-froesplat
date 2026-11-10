"""Validate the requested allocator in the sole locked E13 GPU driver."""
import qgs_common as q
import torch,warnings,gc,time
def main():
    config=q.os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')
    assert 'expandable_segments:True' in config,config
    torch.set_num_threads(8);started=time.time();records=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for count in [8_000_000,9_000_000,10_000_000]:
            value=torch.zeros(count,device='cuda',dtype=torch.float32)
            assert float(value.sum())==0
            torch.cuda.synchronize()
            records.append(dict(elements=count,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
            del value;gc.collect();torch.cuda.empty_cache()
        messages=[str(w.message) for w in caught]
    assert not any('expandable_segments' in m and 'not supported' in m for m in messages),messages
    q.write(q.ROOT/'allocator_validation.json',dict(status='passed',physical_gpu=1,torch_version=torch.__version__,
        allocator=config,varying_size_float32_allocations=records,warnings=messages,elapsed_seconds=time.time()-started,
        changes_to_native_objective_or_representation=False,source='https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst'))
    print('Actual GPU1 expandable allocator validation passed',config,flush=True)
if __name__=='__main__':main()
