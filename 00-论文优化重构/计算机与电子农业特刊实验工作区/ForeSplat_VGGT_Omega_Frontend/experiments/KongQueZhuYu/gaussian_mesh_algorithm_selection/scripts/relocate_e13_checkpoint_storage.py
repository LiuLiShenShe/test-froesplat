"""Move inactive E13 checkpoint directories to NVMe, verifying every byte."""
import datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT / 'E13_QGS'
TARGET = Path('/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS')

def read(path):
    return json.loads(path.read_text())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    lock = (ROOT / 'storage_relocation.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not TARGET.exists(), 'Never overwrite an existing model storage directory'
    state = read(ROOT/'STATUS.json')
    assert state['status']=='running' and 13300 <= state['iteration'] < 13600
    metadata = read(ROOT/'recovery/latest_native_state.provenance.json')
    assert metadata['iteration']==13000
    assert not (ROOT/'recovery/latest_native_state.pth.tmp').exists()
    current = read(ROOT/'recovery/STABILITY_OBSERVER_STATUS.json')
    pid = current['pid']
    assert current['status']=='waiting_for_next_real_checkpoint'
    script = ROOT/'scripts/observe_native_stability.py'
    assert str(script).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f'/proc/{pid}').exists():
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Idle CPU observer did not exit')
    TARGET.mkdir(parents=True)
    records = []
    try:
        for name in ['checkpoints', 'recovery']:
            source = ROOT/name
            assert source.is_dir() and not source.is_symlink()
            files = sorted(p for p in source.rglob('*') if p.is_file())
            snapshot = {str(p.relative_to(source)):(p.stat().st_size, p.stat().st_mtime_ns, digest(p)) for p in files}
            shutil.copytree(source, TARGET/name)
            for relative, (size, mtime, sha) in snapshot.items():
                old, new = source/relative, TARGET/name/relative
                assert (old.stat().st_size, old.stat().st_mtime_ns)==(size,mtime), 'Source changed while copying'
                assert new.stat().st_size==size and digest(new)==sha, 'Relocated bytes do not match source'
                records.append(dict(original_absolute_path=str(old), new_absolute_path=str(new.resolve()), file_size_bytes=size, sha256=sha))
        assert read(ROOT/'recovery/latest_native_state.provenance.json')==metadata
        after = read(ROOT/'STATUS.json')
        assert after['status']=='running' and after['iteration']<13800, 'Insufficient inactive interval before14K save; retain original directories'
        assert not (ROOT/'recovery/latest_native_state.pth.tmp').exists()
        for name in ['checkpoints','recovery']:
            source, staging = ROOT/name, ROOT/(name+'_verified_relocation_staging')
            assert not staging.exists()
            source.rename(staging)
            try:
                source.symlink_to(TARGET/name, target_is_directory=True)
            except BaseException:
                staging.rename(source)
                raise
            assert source.resolve()==(TARGET/name).resolve()
            for record in records:
                original = Path(record['original_absolute_path'])
                if original.is_relative_to(source):
                    assert original.resolve()==Path(record['new_absolute_path'])
            # A cross-filesystem move needs a temporary copy. Remove only the
            # fully verified source staging so large model bytes are not duplicated.
            shutil.rmtree(staging)
        result = dict(status='complete_byte_verified_storage_move', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            original_directory_aliases=[str(ROOT/n) for n in ['checkpoints','recovery']], actual_NVMe_storage=str(TARGET),
            files=records, original_aliases_preserved=True, native_GPU_training_interrupted=False,
            trained_parameters_optimizer_camera_loss_and_densification_unchanged=True,
            scope='Artifact storage only. Early training/checkpoint timings used rotational/data; later saves use root NVMe. Actual wall times include these different I/O conditions and observed shared GPU use.')
        (ROOT/'STORAGE_RELOCATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('E13 checkpoint storage relocated with identical SHA256',len(records),'files',sum(r['file_size_bytes'] for r in records),'bytes',flush=True)
    finally:
        with (ROOT/'logs/native_stability_observer.log').open('a') as stream:
            child = subprocess.Popen(['uv','run','--no-project','--python',str(ROOT/'.venv/bin/python'),'python',str(script)],cwd=OUT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with (OUT/'actual_commands.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(action='CPU_observer_restart_after_storage_validation',old_pid=pid,new_pid=child.pid,
                script=str(script),GPU_training_untouched=True,time=time.time()))+'\n')

if __name__ == '__main__':
    main()
