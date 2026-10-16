"""Verify historical hash records and snapshot all protected model assets."""
from common import *
from concurrent.futures import ThreadPoolExecutor


def build():
    expected = {}
    def add(path, digest):
        if Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}:
            return
        p = str(Path(path).resolve())
        if p in expected and expected[p] != digest:
            raise ValueError(f'Conflicting historical hashes: {p}')
        expected[p] = digest
    for r in read_json(E3 / 'configs/FREEZE.json')['protected']:
        add(r['path'], r['sha256'])
    for directory in [E1, E2]:
        for path in sorted((directory / 'reports').glob('*VALIDATION.json')):
            d = read_json(path)
            for pathkey, hashkey in [('checkpoint_path','checkpoint_sha256'), ('final_PLY_path','final_PLY_sha256'), ('PLY_path','PLY_sha256')]:
                if pathkey in d:
                    add(d[pathkey],d[hashkey])
    for p in sorted((E3 / 'reports').glob('*RUN.json')):
        d = read_json(p)
        if d.get('model_path') and d.get('checkpoint_sha256'):
            add(Path(d['model_path']) / 'chkpnt30000.pth', d['checkpoint_sha256'])
            add(Path(d['model_path']) / 'point_cloud/iteration_30000/point_cloud.ply', d['PLY_sha256'])
    for path,digest in read_json(EF / 'logs/EXECUTION.json')['checked_input_sha256'].items():
        # Phase F records the complete image dataset as execution inputs. The
        # G0 gate concerns frozen model/PLY assets and their manifests, not a
        # second full-image-dataset checksum pass.
        suffix = Path(path).suffix.lower()
        if suffix in {'.ply', '.pth', '.json', '.csv', '.txt'} or Path(path).name == 'cfg_args':
            add(path,digest)
    model_roots = [Path('/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1/phase20e1_outputs'),
                   Path('/data/fj/F2DMAS_phase20e2_data/694743e7_v1/phase20e2_outputs'),
                   Path('/data/fj/F2DMAS_phase20e3_data/1771e53f_v1/training'),
                   Path('/data/fj/F2DMAS_phase20e3_data/1771e53f_v1/meshes')]
    paths = {Path(p) for p in expected}
    for base in model_roots:
        for extension in ['*.ply','*.pth']:
            paths.update(p.resolve() for p in base.rglob(extension))
    # Keep this index scoped to frozen inputs and model assets. Hashing every
    # tracked file would reread unrelated historical binaries and adds no
    # protection beyond git status/diff for this plugin-only phase.
    def check(p):
        actual = sha(p)
        historical = expected.get(str(p), '')
        if historical and actual != historical:
            raise ValueError(f'Historical SHA256 mismatch: {p}')
        return {'path':str(p),'sha256':actual,'bytes':p.stat().st_size,
                'historical_expected_sha256':historical,'historical_match':actual==historical if historical else 'BASELINE_SNAPSHOT',
                'kind':'MODEL_OR_MESH' if p.suffix in ['.ply','.pth'] else 'PROTECTED_FILE'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check,sorted(paths)))
    write_csv(STAGE/'configs/PROTECTED_INPUT_HASHES.csv',checks)
    write_json(STAGE/'reports/PREFLIGHT.json',{'status':'PASS','time':now(),'baseline':BASELINE,
        'checked_files':len(checks),'historical_hashes_verified':len(expected),'total_bytes_hashed':sum(r['bytes'] for r in checks),
        'model_or_mesh_files':sum(r['kind']=='MODEL_OR_MESH' for r in checks),
        'index_sha256':sha(STAGE/'configs/PROTECTED_INPUT_HASHES.csv')})
    print('PREFLIGHT_PASS',len(checks),len(expected),flush=True)


def verify():
    checks = rows(STAGE/'configs/PROTECTED_INPUT_HASHES.csv')
    def check(r):
        actual = sha(r['path'])
        if actual != r['sha256']:
            raise ValueError(f'Protected input changed: {r["path"]}')
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check,checks))
    write_json(STAGE/'reports/FINAL_INPUT_RECHECK.json',{'status':'PASS','time':now(),
        'checked_files':len(checks),'all_unchanged':True,'protected_index_sha256':sha(STAGE/'configs/PROTECTED_INPUT_HASHES.csv')})
    print('FINAL_INPUT_RECHECK_PASS',len(checks),flush=True)


if __name__=='__main__':
    if '--verify' in sys.argv:
        verify()
    else:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASELINE
        assert not (STAGE/'reports/PREFLIGHT.json').exists()
        build()
