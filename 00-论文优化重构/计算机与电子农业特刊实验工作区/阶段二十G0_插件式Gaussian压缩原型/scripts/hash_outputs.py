"""Index reproducible small outputs and external large artifacts."""
from common import *


def main():
    p=plan();records=[]
    for path in sorted(STAGE.rglob('*')):
        if path.is_file() and path.name not in {'CONFIG_AND_MODEL_HASH_INDEX.csv','LARGE_ASSET_INDEX.csv'}:
            records.append({'scope':'phase','path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                            'tracked_candidate':path.suffix not in {'.png','.pt','.npy','.pth','.ply','.so'}})
    large=[]
    roots=[Path(p['source_model']).parent,
           Path(p['source_native_model_dir']),
           DATA/'models',
           DATA/'mesh'/p['mesh']['run_id']]
    allowed={'.ply','.json','.csv','.npy','.txt'}
    for root in roots:
        if not root.exists():continue
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in allowed:
                large.append({'scope':'external_asset','path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                              'type':path.suffix.lower() or 'no_suffix'})
    write_csv(STAGE/'configs/CONFIG_AND_MODEL_HASH_INDEX.csv',records)
    write_csv(STAGE/'configs/LARGE_ASSET_INDEX.csv',large)
    write_json(STAGE/'reports/OUTPUT_HASH_INDEX.json',{'status':'PASS','phase_files':len(records),'external_files':len(large),
        'phase_bytes':sum(x['bytes'] for x in records),'external_bytes':sum(x['bytes'] for x in large),
        'large_float_caches_indexed_by_manifest': [
            str(STAGE/'configs/FLOAT_RENDER_INDEX.csv'),
            str(STAGE/'configs/MESH_FUSION_FLOAT_INDEX.csv')],
        'large_float_caches_rehashed_here': False,
        'source_model_sha256':p['source_model_sha256'],'source_unchanged':sha(p['source_model'])==p['source_model_sha256'],
        'index_sha256':{'phase':sha(STAGE/'configs/CONFIG_AND_MODEL_HASH_INDEX.csv'),
                        'external':sha(STAGE/'configs/LARGE_ASSET_INDEX.csv')}})
    print('HASH_INDEXED',len(records),len(large),flush=True)


if __name__=='__main__':main()
