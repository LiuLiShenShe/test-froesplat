import os,json,zipfile,hashlib,sys,time
from pathlib import Path
source=Path('/tmp/foresplat_KongQueZhuYu/Fast3R_AMB3R_SfM_full208_original_dtype_20261107')
area=Path('/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区')
target=area/'ForeSplat_Fast3R_AMB3R_Frontend/experiments/KongQueZhuYu/attempts/Fast3R_AMB3R_SfM_full208_original_dtype'
start=time.time(); problems=[]; metadata_differences=[]; count=0; total=0; allocated=0; archives=0; shared=0; regular_single=0
for p in source.rglob('*'):
    if not p.is_file() or p.is_symlink(): continue
    rel=p.relative_to(source); q=target/rel; s=p.stat(); count+=1;total+=s.st_size;allocated+=s.st_blocks*512
    if s.st_nlink==1: regular_single+=s.st_blocks*512
    if not q.is_file():problems.append([str(rel),'missing persistent']);continue
    t=q.stat()
    same=(s.st_dev,s.st_ino)==(t.st_dev,t.st_ino)
    shared+=int(same)
    if p.suffix=='.npz':
        archives+=1
        if s.st_size!=t.st_size: problems.append([str(rel),'NPZ size mismatch'])
        else:
            with zipfile.ZipFile(p) as a,zipfile.ZipFile(q) as b:
                aa=[(i.filename,i.CRC,i.file_size,i.compress_size) for i in a.infolist()]
                bb=[(i.filename,i.CRC,i.file_size,i.compress_size) for i in b.infolist()]
            if aa!=bb:problems.append([str(rel),'archive CRC/entry-size mismatch'])
    elif s.st_size!=t.st_size:
        if p.suffix in {'.json','.md','.csv'}:metadata_differences.append([str(rel),'size changed during finalization'])
        else:problems.append([str(rel),'data size mismatch'])
record={'source':str(source),'persistent':str(target),'source_files':count,'source_logical_bytes':total,'source_allocated_bytes':allocated,'minimum_reclaimable_allocated_single_links':regular_single,'source_target_same_inode_files':shared,'NPZ_entry_CRC_sizes_checked':archives,'problems':problems,'metadata_finalization_differences':metadata_differences,'existing_persistence_verification':json.loads((target/'persistence_verification.json').read_text()),'elapsed_seconds':time.time()-start,'verification_scope':'Current file-set/size inclusion and every NPZ central-directory CRC/uncompressed+compressed size; not a full payload checksum re-read. Persistent finalization may add/change metadata. Earlier persistence report includes actual array checks.'}
Path(sys.argv[1]).write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='existing_persistence_verification'},ensure_ascii=False,indent=2))
