"""Execute only the user's enumerated cleanup, with fixed paths and preservation guards."""
import os,sys,stat,shutil,json,hashlib,time
from pathlib import Path
from datetime import datetime

PROJECT=Path('/data/fj/F2DMAS')
AREA=PROJECT/'00-论文优化重构/计算机与电子农业特刊实验工作区'
AUDIT=AREA/'storage_audit_20261108'
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True)
COLMAP=PROJECT/'11-COLMAP-fuse/run_20260302_153856'
EXPERIMENTS=AREA/'ForeSplat_Fast3R_AMB3R_Frontend/experiments/KongQueZhuYu'
TEMP=Path('/tmp/foresplat_KongQueZhuYu/Fast3R_AMB3R_SfM_full208_original_dtype_20261107')
SMOKE=['Fast3R_smoke_v2','Fast3R_ramp16','Fast3R_ramp64','AMB3R_SfM_smoke32_original_dtype']
HEAVY={'pointcloud','pointmaps','confidence','raw_predictions','local_predictions'}
TEXT={'.json','.jsonl','.md','.csv','.yaml','.yml','.txt','.log'}
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def ensure_regular_path(path):
    # Check every component, so neither a root nor nested candidate can redirect deletion.
    for p in [path,*path.parents]:
        if p.is_symlink():raise RuntimeError('Refuse symlink candidate ancestor: '+str(p))
    if not path.exists():raise RuntimeError('Missing approved candidate: '+str(path))
def walk_files(root):
    if root.is_file():yield root;return
    for d,ds,fs in os.walk(root,followlinks=False):
        for name in ds:
            if (Path(d)/name).is_symlink():raise RuntimeError('Nested symlink candidate '+str(Path(d)/name))
        for name in fs:
            p=Path(d)/name
            if p.is_symlink():raise RuntimeError('Nested symlink candidate '+str(p))
            yield p
def estimate(root):
    entries=[];inodes={};directory_blocks=0
    for d,ds,fs in os.walk(root,followlinks=False) if root.is_dir() else []:
        directory_blocks+=os.lstat(d).st_blocks*512
    for p in walk_files(root):
        s=p.stat();entries.append((p,s))
        k=(s.st_dev,s.st_ino)
        item=inodes.setdefault(k,[s.st_blocks*512,s.st_nlink,0]);item[2]+=1
    return {'file_count':len(entries),'logical_bytes':sum(s.st_size for p,s in entries),
            'unique_allocated_bytes':sum(v[0] for v in inodes.values())+directory_blocks,
            'reclaimable_allocated_bytes':sum(v[0] for v in inodes.values() if v[1]==v[2])+directory_blocks}
def tree_metadata_signature(root,exclude=()):
    rows=[]
    for p in walk_files(root):
        if any(p==x or x in p.parents for x in exclude):continue
        s=p.stat();rows.append((str(p.relative_to(root)),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_blocks))
    rows.sort();data=json.dumps(rows,separators=(',',':')).encode()
    return {'files':len(rows),'logical_bytes':sum(r[3] for r in rows),'metadata_sha256':hashlib.sha256(data).hexdigest()}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def space():
    result={}
    for path in ['/data','/tmp']:
        s=os.statvfs(path)
        result[path]={'device':os.stat(path).st_dev,'available_bytes':s.f_bavail*s.f_frsize,'free_bytes':s.f_bfree*s.f_frsize}
    return result

record=json.loads((AUDIT/'colmap_intermediate_validation.json').read_text())
tmpcheck=json.loads((OUT/'temporary_duplicate_preflight.json').read_text())
assert tmpcheck['problems']==[] and tmpcheck['source']==str(TEMP)
assert tmpcheck['source_files']==5145 and tmpcheck['NPZ_entry_CRC_sizes_checked']==1936
assert record['all_20_final_clouds_exist_and_nonempty']
targets=[];fuse_guard={}
for cloud in record['retained_final_clouds']:
    p=Path(cloud['path']);assert p.parent.parent==COLMAP
    s=p.stat();assert s.st_size>0
    fuse_guard[str(p)]={'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns}
for name in ['depth_maps','normal_maps','consistency_graphs']:
    for plant in sorted(Path(p).parent.name for p in fuse_guard):
        p=COLMAP/plant/'dense/stereo'/name
        ensure_regular_path(p);targets.append({'category':'old_colmap_intermediate','path':str(p),'kind':'directory'})
assert len(targets)==60
smoke_excludes={}
for name in SMOKE:
    root=EXPERIMENTS/name;ensure_regular_path(root);ex=[]
    for sub in sorted(HEAVY):
        p=root/sub
        if p.exists():
            ensure_regular_path(p);targets.append({'category':'early_smoke_predictions','path':str(p),'kind':'directory'});ex.append(p)
    raw=root/'official_memory_raw.npz'
    if raw.exists():
        ensure_regular_path(raw);targets.append({'category':'early_smoke_predictions','path':str(raw),'kind':'file'});ex.append(raw)
    smoke_excludes[name]=ex
ensure_regular_path(TEMP);targets.append({'category':'temporary_sfm_duplicate','path':str(TEMP),'kind':'directory'})
for row in targets:row.update(estimate(Path(row['path'])))

formal={}
for name in ['Fast3R','Fast3R_FP32','AMB3R_Base_OFF','AMB3R_Base_ON','AMB3R_SfM','Fast3R_AMB3R_SfM','VGGT_Omega_Existing']:
    path=(EXPERIMENTS/name).resolve(strict=True)
    assert not any(path==Path(t['path']) or Path(t['path']) in path.parents or path in Path(t['path']).parents for t in targets)
    formal[name]={'resolved_path':str(path),'metadata_signature':tree_metadata_signature(path),
                  'record_sha256':{str(p.relative_to(path)):digest(p) for p in path.iterdir() if p.is_file() and p.suffix in TEXT}}
smoke_guard={name:tree_metadata_signature(EXPERIMENTS/name,ex) for name,ex in smoke_excludes.items()}
write('protected_artifacts_before.json',{'final_fused_clouds':fuse_guard,'formal208_bundles':formal,'preserved_smoke_inputs_cameras_reports':smoke_guard})

# Archive text records nested in the approved heavy subdirectories before removing arrays.
archive=[]
for row in targets:
    if row['category']!='early_smoke_predictions':continue
    path=Path(row['path'])
    for p in walk_files(path):
        if p.suffix not in TEXT:continue
        dest=OUT/'retained_smoke_records'/p.relative_to(EXPERIMENTS)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert digest(p)==digest(dest)
        archive.append({'source':str(p),'archive':str(dest),'sha256':digest(dest)})
write('archived_smoke_text_records.json',archive)
write('deletion_manifest.json',{'created_at':datetime.now().isoformat(),'user_authorized':True,'entries':targets,'only_categories':['old_colmap_intermediate','early_smoke_predictions','temporary_sfm_duplicate']})
before=space();write('disk_space_before.json',before)
print('PREFLIGHT_PASSED',len(targets),'targets; formal208 +20 final clouds guarded;',len(archive),'nested text records archived',flush=True)
success=[]
with (OUT/'deletion_log.jsonl').open('a') as log:
    for i,row in enumerate(targets,1):
        path=Path(row['path']);ensure_regular_path(path)
        if row['kind']=='directory':shutil.rmtree(path)
        else:path.unlink()
        assert not path.exists()
        done={'completed_at':datetime.now().isoformat(),'category':row['category'],'path':str(path),'reclaimable_allocated_bytes':row['reclaimable_allocated_bytes'],'removed':True}
        log.write(json.dumps(done,ensure_ascii=False)+'\n');log.flush();success.append(done)
        if i%10==0 or i==len(targets):print('DELETED',i,'/',len(targets),flush=True)

errors=[]
for p,saved in fuse_guard.items():
    s=Path(p).stat();current={'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns}
    if current!=saved:errors.append('Final PLY modified: '+p)
for name,guard in formal.items():
    root=(EXPERIMENTS/name).resolve(strict=True)
    if str(root)!=guard['resolved_path'] or tree_metadata_signature(root)!=guard['metadata_signature']:errors.append('Formal208 tree changed '+name)
    for rel,h in guard['record_sha256'].items():
        if digest(root/rel)!=h:errors.append('Formal208 metadata changed '+name+'/'+rel)
for name,saved in smoke_guard.items():
    root=EXPERIMENTS/name
    if tree_metadata_signature(root)!=saved:errors.append('Preserved smoke record changed '+name)
for a in archive:
    if digest(Path(a['archive']))!=a['sha256']:errors.append('Archive checksum changed '+a['archive'])
for row in targets:
    if Path(row['path']).exists():errors.append('Deletion target remains '+row['path'])
after=space();write('disk_space_after.json',after)
categories={}
for row in targets:
    c=categories.setdefault(row['category'],{'targets':0,'removed_files':0,'reclaimable_allocated_bytes':0})
    c['targets']+=1;c['removed_files']+=row['file_count'];c['reclaimable_allocated_bytes']+=row['reclaimable_allocated_bytes']
result={'status':'complete' if not errors else 'preservation_check_failed','errors':errors,'user_authorized_categories':list(categories),'removed_targets':len(success),'categories':categories,'final_fused_clouds_preserved':len(fuse_guard),'formal208_bundles_preserved':len(formal),'smoke_inputs_cameras_reports_preserved':len(smoke_guard),'archived_nested_smoke_text_records':len(archive),'space_before':before,'space_after':after,'available_delta':{p:after[p]['available_bytes']-before[p]['available_bytes'] for p in before},'measurement_note':'Reclaimable blocks computed per selected inode incl dirs. df/statvfs delta also includes concurrent unrelated filesystem activity. No CUDA or model execution.'}
write('CLEANUP_RESULT.json',result)
for name in SMOKE:
    (EXPERIMENTS/name/'CLEANUP_NOTICE.md').write_text('# 历史冒烟/爬坡结果存储清理\n\n用户于2026-11-08授权清理早期大型预测、置信度和点云；本目录原日志、报告、参数、输入RGB/Mask、相机与可视化保留。原status记录的是历史推理完成状态，原bundle引用的大型数组现已按授权删除，不能再视为完整Geometry Bundle。\n\n清理记录：'+str(OUT/'CLEANUP_RESULT.json')+'\n局部调用文本记录备份：'+str(OUT/'retained_smoke_records'/name)+'\n')
assert not errors,errors
summary=['# 已授权存储清理执行报告','', '仅执行用户选定的三类清理；没有使用GPU、conda或重跑模型。','', '| 类别 | 删除文件 | 释放分配块 |','|---|---:|---:|']
for category,value in categories.items():summary.append(f'| {category} | {value["removed_files"]:,} | {value["reclaimable_allocated_bytes"]/1024**3:.2f} GiB |')
summary+=['',f'数据盘可用空间变化：{result["available_delta"]["/data"]/1024**3:.2f} GiB；系统盘可用空间变化：{result["available_delta"]["/tmp"]/1024**3:.2f} GiB。其他任务可能同时影响df，逐项分配块见清单。','',f'保留验证通过：20份最终融合PLY、7份正式208 Bundle、4套早期运行的报告/参数/RGB/Mask/相机/可视化。{len(archive)}份大型预测目录中的文本记录另存备份。','', '所有模型主权重、环境、缓存、E1–E6 Checkpoint、原始图像/Mask/GT、根因审计结果均不在删除清单中。','', '历史存储审计是清理前快照，未改写。旧COLMAP直接重融合需重新生成深度/法向；早期Smoke Bundle的数组已清理，目录新增CLEANUP_NOTICE记录。','', '[删除清单](deletion_manifest.json)、[逐项执行日志](deletion_log.jsonl)、[最终核验](CLEANUP_RESULT.json)、[删除前保留证据](protected_artifacts_before.json)、[临时重复副本复核](temporary_duplicate_preflight.json)。','', '来源：本机文件系统、既有持久化核验和用户明确授权。博查检索返回403额度不足，无搜索结果可引用。']
(OUT/'CLEANUP_REPORT.md').write_text('\n'.join(summary)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
