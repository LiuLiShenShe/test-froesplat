"""Read-only project metadata inventory, no symlink traversal or file payload reads."""
import os, sys, stat, json, csv, time
from pathlib import Path
from collections import defaultdict, Counter

root=os.path.abspath(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
device=os.stat(root).st_dev; start=time.time()
dirs={}; parents={}; hard={}; large=[]; links=[]; errors=[]; cats=defaultdict(lambda:[0,0])
stack=[root]; files=0; checkpoint=[]; special=[]
cache_names={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.ipynb_checkpoints'}
while stack:
    path=stack.pop()
    try:
        st=os.lstat(path)
        dirs[path]=[st.st_blocks*512,st.st_size,0,st.st_blocks*512]
        parents[path]=os.path.dirname(path) if path!=root else None
        with os.scandir(path) as iterator:
            for entry in iterator:
                p=entry.path
                try: s=entry.stat(follow_symlinks=False)
                except OSError as exc: errors.append([p,str(exc)]); continue
                if stat.S_ISDIR(s.st_mode):
                    if s.st_dev==device: stack.append(p)
                    else: errors.append([p,'different filesystem skipped'])
                    continue
                if stat.S_ISLNK(s.st_mode):
                    dirs[path][0]+=s.st_blocks*512; dirs[path][1]+=s.st_size; dirs[path][3]+=s.st_blocks*512
                    links.append({'path':p,'target':os.readlink(p),'allocated':s.st_blocks*512})
                    continue
                if not stat.S_ISREG(s.st_mode): continue
                size=s.st_size; alloc=s.st_blocks*512
                files+=1; dirs[path][0]+=alloc; dirs[path][1]+=size; dirs[path][2]+=1
                ext=Path(p).suffix.lower() or '(none)'; cats[ext][0]+=alloc; cats[ext][1]+=1
                if s.st_nlink==1: dirs[path][3]+=alloc
                else:
                    item=hard.setdefault((s.st_dev,s.st_ino),{'allocated':alloc,'logical':size,'nlink':s.st_nlink,'paths':[]})
                    item['paths'].append(p)
                rec={'path':p,'logical_bytes':size,'allocated_bytes':alloc,'nlink':s.st_nlink,'device':s.st_dev,'inode':s.st_ino,'mtime':s.st_mtime}
                if alloc>=16*1024**2: large.append(rec)
                if ('chkpnt' in entry.name.lower() or 'checkpoint' in entry.name.lower()) and ext in {'.pth','.pt','.npz','.ckpt','.safetensors'}: checkpoint.append(rec)
                if p.endswith(('.incomplete','.tmp','.temp','.bak','.old')) or entry.name.startswith('core.'): special.append(rec)
    except OSError as exc: errors.append([path,str(exc)])
    if time.time()-start>0 and len(dirs)%2000==0: print(f'dirs={len(dirs)} files={files} elapsed={time.time()-start:.1f}s',flush=True)

# Reclaimable blocks are assigned to the lowest ancestor containing ALL hard links.
# If st_nlink exceeds observed paths, at least one external link survives, so zero.
for group in hard.values():
    paths=group['paths']
    if len(paths)==group['nlink']:
        lca=os.path.commonpath([os.path.dirname(p) for p in paths])
        if lca in dirs: dirs[lca][3]+=group['allocated']
for path in sorted(dirs,key=lambda x:x.count('/'),reverse=True):
    parent=parents[path]
    if parent in dirs:
        for k in range(4): dirs[parent][k]+=dirs[path][k]
unique={p:v[0] for p,v in dirs.items()}
for group in hard.values():
    occurrences=Counter()
    for p in group['paths']:
        d=os.path.dirname(p)
        while d in dirs:
            occurrences[d]+=1; d=parents[d]
    for d,n in occurrences.items():
        if n>1: unique[d]-=(n-1)*group['allocated']

fields=['path','unique_allocated_bytes','reclaimable_allocated_bytes','entry_allocated_bytes','logical_bytes','files']
with (out/'directories.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for p,v in sorted(dirs.items(),key=lambda item:unique[item[0]],reverse=True):
        w.writerow(dict(zip(fields,[p,unique[p],v[3],v[0],v[1],v[2]])))
with (out/'large_files.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','logical_bytes','allocated_bytes','nlink','device','inode','mtime']);w.writeheader();w.writerows(sorted(large,key=lambda x:x['allocated_bytes'],reverse=True))
groups=[v for v in hard.values() if v['allocated']>=1024**2]
(out/'hardlink_groups.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
(out/'symlinks.json').write_text(json.dumps(links,ensure_ascii=False,indent=2))
(out/'checkpoints.json').write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2))
(out/'temporary_named_files.json').write_text(json.dumps(special,ensure_ascii=False,indent=2))
top=[{'path':p, 'unique_allocated_GiB':round(unique[p]/1024**3,3),'reclaimable_GiB':round(v[3]/1024**3,3)} for p,v in dirs.items() if parents[p]==root]
cache_dirs=[{'path':p,'allocated':unique[p],'reclaimable':v[3]} for p,v in dirs.items() if os.path.basename(p) in cache_names]
summary={'root':root,'elapsed_seconds':time.time()-start,'files':files,'directories':len(dirs),'unique_allocated_bytes':unique[root],'reclaimable_if_entire_project_removed':dirs[root][3],'hardlinked_inodes':len(hard),'unobserved_external_hardlink_inodes':sum(len(v['paths'])<v['nlink'] for v in hard.values()),'top_level':sorted(top,key=lambda x:x['unique_allocated_GiB'],reverse=True),'cache_directories':cache_dirs,'errors':errors,'measurement':'st_blocks * 512; recursive unique inodes, no symlink traversal; reclaimable requires all st_nlink entries under candidate; no file data hashed or deleted','suffixes_including_hardlink_entries':dict(cats)}
(out/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in {'cache_directories','suffixes_including_hardlink_entries'}},ensure_ascii=False,indent=2))
