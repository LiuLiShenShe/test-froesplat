"""Validate persisted audit metadata and links; never rerun inference or training."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['01_experiment_provenance.md','02_camera_geometry_audit.md',
            '03_frontend_geometry_audit.md','04_gggs_optimization_audit.md',
            '05_render_evaluation_audit.md','ROOT_CAUSE_ANALYSIS.md','evidence_summary.csv','controlled_verification.csv']

def main():
    errors=[]; report_hashes={}; scripts=[]; link_count=0; json_count=0; jsonl=[]
    for name in REQUIRED:
        p=ROOT/name
        if not p.is_file():
            errors.append(f'Missing required {name}')
            continue
        if name.endswith('.md'):
            text=p.read_text()
            if '<!-- TARGETED_CONTROLS -->' in text or '<!-- FINAL_DECISION -->' in text:
                errors.append(f'Unfinished report markers {name}')
        report_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in ROOT.rglob('*.md'):
        if 'runtime_checkpoint' in p.parts:
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
            target=target.strip().split(' "')[0].strip('<>')
            if target.startswith(('http:','https:','#','mailto:')):
                continue
            path_part=target.split('#')[0]
            if re.search(r':\d+$',path_part):
                path_part=re.sub(r':\d+$','',path_part)
            path=Path(path_part)
            path=path if path.is_absolute() else p.parent/path
            link_count+=1
            if not path.exists():
                errors.append(f'Broken link {p.relative_to(ROOT)} -> {target}')
    for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'controlled_fixes').rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        ast.parse(p.read_text(),filename=str(p))
        scripts.append(str(p.relative_to(ROOT)))
    for p in ROOT.rglob('*.json'):
        if p.name=='AUDIT_DELIVERY_VALIDATION.json':
            continue
        raw=p.read_text()
        try:
            json.loads(raw,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            json_count+=1
        except json.JSONDecodeError:
            if p.name!='multi_view.json':
                errors.append(f'Invalid JSON {p.relative_to(ROOT)}')
                continue
            try:
                rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
                if len(rows)!=208:
                    errors.append(f'Native208 JSONL size mismatch {p.relative_to(ROOT)}')
                jsonl.append({'path':str(p.relative_to(ROOT)), 'rows':len(rows)})
            except ValueError:
                errors.append(f'Invalid JSONL {p.relative_to(ROOT)}')
        except ValueError:
            errors.append(f'Nonfinite JSON constant {p.relative_to(ROOT)}')
    controls=[]
    for folder, names in [('view_count_control_v1',['Omega_70','Omega_110','Standard_70','Standard_110']),
                          ('background_control_v1',['Omega_RAP_baseline_24','Omega_background_restored_24',
                                                    'Standard_RAP_baseline_24','Standard_background_restored_24'])]:
        for name in names:
            base=ROOT/'controlled_fixes'/folder/name
            try:
                status=json.loads((base/'status.json').read_text())
                cameras=json.loads((base/'cameras.json').read_text())['frames']
                params=json.loads((base/'run_parameters.json').read_text())
                runtime=json.loads((base/'metadata.json').read_text())
                n=int(status['input_views'])
                if status.get('status')!='complete' or not status.get('GPU_inference_executed'):
                    errors.append(f'Control not actually completed {name}')
                if runtime.get('CUDA_VISIBLE_DEVICES')!='1' or runtime.get('training') is not False:
                    errors.append(f'GPU/training policy mismatch {name}')
                source=[r['source_index'] for r in cameras]
                if len(source)!=n or len(set(source))!=n or source!=sorted(source):
                    errors.append(f'Control ordered source mismatch {name}')
                for channel in ['depth','confidence','model_mask']:
                    paths=list((base/'per_frame').glob(channel+'_crop_*.npy'))
                    if len(paths)!=n:
                        errors.append(f'{name} {channel} files {len(paths)} != {n}')
                    for p in paths:
                        arr=np.load(p,mmap_mode='r',allow_pickle=False)
                        if arr.shape!=(cameras[0]['model_height'],cameras[0]['model_width']):
                            errors.append(f'Control shape mismatch {p.relative_to(ROOT)}')
                controls.append({'name':name,'views':n,'physical_GPU':'1','training':False,'status':status['status']})
            except (OSError,ValueError,KeyError) as exc:
                errors.append(f'Control metadata validation {name}: {exc}')
    with (ROOT/'evidence_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        evidence=list(csv.DictReader(f))
    for row in evidence:
        if not (ROOT/row['artifact']).exists():
            errors.append(f'Missing evidence artifact {row["id"]}')
    result={'status':'passed' if not errors else 'failed','errors':errors,
            'report_sha256':report_hashes,'python_AST_count':len(scripts),
            'local_markdown_links_checked':link_count,'portable_JSON_checked':json_count,
            'native_JSONL':jsonl,'targeted_inference_controls':controls,'evidence_rows':len(evidence),
            'scope':'Metadata, shape headers, link validity and script syntax only. No raw checkpoint/prediction payload reread or algorithm work.'}
    (ROOT/'AUDIT_DELIVERY_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'errors':errors,'scripts':len(scripts),'links':link_count,'controls':len(controls)},ensure_ascii=False))
    if errors:
        raise SystemExit(1)

if __name__=='__main__':
    main()
