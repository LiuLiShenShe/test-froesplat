"""Check real native PLY payload lengths and PTH ZIP structure without tensor loads."""
import csv,json,struct,time,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
SIZES={'char':1,'uchar':1,'int8':1,'uint8':1,'short':2,'ushort':2,'int16':2,'uint16':2,'int':4,'uint':4,'int32':4,'uint32':4,'float':4,'float32':4,'double':8,'float64':8}

def ply(path):
    counts={};sizes={};element=None;fmt=None
    with path.open('rb') as stream:
        for _ in range(300):
            line=stream.readline().decode('ascii').strip();tokens=line.split()
            if tokens[:1]==['format']:fmt=tokens[1]
            elif tokens[:1]==['element']:
                element=tokens[1];counts[element]=int(tokens[2]);sizes[element]=0
            elif tokens[:1]==['property']:
                assert tokens[1]!='list','Native/point-cloud vertex records must have scalar fields'
                sizes[element]+=SIZES[tokens[1]]
            elif line=='end_header':header=stream.tell();break
        else:raise AssertionError('Missing actual PLY header end')
    assert fmt in ['binary_little_endian','binary_big_endian'],fmt
    expected=header+sum(counts[k]*sizes[k] for k in counts)
    assert path.stat().st_size==expected,('Native PLY payload length differs',expected,path.stat().st_size)
    return dict(status='passed_exact_native_payload_length',format=fmt,header_bytes=header,vertex_count=counts.get('vertex'),bytes_per_vertex=sizes.get('vertex'),expected_file_bytes=expected)

def pth(path):
    with zipfile.ZipFile(path) as archive:
        items=archive.infolist();names=[x.filename for x in items]
        assert any(n.endswith('/data.pkl') for n in names),'Native tensor archive missing pickle record'
        assert any('/data/' in n for n in names),'Native tensor storage records missing'
        for item in items:
            assert 0<=item.header_offset<path.stat().st_size
            assert item.header_offset+30+item.compress_size<=path.stat().st_size
        return dict(status='passed_ZIP_directory_structure_only',archive_entries=len(items),tensor_storage_entries=sum('/data/' in n for n in names),scope='Central-directory and entry ranges; tensor CRC and numeric provenance are separately audited, not claimed here')

def main():
    rows=list(csv.DictReader((OUT/'artifact_index/ALL_ARTIFACTS.csv').open()));records=[];seen=set()
    for row in rows:
        if row['artifact_type'] not in ['Gaussian PLY','Gaussian Model','Point Cloud PLY','Checkpoint']:continue
        path=Path(row['absolute_path']);key=str(path.resolve())
        if key in seen or path.suffix not in ['.ply','.pth','.pt']:continue
        if 'rolling' in row['status']:continue
        seen.add(key)
        try:
            result=ply(path) if path.suffix=='.ply' else pth(path)
        except Exception as error:
            known_bad='archive_unreadable' in row['status'] and '/failure_exports/' in key
            result=dict(status='known_unreadable_failed_export_confirmed' if known_bad else 'failed_structure_check',error=type(error).__name__+': '+str(error),usable_resume_checkpoint=False)
        records.append(dict(experiment_id=row['experiment_id'],artifact_type=row['artifact_type'],absolute_path=key,file_size_bytes=path.stat().st_size,**result))
    failures=[r for r in records if r['status']=='failed_structure_check']
    result=dict(status='passed_with_explicitly_unreadable_failed_export' if not failures and any(r['status']=='known_unreadable_failed_export_confirmed' for r in records) else 'passed' if not failures else 'failed',observed_time=time.time(),unique_actual_native_files=len(records),records=records,failures=failures,original_files_unchanged=True)
    target=OUT/'artifact_index/NATIVE_ARTIFACT_STRUCTURE_AUDIT.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'],'actual native PLY/PTH structures',len(records),'failures',len(failures),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
