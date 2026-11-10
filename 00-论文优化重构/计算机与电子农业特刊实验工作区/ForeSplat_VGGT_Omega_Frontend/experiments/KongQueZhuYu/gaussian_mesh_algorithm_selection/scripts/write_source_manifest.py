"""Record real pinned source revisions, tracked patches and execution-script hashes."""
import datetime,hashlib,json,subprocess
from pathlib import Path
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1]
DEST=OUT/'source_manifest';DEST.mkdir(exist_ok=True)
repos=[
    ('GGGS',OUT/'P1_GGGS_Tetrahedra/official','23b418c2f0f19b30b251acad6fe13445dae968c7','https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids'),
    ('QGS',OUT/'E13_QGS/official','74d05c945e99fcaef7afe5a8831903be71ad9b55','https://github.com/will-zzy/QGS'),
    ('2D_SuGaR',OUT/'E14_2D_SuGaR/official','e62770ec6ab50076ebb6c0e3b4ac39f777f14d27','https://github.com/prajwalcr/2d-sugar'),
    ('GS2Mesh',OUT/'P3_GS2Mesh/official','560f3e8a2349aaad24fe9444ef1ca2770aeec5ab','https://github.com/yanivw12/gs2mesh')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])
records=[]
for name,repo,expected,url in repos:
    assert repo.is_dir(),repo
    commit=git(repo,'rev-parse','HEAD').decode().strip();assert commit==expected,(name,commit)
    diff=git(repo,'diff','HEAD','--');patch=DEST/(name+'_tracked_worktree.patch');patch.write_bytes(diff)
    submodules=git(repo,'submodule','status','--recursive').decode(errors='replace')
    records.append(dict(method=name,official_url=url,official_commit=commit,actual_source=str(repo.resolve()),tracked_patch=str(patch.resolve()),tracked_patch_sha256=sha(patch),submodule_status=submodules))

scripts=[]
for folder in [OUT/'scripts',*[OUT/x/'scripts' for x in ['P3_GS2Mesh','E13_QGS','E14_2D_SuGaR']]]:
    for path in sorted(folder.glob('*.py')):
        scripts.append(dict(path=str(path.resolve()),relative_path=str(path.relative_to(OUT)),file_size_bytes=path.stat().st_size,sha256=sha(path)))
patches=[]
for root in [OUT/x for x in ['P1_GGGS_Tetrahedra','P2_GGGS_Cleaning/Tetrahedra','P3_GS2Mesh','E12_GGGS_EO','E12_GGGS_EO/Tetrahedra','E13_QGS','E14_2D_SuGaR']]:
    for pattern in ['*.patch','patches/*.diff','patches/*.patch']:
        for path in sorted(root.glob(pattern)):
            patches.append(dict(path=str(path.resolve()),relative_path=str(path.relative_to(OUT)),file_size_bytes=path.stat().st_size,sha256=sha(path)))
adapted=[]
for label,folder in [('GGGS_corrected',OUT.parent/'geometry_rgb_dual_frontend/camera_interface/gggs_corrected'),('QGS_adapter',OUT/'E13_QGS/adapted'),('E12_EO',OUT/'E12_GGGS_EO/source')]:
    if not folder.is_dir():continue
    for rel in ['train.py','train_foresplat.py','scene/cameras.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','utils/loss_utils.py','utils/graphics_utils.py']:
        path=folder/rel
        if path.is_file():adapted.append(dict(runtime=label,path=str(path.resolve()),sha256=sha(path)))
write_status(DEST/'SOURCE_AND_SCRIPT_MANIFEST.json',dict(status='actual_current_sources',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repositories=records,scripts=scripts,runtime_sources=adapted,adaptation_patches=patches,
    note='Current source snapshot, not a retroactive claim about scripts loaded in already running processes. Individual actual training parameter/source-hash and patch records remain authoritative. Kernel/camera/loss hashes and checkpoint hashes are recorded by experiment audits. Ordinary point clouds are not native Gaussian representations.'))
print('Pinned source revisions and actual current scripts recorded',len(records),len(scripts),len(adapted),flush=True)
