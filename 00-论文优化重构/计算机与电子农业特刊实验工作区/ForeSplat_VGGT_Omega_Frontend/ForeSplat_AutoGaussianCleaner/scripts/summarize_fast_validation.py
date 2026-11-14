"""Copy small measured reports and build a reviewable index; never read PLY payloads."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/fast_ply_validation'
OUT=ROOT/'reports/fast_ply_20261114'
OUT.mkdir(parents=True,exist_ok=True)
resources={};index={};quality={}
for folder in sorted((BASE/'v2').iterdir()):
    report=folder/'cleanup_report.json'
    if not report.exists():continue
    plant=folder.name;resources[plant]=json.loads(report.read_text())
    index[plant]=dict(native_ply=str(folder/'cleaned_gaussians.ply'),cleanup_report=str(report))
    paired=folder/'evaluation/paired_quality.json'
    if paired.exists():
        measured=json.loads(paired.read_text());quality[plant]=measured
        shutil.copyfile(paired,OUT/(plant+'_paired_quality.json'))
        index[plant]['paired_evaluation']=str(paired)
        index[plant]['observed_contact']=str(folder/'evaluation/observed_contact.png')
probe=BASE/'anchor_probe/KongQueZhuYu/evaluation/paired_quality.json'
if probe.exists():
    shutil.copyfile(probe,OUT/'KongQueZhuYu_anchor_probe_quality.json')
    index['anchor_probe']=dict(paired_evaluation=str(probe),status='rejected: extra dark foreground error and weak incremental artifact removal; removed from default module',
                               cleanup_report=str(probe.parents[1]/'cleanup_report.json'),
                               native_ply=str(probe.parents[1]/'cleaned_gaussians.ply'),
                               source_patch=str(OUT/'rejected_anchor_probe.patch'))
(OUT/'measured_resources.json').write_text(json.dumps(resources,ensure_ascii=False,indent=2)+'\n')
(OUT/'artifact_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
for plant,j in quality.items():
    print(plant)
    for name,r in j['results'].items():
        if name.startswith('ablation_'):continue
        m=r['mean']
        print(name,r['gaussians'],'FG',round(m['fg_psnr'],5),'edge',round(m['boundary_coverage'],6),
              'thin',round(m['thin_coverage'],6),'hole',round(m['new_fg_alpha_hole_fraction'],6),
              'bg',round(m['bg_alpha_mass_per_fg'],6),'free_black',r['historical_free_black_pixels'])
