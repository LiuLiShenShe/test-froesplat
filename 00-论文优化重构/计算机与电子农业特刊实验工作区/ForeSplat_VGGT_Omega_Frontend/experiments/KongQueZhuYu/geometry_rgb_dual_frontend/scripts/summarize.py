"""Common208 comparisons and native-pixel visual evidence, no rerender."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OMEGA=ROOT.parents[2];AUDIT=ROOT.parent/'root_cause_audit'
sys.path.insert(0,str(AUDIT/'scripts'))
from render_evaluate_common import psnr_region,gradients
EXPERIMENTS=['E1','E5','E7','E8','E9','E10']
NEW={'E7':'E7_Omega_Original_Interface','E8':'E8_Omega_Corrected_Interface','E9':'E9_Standard_Original_Interface','E10':'E10_Standard_Corrected_Interface'}
ids=[i for i in range(210) if i not in (44,206)];e6_ids=[ids[i] for i in range(0,208,26)]
OUT=ROOT/'render_comparison/visualizations';OUT.mkdir(exist_ok=True)
def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def montage(panels,labels,path,resize=None):
    ims=[Image.fromarray((np.clip(p,0,1)*255).round().astype(np.uint8)) for p in panels]
    if resize:ims=[im.resize(resize,Image.Resampling.NEAREST) for im in ims]
    w,h=ims[0].size;canvas=Image.new('RGB',(w*len(ims),h+24),'white');draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(ims,labels)):canvas.paste(im,(w*i,24));draw.text((w*i+8,5),label,fill='black')
    canvas.save(path)
def main():
    per={};rows=[]
    for e in EXPERIMENTS:
        r=ROOT/'render_comparison'/e;report=json.loads((r/'render_complete.json').read_text());metrics=list(csv.DictReader((r/'per_view_metrics.csv').open()));assert len(metrics)==208
        per[e]={int(v['source_id']):v for v in metrics};summary=dict(report['summary']);summary.pop('representative_source_ids',None);summary['representative_views']=len(report['summary']['representative_source_ids'])
        if e in NEW:
            train=json.loads((ROOT/NEW[e]/'densification_events.complete.json').read_text())
            with np.load(ROOT/NEW[e]/'checkpoints/final_gaussians/gaussians.npz') as z:
                summary.update(training_seconds=float(z['training_seconds']),peak_training_vram_gib=float(z['peak_vram_mib'])/1024,training_complete_iteration=30000)
            summary.update(full_training_command_wall_seconds=train['wall_seconds'],peak_full_training_command_allocated_gib=train['peak_allocated_gib'],peak_full_training_command_reserved_gib=train['peak_reserved_gib'])
            events=[json.loads(line) for line in (ROOT/NEW[e]/'densification_events.jsonl').read_text().splitlines()]
            summary.update(zero_Q_events=sum(v['Q']==0 for v in events),last_densification_gaussians=events[-1]['after'])
        else:
            old=ROOT.parent/'E1_GGGS_Baseline/checkpoints' if e=='E1' else OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison/E5_Standard_VGGT_GGGS_Baseline/checkpoints'
            with np.load(old/'final_gaussians/gaussians.npz') as z:summary.update(training_seconds=float(z['training_seconds']),peak_training_vram_gib=float(z['peak_vram_mib'])/1024,training_complete_iteration=30000)
        rows.append(summary)
    for sid in ids:
        assert len({per[e][sid]['gt_sha256'] for e in EXPERIMENTS})==1
        assert len({per[e][sid]['mask_sha256'] for e in EXPERIMENTS})==1
    csvwrite(ROOT/'reconstruction_comparison.csv',rows)
    paired=[]
    for a,b in [('E1','E7'),('E5','E9'),('E7','E8'),('E9','E10'),('E1','E8'),('E5','E10'),('E10','E8')]:
        for metric in ['full_psnr','fg_mask_psnr','inner_fg_edge5_psnr','ssim','lpips_alex','fg_gradient_retention','fg_gradient_mae','fg_gradient_ratio_abs_error']:
            diff=np.array([float(per[b][sid][metric])-float(per[a][sid][metric]) for sid in ids])
            paired.append(dict(reference=a,condition=b,metric=metric,mean_delta=float(diff.mean()),median_delta=float(np.median(diff)),p10_delta=float(np.quantile(diff,.1)),p90_delta=float(np.quantile(diff,.9)),fraction_improved=float((diff<0 if metric in ['lpips_alex','fg_gradient_mae','fg_gradient_ratio_abs_error'] else diff>0).mean())))
    csvwrite(ROOT/'render_comparison/paired_effects.csv',paired)
    cmap=plt.get_cmap('turbo');crop_records=[];leaf_metrics=[]
    # E6's exact eight source IDs, selected from the shared usable order.
    for sid in e6_ids:
        inputs={}
        for e in EXPERIMENTS:
            with np.load(ROOT/'render_comparison'/e/'views'/f'crop_{sid:04d}'/'raw_render_outputs.npz') as z:inputs[e]={k:z[k] for k in ['render_rgb','gt_rgb','foreground_mask','alpha','depth_median','normal_camera']}
        gt=inputs['E1']['gt_rgb'];fg=inputs['E1']['foreground_mask'];h,w=fg.shape
        panels=[gt]+[inputs[e]['render_rgb'] for e in EXPERIMENTS];labels=['GT']+EXPERIMENTS
        montage(panels,labels,OUT/f'source_{sid:04d}_whole_plant.png',resize=(270,480))
        # Fixed native-pixel leaf window inherited from the audit's original ROI.
        box=[270,285,540,615];montage([p[box[1]:box[3],box[0]:box[2]] for p in panels],labels,OUT/f'source_{sid:04d}_leaf_native.png')
        gray=gt@np.array([.2126,.7152,.0722]);grad=np.hypot(ndimage.sobel(gray,1),ndimage.sobel(gray,0));inner=fg&~ndimage.binary_erosion(fg,iterations=5);yy=np.nonzero(fg)[0];candidate=inner.copy();candidate[int(yy.min()+.6*(yy.max()-yy.min())):]=False
        cy,cx=np.unravel_index(np.argmax(ndimage.uniform_filter(grad,size=15)*candidate),fg.shape);x0=max(0,min(cx-160,w-320));y0=max(0,min(cy-160,h-320));crop_records.append(dict(source_id=sid,x0=int(x0),y0=int(y0),side=320,selection='Same GT-only edge crop for all six native renders; no interpolation'))
        montage([p[y0:y0+320,x0:x0+320] for p in panels],labels,OUT/f'source_{sid:04d}_leaf_edge_native320.png')
        # Read-only scores of the same displayed native crops. These are eight
        # representative GT-only regions, not a new leaf-instance segmentation.
        gt_gradient=gradients(gt)
        for region,bounds in [('fixed_leaf_window',box),('GT_selected_upper_edge320',[x0,y0,x0+320,y0+320])]:
            selected=np.zeros_like(fg);selected[bounds[1]:bounds[3],bounds[0]:bounds[2]]=True
            pixels=fg&selected;edge_pixels=inner&selected
            for e in EXPERIMENTS:
                rgb=inputs[e]['render_rgb'];error=(rgb-gt)**2;gradient=gradients(rgb)
                leaf_metrics.append(dict(experiment=e,source_id=sid,region=region,x0=int(bounds[0]),y0=int(bounds[1]),x1=int(bounds[2]),y1=int(bounds[3]),foreground_pixels=int(pixels.sum()),inner_edge_pixels=int(edge_pixels.sum()),
                    crop_fg_psnr=psnr_region(error,pixels),crop_inner_edge5_psnr=psnr_region(error,edge_pixels),
                    crop_fg_gradient_retention=float(gradient[pixels].mean()/gt_gradient[pixels].mean()) if pixels.any() else None,
                    crop_fg_gradient_mae=float(np.abs(gradient-gt_gradient)[pixels].mean()) if pixels.any() else None))
        errors=[];normals=[];alphas=[];depths=[]
        # Depth is displayed relative to each camera trajectory's native extent.
        # This is a display normalization only; no points/cameras are transformed.
        for e in EXPERIMENTS:
            v=inputs[e];abs_error=np.abs(v['render_rgb']-gt).mean(2)*fg;heat=np.zeros((*fg.shape,3));heat[...,0]=np.clip(abs_error/.5,0,1);heat[...,1]=np.clip(abs_error/.5,0,1)*.3;errors.append(heat)
            length=np.linalg.norm(v['normal_camera'],axis=2,keepdims=True);normal=np.divide(v['normal_camera'],length,out=np.zeros_like(v['normal_camera']),where=length>1e-8);normal=np.clip(normal*.5+.5,0,1);normal[v['alpha']<=.01]=0;normals.append(normal)
            alphas.append(np.repeat(np.clip(v['alpha'],0,1)[...,None],3,2))
            records=json.loads((ROOT/'render_comparison'/e/'camera_runtime.json').read_text())['records'];centers=np.array([r['camera_center'] for r in records]);extent=np.max(np.linalg.norm(centers-centers.mean(0),axis=1))*1.1
            z=v['depth_median']/extent;depth=cmap(np.clip(z/5,0,1))[...,:3];depth[v['alpha']<=.01]=0;depths.append(depth)
        for typ,images in [('foreground_error',errors),('normal',normals),('alpha',alphas),('depth_per_extent',depths)]:montage(images,EXPERIMENTS,OUT/f'source_{sid:04d}_{typ}.png',resize=(270,480))
    (OUT/'visualization_protocol.json').write_text(json.dumps(dict(source_ids=e6_ids,old_e6_selection='usable indices0,26,52,78,104,130,156,182',overview_interpolation='nearest, display only',leaf_edge_native_pixels=True,crops=crop_records,depth_display='native median depth / own camera extent, common display0..5; arrays retained; no scan alignment'),indent=2))
    csvwrite(ROOT/'render_comparison/representative_leaf_crop_metrics.csv',leaf_metrics)
    # Training counts from the actual observed events, including 7K--15K.
    fig,ax=plt.subplots(figsize=(8,4))
    for e in NEW:
        ev=[json.loads(x) for x in (ROOT/NEW[e]/'densification_events.jsonl').read_text().splitlines()];ax.plot([r['iteration'] for r in ev],[r['after'] for r in ev],label=e)
    ax.set_xlabel('Iteration');ax.set_ylabel('Gaussian count after densification');ax.legend();fig.tight_layout();fig.savefig(ROOT/'geometry_comparison/gaussian_counts.png',dpi=170)
    # Same display rotation in each frontend's first-camera reference frame.
    # Plot normalized coordinates only; no point-cloud alignment is applied.
    from plyfile import PlyData
    fig=plt.figure(figsize=(12,5))
    for panel,dirname in enumerate(['VGGT_Omega_Original_RGB','Standard_VGGT_Original_RGB']):
        out=ROOT/dirname;ply=PlyData.read(out/'foreground.ply',mmap='r')['vertex'].data
        index=np.linspace(0,len(ply)-1,min(80000,len(ply))).astype(int);xyz=np.column_stack([ply[k][index] for k in ['x','y','z']]);rgb=np.column_stack([ply[k][index] for k in ['red','green','blue']])/255
        camera=json.loads((out/'cameras.json').read_text())['frames'][0];e=np.array(camera['extrinsic_model_w2c']);local=xyz@e[:3,:3].T+e[:3,3];center=np.median(local,0);radius=np.quantile(np.linalg.norm(local-center,axis=1),.98);local=(local-center)/radius
        ax=fig.add_subplot(1,2,panel+1,projection='3d');ax.scatter(local[:,0],local[:,1],local[:,2],c=rgb,s=.3,linewidths=0);ax.view_init(elev=15,azim=-60);ax.set_xlim(-1,1);ax.set_ylim(1,-1);ax.set_zlim(-1,1);ax.set_box_aspect([1,1,1]);ax.set_title(dirname)
    fig.tight_layout();fig.savefig(ROOT/'geometry_comparison/foreground_pointclouds.png',dpi=200)
    geo=list(csv.DictReader((ROOT/'frontend_comparison.csv').open()));lookup={r['experiment']:r for r in rows}
    def r(e,key):return lookup[e][key]
    def delta(a,b,key):return r(b,key)-r(a,key)
    report=['# ForeSplat 原始 RGB 双前端 208 帧与完整 GGGS 对照','',
        '两套208帧前端、E8/E10/E7/E9各30,000步及E1/E5六模型统一208训练视角浮点评价均已实际执行。定量表和原生像素图已生成；下面技术结论仍需结合直接查看渲染与点云补充视觉判断。', '',
        '几何 RGB 使用审计适配器：Mask 外为02-FFT原图、Mask 内为逐像素不变的03-SAM前景；同时保存完整原图路径。训练 RGB/Mask、权重、优化损失和增密参数保持原配方。标准模型保留square1024→518，Ω保留balanced@512。','',
        '所有正式实验均在物理GPU1，经uv运行。共同208帧为0–209排除历史损坏44/206，没有增减。各前端使用深度反投影Point Map，保留原始Camera/Depth/Confidence；未新增learned point head或融合算法。Ω沿用既有融合，标准沿用原分层点采样，因此E8/E10并非同融合算法消融。','',
        '|实验|PSNR|真FG PSNR|前景内侧边缘PSNR|SSIM|LPIPS|FG梯度保留|Gaussian数|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in EXPERIMENTS:report.append(f'|{e}|{r(e,"full_psnr"):.4f}|{r(e,"fg_mask_psnr"):.4f}|{r(e,"inner_fg_edge5_psnr"):.4f}|{r(e,"ssim"):.4f}|{r(e,"lpips_alex"):.4f}|{r(e,"fg_gradient_retention"):.4f}|{int(r(e,"gaussian_count")):,}|')
    report+=['','同一评价实现的收益拆分：',f'- 几何RGB收益（原接口）：E7−E1 Full/FG/Edge = {delta("E1","E7","full_psnr"):+.4f}/{delta("E1","E7","fg_mask_psnr"):+.4f}/{delta("E1","E7","inner_fg_edge5_psnr"):+.4f} dB；E9−E5 = {delta("E5","E9","full_psnr"):+.4f}/{delta("E5","E9","fg_mask_psnr"):+.4f}/{delta("E5","E9","inner_fg_edge5_psnr"):+.4f} dB。',
        f'- 独立接口收益：E8−E7 Full/FG/Edge = {delta("E7","E8","full_psnr"):+.4f}/{delta("E7","E8","fg_mask_psnr"):+.4f}/{delta("E7","E8","inner_fg_edge5_psnr"):+.4f} dB；E10−E9 = {delta("E9","E10","full_psnr"):+.4f}/{delta("E9","E10","fg_mask_psnr"):+.4f}/{delta("E9","E10","inner_fg_edge5_psnr"):+.4f} dB。','',
        '全部208逐视角配对差及改善比例见render_comparison/paired_effects.csv。不把高度相关的训练视角当作独立重复实验，不宣称统计显著性或未见新视角泛化。原17.89/13.11旧汇总值仅作历史背景，本报告正式收益使用统一重算E1/E5。','',
        '前端完整指标见frontend_comparison.csv，独立RGB匹配、前景/背景、双向深度、边界/叶缘支持与相机轨迹分别留存。相机与3D表面没有公制真值；闭合间隙为轨迹诊断，不能自动当作真值闭环误差。','',
        '像素接口不仅修复COLMAP内参：真实Camera保留cx/cy，训练resize包含半像素项，P包含非中心主点，CUDA前向/反向渲染及sample_depth射线读取同一相机矩阵。合成投影、薄表面采样与反向梯度、实际两前端416个runtime Camera均已检查，数值见camera_interface/。原/修正版共享相同点云文件和训练RGB。','',
        '所有8个旧E6代表视角的GT|E1|E5|E7|E8|E9|E10整株、原生叶片/叶缘、误差、Alpha、Normal与Depth显示在render_comparison/visualizations/；前端全量未过滤PLY、前景PLY及逐帧Point Map位于两前端目录。没有通过插值改善锐度。','',
        '视觉核验与九项最终技术决策：等待直接查看真实图像和点云后补充。','',
        '来源：本目录真实模型/训练/渲染日志、Checkpoint、逐视角CSV；[VGGT官方源码](https://github.com/facebookresearch/vggt)、[VGGT-Ω官方源码](https://github.com/facebookresearch/vggt-omega)、[GGGS固定源码](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)。博查搜索返回403配额不足；Context7与官方静态源码已核验，实验数值并非训练知识推断。']
    (ROOT/'FINAL_ANALYSIS.md').write_text('\n'.join(report)+'\n')
    print('common208 summaries and E6 eight-view native evidence ready',flush=True)
if __name__=='__main__':main()
