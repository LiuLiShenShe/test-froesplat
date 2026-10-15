"""Index real external artifacts; an index is not a remote data backup."""
from common import *
import shlex,time

def main():
    entries=[];start=time.perf_counter()
    for path in sorted(DATA.rglob('*')):
        if not path.is_file():continue
        entries.append({'asset_scope':'E3_NEW_EXTERNAL','path':str(path),'relative_to_managed_root':str(path.relative_to(DATA)),
                        'bytes':path.stat().st_size,'sha256':sha(path)})
        if len(entries)%100==0:print('HASHED_EXTERNAL_FILES',len(entries),flush=True)
    for sample in SCENES:
        base=Path(reference(sample)['checkpoint_path']).parent
        for suffix in ['chkpnt30000.pth','point_cloud/iteration_30000/point_cloud.ply','input.ply','cameras.json','cfg_args']:
            path=base/suffix
            if path.exists():entries.append({'asset_scope':'INHERITED_M0_REFERENCE','path':str(path),'relative_to_managed_root':'','bytes':path.stat().st_size,'sha256':sha(path)})
    write_csv(STAGE/'LARGE_ASSET_INDEX.csv',entries)
    env={'CUDA_VISIBLE_DEVICES':GPU_INDEX,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
    commands=[]
    for sample in SCENES:
        commands.append({'purpose':'Reference mesh','sample':sample,'method':'M0','command':[str(PYTHON),str(STAGE/'scripts/extract_mesh.py'),'--sample',sample,'--method','M0'],'env':env,'cwd':str(ROOT)})
        for method in ['M1','M2']:
            commands.append({'purpose':'Fresh30000 training','sample':sample,'method':method,'command':[str(PYTHON),str(STAGE/'scripts/train_observed.py'),'--sample',sample,'--method',method],'env':env,'cwd':str(ROOT)})
    write_json(STAGE/'reports/RECOVERY_COMMANDS.json',commands)
    write_json(STAGE/'reports/LARGE_ASSET_INDEX_VERIFICATION.json',{'status':'PASS','files':len(entries),'bytes':sum(x['bytes'] for x in entries),
               'index_sha256':sha(STAGE/'LARGE_ASSET_INDEX.csv'),'wall_seconds':time.perf_counter()-start,'time':now(),
               'remote_backup_created':False,'recovery_limit':'Requires the indexed external files; Git contains scripts/manifests and small outputs, not checkpoint or mesh data'})
    text=f'''# Recovery and reproduction — Phase20E-3

生产代码基准：`{BASELINE}`。本阶段提交只添加阶段目录，不改 R1、2DGS、M2M3、COLMAP、Mask、人工 GT 或论文正文。交付提交号见 Git 历史与最终回复；配置、脚本、报告均可从该提交恢复。

## 已存在的真实恢复路径

- 仓库：`{ROOT}`。
- 原始冻结数据和 Seed0 模型：`/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1/`；模型位于 `phase20e1_outputs/<scene>/<group>/`。
- E2 的原始共同评价浮点文件：`/data/fj/F2DMAS_phase20e2_data/694743e7_v1/`。
- 本阶段独立受管理目录：`{DATA}`。没有覆盖历史资产。
- 新模型：`training/<scene>/M1|M2/attempt_01/chkpnt30000.pth` 和 `point_cloud/iteration_30000/point_cloud.ply`。
- 主 Mesh：`meshes/<scene>/M0|M1|M2/main/raw.ply`、`post.ply`；各训练相机的真实 float RGB/alpha/depth 位于同目录 `fusion_float/`，清单位于 `FUSION_FLOAT_MANIFEST.json`。
- 新 held-out float：`evaluation/<scene>/M1|M2/`；M0 使用 E2 的原始浮点文件，指标 CSV 保留实际路径与 SHA256。
- Mesh 固定视角 CPU 重投影：`mesh_reprojection/<scene>/<method>/`。保留预先规则选定的三个视角 float depth/normal；所有视角均可由保留的 Mesh 与冻结相机重算。
- 预声明敏感性结果：`meshes/<scene>/M0/sensitivity/`（以实际索引路径为准）。保留成功与失败记录，不替换主实验。
- 初始完整 Git index 的较大文本位于 `provenance/BASELINE_GIT_INDEX.txt`；Git 内的 `reports/BASELINE_GIT_INDEX_POINTER.json` 提供 SHA256。

`LARGE_ASSET_INDEX.csv` 为本次真实逐文件 SHA256/字节数索引，共 {len(entries)} 个文件、{sum(x['bytes'] for x in entries):,} 字节。冻结输入和生产源码另由 `configs/FREEZE.json` 的 5868 条受保护 SHA256 记录约束。索引是恢复校验依据，不能代替数据备份。本阶段没有创建远程大文件备份，也没有把大型 checkpoint/PLY/Mesh 上传 GitHub。

## 完整命令与环境

`reports/RECOVERY_COMMANDS.json` 保存 argv、cwd、环境变量；实际执行顺序、开始结束与退出码见 `reports/PREFLIGHT_EXECUTION.json`、`reports/TRAINING_EXECUTION.json`、`reports/POST_EXECUTION.json`，首个 GPU Mesh 另见 `reports/WanNianQing1_M0_LAUNCH.json`。所有 stdout/stderr 均保留在 `logs/`。正式短测试独立保存在 `smoke/`，不是任何正式模型的恢复起点。

指定物理 GPU：index={GPU_INDEX}，UUID=`{GPU_UUID}`。每个 GPU 作业开始前检查其空闲；没有切换 GPU。详细 Python/PyTorch/CUDA/Open3D/扩展版本与扩展 SHA256 见 `reports/ENVIRONMENT.json`。系统资源的只读观测见 `logs/RESOURCE_OBSERVATIONS.csv`；提取分项时间和进程退出等待分开，新增观察包装有额外开销。精确计时边界、PyTorch allocated/reserved 与设备采样的区别见 `RESOURCE_METRIC_SPEC.md`。

所有正式配置均在 `configs/*_TRAIN.json` 和 `configs/*_MESH.json`。M1/M2 初始化、RNG、alpha 和相机抽样对照分别在 `reports/*_INITIALIZATION.json`、`reports/*_CAMERA_SEQUENCE.json`。原生 Seed0 历史初始 RNG 未存档，M0 可复用依据是源代码/配置等价，不宣称拥有历史 RNG 状态哈希。

## 校验与重算

在恢复上述绝对路径和依赖后，以下命令只重算校验和报告，不重新训练或覆盖模型：

```bash
cd {shlex.quote(str(ROOT))}
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 {shlex.quote(str(PYTHON))} {shlex.quote(str(STAGE/'scripts/acceptance.py'))}
CUDA_VISIBLE_DEVICES={GPU_INDEX} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 {shlex.quote(str(PYTHON))} {shlex.quote(str(STAGE/'scripts/recompute_metrics.py'))}
```

`acceptance.py` 独立重载所有正式 checkpoint/PLY 与18个主 Mesh、重算结构并核对受保护输入；`recompute_metrics.py` 从真实 float 和共同 raw RGB/R0 ROI 复算204行指标，需指定 GPU 空闲。两者会更新本阶段校验报告。

训练/提取入口故意拒绝覆盖既有正式输出目录。完整重现实验应使用同一绝对路径布局的空白隔离环境，或另建命名空间并重新冻结仅输出路径配置；不能在已有目录直接重复运行、删除失败证据或覆盖历史模型。不要把“已有资产可重载”和“另行重训得到逐位相同结果”混为一谈；CUDA 执行并非由本阶段证明逐位确定。

无三维几何 GT 或物理尺度。所有 Mesh 坐标、深度和面积保留任意 COLMAP 单位/平方单位；本阶段结束后不进入 Phase20F、不修改论文正文。
'''
    (STAGE/'RECOVERY_AND_REPRODUCTION.md').write_text(text)
    print('RECOVERY_INDEX_COMPLETE',len(entries),flush=True)

if __name__=='__main__':main()
