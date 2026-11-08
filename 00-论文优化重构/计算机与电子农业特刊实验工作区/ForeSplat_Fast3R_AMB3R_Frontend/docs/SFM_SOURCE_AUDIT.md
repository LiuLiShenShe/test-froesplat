# AMB3R-SfM 与 Fast3R 适配源码审计

审计日期：2026-11-07。官方源码固定版本：AMB3R `92c4081f910f98e683503092b85301861519175e`；Fast3R `33104d4b5b8df43795ecded236194958bbdac572`。未修改 `third_party` 网络、注册或融合源码。

## 依据

- [AMB3R 论文正文与补充材料](https://arxiv.org/html/2511.20343)：直接下载 HTML 并阅读第 4.5 节、补充材料第 7 节。第 4.5 节定义 whitening、FPS 聚类、coarse registration、BFS keyframe mapping 和 non-keyframe refinement；补充材料明确复杂薄结构会破坏参考帧几何先验、导致尺度注册不稳定。
- [固定版本 SfM pipeline.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/pipeline.py)：本地逐段阅读，并直接下载相同版本官方 raw 文件复核，共 869 行。
- [SfM run.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/run.py)、[memory.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/memory.py)、[clustering.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/sfm/clustering.py)：阅读真实张量接口、注册状态和内建融合。
- [AMB3R model.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/model.py)、[frontend.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/frontend.py)、[pts_align.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/tools/pts_align.py)：核对原始预测、尺度头和坐标变换。
- [SLAM pipeline.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/slam/pipeline.py)、[SLAM run.py](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/slam/run.py)：核对顺序 VO 和无序 SfM 的区别；本阶段没有以 VO 代替 SfM。
- [Fast3R 论文](https://arxiv.org/abs/2501.13928)：直接阅读已下载 PDF 的 global fusion、local/global point maps、置信度与 PnP 段落。
- [Fast3R models/fast3r.py](https://github.com/facebookresearch/fast3r/blob/33104d4b5b8df43795ecded236194958bbdac572/fast3r/models/fast3r.py)、[multiview_dust3r_module.py](https://github.com/facebookresearch/fast3r/blob/33104d4b5b8df43795ecded236194958bbdac572/fast3r/models/multiview_dust3r_module.py)、[init_im_poses.py](https://github.com/facebookresearch/fast3r/blob/33104d4b5b8df43795ecded236194958bbdac572/fast3r/dust3r/cloud_opt/init_im_poses.py)：直接核对 encoder、官方多视角调用和 RANSAC PnP。
- 通过 `ctx7 library` 再 `ctx7 docs /pytorch/pytorch/v2.5.1` 获取 [PyTorch autocast](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/amp.rst)、[inference_mode](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/autograd.rst)、[CUDA_VISIBLE_DEVICES](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/cuda_environment_variables.rst) 文档。没有升级全局 PyTorch。

博查搜索在主任务返回 HTTP 403 / 配额不足。本审计直接访问用户指定的一手论文与官方仓库，不使用内置 WebSearch/WebFetch。web-access skill 已加载；CDP 前置检查发现未配置浏览器，因此没有使用登录态浏览器，也没有修改用户浏览器配置。

## 实际接口与 README 的区别

`AMB3R_SfM.run()` 创建 `SfMemory(T,H,W)`，随后依次执行：

1. `extract_features()` 调用模型 `extract_amb3r_sfm_features(views)`，需要 `(T,D)` 全局描述向量。
2. 官方 `get_distance_matrix(..., whitening=True)`，再 `find_best_image_clustering`，默认每簇 5–8 张。
3. `process_clusters()` 预测每簇，用置信度与与其他簇的视觉相似性选择初始化种子；低置信帧拆为单帧簇。
4. `initialize_map()` 更换第一视图多次推理，并由官方 `SfMemory.initialize` 进行尺度对齐、置信融合和 keyframe selection。
5. `coarse_registration()` 寻找候选簇，连接已有全局关键帧，进行 `coordinate_alignment`，延期和重试低置信帧。
6. `global_mapping()` 调用 keyframe BFS refinement，再 non-keyframe refinement。完整执行默认一次全局 refinement。

README 示意的 `world_points_conf` 为 `(B,T,H,W,1)`，但真实 pipeline 在 `process_one_cluster`、`build_prediction_dict`、`register_candidate_cluster` 中反复 `.mean(dim=(1,2))`，`SfMemory.conf` 分配为 `(T,H,W)`。真实 AMB3R `DPTHead` 也输出 `(B,T,H,W)`，注释中的尾单例维与实际不一致。

`Fast3RForAMB3RSfM` 公开接口保留用户要求的 `(B,T,H,W,1)`；独立 `PipelineBridge` 仅 `squeeze(-1)`，供未修改的官方 pipeline 使用。不改变置信度值。

## Fast3R 描述向量与局部几何

预训练 Fast3R 配置使用 CroCo encoder，patch 16，embedding 1024。`extract_amb3r_sfm_features` 直接调用原始 `model.encoder(image,true_shape)`，对原始 encoder patch tokens 求均值，返回 `(T,1024)`。仅对 encoder 独立图片分块控制峰值内存，所有图片的描述向量仍进入同一官方聚类。没有新增视觉模型，没有预先抽帧。

每次 `run_amb3r_sfm` 保持该调用内全部视图的 Fast3R Global Fusion，通过官方 `inference(...,profiling=False)` 获取字典中的 `preds`。官方 `profiling=False` 不返回二元组。网络及预训练权重不修改。

每个局部调用的 `world_points` 都位于该调用第一个视图的参考坐标系。不同调用不共享世界坐标。跨簇变换只由官方 `coordinate_alignment`、关键帧锚点和 `SfMemory` 完成。

跨簇注册是随memory更新反复进行的动态对齐与融合，不是一个静态簇只对应一张最终Sim3矩阵。本次未额外截获每次 `coordinate_alignment` 的内部Sim3矩阵；保存的是完整原始local maps/poses、accepted source calls、官方memory阶段状态和最终maps/poses。实际验证覆盖C2W逆变换下的真实前景像素投影、深度正号、旋转合法性和初始化参考相机与identity距离，并有已知非零位姿的逆方向对照；这些验证不被称为每call完整Sim3矩阵trace或GT注册精度。

## Confidence

Fast3R `reg_dense_conf` 在配置 `('exp',1,inf)` 下为 `1+exp(logit)`；AMB3R VGGT `head_act.py` 的 `expp1` 也为 `1+exp(logit)`。二者具有相同值域，可以将原始 Fast3R confidence 直接传给原始 SfM 置信处理；值域相同不表示概率校准或误差条件分布相同，需实测最终分布与注册行为。

`SfMemory.conf` 已经是 `conf_sig=(conf-1)/conf` 的融合结果，值域为 `[0,1)`。不能将其当作原始 confidence 使用 `conf>1`。导出保存原始 memory sig，并用精确代数式 `1/(1-sig)` 恢复统一接口的 confidence 表达；称为“从 memory sig 恢复的 confidence”，不称为网络原始预测。无穷、负值、`sig>=1` 均标为无效，不进行制造高置信度的裁剪。每次网络调用的原始 confidence 另存 `local_predictions/call_*.npz`。

原始 AMB3R `run_amb3r_sfm` 的 `benchmark_conf0` 比较发生在 Frontend 完成、Backend 开始之前；所有帧 raw confidence 均未超出现有基准减 0.01 时返回 `None`，避免一次 Backend。单阶段 Fast3R 接口返回完整预测，不虚构同样的 Frontend/Backend 结构。官方 SfM 的 candidate confidence 最佳帧替换、置信阈值、keyframe/non-keyframe 选择和 remapping 全部保留。原生提前返回率与 Fast3R 局部调用数不是同一种模型内部开销。局部原始存档包含官方接口实际返回的预测；官方提前返回 `None` 的调用只记录状态与时间，不伪造未返回的几何。

## Pose 与内参

Fast3R `fast_pnp` 调用 `cv2.solvePnPRansac` 得到 world-to-camera 的 R/T，最终返回其逆矩阵，即 C2W。PnP 固定为 100 次，第一视图 global head 估计焦距，principal point `(W/2,H/2)`。源码的 `min_conf_thr_percentile=85` 目前未应用于 PnP 有效点筛选，实际有效点仅 `conf>1`，不可把注释称为实际 top15% 策略。

官方 PnP 失败时返回 identity pose，单凭 SE(3) shape 或有限值无法识别失败。适配器复用独立 observer 记录真实 solver 成功标志/有效点/inliers，保留官方求解算法，同时保存投影数值验证。

AMB3R `frontend.py` 由 `pose_encoding_to_extri_intri` 输出 W2C，再 `closed_form_inverse_se3` 得到 `pose`。`coordinate_alignment` 的参数明确命名为 `c2w_local/c2w_global`，源码变换是 `T_global_local @ C2W_local`，与 Fast3R 相同。

官方 SfM 对多次局部结果进行尺度对齐并融合 point maps 与 poses，但没有全局内参优化，也不保存最终 K。本适配记录每次 local call 的 K，在实际内建 memory fusion 被调用时追踪 accepted contribution，最终 bundle 的 K 来自最新参与该帧 memory 的局部预测。初始化为多预测融合。该 K 不被宣称为 SfM 优化后的内参；最终重投影误差明确包含融合几何、融合相机与该局部 K 的不一致。全部原始 K 和参与 call id 可追溯。

## 尺度、失败帧与禁止冒充

AMB3R `frontend.py` 仅将预测的 metric scale 乘到 `depth_metric`。`world_points`、`pts3d_by_unprojection`、`pose` 保留归一化尺度；原生 SfM 用 `world_points` 与 `pose`，跨簇又会进行尺度注册。完整 SfM 地图不能直接标记为公制。每次原生调用的 `depth_metric` 和 scale-head 原始输出保留用于单独研究测量潜力。Fast3R 与 hybrid 始终标记为尺度不确定。未使用扫描点云或任何尺度矫正。

原生局部 SfM 的具体几何字段是 `frontend.point_head` 直接输出的 `world_points` / `world_points_conf`。`frontend.py` 另外由 normalized `depth`、W2C、K 计算 `pts3d_by_unprojection`，但 `model.run_amb3r_sfm` 不把这个替代点图覆盖到 `world_points`。`pipeline.process_one_cluster`、`build_prediction_dict` 及 `SfMemory.update_kf` 明确读取 `world_points`。因此 Base 全208上下文中直接 point head 与 camera head 不一致的结果不能直接推断小簇推理加原生注册后仍相同；原生用的仍是 direct head，但局部视图上下文、adaptive frontend/backend fallback 和官方 memory 对齐/融合均不同。所有局部原字典中的 unprojection 结果另存用于诊断，没有替换主实验几何。

默认官方配置 `allow_unmapped_frames=false` 会把最终低置信失败帧强行写入 memory。runner 选择同一官方配置项 `true`，其余阈值不改，以保留真实 unmapped 状态。可显式 `--force-low-confidence-mapping` 复现原默认。此差异记录在完整运行参数中。除 official unmapped set 外，再检查 memory iteration 与 pose 有效性，保留全部输入序号和失败原因。

原始 memory 中未映射帧是零 point map 与零 pose。原样保存在 `official_memory_raw.npz`；统一 bundle 将未映射帧 canonical points、pose、confidence 置为 NaN，避免把原点假点与 identity 假相机写进有效几何。不会静默删除帧，也不会生成虚假的特征 tracks。

## 实现与运行

- `frontends/sfm/fast3r_adapter.py`：Fast3R 双接口及 confidence shape bridge。
- `frontends/sfm/run.py`：官方 native/hybrid headless runner、只观察的阶段与 memory hooks、每次局部原始输出、全部帧注册状态、最终投影验证及统一 bundle。
- Native 输入 518 长边、14 对齐；hybrid 输入 512 长边、16 对齐；均调用统一 RGB/Mask 等比 resize+padding，未裁切，保留全部可用 208 帧及 210 行原始 manifest。
- 所有 GPU 入口在导入 torch 前设置 `CUDA_VISIBLE_DEVICES=1`，逻辑 `cuda:0` 只表示物理 GPU1。环境安装由独立环境任务管理。

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-amb3r/bin/python frontends/sfm/run.py --model amb3r
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv-fast3r/bin/python frontends/sfm/run.py --model fast3r
```

这份审计说明源码与实现。是否成功注册、是否改善植物薄结构，必须以真实数据运行后的 `status.json`、`registration_status.json`、`geometry_validation.json` 与评价结果判断，不能从接口实现推断。

## 已完成的 CPU 数值检查

在 GPU 调度前进行了三个独立检查，均在入口设置 `CUDA_VISIBLE_DEVICES=1`，未启动 CUDA 模型推理：

1. `.venv-amb3r` 中实际导入官方 pipeline 和 `SfMemory` 成功。
2. 已知非零旋转与平移的合成 C2W，原始像素反投影到世界，再通过导出检查重投影，中位残差 `8.88e-16 px`；将 W2C 误作 C2W 得到 `11.92 px`。此检查能区分位姿方向，不能证明真实数据位姿质量。
3. 相同两视图预测分别进入未经观察的官方 `SfMemory` 和加只观察 hooks 的 memory，实际执行 `initialize` 与 `update`；`pts`、`poses`、`conf`、`iter` 逐项完全相等，accepted call provenance 正确记录。Confidence shape bridge 的数值与几何逐项不变。

真实 Fast3R adapter 的 encoder、PnP 和跨簇注册仍需 GPU1 真实帧试验。本节不将合成检查称为真实冒烟结果。

## Whitening 精度与隔离环境补齐

真实原生首轮保留模型返回的 descriptor dtype。`--descriptor-fp32` 是独立的显式重试选项：只将已经计算好的 pooled descriptors 精确扩展至 FP32，改变后续协方差的算术精度，不重算 encoder、不改 descriptor 值、不修改官方 whitening/聚类代码。`--retry-of` 保存原失败目录引用；重试必须使用新目录，首轮 raw descriptors、distance matrix、错误和日志始终保留。官方 whitening 返回非有限距离时，observer 明确停止并记录，避免将 NaN 当作有效聚类。

CPU 随机合成的低秩 `208×1024` descriptors 复现了风险：BF16 协方差出现大量 `<-1e-6` 特征值，官方 `sqrt(eig+1e-6)` 无效；FP32 在某些特征幅度下也可能出现舍入负特征值。本诊断不是实际植物 descriptor 的结论，必须以真实首轮日志判断，不自动修改 epsilon 或替换官方算法。

## 首株真实全量运行的观测

原生32-view smoke 完整执行全部官方阶段，32/32注册、12关键帧、0 unmapped；仅是先行验证，不代表全序列。随后原生208-view在同一固定官方源码与权重上完整执行，208/208注册、73关键帧、0 unmapped。实际 refined map seed 为 usable189 / original source190，初始化 seed C2W 距 identity 0.000423，最终官方融合后为0.3413；地图参考信息保存到 map_reference.json。

原生全量有1403 local calls：initial31、init8、coarse1205、KF73、non-KF86。352次官方前端比较后提前返回None；32次返回预测使用官方 Backend confidence 下降后的 Frontend fallback。因此该实验为原生自适应 SfM，不能把它说成每次局部预测都是强制 Backend ON。最终前景逐帧重投影中位数的中位数26.037px，旋转det约1；此值含最终融合点图/pose与局部K的差异，并非GT精度。

原生 pipeline wall2644.641s，其中 local model1164.768s，局部原始预测转CPU与写盘1022.578s；allocated峰值11.617GiB、reserved峰值36.475GiB。/data 为共享HDD，现场 /proc/pressure/io some avg60约70%、full约66%。原生输出直接写HDD，hybrid输出先在/tmp NVMe SSD暂存后完整复制和验证；模型时间、原始序列化、bundle导出和持久化复制分别记录。不同I/O位置下的pipeline总时间不能直接当作模型速度排名。

两个真实全208流程的原始descriptor均为FP32，未启用descriptor扩精重试，whitening未产生NaN。实际off-diagonal距离：native mean0.50241536/std0.00179305；Fast encoder mean0.50241548/std0.00014031，后者min0.49966943/max0.50319004。源码在N208<D1024时中心化、全协方差白化再L2 cosine；距离近似常数与白化满行秩点接近simplex的数学推断相符，均值接近N/(2(N−1))。这提示首株的小样本视觉聚类区分性限制，不能据此宣布注册失败；没有改白化、epsilon、特征维度或聚类核心。

Hybrid全208也已完成所有官方计算和SSD完整bundle：原始29簇、19低置信帧拆分为47 coarse簇；第二轮重试15帧、第三轮重试2帧后，最终208/208注册、35关键帧、0unmapped。全部1310 predicted calls具有9885条实际PnP success，缺失观察0；这一solver状态不是几何精度保证。Seed usable/source23，初始/最终参考C2W距离identity分别0.003041/0.234978；最终前景投影帧中位数的中位数27.781px。

Hybrid pipeline2073.706s、local model1260.933s、local raw SSD99.184s；allocated/reserved峰值4.606/5.002GiB。最终CPU bundle导出276.074s，从模型加载（输入已经准备）到bundle结束实测2465.544s，排除原始输入读取与最终HDD复制。完整5145原始文件、52,300,419,491 bytes已逐文件集合/尺寸、全帧camera/RGB/Mask/archive及first/reference/last真实数组等值核验，HDD复制+核验实测3250.702s。随后16份派生指标/图按清单逐SHA同步，最终5162文件，canonical已指向成功persistent attempt。该磁盘时域不能与原生HDD运行混同为模型加速。

已由主任务明确授权使用 uv `--no-deps` 为 `.venv-fast3r` 安装 `evo==1.31.1` 与 `utils3d` 官方固定 commit `c5daf6f6c244d251f252102d09e9b7bcef791a38`，并补齐其缺失依赖，版本与 AMB3R 环境相同。`uv pip check` 的 163 个包全部兼容，真实官方 SfM CPU import 成功。Fast3R 环境 PyTorch 安装前后均为 `2.10.0+cu128`，没有修改全局或其他环境。完整安装记录见 `FAST3R_SFM_ENVIRONMENT.json`。uv 接口通过 Context7 官方文档核对：[环境选择](https://github.com/astral-sh/uv/blob/main/docs/pip/environments.md)、[依赖检查](https://github.com/astral-sh/uv/blob/main/docs/pip/inspection.md)。
