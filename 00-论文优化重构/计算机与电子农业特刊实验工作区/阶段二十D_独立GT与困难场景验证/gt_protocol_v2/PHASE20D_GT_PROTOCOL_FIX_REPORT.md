# Phase20D GT Protocol FIX — visible-region V2

Date:2026-10-12；基准HEAD：`6092e2ea`，原人工资料提交 `4d8dcd52`。
本阶段结果来自真实冻结文件、人工历史分类和实际CPU协议检查，不是分割准确率。

## 结论

```text
GT_PROTOCOL_FIX_IMPLEMENTATION=COMPLETE
GT_PROTOCOL=FORESPLAT_VISIBLE_GT_V2
HUMAN_RECORD_ARCHIVE=INCOMPLETE
GT32_V2_GATE=BLOCKED
READY_FOR_PHASE20D_EVALUATION=NO
INDEPENDENT_TEST_METRICS=NOT_RUN
PHASE20E=NOT_STARTED
```

已修正错误的协议假设：不可见花盆不要求标注；只标整体的植物视角可以经真人
明确确认作等值组件格式转换。**并不认定15个无pot标签的帧漏标，也不要求重画
32张GT。** 用户说明是协议层事实；未提供具体帧对应、确认人/时间，不据此给所有
缺标签帧批量填写 NOT_VISIBLE、APPROVED 或plant派生授权。

## 已完成的真实工作

| 项目 | 实际结果 |
| --- | --- |
| 冻结TEST32输入 | 32帧，8个序列，71个原P6候选身份/掩膜/metadata通过原输入校验 |
| 原人工JSON及字节备份 | 32/32 SHA256与6092归档一致；原RGB/候选哈希不变 |
| 图像坐标 | 全部width2160、height3840；源坐标保留 |
| 原始shape | 75个linestrip；32 potted_plant、21 plant、22 pot、0 blue_cube |
| 缺组件标签客观事实 | 15帧无pot标签；11帧同时无plant标签；不推断实际可见性 |
| 逐shape客观审计 | 顶点9–240；首尾距离0.259259–31.352620像素；无自动距离审批阈值 |
| 真实RGB显示材料 | 107张整帧/逐shape核查叠加图，源/显示SHA和尺度有索引 |
| 自交数值异常 | 17个shape，涉及13帧；21对交叉/相邻回折重叠，21张局部RGB放大图 |
| 其余几何 | 当前未见非法/越界顶点、未知标签或零面积；58个shape仅通过数值几何，不是闭合审批 |
| 历史D类判定 | 原160个YES/NO不变，逐帧推导labels/none；未制造逐图证据 |
| 实际转换尝试 | 32行manifest，0个派生JSON；缺确认时未写“自动GT” |
| 实际V2 Gate | 0/32 VERIFIED，evaluation_allowed=false；退出码2 |
| 验证 | 140项测试通过（V2新增57）；编译检查通过，git diff --check通过 |
| 保护范围 | 原V1 Gate、所有旧completion/指标、生产算法、2DGS/M2M3-GO、论文正文未改 |

自交详情见 `GEOMETRY_EXCEPTION_AUDIT.csv` 和 `INTERSECTION_DETAIL_INDEX.csv`。其中
部分发生在首尾附近，另有局部轮廓内部交叉；不能以“首尾很近”自动判断闭合。
新增相邻边回折检测发现 CaoMei2_0009 两个shape各多一对重叠；只更新这两个shape
的**新V2客观诊断列**，未修改坐标或任何人类字段。首次局部图重建因旧客观诊断
不一致拒绝执行，保留失败日志；最新 `intersection_details_v2/` 为完整有效显示材料。

## 仍需的最小人工输入

1. 转录实际已有的32帧目标身份、pot_visibility、plant可见性、blue_cube存在性，
   真实人员代号/时间/来源。不能由缺标签或照片AI描述替代。
2. 转录75个linestrip的完整闭合区域边界确认；开放或不确定的轮廓保留阻断。
3. 11个缺plant帧只在明确确认pot不可见且整体等于植物时补派生授权；若实际pot
   可见则不得复制整体为plant。无确认时没有派生文件。
4. 复用原人工D类YES/NO，补真实图像特定证据、独立第二真人复核及最终JSON哈希。
   可批量录入实际共享人员/完成时间，不能批量制造判断。
5. 针对17个几何异常只做局部人工核查。坐标自交不会因签字而消失；如确需改点，
   必须另立人工修订版本/来源流程，永久保留原JSON。当前严格转换器不修点。

精简32行补录表为 `HUMAN_REVIEW_REQUIRED_ITEMS.csv`；具体局部清单为
`REMAINING_HUMAN_CONFIRMATIONS.md`。不存在“缺pot必补pot”的补录项。
身份代号公开必须由用户确认；真实身份映射保持私有，本次没有人员信息代填。

## 交付及停止边界

所有新增内容仅在 `gt_protocol_v2/`。协议、源身份、可见性表、逐shape审计、真实
转换manifest、独立真人录入模板、Gate JSON/报告、可重复运行脚本/测试、真实图件
源索引、差异及独立Methods建议均已保存。大图/私有/转换目录忽略Git，恢复步骤
见 `REPRODUCE.md`。Methods仅为独立建议文件，未改原论文正文。

没有冻结算法重评分、候选重新推理、参数调整、TEST32准确率、方法LOCKED结论
或Phase20E训练。本轮未要求提交/推送，未创建提交、分支或更改远端。

## 证据入口

- `logs/INPUT_AND_PROTECTION_CHECKS.json`：原图/原JSON/备份/候选和保护检查。
- `logs/GT_GATE_V2_RESULT.json`：实际32帧阻断原因；不是旧Gate报告替换。
- `logs/CONVERSION_RESULT.json`：实际0转换，不含伪造derived_hash。
- `logs/TEST_ALL_FINAL_2.txt`：140项最终真实测试输出。
- `logs/REPLAY_FINAL_*`：完整新目录重建的实际日志，仍为0转换/阻断Gate。
- `SOURCE_IDENTITIES.csv`、两个图件索引：原始文件与派生展示可追溯SHA256。
- `PROTOCOL_DIFF_AND_METHODS_SUGGESTION.md`：V1→V2差异和条件性Methods表述。

外部搜索API不可用（HTTP403）；本报告依据本地冻结资料与实际脚本，不依赖训练
记忆或搜索断言。图像内容排版描述由Agnes API提供，仅作显示QA，不计入人类记录。
