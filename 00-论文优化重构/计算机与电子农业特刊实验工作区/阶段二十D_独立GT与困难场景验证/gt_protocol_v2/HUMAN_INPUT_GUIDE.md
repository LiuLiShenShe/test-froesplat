# 只补录已有人工工作的确认，不重画32张

已接受用户说明：有些视角完全看不到花盆；只标 potted_plant 的视角可能仅有植物
可见。不要画不存在的 pot，也不要从邻帧或模型补齐。因为未提供具体帧映射，
GT_VISIBILITY_MANIFEST.csv 的 pot_visibility 全部初始 UNVERIFIED，绝不根据缺标签猜。

## 最少三张表 + 可选一次性批量记录

1. **GT_VISIBILITY_MANIFEST.csv（32行）**：填写每帧 pot_visibility、plant_visible、
   blue_cube_present，真实确认人/人类声明/时间/记录来源和 visibility_status=CONFIRMED。
   无 plant 标签且人类明确确认仅植物可见的11帧，还需 plant-only/等值派生确认、
   真实来源与明确规则。其他帧不会自动派生或要求不可见 pot。
2. **LINESTRIP_CONVERSION_AUDIT.csv（75行）**：点击 REVIEW_INDEX.html 查看总览及
   逐形状原图叠加；shape_index 从0开始。只有确实完整闭合外轮廓时，填
   complete_region_boundary=YES、obvious_open_gap=NO、closed_region_confirmed=YES，
   confirmed_by/confirmed_is_human/confirmed_at/confirmed_source/confirmation_reason，
   confirmation_status=CONFIRMED_CLOSED_REGION。不确定就保留 HUMAN_REVIEW_REQUIRED。
   每帧多个shape可在人工实际逐个检查后一次填共享代码/时间，不得自动批量假审批。
   另有17个shape/13帧的数值自交或回折异常，精确边索引及21张局部放大图见
   INTERSECTION_DETAIL_INDEX.csv 和 REMAINING_HUMAN_CONFIRMATIONS.md。只核查这些
   局部问题，不要求整套重标；脚本不会自动改点，几何不合格不能仅靠审批通过。
3. **HUMAN_REVIEW_V2.csv（32行）**：保留预填160个D决策及标签，填写现成目标身份、
   真实 difficulty_evidence_json（每个标签或none都对应图像特定依据）、人类确认、
   可见组件确认和来源声明。真实复核完成且最终导出哈希绑定后才能 approval=APPROVED。
   原人类记录可转录，不从日期/AI描述猜。现有D值不按序列重新复制。

可把真实共享人员代码和实际完成时间只录入 **HUMAN_BATCH_RECORD.json**，并在逐帧表
填同一 batch_record_id。该记录必须确认 actual_human_review_completed=YES、真实来源、
完整范围和 scope_source_identities_sha256。转换后填各最终 derived SHA256 到
scope_final_derived_sha256_by_frame，同时逐帧 annotation_sha256 绑定同一哈希。
它只减少录入重复，不代替每帧判断，不自动把逐帧 approval 变为 APPROVED。

## 具体可见性路径

- pot=NOT_VISIBLE + plant=YES 且有原plant：保留真实plant，不造pot。
- pot=NOT_VISIBLE + plant=YES、只标potted_plant：仅在真人明确plant-only/等值派生
  同意后，复制现有人工轮廓，记录固定规则；不重新描绘整套GT。
- pot=VISIBLE/PARTIALLY_VISIBLE：保留或取得已做的真实pot/plant轮廓，不能把整体
  potted_plant复制给plant；若确实有未标的可见部分，仅该部分由真实人处理。
- 所有未确认的可见性/开放折线/身份/证据：阻止转换或V2Gate，不降低审批条件。

## 隐私和保留

用实际人员可追溯、经用户同意公开的代号；身份映射放 private/ 或库外。正式记录
若含姓名先放私有 records-dir，不要提交公开CSV。原JSON及completion字节备份不改，
新版本输出到独立目录。图像仅显示真实RGB/原路径，不是AI生成GT或真人签名。

## 来源

这里的确认字段由实际标注/复核者提供，AI不填APPROVED/YES/时间/视觉证据。
用户当前提供的是协议层说明，不足以补签具体32帧/75形状。可直接整理已经做过的
真实记录，无需重做不存在的花盆或整套标注。
