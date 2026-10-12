# 最少人工补录清单（不是重新绘制32张GT）

已接受可见区域协议：缺pot标签本身不是错误，不应画不存在的花盆。

1. 32帧：将已有目标身份/实际pot可见性/plant可见性/blue_cube存在性填入V2表，记录实际确认人和时间。
2. 75个linestrip：借助REVIEW_INDEX.html确认是否完整区域边界；共享人员/时间可按真实已完成范围录入。
3. 保留现有160个D类YES/NO；补已有逐图视觉依据、独立真人复核、最终派生JSON哈希。
4. 11帧没有plant标签，仅在真人确认pot不可见且整体仅植物时补等值派生授权；不要求补不可见pot。

额外数值几何异常：17个shape/13帧，21张自交局部图。边序号/坐标可定位；未自动修复。

自交不能靠签字把坐标变为有效polygon。请只核查以下局部异常；原JSON永久保留。
若需要人为改变坐标，必须另立人工修订版本及来源记录；当前转换器不会修改顶点，也不会自动接受修订版。

| frame | shape（0-based）| label | 交叉边 |
| --- | --- | --- | --- |
| CaoMei2_0007 | 0 | potted_plant | [[0, 238]] |
| CaoMei2_0007 | 1 | plant | [[0, 164]] |
| CaoMei2_0008 | 0 | potted_plant | [[75, 77]] |
| CaoMei2_0008 | 2 | pot | [[0, 76]] |
| CaoMei2_0009 | 0 | potted_plant | [[0, 152], [152, 153]] |
| CaoMei2_0009 | 1 | plant | [[0, 118], [118, 119]] |
| CaoMei2_0146 | 0 | potted_plant | [[0, 167], [2, 167]] |
| CaoMei2_0146 | 1 | plant | [[94, 96]] |
| ChangShouHua3_0189 | 0 | potted_plant | [[0, 103]] |
| DouBanLv3_0220 | 0 | potted_plant | [[0, 115]] |
| HongZhang_0229 | 0 | potted_plant | [[0, 84]] |
| WanNianQing1_0227 | 0 | potted_plant | [[0, 143]] |
| WangWenCao2_0014 | 1 | plant | [[0, 98]] |
| WangWenCao2_0015 | 0 | potted_plant | [[43, 49], [44, 48]] |
| XianKeLai3_0117 | 1 | plant | [[163, 165]] |
| XiangPiShu2_0133 | 0 | potted_plant | [[29, 31]] |
| XiangPiShu2_0135 | 1 | plant | [[44, 46]] |

精确逐帧缺字段见HUMAN_REVIEW_REQUIRED_ITEMS.csv；填写说明见HUMAN_INPUT_GUIDE.md。
源SHA256预填不代表人类审批。无确定帧映射，不会把全部缺pot的帧推断为NOT_VISIBLE。
