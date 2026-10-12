# 已有人工工作归档 — 2026-10-12

用户报告已完成人工标注和真人复核。本阶段不否定其已做的工作，不要求重做 32 张。
实际找到 32 份原始 JSON，位于 `annotation_package/rgb/`（682234 字节合计），
不是原 Gate 指定的 `annotations/`。现已建立 32 个本地只读链接；逐文件原字节
SHA256 前后相同，见 `GT32_SOURCE_ARCHIVE.csv`。JSON、RGB、候选均未重写。

## 已可靠归档

- 两表恰好覆盖冻结 32 frame_id；与 challenge_id/RGB 哈希逐行匹配。
- 从现有 JSON 填入当前 `annotation_sha256`，追加图像尺寸和客观几何状态。
  此哈希仅绑定当前导出，不表示审批通过或轮廓已最终冻结。
- 保留 `22073ca5` 的全部 160 个 D 列 YES/NO 单元格；仅从同一帧五列派生标签。
  D1=12、D3=16、D6=15、D11=16、D12=12；16 帧为 `none`。类型允许重叠。
  这只是已有人工判定的转录，不是已经获得文字证据和双人审批的困难分类。
- 不填姓名、审核者代号、人工确认、来源声明、时间、目标描述、视觉证据、审批。
  现有 PENDING 状态不变；JSON 的 `checked=false` 不被解释为人类审核结论。

## 具体导出兼容性问题

32/32 文件的全部 75 个 shape 都是 `linestrip`，原冻结 Gate 只接受 `polygon`。
没有将开放折线填充成多边形、闭合端点、修改 shape_type 或补画缺失区域。
现有点无非有限、越界或退化坐标错误；这不意味着多边形 GT 已有效。
没有合法 raster GT，故 plant/pot 包含性与 potted_clean 非空检查均未通过，
不是误报“空掩膜”。全部文件均未含 blue_cube；实际是否不存在仍需真人确认。

15 帧另缺必需 label（11 帧缺 plant，15 帧缺 pot）：

| frame_id | missing labels |
| --- | --- |
| DouBanLv3_0211 | plant;pot |
| DouBanLv3_0220 | plant;pot |
| DouBanLv3_0221 | plant;pot |
| HongZhang_0134 | plant;pot |
| HongZhang_0200 | plant;pot |
| HongZhang_0202 | plant;pot |
| HongZhang_0229 | plant;pot |
| WanNianQing1_0223 | plant;pot |
| WanNianQing1_0224 | plant;pot |
| WanNianQing1_0225 | plant;pot |
| WanNianQing1_0227 | plant;pot |
| XiangPiShu2_0133 | pot |
| XiangPiShu2_0134 | pot |
| XiangPiShu2_0135 | pot |
| XiangPiShu2_0136 | pot |

先寻找/提供已有的最终兼容导出及组件标注，不要求整集重新描绘。若只有当前导出，
需实际标注者确认折线的预期闭合语义及缺失组件；未经授权和真人确认，AI 不转换。
任何真实人员补充/格式修正都要保留旧版本、记录新 SHA256，并绑定最终审批。

## 只补录真实审核记录

`HUMAN_REVIEW_BATCH_REGISTRATION.csv`：实际标注者、首次审核者、独立第二审核者的
代号、实际完成时间、覆盖范围、身份映射的私有保存位置及公开代号确认。
首次审核者可为标注者，第二审核者必须与两者不同。不能用文件 mtime、Git 日期、
当前日期或用户总体完成声明推测审核日期/人员。

`HUMAN_REVIEW_FRAME_SUPPLEMENT.csv`：预填逐帧原始哈希、已有 D 判断、派生标签、
具体几何问题和缺失字段清单；人工区域为空。填真实目标身份、cube 判定、确认、
各真实困难标签对应的图像特定证据；`none` 也需要真人检查依据。不以通用文本
或重复序列标签代替逐帧记录。仅把实际记录合并回两份正式 CSV，不自动批量批准。

两个表中当前每帧都缺：annotation_origin、annotator、reviewer、second_reviewer、
三个人类声明、manual_annotation_confirmed、target_identity_description、
target_identity_confirmed、plant_confirmed、pot_confirmed、blue_cube_present、
difficulty_manually_confirmed、difficulty_evidence_json、reviewed_at、second_reviewed_at；
approval 仍 PENDING，difficulty status 仍 PENDING_HUMAN_VISUAL_REVIEW。

真实姓名与代号对应表保存在 completion/private/ 或库外私有位置，不能上传。
用户未提供匿名代号的公开确认，故未自动创建身份代号。当前公开材料不含身份。

## Gate

原 `scripts/validate_annotations.py` 未修改；32 JSON 存在，0/32 通过，退出码 2，
evaluation_allowed=false。权威日志为 `logs/annotation_gate_completion.json`；
`GT_GATE_RESULT.json` 是内容相同的交付副本。停止所有依赖 GT 的后续运行。

来源：本地真实 LabelMe JSON、冻结 manifest、22073ca5 人工五列和未修改的原 Gate。
