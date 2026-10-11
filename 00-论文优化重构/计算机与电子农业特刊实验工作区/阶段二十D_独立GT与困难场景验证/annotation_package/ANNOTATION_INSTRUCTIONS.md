# TEST32 人工标注与独立复核说明

状态：`USER_ANNOTATION_REQUIRED`。标注者与复核者必须是真实人员；AI 不能
替代标注人员，也不能签署或冒充第二位复核者。

## 1. 找到原图并确认目标

- `rgb/<Sample>_<Frame>.jpg` 是未经缩放、裁剪、重编码的原始 RGB 的本地链接。
  用 `TEST32_INPUT_MANIFEST.csv` 的 `rgb_path` 和 `rgb_sha256` 恢复/验证原图。
- 32 帧、8 个序列、每序列 4 帧；原始坐标为宽 2160、高 3840。
- `context/<Sample>_<Frame>_prev.jpg` 与 `_next.jpg` 仅帮助识别同一株盆栽，
  不进入独立评价分母，不是附加测试样本。不能借用邻帧 GT。
- 写出目标花盆/植物的位置、外观和跨帧身份依据到 `target_identity_description`。
  不要因居中、面积大、绿色多或模型选中就认定目标；相邻植物必须排除。
- 建议标注及首次复核阶段隐藏候选/R0/R1 预测，避免锚定偏差。
- 不确定目标身份时保留该帧，标记待澄清/拒绝；**不得移除冻结 manifest 的帧**。

## 2. 必需多边形

每张保存 `annotations/<Sample>_<Frame>.json`，标准 JSON 包含 `imageWidth`、
`imageHeight`、`imagePath` 和 `shapes`。

| label | 要求 |
| --- | --- |
| `potted_plant` | 必需：仅目标盆栽的可见整体，包括植物组织和目标盆 |
| `plant` | 必需：同一目标植物的可见组织，排除盆、邻株、背景 |
| `pot` | 必需：目标盆；不能把邻盆算入，也不能从预测推断 |
| `blue_cube` | 真实存在时单独标注；不存在时不造多边形，明确填 NO |

- `shape_type` 固定为 `polygon`；每个多边形至少三个有限且不共线的不同点，
  坐标须满足 `0 <= x < 2160`、`0 <= y < 3840`。不允许缩放后标注再插值。
- 多个可见分离区域可用同一 label 的多个 polygon，但必须属于同一个目标。
  本关卡不支持用另外一个实例的 group_id 来混入邻株，也不支持洞轮廓语义。
  有此问题需由人修正/明确，不能从 SAM 补齐。
- `imagePath` 文件名为 `<Sample>_<Frame>.jpg`；也接受原始 `<Frame>.jpg`，
  文件名本身不构成身份认证，还必须绑定原图 SHA256。
- 建议不嵌入 `imageData`；如果嵌入，必须是同一原始 RGB 字节，不能重新压缩。
- `plant`、`pot` 必须位于 `potted_plant` 内；GT 主口径始终为
  **`potted_clean = potted_plant AND NOT blue_cube`**。
- SAM3、R1、其他自动分割、多边形预测或 AI 输出不得直接充当/生成 GT。
  不提供模型初始化的自动标注模板，不自动填充任何 polygon。

旧 Phase16 instruction card 的 plant/pot“建议”、`linestrip` 以及
“身份不明就移除 manifest”不适用于本阶段。本说明与 Phase20D 用户规范优先；
历史文件保留不变。

## 3. 逐帧人工困难证据

在两个根目录 CSV 中都填写真实判断：`TEST32_ANNOTATION_REVIEW.csv` 和
`DIFFICULTY_HUMAN_VERIFICATION.csv`。困难标签用分号连接，如 `D1;D11`。
允许多标签；全部不符合时明确写 `none`，不要留空。

| code | 类别 | 应写的人类视觉证据（不是自动判定标准） |
| --- | --- | --- |
| D1 | Neighbor-plant adhesion | 哪株邻植株与目标何处接触/交叠，何处应归目标/排除 |
| D3 | Foreground-background confusion | 哪一段植物/盆与哪种背景在外观或边界上难分 |
| D6 | Small target | 目标在整幅图中的相对尺寸、位置，及易漏的可见细节 |
| D11 | Cluttered background | 真实干扰物位置与性质：邻盆、设施、线缆等，不能泛称“复杂” |
| D12 | Low contrast | 目标的哪些边界与相邻背景颜色/亮度接近，以及具体位置 |

`difficulty_evidence_json` 必须为标签到文字证据的 JSON 对象，例如
`{"D1":"人工填写位置和交叠关系","D11":"人工填写实际干扰物位置"}`。
这些是**填写格式示例，不是任何帧的视觉结论**。`none` 也需写人工检查依据。
每个 D 列显式 YES/NO；不从绿色比例/历史 heuristic D1/D11 自动复制标签。
当前 5 类全部没有独立人工确认，图件状态为 `NOT COVERED`。

## 4. 两位真实审核者与哈希绑定

1. 记录真实 `annotator`，人工逐点完成多边形，`annotation_origin=MANUAL_POLYGONS_ONLY`。
2. 第一位 `reviewer` 检查原图身份、目标、三个必需 label、cube、有无邻株、原始坐标。
   第一位审核者可以是标注者，但第二位必须与标注者及第一位审核者不同。
3. 第二位 `second_reviewer` 独立看原图与多边形并核查困难证据；不要仅签名。
4. 实际通过后填写人工身份声明和各确认字段 YES、cube YES/NO，以及两位审核时间
   （带时区的 ISO 时间推荐），才填 `approval=APPROVED`。
5. 用 `sha256sum "annotations/<Sample>_<Frame>.json"` 得到真实最终哈希，填入
   两个 CSV 的 `annotation_sha256`。两表的审核人、标签、证据及哈希必须相同。
6. 任何 polygon/身份/类别修改均使旧批准失效；新版本必须重新独立复核并更新哈希。

CSV 中不能伪造姓名、人类声明、时间、YES 或 APPROVED。脚本只能验证记录和
几何一致性，不能证明填写者真的看过图；签署真实性由实际人员负责。

## 5. 只运行预检，不是准确率评价

从仓库根目录：

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_独立GT与困难场景验证'
/home/test/biosoft/enter/bin/python "$STAGE/scripts/validate_annotations.py" \
  --output "$STAGE/logs/annotation_gate_after_human_review.json"
```

未通过时退出码为 2，输出 `USER_ANNOTATION_REQUIRED`；该命令从不计算准确率。
输出文件已经存在时换用新名称，不覆盖记录。
只有 32/32 合格且原始输入仍一致时才可继续 Phase20D 的冻结 R0/R1、Oracle、
全部逐帧/逐序列统计与另外的预测身份人工核查。不要自行启动 Phase20E。

来源：Phase20D 用户规范；冻结 TEST32 manifest；现有 Phase20C-FIX polygon/GT gate。
