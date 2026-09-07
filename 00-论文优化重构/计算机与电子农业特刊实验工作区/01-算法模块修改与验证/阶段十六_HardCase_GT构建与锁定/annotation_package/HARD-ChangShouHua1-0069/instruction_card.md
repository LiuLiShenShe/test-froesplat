# Instruction Card — HARD-ChangShouHua1-0069

## 标注对象
- **sample / 序列**: ChangShouHua1
- **目标帧**: 0069.jpg
- **split**: DEV（DEV / TEST — 仅用于分组，不影响标注内容）

## 难度提示（启发式，标注时请人工确认）
- primary:   背景杂乱 (D11_cluttered_background)
- secondary: 前景/背景混淆
- raw-image difficulty score: 0.6430
- Phase-15 排名: 17

## 目标身份规则（关键）
1. 本 sample 是时间序列，目标是**同一株个体植物**（跨帧一致）。
2. 先查看 context_prev.jpg / context_next.jpg 确认目标个体身份。
3. 只标注目标个体；相邻植株、背景植被、温室结构一律排除。

## 需要标注的 label
| label | 必需? | 说明 |
|---|---|---|
| `potted_plant` | **必须** | 目标整株（含盆），闭合 linestrip |
| `plant` | 建议 | 仅叶+茎（不含盆） |
| `pot` | 建议 | 花盆/托盘 |
| `blue_cube` | 如存在 | 蓝色标定块（potted_clean 会自动扣除） |

## 原始帧路径（标注完成后，把 .jpg + .json 放同目录）
- 原始帧: `/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames/ChangShouHua1/0069.jpg`
- 上下文前帧（可用）: `/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames/ChangShouHua1/0064.jpg`
- 上下文后帧（可用）: `/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames/ChangShouHua1/0074.jpg`
- **GT 输出目录**: `/data/fj/F2DMAS/03-GT-区分_challenge/ChangShouHua1/`

## 尺寸
图像 3840×2160（高×宽，纵向；JSON 中 imageHeight=3840, imageWidth=2160），
shape_type=linestrip，`imagePath` 字段只填帧名如 `0069.jpg`。

## 若无法确定身份
在 JSON 的 text 字段写明 "ambiguous"，并把该帧从 manifest 移除（通知负责人）。
