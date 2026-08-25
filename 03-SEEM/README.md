# SEEM 图像分割处理报告

## 处理信息

| 项目 | 内容 |
|------|------|
| 处理时间 | 2026-02-28 |
| 输入目录 | /data/fj/02-FFT |
| 输出目录 | /data/fj/03-SEEM |
| 文本提示词 | "a potted plant and a small blue square block" |
| 模型 | SEEM Focal-T v0 |

## 处理结果

| 项目 | 数量 |
|------|------|
| 总图片数 | 4217 |
| 成功处理 | 4217 |
| 失败 | 0 |
| 成功率 | 100% |

## 输出效果

- 保留原图颜色的分割区域
- 背景为纯黑色
- 平均黑色背景占比: ~76%
- 保持原有文件夹结构

## 文件夹结构

```
/data/fj/03-SEEM/
├── BaiZhang/
├── ChangShouHua3/
├── DouBanLv1/
├── DouBanLv2/
├── WanNianQing2/
├── WangWenCao2/
├── XianKeLai2/
├── XianKeLai3/
├── XiangPiShu2/
└── ... (共11个子文件夹)
```

## 使用方法

```bash
cd /data/fj/Segment-Everything-Everywhere-All-At-Once
source venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

# 文本提示分割
python batch_segment.py -i /data/fj/02-FFT -o /data/fj/03-SEEM -t "your text prompt"

# 全景分割
python batch_segment.py -i /data/fj/02-FFT -o /data/fj/03-SEEM --panoptic
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `-i, --input` | 输入目录 |
| `-o, --output` | 输出目录 |
| `-t, --text` | 文本提示（英文） |
| `--panoptic` | 全景分割模式 |
| `-e, --ext` | 输出扩展名（默认: .jpg） |

## 注意事项

1. 文本提示词需使用英文
2. 输出图片保留原图尺寸
3. 分割区域保留原图颜色，