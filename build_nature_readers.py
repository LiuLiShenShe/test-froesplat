#!/usr/bin/env python3
"""Build bilingual Markdown readers for local research-paper PDFs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pdfplumber


DEFAULT_PDFS = [
    Path("/data/fj/F2DMAS/00参考文章/1-s2.0-S2643651526000373-main-2.pdf"),
    Path("/data/fj/F2DMAS/00参考文章/fpls-17-1783465.pdf"),
    Path("/data/fj/F2DMAS/00参考文章/plantphenomics.0235.pdf"),
]

SECTION_RE = re.compile(
    r"^(?:(?:[1-9]|1[0-9])(?:\.\d+){0,4}\.?\s+[A-Z][A-Za-z0-9 ,&()/:;–—-]+|"
    r"(?:Abstract|Introduction|Materials and methods|Methods|Results|Discussion|Conclusion|Conclusions|"
    r"Data availability statement|Data availability|Author contributions|Funding|Conflict of interest|"
    r"Declaration of competing interest|Acknowledgments|Acknowledgements|Supplementary material|References))$",
    re.I,
)

CAPTION_START_RE = re.compile(r"^(Fig\.|Figure|FIGURE|Table|TABLE)\s*\d+", re.I)

TERM_FIXES = {
    "植物现象型": "植物表型",
    "现象型": "表型",
    "现象类型": "表型",
    "现象数据": "表型数据",
    "智能繁殖": "智能育种",
    "繁殖": "育种",
    "标记": "标注",
    "注释": "标注",
    "无人监督": "无监督",
    "不受监督": "无监督",
    "型号化": "表型分析",
    "型方法": "表型方法",
    "型提取": "表型提取",
    "型平台": "表型平台",
    "体素体积": "体素体积",
    "语音体积": "体素体积",
    "语音量": "体素",
    "波素": "体素",
    "音量": "体素体积",
    "面具": "掩膜",
    "分区": "分割",
    "细分": "分割",
    "互动": "交互式",
    "自封": "自遮挡",
    "高斯式分光": "3D Gaussian Splatting",
    "3D高质谱": "3D Gaussian Splatting",
    "神经辐射领域": "神经辐射场",
    "辐射领域": "辐射场",
    "点点": "点云",
    "籽籽粒": "籽粒",
    "谷物": "籽粒",
    "大米": "水稻",
    "恐慌": "穗部",
    "fenootyping": "phenotyping",
    "可差异化": "可微",
}

MANUAL_TRANSLATIONS = {
    "Abstract": "摘要",
    "1. Introduction": "1. 引言",
    "Introduction": "引言",
    "2. Materials and methods": "2. 材料与方法",
    "Materials and methods": "材料与方法",
    "Methods": "方法",
    "3. Results": "3. 结果",
    "Results": "结果",
    "4. Discussion": "4. 讨论",
    "Discussion": "讨论",
    "5. Conclusion": "5. 结论",
    "Conclusion": "结论",
    "Conclusions": "结论",
    "References": "参考文献",
    "Author contributions": "作者贡献",
    "Funding": "基金资助",
    "Data availability": "数据可用性",
    "Data availability statement": "数据可用性声明",
    "Conflict of interest": "利益冲突",
    "Declaration of competing interest": "利益冲突声明",
    "Supplementary material": "补充材料",
}

MANUAL_FIGURE_CAPTIONS = {
    "1-s2.0-S2643651526000373-main-2.pdf": [
        {
            "label": "Fig. 1",
            "original": "Fig. 1. Overview of the Plant3R model's pipeline.",
            "translation": "图 1. Plant3R 模型流程概览。",
        },
        {
            "label": "Fig. 2",
            "original": "Fig. 2. Comparison between SfM and Plant3R at different growth stages of wheat. In order to intuitively demonstrate the feature extraction and sparse point cloud reconstruction capabilities of the Plant3R model, we selected its reconstructed sparse point cloud for visual comparison with the SfM algorithm.",
            "translation": "图 2. 小麦不同生育阶段下 SfM 与 Plant3R 的比较。为直观展示 Plant3R 模型的特征提取与稀疏点云重建能力，作者选取其重建的稀疏点云与 SfM 算法进行可视化对比。",
        },
        {
            "label": "Fig. 3",
            "original": "Fig. 3. 2D image rendering results of wheat's four growth stages under three different algorithms.",
            "translation": "图 3. 三种不同算法在小麦四个生育阶段的二维图像渲染结果。",
        },
        {
            "label": "Fig. 4",
            "original": "Fig. 4. 3D point cloud extraction results of wheat's four growth stages under four different algorithms. From the point cloud visualization results of different models, the fidelity of the Plant3R model is significantly better than Colmap, NeRF and original 3DGS.",
            "translation": "图 4. 四种不同算法在小麦四个生育阶段的三维点云提取结果。从不同模型的点云可视化结果看，Plant3R 模型的保真度显著优于 Colmap、NeRF 和原始 3DGS。",
        },
        {
            "label": "Fig. 5",
            "original": "Fig. 5. Comparison between NeRF, 3DGS and Plant3R in model surface details such as leaf, stem and ear.",
            "translation": "图 5. NeRF、3DGS 与 Plant3R 在叶片、茎秆和穗部等模型表面细节上的比较。",
        },
        {
            "label": "Fig. 6",
            "original": "Fig. 6. Comparison of model-derived and manually measured values of wheat plant height at different growth stages.",
            "translation": "图 6. 小麦不同生育阶段模型推导株高与人工测量株高的比较。",
        },
        {
            "label": "Fig. 7",
            "original": "Fig. 7. Validation of model-derived wheat leaf dimensions against manual measurements across different growth stages. Scatter plots showing the correlation between model-extracted and manually measured values for (a-c) leaf length and (d-f) leaf width across three growth stages (Tillering, Jointing, and Grain Filling). The high R2 values (>0.94) and low RMSE indicate a strong agreement between the model estimations and ground truth.",
            "translation": "图 7. 不同生育阶段模型推导的小麦叶片尺寸与人工测量结果的验证。散点图展示三个生育阶段（分蘖期、拔节期和灌浆期）中，模型提取值与人工测量值在 (a-c) 叶长和 (d-f) 叶宽上的相关性。较高的 R2 值（>0.94）和较低的 RMSE 表明模型估计与真实值高度一致。",
        },
    ],
    "fpls-17-1783465.pdf": [
        {
            "label": "Fig. 1",
            "original": "FIGURE 1. Overview of the end-to-end phenotyping pipeline based on 3D Gaussian Splatting (3DGS).",
            "translation": "图 1. 基于三维高斯泼溅（3DGS）的端到端表型分析流程概览。",
        },
        {
            "label": "Fig. 2",
            "original": "FIGURE 2. The flowchart illustrates the complete data flow from seed preparation on the left through the three-stage core algorithm on the right (lifting, clustering, refinement), ultimately exporting a 3D plant model.",
            "translation": "图 2. 流程图展示了完整的数据流：从左侧的种子点准备，到右侧三阶段核心算法（提升、聚类、精修），最终导出三维植株模型。",
        },
        {
            "label": "Fig. 3",
            "original": "FIGURE 3. Comparison of reconstruction results from (a) dense MVS, (b) NeRF, and (c) 3DGS. Insets show zoomed regions highlighting differences in geometric and photometric fidelity.",
            "translation": "图 3. (a) 稠密 MVS、(b) NeRF 和 (c) 3DGS 的重建结果比较。插图显示局部放大区域，用于突出几何保真度和光度保真度的差异。",
        },
        {
            "label": "Fig. 4",
            "original": "FIGURE 4. Effect of input cue count (rows) and lift-score retention percentile (columns) on plant-background separation. Higher cue counts and stricter percentiles better isolate high-confidence plant regions.",
            "translation": "图 4. 输入提示数量（行）和提升分数保留百分位（列）对植株-背景分离的影响。更多提示和更严格的百分位阈值能更好地隔离高置信植株区域。",
        },
        {
            "label": "Fig. 5",
            "original": "FIGURE 5. Precision-recall curves showing the incremental effect of each LCR-GS component. Each curve adds one processing stage, illustrating improvements from lifting, geometric clustering, NN-retain, and final CIELAB refinement.",
            "translation": "图 5. 精确率-召回率曲线展示各 LCR-GS 组件的递增效果。每条曲线增加一个处理阶段，说明提升、几何聚类、NN 保留以及最终 CIELAB 精修带来的改进。",
        },
        {
            "label": "Fig. 6",
            "original": "FIGURE 6. Ablation results on two plants. Columns show the incremental effects of each LCR-GS stage, with insets highlighting improvements in leaf completeness and background removal.",
            "translation": "图 6. 两株植株上的消融结果。各列展示 LCR-GS 各阶段的递增效果，插图突出叶片完整性和背景去除方面的改进。",
        },
        {
            "label": "Fig. 7",
            "original": "FIGURE 7. Comparison of (a) original RGB frames, (b) the reconstructed 3DGS scene, and (c) extracted plant instances obtained using LCR-GS.",
            "translation": "图 7. (a) 原始 RGB 帧、(b) 重建的 3DGS 场景以及 (c) 使用 LCR-GS 提取的植株实例的比较。",
        },
        {
            "label": "Fig. 8",
            "original": "FIGURE 8. Comparison of organ-level segmentation results across seven representative test plants (a-g). For each plant, the top row shows ground-truth annotations, the middle row shows PTv3 predictions in point-cloud space, and the bottom row shows reconstructed organ instances in the 3DGS representation.",
            "translation": "图 8. 七株代表性测试植株 (a-g) 的器官级分割结果比较。每株植株中，上排为真实标注，中排为点云空间中的 PTv3 预测结果，下排为 3DGS 表示中的重建器官实例。",
        },
        {
            "label": "Fig. 9",
            "original": "FIGURE 9. Validation of computed phenotypic traits against manual measurements. Left: correlation between computed and measured plant height. Right: correlation between computed and measured leaf count.",
            "translation": "图 9. 计算得到的表型性状与人工测量结果的验证。左：计算株高与测量株高的相关性；右：计算叶片数与测量叶片数的相关性。",
        },
        {
            "label": "Fig. 10",
            "original": "FIGURE 10. Pearson correlation matrices of organ- and plant-level traits at the vegetative stage, with significance levels indicated on the right panel.",
            "translation": "图 10. 营养生长期器官级和植株级性状的 Pearson 相关矩阵，右侧面板标示显著性水平。",
        },
        {
            "label": "Fig. 11",
            "original": "FIGURE 11. Qualitative examples of LCR-GS extraction beyond the validated setting. Left: representative 3DGS scene renderings; right: corresponding extracted plant instances. (a) Mid-stage greenhouse muskmelon. (b) Late-stage greenhouse muskmelon, with arrows indicating representative residual or locally incomplete-separation regions. (c) Sweet olive and (d) peanut, with arrows indicating representative non-target residual regions.",
            "translation": "图 11. 验证设置之外的 LCR-GS 提取定性示例。左：代表性 3DGS 场景渲染；右：对应提取的植株实例。(a) 温室甜瓜中期；(b) 温室甜瓜后期，箭头标示代表性残留或局部分离不完整区域；(c) 桂花和 (d) 花生，箭头标示代表性非目标残留区域。",
        },
    ],
    "plantphenomics.0235.pdf": [
        {
            "label": "Fig. 1",
            "original": "Fig. 1. NeRFs are proposed as an alternative to traditional TLS scans for 3D plant reconstruction, offering cost-effective and efficient modeling from images captured at multiple angles using a smartphone camera, in contrast to the higher expense and extensive processing time required by TLS for multiangle scan registration.",
            "translation": "图 1. 作者提出将 NeRF 作为传统 TLS 扫描的替代方案用于三维植株重建。与 TLS 多角度扫描配准所需的高成本和长处理时间相比，NeRF 可利用智能手机从多角度采集的图像进行成本更低、效率更高的建模。",
        },
        {
            "label": "Fig. 3",
            "original": "Fig. 3. Example images input to NeRFs for reconstruction across 3 different scenarios. (A) Scenario I: Indoor single object. (B) Scenario II: Indoor multiple objects. (C) Scenario III: Outdoor scene.",
            "translation": "图 3. 三种不同场景下输入 NeRF 用于重建的示例图像。(A) 场景 I：室内单个对象；(B) 场景 II：室内多个对象；(C) 场景 III：室外场景。",
        },
        {
            "label": "Fig. 4",
            "original": "Fig. 4. Correlation analysis between different metrics with F1 Score via Pearson coefficients: (A) PSNR, (B) SSIM, and (C) LPIPS.",
            "translation": "图 4. 通过 Pearson 系数分析不同指标与 F1 分数的相关性：(A) PSNR，(B) SSIM，(C) LPIPS。",
        },
        {
            "label": "Fig. 5",
            "original": "Fig. 5. Camera pose estimations across 3 different scenarios. (A) Scenario I. (B) Scenario II. (C) Scenario III.",
            "translation": "图 5. 三种不同场景下的相机位姿估计。(A) 场景 I；(B) 场景 II；(C) 场景 III。",
        },
        {
            "label": "Fig. 6",
            "original": "Fig. 6. Precision and recall of 3D reconstruction using different NeRF techniques across different scenarios. Legend: Correct, Missing, Outlier.",
            "translation": "图 6. 不同场景下不同 NeRF 技术进行三维重建的精确率和召回率。图例：正确点、缺失点、离群点。",
        },
        {
            "label": "Fig. 8",
            "original": "Fig. 8. Scenes for validating the early stopping algorithm and their 3D reconstructions: original scenes in the first column, iterative reconstructions in the right column, and optimal iterations in the third column (*).",
            "translation": "图 8. 用于验证早停算法的场景及其三维重建结果：第一列为原始场景，右侧列为迭代重建结果，第三列标示最优迭代次数（*）。",
        },
    ],
}


@dataclass
class Block:
    id: str
    page: int
    type: str
    order: int
    original: str
    translation: str
    confidence: str
    notes: str = ""
    nearby_refs: list[str] | None = None


@dataclass
class PaperContext:
    pdf_path: Path
    out_dir: Path
    asset_dir: Path
    title: str
    authors: str
    journal: str
    doi: str
    metadata: dict


def normalize_text(text: str) -> str:
    text = text.replace("\u0000", "")
    text = text.replace("(cid:0)", "-")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = text.replace("R 2", "R2").replace("m IoU", "mIoU")
    text = text.replace("Ne RF", "NeRF").replace("SA M2", "SAM2")
    text = re.sub(r"\s+%", "%", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\bR2\s*=\s*", "R2 = ", text)
    return normalize_text(text)


def slug_reader_dir(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}_reader")


def pdf_text_pages(pdf_path: Path, raw: bool = False) -> list[str]:
    cmd = ["pdftotext"]
    if raw:
        cmd.append("-raw")
    cmd += ["-enc", "UTF-8", str(pdf_path), "-"]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return [p for p in proc.stdout.split("\f") if p.strip()]


def infer_context(pdf_path: Path) -> PaperContext:
    out_dir = slug_reader_dir(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}
        first_text = pdf.pages[0].extract_text(x_tolerance=1.5, y_tolerance=3) or ""
    title = normalize_text(metadata.get("Title") or "")
    if not title:
        title = infer_title_from_first_page(first_text)
    authors = normalize_text(metadata.get("Author") or infer_authors_from_first_page(first_text))
    subject = normalize_text(metadata.get("Subject") or "")
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", subject + " " + first_text)
    doi = doi_match.group(0).rstrip(".") if doi_match else ""
    journal = infer_journal(subject, first_text, title)
    return PaperContext(
        pdf_path=pdf_path,
        out_dir=out_dir,
        asset_dir=out_dir / "assets",
        title=title or pdf_path.stem,
        authors=authors or "Unknown authors",
        journal=journal,
        doi=doi,
        metadata=metadata,
    )


def infer_title_from_first_page(first_text: str) -> str:
    lines = [normalize_text(x) for x in first_text.splitlines() if normalize_text(x)]
    skip = ("RESEARCH ARTICLE", "TYPE", "PUBLISHED", "DOI", "OPEN ACCESS", "Citation:")
    candidates = [line for line in lines[:20] if not line.startswith(skip) and len(line) > 20]
    return candidates[0] if candidates else ""


def infer_authors_from_first_page(first_text: str) -> str:
    lines = [normalize_text(x) for x in first_text.splitlines() if normalize_text(x)]
    for idx, line in enumerate(lines[:25]):
        if re.search(r"\d,?\s*[A-Z][a-z]+| and |,", line) and not line.startswith(("Citation:", "DOI", "PUBLISHED")):
            if idx > 0 and len(line) > 10:
                return line
    return ""


def infer_journal(subject: str, first_text: str, title: str) -> str:
    if "Front. Plant Sci." in subject or "Frontiers in Plant Science" in first_text:
        return "Frontiers in Plant Science"
    if "Plant Phenomics" in subject or "Plant Phenomics" in first_text:
        m = re.search(r"Plant Phenomics[^.\n]*(?:\d{4}|100\d+|0235)?", subject)
        return normalize_text(m.group(0)) if m else "Plant Phenomics"
    return subject or title


def skip_line(line: str, ctx: PaperContext) -> bool:
    if not line:
        return False
    if line == ctx.journal or line == "Plant Phenomics":
        return True
    if re.fullmatch(r"\d+", line):
        return True
    skip_prefixes = (
        "Contents lists available",
        "journal homepage:",
        "https://doi.org",
        "Received ",
        "Available online",
        "Copyright",
        "Frontiers in Plant Science",
        "frontiersin.org",
        "Lin and Lin",
        "Citation:",
        "OPEN ACCESS",
        "EDITED BY",
        "REVIEWED BY",
        "PUBLISHED",
        "TYPE ",
        "DOI ",
        "COPYRIGHT",
    )
    if line.startswith(skip_prefixes):
        return True
    if "Plant Phenomics" in line and re.search(r"\d{4}|100\d+|0235", line):
        return True
    return False


def paragraphize_lines(lines: list[str], ctx: PaperContext, page_no: int) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            para = dehyphenate(" ".join(buf))
            if keep_paragraph(para):
                paragraphs.append(para)
            buf = []

    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line:
            flush()
            continue
        if skip_line(line, ctx):
            flush()
            continue
        if CAPTION_START_RE.match(line):
            flush()
            continue
        if SECTION_RE.match(line):
            flush()
            paragraphs.append(line)
            continue
        if line.startswith(("• ", "- ")):
            flush()
            buf = [line]
            continue
        buf.append(line)
    flush()
    return paragraphs


def keep_paragraph(text: str) -> bool:
    if len(text) < 3:
        return False
    if re.fullmatch(r"\d+", text):
        return False
    return True


def extract_blocks(ctx: PaperContext) -> tuple[list[Block], list[dict]]:
    blocks: list[Block] = []
    figures: list[dict] = []
    s_idx = 1
    c_idx = 1
    order = 1
    in_refs = False
    pages = pdf_text_pages(ctx.pdf_path, raw=False)
    for page_no, page_text in enumerate(pages, start=1):
        lines = page_text.splitlines()
        for para in paragraphize_lines(lines, ctx, page_no):
            if para.lower() == "references":
                in_refs = True
            block_id = f"S{s_idx:03d}"
            block_type = "section" if SECTION_RE.match(para) else ("reference" if in_refs else "body")
            blocks.append(
                Block(
                    id=block_id,
                    page=page_no,
                    type=block_type,
                    order=order,
                    original=para,
                    translation="",
                    confidence=confidence_for_text(para),
                    nearby_refs=extract_refs(para),
                )
            )
            s_idx += 1
            order += 1
    layout_captions = extract_layout_captions(ctx)
    captions_by_page: dict[int, list[str]] = {}
    for page_no, caption in layout_captions:
        captions_by_page.setdefault(page_no, []).append(caption)
    with pdfplumber.open(ctx.pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            captions = captions_by_page.get(page_no) or extract_captions(page)
            for caption in captions:
                block_id = f"C{c_idx:03d}"
                blocks.append(
                    Block(
                        id=block_id,
                        page=page_no,
                        type="caption",
                        order=order,
                        original=caption,
                        translation="",
                        confidence=confidence_for_text(caption),
                        nearby_refs=extract_refs(caption),
                    )
                )
                c_idx += 1
                order += 1
            figures.extend(extract_page_figures(page, page_no))
    return sorted(blocks, key=lambda b: (b.order, b.id)), figures


def extract_layout_captions(ctx: PaperContext) -> list[tuple[int, str]]:
    pages = pdf_text_pages(ctx.pdf_path, raw=False)
    captions: list[tuple[int, str]] = []
    for page_no, page_text in enumerate(pages, start=1):
        lines = [normalize_text(line) for line in page_text.splitlines()]
        current: list[str] = []
        for line in lines:
            if not line:
                if current:
                    captions.append((page_no, trim_caption(" ".join(current))))
                    current = []
                continue
            if is_true_caption_start(line):
                if current:
                    captions.append((page_no, trim_caption(" ".join(current))))
                current = [line]
                continue
            if current:
                if should_continue_caption(line, current):
                    current.append(line)
                else:
                    captions.append((page_no, trim_caption(" ".join(current))))
                    current = []
        if current:
            captions.append((page_no, trim_caption(" ".join(current))))
    clean: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for page_no, caption in captions:
        if 8 <= len(caption) <= 1200 and (page_no, caption) not in seen:
            clean.append((page_no, caption))
            seen.add((page_no, caption))
    return clean


def is_true_caption_start(line: str) -> bool:
    if not CAPTION_START_RE.match(line):
        return False
    if re.match(r"^Figure\s+\d+\s+(shows|compares|presents|visualizes|illustrates)\b", line, re.I):
        return False
    if re.match(r"^Fig\.\s*\d+[A-Za-z]?,", line):
        return False
    return True


def should_continue_caption(line: str, current: list[str]) -> bool:
    joined = " ".join(current)
    if len(joined) > 900:
        return False
    if SECTION_RE.match(line):
        return False
    if is_true_caption_start(line):
        return False
    if skip_raw_caption_line(line):
        return False
    if re.match(r"^(Abstract|Introduction|Materials and methods|Results|Discussion|Conclusion|References)\b", line, re.I):
        return False
    return True


def skip_raw_caption_line(line: str) -> bool:
    if re.fullmatch(r"\d+", line):
        return True
    if line.startswith(("Frontiers in", "Plant Phenomics", "Lin and Lin", "J. Ma et al.")):
        return True
    return False


def extract_refs(text: str) -> list[str]:
    return sorted(set(re.findall(r"(?:Fig\.|Figure|FIGURE|Table|TABLE)\s*\d+[A-Za-z]?", text)))


def confidence_for_text(text: str) -> str:
    if "cid:" in text or len(re.findall(r"\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b", text)) > 7:
        return "low"
    if re.search(r"[∫∑⃦⊤ℝ𝕊]|Algorithm|=|±|\bP\s*[<=>]", text):
        return "medium"
    return "high"


def extract_captions(page: pdfplumber.page.Page) -> list[str]:
    text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
    lines = [normalize_text(x) for x in text.splitlines()]
    captions: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            continue
        if is_true_caption_start(line):
            if current:
                captions.append(dehyphenate(" ".join(current)))
            current = [line]
            continue
        if current:
            if SECTION_RE.match(line) or len(" ".join(current)) > 1000:
                captions.append(dehyphenate(" ".join(current)))
                current = []
            elif not skip_caption_continuation(line):
                current.append(line)
    if current:
        captions.append(dehyphenate(" ".join(current)))
    clean: list[str] = []
    for cap in captions:
        cap = trim_caption(cap)
        if 8 <= len(cap) <= 1200 and cap not in clean:
            clean.append(cap)
    return clean


def skip_caption_continuation(line: str) -> bool:
    if line.startswith(("Frontiers in", "Plant Phenomics", "Lin and Lin")):
        return True
    if re.fullmatch(r"\d+", line):
        return True
    return False


def trim_caption(caption: str) -> str:
    caption = re.sub(r"\s+\d+$", "", caption)
    caption = re.split(
        r"\s(?=(?:[1-9]|1[0-9])(?:\.\d+){0,3}\.?\s+[A-Z]|"
        r"Abstract\b|Introduction\b|Materials and methods\b|Results\b|Discussion\b|Conclusion\b|References\b)",
        caption,
        maxsplit=1,
    )[0]
    return normalize_text(caption)


def extract_page_figures(page: pdfplumber.page.Page, page_no: int) -> list[dict]:
    boxes: list[tuple[float, float, float, float]] = []
    for img in page.images:
        w = float(img["width"])
        h = float(img["height"])
        top = float(img["top"])
        if page_no == 1 and (w < 120 or h < 80):
            continue
        if w < 35 or h < 25:
            continue
        if top < 55 and h < 65:
            continue
        boxes.append((float(img["x0"]), top, float(img["x1"]), float(img["bottom"])))
    if not boxes:
        return []
    if page_no == 1:
        boxes = [b for b in boxes if (b[2] - b[0]) > 120 and (b[3] - b[1]) > 60]
    if not boxes:
        return []
    if len(boxes) >= 6:
        clusters = cluster_many_panel_boxes(boxes, page.width)
    else:
        clusters = cluster_boxes(boxes, page.width, page.height)
    figures: list[dict] = []
    for idx, bbox in enumerate(clusters, start=1):
        x0, top, x1, bottom = bbox
        if (x1 - x0) < 70 or (bottom - top) < 50:
            continue
        figures.append(
            {
                "page": page_no,
                "image_no": idx,
                "bbox": [
                    round(max(0, x0 - 3), 2),
                    round(max(0, top - 3), 2),
                    round(min(page.width, x1 + 3), 2),
                    round(min(page.height, bottom + 3), 2),
                ],
            }
        )
    return figures


def cluster_many_panel_boxes(boxes: list[tuple[float, float, float, float]], page_w: float) -> list[tuple[float, float, float, float]]:
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    groups: list[list[tuple[float, float, float, float]]] = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        placed = False
        for group in groups:
            gx0, gy0, gx1, gy1 = group_union(group)
            gcx = (gx0 + gx1) / 2
            gcy = (gy0 + gy1) / 2
            # Same visual panel if vertically close or within a broad page-spanning band.
            if abs(cy - gcy) < 135 or (max(box[1], gy0) - min(box[3], gy1) < 60 and abs(cx - gcx) < page_w * 0.45):
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])
    unions = [group_union(group) for group in groups]
    return cluster_boxes(unions, page_w, 10_000)


def group_union(group: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in group)
    y0 = min(b[1] for b in group)
    x1 = max(b[2] for b in group)
    y1 = max(b[3] for b in group)
    return (x0, y0, x1, y1)


def cluster_boxes(boxes: list[tuple[float, float, float, float]], page_w: float, page_h: float) -> list[tuple[float, float, float, float]]:
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    clusters: list[tuple[float, float, float, float]] = []
    for box in boxes:
        placed = False
        for i, existing in enumerate(clusters):
            if should_merge(existing, box, page_w):
                clusters[i] = union_box(existing, box)
                placed = True
                break
        if not placed:
            clusters.append(box)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[float, float, float, float]] = []
        for box in clusters:
            for i, existing in enumerate(merged):
                if should_merge(existing, box, page_w):
                    merged[i] = union_box(existing, box)
                    changed = True
                    break
            else:
                merged.append(box)
        clusters = merged
    return sorted(clusters, key=lambda b: (b[1], b[0]))


def should_merge(a: tuple[float, float, float, float], b: tuple[float, float, float, float], page_w: float) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    h_gap = max(0, max(bx0 - ax1, ax0 - bx1))
    v_gap = max(0, max(by0 - ay1, ay0 - by1))
    overlap_x = max(0, min(ax1, bx1) - max(ax0, bx0))
    overlap_y = max(0, min(ay1, by1) - max(ay0, by0))
    min_w = min(ax1 - ax0, bx1 - bx0)
    min_h = min(ay1 - ay0, by1 - by0)
    if overlap_x > 0.25 * min_w and v_gap < 35:
        return True
    if overlap_y > 0.25 * min_h and h_gap < 35:
        return True
    if h_gap < 25 and v_gap < 25:
        return True
    # Multi-panel figures often span most of the page width as separate tiles.
    union = union_box(a, b)
    if (union[2] - union[0]) > 0.72 * page_w and v_gap < 80:
        return True
    return False


def union_box(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class Translator:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.tokenizer.src_lang = "eng_Latn"
        except Exception:
            self.model = False

    def translate_many(self, texts: list[str]) -> list[str]:
        out: list[str] = []
        self.load()
        for idx, text in enumerate(texts, start=1):
            if text in MANUAL_TRANSLATIONS:
                out.append(MANUAL_TRANSLATIONS[text])
                continue
            if should_not_translate(text):
                out.append(fallback_translate(text))
                continue
            if text in self.cache and not self.cache[text].startswith("【机器初译待精修】"):
                out.append(self.cache[text])
                continue
            if idx == 1 or idx % 25 == 0:
                print(f"Translating block {idx}/{len(texts)}", flush=True)
            translated = self.translate_text(text)
            self.cache[text] = translated
            out.append(translated)
            if idx % 25 == 0:
                self.save()
        self.save()
        return out

    def translate_text(self, text: str) -> str:
        if not self.model or self.tokenizer is None:
            return fallback_translate(text)
        chunks = chunk_text(text, limit=850)
        zh_chunks: list[str] = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("zho_Hans"),
                max_length=512,
                num_beams=1,
                do_sample=False,
            )
            zh_chunks.append(self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
        return " ".join(zh_chunks)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_text(text: str, limit: int = 850) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.;!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 > limit and buf:
            chunks.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def should_not_translate(text: str) -> bool:
    if SECTION_RE.match(text):
        return True
    if text.startswith("[") and re.match(r"^\[\d+\]\s", text):
        return True
    if len(re.findall(r"[∫∑{}<>]|\\bInput:|\\bOutput:", text)) > 2:
        return True
    return False


def fallback_translate(text: str) -> str:
    if text in MANUAL_TRANSLATIONS:
        return MANUAL_TRANSLATIONS[text]
    if SECTION_RE.match(text):
        return f"【标题暂译】{text}"
    if CAPTION_START_RE.match(text):
        return f"图表说明：{text}"
    if text.startswith("[") and re.match(r"^\[\d+\]\s", text):
        return "参考文献条目保留原文，未做逐条翻译。"
    return f"【机器初译待精修】{text}"


def polish_translation(original: str, zh: str) -> str:
    zh = normalize_text(zh)
    for old, new in TERM_FIXES.items():
        zh = zh.replace(old, new)
    for term in ["NeRF", "3DGS", "Gaussian Splatting", "COLMAP", "SAM", "SAM2", "YOLO", "PSNR", "SSIM", "LPIPS", "F1", "IoU", "RMSE", "MAE"]:
        zh = re.sub(r"\s*".join(map(re.escape, term)), term, zh, flags=re.I)
    if original.startswith(("•", "-")) and not zh.startswith(("•", "-")):
        zh = "• " + zh
    return zh


def translate_blocks(ctx: PaperContext, blocks: list[Block]) -> None:
    translatable = [b for b in blocks if b.type != "reference"]
    translator = Translator(ctx.out_dir / "translation_cache.json")
    translations = translator.translate_many([b.original for b in translatable])
    translated_by_id = {b.id: polish_translation(b.original, zh) for b, zh in zip(translatable, translations)}
    for block in blocks:
        if block.type == "reference":
            block.translation = "参考文献条目保留原文，未做逐条翻译。"
        else:
            block.translation = translated_by_id[block.id]


def crop_assets(ctx: PaperContext, figures: list[dict]) -> list[dict]:
    if ctx.asset_dir.exists():
        shutil.rmtree(ctx.asset_dir)
    ctx.asset_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict] = []
    with pdfplumber.open(ctx.pdf_path) as pdf:
        for fig_idx, fig in enumerate(figures, start=1):
            page = pdf.pages[fig["page"] - 1]
            cropped = page.crop(tuple(fig["bbox"]))
            im = cropped.to_image(resolution=220).original.convert("RGB")
            filename = f"fig{fig_idx}.png"
            im.save(ctx.asset_dir / filename, optimize=True)
            fig.update(
                {
                    "id": f"F{fig_idx:03d}",
                    "file": f"assets/{filename}",
                    "width": im.width,
                    "height": im.height,
                    "confidence": "high" if fig.get("image_no", 1) == 1 else "medium",
                }
            )
            enriched.append(fig)
    return enriched


def manual_caption_for_figure(pdf_name: str, fig: dict, idx: int) -> dict | None:
    captions = MANUAL_FIGURE_CAPTIONS.get(pdf_name)
    if not captions:
        return None
    if pdf_name == "plantphenomics.0235.pdf":
        page_caption_idx = {
            (2, 1): 0,
            (4, 1): 1,
            (9, 1): 2,
            (9, 2): 3,
            (11, 1): 4,
            (13, 1): 5,
        }.get((fig["page"], fig.get("image_no", 1)))
        return captions[page_caption_idx] if page_caption_idx is not None else None
    return captions[idx] if idx < len(captions) else None


def pair_figures_with_captions(ctx: PaperContext, figures: list[dict], blocks: list[Block]) -> list[dict]:
    fig_caps = [b for b in blocks if b.type == "caption" and re.match(r"^(Fig\.|Figure|FIGURE)\s*\d+", b.original)]
    page_seen: dict[int, int] = {}
    for idx, fig in enumerate(figures):
        manual_caption = manual_caption_for_figure(ctx.pdf_path.name, fig, idx)
        if manual_caption:
            fig["caption_id"] = "manual-layout"
            fig["caption_original"] = manual_caption["original"]
            fig["caption_translation"] = manual_caption["translation"]
            fig["label"] = manual_caption["label"]
            fig["placed_near"] = find_first_mention(manual_caption["original"], blocks)
            continue
        caption = None
        same_page = [c for c in fig_caps if c.page == fig["page"]]
        if same_page:
            seen = page_seen.get(fig["page"], 0)
            caption = same_page[min(seen, len(same_page) - 1)]
            page_seen[fig["page"]] = seen + 1
        elif idx < len(fig_caps):
            caption = fig_caps[idx]
        if caption:
            fig["caption_id"] = caption.id
            fig["caption_original"] = caption.original
            fig["caption_translation"] = caption.translation
            m = re.match(r"^(Fig\.|Figure|FIGURE)\s*(\d+)", caption.original)
            fig["label"] = f"Fig. {m.group(2)}" if m else fig["id"]
            fig["placed_near"] = find_first_mention(caption.original, blocks)
        else:
            fig["caption_id"] = None
            fig["caption_original"] = "Caption not confidently extracted."
            fig["caption_translation"] = "图注未能可靠抽取。"
            fig["label"] = fig["id"]
            fig["placed_near"] = None
    return figures


def find_first_mention(caption: str, blocks: list[Block]) -> str | None:
    m = re.match(r"^(?:Fig\.|Figure|FIGURE)\s*(\d+)", caption)
    if not m:
        return None
    patterns = [f"Fig. {m.group(1)}", f"Figure {m.group(1)}", f"FIGURE {m.group(1)}"]
    for block in blocks:
        if block.type == "body" and any(p in block.original for p in patterns):
            return f"p.{block.page} {block.id}"
    return None


def build_markdown(ctx: PaperContext, blocks: list[Block], figures: list[dict]) -> str:
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    figure_after: dict[str, list[dict]] = {}
    for fig in figures:
        near = fig.get("placed_near")
        key = near.split()[-1] if near else None
        if key:
            figure_after.setdefault(key, []).append(fig)
        else:
            page_blocks = by_page.get(fig["page"], [])
            if page_blocks:
                figure_after.setdefault(page_blocks[0].id, []).append(fig)

    md: list[str] = []
    md.append("---")
    md.append(f"title: {json.dumps(ctx.title, ensure_ascii=False)}")
    md.append(f"authors: {json.dumps(ctx.authors, ensure_ascii=False)}")
    md.append(f"journal: {json.dumps(ctx.journal, ensure_ascii=False)}")
    md.append(f"doi: {ctx.doi}")
    md.append(f"source_pdf: {ctx.pdf_path}")
    md.append(f"generated: {date.today().isoformat()}")
    md.append("reader_type: bilingual_source_grounded_markdown")
    md.append("---\n")
    md.append(f"# {ctx.title}\n")
    md.append(f"**作者：** {ctx.authors}\n")
    md.append(f"**来源：** {ctx.journal}" + (f"; DOI: {ctx.doi}" if ctx.doi else "") + "\n")
    md.append("**说明：** 本文件为全文中英对照阅读稿。中文为机器初译并经过领域术语规则校正；双栏、公式、表格和复杂多子图区域的低置信点记录在 `translation_notes.md`。\n")
    md.append("## 页面/章节索引\n")
    for block in blocks:
        if block.type == "section":
            md.append(f"- [{block.original}](#{block.id.lower()}) — p.{block.page}")
    md.append("")
    md.append("## 术语表\n")
    md.append("| English | 中文 |")
    md.append("| --- | --- |")
    terms = [
        ("plant phenotyping", "植物表型/植物表型分析"),
        ("3D Gaussian Splatting (3DGS)", "三维高斯泼溅（3DGS）"),
        ("Neural Radiance Fields (NeRF)", "神经辐射场（NeRF）"),
        ("Structure from Motion (SfM)", "运动恢复结构（SfM）"),
        ("COLMAP", "COLMAP"),
        ("point cloud", "点云"),
        ("instance segmentation", "实例分割"),
        ("trait extraction", "性状提取"),
        ("PSNR / SSIM / LPIPS", "PSNR / SSIM / LPIPS 指标"),
    ]
    for en, zh in terms:
        md.append(f"| {en} | {zh} |")
    md.append("\n## 全文中英对照\n")
    for page in sorted(by_page):
        md.append(f"\n## Page {page}\n")
        for block in sorted(by_page[page], key=lambda b: b.order):
            if block.type == "caption" and re.match(r"^(Fig\.|Figure|FIGURE)", block.original):
                continue
            md.append(f'<a id="{block.id}"></a>')
            if block.type == "section":
                md.append(f"### {block.original}")
            elif block.type == "caption":
                md.append(f"### {block.original.split('.')[0] if '.' in block.original else block.original}")
            md.append(f"**Source:** p.{block.page} {block.id}  \n**Type:** {block.type}  \n**Confidence:** {block.confidence}\n")
            md.append(f"**Original:** {block.original}\n")
            md.append(f"**中文:** {block.translation}\n")
            for fig in figure_after.get(block.id, []):
                md.extend(render_figure_block(fig))
    md.append("\n## 阅读提示\n")
    md.append("- 先读摘要、方法流程图和结果图表，再回到方法细节，可更快抓住论文贡献。")
    md.append("- 对图像重建/分割论文，重点核对数据采集方式、3D 表示、分割或性状提取流程、评价指标和失败案例。")
    md.append("- 公式、表格和复杂多子图页面已经保留原文锚点；若要精校中文，优先处理 `translation_notes.md` 中标为 low/medium 的块。")
    return "\n".join(md) + "\n"


def render_figure_block(fig: dict) -> list[str]:
    label = fig.get("label") or fig["id"]
    title = re.sub(r"^(图表说明：|图注：)", "", fig.get("caption_translation", label))
    title = re.sub(r"^图\s*\d+[.．]\s*", "", title)
    return [
        f'<a id="{fig["id"]}"></a>',
        f"### {label}. {title[:90]}",
        f"**Placed near:** {fig.get('placed_near') or 'p.' + str(fig['page'])}  ",
        f"**Source:** p.{fig['page']} {fig.get('caption_id') or fig['id']}  ",
        f"**Crop confidence:** {fig.get('confidence', 'medium')}\n",
        f"![{label}]({fig['file']})\n",
        f"**Original caption:** {fig.get('caption_original', '')}\n",
        f"**中文图注:** {fig.get('caption_translation', '')}\n",
        "**Reading note:** 重点查看该图如何支撑相邻正文中的流程、比较、消融或性状提取结果。\n",
    ]


def write_source_map(ctx: PaperContext, blocks: list[Block], figures: list[dict]) -> None:
    data = {
        "source_pdf": str(ctx.pdf_path),
        "metadata": ctx.metadata,
        "title": ctx.title,
        "authors": ctx.authors,
        "journal": ctx.journal,
        "doi": ctx.doi,
        "blocks": [asdict(b) for b in blocks],
        "figures": figures,
    }
    (ctx.out_dir / "source_map.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes(ctx: PaperContext, blocks: list[Block], figures: list[dict]) -> None:
    low = [b for b in blocks if b.confidence != "high"]
    notes = [
        "# Translation and Extraction Notes\n",
        f"- Source PDF: `{ctx.pdf_path}`",
        "- PDF type: selectable-text PDF.",
        "- Paper type: 3D reconstruction / plant phenotyping methods paper.",
        "- Translation method: NLLB machine translation with domain-term post-processing.",
        "- Draft-mode caveat: equations, tables, references, and complex multi-panel figure pages may need human polishing.",
        f"- Text/caption blocks: {len(blocks)}; figure crops: {len(figures)}.",
        "\n## Low/Medium Confidence Blocks\n",
    ]
    if low:
        for block in low[:240]:
            notes.append(f"- `{block.id}` p.{block.page} ({block.type}, {block.confidence}): {block.original[:180]}")
    else:
        notes.append("- None.")
    notes.append("\n## Figure Crop Notes\n")
    for fig in figures:
        notes.append(f"- `{fig['id']}` p.{fig['page']} `{fig['file']}` bbox={fig['bbox']} caption={fig.get('caption_id')}")
    notes.append("\n## Known Limitations\n")
    notes.append("- Tables are represented as caption/source blocks and nearby prose; exact cell-level table reconstruction is not guaranteed.")
    notes.append("- Figure crops use PDF image-object clustering. For pages composed of many subimages, crops may cover the whole visual panel instead of individual subpanels.")
    notes.append("- References are retained as source blocks when extractable but are not translated.")
    (ctx.out_dir / "translation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def verify_outputs(ctx: PaperContext, figures: list[dict]) -> None:
    paper = ctx.out_dir / "paper.md"
    source_map = ctx.out_dir / "source_map.json"
    notes = ctx.out_dir / "translation_notes.md"
    text = paper.read_text(encoding="utf-8")
    assert "**Original:**" in text and "**中文:**" in text
    assert text.count("**Original:**") == text.count("**中文:**")
    for fig in figures:
        assert (ctx.out_dir / fig["file"]).exists(), fig["file"]
        assert fig["id"] in text, fig["id"]
    data = json.loads(source_map.read_text(encoding="utf-8"))
    assert data["blocks"]
    assert notes.exists()


def build_reader(pdf_path: Path) -> PaperContext:
    ctx = infer_context(pdf_path)
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== Building {pdf_path.name} ==")
    print(f"Output: {ctx.out_dir}")
    blocks, figures = extract_blocks(ctx)
    translate_blocks(ctx, blocks)
    figures = crop_assets(ctx, figures)
    figures = pair_figures_with_captions(ctx, figures, blocks)
    (ctx.out_dir / "paper.md").write_text(build_markdown(ctx, blocks, figures), encoding="utf-8")
    write_source_map(ctx, blocks, figures)
    write_notes(ctx, blocks, figures)
    verify_outputs(ctx, figures)
    print(f"Wrote {ctx.out_dir} blocks={len(blocks)} figures={len(figures)}")
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", type=Path)
    args = parser.parse_args()
    pdfs = args.pdfs or DEFAULT_PDFS
    for pdf in pdfs:
        build_reader(pdf)


if __name__ == "__main__":
    main()
