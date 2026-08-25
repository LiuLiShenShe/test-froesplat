#!/usr/bin/env python3
"""Build a bilingual Markdown reader for the IPENS Plant Phenomics PDF."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import pdfplumber
from PIL import Image


PDF_PATH = Path("/data/fj/F2DMAS/00参考文章/1-s2.0-S2643651525001128-main.pdf")
OUT_DIR = Path("/data/fj/F2DMAS/00参考文章/1-s2.0-S2643651525001128-main_reader")
ASSET_DIR = OUT_DIR / "assets"


TITLE = "IPENS: Interactive unsupervised framework for rapid plant phenotyping extraction via NeRF-SAM2 fusion"
AUTHORS = "Wentao Song; He Huang; Fang Qu; Jiaqi Zhang; Longhui Fang; Yuwei Hao; Chenyang Peng; Youqiang Sun"
DOI = "10.1016/j.plaphe.2025.100106"
JOURNAL = "Plant Phenomics 7 (2025) 100106"


TERM_FIXES = {
    "植物现象型": "植物表型",
    "植物口哨": "植物表型",
    "现象型": "表型",
    "智能繁殖": "智能育种",
    "繁殖": "育种",
    "特征改进": "性状改良",
    "特性改进": "性状改良",
    "点云": "点云",
    "辐射场": "辐射场",
    "分割": "分割",
    "无人监督": "无监督",
    "交互式": "交互式",
    "谷物": "籽粒",
    "粒级": "籽粒级",
    "籽籽粒": "籽粒",
    "米粒": "水稻籽粒",
    "大米粒": "水稻籽粒",
    "水稻和小麦": "水稻和小麦",
    "叶面积": "叶片表面积",
    "体素体积": "体素体积",
    "语音体积": "体素体积",
    "语音量": "体素",
    "音量": "体素体积",
    "平均交叉比": "平均交并比",
    "交叉联盟": "交并比",
    "型号化": "表型分析",
    "型方法": "表型方法",
    "型提取": "表型提取",
    "型平台": "表型平台",
    "型结构技术": "表型技术",
    "现象类型": "表型",
    "现象数据": "表型数据",
    "互动": "交互式",
    "面具": "掩膜",
    "框架": "框架",
    "注释": "标注",
    "不受监督": "无监督",
    "零射": "零样本",
    "子粒": "籽粒",
    "小叶子": "穗部",
    "分区": "分割",
    "细分": "分割",
    "位定位": "定位",
    "高斯式分光": "3D Gaussian Splatting",
    "表表型": "表型",
    "农业型化": "农业表型分析",
    "现象类表型": "表型",
    "3D高质谱": "3D Gaussian Splatting",
    "被强奸的植物": "油菜植株",
    "大米和小麦": "水稻和小麦",
    "点点": "点云",
    "自封": "自遮挡",
    "快点漂移": "提示点漂移",
    "无监督的驱动": "无监督提示",
    "可差异化染": "可微渲染",
    "恐慌": "穗部",
    "异形": "表型",
    "fenootyping": "phenotyping",
    "波素": "体素",
    "米数据": "水稻数据",
    "米 78.18%": "水稻 78.18%",
    "米 23.41%": "水稻 23.41%",
    "米 55.82%": "水稻 55.82%",
    "干子": "茎",
}


MANUAL_TRANSLATIONS = {
    TITLE: "IPENS：通过 NeRF-SAM2 融合实现快速植物表型提取的交互式无监督框架",
    "Research Article": "研究论文",
    AUTHORS: "Wentao Song；He Huang；Fang Qu；Jiaqi Zhang；Longhui Fang；Yuwei Hao；Chenyang Peng；Youqiang Sun",
    "Affiliations: Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei, China; University of Science and Technology of China, Hefei, China; Anhui Agricultural University, Hefei, China; Anhui Jianzhu University, Hefei, China.": "作者单位：中国科学院合肥物质科学研究院；中国科学技术大学；安徽农业大学；安徽建筑大学。",
    "A R T I C L E I N F O": "文章信息",
    "A B S T R A C T": "摘要",
    "Keywords: Rice and wheat phenotype; NeRF; SAM2; 3D instance segmentation; Unsupervised": "关键词：水稻和小麦表型；NeRF；SAM2；3D 实例分割；无监督",
    "1. Introduction": "1. 引言",
    "2. Materials and methods": "2. 材料与方法",
    "2.1. Overview of the method": "2.1. 方法概览",
    "2.2. Data acquisition and reconstruction": "2.2. 数据采集与重建",
    "2.3. Dataset construction": "2.3. 数据集构建",
    "2.4. Neural Radiance Fields": "2.4. 神经辐射场",
    "2.5. Pipeline of the interactive model in IPENS": "2.5. IPENS 中交互模型的流程",
    "2.6. 3D Mask Representation and loss function": "2.6. 3D 掩膜表示与损失函数",
    "2.7. Auxiliary optimization strategy": "2.7. 辅助优化策略",
    "2.8. Phenotypic data extraction method": "2.8. 表型数据提取方法",
    "2.8.1. Voxel volumes of grains and panicles": "2.8.1. 籽粒与穗部的体素体积",
    "2.8.2. Leaf surface area": "2.8.2. 叶片表面积",
    "2.8.3. Leaf length and width": "2.8.3. 叶长与叶宽",
    "2.9. Evaluation metrics": "2.9. 评价指标",
    "3. Experimental setup and results": "3. 实验设置与结果",
    "3.1. Quantitative experiment": "3.1. 定量实验",
    "3.1.1. 3D segmentation performance": "3.1.1. 3D 分割性能",
    "3.1.2. Time performance analysis": "3.1.2. 时间性能分析",
    "3.2. Phenotypic analysis": "3.2. 表型分析",
    "3.2.1. Analysis of rice grain voxel volume": "3.2.1. 水稻籽粒体素体积分析",
    "3.2.2. Analysis of wheat panicle voxel volume": "3.2.2. 小麦穗体素体积分析",
    "3.2.3. Leaf phenotypic analysis": "3.2.3. 叶片表型分析",
    "4. Discussion": "4. 讨论",
    "4.1. Interpretation of 3D segmentation performance": "4.1. 3D 分割性能解释",
    "4.2. Effectiveness of the proposed method": "4.2. 所提方法的有效性",
    "4.3. Limitation and future prospects": "4.3. 局限性与未来展望",
    "5. Conclusion": "5. 结论",
    "5.1. Multi-species point cloud extraction visualization": "5.1. 多物种点云提取可视化",
    "5.2. Time consumption of 3D reconstruction models": "5.2. 3D 重建模型耗时",
    "Author contributions": "作者贡献",
    "Funding": "基金资助",
    "Data availability": "数据可用性",
    "Declaration of competing interest": "利益冲突声明",
    "Appendix A. Supplementary data": "附录 A. 补充数据",
    "References": "参考文献",
    "Fig. 1. Overall workflow of IPENS: Data preparation, model & method, and phenotyping extraction.": "图 1. IPENS 的总体流程：数据准备、模型与方法，以及表型提取。",
    "Fig. 2. Data acquisition and instance segmentation.": "图 2. 数据采集与实例分割。",
    "Fig. 4. Illustration of residual handling and SSIM-Based fast prompt frame detection process.": "图 4. 残差处理与基于 SSIM 的快速提示帧检测过程示意图。",
    "Fig. 5. (a) Radar chart of evaluation metrics across rice organs. (b) Radar chart of evaluation metrics across wheat organs.": "图 5. (a) 水稻器官各评价指标的雷达图。(b) 小麦器官各评价指标的雷达图。",
    "Fig. 8. Indoor high-throughput phenotyping chamber system and outdoor field data acquisition vehicle.": "图 8. 室内高通量表型舱系统与室外田间数据采集车。",
}

ABSTRACT_EN = (
    "Advanced plant phenotyping technologies are vital for trait improvement and accelerating intelligent breeding. "
    "Due to the species diversity of plants, existing methods heavily rely on large-scale high-precision manually annotated data. "
    "For self-occluded objects at the grain level, unsupervised methods often prove ineffective. "
    "This study proposes IPENS, an interactive unsupervised multi-target point cloud extraction method. "
    "It utilizes radiance field information to lift 2D masks, segmented by SAM2 (Segment Anything Model 2), into 3D space for target point cloud extraction. "
    "A multi-target collaborative optimization strategy addresses the challenge of segmenting multiple targets from a single interaction. "
    "On a rice dataset, IPENS achieves a grain-level segmentation mean Intersection over Union (mIoU) of 63.72%. "
    "For phenotypic trait estimation, it achieves a grain voxel volume coefficient of determination R2 = 0.7697 (Root Mean Square Error, RMSE = 0.0025), "
    "leaf surface area R2 = 0.84 (RMSE = 18.93), and leaf length and width prediction accuracies of R2 = 0.97 and R2 = 0.87 (RMSE = 1.49 and 0.21). "
    "On a wheat dataset, IPENS further improves segmentation performance to a mIoU of 89.68%, with exceptional phenotypic estimation results: "
    "panicle voxel volume R2 = 0.9956 (RMSE = 0.0055), leaf surface area R2 = 1.00 (RMSE = 0.67), and leaf length and width predictions reaching "
    "R2 = 0.99 and R2 = 0.92 (RMSE = 0.23 and 0.15). "
    "Without requiring annotated data, IPENS rapidly extracts grain-level point clouds for multiple targets within 3 min using single-round image interactions. "
    "These features make IPENS a high-quality, non-invasive phenotypic extraction solution for rice and wheat, offering significant potential to enhance intelligent breeding."
)

MANUAL_TRANSLATIONS[ABSTRACT_EN] = (
    "先进的植物表型技术对于性状改良和加速智能育种至关重要。由于植物物种多样性，现有方法高度依赖大规模、高精度的人工标注数据。"
    "对于籽粒级别存在自遮挡的目标，无监督方法往往效果有限。本文提出 IPENS，一种交互式无监督多目标点云提取方法。"
    "该方法利用辐射场信息，将 SAM2（Segment Anything Model 2）分割得到的 2D 掩膜提升到 3D 空间，从而提取目标点云。"
    "多目标协同优化策略解决了单次交互中分割多个目标的挑战。在水稻数据集上，IPENS 的籽粒级分割平均交并比（mIoU）达到 63.72%。"
    "在表型性状估计方面，籽粒体素体积的决定系数 R2 = 0.7697（RMSE = 0.0025），叶片表面积 R2 = 0.84（RMSE = 18.93），"
    "叶长和叶宽预测精度分别为 R2 = 0.97 与 R2 = 0.87（RMSE = 1.49 与 0.21）。在小麦数据集上，IPENS 将分割性能进一步提升到 mIoU = 89.68%，"
    "并取得优异的表型估计结果：穗部体素体积 R2 = 0.9956（RMSE = 0.0055），叶片表面积 R2 = 1.00（RMSE = 0.67），"
    "叶长和叶宽预测分别达到 R2 = 0.99 与 R2 = 0.92（RMSE = 0.23 与 0.15）。在不需要标注数据的情况下，IPENS 通过单轮图像交互，"
    "可在 3 分钟内快速提取多个目标的籽粒级点云。这些特性使 IPENS 成为面向水稻和小麦的高质量、非侵入式表型提取方案，并具有促进智能育种的重要潜力。"
)


SECTION_RE = re.compile(
    r"^(?:(?:[1-5](?:\.\d+){0,3})\.?\s+[A-Z][A-Za-z0-9 ,&()/-]+|"
    r"Author contributions|Funding|Data availability|Declaration of competing interest|"
    r"Appendix A\. Supplementary data|References)$"
)


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


def normalize_text(text: str) -> str:
    text = text.replace("\u0000", "")
    text = text.replace("(cid:0)", "-")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = text.replace("R 2", "R2").replace("m IoU", "mIoU")
    text = text.replace("Ne RF", "NeRF").replace("SA M2", "SAM2")
    return text.strip()


def dehyphenate(text: str) -> str:
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\bR2\s*=\s*", "R2 = ", text)
    text = re.sub(r"\s+%", "%", text)
    return normalize_text(text)


def line_text(line: list[dict]) -> str:
    return normalize_text(" ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])))


def group_lines(words: list[dict]) -> list[list[dict]]:
    lines: list[list[dict]] = []
    current: list[dict] = []
    last_top: float | None = None
    for word in sorted(words, key=lambda w: (round(w["top"] / 3) * 3, w["x0"])):
        top = round(word["top"] / 3) * 3
        if last_top is None or abs(top - last_top) <= 1:
            current.append(word)
        else:
            if current:
                lines.append(current)
            current = [word]
        last_top = top
    if current:
        lines.append(current)
    return lines


def extract_column_paragraphs(page: pdfplumber.page.Page, page_no: int) -> list[str]:
    """Extract reading paragraphs from a two-column Elsevier paper page."""
    crop_top = 45 if page_no > 1 else 145
    crop_bottom = 750
    body_words = [
        w
        for w in page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False, use_text_flow=False)
        if crop_top <= w["top"] <= crop_bottom
    ]
    if page_no == 1:
        # Page 1 has title/metadata across the page and only the lower body uses columns.
        return extract_first_page(page)

    paragraphs: list[str] = []
    for x0, x1 in [(35, 292), (303, 560)]:
        col_words = [w for w in body_words if x0 <= ((w["x0"] + w["x1"]) / 2) <= x1]
        lines = []
        for line in group_lines(col_words):
            txt = line_text(line)
            if not txt:
                continue
            y = sum(w["top"] for w in line) / len(line)
            if should_skip_line(txt, y):
                continue
            lines.append((y, txt))
        paragraphs.extend(lines_to_paragraphs(lines))
    return [p for p in paragraphs if keep_paragraph(p)]


def extract_first_page(page: pdfplumber.page.Page) -> list[str]:
    words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False, use_text_flow=False)
    lines = []
    for line in group_lines([w for w in words if 145 <= w["top"] <= 730]):
        txt = line_text(line)
        if not txt:
            continue
        y = sum(w["top"] for w in line) / len(line)
        lines.append((y, txt))
    raw = [txt for _, txt in sorted(lines)]
    out: list[str] = []
    # Manually stabilize title-page metadata because it spans different regions.
    out.append("Research Article")
    out.append(TITLE)
    out.append(AUTHORS)
    out.append(
        "Affiliations: Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei, China; "
        "University of Science and Technology of China, Hefei, China; Anhui Agricultural University, Hefei, China; "
        "Anhui Jianzhu University, Hefei, China."
    )
    out.append("Keywords: Rice and wheat phenotype; NeRF; SAM2; 3D instance segmentation; Unsupervised")
    abstract_lines = []
    in_abs = False
    for txt in raw:
        if "A B S T R A C T" in txt:
            in_abs = True
            txt = txt.split("A B S T R A C T", 1)[-1].strip()
        if in_abs:
            if txt.startswith("1. Introduction") or "Corresponding author" in txt:
                break
            if txt.startswith("Keywords:") or txt in {"A R T I C L E I N F O", "A B S T R A C T"}:
                continue
            # Drop left-column keyword fragments that share the abstract y range.
            if txt in {"Rice and wheat phenotype", "NeRF", "SAM2", "3D instance segmentation", "Unsupervised"}:
                continue
            abstract_lines.append(txt)
    abstract = dehyphenate(" ".join(abstract_lines))
    abstract = re.sub(r"^Advanced", "Advanced", abstract)
    if abstract:
        out.append(abstract)
    intro_lines = []
    start_intro = False
    for txt in raw:
        if txt.startswith("1. Introduction"):
            start_intro = True
            intro_lines.append("1. Introduction")
            rest = txt[len("1. Introduction") :].strip()
            if rest:
                intro_lines.append(rest)
            continue
        if start_intro:
            if txt.startswith("* Corresponding author") or txt.startswith("https://doi.org"):
                break
            if any(
                marker in txt
                for marker in [
                    "E-mail address:",
                    "Contents lists available",
                    "Plant Phenomics",
                    "journal homepage",
                    "Published by Elsevier",
                    "Received ",
                    "Available online",
                ]
            ):
                continue
            intro_lines.append(txt)
    out.extend(lines_to_paragraphs([(i, t) for i, t in enumerate(intro_lines)]))
    return [p for p in out if keep_paragraph(p)]


def should_skip_line(txt: str, y: float) -> bool:
    if txt.startswith("W. Song et al.") or txt == JOURNAL:
        return True
    if re.fullmatch(r"\d+", txt):
        return True
    if txt.startswith("Fig. ") or txt.startswith("Table "):
        return True
    if txt.startswith("Algorithm "):
        return False
    if y > 745 and re.fullmatch(r"\d+", txt):
        return True
    return False


def keep_paragraph(p: str) -> bool:
    if len(p) < 3:
        return False
    if re.fullmatch(r"\d+", p):
        return False
    if p.startswith("Plant Phenomics 7"):
        return False
    return True


def lines_to_paragraphs(lines: list[tuple[float, str]]) -> list[str]:
    paragraphs: list[str] = []
    buf = ""
    for _, txt in sorted(lines):
        txt = normalize_text(txt)
        if not txt:
            continue
        is_heading = bool(SECTION_RE.match(txt)) or txt.startswith("Algorithm ")
        is_list = txt.startswith("• ")
        if is_heading:
            if buf:
                paragraphs.append(dehyphenate(buf))
                buf = ""
            paragraphs.append(txt)
            continue
        if is_list:
            if buf:
                paragraphs.append(dehyphenate(buf))
            buf = txt
            continue
        if buf and txt.startswith("• "):
            paragraphs.append(dehyphenate(buf))
            buf = txt
            continue
        buf = f"{buf} {txt}".strip()
    if buf:
        paragraphs.append(dehyphenate(buf))
    return paragraphs


def extract_blocks() -> tuple[list[Block], list[dict], dict]:
    blocks: list[Block] = []
    figures: list[dict] = []
    metadata: dict = {}
    s_idx = 1
    c_idx = 1
    order = 1
    raw_by_page = extract_raw_pages()
    in_references = False
    for page_no, page_text in enumerate(raw_by_page, start=1):
        for para in raw_page_paragraphs(page_text, page_no):
            if para == "References":
                in_references = True
            block_id = f"S{s_idx:03d}"
            block_type = "section" if SECTION_RE.match(para) else ("reference" if in_references else "body")
            blocks.append(
                Block(
                    id=block_id,
                    page=page_no,
                    type=block_type,
                    order=order,
                    original=para,
                    translation="",
                    confidence=confidence_for_text(para),
                    nearby_refs=sorted(set(re.findall(r"(?:Fig\.|Table)\s*S?\d+[a-z]?", para))),
                )
            )
            s_idx += 1
            order += 1
    with pdfplumber.open(PDF_PATH) as pdf:
        metadata = pdf.metadata or {}
        for page_no, page in enumerate(pdf.pages, start=1):
            captions = extract_captions(page, page_no)
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
                        nearby_refs=sorted(set(re.findall(r"(?:Fig\.|Table)\s*S?\d+[a-z]?", caption))),
                    )
                )
                c_idx += 1
                order += 1
            figures.extend(extract_figures_for_page(page, page_no))
    blocks.sort(key=lambda b: (b.order, b.id))
    return blocks, figures, metadata


def extract_raw_pages() -> list[str]:
    proc = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(PDF_PATH), "-"],
        check=True,
        text=True,
        capture_output=True,
    )
    pages = proc.stdout.split("\f")
    return [p for p in pages if p.strip()]


def raw_page_paragraphs(text: str, page_no: int) -> list[str]:
    raw_lines = text.splitlines()
    lines = [normalize_text(line) for line in raw_lines]
    if page_no == 1:
        return first_page_raw_paragraphs([line for line in lines if line])
    return paragraphize_default_lines(lines)


def first_page_raw_paragraphs(lines: list[str]) -> list[str]:
    out = [
        "Research Article",
        TITLE,
        AUTHORS,
        "Affiliations: Hefei Institutes of Physical Science, Chinese Academy of Sciences, Hefei, China; University of Science and Technology of China, Hefei, China; Anhui Agricultural University, Hefei, China; Anhui Jianzhu University, Hefei, China.",
        "Keywords: Rice and wheat phenotype; NeRF; SAM2; 3D instance segmentation; Unsupervised",
        ABSTRACT_EN,
        "1. Introduction",
    ]
    try:
        start = lines.index("1. Introduction") + 1
    except ValueError:
        start = 0
    intro_lines = []
    for line in lines[start:]:
        if skip_raw_line(line):
            continue
        if line.startswith("* Corresponding author") or line.startswith("E-mail address:") or line.startswith("Contents lists"):
            break
        intro_lines.append(line)
    out.extend(lines_to_paragraphs([(i, line) for i, line in enumerate(intro_lines)]))
    return [p for p in out if keep_paragraph(p)]


def paragraphize_default_lines(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            para = dehyphenate(" ".join(buf))
            if keep_paragraph(para):
                paragraphs.append(para)
            buf = []

    for line in lines:
        line = normalize_text(line)
        if not line:
            flush()
            continue
        if skip_raw_line(line):
            flush()
            continue
        if line.startswith("Fig. ") or line.startswith("Table "):
            flush()
            continue
        if SECTION_RE.match(line) or line.startswith("Algorithm "):
            flush()
            paragraphs.append(line)
            continue
        if line.startswith("• "):
            flush()
            buf = [line]
            continue
        buf.append(line)
    flush()
    return paragraphs


def skip_raw_line(line: str) -> bool:
    if line in {"A R T I C L E I N F O", "A B S T R A C T", "Keywords:"}:
        return True
    if line in {"a", "b", "c", "d", ","}:
        return True
    if line.startswith("W. Song et al.") or line == JOURNAL or line == "Plant Phenomics":
        return True
    if line.startswith("journal homepage:") or line.startswith("Contents lists available"):
        return True
    if line.startswith("https://doi.org") or line.startswith("Received ") or line.startswith("Available online"):
        return True
    if line.startswith("2643-6515/") or line.startswith("license ("):
        return True
    if re.fullmatch(r"\d+", line):
        return True
    return False


def confidence_for_text(text: str) -> str:
    if "cid:" in text or len(re.findall(r"\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b", text)) > 5:
        return "low"
    if re.search(r"[∫∑⃦⊤ℝ𝕊]|Algorithm|=", text):
        return "medium"
    return "high"


def extract_captions(page: pdfplumber.page.Page, page_no: int) -> list[str]:
    text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
    text = normalize_text(text)
    candidates = []
    for match in re.finditer(r"(Fig\.\s*\d+\.[^£]+?|Table\s*\d+[^£]+?)(?=(?:\s(?:Fig\.|Table)\s*\d+|\s\d+\.\d+\.|\s[A-Z][a-z]+ contributions|$))", text):
        cap = match.group(1).strip()
        cap = re.sub(r"\s+\d+$", "", cap)
        if cap.startswith("Fig."):
            cap = cap.split(" W. Song et al.", 1)[0]
            # Stop accidental body continuation at common sentence starts.
            cap = re.split(r"\s(?:grains, with|effectively fills|Table \d+[a-z]? shows|This paper conducts)", cap)[0]
        if cap.startswith("Table"):
            cap = re.split(r"\s(?:Tables? \d+[a-z]? presents|Fig\. \d+|The results show)", cap)[0]
        if 10 <= len(cap) <= 900 and cap not in candidates:
            candidates.append(cap)
    manual = {
        3: ["Fig. 1. Overall workflow of IPENS: Data preparation, model & method, and phenotyping extraction."],
        4: [
            "Fig. 2. Data acquisition and instance segmentation.",
            "Table 1. 3D instance statistics results instance statistics results.",
        ],
        5: [
            "Fig. 3. Pipeline of the interactive model. Given a radiance field trained on rice or wheat, the model first takes manual inputs or YOLO prompts as input. It then uses SAM2 to generate 2D masks for the image sequence. Based on the radiance field information, a mask inverse rendering process is performed iteratively, ultimately obtaining the 3D masks."
        ],
        6: ["Fig. 4. Illustration of residual handling and SSIM-Based fast prompt frame detection process."],
        9: [
            "Table 2. Comparison of mainstream segmentation methods on the MMR and MMW datasets.",
            "Fig. 5. (a) Radar chart of evaluation metrics across rice organs. (b) Radar chart of evaluation metrics across wheat organs.",
        ],
        10: [
            "Fig. 6. Comparison of target inference time, taking rice as an example. (a) SA3D single-target point cloud segmentation time. (b)-(c) IPENS segmentation time for different organs. (d)-(g) IPENS simultaneous segmentation time for 2 to 5 multi-target.",
            "Table 3. Time consumption in the IPENS workflow.",
            "Fig. 7. (a) Correlation between labeled and predicted grain voxel volume extracted by model. (b) Correlation between labeled and predicted panicle voxel volume extracted by model.",
        ],
        11: ["Table 4. Comparison of rice and wheat leaf surface area, length, and width (in cm2 and cm)."],
        12: [
            "Table 5. GPU memory consumption under varying numbers of targets.",
            "Fig. 8. Indoor high-throughput phenotyping chamber system and outdoor field data acquisition vehicle.",
        ],
    }
    if page_no in manual:
        candidates = [c for c in candidates if not c.startswith("Fig.")]
    for cap in manual.get(page_no, []):
        if cap not in candidates:
            candidates.append(cap)
    return candidates


def extract_figures_for_page(page: pdfplumber.page.Page, page_no: int) -> list[dict]:
    figures = []
    for img_no, img in enumerate(page.images, start=1):
        # Skip publisher logos on the first page.
        if page_no == 1:
            continue
        bbox = (
            max(0, img["x0"] - 2),
            max(0, img["top"] - 2),
            min(page.width, img["x1"] + 2),
            min(page.height, img["bottom"] + 2),
        )
        figures.append(
            {
                "page": page_no,
                "image_no": img_no,
                "bbox": [round(v, 2) for v in bbox],
            }
        )
    return figures


def translate_blocks(blocks: list[Block]) -> None:
    translatable = [b for b in blocks if b.type != "reference"]
    translations = batch_translate([b.original for b in translatable])
    translated_by_id = {
        block.id: polish_translation(block.original, zh)
        for block, zh in zip(translatable, translations)
    }
    for block in blocks:
        if block.type == "reference":
            block.translation = "参考文献条目保留原文，未做逐条翻译。"
        else:
            block.translation = translated_by_id[block.id]


def batch_translate(texts: list[str]) -> list[str]:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except Exception:
        return [fallback_translate(t) for t in texts]

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    model_name = "facebook/nllb-200-distilled-600M"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        tokenizer.src_lang = "eng_Latn"
        out: list[str] = []
        cache_path = OUT_DIR / "translation_cache.json"
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for idx, text in enumerate(texts, start=1):
            if text in MANUAL_TRANSLATIONS:
                out.append(MANUAL_TRANSLATIONS[text])
                continue
            if should_not_machine_translate(text):
                out.append(fallback_translate(text))
                continue
            if text in cache and not str(cache[text]).startswith("【机器初译待精修】"):
                out.append(cache[text])
                continue
            if idx == 1 or idx % 25 == 0:
                print(f"Translating block {idx}/{len(texts)}", flush=True)
            chunks = chunk_text(text, limit=850)
            zh_chunks = []
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
                    max_length=512,
                    num_beams=1,
                    do_sample=False,
                )
                zh_chunks.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
            translated = " ".join(zh_chunks)
            cache[text] = translated
            out.append(translated)
            if idx % 25 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception as exc:
        return [fallback_translate(t) + f" [机器翻译不可用：{type(exc).__name__}]" for t in texts]


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


def should_not_machine_translate(text: str) -> bool:
    if SECTION_RE.match(text):
        return True
    if text.startswith("Algorithm "):
        return True
    if len(re.findall(r"[∫∑{}<>]|\\bInput:|\\bOutput:", text)) > 2:
        return True
    if text.startswith("[") and re.match(r"^\[\d+\]", text):
        return True
    return False


def fallback_translate(text: str) -> str:
    if text in MANUAL_TRANSLATIONS:
        return MANUAL_TRANSLATIONS[text]
    if SECTION_RE.match(text):
        return MANUAL_TRANSLATIONS.get(text, f"【标题暂译】{text}")
    if text.startswith("Algorithm "):
        return text.replace("Algorithm", "算法")
    if text.startswith("Table "):
        return MANUAL_TRANSLATIONS.get(text, f"表格说明：{text}")
    if text.startswith("Fig. "):
        return MANUAL_TRANSLATIONS.get(text, f"图注：{text}")
    return f"【机器初译待精修】{text}"


def polish_translation(original: str, zh: str) -> str:
    zh = normalize_text(zh)
    for old, new in TERM_FIXES.items():
        zh = zh.replace(old, new)
    replacements = {
        "IPENS": "IPENS",
        "NeRF": "NeRF",
        "SAM2": "SAM2",
        "YOLOv11": "YOLOv11",
        "COLMAP": "COLMAP",
        "mIoU": "mIoU",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "SSIM": "SSIM",
        "PSNR": "PSNR",
        "LPIPS": "LPIPS",
        "FPS": "FPS",
    }
    for term in replacements:
        # Preserve exact Latin technical token if NLLB spaces it oddly.
        zh = re.sub(r"\s*".join(map(re.escape, term)), term, zh, flags=re.I)
    if original.startswith("•") and not zh.startswith("•"):
        zh = "• " + zh
    if original.startswith("[") and re.match(r"^\[\d+\]", original):
        zh = "参考文献条目保留原文，未翻译。"
    return zh


def crop_assets(figures: list[dict]) -> list[dict]:
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    enriched: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        fig_idx = 1
        for fig in figures:
            page = pdf.pages[fig["page"] - 1]
            bbox = tuple(fig["bbox"])
            cropped = page.crop(bbox)
            im = cropped.to_image(resolution=220).original.convert("RGB")
            filename = f"fig{fig_idx}.png"
            path = ASSET_DIR / filename
            im.save(path, optimize=True)
            fig.update(
                {
                    "id": f"F{fig_idx:03d}",
                    "file": f"assets/{filename}",
                    "width": im.width,
                    "height": im.height,
                    "confidence": "high",
                }
            )
            enriched.append(fig)
            fig_idx += 1
    return enriched


def pair_figures_with_captions(figures: list[dict], blocks: list[Block]) -> list[dict]:
    fig_captions = [b for b in blocks if b.type == "caption" and b.original.startswith("Fig.")]
    page_seen: dict[int, int] = {}
    for i, fig in enumerate(figures):
        caption = None
        same_page = [c for c in fig_captions if c.page == fig["page"]]
        if same_page:
            page_index = page_seen.get(fig["page"], 0)
            caption = same_page[min(page_index, len(same_page) - 1)]
            page_seen[fig["page"]] = page_index + 1
        elif i < len(fig_captions):
            caption = fig_captions[i]
        if caption:
            fig["caption_id"] = caption.id
            fig["caption_original"] = caption.original
            fig["caption_translation"] = caption.translation
            fig["label"] = re.match(r"Fig\.\s*\d+", caption.original).group(0) if re.match(r"Fig\.\s*\d+", caption.original) else fig["id"]
            fig["placed_near"] = find_first_mention(caption.original, blocks)
        else:
            fig["caption_id"] = None
            fig["caption_original"] = "Caption not confidently extracted."
            fig["caption_translation"] = "图注未能可靠抽取。"
            fig["label"] = fig["id"]
            fig["placed_near"] = None
    return figures


def find_first_mention(caption: str, blocks: list[Block]) -> str | None:
    m = re.match(r"(Fig\.\s*\d+)", caption)
    if not m:
        return None
    label = m.group(1)
    for block in blocks:
        if block.type == "body" and label in block.original:
            return f"p.{block.page} {block.id}"
    return None


def build_markdown(blocks: list[Block], figures: list[dict], metadata: dict) -> str:
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
            # Place at the first block from the figure page if no first mention was found.
            page_blocks = by_page.get(fig["page"], [])
            if page_blocks:
                figure_after.setdefault(page_blocks[0].id, []).append(fig)

    md: list[str] = []
    md.append("---")
    md.append(f"title: {json.dumps(TITLE, ensure_ascii=False)}")
    md.append(f"authors: {json.dumps(AUTHORS, ensure_ascii=False)}")
    md.append(f"journal: {json.dumps(JOURNAL, ensure_ascii=False)}")
    md.append(f"doi: {DOI}")
    md.append(f"source_pdf: {PDF_PATH}")
    md.append(f"generated: {date.today().isoformat()}")
    md.append("reader_type: bilingual_source_grounded_markdown")
    md.append("---\n")
    md.append(f"# {TITLE}\n")
    md.append(f"**中文题名：** {MANUAL_TRANSLATIONS[TITLE]}\n")
    md.append(f"**来源：** {JOURNAL}; DOI: {DOI}\n")
    md.append("**说明：** 本文件为全文中英对照阅读稿。正文翻译为机器初译并经过领域术语规则校正；双栏公式、表格与参考文献的低置信区域已在 `translation_notes.md` 标注。\n")
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
        ("Neural Radiance Fields (NeRF)", "神经辐射场（NeRF）"),
        ("Segment Anything Model 2 (SAM2)", "Segment Anything Model 2（SAM2）"),
        ("3D instance segmentation", "三维实例分割"),
        ("point cloud extraction", "点云提取"),
        ("mask inverse rendering", "掩膜逆渲染"),
        ("multi-target collaborative optimization", "多目标协同优化"),
        ("voxel volume", "体素体积"),
        ("leaf surface area", "叶片表面积"),
        ("Structural Similarity Index Measure (SSIM)", "结构相似性指数（SSIM）"),
    ]
    for en, zh in terms:
        md.append(f"| {en} | {zh} |")
    md.append("")
    md.append("## 全文中英对照\n")
    for page in sorted(by_page):
        md.append(f"\n## Page {page}\n")
        for block in sorted(by_page[page], key=lambda b: b.order):
            anchor = block.id
            if block.type == "caption":
                # Captions are displayed in figure blocks or table/caption notes.
                if block.original.startswith("Fig."):
                    continue
                md.append(f'<a id="{anchor}"></a>')
                md.append(f"### {block.original.split('.')[0] if '.' in block.original else block.original}")
                md.append(f"**Source:** p.{block.page} {block.id}  \n**Type:** {block.type}  \n**Confidence:** {block.confidence}\n")
                md.append(f"**Original:** {block.original}\n")
                md.append(f"**中文:** {block.translation}\n")
                continue
            heading_prefix = "### " if block.type == "section" else ""
            md.append(f'<a id="{anchor}"></a>')
            if heading_prefix:
                md.append(f"{heading_prefix}{block.original}")
            md.append(f"**Source:** p.{block.page} {block.id}  \n**Type:** {block.type}  \n**Confidence:** {block.confidence}\n")
            md.append(f"**Original:** {block.original}\n")
            md.append(f"**中文:** {block.translation}\n")
            for fig in figure_after.get(block.id, []):
                md.extend(render_figure_block(fig))
    md.append("\n## 阅读提示\n")
    md.append("- IPENS 的核心思想是把 SAM2 在多视角图像序列上得到的 2D 掩膜，通过 NeRF 的辐射场表示和可微渲染约束提升到 3D 空间。")
    md.append("- 与完全监督 3D 分割相比，IPENS 的价值在于减少精细 3D 标注依赖；与单目标交互式方法相比，它强调一次交互中的多目标协同提取。")
    md.append("- 结果部分要重点对照 Table 2、Fig. 5、Fig. 7 和 Table 4：水稻籽粒分割相对困难，小麦穗部与叶片表型估计表现更强。")
    md.append("- 局限性主要来自显存随同步目标数增加而快速增长，以及野外光照/风扰对 NeRF 重建质量的影响。")
    return "\n".join(md) + "\n"


def render_figure_block(fig: dict) -> list[str]:
    label = fig.get("label") or fig["id"]
    title = fig.get("caption_translation", label)
    title = re.sub(r"^图\s*\d+[.．]\s*", "", title)
    lines = [
        f'<a id="{fig["id"]}"></a>',
        f"### {label}. {title[:90]}",
        f"**Placed near:** {fig.get('placed_near') or 'p.' + str(fig['page'])}  ",
        f"**Source:** p.{fig['page']} {fig.get('caption_id') or fig['id']}  ",
        f"**Crop confidence:** {fig.get('confidence', 'medium')}\n",
        f"![{label}]({fig['file']})\n",
        f"**Original caption:** {fig.get('caption_original', '')}\n",
        f"**中文图注:** {fig.get('caption_translation', '')}\n",
        f"**Reading note:** 重点查看该图如何支撑相邻正文中的方法流程、实验比较或平台应用描述。\n",
    ]
    return lines


def write_source_map(blocks: list[Block], figures: list[dict], metadata: dict) -> None:
    data = {
        "source_pdf": str(PDF_PATH),
        "metadata": metadata,
        "blocks": [asdict(b) for b in blocks],
        "figures": figures,
    }
    (OUT_DIR / "source_map.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes(blocks: list[Block], figures: list[dict]) -> None:
    low = [b for b in blocks if b.confidence != "high"]
    notes = []
    notes.append("# Translation and Extraction Notes\n")
    notes.append(f"- Source PDF: `{PDF_PATH}`")
    notes.append("- PDF type: selectable-text PDF with two-column Elsevier layout.")
    notes.append("- Paper type: methods / algorithm paper for interactive unsupervised 3D plant phenotyping.")
    notes.append("- Translation method: NLLB machine translation with domain-term post-processing; section titles and recurring captions are manually stabilized.")
    notes.append("- Draft-mode caveat: equations, algorithms, dense tables, and references were preserved with lower translation confidence where layout extraction was noisy.")
    notes.append(f"- Text/caption blocks: {len(blocks)}; figure crops: {len(figures)}.")
    notes.append("\n## Low/Medium Confidence Blocks\n")
    for block in low[:200]:
        notes.append(f"- `{block.id}` p.{block.page} ({block.type}, {block.confidence}): {block.original[:180]}")
    if not low:
        notes.append("- None.")
    notes.append("\n## Figure Crop Notes\n")
    for fig in figures:
        notes.append(f"- `{fig['id']}` p.{fig['page']} `{fig['file']}` bbox={fig['bbox']} caption={fig.get('caption_id')}")
    notes.append("\n## Known Limitations\n")
    notes.append("- Tables are represented as caption/source blocks and nearby prose; exact cell-level table recreation was not guaranteed for all tables because the PDF interleaves table columns with body text.")
    notes.append("- References are retained as source blocks when extractable but are not fully translated, to avoid low-value noisy translation.")
    notes.append("- Formula-heavy paragraphs may preserve original symbols with only partial Chinese explanation.")
    (OUT_DIR / "translation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def verify_outputs(figures: list[dict]) -> None:
    paper = OUT_DIR / "paper.md"
    source_map = OUT_DIR / "source_map.json"
    notes = OUT_DIR / "translation_notes.md"
    text = paper.read_text(encoding="utf-8")
    assert "**Original:**" in text and "**中文:**" in text
    for fig in figures:
        assert (OUT_DIR / fig["file"]).exists(), fig["file"]
        assert fig["id"] in text, fig["id"]
    json.loads(source_map.read_text(encoding="utf-8"))
    assert notes.exists()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks, figures, metadata = extract_blocks()
    translate_blocks(blocks)
    figures = crop_assets(figures)
    figures = pair_figures_with_captions(figures, blocks)
    (OUT_DIR / "paper.md").write_text(build_markdown(blocks, figures, metadata), encoding="utf-8")
    write_source_map(blocks, figures, metadata)
    write_notes(blocks, figures)
    verify_outputs(figures)
    print(f"Wrote {OUT_DIR}")
    print(f"Blocks: {len(blocks)}; figures: {len(figures)}")


if __name__ == "__main__":
    main()
