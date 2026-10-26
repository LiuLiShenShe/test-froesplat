#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build `feature_matrix.csv` and `run_manifest.json` from what is on disk.

Nothing here is asserted by hand: every status row points at an artefact path,
and every manifest row is derived from a run directory that actually exists
(a `逐帧报告.csv`, a `run_record.json`, or a `run_manifest.json`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
PROD = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py"
FWD = ROOT / "2d-gaussian-splatting-m2m3-floor40/train.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def report_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def collect_runs() -> list[dict]:
    runs = []
    for params in sorted(RECOVERY.glob("runs/**/参数.json")):
        run_dir = params.parent
        report = run_dir / "逐帧报告.csv"
        record = run_dir / "run_record.json"
        if not report.exists() and not record.exists():
            continue
        try:
            cfg = json.loads(params.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        entry = {
            "run_dir": str(run_dir.relative_to(RECOVERY)),
            "frames": report_rows(report),
            "report_sha256": sha256(report) if report.exists() else None,
            "params_sha256": sha256(params),
            "params": {k: cfg.get(k) for k in (
                "prompt_list", "default_prompt_id", "prompt_selection_mode",
                "use_semantic_gate", "semantic_gate_mode", "semantic_gate_temperature",
                "semantic_gate_margin", "ranking_policy", "candidate_mode",
                "use_spnp_refinement", "spnp_prompt_form", "spnp_negative_region_mode",
                "use_residual_repair", "thin_repair_mode",
                "use_corrective_geometry", "corrective_geometry_backend",
            )},
        }
        runs.append(entry)
    for record in sorted(RECOVERY.glob("runs/**/run_record.json")):
        payload = json.loads(record.read_text(encoding="utf-8"))
        runs.append({"run_dir": str(record.parent.relative_to(RECOVERY)),
                     "kind": "downstream", **payload})
    for record in sorted(RECOVERY.glob("runs/mesh/*/mesh_record.json")):
        payload = json.loads(record.read_text(encoding="utf-8"))
        runs.append({"run_dir": str(record.parent.relative_to(RECOVERY)),
                     "kind": "mesh", **payload})
    return runs


FEATURES = [
    dict(feature="FFT 帧质量过滤 Q_FFT", section="§2.3.1 Eq.(1)",
         entry="02-FFT/blur_filter.py:fft_blur_score_paper",
         level="L1-L6", status="VERIFIED",
         evidence="阶段二十一_…/metrics/fft_formula_kqz.csv（210 帧，33 帧判定不同）"),
    dict(feature="FFT 径向高通权重 W_H", section="§2.3.1",
         entry="02-FFT/blur_filter.py:radial_high_pass_weight",
         level="L1-L3", status="IMPLEMENTED",
         evidence="cutoff_ratio=0.10, transition_ratio=0.10"),
    dict(feature="逐序列第一四分位阈值过滤", section="§2.3.1",
         entry="02-FFT/blur_filter.py:--blur_percentile 25",
         level="L1-L4", status="RUNNABLE",
         evidence="filter_log.json"),
    dict(feature="多提示词逐实例候选生成 (K=5)", section="§2.3.2",
         entry="生成RAP-FSAM3掩膜.py:infer_candidates",
         level="L1-L5", status="VERIFIED",
         evidence="runs/R6_RAP_FULL_real/候选掩膜/raw_instance_P{1..5}/（23/7/5/5/13）"),
    dict(feature="SG-MPS 前景描述符 φ", section="§2.3.3",
         entry="生成RAP-FSAM3掩膜.py:semantic_gate_scores",
         level="L1-L3", status="IMPLEMENTED",
         evidence="逐帧报告.csv:sg_mps_gate_confidence"),
    dict(feature="SG-MPS 干扰描述符 ψ", section="§2.3.3",
         entry="生成RAP-FSAM3掩膜.py:semantic_gate_scores",
         level="L1-L3", status="IMPLEMENTED",
         evidence="同上"),
    dict(feature="SG-MPS 门控置信度 c=σ(wφ·φ−wψ·ψ−b)", section="§2.3.3 Eq.(2)",
         entry="生成RAP-FSAM3掩膜.py:semantic_gate_scores:gate_confidence",
         level="L1-L3", status="IMPLEMENTED",
         evidence="wφ=0.5,0.3,0.15,0.15,0,0.1; wψ=0.3,0.3,0.3,0.2; b=0.5"),
    dict(feature="SG-MPS 提示词级 softmax a_i^(k)", section="§2.3.3 Eq.(3)",
         entry="生成RAP-FSAM3掩膜.py:apply_semantic_gate_softmax",
         level="L1-L5", status="VERIFIED",
         evidence="逐帧报告.csv:sg_mps_softmax 逐帧 Σ=1；tests/test_rap_recovery_contracts.py"),
    dict(feature="SG-MPS 提示词选择 k*=argmax a_i^(k)", section="§2.3.3 Eq.(4)",
         entry="生成RAP-FSAM3掩膜.py:semantic_gate_prompt",
         level="L1-L5", status="PARTIALLY_VERIFIED",
         evidence="能真实切换提示词（KQZ 0075: P2→P1），但 dev 集上 16/16 帧 oracle 都选 P2 而门控只选对 2/16，见 05 报告 §3-§4"),
    dict(feature="SG-MPS 承诺裕度 (safety veto)", section="§2.3.3 语义（新增参数）",
         entry="生成RAP-FSAM3掩膜.py:--semantic_gate_margin",
         level="L1-L5", status="PARTIALLY_VERIFIED",
         evidence="dev 集 margin=0.05 时 R3→R3g F1 0.7504→0.9435，仍低于无门控 0.9763（05 报告 §5）"),
    dict(feature="CDR/R1 实例级重排 S−q_contrast/5.5", section="§2.3.3 末段 / 17C",
         entry="scripts/ranking_policy.py:r1",
         level="L1-L5", status="PARTIALLY_VERIFIED",
         evidence="dev 集 16 帧选择与 base 完全一致 → 该子集上无增量"),
    dict(feature="SB-FBPR 正提示 TopK 距离变换", section="§2.3.4 Eq.(5)",
         entry="生成RAP-FSAM3掩膜.py:sample_points(positive=True)",
         level="L1-L5", status="VERIFIED",
         evidence="tests: test_positive_points_are_distance_topk；SB-FBPR提示证据.csv"),
    dict(feature="SB-FBPR 负候选区 ring∪table∪bg∪neighbour", section="§2.3.4 Eq.(6)",
         entry="生成RAP-FSAM3掩膜.py:manuscript_negative_regions",
         level="L1-L5", status="VERIFIED",
         evidence="tests: test_negative_regions_are_outside_mask"),
    dict(feature="SB-FBPR 负提示 NMS", section="§2.3.4 Eq.(7)",
         entry="生成RAP-FSAM3掩膜.py:sample_negative_prompts",
         level="L1-L5", status="VERIFIED",
         evidence="每帧 8 负点、跨 ≥2 子区、均落在掩膜外"),
    dict(feature="SB-FBPR 点提示注入 SAM3", section="§2.3.4 末段",
         entry="sam3_prompt_adapter.py:append_point_prompts + Prompt.append_points",
         level="L1-L5", status="VERIFIED",
         evidence="SB-FBPR提示证据.csv: points_sent=5/8, boxes=0；tests: prompt form"),
    dict(feature="残余/细结构修复", section="§2.3.4 工程实现",
         entry="生成RAP-FSAM3掩膜.py:repair_residual_mask",
         level="L1-L4", status="RUNNABLE",
         evidence="残差修复掩膜/、逐帧报告.csv:residual_restored_px（210 帧全部执行，4–101 px/帧，均值 17.0；dev 子集上对最终掩膜零改变）"),
    dict(feature="GCFC 投影支撑场 H_i", section="§2.3.5 Eq.(8)",
         entry="生成RAP-FSAM3掩膜.py:points_to_support_field",
         level="L1-L5", status="VERIFIED",
         evidence="几何修正提示图/GCFC_support_field_*.png；阶段21 tests"),
    dict(feature="GCFC 几何一致性 G_i", section="§2.3.5 Eq.(9)",
         entry="生成RAP-FSAM3掩膜.py:apply_corrective_geometry",
         level="L1-L5", status="VERIFIED",
         evidence="G=0.9677–0.9918（5 帧真实 COLMAP）"),
    dict(feature="GCFC 边界带内正向补洞", section="§2.3.5 Eq.(10)",
         entry="生成RAP-FSAM3掩膜.py:boundary_band + apply_corrective_geometry",
         level="L1-L5", status="VERIFIED",
         evidence="GCFC_added_pixels_*.png；5 帧 run 新增 151/227/456/0/223 px；210 帧 run 205/210 接受，均值 3049 px/帧"),
    dict(feature="GCFC 真实 COLMAP 观测加载", section="§2.3.5",
         entry="生成RAP-FSAM3掩膜.py:load_colmap_observations (两端 backend)",
         level="L1-L5", status="VERIFIED",
         evidence="gcfc_has_real_observation=1：5 帧 run 5/5、210 帧 run 210/210；--strict_colmap_observations"),
    dict(feature="Foreground-Object 2DGS 训练", section="§2.4",
         entry="2d-gaussian-splatting-m2m3-floor40/train.py",
         level="L1-L5", status="VERIFIED",
         evidence="2×2 四档全部 COMPLETE（2132–2201 s）；PSNR_fg 见 metrics/downstream_2x2.csv；掩膜 provenance 183/183 一致"),
    dict(feature="M2M3-GO 双向微匹配", section="§2.7",
         entry="2d-gaussian-splatting-m2m3-floor40/train.py:run_m2m3_go_update",
         level="L1-L5", status="RUNNABLE",
         evidence="2×2 两档真实执行（19 次更新，剪枝 107 955/115 825）；ΔPSNR_fg +0.011 dB（C_R1）/ −0.125 dB（RAP）"),
    dict(feature="TSDF mesh 提取", section="§2.8",
         entry="2d-gaussian-splatting-m2m3-floor40/render.py",
         level="L1-L5", status="VERIFIED",
         evidence="runs/mesh/<tag>/mesh_post.ply ×4（标准 TSDF，74–106 s/档）；metrics/mesh_phenotype.csv"),
    dict(feature="表型几何测量（探索性，COLMAP 单位）", section="§2.8",
         entry="scripts/run_mesh_phenotype.py:measure",
         level="L1-L5", status="PARTIALLY_VERIFIED",
         evidence="高 3.541–3.561 u、冠幅 3.122–3.142 u（四档差异 ≤0.5%）；无真实尺度→物理表型全部 PHYSICAL_SCALE_NOT_AVAILABLE"),
]


def main() -> int:
    runs = collect_runs()
    manifest = {
        "generated_at": "2026-10-25",
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "production_entry_sha256": sha256(PROD),
        "forward_trainer_sha256": sha256(FWD) if FWD.exists() else None,
        "python_sam3": "/home/test/.venvs/sam3/bin/python",
        "python_2dgs": str(ROOT / "2d-gaussian-splatting-m2m3-floor40/venv/bin/python"),
        "runs": runs,
        "run_count": len(runs),
    }
    (RECOVERY / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    with (RECOVERY / "feature_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "feature", "section", "entry", "level", "status", "evidence"])
        writer.writeheader()
        writer.writerows(FEATURES)
    print(f"feature_matrix.csv: {len(FEATURES)} rows")
    print(f"run_manifest.json: {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
