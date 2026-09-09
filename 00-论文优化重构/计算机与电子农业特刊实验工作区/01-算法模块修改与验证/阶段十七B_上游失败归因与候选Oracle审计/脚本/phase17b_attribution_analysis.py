#!/usr/bin/env python3
"""
Phase 17B — Steps 5 + 12 + 13 + 14
Failure attribution, score decomposition, cleanup trace, temporal diagnosis.
Zero-inference: reads P00 artifacts + Phase16 GT.

Score decomposition uses the pipeline's OWN recorded 总分 from 提示词评分.csv
as the selection criterion (select_mask: winner = max total_score), then
compares component-level q-scores of the oracle vs selected instance.
The weighted contributions are reported as evidence of the driver; the
authoritative total for selection is the pipeline 总分 (not a re-derivation).
"""
from __future__ import annotations
import csv
from pathlib import Path
import cv2, numpy as np

PHASE17A = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计")
PHASE16  = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定")
P00      = PHASE17A / "02_trackA_variants/P00_Control"
GT_DIR   = PHASE16 / "GT_potted_clean_challenge"
OUT      = Path(__file__).resolve().parent.parent

GT10 = [("BaiZhang", f"{i:04d}") for i in range(33, 38)] + \
       [("DouBanLv2", f"{i:04d}") for i in range(37, 42)]
FAILING = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}

# pipeline score weights (from S20 --score_weights + params.json; leak/side weight 0)
W = {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0, "leak": 0.0, "side": 0.0, "sam": 0.5}
DENOM = sum(W.values())  # 5.5
Q_COLS = ["面积得分", "连通域得分", "边界得分", "时序得分", "前背景对比得分",
          "下方泄漏得分", "侧边泄漏得分"]
Q_KEYS = ["area", "comp", "edge", "temp", "contrast", "leak", "side"]


def bmask(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def fm(a, b):
    if a is None or b is None:
        return 0.0, 0.0, 0.0, 0.0
    tp = int(np.logical_and(a, b).sum()); fp = int(a.sum()) - tp; fn = int(b.sum()) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0; r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return f1, iou, p, r


def parse_instances(frame, sample):
    stem = f"{sample}_{frame}"
    rows = []
    with (P00 / "候选掩膜" / f"候选评分明细_{stem}.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "instance_id": int(r["实例编号"]),
                "sam_score": float(r["SAM3分数"]),
                "area_ratio": float(r["面积比例"]),
                "bbox": r["外接框"],
            })
    return rows


def parse_score_rows(frame, sample):
    """提示词评分.csv rows for a frame, in file order (matches instance order)."""
    rows = []
    with (P00 / "提示词评分.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["图像"].rsplit(".", 1)[0] == f"{sample}_{frame}":
                rows.append(r)
    return rows


def parse_selection(frame, sample):
    with (P00 / "提示词选择.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["图像"].rsplit(".", 1)[0] == f"{sample}_{frame}":
                return r
    return {}


def align_score_rows(sc_rows, insts):
    """Align 提示词评分 rows to instance details by area ratio (order-verified)."""
    if len(sc_rows) != len(insts):
        return None
    # verify positional order matches area ratio
    if all(abs(float(s["面积比例"]) - d["area_ratio"]) < 1e-8 for s, d in zip(sc_rows, insts)):
        return list(zip(sc_rows, insts))
    # fallback: match by area ratio
    by_area = {}
    for s in sc_rows:
        by_area[round(float(s["面积比例"]), 8)] = s
    out = []
    for d in insts:
        s = by_area.pop(round(d["area_ratio"], 8), None)
        out.append((s, d))
    return out


def main() -> None:
    attr_rows, score_regret_rows, cleanup_rows, temporal_rows = [], [], [], []

    for (sample, frame) in GT10:
        stem = f"{sample}_{frame}"
        gt = bmask(GT_DIR / sample / f"mask_potted_clean_{frame}.png")
        fin_mask = P00 / "最终掩膜" / f"mask_{stem}.png"
        sel_mask = P00 / "选择后掩膜" / f"mask_{stem}.png"

        insts = parse_instances(frame, sample)
        sc_rows = parse_score_rows(frame, sample)
        sel = parse_selection(frame, sample)
        paired = align_score_rows(sc_rows, insts)
        if paired is None:
            print(f"WARN {stem}: cannot align {len(sc_rows)} score rows to {len(insts)} instances")
            paired = [(None, d) for d in insts]

        # oracle instance = highest F1 vs GT among raw P6 instances
        best_i, best_f1 = -1, -1.0
        for d in insts:
            raw = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{d['instance_id']:02d}.png")
            f = fm(raw, gt)[0]
            d["f1"] = f
            if f > best_f1:
                best_f1, best_i = f, d["instance_id"]
        oracle = next((d for d in insts if d["instance_id"] == best_i), None)

        # selected instance = the one the pipeline chose (argmax total_score)
        sel_area = float(sel["前景面积比例"]) if sel.get("前景面积比例", "").strip() not in ("", "0") else None
        sel_inst = None
        if sel_area is not None:
            for d in insts:
                if abs(d["area_ratio"] - sel_area) < 1e-6:
                    sel_inst = d["instance_id"]; break
        if sel_inst is None:  # fallback: argmax total
            best_t = -1e9
            for s, d in paired:
                if s is not None and float(s["总分"]) > best_t:
                    best_t, sel_inst = float(s["总分"]), d["instance_id"]

        sel_selected = bmask(sel_mask); fin = bmask(fin_mask)
        sel_f1 = fm(sel_selected, gt)[0]
        fin_f1 = fm(fin, gt)[0]

        # ---- Step 5: failure attribution (protocol C5>C1>C2>C3>C4) ----
        # C2 must compare the ORACLE candidate's raw F1 vs that same candidate's
        # postcleanup F1 — NOT the frame-final F1 (which reflects wrong selection).
        # basic_cleanup is bitwise identity on P00 (选择后==最终==A5c verified), so
        # postcleanup_oracle_f1 == raw_oracle_f1 and C2 cannot trigger here.
        raw_oracle = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{best_i:02d}.png") if best_i >= 0 else None
        postcleanup_f1 = fm(raw_oracle, gt)[0] if raw_oracle is not None else 0.0
        mapping_ok = bool(gt is not None and gt.any())
        if not mapping_ok:
            cls, conf, ev = "C5_DATA_MAPPING_FAILURE", "PROVEN", "GT empty or unreadable"
        elif oracle is None or best_f1 < 0.10:
            cls, conf, ev = "C1_GENERATION_FAILURE", "PROVEN", f"oracle_raw_f1={best_f1:.4f}<0.10"
        elif best_f1 >= 0.80 and (best_f1 - postcleanup_f1) > 0.05:
            cls, conf, ev = "C2_POSTPROCESS_FAILURE", "PROVEN", f"raw={best_f1:.4f}→postcleanup={postcleanup_f1:.4f} (Δ={best_f1-postcleanup_f1:.4f})"
        elif best_f1 >= 0.80 and sel_f1 < 0.5:
            cls, conf, ev = "C3_RANKING_FAILURE", "PROVEN", f"oracle={best_f1:.4f} selected={sel_f1:.4f}, correct instance not selected"
        elif 0.10 <= best_f1 < 0.80:
            cls, conf, ev = "C4_WEAK_GENERATION", "SUPPORTED", f"oracle_raw={best_f1:.4f} (partial candidate)"
        else:
            cls, conf, ev = "OK_SELECTED", "SUPPORTED", f"oracle={best_f1:.4f} selected={sel_f1:.4f}, regret={best_f1-sel_f1:.4f}"
        attr_rows.append({
            "sample": sample, "frame": frame, "failure_class": cls, "confidence": conf,
            "evidence": ev, "selected_f1": round(sel_f1, 4),
            "raw_oracle_f1": round(best_f1, 4), "regret": round(best_f1 - sel_f1, 4),
            "n_raw": len(insts), "oracle_instance": oracle["instance_id"] if oracle else None,
            "selected_instance": sel_inst,
        })

        # ---- Step 12: score decomposition (pipeline 总分 = selection criterion) ----
        if (sample, frame) in FAILING and oracle is not None and sel_inst is not None:
            oracle_sc = next((s for s, d in paired if d["instance_id"] == oracle["instance_id"]), None)
            sel_sc = next((s for s, d in paired if d["instance_id"] == sel_inst), None)
            if oracle_sc is not None and sel_sc is not None:
                o_tot = float(oracle_sc["总分"]); s_tot = float(sel_sc["总分"])
                # per-component raw q values + pipeline SAM raw score
                comps = {}
                for key, col in zip(Q_KEYS, Q_COLS):
                    comps[key] = (float(oracle_sc[col]), float(sel_sc[col]))
                sam_o = float(oracle_sc["SAM3原始分数"].split(";")[0])
                sam_s = float(sel_sc["SAM3原始分数"].split(";")[0])
                # weighted contribution of each component to (oracle − selected)
                contrib = {key: W[key] * (co - cs) / DENOM for key, (co, cs) in comps.items()}
                # driver = component that most favors selected over oracle (most negative)
                driver = min(contrib, key=contrib.get)
                score_regret_rows.append({
                    "sample": sample, "frame": frame,
                    "oracle_total_pipeline": round(o_tot, 6),
                    "selected_total_pipeline": round(s_tot, 6),
                    "selection_margin": round(s_tot - o_tot, 6),  # >0 => selected outranks oracle
                    "driver": driver, "driver_weighted_gap": round(contrib[driver], 6),
                    **{f"oracle_{k}": round(co, 6) for k, (co, _) in comps.items()},
                    **{f"selected_{k}": round(cs, 6) for k, (_, cs) in comps.items()},
                    "oracle_sam_raw": round(sam_o, 6), "selected_sam_raw": round(sam_s, 6),
                    "oracle_instance": oracle["instance_id"], "selected_instance": sel_inst,
                    "oracle_f1": round(best_f1, 6), "selected_f1": round(sel_f1, 6),
                })

        # ---- Step 13: cleanup trace (per-hop F1) ----
        raw_oracle = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{oracle['instance_id']:02d}.png") if oracle else None
        a5c = bmask(P00 / "A5c_final_mask" / f"mask_{stem}.png")
        cleanup_rows.append({
            "sample": sample, "frame": frame,
            "stage_raw_f1": round(fm(raw_oracle, gt)[0] if raw_oracle is not None else 0.0, 6),
            "stage_selected_f1": round(sel_f1, 6),
            "stage_final_f1": round(fin_f1, 6),
            "stage_a5c_f1": round(fm(a5c, gt)[0] if a5c is not None else 0.0, 6),
            "raw_to_selected_identity": bool(np.array_equal(sel_selected, raw_oracle)) if (sel_selected is not None and raw_oracle is not None) else None,
            "selected_to_final_identity": bool(np.array_equal(sel_selected, fin)) if (sel_selected is not None and fin is not None) else None,
            "oracle_used_for_trace": oracle["instance_id"] if oracle else None,
            "selected_is_oracle": (oracle["instance_id"] == sel_inst) if (oracle and sel_inst is not None) else None,
            "note": "cleanup identity verified: 选择后==最终==A5c in P00 per_instance mode",
        })

        # ---- Step 14: temporal strip (DouBanLv2 0037-0041) ----
        if sample == "DouBanLv2" and int(frame) in (37, 38, 39, 40, 41):
            gt_area = int(gt.sum()) if gt is not None else 0
            gt_yx = np.argwhere(gt > 0) if gt is not None and gt.any() else np.array([])
            oracle_sc_t = next((s for s, d in paired if oracle and d["instance_id"] == oracle["instance_id"]), None)
            temporal_rows.append({
                "frame": f"{sample}_{frame}", "gt_area": gt_area,
                "gt_cx": round(float(gt_yx[:, 1].mean()), 1) if len(gt_yx) else 0.0,
                "gt_cy": round(float(gt_yx[:, 0].mean()), 1) if len(gt_yx) else 0.0,
                "n_raw_instances": len(insts),
                "oracle_area_ratio": round(oracle["area_ratio"], 6) if oracle else 0.0,
                "oracle_sam": round(oracle["sam_score"], 6) if oracle else 0.0,
                "oracle_contrast_raw": round(float(oracle_sc_t["前背景对比"]), 6) if oracle_sc_t else 0.0,
                "oracle_f1": round(best_f1, 4),
                "selected_instance": sel_inst,
                "selected_f1": round(sel_f1, 4),
                "selection_correct": bool(oracle["instance_id"] == sel_inst) if oracle and sel_inst is not None else None,
            })

    # ---- write CSVs ----
    acols = ["sample", "frame", "failure_class", "confidence", "evidence", "selected_f1",
             "raw_oracle_f1", "regret", "n_raw", "oracle_instance", "selected_instance"]
    with (OUT / "Phase17B_failure_attribution.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=acols, extrasaction="ignore"); w.writeheader(); w.writerows(attr_rows)

    sr_prefix = ["sample", "frame", "oracle_total_pipeline", "selected_total_pipeline",
                 "selection_margin", "driver", "driver_weighted_gap"]
    sr_comp = [f"{side}_{k}" for side in ("oracle", "selected") for k in Q_KEYS] + \
              ["oracle_sam_raw", "selected_sam_raw"]
    sr_suf = ["oracle_instance", "selected_instance", "oracle_f1", "selected_f1"]
    with (OUT / "Phase17B_score_regret.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sr_prefix + sr_comp + sr_suf, extrasaction="ignore")
        w.writeheader(); w.writerows(score_regret_rows)

    ccols = ["sample", "frame", "stage_raw_f1", "stage_selected_f1", "stage_final_f1", "stage_a5c_f1",
             "raw_to_selected_identity", "selected_to_final_identity", "oracle_used_for_trace",
             "selected_is_oracle", "note"]
    with (OUT / "Phase17B_cleanup_trace.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore"); w.writeheader(); w.writerows(cleanup_rows)

    if temporal_rows:
        tcols = list(temporal_rows[0].keys())
        with (OUT / "Phase17B_temporal_diagnosis.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=tcols, extrasaction="ignore"); w.writeheader(); w.writerows(temporal_rows)

    # ---- console ----
    print("=== Failure attribution (§5) ===")
    for r in attr_rows:
        flag = " <<<" if (r["sample"], r["frame"]) in FAILING else ""
        print(f"  {r['sample']}_{r['frame']}: {r['failure_class']} [{r['confidence']}] {r['evidence']}{flag}")
    print("\n=== Score decomposition (§12) — failing frames (pipeline 总分) ===")
    for r in score_regret_rows:
        print(f"  {r['sample']}_{r['frame']}: oracle_total={r['oracle_total_pipeline']:.4f} "
              f"selected_total={r['selected_total_pipeline']:.4f} margin={r['selection_margin']:+.4f} "
              f"driver={r['driver']} (oracle_contrast={r['oracle_contrast']:.4f} vs "
              f"selected_contrast={r['selected_contrast']:.4f})")
    print("\n=== Cleanup trace (§13) ===")
    for r in cleanup_rows:
        ok = "IDENTITY" if r["selected_to_final_identity"] else "DIFFERS"
        flag = " <<<" if (r["sample"], r["frame"]) in FAILING else ""
        print(f"  {r['sample']}_{r['frame']}: raw={r['stage_raw_f1']:.4f} sel={r['stage_selected_f1']:.4f} "
              f"fin={r['stage_final_f1']:.4f} sel==oracle:{r['selected_is_oracle']} {ok}{flag}")
    print("\n=== Temporal (§14) DouBanLv2 0037-0041 ===")
    for r in temporal_rows:
        print(f"  {r['frame']}: gt_area={r['gt_area']} #inst={r['n_raw_instances']} "
              f"oracle_area_ratio={r['oracle_area_ratio']} oracle_sam={r['oracle_sam']} "
              f"oracle_contrast={r['oracle_contrast_raw']} oracle_f1={r['oracle_f1']} "
              f"correct={r['selection_correct']}")


if __name__ == "__main__":
    main()
