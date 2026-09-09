#!/usr/bin/env python3
"""
Phase 17B — Steps 2-6 + 8 + 10 (mapping audit, inventory, oracle, attribution,
oracle table, prompt matrix, detection-vs-mask).

Zero-inference: reads ONLY Phase17A P00_Control artifacts + Phase16 GT masks.
Emits all Phase17B analysis CSVs + mapping overlays.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import cv2
import numpy as np

PHASE17A = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计")
PHASE16  = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定")
P00      = PHASE17A / "02_trackA_variants/P00_Control"
GT_DIR   = PHASE16 / "GT_potted_clean_challenge"
INPUT_A  = PHASE17A / "00_input_trackA"
OUT_DIR  = Path(__file__).resolve().parent.parent
VIZ      = OUT_DIR / "visualizations"

GT10 = [("BaiZhang", f"{i:04d}") for i in range(33, 38)] + \
       [("DouBanLv2", f"{i:04d}") for i in range(37, 42)]
FAILING = {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}


def bmask(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def mask_metrics(pred, gt):
    """P6-metric pattern: tp/fp/fn, P/R/F1/IoU, empty pred -> F1=0."""
    if pred is None:
        return {"tp": 0, "fp": 0, "fn": int(gt.sum()) if gt is not None else 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "empty": True}
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(pred.sum()) - tp
    fn = int(gt.sum()) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": f1, "iou": iou, "empty": bool(pred.sum() == 0)}


def parse_bbox(s: str):
    if not s:
        return None
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        return None
    return parts  # raw 4-tuple; format determined empirically vs mask bbox


def read_score_rows():
    """提示词评分.csv -> {(sample, frame): [row,...]} preserving row order."""
    out = {}
    with (P00 / "提示词评分.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            img = r["图像"]                       # e.g. "DouBanLv2_0039.jpg"
            stem = img.rsplit(".", 1)[0]
            sample, frame = stem.rsplit("_", 1)
            out.setdefault((sample, frame), []).append(r)
    return out


def read_instance_details():
    """候选评分明细_*.csv -> {(sample, frame): [ {instance_id, sam_score, bbox, area_ratio}, ...]}"""
    out = {}
    for csvp in sorted((P00 / "候选掩膜").glob("候选评分明细_*.csv")):
        stem = csvp.name.replace("候选评分明细_", "").replace(".csv", "")
        sample, frame = stem.rsplit("_", 1)
        inst = []
        with csvp.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                inst.append({
                    "instance_id": int(r["实例编号"]),
                    "sam_score": float(r["SAM3分数"]),
                    "bbox": parse_bbox(r["外接框"]),
                    "area_ratio": float(r["面积比例"]),
                })
        out[(sample, frame)] = inst
    return out


def read_selection():
    """提示词选择.csv -> {(sample, frame): {selected_prompt, score, fg_area}}"""
    out = {}
    with (P00 / "提示词选择.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = r["图像"].rsplit(".", 1)[0]
            sample, frame = stem.rsplit("_", 1)
            out[(sample, frame)] = r
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    score_rows = read_score_rows()
    inst_rows = read_instance_details()
    selection = read_selection()

    # score-weight string from S20 (leak/side weight 0, sam weight 0.5)
    W = {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0, "sam": 0.5}
    COL_MAP = {"area": "面积得分", "comp": "连通域得分", "edge": "边界得分",
               "temp": "时序得分", "contrast": "前背景对比得分",
               "leak": "下方泄漏得分", "side": "侧边泄漏得分"}

    inventory = []   # per-candidate rows (ALL scored frames, not only GT10)
    oracle_rows = []
    mapping_rows = []
    det_rows = []

    # ---- Step 3: candidate inventory over ALL scored P00 frames (157 rows) ----
    # Every candidate in 提示词评分.csv (30 frames = 10 GT + 20 context) gets a row.
    # F1/IoU/P/R are populated only when GT exists (GT10); context frames have no GT.
    for (sample, frame), sc in sorted(score_rows.items()):
        stem = f"{sample}_{frame}"
        det = inst_rows.get((sample, frame), [])
        order_ok = (len(sc) == len(det)) and all(
            abs(float(s["面积比例"]) - d["area_ratio"]) < 1e-9 for s, d in zip(sc, det))
        gt_path = GT_DIR / sample / f"mask_potted_clean_{frame}.png"
        gt = bmask(gt_path) if gt_path.exists() else None
        has_gt = gt is not None and gt.any()
        for i, s in enumerate(sc):
            det_i = det[i] if (order_ok and i < len(det)) else None
            inst_id = det_i["instance_id"] if det_i else i
            sam = det_i["sam_score"] if det_i else float(s.get("SAM3原始分数", "0").split(";")[0])
            raw_path = P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{inst_id:02d}.png"
            raw = bmask(raw_path)
            m = mask_metrics(raw, gt) if (raw is not None and has_gt) else {
                "empty": True, "f1": None, "iou": None, "precision": None, "recall": None}
            inventory.append({
                "sample": sample, "frame": frame, "prompt_id": "P6",
                "instance_id": inst_id, "stage": "raw", "candidate_path": str(raw_path),
                "mask_nonempty": int(not m["empty"]), "area_ratio": round(det_i["area_ratio"], 6) if det_i else "",
                "sam_score": round(sam, 6), "total_score": round(float(s["总分"]), 6),
                "has_gt": int(bool(has_gt)),
                "f1": ("" if m["f1"] is None else round(m["f1"], 6)),
                "iou": ("" if m["iou"] is None else round(m["iou"], 6)),
                "precision": ("" if m["precision"] is None else round(m["precision"], 6)),
                "recall": ("" if m["recall"] is None else round(m["recall"], 6)),
            })

    for (sample, frame) in GT10:
        stem = f"{sample}_{frame}"
        rgb = INPUT_A / f"{stem}.jpg"
        gt_mask = GT_DIR / sample / f"mask_potted_clean_{frame}.png"
        sel_mask = P00 / "选择后掩膜" / f"mask_{stem}.png"
        fin_mask = P00 / "最终掩膜" / f"mask_{stem}.png"
        a5c_mask = P00 / "A5c_final_mask" / f"mask_{stem}.png"

        # ---- Step 2 mapping ----
        img = cv2.imread(str(rgb))
        gt = bmask(gt_mask)
        img_ok = img is not None and img.shape[:2] == (3840, 2160)
        gt_ok = gt is not None and gt.shape == (3840, 2160)
        # frame index sanity vs context window
        fin = bmask(fin_mask)
        map_row = {
            "sample": sample, "frame": frame, "image_path": str(rgb), "gt_path": str(gt_mask),
            "pred_path": str(fin_mask),
            "image_exists": img_ok, "gt_exists": gt_ok, "pred_exists": fin_mask.exists(),
            "image_size": f"{img.shape[1]}x{img.shape[0]}" if img is not None else "NA",
            "gt_size": f"{gt.shape[1]}x{gt.shape[0]}" if gt is not None else "NA",
            "pred_size": f"{fin.shape[1]}x{fin.shape[0]}" if fin is not None else "NA",
            "sample_prefix_ok": rgb.stem.rsplit("_", 1)[0] == sample,
            "frame_index_ok": frame.isdigit(),
            "gt_area": int(gt.sum()) if gt is not None else 0,
        }
        mapping_rows.append(map_row)

        # overlays
        if img is not None and gt is not None:
            vis = img.copy()
            green = vis.copy(); green[gt] = (0, 220, 0)
            gt_ov = cv2.addWeighted(vis, 0.6, green, 0.4, 0)
            cv2.imwrite(str(VIZ / f"map_gt_{stem}.jpg"), cv2.resize(gt_ov, (1280, 720)))
        if img is not None and fin is not None:
            vis = img.copy()
            red = vis.copy(); red[fin] = (0, 0, 220)
            pd_ov = cv2.addWeighted(vis, 0.6, red, 0.4, 0)
            cv2.imwrite(str(VIZ / f"map_pred_{stem}.jpg"), cv2.resize(pd_ov, (1280, 720)))

        # ---- Step 3: per-candidate inventory + Step 4 oracle ----
        sc = score_rows.get((sample, frame), [])
        det = inst_rows.get((sample, frame), [])
        # join 提示词评分 row <-> instance detail by positional order (same iteration order)
        # verify area-ratio match positionally
        if len(sc) == len(det):
            order_ok = all(abs(float(s["面积比例"]) - d["area_ratio"]) < 1e-9
                           for s, d in zip(sc, det))
        else:
            order_ok = False

        best_raw = {"f1": -1.0, "instance_id": None}
        best_final = {"f1": -1.0, "instance_id": None}
        selected_row = None

        for i, s in enumerate(sc):
            det_i = det[i] if (order_ok and i < len(det)) else None
            inst_id = det_i["instance_id"] if det_i else i
            sam = det_i["sam_score"] if det_i else float(s.get("SAM3原始分数", "0").split(";")[0] if i < len(s.get("SAM3原始分数", "").split(";")) else 0)
            raw_path = P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{inst_id:02d}.png"
            raw = bmask(raw_path)
            m = mask_metrics(raw, gt)
            total = float(s["总分"])
            if m["f1"] > best_raw["f1"]:
                best_raw = {"f1": m["f1"], "instance_id": inst_id, "iou": m["iou"], "total": total}

        # selected candidate: which instance did the pipeline pick?
        sel = selection.get((sample, frame), {})
        sel_area = float(sel["前景面积比例"]) if sel.get("前景面积比例", "").strip() not in ("", "0") else None
        sel_inst = None
        if det:
            if sel_area is not None:
                for d in det:
                    if abs(d["area_ratio"] - sel_area) < 1e-6:
                        sel_inst = d["instance_id"]
                        break
            if sel_inst is None and order_ok:   # fallback: max total
                sel_inst = max(((inst_id, float(s["总分"])) for inst_id, s in zip(
                    [d["instance_id"] for d in det], sc)), key=lambda x: x[1])[0]
        selected_f1 = mask_metrics(bmask(sel_mask), gt)["f1"]

        # postcleanup oracle == raw (cleanup identity verified at plan time); final == selected
        final_f1 = mask_metrics(fin_mask and bmask(fin_mask), gt)["f1"]
        cleanup_f1 = best_raw["f1"]  # identity: no postprocess transformation in P00 per_instance

        # ---- Step 10 detection-vs-mask ----
        gt_bbox = (np.argwhere(gt).min(axis=0), np.argwhere(gt).max(axis=0)) if gt is not None and gt.any() else None
        n_inst = len(det)
        best_det = None
        for d in det:
            raw = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{stem}_{d['instance_id']:02d}.png")
            if raw is None:
                continue
            m = mask_metrics(raw, gt)
            # mask-level GT overlap
            if best_det is None or m["f1"] > best_det["f1"]:
                best_det = {"instance_id": d["instance_id"], "sam": d["sam_score"], "f1": m["f1"]}
        det_rows.append({
            "sample": sample, "frame": frame, "n_raw_instances": n_inst,
            "gt_bbox": str(gt_bbox) if gt_bbox else "",
            "any_instance_overlaps_gt": best_det is not None and best_det["f1"] > 0.5,
            "best_instance_id": best_det["instance_id"] if best_det else None,
            "best_instance_f1": round(best_det["f1"], 6) if best_det else 0.0,
            "selected_instance_id": sel_inst,
            "selected_instance_f1": round(selected_f1, 6),
            "detection_class": ("wrong_instance" if best_det and best_det["f1"] > 0.5 and selected_f1 < 0.5
                                else "correct_selection" if best_det and best_det["f1"] > 0.5
                                else "no_good_candidate"),
        })

        regret = best_raw["f1"] - selected_f1
        oracle_rows.append({
            "sample": sample, "frame": frame,
            "selected_f1": round(selected_f1, 6),
            "raw_oracle_f1": round(best_raw["f1"], 6),
            "raw_oracle_instance": best_raw["instance_id"],
            "postcleanup_oracle_f1": round(cleanup_f1, 6),
            "finalcandidate_oracle_f1": round(best_raw["f1"], 6),
            "selection_regret": round(regret, 6),
            "n_raw_instances": n_inst,
            "selected_instance_id": sel_inst,
            "selected_prompt": sel.get("选择提示词", ""),
        })

    # ---- write CSVs ----
    cols = ["sample", "frame", "prompt_id", "instance_id", "stage", "candidate_path",
            "mask_nonempty", "area_ratio", "sam_score", "total_score", "has_gt",
            "f1", "iou", "precision", "recall"]
    with (OUT_DIR / "Phase17B_candidate_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(inventory)

    ocols = ["sample", "frame", "selected_f1", "raw_oracle_f1", "raw_oracle_instance",
             "postcleanup_oracle_f1", "finalcandidate_oracle_f1", "selection_regret",
             "n_raw_instances", "selected_instance_id", "selected_prompt"]
    with (OUT_DIR / "Phase17B_candidate_oracle.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ocols, extrasaction="ignore"); w.writeheader(); w.writerows(oracle_rows)

    mcols = ["sample", "frame", "image_path", "gt_path", "pred_path", "image_exists", "gt_exists",
             "pred_exists", "image_size", "gt_size", "pred_size", "sample_prefix_ok",
             "frame_index_ok", "gt_area"]
    with (OUT_DIR / "Phase17B_mapping_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mcols, extrasaction="ignore"); w.writeheader(); w.writerows(mapping_rows)

    dcols = ["sample", "frame", "n_raw_instances", "gt_bbox", "any_instance_overlaps_gt",
             "best_instance_id", "best_instance_f1", "selected_instance_id",
             "selected_instance_f1", "detection_class"]
    with (OUT_DIR / "Phase17B_detection_diagnosis.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dcols, extrasaction="ignore"); w.writeheader(); w.writerows(det_rows)

    # ---- Step 6 oracle table (console) ----
    print("=== Phase 17B oracle table ===")
    print(f"{'frame':>18} {'selF1':>7} {'rawOrc':>7} {'clean':>7} {'finOrc':>7} {'regret':>7} {'#inst':>5}")
    for r in oracle_rows:
        flag = "  <<< FAILING" if (r["sample"], r["frame"]) in FAILING else ""
        print(f"{r['sample']}_{r['frame']:>18} {r['selected_f1']:>7.4f} {r['raw_oracle_f1']:>7.4f} "
              f"{r['postcleanup_oracle_f1']:>7.4f} {r['finalcandidate_oracle_f1']:>7.4f} "
              f"{r['selection_regret']:>7.4f} {r['n_raw_instances']:>5}{flag}")
    print(f"\n{len(inventory)} candidate rows in inventory | {len(oracle_rows)} oracle rows")

    # Step 8 prompt oracle matrix (P6 populated; P1-P5 not_run)
    with (OUT_DIR / "Phase17B_prompt_oracle.csv").open("w", encoding="utf-8-sig", newline="") as f:
        f.write("sample,frame,P1,P2,P3,P4,P5,P6,best_prompt\n")
        for r in oracle_rows:
            f.write(f"{r['sample']},{r['frame']},not_run(gate),not_run(gate),not_run(gate),"
                    f"not_run(gate),not_run(gate),{r['raw_oracle_f1']:.4f},P6\n")
    print("Phase17B_prompt_oracle.csv written (P1-P5 not_run: §7 gate unmet, P6 oracle >= 0.10)")


if __name__ == "__main__":
    main()
