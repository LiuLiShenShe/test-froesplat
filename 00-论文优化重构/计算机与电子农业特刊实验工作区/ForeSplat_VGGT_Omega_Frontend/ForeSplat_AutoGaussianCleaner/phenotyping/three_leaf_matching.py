"""Human matching workflow for the three handwritten calibration leaves per plant.

This module only reads existing RGB/projection/leaf-instance/phenotype artifacts.  It
never changes a Leaf_NNN ID or a reconstruction file.  The only writable artifacts
are the explicit human matching tables under ``summary``.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from batch_common import MANIFEST, OUT, ROOT, Context, read, write
from batch_measure import csvread, csvwrite

MATCH_FIELDS = [
    "plant_id", "handwritten_number", "leaf_id", "reference_view",
    "clicked_pixel_x", "clicked_pixel_y", "match_status", "human_confirmed",
    "confirmed_at", "notes", "history_row_id", "history_confirmed",
    "history_source", "history_variety", "history_excel_row",
    "manual_leaf_length_cm", "manual_leaf_width_cm",
    "capture_session", "measured_at", "measurement_tool", "operator",
]

PLANT_VARIETY = {
    "CaoMei": "草莓", "ChangShouHua": "长寿花", "DouBanLv": "豆苗绿",
    "HongZhang": "红掌", "KongQueZhuYu": "孔雀竹芋", "WanNianQing": "万年青",
    "WangWenCao": "网纹草", "XianKeLai": "仙客来", "XiangPiShu": "橡皮树",
}
WORKBOOK = ROOT / "植株数据.xlsx"
HISTORICAL_MATCHING = (
    ROOT / "00-论文优化重构" / "计算机与电子农业特刊实验工作区" /
    "阶段二十F_植物表型与尺度恢复" / "HISTORICAL_PHENOTYPE_MATCHING.csv"
)


def _plant_record(plant: str) -> dict:
    return next(item for item in read(MANIFEST)["plants"] if item["plant"] == plant)


def _match_path(plant: str) -> Path:
    return OUT / "summary" / "three_leaf_matches.csv"


def _empty_row(plant: str, number: int) -> dict:
    return {
        "plant_id": plant, "handwritten_number": str(number), "leaf_id": "",
        "reference_view": "", "clicked_pixel_x": "", "clicked_pixel_y": "",
        "match_status": "pending", "human_confirmed": "False", "confirmed_at": "",
        "notes": "", "history_row_id": "", "history_confirmed": "False",
        "history_source": "", "history_variety": "", "history_excel_row": "",
        "manual_leaf_length_cm": "", "manual_leaf_width_cm": "",
        "capture_session": "", "measured_at": "", "measurement_tool": "", "operator": "",
    }


def ensure_matches() -> list[dict]:
    """Create or migrate the fixed 19 x 3 table without losing existing entries."""
    path = _match_path("")
    existing = csvread(path) if path.exists() else []
    by_key = {(r.get("plant_id", ""), str(r.get("handwritten_number", ""))): r for r in existing}
    rows = []
    for item in read(MANIFEST)["plants"]:
        for number in range(1, 4):
            key = (item["plant"], str(number))
            row = _empty_row(*key) if key not in by_key else dict(_empty_row(*key), **by_key[key])
            row["plant_id"], row["handwritten_number"] = key
            rows.append({field: str(row.get(field, "")) for field in MATCH_FIELDS})
    csvwrite(path, rows, fields=MATCH_FIELDS)
    return rows


def save_matches(rows: list[dict]) -> None:
    csvwrite(_match_path(""), rows, fields=MATCH_FIELDS)


def _catalog(ctx: Context) -> tuple[dict, dict[str, dict], np.ndarray]:
    catalog = read(ctx.out / "leaf_instances" / "leaf_instances.json")
    by_id = {item["leaf_id"]: item for item in catalog.get("instances", [])}
    labels = np.load(ctx.out / "leaf_instances" / "triangle_leaf_ids.npy", mmap_mode="r")
    return catalog, by_id, labels


def views(plant: str) -> list[dict]:
    ctx = Context(plant)
    data, cameras = ctx.cameras()
    catalog = read(ctx.out / "leaf_instances" / "leaf_instances.json")
    preferred = {item.get("reference_view") for item in catalog.get("instances", [])}
    return [
        {"view": name, "width": data["width"], "height": data["height"], "recommended": name in preferred}
        for name in sorted(cameras)
    ]


def _leaf_id_from_label(label: int, by_id: dict[str, dict]) -> str | None:
    if label <= 0:
        return None
    leaf = f"Leaf_{int(label):03d}"
    return leaf if leaf in by_id else None


def pixel_candidates(plant: str, view: str, x: float, y: float, radius: int = 48) -> dict:
    """Read the existing triangle projection cache and return all local candidates.

    The nearest label is deliberately not auto-confirmed.  The response includes
    direct-hit and neighbourhood evidence so the reviewer can choose another leaf.
    """
    ctx = Context(plant)
    data, cameras = ctx.cameras()
    if view not in cameras:
        raise ValueError("Unknown registered RGB view")
    w, h = int(data["width"]), int(data["height"])
    x, y = float(x), float(y)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Pixel is outside the RGB image")
    cache = ctx.out / "plant_only" / "projection_cache" / f"{view}.npz"
    if not cache.exists():
        return {"clicked": {"x": x, "y": y}, "direct_leaf_id": None, "candidates": [], "status": "projection_cache_missing"}
    with np.load(cache) as data_cache:
        triangles = np.asarray(data_cache["triangle_id"])
    catalog, by_id, labels = _catalog(ctx)
    iy, ix = int(round(y)), int(round(x))
    direct_face = int(triangles[iy, ix])
    direct_label = int(labels[direct_face]) if 0 <= direct_face < len(labels) else 0
    direct_leaf = _leaf_id_from_label(direct_label, by_id)
    # Include the exact pixel and concentric neighbourhoods.  Multiple labels are
    # retained; a nearby candidate is never silently promoted to the direct hit.
    candidates: dict[str, dict] = {}
    yy, xx = np.ogrid[:triangles.shape[0], :triangles.shape[1]]
    for rr in (0, 8, 16, 32, int(radius)):
        if rr == 0:
            ys, xs = np.array([iy]), np.array([ix])
        else:
            region = (xx - ix) ** 2 + (yy - iy) ** 2 <= rr * rr
            ys, xs = np.where(region)
        face_values = triangles[ys, xs].astype(np.int64)
        valid = (face_values >= 0) & (face_values < len(labels))
        for py, px, face in zip(ys[valid], xs[valid], face_values[valid]):
            label = int(labels[int(face)])
            leaf_id = _leaf_id_from_label(label, by_id)
            if not leaf_id:
                continue
            item = candidates.setdefault(leaf_id, {"leaf_id": leaf_id, "hits": 0, "nearest_distance_px": 1e9, "nearest_pixel_x": int(px), "nearest_pixel_y": int(py), "direct": leaf_id == direct_leaf})
            item["hits"] += 1
            distance_px = math.hypot(float(px) - x, float(py) - y)
            if distance_px < item["nearest_distance_px"]:
                item.update(nearest_distance_px=distance_px, nearest_pixel_x=int(px), nearest_pixel_y=int(py))
    result = sorted(candidates.values(), key=lambda item: (not item["direct"], item["nearest_distance_px"], -item["hits"]))
    return {
        "clicked": {"x": x, "y": y}, "direct_leaf_id": direct_leaf,
        "direct_face_id": direct_face, "candidates": result[:12],
        "status": "direct_hit" if direct_leaf else ("nearby_candidates" if result else "no_leaf_candidate"),
    }


def leaf_detail(plant: str, leaf_id: str, view: str = "") -> dict:
    ctx = Context(plant)
    catalog, by_id, _ = _catalog(ctx)
    if leaf_id not in by_id:
        raise ValueError("Unknown stable Leaf ID")
    item = by_id[leaf_id]
    if view and view not in set(item.get("views", [])):
        # A mask may not exist in every RGB view; the UI can still show the mesh.
        mask_available = False
    else:
        mask_available = bool(view and (ctx.out / "leaf_instances" / leaf_id / "masks" / f"{view}.png").exists())
    measurements = csvread(ctx.out / "phenotype" / "leaf_measurements.csv")
    measurement = next((row for row in measurements if row.get("leaf_id") == leaf_id), {})
    return {
        "plant_id": plant, "leaf_id": leaf_id, "views": item.get("views", []),
        "reference_view": item.get("reference_view", ""), "faces": item.get("faces", ""),
        "mesh_url": f"/{plant}/leaf_instances/{leaf_id}/viewer.html",
        "ply_url": f"/{plant}/leaf_instances/{leaf_id}/leaf_cm.ply",
        "reference_url": f"/{plant}/leaf_instances/{leaf_id}/reference.png",
        "mask_url": f"/api/match/mask?plant={plant}&leaf_id={leaf_id}&view={view}" if mask_available else "",
        "measurement": measurement,
    }


def history_candidates(plant: str, number: int) -> list[dict]:
    """Return workbook rows; never resolve same-variety duplicates automatically."""
    variety = PLANT_VARIETY.get(plant.rstrip("0123456789"), "")
    if not WORKBOOK.exists():
        return []
    from openpyxl import load_workbook
    ws = load_workbook(WORKBOOK, read_only=True, data_only=True).active
    headers = [str(value or "") for value in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    fields = {
        "leaf_length_cm": f"叶长真值{number}", "leaf_width_cm": f"叶宽真值{number}",
        "plant_height_cm": "株高真值", "canopy_width_cm": "冠幅真值",
    }
    indexes = {key: headers.index(column) for key, column in fields.items() if column in headers}
    historical_rows = {}
    if HISTORICAL_MATCHING.exists():
        for record in csvread(HISTORICAL_MATCHING):
            if record.get("new_mesh_sample_id") == plant and str(record.get("excel_row", "")):
                historical_rows[int(record["excel_row"])] = record
    candidates = []
    for excel_row, values in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
        if str(values[0] or "").strip() != variety:
            continue
        historical = historical_rows.get(excel_row, {})
        candidates.append({
            "history_row_id": f"Sheet1!A{excel_row}", "excel_row": excel_row,
            "variety": variety, "source": str(WORKBOOK), "candidate_ambiguous": True,
            "leaf_slot": int(number),
            "leaf_length_cm": values[indexes["leaf_length_cm"]] if "leaf_length_cm" in indexes else "",
            "leaf_width_cm": values[indexes["leaf_width_cm"]] if "leaf_width_cm" in indexes else "",
            "plant_height_cm": values[indexes["plant_height_cm"]] if "plant_height_cm" in indexes else "",
            "canopy_width_cm": values[indexes["canopy_width_cm"]] if "canopy_width_cm" in indexes else "",
            "historical_matching_status": historical.get("matching_status", "NOT_LINKED") or "NOT_LINKED",
            "historical_leaf_identity_status": historical.get("leaf_identity_status", "") or "",
            "all_values": {headers[i]: values[i] for i in range(len(headers))},
        })
    return candidates


def current_state(plant: str) -> dict:
    rows = ensure_matches()
    return {"plant_id": plant, "rows": [row for row in rows if row["plant_id"] == plant], "all_rows": rows}


def _measurement_row(plant: str, leaf_id: str) -> dict:
    path = OUT / plant / "phenotype" / "leaf_measurements.csv"
    if not leaf_id or not path.exists():
        return {}
    return next((r for r in csvread(path) if r.get("leaf_id") == leaf_id), {})


def metric_candidates(plant: str, number: int, history_row_id: str, limit: int = 12) -> dict:
    """Rank existing leaf meshes against one explicitly selected workbook slot.

    This is a measurement-based shortlist, not an identity confirmation: the
    automatic geometry is often partial and the workbook row may describe a
    different specimen of the same cultivar.
    """
    selected = next((r for r in history_candidates(plant, number)
                     if r["history_row_id"] == history_row_id), None)
    if selected is None:
        raise ValueError("Select a valid historical Excel row to rank by measurements")
    target_length = selected.get("leaf_length_cm")
    target_width = selected.get("leaf_width_cm")
    try:
        target_length = float(target_length)
    except (TypeError, ValueError):
        target_length = None
    try:
        target_width = float(target_width)
    except (TypeError, ValueError):
        target_width = None
    if not any(v is not None and math.isfinite(v) and v > 0 for v in (target_length, target_width)):
        return {"plant_id": plant, "handwritten_number": int(number), "history_row_id": history_row_id,
                "candidates": [], "status": "historical_dimensions_missing"}
    ctx = Context(plant)
    records = csvread(ctx.out / "phenotype" / "leaf_measurements.csv")
    ranked = []
    for row in records:
        try:
            auto_length = float(row.get("leaf_length_cm", ""))
            if not math.isfinite(auto_length) or auto_length <= 0:
                auto_length = None
        except (TypeError, ValueError):
            auto_length = None
        try:
            auto_width = float(row.get("leaf_width_cm", ""))
            if not math.isfinite(auto_width) or auto_width <= 0:
                auto_width = None
        except (TypeError, ValueError):
            auto_width = None
        length_error = abs(auto_length - target_length) / target_length if auto_length and target_length else None
        width_error = abs(auto_width - target_width) / target_width if auto_width and target_width else None
        errors = [e for e in (length_error, width_error) if e is not None]
        if not errors:
            continue
        ranked.append({
            "leaf_id": row["leaf_id"], "automatic_leaf_length_cm": row.get("leaf_length_cm", ""),
            "automatic_leaf_width_cm": row.get("leaf_width_cm", ""), "automatic_leaf_area_cm2": row.get("leaf_area_cm2", ""),
            "manual_leaf_length_cm": selected.get("leaf_length_cm", ""),
            "manual_leaf_width_cm": selected.get("leaf_width_cm", ""),
            "length_relative_error": length_error, "width_relative_error": width_error,
            "mean_relative_error": float(sum(errors) / len(errors)), "traits_compared": len(errors),
            "measurement_status": row.get("leaf_measurement_status", ""),
            "formal_whole_leaf_valid": row.get("formal_whole_leaf_valid", ""),
            "history_row_id": history_row_id, "history_specimen_match_status": selected.get("historical_matching_status", ""),
        })
    ranked.sort(key=lambda r: (r["mean_relative_error"], -r["traits_compared"], r["leaf_id"]))
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return {"plant_id": plant, "handwritten_number": int(number), "history_row_id": history_row_id,
            "candidate_count": len(ranked), "candidates": ranked[:max(1, min(int(limit), 50))],
            "status": "ranked_by_size_only" if ranked else "no_automatic_dimensions_available",
            "warning": "尺寸接近仅用于缩小候选范围，不代表叶片身份；需要同时核对实物照片、Mesh 和历史样本来源。"}


def export_three_leaf_tables() -> dict:
    rows = ensure_matches()
    measurements = []
    errors = []
    for row in rows:
        auto = _measurement_row(row["plant_id"], row["leaf_id"])
        confirmed = row["human_confirmed"].lower() == "true"
        history_confirmed = row["history_confirmed"].lower() == "true"
        measurements.append({
            "plant_id": row["plant_id"], "handwritten_number": row["handwritten_number"],
            "leaf_id": row["leaf_id"], "match_status": row["match_status"],
            "human_confirmed": row["human_confirmed"], "reference_view": row["reference_view"],
            "history_row_id": row["history_row_id"], "history_confirmed": row["history_confirmed"],
            "manual_leaf_length_cm": row.get("manual_leaf_length_cm", ""),
            "manual_leaf_width_cm": row.get("manual_leaf_width_cm", ""),
            "capture_session": row.get("capture_session", ""), "measured_at": row.get("measured_at", ""),
            "measurement_tool": row.get("measurement_tool", ""), "operator": row.get("operator", ""),
            "automatic_leaf_length_cm": auto.get("leaf_length_cm", ""),
            "automatic_leaf_width_cm": auto.get("leaf_width_cm", ""),
            "automatic_leaf_area_cm2": auto.get("leaf_area_cm2", ""),
            "measurement_status": auto.get("leaf_measurement_status", "") if auto else "",
            "notes": row.get("notes", ""),
        })
        if confirmed and history_confirmed and row.get("leaf_id"):
            for trait, manual_key, auto_key in [
                ("leaf_length_cm", "manual_leaf_length_cm", "leaf_length_cm"),
                ("leaf_width_cm", "manual_leaf_width_cm", "leaf_width_cm"),
            ]:
                manual_value, automatic = row.get(manual_key, ""), auto.get(auto_key, "")
                if manual_value not in ("", None) and automatic not in ("", None):
                    value = float(automatic) - float(manual_value)
                    errors.append({"plant_id": row["plant_id"], "handwritten_number": row["handwritten_number"], "leaf_id": row["leaf_id"], "trait": trait, "manual_cm": manual_value, "automatic_cm": automatic, "error_cm": value, "history_row_id": row["history_row_id"]})
    summary = []
    size_suggestions = []
    for row in rows:
        for workbook_row in history_candidates(row["plant_id"], int(row["handwritten_number"])):
            try:
                response = metric_candidates(row["plant_id"], int(row["handwritten_number"]), workbook_row["history_row_id"], limit=12)
            except ValueError:
                continue
            for candidate in response["candidates"]:
                size_suggestions.append({
                    "plant_id": row["plant_id"], "handwritten_number": row["handwritten_number"],
                    "history_row_id": candidate["history_row_id"], "history_candidate_is_selected": str(row.get("history_row_id", "") == candidate["history_row_id"] and row.get("history_confirmed", "").lower() == "true"),
                    "history_specimen_match_status": candidate["history_specimen_match_status"],
                    "rank": candidate["rank"], "leaf_id": candidate["leaf_id"],
                    "manual_leaf_length_cm": candidate["manual_leaf_length_cm"], "automatic_leaf_length_cm": candidate["automatic_leaf_length_cm"],
                    "manual_leaf_width_cm": candidate["manual_leaf_width_cm"], "automatic_leaf_width_cm": candidate["automatic_leaf_width_cm"],
                    "mean_relative_error": candidate["mean_relative_error"], "measurement_status": candidate["measurement_status"],
                    "formal_whole_leaf_valid": candidate["formal_whole_leaf_valid"],
                })
    for trait in ("leaf_length_cm", "leaf_width_cm"):
        vals = [float(r["error_cm"]) for r in errors if r["trait"] == trait]
        # R² is left blank for a single or constant reference sample, matching
        # batch_manual.py's established rule.
        manual_values = [float(r["manual_cm"]) for r in errors if r["trait"] == trait]
        ss = float(np.sum((np.asarray(manual_values) - np.mean(manual_values)) ** 2)) if manual_values else 0.0
        r2 = float(1.0 - np.sum(np.square(vals)) / ss) if len(vals) >= 2 and ss > 0 else ""
        summary.append({"trait": trait, "n_pairs": len(vals), "MAE": float(np.mean(np.abs(vals))) if vals else "", "RMSE": float(np.sqrt(np.mean(np.square(vals)))) if vals else "", "R2": r2, "status": "NO_VERIFIED_MANUAL_PAIRS" if not vals else "PAIRED_BY_HUMAN"})
    folder = OUT / "summary"
    csvwrite(folder / "three_leaf_measurements.csv", measurements)
    csvwrite(folder / "three_leaf_metric_suggestions.csv", size_suggestions, fields=[
        "plant_id", "handwritten_number", "history_row_id", "history_candidate_is_selected", "history_specimen_match_status", "rank", "leaf_id",
        "manual_leaf_length_cm", "automatic_leaf_length_cm", "manual_leaf_width_cm", "automatic_leaf_width_cm",
        "mean_relative_error", "measurement_status", "formal_whole_leaf_valid",
    ])
    csvwrite(folder / "three_leaf_errors.csv", errors, fields=["plant_id", "handwritten_number", "leaf_id", "trait", "manual_cm", "automatic_cm", "error_cm", "history_row_id"])
    csvwrite(folder / "three_leaf_error_summary.csv", summary)
    _write_overview(rows, measurements)
    return {"matches": rows, "measurements": measurements, "errors": errors, "error_summary": summary}


def _write_overview(rows: list[dict], measurements: list[dict]) -> None:
    by_key = {(r["plant_id"], r["handwritten_number"]): r for r in rows}
    measurement_by_key = {(r["plant_id"], r["handwritten_number"]): r for r in measurements}
    cards = []
    for item in read(MANIFEST)["plants"]:
        plant = item["plant"]
        ctx = Context(plant)
        _, cameras = ctx.cameras()
        catalog = read(ctx.out / "leaf_instances" / "leaf_instances.json")
        instances = {entry["leaf_id"]: entry for entry in catalog.get("instances", [])}
        fallback_view = next((entry.get("reference_view") for entry in catalog.get("instances", [])
                              if entry.get("reference_view") in cameras), next(iter(sorted(cameras)), ""))
        for number in range(1, 4):
            row = by_key[(plant, str(number))]
            leaf = row.get("leaf_id", "")
            instance = instances.get(leaf, {})
            view = row.get("reference_view") or instance.get("reference_view") or fallback_view
            query = f"plant={plant}&view={view}"
            rgb_url = f"/api/match/rgb?{query}" if view else ""
            mask_path = ctx.out / "leaf_instances" / leaf / "masks" / f"{view}.png" if leaf and view else None
            mask_url = f"/api/match/mask?{query}&leaf_id={leaf}" if mask_path and mask_path.exists() else ""
            measurement = measurement_by_key[(plant, str(number))]
            viewer = f"/{plant}/leaf_instances/{leaf}/viewer.html" if leaf else ""
            edit_url = f"/match?plant={plant}&view={view}&number={number}"
            mask_layer = f'<img class="mask" loading="lazy" src="{html.escape(mask_url)}" alt="Leaf mask">' if mask_url else ""
            image = (f'<div class="photo"><img loading="lazy" src="{html.escape(rgb_url)}" alt="{plant} {number}号原始照片">{mask_layer}</div>'
                     if rgb_url else '<div class="photo empty">无可用 RGB 视角</div>')
            mesh_link = f'<a href="{html.escape(viewer)}" target="_blank">打开独立 Mesh / 选点</a>' if viewer else "尚未匹配 Leaf_NNN"
            auto_text = " · ".join([
                f"长 {html.escape(str(measurement.get('automatic_leaf_length_cm') or '—'))} cm",
                f"宽 {html.escape(str(measurement.get('automatic_leaf_width_cm') or '—'))} cm",
                f"面积 {html.escape(str(measurement.get('automatic_leaf_area_cm2') or '—'))} cm²",
            ])
            manual_text = f"人工长 {html.escape(str(measurement.get('manual_leaf_length_cm') or '—'))} cm · 宽 {html.escape(str(measurement.get('manual_leaf_width_cm') or '—'))} cm"
            confirmed = row["human_confirmed"].lower() == "true"
            cards.append(
                f'<article class="card"><header><b>{html.escape(plant)} · {number}号</b>'
                f'<span class="{ "confirmed" if confirmed else "pending" }">{html.escape(row["match_status"])}</span></header>'
                f'{image}<div class="details"><b>{html.escape(leaf or "未匹配")}</b> · 视角 {html.escape(view or "—")}<br>'
                f'{mesh_link}<p>{auto_text}</p><p>{manual_text}</p>'
                f'<a href="{html.escape(edit_url)}">打开并核对此编号</a></div></article>'
            )
    confirmed_count = sum(row["human_confirmed"].lower() == "true" for row in rows)
    page = """<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ForeSplat 57 片标定叶片总览</title>
    <style>body{font:15px system-ui;margin:20px;background:#f4f7f6;color:#16312b}a{color:#145e49}nav{position:sticky;top:0;background:#f4f7f6;padding:10px 0;z-index:2}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}.card{background:#fff;border:1px solid #cbd8d2;border-radius:8px;overflow:hidden}.card header{display:flex;justify-content:space-between;padding:10px;background:#eaf2ee}.confirmed{color:#08643d}.pending{color:#80611d}.photo{height:300px;position:relative;background:#1b241f;overflow:hidden}.photo img{width:100%;height:100%;object-fit:contain;position:absolute;inset:0}.photo .mask{opacity:.5;mix-blend-mode:screen;object-fit:fill;pointer-events:none}.photo.empty{display:grid;place-items:center;color:#fff}.details{padding:10px}.details p{margin:6px 0;color:#40574d;font-size:13px}</style>
    <h1>ForeSplat · 57 片人工标定叶片</h1><nav><b>已确认 """ + str(confirmed_count) + "/57</b> · <a href='/match'>打开交互匹配页面</a> · <a href='three_leaf_matches.csv'>匹配 CSV</a> · <a href='three_leaf_measurements.csv'>测量 CSV</a> · <a href='three_leaf_metric_suggestions.csv'>尺寸候选排序</a> · <a href='three_leaf_errors.csv'>误差 CSV</a> · <a href='three_leaf_error_summary.csv'>误差汇总</a></nav><div class='grid'>" + "".join(cards) + "</div>"
    (OUT / "summary" / "three_leaf_match_overview.html").write_text(page, encoding="utf-8")


def confirm(payload: dict) -> dict:
    plant = str(payload.get("plant_id", ""))
    number = str(payload.get("handwritten_number", ""))
    if plant not in {r["plant"] for r in read(MANIFEST)["plants"]} or number not in {"1", "2", "3"}:
        raise ValueError("Unknown plant or handwritten number")
    leaf_id = str(payload.get("leaf_id", "")).strip()
    if leaf_id:
        ctx = Context(plant)
        catalog, by_id, _ = _catalog(ctx)
        if leaf_id not in by_id:
            raise ValueError("Leaf ID is not present in this plant's existing instance catalog")
        view = str(payload.get("reference_view", ""))
        if view and view not in ctx.cameras()[1]:
            raise ValueError("Reference view is not a registered RGB camera for this plant")
    rows = ensure_matches()
    row = next(r for r in rows if r["plant_id"] == plant and r["handwritten_number"] == number)
    status = str(payload.get("match_status", "confirmed" if payload.get("human_confirmed") else "pending"))
    if status not in {"pending", "confirmed", "not_found", "candidate_selected", "cancelled"}:
        raise ValueError("Invalid match status")
    human_confirmed = payload.get("human_confirmed") is True and status == "confirmed"
    if human_confirmed and leaf_id:
        duplicate = next((r for r in rows if r["plant_id"] == plant and
                          r["handwritten_number"] != number and r["leaf_id"] == leaf_id and
                          r["human_confirmed"].lower() == "true"), None)
        if duplicate:
            raise ValueError(f"{leaf_id} is already confirmed for handwritten number {duplicate['handwritten_number']}")
    history_candidate = None
    if human_confirmed and leaf_id and payload.get("history_confirmed") is True:
        history_row_id = str(payload.get("history_row_id", ""))
        history_candidate = next((candidate for candidate in history_candidates(plant, int(number))
                                  if candidate["history_row_id"] == history_row_id), None)
        if history_candidate is None:
            raise ValueError("Select a valid historical Excel row before confirming the measurement pair")
    row.update({
        "leaf_id": leaf_id if status not in {"not_found", "cancelled"} else "",
        "reference_view": str(payload.get("reference_view", "")),
        "clicked_pixel_x": str(payload.get("clicked_pixel_x", "")),
        "clicked_pixel_y": str(payload.get("clicked_pixel_y", "")),
        "match_status": status,
        "human_confirmed": "True" if human_confirmed else "False",
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if human_confirmed else "",
        "notes": str(payload.get("notes", ""))[:2000],
        "history_row_id": history_candidate["history_row_id"] if history_candidate else str(payload.get("history_row_id", "")),
        "history_confirmed": "True" if history_candidate else "False",
        "history_source": history_candidate["source"] if history_candidate else str(payload.get("history_source", "")),
        "history_variety": history_candidate["variety"] if history_candidate else str(payload.get("history_variety", "")),
        "history_excel_row": str(history_candidate["excel_row"]) if history_candidate else str(payload.get("history_excel_row", "")),
    })
    # Keep historical values explicit in the exported table so a later revision is auditable.
    manual_length = history_candidate["leaf_length_cm"] if history_candidate else payload.get("manual_leaf_length_cm", "")
    manual_width = history_candidate["leaf_width_cm"] if history_candidate else payload.get("manual_leaf_width_cm", "")
    row["manual_leaf_length_cm"] = "" if manual_length is None else str(manual_length)
    row["manual_leaf_width_cm"] = "" if manual_width is None else str(manual_width)
    row["capture_session"] = str(payload.get("capture_session", ""))[:2000]
    row["measured_at"] = str(payload.get("measured_at", ""))[:2000]
    row["measurement_tool"] = str(payload.get("measurement_tool", ""))[:2000]
    row["operator"] = str(payload.get("operator", ""))[:2000]
    save_matches(rows)
    # A confirmed historical row is also mirrored into the existing manual
    # reference table only after the same-capture protocol fields are present.
    # This reuses batch_manual.py for the established MAE/RMSE/R² rules.
    protocol = [row.get(key, "").strip() for key in ["capture_session", "measured_at", "measurement_tool", "operator"]]
    if row["history_confirmed"] == "True" and row["human_confirmed"] == "True" and row["leaf_id"] and all(protocol):
        manual_path = OUT / plant / "phenotype" / "manual_reference.csv"
        if manual_path.exists():
            manual_rows = csvread(manual_path)
            target = next((item for item in manual_rows if item.get("leaf_id") == row["leaf_id"]), None)
            if target is not None:
                target.update({"leaf_length_cm": row["manual_leaf_length_cm"], "leaf_width_cm": row["manual_leaf_width_cm"], "capture_session": row["capture_session"], "measured_at": row["measured_at"], "measurement_tool": row["measurement_tool"], "operator": row["operator"], "pair_confirmed": "True", "notes": "Historical Excel row explicitly selected in /match; same plant and Leaf_NNN confirmed by reviewer."})
                csvwrite(manual_path, manual_rows)
                from batch_manual import main as batch_manual_main
                batch_manual_main(Context(plant))
                from batch_report import summarize
                summarize()
    return export_three_leaf_tables()
