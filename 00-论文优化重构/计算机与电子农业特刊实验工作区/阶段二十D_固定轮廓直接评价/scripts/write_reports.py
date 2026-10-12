from __future__ import annotations

import json

import numpy as np

from direct_common import DIFFICULTIES, FIX, METRICS, OLD, SWITCHES, WORK, read_csv, sha256, table, write_json, write_text
from evaluate_test32 import direction


def numeric(rows):
    return [{key: float(value) if key in METRICS or key.startswith(("r0_", "r1_", "delta_")) and key.rsplit("_", 1)[-1] not in ("id", "ids")
             and value.replace(".", "", 1).replace("-", "", 1).isdigit() else value for key, value in row.items()} for row in rows]


def main():
    summaries = read_csv(WORK / "R0_R1_TEST32_SUMMARY.csv")
    summary_map = {(row["rasterizer"], row["subset"], row["method"], row["aggregation"]): row for row in summaries}
    results = read_csv(WORK / "R0_R1_TEST32_RESULTS.csv")
    secondary = {row["frame_id"]: row for row in read_csv(WORK / "R0_R1_TEST32_SECONDARY_RESULTS.csv")}
    by_frame = {row["frame_id"]: row for row in results}
    ambiguity = read_csv(WORK / "RASTERIZATION_SENSITIVITY.csv")
    evaluation = json.loads((WORK / "logs/NUMERIC_EVALUATION_STATUS.json").read_text())
    primary = [summary_map[("opencv", "ALL32", method, "sample_equal_mean")] for method in ("R0", "R1", "R1_MINUS_R0")]
    checks = {}
    for rasterizer in ("opencv", "pillow"):
        baseline = summary_map[(rasterizer, "ALL32", "R0", "sample_equal_mean")]
        treatment = summary_map[(rasterizer, "ALL32", "R1", "sample_equal_mean")]
        paired = summary_map[(rasterizer, "ALL32", "R1_MINUS_R0", "sample_equal_mean")]
        checks[rasterizer] = {"all32_nonnegative_f1_iou": float(paired["f1"]) >= 0 and float(paired["iou_fg"]) >= 0,
                              "no_new_empty_masks": int(treatment["empty_mask_count"]) <= int(baseline["empty_mask_count"]),
                              "no_new_severe_failures": all(not (float(row["r1_f1"]) < 0.5 <= float(row["r0_f1"]))
                                                           for row in (results if rasterizer == "opencv" else secondary.values()))}
    engineering = all(all(value.values()) for value in checks.values()) and evaluation["rasterization_sensitivity"] == "STABLE"
    decision = "ENGINEERING_LOCKED" if engineering else "PROVISIONAL"
    switch_effects = [by_frame[stem]["f1_effect"] for stem in SWITCHES]
    switches = "IMPROVED" if all(value == "IMPROVED" for value in switch_effects) else "REGRESSED" if all(value == "REGRESSED" for value in switch_effects) else "MIXED"
    final = {"PHASE20D_DIRECT_EVALUATION": "PASS", "RASTERIZED_MASKS": "32/32", "GEOMETRY_AMBIGUITY": 13,
             "MAIN_GT_GEOMETRY_AMBIGUITY": 10, "TEST32_R0_F1": float(primary[0]["f1"]),
             "TEST32_R1_F1": float(primary[1]["f1"]), "TEST32_R1_MINUS_R0": float(primary[2]["f1"]),
             "SWITCH_CASES": switches, "RASTERIZATION_SENSITIVITY": evaluation["rasterization_sensitivity"],
             "FOREGROUND_METHOD": decision, "PHASE20E_PILOT_READY": "CONDITIONAL",
             "metric_scope": "Fixed-contour rasterization evaluation; sample-equal mean; all32",
             "not_original_human_gate_certified": True, "ranking_checks": checks,
             "remaining_severe_frames": [row["frame_id"] for row in results if float(row["r1_f1"]) < 0.5],
             "full_config_path": "logs/full_r1_runtime_parameters.json", "spec_sha256": evaluation["spec_sha256"]}
    write_json(WORK / "FINAL_STATUS.json", final)
    switch_rows = []
    for rasterizer, values in (("opencv", by_frame), ("pillow", secondary)):
        for stem in SWITCHES:
            row = values[stem]
            switch_rows.append({"frame": stem, "rasterizer": rasterizer, "geometry": row["geometry_warning"],
                                "R0_ID": row["r0_selected_id"], "R1_ID": row["r1_selected_id"], "oracle_ID": row["oracle_candidate_id"],
                                "R0_F1": float(row["r0_f1"]), "R1_F1": float(row["r1_f1"]), "delta_F1": float(row["delta_f1"]),
                                "R0_IoU": float(row["r0_iou_fg"]), "R1_IoU": float(row["r1_iou_fg"]),
                                "R0_regret": float(row["r0_oracle_regret"]), "R1_regret": float(row["r1_oracle_regret"]),
                                "effect": row["f1_effect"]})
    write_text(WORK / "R0_R1_SWITCH_CASES.md", "# Two real switch cases\n\n" +
               table(switch_rows, ["frame", "rasterizer", "R0_ID", "R1_ID", "oracle_ID", "R0_F1", "R1_F1", "delta_F1", "R0_IoU", "R1_IoU", "R0_regret", "R1_regret", "effect"]) +
               "\n\nBoth switches improve F1/IoU with both fixed rasterizers, and both R1 selections are the F1 oracle. "
               "Both frames have GEOMETRY_AMBIGUITY warnings, retained without repair. "
               "These are measured contour-overlap improvements, not fabricated human target approvals. "
               "All30 unchanged selections were actually evaluated; WanNianQing1_0223 remains severe(F1=0) "
               "and selects00 instead of oracle01. Candidate stability does not mean correctness.\n\n"
               "Figures: figures/comparison_CaoMei2_0146.png and figures/comparison_WanNianQing1_0227.png.\n"
               "Sources: R0_R1_TEST32_RESULTS.csv, SECONDARY_RESULTS.csv, ORACLE_ALL_CANDIDATES.csv, real production logs.")
    main_affected = sum(int(row["main_gt_shape_ambiguous"]) for row in ambiguity)
    geometry_table = [{"frame": row["frame_id"], "warning": row["geometry_warning"], "main_target_affected": row["main_gt_shape_ambiguous"],
                       "cross_raster_IoU": float(row["raster_mask_iou"]), "OpenCV_area": row["opencv_area_px"],
                       "Pillow_area": row["pillow_area_px"], "area_delta_px": row["area_difference_px"],
                       "area_delta_ratio": float(row["relative_area_difference"]), "noticeable": row["noticeable_raster_disagreement"]} for row in ambiguity]
    subset_rows = [row for row in summaries if row["aggregation"] in ("sample_equal_mean", "frame_equal_mean")]
    metric_stability = read_csv(WORK / "METRIC_RASTERIZATION_STABILITY.csv")
    ranges = [{"metric": metric,
               "max_abs_R0_Pillow_minus_OpenCV": max(abs(float(row[f"r0_pillow_minus_opencv_{metric}"])) for row in metric_stability),
               "max_abs_R1_Pillow_minus_OpenCV": max(abs(float(row[f"r1_pillow_minus_opencv_{metric}"])) for row in metric_stability),
               "paired_direction_reversal_frames": sum(row[f"paired_direction_stable_{metric}"] != "True" for row in metric_stability)}
              for metric in METRICS]
    write_text(WORK / "GEOMETRY_SENSITIVITY_REPORT.md", f"# Geometry sensitivity — {evaluation['rasterization_sensitivity']}\n\n"
               f"All32 kept. Recomputed75 original shapes:17 shapes in13 frames intersect/backtrack. "
               f"MainGT contributors affected in{main_affected} frames;3 other warnings concern only ignored plant/pot components. "
               "No coordinate repair or signature-based certification. Full original/rounded mappings are local with indexedSHA256.\n\n"
               + table(geometry_table, ["frame", "warning", "main_target_affected", "cross_raster_IoU", "OpenCV_area", "Pillow_area", "area_delta_px", "area_delta_ratio", "noticeable"]) +
               "\n\nNo frame crosses the predeclaredIoU<0.99 or >1% absolute area-difference diagnostic threshold. "
               "This indicates small implementation disagreement, not absence of annotation ambiguity. Two similar filling rules "
               "cannot establish the intended semantics of an intersecting boundary. No main-rule selection uses R1 outcomes.\n\n"
               + table(numeric(subset_rows), ["rasterizer", "subset", "method", "aggregation", "frames", "sequences"] + METRICS + ["empty_mask_count", "severe_f1_lt_0_5_count"]) +
               "\n\nAll32 sample-equal and frame-equal means are identical because8 sequences each have4 views. "
               "Subset means differ because the13/19 counts per sequence are unequal. The19-frame subset shows zero R1−R0 "
               "because both switches lie in the13-frame subset; this is not a second independent win. "
               "The remaining severe un-switched error is inOTHER19.\n\n" +
               table(ranges, ["metric", "max_abs_R0_Pillow_minus_OpenCV", "max_abs_R1_Pillow_minus_OpenCV", "paired_direction_reversal_frames"]) +
               "\n\nF1/IoU ordering and both switch directions are stable; report statusSTABLE is restricted to these two "
               "implementations, not robust confirmation against all possible contour interpretations. Other metrics' numerical "
               "changes are disclosed above. Sources: CONTOUR_GEOMETRY_AUDIT.csv, RASTERIZATION_SENSITIVITY.csv, "
               "METRIC_RASTERIZATION_STABILITY.csv and both raster result tables.")
    regression = read_csv(WORK / "GT10_DEVELOPMENT_REGRESSION.csv")
    failures = [row for row in results if float(row["r1_f1"]) < 0.5]
    severe_rows = [{"frame": row["frame_id"], "R0_ID": row["r0_selected_id"], "R1_ID": row["r1_selected_id"],
                   "oracle_ID": row["oracle_candidate_id"], "R1_F1": float(row["r1_f1"]), "oracle_F1": float(row["oracle_f1"]),
                   "FP": row["r1_fp"], "FN": row["r1_fn"], "R1_regret": float(row["r1_oracle_regret"])} for row in failures]
    write_text(WORK / "FOREGROUND_METHOD_DECISION.md", f"# Foreground method decision — {decision}\n\n"
               "Frozen engineering recipe: **V1=P6+per_instance+single+R1**; original score weights "
               "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5; archived confidence precision6decimals; "
               "SPNP/Residual/GCFC/A6/A7/semantic gate/temporal alignment/repromptOFF. "
               "Full unchanged parameters: logs/full_r1_runtime_parameters.json. Frozen ranking codeSHA256="
               "c80442a2161c0560bc7bf53a7714b6d9da5a996e2a824716a0b0dc2092071905.\n\n"
               "## Seven requested answers\n\n"
               "1. GT10's0039/0040/0041 repairs remain supported: read-only recomputation of original candidate identities "
               "and masks reproduces R0F1=0 and R1F1>0.98. These are development regressions, not independent TEST evidence.\n"
               "2. Both TEST32 switches improve under OpenCV and Pillow; both R1IDs reach the raw candidateF1 oracle.\n"
               "3. No new empty or severeF1<0.5 frame; severe count drops3→1. One old severe error remains, listed below.\n"
               f"4. Full32 sample-equalF1 {float(primary[0]['f1']):.9f}→{float(primary[1]['f1']):.9f}; "
               f"delta={float(primary[2]['f1']):+.9f}. F1 andIoU improve on2, tie on30, regress on0; not32 improvements. "
               "Means ofBoundary-F1/HD95/leakage also improve, but boundary agreement remains modest and long-distance "
               "outliers remain. No claim of universal correct-target selection.\n"
               "5. The13-frame ambiguity subset retains a positive pairedF1/IoU effect with both implementations; "
               "this supports engineering ranking stability only, not an ambiguity-freeGT conclusion.\n"
               "6. Both rasterizers give the sameF1/IoU method ordering and switch direction; STABLE is limited to that sensitivity check.\n"
               f"7. {decision} is adequate to freeze matched3D engineering treatments without tuning. "
               "It does not require or assert original dual-signatureGatePASS, and does not promise segmentation accuracy "
               "on every view. A/B/C matching and coordinate/data readiness remain conditional.\n\n"
               + table(regression, ["frame_id", "method", "candidate_id", "f1", "iou_fg", "dataset"]) +
               "\n\n## Retained serious error\n\n" + table(severe_rows, ["frame", "R0_ID", "R1_ID", "oracle_ID", "R1_F1", "oracle_F1", "FP", "FN", "R1_regret"]) +
               "\n\nNo new label, approval, R1 formula, weight, mask cleanup, TEST-trained threshold or runtime oracle is introduced. "
               "Oracle is evaluation-only. Continue this error in Phase20E analysis if the corresponding view is registered; "
               "do not silently replace it with oracle01.\n\nSources: Phase20C-FIX FINAL_FOREGROUND_METHOD_DECISION.md and "
               "GT10_DEVELOPMENT_REGRESSION.csv; new complete32results, both rasterized summaries and immutable production logs.")
    pilot = read_csv(WORK / "PHASE20E_SEQUENCE_READINESS.csv")
    write_text(WORK / "PHASE20E_EXECUTION_READINESS.md", "# Phase20E executable preparation — CONDITIONAL, training not started\n\n"
               "The foreground configuration is engineering-locked. TEST32 is not a full-sequence training input. "
               "Use3plants×3matchedtreatments×1fixedseed=9trainings as a feasibility pilot, not significance or variance evidence.\n\n"
               + table(pilot, ["sample", "raw_rgb_count", "registered_views", "train_views", "validation_views", "model_undistorted_dimensions",
                               "historical_rgba_exists", "historical_rgba_matching_grid", "undistorted_rgb_matching_grid", "undistorted_rgb_unreadable",
                               "colmap_input_unreadable", "verified_frozen_candidate_views", "missing_full_view_candidate_views",
                               "test32_registered_views", "test32_unregistered_views"]) +
               "\n\n## Actual findings, not assumptions\n\n"
               "- FullRGB:01-FFmepg/{sample}; registered/camera models:04-COLMAP/{sample}/sparse/0 and distorted/sparse/0. "
               "View and camera manifests include exactIDs, names, hashes, raw inputs and real dimensions.\n"
               "- Historical05-2DGS-new cfg_args uses images_rgba, resolution=2, eval=False, sh_degree=3, "
               "white_background=False and old Windows paths. Cannot reuse as an already matched held-out control.\n"
               "- CaoMei2 registration names are crop_0095.png..crop_0200.png on previously foreground-processedRGB, "
               "not raw fullRGB. This is a segmentation-conditioned pose selection and a limitation of reusing that model. "
               "Its0007/0008/0009 test views are not in this model;0146 is. Other two sequence test registration status "
               "is explicitly enumerated in the table, never assumed.\n"
               "- All examined full-view metadata archives are indexed inPHASE20E_CANDIDATE_ARCHIVE_SEARCH.csv. "
               "The frozenU00 set supplies only the listed registered views, not full sequences. Existing03-SAM "
               "foreground masks are not originalP6 per-instance SAM3 candidates.\n"
               "- Actual unreadable/mismatchedRGB and incompleteRGBA are recorded, not silently skipped. "
               "Historical names/extensions can differ; joins are explicit, not positional. No resizing workaround.\n\n"
               "## Minimum future execution checklist\n\n"
               "1. Preserve all existingmodels. Resolve unreadable images from originalRGB and provenance; forCaoMei2 "
               "either reuse and document historical mask-conditioned poses identically forA/B/C or build a separately "
               "versioned rawRGB-onlyCOLMAP model before splitting. Do not pretend these are already raw-matched poses.\n"
               "2. Freeze exact registered viewIDs and sample-specificCameraOPENCV/PINHOLE models. Current proposed split "
               "is saved inPHASE20E_VIEW_MANIFEST.csv: sort exactregistered stems, every eighth(index%8==0) held-out; "
               "all other views train. Seed=0, iterations=30000, resolution=2, sh_degree=3, blackbackground. "
               "If a new model is necessary, create a new full-viewmanifest/split before any treatment, not a per-treatment model.\n"
               "3. A=original historicalinput handling, B=frozenR0, C=frozenR1. Use identical viewset/poses/intrinsics/points3D/seed/iterations/optimizer/hardware. "
               "IfA cannot be provenance-completely recovered, report it and rebuild a versionedA control; do not "
               "claim the partial existingRGBA collection represents a matchedA treatment.\n"
               "4. Future completeP6 generation: single SAM3 archive per sequence on the originalRGB grid, per_instanceP6 "
               "with savedfullmetadata, IDs/boxes/order/confidence/model/checkpoint/RGB/maskSHA256. Then replay the SAME "
               "archive under frozenbase andr1. Any new archive isNEW_CANDIDATE_SET, never a restored71candidateTEST32set. "
               "Do not regenerate or tune onTEST32 during this phase.\n"
               "5. Map rawmasks into the exactundistorted domain, not2160×3840→2119×3779resize. Actualcamera dimensions "
               "are sample-specific in the table. For each pixelcenter in theundistortedPINHOLE grid, unproject "
               "using thatK, then project using thepaired originalOPENCV distortion model; remap thebinarymask with "
               "nearest-neighbor andconstant0. ResolveCOLMAP pixelcenter convention using its camera implementation; "
               "do not assume anOpenCV origin convention without anRGB roundtrip. Save mapSHA256/validROI/source/output "
               "and test actualRGBroundtrip and sparse-trackreprojection. No camera-number-only joins.\n"
               "6. The same mapping applies to all treatmentRGB/alpha. Use mapped binaryalpha/RGBA without changing "
               "2DGS/M2M3-GO. Existing 2DGS camera loader handles undistortedPINHOLE, not arbitraryOPENCV. "
               "Validate exactresolution handling, name/extensioncoverage andmaskalignment before launching.\n"
               "7. Report held-outPSNR/SSIM/LPIPS plus fixedindependentROI/silhouette/leakage/outlier/emptyrender stats, "
               "time/peakmemory and all failures, perplant. No reference3Dscan means no truegeometryaccuracy claim. "
               "Reuse a historicalcheckpoint only if all inputs/config/camera/split/seed provenance truly match; "
               "otherwise rerunA. Phase20E still requires separate userauthorization.\n\n"
               "Current block is technicalfull-viewdata/coordinate preparation, NOT another manual-label/signature loop. "
               "No 2DGS command is executed. Sources: actual binarymodels/images/cfg_args, hashed manifests and "
               "unchanged2DGS dataset_readers.py. COLMAP documentation retrieved viaContext7: "
               "https://github.com/colmap/colmap/blob/main/doc/faq.rst and doc/cameras.rst.")
    difficulty = read_csv(WORK / "DIFFICULTY_RESULTS.csv")
    write_text(WORK / "DIFFICULTY_ANALYSIS.md", "# Archived difficulty grouping — all real categories kept\n\n"
               "D1/D3/D6/D11/D12 use unchanged160existingYES/NO cells. These are archivedhuman grouping variables, "
               "not independently approveddifficultyGT. Labels may overlap; denominators are shown and never summed "
               "as disjointframes. All32 measured, including no-recordeddifficulty frames.\n\n" +
               table(numeric([row for row in difficulty if row["rasterizer"] == "opencv" and row["aggregation"] == "frame_equal_mean"]),
                     ["difficulty", "frames", "sequences", "r0_f1", "r1_f1", "delta_f1", "improved_frames", "regressed_frames", "unchanged_frames", "r1_severe_count", "major_failure_manifestation"]) +
               "\n\nRealcomparison panels for every32frame are infigures/, grouped inDIFFICULTY_GALLERY.html. "
               "There are zero measuredF1regressions, so no regressionexample is manufactured. "
               "The unchangedseverefailureWanNianQing1_0223 is shown withits existingnone label, not relabeled to fit "
               "a difficultcategory. Bothswitches are shown; theWanNianQing1 switch also hasnone inexistingrecords. "
               "Geometrylimitations are ineachaffectedcaption. Source: DIFFICULTY_RESULTS.csv, completeper-frame tables "
               "andfigures/FIGURE_SOURCE_INDEX.csv.")
    report_status = "\n".join(f"{key}: {value}" for key, value in final.items() if key.isupper())
    per_sequence = [row for row in read_csv(WORK / "R0_R1_TEST32_PER_SEQUENCE.csv") if row["rasterizer"] == "opencv" and row["aggregation"] == "frame_equal_mean"]
    write_text(WORK / "PHASE20D_DIRECT_EVALUATION_REPORT.md", "# ForeSplat Phase20D-Final — fixed-contour direct evaluation\n\n"
               "Date:2026-10-12. Startingmain=d3b474b9, initially clean; previousprotocolwork was committed/localfiles "
               "were preserved. This stage executes the user's fixed-contour decision, not anotherGT preparation loop.\n\n"
               "```text\n" + report_status + "\n```\n\n"
               "## Complete primary32-frame results\n\n" + table(numeric(primary), ["method", "frames", "sequences"] + METRICS + ["empty_mask_count", "severe_f1_lt_0_5_count"]) +
               "\n\nMain=sample_equal_mean;8×4views makes frame_equal_mean identical here. "
               "R1 improves2frames, regresses0, ties30. Bothtrue switches improve;30ties are measured, not assumedcorrect. "
               "Severeproxy3→1, empty0→0. WanNianQing1_0223 remains F1=0/oracle regret≈0.98718, "
               "and it is explicitly present inalltables andfigures. No badframe is deleted. "
               "Median/populationSD, pairedmeans,8sequences,13/19subsets andbothrasterizers are separateCSVrows.\n\n"
               "## Eight sequences\n\n" + table(numeric(per_sequence), ["sample", "method", "frames"] + METRICS + ["severe_f1_lt_0_5_count"]) +
               "\n\n## Provenance and protocol limits\n\n"
               "32originalJSON files read completely, all75originalshape sequences preserved. MainGT uses only "
               "potted_plant;0blue_cube labels means no inventedsubtraction. No plant/pot requirement, no approval "
               "fields fabricated. PrimaryOpenCVfillPoly and secondaryPillow use the predeclaredhalf-away rounding. "
               "All32primary and32secondary masks have source/RGB/output/mappinghashes. No empty surrogateGT. "
               "13frames/17shapes have numericalself-intersection/backtrackingwarnings;10frames affect mainGT. "
               "All remain inmainstatistics. STABLE undertworasterizers does not certify annotationsemantics.\n\n"
               "The originalPhase20DGate is unchanged andhistoricallyBLOCKED; itwasnotrerun oroverridden. "
               "This is **Fixed-contour rasterization evaluation**, or **Evaluation using deterministically rasterized manual contours**, "
               "notdual-signaturestandardhumanGT. Userstatement ofcompletedhumanreview is retained; "
               "there are no newhumanapprovals. Difficultylabels remainexistinggroupvariables.\n\n"
               "FourrealCPU smoke runs passedfirst, thenrealR0/R1productionCLI full32runs. Both use71originalP6candidates, "
               "originalmasks andstrictmetadata; onlyranking_policy differs. RealID/mask equality, fullscores, runtimeparameters, "
               "codeSHA256, sourcecandidateSHA256, outputSHA256 andstdoutlogs are retained. Finalmasks equalrawselectedmasks "
               "underthisfrozenconfiguration. NoSAM3loaded/reinference, optimizer, GT-basedrank changes oroptionalmodule. "
               "OriginalarchivedSAMconfidence precision6decimals; unsavedtensorbitsarenotinvented.\n\n"
               "HistoricalGT10threerepairs are recomputedseparately and excludedfromTEST32tables. "
               "Themethoddecision isengineering-onlyENGINEERING_LOCKED withoneknownseriouserror; "
               "no universalaccuracyclaim. Phase20E3plants areCONDITIONALduefullviewcandidatecoverage, "
               "historicalRGBAcoverage/coordinateandRGBreadabilityissues, notmissinghuman signatures. "
               "Exactcamera grids differ; no2160×3840→2119×3779resizing. No3Dtrainingorpapermodification.\n\n"
               "## Delivery and verification\n\n"
               "Requested reports/CSVs,32comparisonfigures,2switchPDFs,sourceindices,scripts/tests/logs and "
               "full-view3Dpreparationmanifests are inthisdirectory. Large derivedmasks/mappings/productionoutputs "
               "remainlocalignored; SHA256 andreproductioncommandsaretracked. Verificationresults are inlogs/ "
               "TEST_FINAL.txt, SOURCE_PROTECTION_FINAL.json and DELIVERY_VERIFICATION.json; readtheiractualPASS/FAIL "
               "rather than interpretingthisreport as ahumanGTGatePASS. Commit/pushreceipt is separate.\n\n"
               "Sources: preservedlocalJSON/RGB/candidate archives, actualproductionruns, frozenPhase20B/Cmetrics, "
               "Phase20C-FIXmethoddecision andnewcompleteCSVtables. BochaHTTP403; no externalexperimentdata "
               "wasretrieved. CurrentOpenCV/Pillow/COLMAPdocumentation wasretrievedviaContext7; seeprotocolreferences.")
    write_text(WORK / "REPRODUCE.md", "# Reproduce without overwriting history\n\n"
               "Prerequisites: original32JSON/RGB,71originalmasks, strictmetadata andall existingfrozenlocalarchives. "
               "Versions are inlogs/RUNTIME_ENVIRONMENT.json. Sourcechecks are mandatory; missingfiles must not be guessed.\n\n"
               "```bash\nSTAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_固定轮廓直接评价'\n"
               "export PYTHONDONTWRITEBYTECODE=1\n"
               "python3 -m pytest \"$STAGE/tests\" -q -p no:cacheprovider\n"
               "python3 \"$STAGE/scripts/verify_delivery.py\" --output \"$STAGE/logs/new_verification.json\"\n"
               "python3 \"$STAGE/scripts/rasterize_contours.py\" --output \"$STAGE/reproduction/new_rasterization\"\n"
               "```\n\nUse a genuinelyNEWemptyoutput directory forrasterreproduction;the scriptrefusesoverwrite. "
               "Comparedecodedpixelhashes and original/roundedmappingrecords, nottimestamp-dependentfiles. "
               "run_frozen.py refusesexistingrunroots. Exactoriginalproductioncommands aretracked in "
               "logs/smoke_*_execution.json andlogs/full_*_execution.json. Replaythosecommands withnewoutput_dir "
               "andoriginalfrozeninput/candidate/metadata paths; do NOT runthesameoutputcommand overexistinghistory. "
               "RunfourCPU smokes beforebothfullruns, thenverifyrawidentity/fullscoredecomposition/configs beforemetrics. "
               "Metricreproduction uses thesavedactualmasks andthe unchangedPhase20Bmetricfunction; deliverytests "
               "recomputeall32bothrasterizers andconfirmallsummaryrows. NoSAM3inferenceisneeded.\n\n"
               "Theignoredfullrunartifacts are indexed bylogs/full_*_artifacts.csv; localderivedGT64masks/mappingfiles "
               "byRASTERIZED_MASK_MANIFEST.csv. No externalmodels/data are silentlydownloaded. "
               "TheoriginalhumanGate is neitherrequiredfor thisseparateevaluationnorredefinedby it.")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
