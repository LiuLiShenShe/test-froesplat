# ForeSplat Phase20D-Final — fixed-contour direct evaluation

Date:2026-10-12. Startingmain=d3b474b9, initially clean; previousprotocolwork was committed/localfiles were preserved. This stage executes the user's fixed-contour decision, not anotherGT preparation loop.

## Final validation evidence

- `logs/TEST_FINAL.txt`: **43 tests passed**, including independent per-sequence,
  overlapping difficulty, cross-raster area/IoU and frozen score decomposition checks.
- `logs/DELIVERY_VERIFICATION.json`: **PASS**; 4756 protected sources unchanged,
  64 raster masks reproduced, 128 full metric rows recomputed, 108 aggregate and
  288 per-sequence rows verified, 142 oracle candidate evaluations retained,
  32 real comparison figures and 532 registered Phase20E views audited.
- `logs/SOURCE_PROTECTION_FINAL.json`: **PASS**. Original JSON, RGB, candidate
  masks/metadata, production ranking, historical results and GT Gate remain unchanged.
- `CODE_SHA256_MANIFEST.csv` and `logs/RUNTIME_ENVIRONMENT.json` record actual
  delivered script hashes and installed runtime versions. All 16 Python files parse.
- `VISUAL_QA.md` records direct native image inspection and correction of earlier
  external-description errors. Neither machine inspection nor this evaluation is
  a substitute for archived human signatures. No new external vision call is used.

Repository delivery covers this new stage only. The separately user-authorized
vision-rule change is in `/home/test/.codex/AGENTS.md`; that credential-containing
global file is not added to the project repository.

```text
PHASE20D_DIRECT_EVALUATION: PASS
RASTERIZED_MASKS: 32/32
GEOMETRY_AMBIGUITY: 13
MAIN_GT_GEOMETRY_AMBIGUITY: 10
TEST32_R0_F1: 0.8888725908186376
TEST32_R1_F1: 0.9504480538376888
TEST32_R1_MINUS_R0: 0.061575463019051174
SWITCH_CASES: IMPROVED
RASTERIZATION_SENSITIVITY: STABLE
FOREGROUND_METHOD: ENGINEERING_LOCKED
PHASE20E_PILOT_READY: CONDITIONAL
```

## Complete primary32-frame results

| method | frames | sequences | f1 | iou_fg | miou | hd95_px | boundary_f1 | leakage_energy | oracle_regret | empty_mask_count | severe_f1_lt_0_5_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 32 | 8 | 0.88887259 | 0.87222299 | 0.88821850 | 390.60436147 | 0.24737138 | 0.00747145 | 0.09242488 | 0 | 3 |
| R1 | 32 | 8 | 0.95044805 | 0.93290547 | 0.93199853 | 227.26327252 | 0.26349097 | 0.00666420 | 0.03084942 | 0 | 1 |
| R1_MINUS_R0 | 32 | 8 | 0.06157546 | 0.06068247 | 0.04378003 | -163.34108895 | 0.01611960 | -0.00080725 | -0.06157546 | 0 | -2 |

Main=sample_equal_mean;8×4views makes frame_equal_mean identical here. R1 improves2frames, regresses0, ties30. Bothtrue switches improve;30ties are measured, not assumedcorrect. Severeproxy3→1, empty0→0. WanNianQing1_0223 remains F1=0/oracle regret≈0.98718, and it is explicitly present inalltables andfigures. No badframe is deleted. Median/populationSD, pairedmeans,8sequences,13/19subsets andbothrasterizers are separateCSVrows.

## Eight sequences

| sample | method | frames | f1 | iou_fg | miou | hd95_px | boundary_f1 | leakage_energy | oracle_regret | severe_f1_lt_0_5_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CaoMei2 | R0 | 4 | 0.73920744 | 0.72869645 | 0.83683396 | 402.42270446 | 0.31834805 | 0.00916404 | 0.24635121 | 1 |
| CaoMei2 | R1 | 4 | 0.98555865 | 0.97154056 | 0.98384002 | 17.08272958 | 0.40012956 | 0.00109330 | 0.00000000 | 0 |
| CaoMei2 | R1_MINUS_R0 | 4 | 0.24635121 | 0.24284411 | 0.14700606 | -385.33997488 | 0.08178151 | -0.00807075 | -0.24635121 | -1 |
| ChangShouHua3 | R0 | 4 | 0.98592971 | 0.97225994 | 0.96973683 | 132.84876251 | 0.20262566 | 0.00961842 | 0.00000000 | 0 |
| ChangShouHua3 | R1 | 4 | 0.98592971 | 0.97225994 | 0.96973683 | 132.84876251 | 0.20262566 | 0.00961842 | 0.00000000 | 0 |
| ChangShouHua3 | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |
| DouBanLv3 | R0 | 4 | 0.98062269 | 0.96200410 | 0.96768582 | 79.97158051 | 0.23055923 | 0.01024547 | 0.00000000 | 0 |
| DouBanLv3 | R1 | 4 | 0.98062269 | 0.96200410 | 0.96768582 | 79.97158051 | 0.23055923 | 0.01024547 | 0.00000000 | 0 |
| DouBanLv3 | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |
| HongZhang | R0 | 4 | 0.97290406 | 0.94727433 | 0.91372420 | 292.88723755 | 0.16746807 | 0.00984637 | 0.00000000 | 0 |
| HongZhang | R1 | 4 | 0.97290406 | 0.94727433 | 0.91372420 | 292.88723755 | 0.16746807 | 0.00984637 | 0.00000000 | 0 |
| HongZhang | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |
| WanNianQing1 | R0 | 4 | 0.49234741 | 0.48494510 | 0.49873401 | 2012.67795563 | 0.08002471 | 0.01508186 | 0.49304786 | 2 |
| WanNianQing1 | R1 | 4 | 0.73859991 | 0.72756078 | 0.70196820 | 1091.28921890 | 0.12719996 | 0.01669464 | 0.24679537 | 1 |
| WanNianQing1 | R1_MINUS_R0 | 4 | 0.24625249 | 0.24261568 | 0.20323419 | -921.38873672 | 0.04717525 | 0.00161277 | -0.24625249 | -1 |
| WangWenCao2 | R0 | 4 | 0.98855807 | 0.97737543 | 0.98720017 | 13.41387486 | 0.49048249 | 0.00095073 | 0.00000000 | 0 |
| WangWenCao2 | R1 | 4 | 0.98855807 | 0.97737543 | 0.98720017 | 13.41387486 | 0.49048249 | 0.00095073 | 0.00000000 | 0 |
| WangWenCao2 | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |
| XianKeLai3 | R0 | 4 | 0.97714560 | 0.95536350 | 0.96839720 | 119.25399017 | 0.28280591 | 0.00200180 | 0.00000000 | 0 |
| XianKeLai3 | R1 | 4 | 0.97714560 | 0.95536350 | 0.96839720 | 119.25399017 | 0.28280591 | 0.00200180 | 0.00000000 | 0 |
| XianKeLai3 | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |
| XiangPiShu2 | R0 | 4 | 0.97426573 | 0.94986509 | 0.96343577 | 71.35878611 | 0.20665690 | 0.00286290 | 0.00000000 | 0 |
| XiangPiShu2 | R1 | 4 | 0.97426573 | 0.94986509 | 0.96343577 | 71.35878611 | 0.20665690 | 0.00286290 | 0.00000000 | 0 |
| XiangPiShu2 | R1_MINUS_R0 | 4 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0 |

## Provenance and protocol limits

32originalJSON files read completely, all75originalshape sequences preserved. MainGT uses only potted_plant;0blue_cube labels means no inventedsubtraction. No plant/pot requirement, no approval fields fabricated. PrimaryOpenCVfillPoly and secondaryPillow use the predeclaredhalf-away rounding. All32primary and32secondary masks have source/RGB/output/mappinghashes. No empty surrogateGT. 13frames/17shapes have numericalself-intersection/backtrackingwarnings;10frames affect mainGT. All remain inmainstatistics. STABLE undertworasterizers does not certify annotationsemantics.

The originalPhase20DGate is unchanged andhistoricallyBLOCKED; itwasnotrerun oroverridden. This is **Fixed-contour rasterization evaluation**, or **Evaluation using deterministically rasterized manual contours**, notdual-signaturestandardhumanGT. Userstatement ofcompletedhumanreview is retained; there are no newhumanapprovals. Difficultylabels remainexistinggroupvariables.

FourrealCPU smoke runs passedfirst, thenrealR0/R1productionCLI full32runs. Both use71originalP6candidates, originalmasks andstrictmetadata; onlyranking_policy differs. RealID/mask equality, fullscores, runtimeparameters, codeSHA256, sourcecandidateSHA256, outputSHA256 andstdoutlogs are retained. Finalmasks equalrawselectedmasks underthisfrozenconfiguration. NoSAM3loaded/reinference, optimizer, GT-basedrank changes oroptionalmodule. OriginalarchivedSAMconfidence precision6decimals; unsavedtensorbitsarenotinvented.

HistoricalGT10threerepairs are recomputedseparately and excludedfromTEST32tables. Themethoddecision isengineering-onlyENGINEERING_LOCKED withoneknownseriouserror; no universalaccuracyclaim. Phase20E3plants areCONDITIONALduefullviewcandidatecoverage, historicalRGBAcoverage/coordinateandRGBreadabilityissues, notmissinghuman signatures. Exactcamera grids differ; no2160×3840→2119×3779resizing. No3Dtrainingorpapermodification.

## Delivery and verification

Requested reports/CSVs,32comparisonfigures,2switchPDFs,sourceindices,scripts/tests/logs and full-view3Dpreparationmanifests are inthisdirectory. Large derivedmasks/mappings/productionoutputs remainlocalignored; SHA256 andreproductioncommandsaretracked. Verificationresults are inlogs/ TEST_FINAL.txt, SOURCE_PROTECTION_FINAL.json and DELIVERY_VERIFICATION.json; readtheiractualPASS/FAIL rather than interpretingthisreport as ahumanGTGatePASS. Commit/pushreceipt is separate.

Sources: preservedlocalJSON/RGB/candidate archives, actualproductionruns, frozenPhase20B/Cmetrics, Phase20C-FIXmethoddecision andnewcompleteCSVtables. BochaHTTP403; no externalexperimentdata wasretrieved. CurrentOpenCV/Pillow/COLMAPdocumentation wasretrievedviaContext7; seeprotocolreferences.
