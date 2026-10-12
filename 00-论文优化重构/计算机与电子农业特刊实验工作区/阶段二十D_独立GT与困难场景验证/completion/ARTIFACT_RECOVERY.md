# Completion artifact recovery

Public: scripts/tests, explicit blocked result schemas, geometry/source/backup indexes,
objective review CSV updates, reports and actual small preflight/protection logs. Local
ignored: original RGB,71 candidate masks,32 existing manual JSONs, canonical JSON links,
the byte-exact local_gt_backup/ files and any future private reviewer identity mapping.
No personal identity, RGB, SAM checkpoint or mask cache is uploaded in this completion.

`GT32_SOURCE_ARCHIVE.csv` identifies every current manual source and SHA256 before/after.
`GT32_LOCAL_BACKUP_INDEX.csv` records immutable-content local backups (682234 bytes in
total). Backups are exact copies of existing human exports, not new or approved GT;
restore from the original experiment archive or these local paths and verify SHA256.
Never overwrite an existing JSON/link with a different version. A real person's corrected
export must be stored as a new version with recorded provenance and actual re-approval.

32 canonical `annotation_package/annotations/<frame_id>.json` paths are local symlinks
to the original human files beside RGB. To relocate a checkout, restore original bytes,
then make the links only if the destinations are absent or already identify the same
source. No conversion of linestrip to polygon, point closing, label fill or GT fabrication
is part of recovery. Full raw-source recovery remains in the parent SOURCE_HASHES.csv,
TEST32_INPUT_MANIFEST.csv, TEST32_CANDIDATE_PROVENANCE.csv and Phase20C-FIX ARTIFACT_RECOVERY.md.

Commands executed in this completion, from repository root:

```bash
STAGE='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十D_独立GT与困难场景验证'
PYTHON=/home/test/biosoft/enter/bin/python
$PYTHON "$STAGE/completion/scripts/archive_existing_gt.py" \
  --patch-output "$STAGE/completion/logs/objective_review_archive.patch"
apply_patch < "$STAGE/completion/logs/objective_review_archive.patch"
$PYTHON "$STAGE/scripts/validate_annotations.py" \
  --output "$STAGE/completion/logs/annotation_gate_completion.json"
$PYTHON "$STAGE/completion/scripts/document_blocked_completion.py"
$PYTHON "$STAGE/completion/scripts/backup_manual_exports.py"
$PYTHON "$STAGE/completion/scripts/verify_completion.py" \
  --output "$STAGE/completion/logs/FINAL_PROTECTION_CHECK.json"
```

These creation scripts use exclusive output creation. Do not blindly rerun them into
existing artifacts; use an isolated new output/archive version and retain originals.
Original gate exit2 is expected; it is NOT a successful independent evaluation. Future
gate rechecks must use fresh output names. No prediction/metric/figure/3D artifact exists.

Review-table before state is recoverable from22073ca5, but do not reset user's real
edits. All human entries must be supplied by actual people; backup hashes and checked
flags cannot create an approval. Private identities stay outside public Git.
