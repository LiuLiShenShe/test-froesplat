# V1 → V2 differences and standalone Methods suggestion

This file does not edit the manuscript or overwrite any earlier findings/metrics.

| item | V1 requirement | explicit V2 correction |
| --- | --- | --- |
| target region | required whole potted-plant label, visibility unstated | required **visible** target-instance foreground in every frame |
| pot polygon | required in all32 | required only for human-confirmed VISIBLE/PARTIALLY_VISIBLE pots |
| absent pot | geometry rejection | visibility UNVERIFIED until real human decision; NOT_VISIBLE permits absence |
| absent plant | rejection | existing potted_plant may equal plant only with explicit plant-only/NOT_VISIBLE consent and recorded derivation |
| linestrip | rejected without conversion | preserve originals; audit and independently human-confirm closed exterior; only shape-type/implicit closure conversion |
| GT formula | potted_plant AND NOT blue_cube | identical |
| frozen frames/RGB/candidates |32/8/71, hashes | identical; original input verifier reused |
| human review | real annotator, independent second reviewer, identity and evidence | retained; add visibility, outline and derivation provenance and final hash binding |
| production algorithm/results | frozen | unchanged, independent evaluation not executed |

The earlier completion report is an accurate record of what **V1** required, but “missing
pot/plant” must not be described as proven annotation negligence. The user's visibility
explanation corrects that protocol assumption. V2 does not authorize hallucinated hidden
parts, GT filling from predictions, automatic category evidence, changed ranking or posthoc
TEST32 tuning. The independent-study report must later name V2 and disclose this correction.

## Suggested Methods wording — conditional draft only

Use this text only after the actual corresponding records and Gate are complete. No
verified human-review claim is currently made by the draft:

> Ground-truth annotations follow a visible-region protocol for the designated target
> potted-plant instance. The potted_plant region includes only its observed foreground;
> occluded or out-of-view plant and pot regions are not reconstructed. Plant tissue and
> the pot are recorded when visible, with pot visibility separately documented. For
> explicitly human-confirmed plant-only views with no visible pot, a missing plant mask
> may be obtained by exact duplication of the existing manual potted_plant region, with
> the source, consent and deterministic derivation retained. Legacy linestrip outlines
> are converted to polygon regions only after individual human confirmation of their
> intended closed boundary; all vertex coordinates remain unchanged. The primary
> evaluation region is potted_plant excluding a separately annotated visible blue cube.
> Source and derived annotation hashes, target identity, visual difficulty evidence and
> independent human review records are retained before frozen-method evaluation.

Do not add F1 values, universal generalization, final method lock or “all annotations
verified” until actual independent results exist. Mention no-GT prior image behavior
checks and8 correlated sequences in the later evaluation limitations, not a fake unseen
external-data claim. No original manuscript text was edited here.
