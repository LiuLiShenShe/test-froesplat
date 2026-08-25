# Fig.4 text labels

## Title

Fig. 4 | Prompt robustness and semantic-bias analysis

## Column headers

| Column | Header | Prompt text | Metric |
|---:|---|---|---|
| 1 | Raw image |  |  |
| 2 | Green-region prompt | green plant | F1 = 0.7972 |
| 3 | Plant-without-container prompt | entire plant excluding pot | F1 = 0.9692 |
| 4 | Organ-level prompt | leaves and stems | F1 = 0.6471 |
| 5 | Seedling-oriented prompt | crop seedling | F1 = 0.0768 |
| 6 | Background-excluding plant prompt | plant body without background | F1 = 0.4051 |
| 7 | RAP-FSAM3 output |  |  |

## Row labels

- DouBanLv1
- ChangShouHua2
- CaoMei1

## Cell annotations

Use red labels for single-prompt failure modes:

- bg leakage
- missing structure
- missing flowers
- mature-plant bias
- under-seg.

Use green labels for RAP-FSAM3 output:

- stable prior
- improved boundary

## Bottom sentence

Direct single-prompt VFM segmentation is semantically unstable across agricultural plant structures; RAP-FSAM3 provides a stable foreground prior.

## Caption draft

Prompt robustness and semantic-bias analysis of direct VFM prompting. Three representative agricultural plant structures were segmented with five single text prompts and compared with the RAP-FSAM3 foreground prior. F1 values in the column headers are computed on the FourSample_GT20 prompt-robustness benchmark. The green-region prompt tends to include semantically plausible but irrelevant green background, organ-level prompts miss complete plant structures, seedling-oriented prompts fail on mature potted plants, and background-excluding prompts are often overly conservative. RAP-FSAM3 mitigates these prompt-induced biases by selecting and refining a stable foreground prior.

## Short Fig.3 distinction

Fig. 3 compares different VFM methods, whereas Fig. 4 isolates prompt sensitivity within the same VFM.
