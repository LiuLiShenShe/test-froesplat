# Notes: CompAg rewrite

## CompAg official requirements checked

Source: ScienceDirect Guide for Authors, Computers and Electronics in Agriculture.

- Scope: the journal emphasizes advances in computer hardware, software, electronic instrumentation and control systems for agricultural problems, including horticulture and controlled environments.
- Suitability: novelty and innovation in computers/electronics for agriculture are emphasized; merely applying existing technology to a crop is not enough.
- Article type: original research papers are accepted.
- File format: editable files are required; Word should use single-column layout, LaTeX may use double-column.
- Abstract: concise and factual, not exceeding 250 words, states purpose, principal results and major conclusions; avoid references and define uncommon abbreviations.
- Keywords: 1-7 keywords, written in English; avoid overly broad multi-word phrases where possible.
- Highlights: required; 3-5 bullet points; each maximum 85 characters including spaces.
- Graphical abstract: encouraged, separate file; 531 x 1328 pixels or proportionally larger.
- Tables: editable text, numbered in order, with captions, no vertical rules or cell shading.
- Figures: separate files, numbered in order, with captions; explain symbols and abbreviations.
- Research data: Option C applies; deposit data and cite/link it, or explain why it cannot be shared.
- Article sections: clearly defined and numbered; subsections should use 1.1, 1.1.1 etc.; abstract is not numbered.
- Acknowledgements: separate section directly before references.
- CRediT author contributions are required for corresponding authors.
- AI use declaration: required if AI tools were used in manuscript preparation.

## Reference article patterns

- IPENS opens from intelligent breeding and plant phenotyping, then names annotation and self-occlusion bottlenecks, then introduces NeRF-SAM2 fusion and reports phenotype metrics.
- LCR-GS opens from automated quantification in greenhouse scenes, then states the need to separate individual plants from shared scene-level reconstructions, then reports compact per-plant representations and trait validation.
- Plant3R opens from precision reconstruction for smart agriculture, then names low efficiency and high-quality-data dependency, then introduces a 3D reconstruction framework and validates phenotype traits.
- NeRF plant reconstruction paper starts from plant 3D geometry as a plant-science need, not only visual rendering, and emphasizes structural detail, cost and scalability.

## Manuscript rewrite stance

- The paper should read as a plant phenotyping paper with a computer/electronics innovation, not as a pure graphics paper.
- The central technical innovation is redefining 2DGS supervision around mask-defined plant foreground.
- The agricultural consequence is reliable low-cost, non-destructive, reusable structural trait measurement from ordinary RGB images under indoor/semi-controlled complex backgrounds.
- Boundary: evidence is 20 sequences and 21 plants; main ablations are representative-sample based; leaf width remains boundary-sensitive.
