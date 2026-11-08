# AMB3R saved depth-channel diagnostic

These supplementary channels retain official world-frame depth unprojection and saved cameras. No new fusion, fit, metric conversion or primary-result replacement is performed. Separate native depth-confidence and same-world-confidence controls distinguish channel support from point-head confidence changes. Algebraic self-projection is not3D accuracy.

| Method | Confidence field | Foreground coverage | Edge coverage | Adjacent depth agreement | Normalized point distance |
| --- | --- | --- | --- | --- | --- |
| AMB3R_Base_OFF | depth_conf | 0.9981440774308256 | 0.9960437655900742 | 0.6791290057518488 | 0.032392699271440506 |
| AMB3R_Base_OFF | world_conf | 0.2067651596663165 | 0.08106838575917587 | 0.8470805588508228 | 0.029372352175414562 |
| AMB3R_Base_ON | depth_conf | 1.0 | 1.0 | 0.7483852025836759 | 0.0263943150639534 |
| AMB3R_Base_ON | world_conf | 0.9481712590962934 | 0.9063178269461272 | 0.747754643194047 | 0.02639830205589533 |