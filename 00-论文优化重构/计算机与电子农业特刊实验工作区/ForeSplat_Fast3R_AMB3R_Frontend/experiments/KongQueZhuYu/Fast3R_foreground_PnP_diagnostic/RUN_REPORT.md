# Foreground-gated Fast3R PnP diagnostic

The primary experiment and all saved raw confidence/pointmaps/cameras remain unchanged. Only a temporary confidence copy marks background pixels invalid for the official PnP validity test. Foreground scores are retained exactly; masks are used without erosion or further transforms. No GPU is used.

The original official PnP API runs with100 iterations and first-view/global-head focal estimation. Focal length is re-estimated from the same head using the temporary validity copy, so this control isolates foreground validity in the camera estimation procedure rather than only changing RANSAC correspondences.

| Diagnostic | Original official all-pixel PnP | Foreground-gated diagnostic |
| --- | --- | --- |
| pnp_success_rate | 1.0 | 1.0 |
| foreground_reprojection_p50_px | 27.38345724598863 | 20.379471305052636 |
| rotation_step_p90_deg | 99.25315399169925 | 110.9032989501953 |
| adjacent_point_distance_p50_normalized | 0.05512934923171997 | 0.04908839613199234 |
| adjacent_depth_agreement_fraction | 0.8234080948698118 | 0.806386213887481 |

A gain in camera/pointmap agreement would identify background sensitivity in PnP. It would not prove that raw global leaf geometry is correct. The raw whole-plant side/top pointmaps are identical in both cases, so any collapsed/banded raw geometry remains. This is not scanned-GT accuracy, bundle adjustment or a new foreground confidence calibration.

Sources: saved Fast3R bundle and official camera API in `third_party/fast3r/fast3r/models/multiview_dust3r_module.py`; no source network edits or pretrained weight changes.