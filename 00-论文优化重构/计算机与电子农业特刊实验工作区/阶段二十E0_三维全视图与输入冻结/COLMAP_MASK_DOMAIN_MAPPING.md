# COLMAP mask coordinate mapping — empirical verification

For each target pixel `(column+0.5,row+0.5)`, call the actual target PINHOLE
`cam_from_img`; append z=1 and project with the original OPENCV `img_from_cam`.
Subtract 0.5 from the source projection to address the image array. No assumed
OpenCV principal-point convention and no resolution-only mask resize are used.
Mask sampling is nearest neighbor with invalid pixels zero. RGB is direct bilinear
interpolation with COLMAP color rounding. Maps use float64 and have explicit SHA256.

Native `pycolmap.undistort_camera` must reproduce the exact historical output
dimensions/intrinsics. Two original RGB views per scene are compared against
`pycolmap.undistort_image` with direct warping and bilinear sampling. All original
and undistorted sparse image observations are converted with the same official
camera functions; actual 3D points are reprojected using unchanged image poses.
Damaged registered input 0136 is independently matched to its original COLMAP
observations with real CPU SIFT and neighboring negative controls; its conflicting
historical derived RGB is rejected instead of using a wrong color reference.

Existing historical images may be foreground-replaced/JPEG-encoded, and several
are unreadable. They are compared and recorded, not assumed equal to raw color.
The canonical raw RGB version is generated for all groups from verified original
files on the exact model grid. Existing files/models are never overwritten.

Validation: per-scene `reports/*_SPARSE_PROJECTION.json`, `*_RGB_COMPARISONS.csv`,
`*_BUILD.json`; exact mapping/source identities in `CAMERA_REMAP_MANIFEST.csv`.
Official sources fetched via Context7 and direct source retrieval:
https://github.com/colmap/colmap/blob/main/src/colmap/image/warp.cc
https://github.com/colmap/colmap/blob/main/src/colmap/image/undistortion.cc
https://github.com/colmap/colmap/blob/main/python/README.md
