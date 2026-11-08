# Omega: completed 208-view geometry inference and export

Actual new inference on the exact 208 historical frame IDs and order, with restored original background. Existing audit geometry_rgb_adapter preserves RAP foreground pixels inside the unchanged mask. Training RGB remains the historical RAP files. Physical GPU1, uv, seed0.

Original preprocessing: balanced. Official adapter call 85.423144s; peak Torch allocation 21.679773GiB.

Camera/K/Depth/Confidence and full depth-unprojected Point Maps are saved. Unfiltered PLY: 54,951,936; foreground/confidence-filtered PLY: 13,640,966. Original and corrected COLMAP datasets share the same initialization PLY and RGB/Mask files. Existing frontend fusion/sampling is preserved; no learned point head was introduced.

See metadata.json, export_complete.json, cameras.json, pixel_transforms.json and shared_manifest.csv.
