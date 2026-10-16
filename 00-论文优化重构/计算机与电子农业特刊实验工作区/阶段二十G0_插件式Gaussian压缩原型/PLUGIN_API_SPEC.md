# Plugin API Specification

## Contract

```python
compress_model(input_model, output_dir, method, prune_ratio, config) -> CompressionResult
```

`input_model` is a frozen PLY or a complete 2DGS model directory. `output_dir` is a new, independent directory. `method` selects a registered scorer. `prune_ratio` is a decimal-like value in `[0, 1)`. The implementation removes `floor(N * Decimal(str(prune_ratio)))` Gaussians and retains exactly the remainder.

`config` must be JSON serializable. `enabled` is an explicit boolean. When it is false, the method and ratio are ignored for scoring and the output PLY is byte-identical to the source. `expected_source_sha256` can bind a run to a frozen asset. `active_sh_degree` is metadata for the native loader and cannot exceed the PLY's inferred maximum degree. `recovery=true` is rejected because no post-pruning training is performed in Phase 20G-0.

## Output invariants

- `pruned_model.ply` is independently written or byte-copied.
- `retained_indices.npy` and `removed_indices.npy` contain unique original row indices and form an exact partition.
- `importance_scores.npy` is finite and has one value per source Gaussian when pruning is enabled.
- All source scalar properties, including XYZ, opacity, two scales, quaternion, DC, rest SH, normals, and unknown numeric scalar properties, are copied by retained row.
- SH degree, PLY element/schema, source SHA256, output SHA256, timing, and capacity metadata are recorded.
- Input bytes are rehashed after compression; a source mutation fails the run.

## Error behavior

Existing output directories, invalid ratios, non-finite scores, invalid quaternion/scale activations, 3DGS three-scale PLYs, provenance mismatches, and blocked conference methods fail explicitly. A failed run retains `compression_config.json` with `status=FAILED` and traceback as negative evidence.
