"""Geometry RGB and RAP appearance RGB have explicit independent roles.

Restore only outside the existing binary foreground Mask from the exactly
matched original RGB. No resizing, crop, exposure change, segmentation, or model
invocation. The returned geometry RGB keeps every foreground RAP pixel intact;
the downstream appearance/training source stays the original RAP RGB file.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

def _source_id(path: Path) -> int:
    match=re.fullmatch(r'(?:crop_|mask_)?(\d+)',path.stem)
    if not match:raise ValueError(f'Require explicit numeric source name, not fuzzy pairing: {path.name}')
    return int(match[1])

def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

@dataclass
class GeometryRGBResult:
    geometry_rgb: np.ndarray
    foreground_mask: np.ndarray
    metadata: dict

def restore_geometry_rgb(
    rap_rgb_path: str | Path,
    mask_path: str | Path,
    matched_raw_rgb_path: str | Path,
    *,
    appearance_rgb_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> GeometryRGBResult:
    """Return exact original-grid context restoration and source-role metadata.

    appearance_rgb_path defaults to RAP RGB; if supplied separately it must
    decode to the same RAP RGB, enforcing unchanged downstream image content.
    output_path is optional and must be a new file outside all source paths.
    A missing/mismatched source raises; there is no fallback or resizing.
    """
    rap=Path(rap_rgb_path);mask_file=Path(mask_path);raw=Path(matched_raw_rgb_path)
    appearance=Path(appearance_rgb_path) if appearance_rgb_path is not None else rap
    for path in (rap,mask_file,raw,appearance):
        if not path.is_file():raise FileNotFoundError(path)
    ids=[_source_id(path) for path in (rap,mask_file,raw,appearance)]
    if len(set(ids))!=1:raise ValueError(f'Source-name pairing mismatch: {ids}')
    with Image.open(rap) as im:
        if im.mode!='RGB':raise ValueError('RAP RGB must have explicit RGB pixels; no silent alpha compositing')
        fg=np.array(im)
    with Image.open(raw) as im:original=np.array(im.convert('RGB'))
    with Image.open(mask_file) as im:mask_values=np.array(im.convert('L'))
    with Image.open(appearance) as im:appearance_rgb=np.array(im.convert('RGB'))
    if original.shape!=fg.shape or mask_values.shape!=fg.shape[:2] or appearance_rgb.shape!=fg.shape:
        raise ValueError('Raw/RAP/Mask/appearance dimensions differ; no geometry resampling is allowed')
    if not np.array_equal(appearance_rgb,fg):raise ValueError('Downstream appearance source must preserve the entire original RAP RGB')
    unique=set(np.unique(mask_values).tolist())
    if not unique.issubset({0,255}):raise ValueError(f'Require existing 0/255 binary RAP Mask compatible with original model preprocessors, got {sorted(unique)}')
    mask=mask_values>0
    restored=np.where(mask[...,None],fg,original).astype(np.uint8)
    changed=np.any(restored!=fg,axis=-1)
    if np.any(changed&mask) or not np.array_equal(restored[~mask],original[~mask]):raise AssertionError('Background-only restoration invariant violated')
    output=None
    if output_path is not None:
        output=Path(output_path)
        if output.suffix.lower()!='.png':raise ValueError('Use lossless .png to preserve exact RGB pixels')
        if output.resolve() in {p.resolve() for p in (rap,mask_file,raw,appearance)}:raise ValueError('Refuse to write a source image or Mask')
        if output.exists():raise FileExistsError(f'Refuse overwrite: {output}')
        output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(restored).save(output)
        with Image.open(output) as im:roundtrip=np.array(im.convert('RGB'))
        if not np.array_equal(roundtrip,restored):raise AssertionError('Export codec did not preserve exact RGB; use lossless PNG')
    metadata=dict(schema='foresplat.geometry_rgb_adapter.v1',source_index=ids[0],image_name=rap.name,
        geometry_rgb_source=dict(strategy='RAP pixels inside Mask + matched raw RGB pixels outside Mask',rap_rgb=str(rap),matched_raw_rgb=str(raw),
                                 prepared_geometry_rgb=str(output) if output is not None else None,decoded_rgb_sha256=hashlib.sha256(restored.tobytes()).hexdigest()),
        downstream_appearance_rgb_source=dict(path=str(appearance),role='unchanged RAP foreground RGB for later appearance/training; geometry context is not substituted here',
                                              decoded_rgb_sha256=hashlib.sha256(appearance_rgb.tobytes()).hexdigest()),
        foreground_mask_source=dict(path=str(mask_file),unchanged=True,binary_values=sorted(unique)),
        source_file_sha256={str(path):_file_sha(path) for path in dict.fromkeys([rap,mask_file,raw,appearance])},
        original_resolution_wh=[fg.shape[1],fg.shape[0]],geometry_to_appearance_pixel_transform=np.eye(3).tolist(),
        foreground_changed_pixels=int((changed&mask).sum()),background_changed_pixels=int((changed&~mask).sum()),
        foreground_pixels=int(mask.sum()),background_pixels=int((~mask).sum()),
        original_files_modified=False,resize=False,crop=False,exposure_compensation=False,mask_erosion=False,network_invoked=False,
        resampling_note='Original-grid foreground pixels are exact. Subsequent unchanged model RGB resampling can blend changed outside context at the boundary; Mask interpolation remains nearest in the model adapter.')
    if output is not None:output.with_suffix(output.suffix+'.metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return GeometryRGBResult(restored,mask,metadata)
