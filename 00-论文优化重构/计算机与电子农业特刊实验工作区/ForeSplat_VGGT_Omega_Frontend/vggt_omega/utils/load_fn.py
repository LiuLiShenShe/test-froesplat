# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Image / mask loading and preprocessing for VGGT-Omega.

The original community functions (``load_and_preprocess_images`` and its
private helpers) are kept byte-for-byte identical so that the released
checkpoint sees exactly the input distribution it was trained on.

ForeSplat extension (added by the ForeSplat VGGT-Omega frontend adapter):

``load_and_preprocess_images_with_masks`` loads RAP-FSAM3 foreground RGB
images *and* their binary foreground masks and applies **exactly the same
geometric transform** to both (same center crop, same resize scale, same
padding, same patch alignment).  The RGB branch keeps BICUBIC resampling
while the binary mask branch uses NEAREST resampling and zero padding.

For every frame an auditable record is produced that stores the crop /
resize / padding geometry and the affine pixel transform

    p_model = A @ p_original

together with its inverse, so that camera intrinsics and 2D observations can
be mapped back to the original image resolution later on.
"""

import os
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=512, patch_size=16):
    """Load images for VGGT-Omega inference.

    `balanced` keeps the total token count close to image_resolution**2.
    `max_size` resizes the longest side to image_resolution.
    Both modes first center-crop extreme aspect ratios into [0.5, 2.0].
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["balanced", "max_size"]:
        raise ValueError("Mode must be either 'balanced' or 'max_size'")
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        image = _crop_to_supported_aspect_ratio(_load_rgb_image(image_path))
        width, height = image.size
        aspect_ratio = height / max(width, 1)

        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
        else:
            target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)

        print(f"Resizing image {image_path} from ({width}, {height}) to ({target_w}, {target_h})")
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        image = to_tensor(image)

        shapes.add((image.shape[1], image.shape[2]))
        images.append(image)

    if len(shapes) > 1:
        warnings.warn(f"Found images with different shapes: {shapes}; padding to a common size.", stacklevel=2)
        images = _pad_images_to_common_size(images, shapes)

    return torch.stack(images)


def _load_rgb_image(image_path):
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _crop_to_supported_aspect_ratio(image, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    width, height = image.size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height))

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height))

    return image


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio, image_resolution, patch_size):
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _round_to_patch_multiple(value, patch_size):
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _pad_images_to_common_size(images, shapes):
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded_images = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_images.append(image)

    return padded_images


# ---------------------------------------------------------------------------
# ForeSplat extension: mask-aware preprocessing
# ---------------------------------------------------------------------------

def _crop_box_for_supported_aspect(image_size, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    """Return ``((left, top, right, bottom), crop_applied)``.

    Mimics the arithmetic of :func:`_crop_to_supported_aspect_ratio` exactly so
    that RGB image and binary mask always share the *same* crop region.
    """
    width, height = image_size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return (left, 0, left + crop_width, height), True

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return (0, top, width, top + crop_height), True

    return (0, 0, width, height), False


def _load_foreground_rgb_image(image_path, background="black"):
    """Load a RAP-FSAM3 foreground image as RGB, compositing RGBA over ``background``.

    RAP-FSAM3 writes its foreground composites over a *black* background (and
    its RGBA debug export keeps the un-composited RGB under ``alpha = mask``),
    so "black" reproduces the original RAP-FSAM3 RGB background form.  The
    community loader composites RGBA over white instead - it must NOT be used
    for RAP-FSAM3 outputs.
    """
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            fill = (0, 0, 0, 255) if background == "black" else (255, 255, 255, 255)
            canvas = Image.new("RGBA", image.size, fill)
            image = Image.alpha_composite(canvas, image)
        return image.convert("RGB")


def _load_binary_mask_image(mask_path, mask_threshold=127):
    """Load a mask as a 0/255 binary 8-bit image."""
    with Image.open(mask_path) as mask:
        if mask.mode not in ("L", "1"):
            mask = mask.convert("L")
        binary = (np.asarray(mask) > mask_threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def _pixel_transform(crop_left, crop_top, target_w, target_h, cropped_width, cropped_height, pad_left, pad_top):
    """Affine ``A`` with ``p_model = A @ p_original`` (homogeneous pixel indices).

    Derivation, matching PIL's pixel-centre convention (output pixel centre
    ``i + 0.5`` samples input coordinate ``(i + 0.5) / scale``, scale =
    target/cropped)::

        x_crop        = x_orig - crop_left
        x_resized     = (x_crop + 0.5) * s_x - 0.5
        x_model       = x_resized + pad_left

    hence::

        x_model = s_x * x_orig + (pad_left - s_x * crop_left + 0.5 * s_x - 0.5)
    """
    s_x = float(target_w) / float(cropped_width)
    s_y = float(target_h) / float(cropped_height)
    t_x = float(pad_left) - s_x * float(crop_left) + 0.5 * s_x - 0.5
    t_y = float(pad_top) - s_y * float(crop_top) + 0.5 * s_y - 0.5
    return np.array([[s_x, 0.0, t_x], [0.0, s_y, t_y], [0.0, 0.0, 1.0]], dtype=np.float64)


def _invert_pixel_transform(pixel_transform):
    A = np.asarray(pixel_transform, dtype=np.float64).reshape(3, 3)
    return np.linalg.inv(A)


def load_and_preprocess_images_with_masks(
    image_path_list,
    mask_path_list,
    mode="balanced",
    image_resolution=512,
    patch_size=16,
    background="black",
    mask_threshold=127,
):
    """Load RAP-FSAM3 foreground images + binary masks for VGGT-Omega inference.

    The RGB branch reproduces :func:`load_and_preprocess_images` exactly
    (identical center crop, identical BICUBIC resize, identical 1.0 padding,
    identical patch-size alignment).  The mask branch applies the *same*
    geometry but uses NEAREST resampling (never bicubic on a binary mask) and
    zero padding.

    Returns ``(images, masks, records)``:

    * ``images``     (N, 3, H_model, W_model) float32 in [0, 1] - model input;
    * ``masks``      (N, H_model, W_model) bool - foreground at model resolution;
    * ``records``    one dict per frame with crop / resize / pad bookkeeping,
      ``model_width`` / ``model_height``, and the affine ``pixel_transform``
      (3x3) with ``p_model = A p_original``.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mask_path_list is None or len(mask_path_list) != len(image_path_list):
        raise ValueError(
            f"mask_path_list must provide exactly one mask per image "
            f"({None if mask_path_list is None else len(mask_path_list)} masks "
            f"for {len(image_path_list)} images)"
        )
    if mode not in ("balanced", "max_size"):
        raise ValueError("Mode must be either 'balanced' or 'max_size'")
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")
    if background not in ("black", "white"):
        raise ValueError("background must be 'black' or 'white'")

    records = []
    rgb_tensors = []
    mask_tensors = []
    shapes = set()

    for image_path, mask_path in zip(image_path_list, mask_path_list):
        original_image = _load_foreground_rgb_image(image_path, background=background)
        original_mask = _load_binary_mask_image(mask_path, mask_threshold=mask_threshold)
        width, height = original_image.size
        if original_mask.size != original_image.size:
            raise ValueError(
                f"Mask {mask_path} has size {original_mask.size} but image "
                f"{image_path} has size {original_image.size}"
            )

        # --- identical crop box for RGB and mask ----------------------------
        crop_box, crop_applied = _crop_box_for_supported_aspect(original_image.size)
        image = original_image.crop(crop_box)
        mask = original_mask.crop(crop_box)
        crop_left, crop_top = crop_box[0], crop_box[1]
        cropped_width, cropped_height = image.size
        aspect_ratio = cropped_height / max(cropped_width, 1)

        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
        else:
            target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)

        print(
            f"Resizing image {image_path} from ({cropped_width}, {cropped_height}) to "
            f"({target_w}, {target_h}) (center-crop {crop_box} from original ({width}, {height}))"
        )
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        mask = mask.resize((target_w, target_h), Image.Resampling.NEAREST)

        image_tensor = TF.ToTensor()(image)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8) > 0)
        if mask_tensor.shape != (target_h, target_w):
            raise RuntimeError(
                f"mask preprocessing produced {tuple(mask_tensor.shape)}, expected {(target_h, target_w)}"
            )

        shapes.add((image_tensor.shape[1], image_tensor.shape[2]))
        rgb_tensors.append(image_tensor)
        mask_tensors.append(mask_tensor)
        records.append(
            {
                "image_name": os.path.basename(str(image_path)),
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "original_width": int(width),
                "original_height": int(height),
                "crop_left": int(crop_left),
                "crop_top": int(crop_top),
                "crop_width": int(cropped_width),
                "crop_height": int(cropped_height),
                "crop_applied": bool(crop_applied),
                "resized_width": int(target_w),
                "resized_height": int(target_h),
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
                "model_width": int(target_w),
                "model_height": int(target_h),
                "preprocess_mode": mode,
                "image_resolution": int(image_resolution),
                "patch_size": int(patch_size),
                "rgb_resize_filter": "bicubic",
                "mask_resize_filter": "nearest",
                "mask_threshold": int(mask_threshold),
                "foreground_background": background,
                "pixel_transform": _pixel_transform(
                    crop_left=crop_left,
                    crop_top=crop_top,
                    target_w=target_w,
                    target_h=target_h,
                    cropped_width=cropped_width,
                    cropped_height=cropped_height,
                    pad_left=0,
                    pad_top=0,
                ).tolist(),
            }
        )

    if len(shapes) > 1:
        warnings.warn(
            f"Found images with different shapes: {shapes}; padding to a common size.", stacklevel=2
        )
        rgb_tensors, mask_tensors, records = _pad_frames_to_common_size(rgb_tensors, mask_tensors, records, shapes)

    for record in records:
        record["pixel_transform_inverse"] = _invert_pixel_transform(record["pixel_transform"]).tolist()
        A = np.asarray(record["pixel_transform"], dtype=np.float64)
        record["scale_x"] = float(A[0, 0])
        record["scale_y"] = float(A[1, 1])
        record["valid_left"] = int(record["pad_left"])
        record["valid_top"] = int(record["pad_top"])
        record["valid_width"] = int(record["resized_width"])
        record["valid_height"] = int(record["resized_height"])

    images = torch.stack(rgb_tensors)
    masks = torch.stack(mask_tensors)
    return images, masks, records


def _pad_frames_to_common_size(rgb_tensors, mask_tensors, records, shapes):
    """Pad RGB with 1.0 (white, community behaviour) and masks with 0 (background)."""
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded_rgb, padded_masks = [], []
    for image, mask, record in zip(rgb_tensors, mask_tensors, records):
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left

        if h_padding > 0 or w_padding > 0:
            image = torch.nn.functional.pad(
                image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
            mask = torch.nn.functional.pad(
                mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=False
            )

        record.update(
            {
                "pad_left": int(pad_left),
                "pad_top": int(pad_top),
                "pad_right": int(pad_right),
                "pad_bottom": int(pad_bottom),
                "model_width": int(image.shape[2]),
                "model_height": int(image.shape[1]),
                "pixel_transform": _pixel_transform(
                    crop_left=record["crop_left"],
                    crop_top=record["crop_top"],
                    target_w=record["resized_width"],
                    target_h=record["resized_height"],
                    cropped_width=record["crop_width"],
                    cropped_height=record["crop_height"],
                    pad_left=pad_left,
                    pad_top=pad_top,
                ).tolist(),
            }
        )
        padded_rgb.append(image)
        padded_masks.append(mask)

    return padded_rgb, padded_masks, records


def pixel_transform_to_matrix(pixel_transform):
    """Normalise a stored transform (nested list / ndarray) into a 3x3 matrix."""
    return np.asarray(pixel_transform, dtype=np.float64).reshape(3, 3)


def pixel_transform_inverse(pixel_transform):
    """Return ``A^-1`` of a stored ``pixel_transform``."""
    return np.linalg.inv(pixel_transform_to_matrix(pixel_transform))


def apply_pixel_transform(pixel_transform, xy):
    """Apply ``A`` to ``(..., 2)`` pixel coordinates."""
    A = pixel_transform_to_matrix(pixel_transform)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float64)], axis=1)
    mapped = homogeneous @ A.T
    return (mapped[:, :2] / mapped[:, 2:3]).reshape(xy.shape)


def transform_intrinsics_model_to_original(intrinsics_model, pixel_transform):
    """Map a model-resolution intrinsic matrix back to the original resolution.

    ``p_original = A^-1 p_model`` and ``p_model = K_model X`` give
    ``K_original = A^-1 K_model`` (the pixel-centre offsets of the PIL
    crop/resize/pad pipeline are folded into ``A``).
    """
    A_inv = pixel_transform_inverse(pixel_transform)
    K = np.asarray(intrinsics_model, dtype=np.float64).reshape(3, 3)
    return A_inv @ K


def original_to_model_coordinates(pixel_transform, xy):
    """Forward transform: map original-resolution pixel coordinates to model input pixels."""
    return apply_pixel_transform(pixel_transform, xy)


def model_to_original_coordinates(pixel_transform, xy):
    """Map model-resolution pixel coordinates back to original-resolution pixels (``A^-1``)."""
    A_inv = pixel_transform_inverse(pixel_transform)
    return apply_pixel_transform(A_inv.tolist(), xy)


