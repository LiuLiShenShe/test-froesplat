# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from dvlt.util.download import download_file_from_url


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")


_SKYSEG_ONNX_PATH = "skyseg.onnx"
_SKYSEG_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


def _load_skyseg_session(
    intra_op_num_threads: Optional[int] = 4,
    inter_op_num_threads: Optional[int] = 1,
) -> "onnxruntime.InferenceSession":
    """Load the skyseg ONNX model, downloading on first use.

    Defaults pin a small thread pool to avoid ORT affinity warnings when
    sessions share the host with other ORT users (e.g. a Gradio demo). Set
    either kwarg to ``None`` to let ORT auto-configure.
    """
    if not os.path.exists(_SKYSEG_ONNX_PATH):
        print(f"Downloading {_SKYSEG_ONNX_PATH}...")
        download_file_from_url(_SKYSEG_URL, _SKYSEG_ONNX_PATH)

    sess_opts = onnxruntime.SessionOptions()
    if intra_op_num_threads is not None:
        sess_opts.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        sess_opts.inter_op_num_threads = inter_op_num_threads

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return onnxruntime.InferenceSession(_SKYSEG_ONNX_PATH, sess_options=sess_opts, providers=providers)


def apply_sky_segmentation(
    conf: np.ndarray,
    images: torch.Tensor,
    intra_op_num_threads: Optional[int] = 4,
    inter_op_num_threads: Optional[int] = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Multiply ``conf`` by a non-sky binary mask predicted by skyseg.

    Args:
        conf: (S, H, W) confidence scores.
        images: (S, 3, H, W) RGB images in [0, 1].
        intra_op_num_threads, inter_op_num_threads: ORT thread-pool sizes.
            Defaults (4, 1) avoid affinity warnings when sharing the host
            with other ORT sessions; set either to ``None`` for auto.
        verbose: Print download / inference progress.

    Returns:
        Updated ``conf`` with sky pixels zeroed.
    """
    S, H, W = conf.shape
    skyseg_session = _load_skyseg_session(
        intra_op_num_threads=intra_op_num_threads,
        inter_op_num_threads=inter_op_num_threads,
    )
    if verbose:
        print(f"ONNX Runtime using providers: {skyseg_session.get_providers()}")
        print("Generating sky masks...")

    sky_masks = []
    iterator = (images.permute(0, 2, 3, 1)[..., [2, 1, 0]][:S] * 255).byte()
    if verbose:
        iterator = tqdm(iterator)
    for image in iterator:
        sky_mask = segment_sky_nodisk(image.cpu().numpy(), skyseg_session)
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))
        sky_masks.append(sky_mask)

    sky_mask_array = np.array(sky_masks)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    return conf * sky_mask_binary


def segment_sky_nodisk(image, onnx_session):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image: input image
        onnx_session: ONNX runtime session with loaded model

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result
