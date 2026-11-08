# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-model registry used by the Gradio demo (and any future batch script).

Provides a curated list of ``(label, experiment-yaml)`` pairs that ship with
ckpt paths pinned, a Hydra-compose helper that loads one of those YAMLs on
demand, and a small ``ModelEntry`` that lazily instantiates / loads /
unloads the model so peak GPU memory stays bounded when iterating across
multiple checkpoints in batch.

Curated list (``DEFAULT_INFERENCE_MODELS``) deliberately excludes pure
training recipes (e.g. ``dvlt-large``, ``dvlt-large-depthconv-stage2``,
``dvlt-large-ablation``, ``default.yaml``); use the inference-only
``dvlt.yaml`` instead.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


# Anchor experiment lookups to ``<repo>/src/dvlt/config/experiments`` so this
# module works no matter where it's imported from.
EXPERIMENTS_DIR: str = str((Path(__file__).resolve().parent.parent / "config" / "experiments"))


# Order is the dropdown order; the first entry is the default selection
# unless overridden. Every config below pins ``trainer.ckpt_dir`` to a
# public/local checkpoint.
DEFAULT_INFERENCE_MODELS: list[tuple[str, str]] = [
    ("DVLT", "dvlt"),
    ("VGGT 1B", "vggt"),
    ("VGGT-Omega 1B", "vggt_omega"),
    ("DA3 Base", "da3-base"),
    ("DA3 Large", "da3-large"),
    ("DA3 Giant", "da3-giant"),
    ("Pi3", "pi3"),
    ("Pi3X", "pi3x"),
    ("MapAnything", "mapanything"),
]


__all__ = [
    "EXPERIMENTS_DIR",
    "DEFAULT_INFERENCE_MODELS",
    "ModelEntry",
    "compose_experiment",
    "parse_model_spec",
]


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a ``LABEL:CONFIG_NAME`` string (e.g. ``--add-model`` CLI flag)."""
    parts = spec.split(":", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(f"Bad --add-model spec {spec!r}: expected LABEL:CONFIG_NAME")
    return parts[0].strip(), parts[1].strip()


def compose_experiment(config_name: str, cfg_dir: str = EXPERIMENTS_DIR) -> Any:
    """Compose an experiment Hydra config by name from ``cfg_dir``.

    The ``@cli`` decorator runs its own ``compose`` and tears the global
    Hydra context down before main(). We re-initialize here to load
    additional configs on demand without disturbing that flow.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(cfg_dir)):
        return hydra_compose(config_name=config_name, overrides=[], return_hydra_config=False)


@dataclass
class ModelEntry:
    """One row in the dropdown. ``model`` is populated on first ``ensure_loaded``."""

    label: str
    config_name: Optional[str] = None  # name of YAML to compose lazily; None when ``config`` is already set
    config: Any = None  # may be pre-supplied (e.g. for the primary entry from --config-name)
    model: Any = None
    img_size: int = 518
    patch_size: int = 14

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self, accelerator: Accelerator) -> None:
        if self.is_loaded:
            return
        if self.config is None:
            if not self.config_name:
                raise ValueError(f"Model '{self.label}' has no config_name and no pre-composed config.")
            logger.info(f"[{self.label}] Composing config '{self.config_name}'")
            self.config = compose_experiment(self.config_name)
        cfg = self.config
        logger.info(f"[{self.label}] Instantiating model")
        model = instantiate(cfg.model)

        ckpt = getattr(getattr(cfg, "trainer", None), "ckpt_dir", "") or ""
        if ckpt:
            logger.info(f"[{self.label}] Loading checkpoint from {ckpt}")
            model.load_pretrained(ckpt, strict=True)
        else:
            logger.warning(f"[{self.label}] No checkpoint specified — running with random weights.")
        model.setup_test(accelerator)

        self.model = model
        self.img_size = int(cfg.data.image_size)
        self.patch_size = int(cfg.data.patch_size)
        logger.info(f"[{self.label}] Using img_size={self.img_size}, patch_size={self.patch_size}")

    def unload(self) -> None:
        """Drop the model and free GPU memory. ``ensure_loaded`` rebuilds later."""
        if self.model is None:
            return
        logger.info(f"[{self.label}] Unloading model")
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
