# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw dataset-source readers and the ``dataset_from_config`` factory."""

import importlib

from omegaconf.dictconfig import DictConfig

from dvlt.data.sources.datasets.base import BaseDataset


def dataset_from_config(config: DictConfig, **additional_params) -> BaseDataset:
    """Instantiate the reader named by ``config.target`` with ``config.params``.

    Targets not resolvable under ``dvlt.data.sources.datasets`` fall back to
    :class:`GenericDataset`.
    """
    assert "target" in config, "Expected key `target` to be the class name under dvlt.data.sources.datasets."

    target = config.target
    params = config.get("params", dict())
    params.update(additional_params)

    module_name, cls_name = target.rsplit(".", 1)

    try:
        dataset_module = importlib.import_module(f"dvlt.data.sources.datasets.{module_name}")
        dataset_cls = getattr(dataset_module, cls_name)
    except (ImportError, AttributeError):
        from dvlt.data.sources.datasets.generic import GenericDataset

        return GenericDataset(target=target, **params)

    dataset_inst = dataset_cls(**params)
    assert isinstance(dataset_inst, BaseDataset), f"{cls_name} is not a BaseDataset."
    return dataset_inst
