# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Testing script that runs the validation loop used during training."""

import hashlib
import json
import logging
import os
import re

import torch
import wandb
from hydra.utils import instantiate

from dvlt.config.cli import cli
from dvlt.config.schema import register_configs
from dvlt.engine.trainer import get_checkpoints


logger = logging.getLogger(__name__)

register_configs()


def _collect_metrics(validation_dir: str) -> dict[str, dict[str, float]]:
    """Read metrics.json from each dataset subdir. Returns {dataset: {metric: value}}."""
    result = {}
    if not os.path.isdir(validation_dir):
        return result
    for dataset_name in sorted(os.listdir(validation_dir)):
        metrics_file = os.path.join(validation_dir, dataset_name, "metrics.json")
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file) as f:
            all_metrics = json.load(f)
        flat = {}
        for key, value in all_metrics.items():
            if isinstance(value, dict):
                flat.update({f"{key}_{k}": v for k, v in value.items() if not k.endswith("_std")})
            else:
                flat[key] = value
        result[dataset_name] = flat
    return result


def _log_results_to_wandb(
    output_dir: str, wandb_project: str, run_name: str, step_interval: int, suffix: str | None = None
) -> None:
    """Find all test_results-<step> dirs and log their metrics to a wandb run with -val suffix."""
    pattern = rf"test_results-{re.escape(suffix)}-(\d+)$" if suffix else r"test_results-(\d+)$"
    results = []
    for entry in os.listdir(output_dir):
        match = re.match(pattern, entry)
        if match:
            step = int(match.group(1))
            full_path = os.path.join(output_dir, entry)
            if os.path.isdir(full_path):
                results.append((full_path, step))
    results.sort(key=lambda x: x[1])

    if step_interval > 0:
        skipped = [s for _, s in results if s % step_interval != 0]
        results = [(p, s) for p, s in results if s % step_interval == 0]
        if skipped:
            logger.info(f"Wandb logging: skipping non-interval steps {skipped}")

    if not results:
        logger.warning(f"No test_results directories found in {output_dir} to log")
        return

    val_run_name = f"{run_name}-val-{suffix}" if suffix else f"{run_name}-val"
    stable_id = hashlib.sha256(val_run_name.encode()).hexdigest()[:16]

    wandb.init(
        project=wandb_project,
        name=val_run_name,
        id=stable_id,
        resume="allow",
        settings=wandb.Settings(init_timeout=300),
    )

    for path, step in results:
        metrics_by_dataset = _collect_metrics(path)
        row = {}
        for dataset_name, flat in metrics_by_dataset.items():
            row.update({f"val/{dataset_name}/{k}": v for k, v in flat.items()})
        wandb.log(row, step=step)
        logger.info(f"Logged {len(row)} metric(s) at step {step}")

    wandb.finish()
    logger.info("Wandb logging complete.")


@cli(
    config_path="../config/experiments",
    config_name="default",
    version_base=None,
    extra_args=[
        (
            "--step-interval",
            {
                "type": str,
                "default": None,
                "help": "Evaluate multiple checkpoints: 'all' for every checkpoint, or an integer (e.g. 10000) to only evaluate at multiples of that step.",
            },
        ),
        (
            "--log-to-wandb",
            {
                "action": "store_true",
                "default": False,
                "help": "Log test results to wandb as a separate -val run after evaluation.",
            },
        ),
        (
            "--suffix",
            {
                "type": str,
                "default": None,
                "help": "Suffix for results dirs and wandb run name. Uses test_results-{suffix}-{step} and logs to a new wandb run named {run}-val-{suffix}.",
            },
        ),
    ],
)
def main(config, step_interval=None, log_to_wandb=False, suffix=None):
    """Main testing function using structured configs."""
    model = instantiate(config.model)
    data = instantiate(config.data)
    trainer = instantiate(config.trainer, model=model, data=data)
    trainer.setup(mode="test")

    if step_interval is None:
        trainer.test()
        return

    # Parse step_interval: "all" = no filter, "<int>" = filter to multiples
    interval_value = 0
    if step_interval != "all":
        try:
            interval_value = int(step_interval)
        except ValueError as err:
            raise ValueError(f"--step-interval must be 'all' or an integer, got '{step_interval}'") from err

    output_dir = trainer.output_dir
    checkpoints = get_checkpoints(output_dir)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output_dir}")

    if interval_value > 0:
        filtered = []
        for ckpt_name in checkpoints:
            step = int(ckpt_name.split("-")[1])
            if step % interval_value == 0:
                filtered.append(ckpt_name)
            else:
                logger.info(f"Skipping {ckpt_name} (step {step} not a multiple of {interval_value})")
        checkpoints = filtered

    logger.info(f"Evaluating {len(checkpoints)} checkpoint(s): {checkpoints}")
    for ckpt_name in checkpoints:
        step = int(ckpt_name.split("-")[1])
        ckpt_path = os.path.join(output_dir, ckpt_name)
        results_tag = f"test_results-{suffix}-{step}" if suffix else f"test_results-{step}"
        results_path = os.path.join(output_dir, results_tag)

        if os.path.exists(results_path):
            logger.info(f"Skipping {ckpt_name} — results already exist at {results_path}")
            continue

        logger.info(f"=== Evaluating {ckpt_name} (step {step}) ===")
        trainer.model.load_pretrained(ckpt_path, strict=True)
        trainer.model.setup_test(trainer.accelerator)
        trainer._test_loop(mode="test", step=step, output_dir=results_path)
        torch.cuda.empty_cache()

    trainer.accelerator.end_training()

    if log_to_wandb and trainer.accelerator.is_main_process:
        run_name = f"{trainer.experiment_name}_{trainer.timestamp}"
        _log_results_to_wandb(
            str(output_dir),
            trainer.wandb_project_name,
            run_name,
            interval_value,
            suffix=suffix,
        )


if __name__ == "__main__":
    main()
