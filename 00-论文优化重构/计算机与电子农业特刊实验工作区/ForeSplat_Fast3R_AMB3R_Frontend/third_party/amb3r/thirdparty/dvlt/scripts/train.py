# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training script."""

from hydra.utils import instantiate

from dvlt.config.cli import cli
from dvlt.config.schema import register_configs


register_configs()


@cli(config_path="../config/experiments", config_name="default", version_base=None)
def main(config):
    """Main training function using structured configs."""
    # Instantiate the model, data and trainer
    model = instantiate(config.model)
    data = instantiate(config.data)
    trainer = instantiate(config.trainer, model=model, data=data)

    # Train the model
    trainer.setup(mode="train", config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
