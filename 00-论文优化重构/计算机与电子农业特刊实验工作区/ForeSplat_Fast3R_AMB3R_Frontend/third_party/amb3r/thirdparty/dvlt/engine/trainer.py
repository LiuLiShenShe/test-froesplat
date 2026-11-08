# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import copy
import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    broadcast_object_list,
    set_seed,
)
from omegaconf import OmegaConf
from torch import nn
from torch.utils.collect_env import get_pretty_env_info

from dvlt.common.constants import DataField
from dvlt.common.tensor import to_device
from dvlt.config.schema import Config
from dvlt.data.module import DataModule
from dvlt.engine.profiler import Profiler
from dvlt.engine.progress import ProgressBar
from dvlt.engine.scheduler import SchedulerType, get_scheduler
from dvlt.model.base import Module
from dvlt.model_components import set_attn_backend
from dvlt.util.sanity_check import MockAcceleratorTrackers


logger = get_logger(__name__)


class SingleBatchIterator:
    """Iterator that yields fresh copies of a single batch indefinitely."""

    def __init__(self, batch, length, device):
        self.batch = batch
        self.length = length
        self.device = device
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        # Create a copy and then move to device - more efficient than copying on device
        batch_copy = copy.deepcopy(self.batch)
        return to_device(batch_copy, self.device, non_blocking=True)

    def __len__(self):
        return self.length


class Trainer:
    """Trainer class.

    This class is responsible for training a model, logging, checkpointing, etc.
    """

    def __init__(
        self,
        model: Module,
        data: DataModule,
        # callbacks
        callbacks: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        # prediction caching
        load_cached_predictions: bool = False,
        write_predictions: bool = False,
        # logging
        output_dir: str = "outputs",
        experiment_name: str = "unnamed",
        timestamp: Optional[str] = None,
        logging_dir: str = "logs",
        experiment_logger: Tuple[str, ...] = ("wandb",),
        wandb_project_name: str = "dvlt",
        tqdm: bool = False,
        print_step_interval: int = 10,
        # training loop
        seed: Optional[int] = None,
        max_train_steps: int = 50_000,
        validation_steps: int = 5000,
        validation_batches: int = 0,  # 0 means use all batches
        ckpt_dir: str = "",
        # debugging
        single_batch_overfit: bool = False,
        sanity_check: bool = False,
        # checkpointing
        checkpointing_steps: int = 5000,
        checkpoints_total_limit: int = 2,
        resume_from_checkpoint: Optional[str] = None,
        # ddp
        find_unused_parameters: bool = False,
        static_graph: bool = False,
        # training & speed optimizations
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        gradient_bucket_view: bool = True,
        allow_tf32: bool = False,
        cudnn_deterministic: bool = False,
        cudnn_benchmark: bool = True,
        mixed_precision: str = "no",
        attn_backend: str = "auto",
        set_grads_to_none: bool = False,
        # learning rate & scheduler
        learning_rate: float = 1e-5,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        lr_num_cycles: float = 0.5,
        lr_power: float = 1.0,
        lr_min_ratio: float = 0.01,
        lr_param_group_multipliers: Optional[Dict[str, float]] = None,
        scale_lr: Optional[str] = None,
        # optimizer & grad
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        # profiling
        profiler: str = "minimal",
        profiler_dir: str = "profiler_stats",
        pytorch_profile_memory: bool = True,
        pytorch_with_stack: bool = True,
        pytorch_record_shapes: bool = True,
    ):
        """Trainer init.

        Args:
            model: The model to use.
            data: The data module to use.
            output_dir: Output directory.
            experiment_name: Experiment name.
            timestamp: Timestamp.
            logging_dir: Logging subdir.
            experiment_logger: Experiment logger(s) to use: tensorboard or wandb.
            wandb_project_name: Wandb project name.
            tqdm: Whether to use tqdm for progress bar. If False, the progress bar print sequentially every 10 steps.
            print_step_interval: Print progress bar every X steps.
            seed: Random seed.
            max_train_steps: Total number of training steps to perform.
            validation_steps: Run validation every X steps.
            validation_batches: Number of batches to validate on. 0 means use all batches.
            ckpt_dir: Pretrained checkpoint to init the model from.
            single_batch_overfit: Whether to train on a single batch repeatedly (for debugging).
            sanity_check: Whether to run in sanity check mode (temp dirs, mock loggers, limited steps).
            find_unused_parameters: Whether to find unused parameters in DDP.
            static_graph: Whether to use static graph in DDP.
            checkpointing_steps: Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`.
            checkpoints_total_limit: Max number of checkpoints to store. Set to -1 to keep all.
            resume_from_checkpoint: Whether training should be resumed from a previous checkpoint. Use 'latest', a checkpoint path or name (e.g. 'checkpoint-100').
            gradient_accumulation_steps: Number of updates steps to accumulate before performing a backward/update pass.
            gradient_checkpointing: Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
            gradient_bucket_view: Whether or not to use gradient bucket view to save memory.
            allow_tf32: Whether or not to allow TF32 on Ampere GPUs. Note that this trades off numerical precision for speed.
            cudnn_deterministic: Whether to use deterministic cudnn operations.
            cudnn_benchmark: Whether to use cudnn benchmark mode.
            mixed_precision: Whether to use mixed precision. Choose between no, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU.
                Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command.
                Use this argument to override the accelerate config.
            set_grads_to_none: Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain behaviors,
                so disable this argument if it causes any problems.
            learning_rate: Initial learning rate (after the potential warmup period) to use.
            lr_scheduler: One of linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup.
            lr_warmup_steps: Number of steps for the warmup in the lr scheduler.
            lr_num_cycles: Number of hard resets of the lr in cosine_with_restarts scheduler.
            lr_power: Power factor of the polynomial scheduler.
            lr_param_group_multipliers: Multipliers for the learning rate of different parameter groups.
            scale_lr: Scale learning rate based on batch size.
            adam_beta1: The beta1 parameter for the Adam optimizer.
            adam_beta2: The beta2 parameter for the Adam optimizer.
            adam_weight_decay: Weight decay to use.
            adam_epsilon: Epsilon value for the Adam optimizer.
            max_grad_norm: Maximum gradient norm.
            profiler: Profiler mode ('minimal', 'full', 'pytorch', 'memory').
            profiler_dir: Relative directory within output_dir to save profiler data.
            pytorch_profile_memory: For PyTorch profiler, whether to track memory usage (increases output size).
            pytorch_with_stack: For PyTorch profiler, whether to record stack traces (increases output size).
            pytorch_record_shapes: For PyTorch profiler, whether to record tensor shapes.
        """
        # Store model and data
        self.model = model
        self.data_module = data
        self.callbacks = {k: v for k, v in (callbacks or {}).items() if v is not None}
        self.load_cached_predictions = load_cached_predictions
        self.write_predictions = write_predictions

        # Store all parameters
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.experiment_logger = experiment_logger
        self.wandb_project_name = wandb_project_name
        self.tqdm = tqdm
        self.print_step_interval = print_step_interval
        self.seed = seed
        self.max_train_steps = max_train_steps
        self.validation_steps = validation_steps
        self.validation_batches = validation_batches
        self.ckpt_dir = ckpt_dir
        self.single_batch_overfit = single_batch_overfit
        self.sanity_check = sanity_check
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph
        self.checkpointing_steps = checkpointing_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.resume_from_checkpoint = resume_from_checkpoint
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_bucket_view = gradient_bucket_view
        self.allow_tf32 = allow_tf32
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_benchmark = cudnn_benchmark
        self.mixed_precision = mixed_precision
        self.attn_backend = attn_backend
        self.set_grads_to_none = set_grads_to_none
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_num_cycles = lr_num_cycles
        self.lr_power = lr_power
        self.lr_min_ratio = lr_min_ratio
        self.lr_param_group_multipliers = lr_param_group_multipliers
        self.scale_lr = scale_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.profiler_mode = profiler
        self.profiler_dir = profiler_dir
        self.profiler_profile_memory = pytorch_profile_memory
        self.profiler_with_stack = pytorch_with_stack
        self.profiler_record_shapes = pytorch_record_shapes

        # Validate configuration
        assert self.mixed_precision in ["no", "fp16", "bf16"]
        assert self.lr_scheduler in [e.value for e in SchedulerType]
        assert all(logger in ["wandb", "tensorboard"] for logger in self.experiment_logger)
        self.accelerator = None
        self.wandb_id = None

    def setup(self, mode: str = "train", config: Optional[Config] = None):
        """
        Setup the trainer.

        Args:
            mode: Mode to run the trainer in. Determines the name of the log file. One of "train", "test".
            config: Configuration used to initialize trainer, model, data.
        """
        # set paths to absolute paths
        if self.sanity_check:
            # Use temporary directory for sanity check mode
            self.output_dir = Path(tempfile.mkdtemp(prefix="sanity_check_"))
            # Use standard logging since accelerate logger isn't ready yet
            print(f"[SANITY CHECK] Using temporary output directory: {self.output_dir}")
        else:
            self.output_dir = Path(self.output_dir, self.experiment_name, self.timestamp)
        self.logging_dir = Path(self.output_dir, self.logging_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # setup accelerator
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir, logging_dir=self.logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_bucket_view,
            static_graph=self.static_graph,
        )
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=list(self.experiment_logger),
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
        )
        self.output_dir = broadcast_object_list([self.output_dir])[0]

        # save the full configuration as yaml file in the output directory
        if config is not None and self.accelerator.is_main_process:
            if mode == "train" or not (self.output_dir / "config.yaml").exists():
                config.trainer.timestamp = self.timestamp
                with open(self.output_dir / "config.yaml", "w") as f:
                    OmegaConf.save(config, f)

        # store current git commit hash
        if mode == "train" and self.accelerator.is_main_process:
            commit_sha_file = self.output_dir / "git_sha.txt"
            if commit_sha_file.exists():
                with open(commit_sha_file, "r") as f:
                    git_sha = f.read().strip()
                if not git_sha == get_git_sha():
                    logger.warning(
                        f"Git commit hash mismatch: {git_sha} != {get_git_sha()}. You're resuming training from a different commit."
                    )
            else:
                git_sha = get_git_sha()
                with open(commit_sha_file, "w") as f:
                    f.write(git_sha)

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        # setup logging to file in the output directory
        logger.info(f"Saving outputs to: {self.output_dir}")
        if self.accelerator.is_main_process:
            log_file = os.path.join(self.output_dir, f"{mode}.log")
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

        # print accelerator state
        logger.info(self.accelerator.state, main_process_only=False)

        # If passed along, set the training seed now.
        if self.seed is not None:
            set_seed(self.seed + self.accelerator.process_index)
            logger.info(f"Set seed to: {self.seed}")

        # Print environment info
        if mode == "train":
            logger.info("***** Environment Info *****", main_process_only=True)
            logger.info(get_pretty_env_info(), main_process_only=True)

            # Check CUDA memory allocator config
            cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set")
            logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}", main_process_only=True)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        # Create profiler
        profile_dir = os.path.join(self.output_dir, self.profiler_dir)
        os.makedirs(profile_dir, exist_ok=True)

        self.profiler = Profiler(
            mode=self.profiler_mode,
            output_dir=profile_dir,
            profile_memory=self.profiler_profile_memory,
            with_stack=self.profiler_with_stack,
            record_shapes=self.profiler_record_shapes,
        )

    def fit(self):
        # Override settings for sanity check mode
        if self.sanity_check:
            logger.info("[SANITY CHECK] Running in sanity check mode")
            # Override parameters to test all functions quickly
            self.max_train_steps = 11  # Run 11 steps to trigger step 10 visualization
            self.validation_steps = 11  # Run validation after 11 steps
            self.validation_batches = 2  # Use only 2 validation batches
            self.model.log_every_n_steps = 1  # Enable log_train visualization at step 10

            logger.info(
                "[SANITY CHECK] Overriding: max_train_steps=11, validation_steps=11, validation_batches=2, log_every_n_steps=1"
            )

        learning_rate = self.learning_rate
        if self.scale_lr is not None:
            total_batch_size = (
                self.gradient_accumulation_steps * self.data_module.images_per_batch * self.accelerator.num_processes
            )
            if self.scale_lr == "linear":
                scale_factor = total_batch_size
            elif self.scale_lr == "sqrt":
                scale_factor = math.sqrt(total_batch_size)
            else:
                raise ValueError(f"Invalid LR scaling method {self.scale_lr}")
            learning_rate = learning_rate * scale_factor

        optimizer_class = torch.optim.AdamW

        logger.info(f"Initializing optimizer {str(optimizer_class)}")
        lr_param_group_multipliers = self.lr_param_group_multipliers or {}

        # Build id(param) -> group-name map from the model-provided groups so we
        # can preserve LR multipliers while splitting each group into decay /
        # no-decay sub-groups. 1D params (LayerNorm gamma/beta, LayerScale gamma)
        # and biases are excluded from weight decay (MAE / DINOv2 / Pi3 convention).
        # Note: ``self.model`` is the ``Module`` wrapper; iterate the underlying
        # ``nn.Module`` via ``self.model.model`` to get ``named_parameters()``.
        param_to_group: Dict[int, str] = {}
        for group_name, params in self.model.get_param_groups().items():
            for p in params:
                param_to_group[id(p)] = group_name

        buckets: Dict[Tuple[str, bool], List[nn.Parameter]] = {}
        for name, p in self.model.model.named_parameters():
            group_name = param_to_group.get(id(p))
            if group_name is None:
                continue  # not in any trainable group (e.g., frozen in finetune)
            is_decay = (p.ndim > 1) and (not name.endswith(".bias"))
            buckets.setdefault((group_name, is_decay), []).append(p)

        param_groups = []
        n_decay = n_no_decay = 0
        for (group_name, is_decay), ps in buckets.items():
            lr_multiplier = lr_param_group_multipliers.get(group_name, 1.0)
            wd = self.adam_weight_decay if is_decay else 0.0
            param_groups.append(
                {
                    "params": ps,
                    "lr": learning_rate * lr_multiplier,
                    "weight_decay": wd,
                }
            )
            n = sum(p.numel() for p in ps)
            if is_decay:
                n_decay += n
            else:
                n_no_decay += n
        logger.info(
            f"Optimizer param split: decay={n_decay:,} no_decay={n_no_decay:,} "
            f"(WD={self.adam_weight_decay} applied to >1D non-bias params)"
        )

        optimizer = optimizer_class(
            param_groups,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
        )

        # Initialize data module if not provided in init
        train_dataloader = self.data_module.train_dataloader(accelerator=self.accelerator)

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.lr_num_cycles,
            power=self.lr_power,
            min_lr_ratio=self.lr_min_ratio,
        )

        # load pre-trained weights if any
        if self.ckpt_dir != "":
            self.model.load_pretrained(self.ckpt_dir)

        logger.info("***** Model summary *****")
        self.model.print_summary()

        # Move model to device, prepare for training
        self.model.setup_train(self.accelerator, gradient_checkpointing=self.gradient_checkpointing)

        # Set attention backend globally for all Attention modules
        set_attn_backend(self.attn_backend)
        logger.info(f"  Attention backend: {self.attn_backend}")

        # Prepare everything with accelerator
        optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.sanity_check:
            # Use mock trackers for sanity check mode (override accelerator's empty trackers)
            self.accelerator.trackers = MockAcceleratorTrackers()
            logger.info("[SANITY CHECK] Using mock trackers")
        elif self.accelerator.is_main_process:
            job_name = f"{self.experiment_name}_{time.strftime('%Y-%m-%d-%H_%M_%S')}"
            config_path = os.path.join(self.output_dir, "config.yaml")
            if os.path.exists(config_path):
                config = OmegaConf.load(config_path)
                config_dict = OmegaConf.to_container(config, resolve=True)
            else:
                config_dict = {}

            # If we're resuming from a checkpoint but don't yet have a wandb_id,
            # try to extract it from the checkpoint directory's wandb logs
            if self.resume_from_checkpoint and not self.wandb_id:
                self.wandb_id = self._extract_wandb_id_from_output_dir()
            self.accelerator.init_trackers(
                self.wandb_project_name,
                init_kwargs={
                    "wandb": {
                        "name": job_name,
                        "config": config_dict,
                        "dir": str(self.logging_dir),
                        "id": self.wandb_id,
                    }
                },
            )
        # For non-main processes in regular mode, ensure empty trackers list exists
        elif not hasattr(self.accelerator, "trackers"):
            self.accelerator.trackers = []

        # Train!
        total_batch_size = (
            self.data_module.images_per_batch * self.accelerator.num_processes * self.gradient_accumulation_steps
        )
        total_params = sum(p.numel() for p in self.model.get_params())
        trainable_params = sum(p.numel() for p in self.model.get_trainable_params())
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.data_module.images_per_batch}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        global_step = 0
        # Potentially load in the weights and states from a previous save
        if self.resume_from_checkpoint:
            if self.resume_from_checkpoint == "latest":
                # Get the most recent checkpoint
                dirs = os.listdir(self.output_dir) if os.path.isdir(self.output_dir) else []
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                resume_path = os.path.join(self.output_dir, path) if path is not None else None

            elif os.path.dirname(self.resume_from_checkpoint) == "":
                resume_path = os.path.join(self.output_dir, self.resume_from_checkpoint)
            else:
                resume_path = self.resume_from_checkpoint

            if resume_path is None or not os.path.exists(resume_path):
                logger.warn(f"Checkpoint '{resume_path}' does not exist. Starting a new training run.")
            else:
                assert os.path.basename(resume_path).startswith(
                    "checkpoint-"
                ), "Checkpoint name must start with 'checkpoint-'"
                logger.info(f"Resuming from checkpoint {resume_path}")
                self.accelerator.load_state(resume_path)
                global_step = int(os.path.basename(resume_path).split("-")[1])
                logger.info(f"Resuming from global step {global_step}")

        self._train_loop(train_dataloader, optimizer, lr_scheduler, global_step, self.max_train_steps)

        # Create the pipeline using using the trained modules and save it.
        os.makedirs(os.path.join(self.output_dir, "final_model"), exist_ok=True)
        self.model.save_pretrained(self.accelerator, os.path.join(self.output_dir, "final_model"))

        self.accelerator.end_training()

        # Cleanup for sanity check mode
        if self.sanity_check:
            logger.info(f"[SANITY CHECK] Cleaning up temporary directory: {self.output_dir}")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            logger.info("[SANITY CHECK] Sanity check completed successfully!")

    def _train_loop(self, train_dataloader, optimizer, lr_scheduler, global_step, max_train_steps):
        # For profiling modes, limit the number of steps
        if self.profiler.step_limit is not None:
            logger.info(f"Profiling mode '{self.profiler_mode}' active - limiting to {self.profiler.step_limit} steps")
            max_train_steps = min(max_train_steps, global_step + self.profiler.step_limit)

        # Handle single batch overfit mode
        single_batch = None
        if self.single_batch_overfit:
            logger.info("Running in single batch overfit mode")
            # Get a single batch and use it repeatedly
            single_batch = next(iter(train_dataloader))

            # Make sure all processes use the same batch in distributed training
            if self.accelerator.num_processes > 1:
                single_batch = broadcast_object_list([single_batch])[0]
                # Keep the batch on CPU for efficient copying, device transfer happens in iterator
            train_dataloader = SingleBatchIterator(single_batch, 1_000_000, self.accelerator.device)

        progress_bar = ProgressBar(
            range(0, max_train_steps),
            initial=global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
            use_tqdm=self.tqdm,
            print_step_interval=self.print_step_interval,
        )

        dataloader_iter = iter(train_dataloader)
        while global_step < max_train_steps:
            self.profiler.start_step(global_step)
            self.profiler.start_data_loading()

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Restart the dataloader if it's exhausted but we haven't reached max_train_steps
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)

            self.profiler.stop_data_loading()

            with self.accelerator.accumulate(self.model.model):
                # NOTE: You could add autocast context manager here to keep using mixed precision outside of
                # model.forward, e.g. for complex loss calculations.
                # See https://huggingface.co/docs/accelerate/main/en/basic_tutorials/migration#mixed-precision
                self.profiler.start_forward()
                loss, pbar_logs, tracker_logs, predictions = self.model.train_step(batch, global_step, self.accelerator)
                self.profiler.stop_forward()

                if not torch.isfinite(loss):
                    raise ValueError(f"Train Loss is not finite: {loss}, {pbar_logs}.")

                self.profiler.start_backward()
                self.accelerator.backward(loss)
                self.profiler.stop_backward()

                # Call callbacks for logging (visualization, parameter stats, etc.)
                if self.callbacks:
                    for callback in self.callbacks.values():
                        callback.on_train_batch(batch, predictions, global_step, self)

                del batch, predictions

                # Time the optimizer operations
                self.profiler.start_optimizer()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.get_trainable_params(), self.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=self.set_grads_to_none)
                self.profiler.stop_optimizer()

            # Gather metrics from all processes with synchronized keys to avoid NCCL desync
            total_loss = self.accelerator.gather(loss.detach()).mean().item()
            global_pbar_keys, global_tracker_keys = _sync_and_verify_metric_keys(pbar_logs, tracker_logs)
            pbar_logs = {k: self.accelerator.gather(pbar_logs[k].detach()).mean().item() for k in global_pbar_keys}
            tracker_logs = {
                k: self.accelerator.gather(tracker_logs[k].detach()).mean().item() for k in global_tracker_keys
            }
            timing_metrics = self.profiler.end_step(global_step)
            current_lr = min(lr_scheduler.get_last_lr(), key=lambda x: abs(x - self.learning_rate))
            progress_bar.set_postfix(
                **{
                    "loss": total_loss,
                    **pbar_logs,
                    **timing_metrics,
                    "lr": current_lr,
                }
            )
            self.accelerator.log(
                {
                    "train/loss": total_loss,
                    **{"train/" + k: v for k, v in tracker_logs.items()},
                    **timing_metrics,
                    "lr": current_lr,
                },
                step=global_step,
            )
            del loss, pbar_logs, tracker_logs

            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % self.checkpointing_steps == 0:
                    # save checkpoint first, then remove old ones only after successful save
                    save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                    # ALL processes must call save_state for distributed training
                    self.accelerator.save_state(save_path)
                    if self.accelerator.is_main_process:
                        logger.info(f"Saved state to {save_path}")
                        # only remove old checkpoints after successful save
                        self.remove_checkpoints()

                # Skip validation in single batch overfit mode
                if global_step % self.validation_steps == 0 and not self.single_batch_overfit:
                    self.model.setup_test(self.accelerator)
                    torch.cuda.empty_cache()
                    save_path = os.path.join(self.output_dir, f"validation-{global_step}")
                    if self.accelerator.is_main_process:
                        self.remove_validation_result()
                        os.makedirs(save_path, exist_ok=True)
                    self.accelerator.wait_for_everyone()
                    self._test_loop(mode="val", step=global_step, output_dir=save_path)
                    self.model.setup_train(self.accelerator, gradient_checkpointing=self.gradient_checkpointing)
                    torch.cuda.empty_cache()

        # For profiling modes, print a clear completion message
        if self.profiler.step_limit is not None and self.accelerator.is_main_process:
            logger.info(f"Profiling data saved to {os.path.join(self.output_dir, self.profiler_dir)}")
            # Print viewing instructions
            if self.profiler_mode == "pytorch":
                Profiler.print_viewing_instructions(os.path.join(self.output_dir, self.profiler_dir))
            elif self.profiler_mode == "memory" and getattr(self.profiler, "memory_snapshot_path", None):
                Profiler.print_memory_viewing_instructions(
                    self.profiler.memory_snapshot_path,
                    getattr(self.profiler, "memory_html_path", None),
                )

        progress_bar.close()

    def test(self):
        """Test the model."""
        if self.sanity_check:
            logger.info("[SANITY CHECK] Running test in sanity check mode")
            # Override validation_batches for sanity check
            self.validation_batches = 2

        test_results_dir = "test_results"
        if self.ckpt_dir:
            ckpt = self.ckpt_dir
            logger.info(f"Using explicitly provided trainer.ckpt_dir={ckpt}")
            # Extract step number from checkpoint dir name (e.g. "checkpoint-60000" -> "60000")
            ckpt_basename = os.path.basename(ckpt.rstrip("/"))
            if ckpt_basename.startswith("checkpoint-"):
                step_suffix = ckpt_basename.split("checkpoint-", 1)[1]
                test_results_dir = f"test_results-{step_suffix}"
        else:
            ckpt = get_checkpoint_dir(self.output_dir)
            if ckpt is None:
                raise ValueError(
                    "No checkpoint found in output_dir and ckpt_dir is empty. Please provide a checkpoint."
                )
        self.model.load_pretrained(ckpt, strict=True)
        self.model.setup_test(self.accelerator)
        self._test_loop(mode="test", output_dir=os.path.join(self.output_dir, test_results_dir))
        self.accelerator.end_training()

        # Cleanup for sanity check mode
        if self.sanity_check:
            logger.info(f"[SANITY CHECK] Cleaning up temporary directory: {self.output_dir}")
            shutil.rmtree(self.output_dir, ignore_errors=True)
            logger.info("[SANITY CHECK] Test sanity check completed successfully!")

    @torch.no_grad()
    def _test_loop(self, mode: Literal["test", "val"] = "test", step=None, output_dir=None):
        """Loop for validation or testing."""
        test_dataloaders = self.data_module.test_dataloaders(accelerator=self.accelerator)
        distributed_eval = getattr(self.data_module, "distributed_eval", True)

        logger.info(f"***** Running {mode} *****")
        for dataset_name, test_dataloader in test_dataloaders.items():
            current_output_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(current_output_dir, exist_ok=True)

            # Call callback hooks at dataset start
            for callback in self.callbacks.values():
                callback.on_test_dataset_start(self, dataset_name)

            logger.info(f"  Test Dataset {dataset_name}")
            logger.info(f"  Num examples = {len(test_dataloader)}")

            max_eval_steps = len(test_dataloader)
            if self.validation_batches > 0 and (mode == "val" or self.sanity_check):
                max_eval_steps = min(max_eval_steps, self.validation_batches)

            # For profiling modes, limit the number of steps
            if self.profiler.step_limit is not None and mode == "test":
                logger.info(f"PyTorch profiler active - limiting evaluation to {self.profiler.step_limit} steps")
                max_eval_steps = min(max_eval_steps, self.profiler.step_limit)

            if mode == "val":
                if self.sanity_check:
                    # Ensure mock trackers are available for validation
                    if not hasattr(self.accelerator, "trackers") or not self.accelerator.trackers:
                        self.accelerator.trackers = MockAcceleratorTrackers()
                trackers = self.accelerator.trackers
                log_idx = random.randint(0, max_eval_steps - 1)
            else:
                trackers = None
                log_idx = None

            progress_bar = ProgressBar(
                range(0, max_eval_steps),
                desc=f"{mode} steps",
                disable=not self.accelerator.is_local_main_process,
                use_tqdm=self.tqdm,
                print_step_interval=self.print_step_interval,
            )

            # Initialize profiler for first iteration
            self.profiler.start_step(0)
            self.profiler.start_data_loading()

            for idx, test_example in enumerate(test_dataloader):
                self.profiler.stop_data_loading()
                self.profiler.start_forward()

                # When distributed_eval is False, dataloader is not prepared,
                # so we need to manually move batch to device
                if not distributed_eval:
                    test_example = to_device(test_example, self.accelerator.device, non_blocking=True)

                # Try to load cached predictions if enabled
                predictions = None
                seq_name = (
                    test_example.get(DataField.SEQ_NAME, [f"batch_{idx:05d}"])[0]
                    if isinstance(test_example.get(DataField.SEQ_NAME), list)
                    else test_example.get(DataField.SEQ_NAME, f"batch_{idx:05d}")
                )
                cache_path = os.path.join(current_output_dir, f"{seq_name}.pth")

                if self.load_cached_predictions and os.path.exists(cache_path):
                    logger.info(f"Loading cached predictions for {seq_name}")
                    with open(cache_path, "rb") as f:
                        predictions = torch.load(f, map_location=self.accelerator.device)
                    self.profiler.stop_forward()
                else:
                    # Run normal test step (now returns just predictions)
                    with self.accelerator.autocast():
                        predictions = self.model.test_step(test_example, self.accelerator)
                    self.profiler.stop_forward()

                    # Write predictions to cache if enabled
                    if self.write_predictions and self.accelerator.is_main_process:
                        with open(cache_path, "wb") as f:
                            torch.save(predictions, f)
                timing_metrics = self.profiler.end_step(idx)

                # Call callbacks (including visualization and evaluators)
                callback_logs = {}
                for callback_name, callback in self.callbacks.items():
                    callback_metrics = callback.on_test_batch(
                        test_example,
                        predictions,
                        current_output_dir,
                        idx,
                        self,  # Pass trainer instance
                        trackers=(
                            self.accelerator.trackers if idx == log_idx and self.accelerator.is_main_process else []
                        ),
                        global_step=step,
                        dataset_name=dataset_name,
                    )
                    # Prefix metrics with callback name
                    callback_logs.update(
                        {f"{callback_name}_{k}": v for k, v in callback_metrics.items() if not math.isnan(v)}
                    )

                logs = {
                    **callback_logs,
                    **timing_metrics,
                }
                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                if idx >= max_eval_steps - 1:
                    if self.profiler.step_limit is not None and mode == "test" and self.accelerator.is_main_process:
                        logger.info(f"Profiling data saved to {os.path.join(self.output_dir, self.profiler_dir)}")
                        # Print viewing instructions if using PyTorch profiler
                        if self.profiler_mode == "pytorch":
                            Profiler.print_viewing_instructions(os.path.join(self.output_dir, self.profiler_dir))
                        elif self.profiler_mode == "memory" and getattr(self.profiler, "memory_snapshot_path", None):
                            Profiler.print_memory_viewing_instructions(
                                self.profiler.memory_snapshot_path,
                                getattr(self.profiler, "memory_html_path", None),
                            )
                    break

                self.profiler.start_step(idx + 1)
                self.profiler.start_data_loading()

            # End the final profiler step (we started one at the end of the last iteration)
            if idx < len(test_dataloader) - 1:  # Only if we broke out of the loop early
                self.profiler.stop_data_loading()
                self.profiler.end_step()

            progress_bar.close()

            # Call callback hooks at dataset end
            all_metrics = {}
            for callback_name, callback in self.callbacks.items():
                metrics = callback.on_test_dataset_end(self, current_output_dir)
                if metrics:
                    all_metrics[callback_name] = metrics

            # Save metrics
            if self.accelerator.is_main_process and all_metrics:
                # log metrics to file/tensorboard/wandb
                with open(os.path.join(current_output_dir, "metrics.json"), "w") as f:
                    json.dump(all_metrics, f, indent=4, sort_keys=True)

                if trackers is not None:
                    # Flatten nested callback metrics for logging. Skip nested
                    # dicts (e.g. ``per_sequence``) and ``_std`` entries since
                    # wandb/tensorboard only accept scalar values.
                    flat_metrics = {}
                    for key, value in all_metrics.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                if isinstance(v, dict) or k.endswith("_std"):
                                    continue
                                flat_metrics[f"{key}_{k}"] = v
                        else:
                            flat_metrics[key] = value

                    for tracker in trackers:
                        tracker.log({f"{mode}/{dataset_name}/{k}": v for k, v in flat_metrics.items()}, step=step)

    def get_checkpoints(self) -> list[str]:
        """Given an output directory, get all checkpoint dir names that exist."""
        checkpoints = os.listdir(self.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        return checkpoints

    def remove_validation_result(self):
        val_result = os.listdir(self.output_dir)
        val_result = [d for d in val_result if d.startswith("validation")]
        if val_result:
            shutil.rmtree(os.path.join(self.output_dir, val_result[0]), ignore_errors=True)

    def remove_checkpoints(self):
        # after we save a new checkpoint, we need to have at _most_ `checkpoints_total_limit` checkpoints
        if self.checkpoints_total_limit < 0:
            return

        checkpoints = self.get_checkpoints()
        if len(checkpoints) > self.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - self.checkpoints_total_limit
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(self.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint, ignore_errors=True)

    def _extract_wandb_id_from_output_dir(self) -> Optional[str]:
        """Extract wandb ID from a specific checkpoint path by looking at wandb run directories.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            The wandb ID if found, None otherwise.
        """
        # Look for wandb logs in the output directory
        wandb_logs_dir = os.path.join(self.output_dir, "logs", "wandb")
        if not os.path.exists(wandb_logs_dir):
            return None

        # Find wandb run directories
        run_dirs = [
            d
            for d in os.listdir(wandb_logs_dir)
            if d.startswith("run-") and os.path.isdir(os.path.join(wandb_logs_dir, d))
        ]

        if not run_dirs:
            return None

        # Sort by modification time to get the most recent run
        run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(wandb_logs_dir, x)), reverse=True)

        # Extract wandb ID from the most recent run directory name
        # Format is: run-{timestamp}-{wandb_id}
        latest_run_dir = run_dirs[0]
        parts = latest_run_dir.split("-")
        if len(parts) >= 3:
            wandb_id = parts[-1]  # Last part is the wandb ID
            logger.info(f"Extracted wandb run id '{wandb_id}' from checkpoint directory")
            return wandb_id
        else:
            return None


def get_checkpoint_dir(model_dir: str) -> str:
    if os.path.exists(os.path.join(model_dir, "final_model")):
        return os.path.join(model_dir, "final_model")
    ckpts = get_checkpoints(model_dir)
    if len(ckpts) > 0:
        return os.path.join(model_dir, ckpts[-1])
    return None


def get_checkpoints(output_dir: str) -> list[str]:
    """Given an output directory, get all checkpoint dir names that exist."""
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return checkpoints


def get_git_sha() -> str:
    """Get the git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip("\n")
    except subprocess.CalledProcessError:
        warnings.warn("Could not get git commit sha.", stacklevel=2)
        return "unknown"


def cleanup_memory():
    """Cleanup GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _sync_and_verify_metric_keys(pbar_logs: dict, tracker_logs: dict) -> tuple[list[str], list[str]]:
    """Synchronize metric keys across ranks and verify local presence.

    Returns:
        Tuple of (global_pbar_keys, global_tracker_keys) to use for ordered gather.
    Raises:
        RuntimeError if this rank is missing any synchronized keys.
    """
    local_pbar_keys = list(pbar_logs.keys())
    local_tracker_keys = list(tracker_logs.keys())
    global_pbar_keys, global_tracker_keys = broadcast_object_list([local_pbar_keys, local_tracker_keys])

    missing_pbar = [k for k in global_pbar_keys if k not in pbar_logs]
    missing_tracker = [k for k in global_tracker_keys if k not in tracker_logs]
    if len(missing_pbar) > 0 or len(missing_tracker) > 0:
        raise RuntimeError(
            f"Metric key mismatch across ranks. Missing on this rank: "
            f"pbar_keys={missing_pbar}, tracker_keys={missing_tracker}. "
            f"Local pbar keys={sorted(local_pbar_keys)}, global pbar keys={sorted(global_pbar_keys)}; "
            f"Local tracker keys={sorted(local_tracker_keys)}, global tracker keys={sorted(global_tracker_keys)}."
        )

    return global_pbar_keys, global_tracker_keys
