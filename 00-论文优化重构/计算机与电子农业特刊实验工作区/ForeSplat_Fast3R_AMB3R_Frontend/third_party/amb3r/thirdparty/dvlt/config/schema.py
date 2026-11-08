# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Configuration schemas for models, data, and trainer using Hydra structured configs."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import nn


@dataclass
class LossConfig:
    """Configuration for a single loss in MultiTaskLoss.

    Attributes:
        loss_class: The loss class to instantiate.
        weight: Weight to apply to this loss.
        enabled: Whether this loss is enabled.
        kwargs: Additional keyword arguments for the loss constructor.
    """

    loss_class: Type[nn.Module]
    weight: float = 1.0
    enabled: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Base model configuration.

    This is a minimal base configuration that all models should inherit from.
    It doesn't anticipate any specific attributes that specialized models might need.

    Args:
        freeze: List of module/parameter paths to freeze. None means no freezing.
        log_params: Parameter logging configuration:
            - None (default): No parameter logging
            - [] (empty list): Log all trainable parameters
            - ['module1', 'param.weight']: Log specific modules/parameters only
    """

    _target_: str = MISSING  # Must be specified by concrete model configs
    freeze: Optional[List[str]] = None
    log_params: Optional[List[str]] = None
    log_every_n_steps: int = 50
    gradient_checkpointing_config: Optional[Dict[str, Any]] = None


@dataclass
class VGGTModelConfig(ModelConfig):
    """VGGT model configuration."""

    _target_: str = "dvlt.model.vggt.model.VGGT"
    camera_head: bool = True
    depth_head: bool = True
    point_head: bool = True
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    dpt_chunk_size: Optional[int] = 24
    loss: Optional[Dict[str, Any]] = None
    conditioning_probs: Dict[str, float] = field(default_factory=dict)
    aggregator_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VGGTOmegaModelConfig(ModelConfig):
    """VGGT-Omega model configuration."""

    _target_: str = "dvlt.model.vggt_omega.model.VGGTOmega"
    camera_head: bool = True
    depth_head: bool = True
    enable_alignment: bool = False
    patch_size: int = 16
    embed_dim: int = 1024


@dataclass
class DA3ModelConfig(ModelConfig):
    """DA3 model configuration."""

    _target_: str = "dvlt.model.da3.model.DA3"
    da3_cfg: Dict[str, Any] = field(default_factory=dict)
    loss: Optional[Dict[str, Any]] = None


@dataclass
class DVLTModelConfig(ModelConfig):
    """Déjà View Looping Transformer (DVLT) configuration.

    Recurrent transformer that loops K shared attention block pairs (frame +
    global attention) with discrete depth indexing. Each step k has a learned
    embedding that produces element-wise scale vectors centered at 1.
    """

    _target_: str = "dvlt.model.dvlt.model.DVLT"
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 768
    num_steps: int = 16
    min_steps: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    patch_embed: str = "dinov2_vitb14_reg"
    load_patch_embed_weights: bool = True
    decoder_depth: int = 2
    decoder_embed_dim: int = 384
    decoder_num_heads: int = 6
    camera_head: bool = True
    drop_path: float = 0.1
    stochastic_depth: float = 0.3
    stochastic_depth_mode: str = "random"  # "random" (drop individual steps) or "prefix" (contiguous)
    sync_stochastic_depth: bool = True  # sync step count across ranks (less diversity, better GPU util)
    # Recurrence mode:
    #   "gated"          — per-step s_attn / s_mlp / s_out scaling (raptor, default)
    #   "no_sout"        — drops s_out
    #   "no_depthscale"  — shared blocks, no depth scaling
    #   "none"           — distinct block per step, no depth scaling
    recurrence_mode: str = "gated"
    time_conditioning: str = "interval"  # "continuous" | "interval" (gated / no_sout modes only)
    k_sampling: Optional[str] = "linspace"  # None | "linspace" (variable-K linspace grid on [0,1])
    k_sampler_beta_a: int = 2  # Beta(a, b) shape 'a' for K sampling; E[K] = min + a/(a+b) * span
    k_sampler_beta_b: int = 1  # Beta(a, b) shape 'b' for K sampling
    inference_steps: Optional[int] = None  # override step count at inference (defaults to num_steps)
    decoder_head_type: str = "linear"  # "linear" (pixel-shuffle) | "conv" (Pi3X/MoGe progressive upsample)
    depth_head_type: Optional[str] = (
        "conv"  # overrides decoder_head_type for depth decoder; "conv" matches the released stage-2 checkpoint
    )
    finetune_mode: Optional[str] = None  # "depth_output" | "depth_decoder" | "depth_decoder_recurrent" | "all_heads"
    reset_depth_decoder_transformer: bool = False  # after load_pretrained, reinit depth_decoder proj_in/blocks/norm
    # Depth-decoder-only overrides (ray/camera heads stay at the shared decoder_* values so their
    # checkpoint weights keep loading). Leave None to inherit the shared decoder_* defaults.
    depth_decoder_depth: Optional[int] = None
    depth_decoder_embed_dim: Optional[int] = None  # set = embed_dim to remove the input bottleneck
    depth_decoder_num_heads: Optional[int] = None
    decoder_init_values: Optional[float] = None  # LayerScale init for ray+depth decoder blocks (0 / None disables)
    decode_chunk_size: Optional[int] = 128  # chunk size along B*S for ray/depth decoders (None disables)
    use_depth_conf_for_pose: bool = False  # use depth_conf as RANSAC weight in rays_to_pose (default: uniform)
    world_points_from_rays: bool = (
        False  # override WORLD_POINTS <- WORLD_POINTS_DIRECT (= ray_origin + ray_dir * depth)
    )
    loss: Optional[Dict[str, Any]] = None


@dataclass
class MapAnythingModelConfig(ModelConfig):
    """MapAnything model configuration (inference only)."""

    _target_: str = "dvlt.model.mapanything.model.MapAnythingWrapper"
    infer_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class Pi3ModelConfig(ModelConfig):
    """Pi3 model configuration."""

    _target_: str = "dvlt.model.pi3.model.Pi3"
    pos_type: str = "rope100"
    decoder_size: str = "large"
    use_pi3x: bool = False
    use_cam_cond_at_test: bool = False


@dataclass
class DataConfig:
    """Data module configuration."""

    _target_: str = "dvlt.data.module.DataModule"
    train_datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    train_config: Dict[str, Any] = field(default_factory=dict)
    test_datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_config: Dict[str, Any] = field(default_factory=dict)
    image_size: int = 518
    patch_size: int = 14
    tokens_per_batch: int | None = None
    images_per_batch: int = 48
    images_per_element: Any = field(default_factory=lambda: (2, 24))  # Could be tuple or int
    aspect_ratios: Any = field(default_factory=lambda: (0.33, 1.0))  # Could be tuple or float
    train_num_workers: int = 2
    train_prefetch_factor: int = 2
    test_num_workers: int = 2
    collate_fn: str = "dvlt.data.collate.default_collate_fn"  # Reference as string
    infinite_sampling: bool = True
    distributed_eval: bool = True
    pin_memory: bool = False


@dataclass
class UserConfig:
    """User-specific configuration.

    This configuration holds user-specific settings like data paths
    that should not be hardcoded in the experiment configurations.
    """

    data_root: str = MISSING


@dataclass
class TrainerConfig:
    """Trainer configuration."""

    _target_: str = "dvlt.engine.trainer.Trainer"
    # Callbacks
    callbacks: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    # Prediction caching
    load_cached_predictions: bool = False
    write_predictions: bool = False
    # Logging
    output_dir: str = "outputs"
    experiment_name: str = "unnamed"
    timestamp: Optional[str] = None
    logging_dir: str = "logs"
    experiment_logger: Tuple[str, ...] = ("wandb",)
    wandb_project_name: str = "dvlt"
    tqdm: bool = False
    print_step_interval: int = 10
    # Training loop
    seed: Optional[int] = None
    max_train_steps: int = 50_000
    validation_steps: int = 5000
    validation_batches: int = 0
    ckpt_dir: str = ""
    # Debugging
    single_batch_overfit: bool = False
    sanity_check: bool = False
    # DDP
    find_unused_parameters: bool = False
    static_graph: bool = False
    # Checkpointing
    checkpointing_steps: int = 5000
    checkpoints_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None
    # Training optimizations
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    gradient_bucket_view: bool = True
    allow_tf32: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    mixed_precision: str = "no"
    attn_backend: str = "auto"  # "auto", "flash", "fa3"
    set_grads_to_none: bool = False
    # Learning rate & scheduler
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: float = 0.5
    lr_power: float = 1.0
    lr_min_ratio: float = 0.01
    lr_param_group_multipliers: Optional[Dict[str, float]] = None
    scale_lr: Optional[str] = None
    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    # Profiling
    profiler: str = "minimal"
    profiler_dir: str = "profiler_stats"
    pytorch_profile_memory: bool = True
    pytorch_with_stack: bool = True
    pytorch_record_shapes: bool = True


@dataclass
class Config:
    """Main configuration.

    The model field is annotated as the base ModelConfig but can be
    overridden with any configuration class that inherits from ModelConfig.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    user: UserConfig = field(default_factory=UserConfig)


def register_configs():
    """Register structured configs with Hydra's config store."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config, group="schema")
    cs.store(name="base", group="model", node=ModelConfig)
    cs.store(name="vggt", group="model", node=VGGTModelConfig)
    cs.store(name="vggt_omega", group="model", node=VGGTOmegaModelConfig)
    cs.store(name="dvlt", group="model", node=DVLTModelConfig)
    cs.store(name="pi3", group="model", node=Pi3ModelConfig)
    cs.store(name="da3", group="model", node=DA3ModelConfig)
    cs.store(name="mapanything", group="model", node=MapAnythingModelConfig)
    cs.store(name="base", group="data", node=DataConfig)
    cs.store(name="base", group="trainer", node=TrainerConfig)
    cs.store(name="base", group="user", node=UserConfig)
