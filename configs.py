from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    train_subset: int | None = None
    val_subset: int | None = None
    calib_subset: int | None = None
    test_subset: int | None = None
    augment: bool = True
    event_bins: int = 8
    event_resize: int = 32
    seed: int = 0
    val_fraction: float = 0.1
    calibration_fraction: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "resnet"
    num_classes: int = 10
    in_channels: int = 3
    max_time_steps: int = 8
    input_encoding: str = "direct"
    stem_channels: int = 128
    stage_channels: tuple[int, ...] = (128, 256, 512)
    blocks_per_stage: tuple[int, ...] = (3, 3, 2)
    widen_factor: int = 10
    wrn_depth: int = 28
    dropout_rate: float = 0.0
    vgg_classifier_dims: tuple[int, ...] = (4096, 4096)
    stochastic: bool = True
    learnable_sigma: bool = True
    use_dynamic_sigma: bool = True
    use_shortcut_lif: bool = True
    v_threshold: float = 0.5
    lif_tau: float = 2.0
    detach_reset: bool = True
    sigma_init: float = 0.55
    sigma_init_space: str = "raw"
    fixed_sigma_raw: float = -10.0
    normalization_mean: tuple[float, ...] | None = (0.4914, 0.4822, 0.4465)
    normalization_std: tuple[float, ...] | None = (0.2471, 0.2435, 0.2616)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    base_lr: float = 0.125
    sigma_lr_scale: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    use_amp: bool = True
    amp_dtype: str = "float16"
    gradient_clip: float = 1.0
    clean_ratio_start: float = 0.5
    clean_ratio_end: float = 0.1
    entropy_lambda: float = 0.01
    epsilon: float = 4 / 255
    alpha: float = 4 / 255
    random_start_scale: float = 0.1
    use_parseval: bool = True
    parseval_beta: float = 0.004
    parseval_interval: int = 5
    mixed_time_steps: tuple[int, ...] = (8,)
    seed: int = 0
    device: str = field(default_factory=_default_device)
    output_dir: str = "vt_research_suite/outputs"
    checkpoint_name: str = "model.pt"
    eval_interval: int = 1
    selection_metric: str = "clean"


@dataclass(frozen=True)
class AttackConfig:
    attack_name: str = "pgd"
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 10
    random_start: bool = True
    eot_iters: int = 4
    exit_aware: bool = False
    objective: str = "untargeted"
    wrong_exit_weight: float = 0.25
    compute_weight: float = 1.0
    preserve_label_weight: float = 0.5
    force_full_trace: bool = True
    attack_time_steps: int | None = None
    autoattack_version: str | None = None
    autoattack_seed: int | None = 0
    autoattack_verbose: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    vt_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
    fixed_time_steps: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    stopping_rules: tuple[str, ...] = (
        "max_softmax",
        "margin",
        "entropy",
        "temperature_softmax",
        "evidential_surrogate",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    max_batches: int | None = None
    compute_metrics: bool = True
    exact_vt_compute: bool = True
    match_metric: str = "avg_synops"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    vt_enabled: bool = True
    learnable_sigma: bool = False
    use_dynamic_sigma: bool = False
    use_parseval: bool = False
    use_mixed_time_training: bool = False


def default_ablation_specs(max_time_steps: int) -> list[AblationSpec]:
    return [
        AblationSpec("baseline_robust_snn", False, False, False, False, False),
        AblationSpec("baseline_plus_vt", True, False, False, False, False),
        AblationSpec("baseline_plus_vt_plus_dynamic_sigma", True, True, True, False, False),
        AblationSpec("baseline_plus_vt_plus_parseval", True, False, False, True, False),
        AblationSpec("full_model", True, True, True, True, True),
    ]


def apply_ablation(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    spec: AblationSpec,
) -> tuple[ModelConfig, TrainingConfig]:
    mixed_steps = (
        tuple(range(1, model_config.max_time_steps + 1))
        if spec.use_mixed_time_training
        else (model_config.max_time_steps,)
    )
    model_variant = replace(
        model_config,
        learnable_sigma=spec.learnable_sigma,
        use_dynamic_sigma=spec.use_dynamic_sigma,
    )
    training_variant = replace(
        training_config,
        use_parseval=spec.use_parseval,
        mixed_time_steps=mixed_steps,
        checkpoint_name=f"{spec.name}.pt",
    )
    return model_variant, training_variant


def ensure_output_dir(training_config: TrainingConfig) -> Path:
    path = Path(training_config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
