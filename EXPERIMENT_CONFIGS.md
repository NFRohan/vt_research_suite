# Experiment Configs

Copy-paste these blocks into the main config cell of [vt_research_direction_kaggle_all_in_one.ipynb](/d:/GitHub/Brain-Cog/examples/Snn_safety/RandHet-SNN/vt_research_suite/vt_research_direction_kaggle_all_in_one.ipynb).

Notes:
- For final runs, keep `exact_vt_compute=True` and `match_metric="avg_synops"`.
- If `REQUIRE_AUTOATTACK_FOR_FINAL=True`, make sure the install cell has `RUN_OPTIONAL_INSTALLS = True` or the packages are already available.
- On H100, the `batch_size=128` configs below are a reasonable starting point.

## ResNet Final Run

```python
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")
DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

RUN_TRAINING = True
RUN_ABLATION_TRAINING = False
RUN_ARCHITECTURE_TRAINING = False
RUN_MULTI_SEED = False
RUN_EXTRA_DATASETS = False
PERSIST_OUTPUTS = True
REQUIRE_AUTOATTACK_FOR_FINAL = True
SHOW_NOTEBOOK_PROGRESS = True

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=128,
    num_workers=2,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=None,
    calib_subset=None,
    test_subset=None,
)

model_cfg = ModelConfig(
    architecture="resnet",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    stem_channels=128,
    stage_channels=(128, 256, 512),
    blocks_per_stage=(3, 3, 2),
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=100,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name="resnet_full_model.pt",
    mixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    selection_metric="clean_plus_robust",
    use_amp=True,
    eval_interval=1,
    epsilon=8/255,
    alpha=8/255,
)

eval_cfg = EvaluationConfig(
    vt_thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    fixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    compute_metrics=True,
    exact_vt_compute=True,
    match_metric="avg_synops",
    max_batches=None,
    seeds=(0, 1, 2),
)

selection_attack_cfg = AttackConfig(epsilon=8/255, alpha=2/255, steps=10, eot_iters=1)
adaptive_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=20,
    eot_iters=8,
    exit_aware=True,
    force_full_trace=True,
    attack_time_steps=model_cfg.max_time_steps,
)
compute_attack_cfg = replace(adaptive_attack_cfg, objective="compute_overuse", preserve_label_weight=1.0)
autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8/255,
    exit_aware=adaptive_attack_cfg.exit_aware,
    force_full_trace=adaptive_attack_cfg.force_full_trace,
    attack_time_steps=model_cfg.max_time_steps,
    eot_iters=adaptive_attack_cfg.eot_iters,
)
```

## VGG Final Run

```python
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")
DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

RUN_TRAINING = True
RUN_ABLATION_TRAINING = False
RUN_ARCHITECTURE_TRAINING = False
RUN_MULTI_SEED = False
RUN_EXTRA_DATASETS = False
PERSIST_OUTPUTS = True
REQUIRE_AUTOATTACK_FOR_FINAL = True
SHOW_NOTEBOOK_PROGRESS = True

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=128,
    num_workers=2,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=None,
    calib_subset=None,
    test_subset=None,
)

model_cfg = ModelConfig(
    architecture="vgg",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    stem_channels=64,
    stage_channels=(64, 128, 256, 512, 512),
    blocks_per_stage=(1, 1, 2, 2, 2),
    dropout_rate=0.5,
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=100,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name="vgg_full_model.pt",
    mixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    selection_metric="clean_plus_robust",
    use_amp=True,
    eval_interval=1,
    epsilon=8/255,
    alpha=8/255,
)

eval_cfg = EvaluationConfig(
    vt_thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    fixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    compute_metrics=True,
    exact_vt_compute=True,
    match_metric="avg_synops",
    max_batches=None,
    seeds=(0, 1, 2),
)

selection_attack_cfg = AttackConfig(epsilon=8/255, alpha=2/255, steps=10, eot_iters=1)
adaptive_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=20,
    eot_iters=8,
    exit_aware=True,
    force_full_trace=True,
    attack_time_steps=model_cfg.max_time_steps,
)
compute_attack_cfg = replace(adaptive_attack_cfg, objective="compute_overuse", preserve_label_weight=1.0)
autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8/255,
    exit_aware=adaptive_attack_cfg.exit_aware,
    force_full_trace=adaptive_attack_cfg.force_full_trace,
    attack_time_steps=model_cfg.max_time_steps,
    eot_iters=adaptive_attack_cfg.eot_iters,
)
```

## WRN Final Run

```python
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")
DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

RUN_TRAINING = True
RUN_ABLATION_TRAINING = False
RUN_ARCHITECTURE_TRAINING = False
RUN_MULTI_SEED = False
RUN_EXTRA_DATASETS = False
PERSIST_OUTPUTS = True
REQUIRE_AUTOATTACK_FOR_FINAL = True
SHOW_NOTEBOOK_PROGRESS = True

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=128,
    num_workers=2,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=None,
    calib_subset=None,
    test_subset=None,
)

model_cfg = ModelConfig(
    architecture="wrn",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    stem_channels=16,
    widen_factor=10,
    wrn_depth=28,
    stage_channels=(160, 320, 640),
    blocks_per_stage=(4, 4, 4),
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=100,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name="wrn_full_model.pt",
    mixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    selection_metric="clean_plus_robust",
    use_amp=True,
    eval_interval=1,
    epsilon=8/255,
    alpha=8/255,
)

eval_cfg = EvaluationConfig(
    vt_thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    fixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    compute_metrics=True,
    exact_vt_compute=True,
    match_metric="avg_synops",
    max_batches=None,
    seeds=(0, 1, 2),
)

selection_attack_cfg = AttackConfig(epsilon=8/255, alpha=2/255, steps=10, eot_iters=1)
adaptive_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=20,
    eot_iters=8,
    exit_aware=True,
    force_full_trace=True,
    attack_time_steps=model_cfg.max_time_steps,
)
compute_attack_cfg = replace(adaptive_attack_cfg, objective="compute_overuse", preserve_label_weight=1.0)
autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8/255,
    exit_aware=adaptive_attack_cfg.exit_aware,
    force_full_trace=adaptive_attack_cfg.force_full_trace,
    attack_time_steps=model_cfg.max_time_steps,
    eot_iters=adaptive_attack_cfg.eot_iters,
)
```

## Cross-Architecture Sweep

Start from the `ResNet Final Run` block above, then change only:

```python
RUN_ARCHITECTURE_TRAINING = True
RUN_TRAINING = True
RUN_MULTI_SEED = False
RUN_ABLATION_TRAINING = False
```

The architecture sweep cell will train or load `resnet`, `vgg`, and `wrn` automatically.

## Multi-Seed Final Run

Start from your chosen single-backbone block above, then change only:

```python
RUN_MULTI_SEED = True
RUN_TRAINING = True
RUN_ARCHITECTURE_TRAINING = False
RUN_ABLATION_TRAINING = False

eval_cfg = replace(
    eval_cfg,
    seeds=(0, 1, 2),
    exact_vt_compute=True,
    match_metric="avg_synops",
)
```

## CIFAR-100 Variant

Start from any single-backbone block above, then change only:

```python
dataset_cfg = replace(dataset_cfg, name="cifar100")
model_cfg = replace(model_cfg, num_classes=100)
training_cfg = replace(training_cfg, checkpoint_name="your_backbone_cifar100.pt")
```

## Quick Debug Run

```python
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")
DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

RUN_TRAINING = True
RUN_ABLATION_TRAINING = False
RUN_ARCHITECTURE_TRAINING = False
RUN_MULTI_SEED = False
RUN_EXTRA_DATASETS = False
PERSIST_OUTPUTS = True
REQUIRE_AUTOATTACK_FOR_FINAL = False
SHOW_NOTEBOOK_PROGRESS = True

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=64,
    num_workers=2,
    seed=0,
    augment=True,
    train_subset=5000,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=1000,
    calib_subset=1000,
    test_subset=1000,
)

model_cfg = ModelConfig(
    architecture="resnet",
    num_classes=10,
    in_channels=3,
    max_time_steps=4,
    stem_channels=64,
    stage_channels=(64, 128, 256),
    blocks_per_stage=(1, 1, 1),
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=10,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name="resnet_debug.pt",
    mixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    selection_metric="clean_plus_robust",
    use_amp=True,
    eval_interval=1,
    epsilon=8/255,
    alpha=8/255,
)

eval_cfg = EvaluationConfig(
    vt_thresholds=(0.6, 0.7, 0.8, 0.9),
    fixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    compute_metrics=True,
    exact_vt_compute=False,
    match_metric="avg_synops",
    max_batches=20,
    seeds=(0, 1, 2),
)

selection_attack_cfg = AttackConfig(epsilon=8/255, alpha=2/255, steps=3, eot_iters=1)
adaptive_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=5,
    eot_iters=2,
    exit_aware=True,
    force_full_trace=True,
    attack_time_steps=model_cfg.max_time_steps,
)
compute_attack_cfg = replace(adaptive_attack_cfg, objective="compute_overuse", preserve_label_weight=1.0)
autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8/255,
    exit_aware=adaptive_attack_cfg.exit_aware,
    force_full_trace=adaptive_attack_cfg.force_full_trace,
    attack_time_steps=model_cfg.max_time_steps,
    eot_iters=adaptive_attack_cfg.eot_iters,
)
```
