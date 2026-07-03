from __future__ import annotations

import json
from pathlib import Path

from build_split_kaggle_notebooks import (
    _code_cell,
    _driver_imports_cell,
    _load_source_cells,
    _markdown_cell,
    _notebook_payload,
    _read_module,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "vt_vanilla_resnet18_kaggle_all_in_one.ipynb"


def _optional_install_cell() -> dict:
    return _code_cell(
        """RUN_OPTIONAL_INSTALLS = False

OPTIONAL_PACKAGES = {
    "autoattack": ("git+https://github.com/fra31/auto-attack.git", True),
}

if RUN_OPTIONAL_INSTALLS:
    import subprocess
    import sys

    for module_name, install_spec in OPTIONAL_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            install_target, no_deps = install_spec
            command = [sys.executable, "-m", "pip", "install", "-q", install_target]
            if no_deps:
                command.append("--no-deps")
            print(f"Installing {install_target} ...")
            subprocess.check_call(command)
"""
    )


def _base_notebook_cells() -> list[dict]:
    module_order = [
        "configs",
        "data",
        "stopping",
        "model",
        "metrics",
        "attacks",
        "training",
        "experiments",
    ]

    cells = [
        _markdown_cell(
            "# VT Research Suite: Vanilla ResNet-18 Kaggle All-In-One\n\n"
            "This notebook is a separate deterministic VT pipeline built around a vanilla ResNet-18-style SNN backbone.\n\n"
            "Key choices:\n"
            "- deterministic LIF neurons\n"
            "- direct input encoding\n"
            "- no stochastic noise\n"
            "- no Parseval regularization\n"
            "- non-EOT PGD-20 and standard AutoAttack\n"
            "- full calibration and clean evaluation, with a class-balanced 2000-sample attack subset"
        ),
        _optional_install_cell(),
    ]

    for module_name in module_order:
        pretty_name = module_name.replace("_", " ").title()
        cells.append(_markdown_cell(f"## {pretty_name}"))
        _, source = _read_module(module_name)
        cells.append(_code_cell(source))
    return cells


def _config_cell() -> dict:
    return _code_cell(
        """from dataclasses import replace

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_vanilla_resnet18_outputs") if Path("/kaggle/working").exists() else Path("vt_vanilla_resnet18_outputs")
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

RUN_FULL_TEST_THRESHOLD_SWEEP = True
RUN_MULTI_THRESHOLD_ATTACKS = True
RUN_AUTOATTACK_ON_ALL_SELECTED_THRESHOLDS = False
SELECTED_THRESHOLD_COUNT = 3

ATTACK_SUBSET_SAMPLES = 2000
ATTACK_SUBSET_BALANCED = True
ATTACK_SUBSET_SEED = 0

CALIBRATION_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
ATTACK_BATCH_SIZE = 128
EXACT_VT_BATCH_SIZE = 512

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=256,
    num_workers=4,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=None,
    calib_subset=None,
    test_subset=None,
)

model_cfg = ModelConfig(
    architecture="vanilla_resnet18",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    input_encoding="direct",
    stem_channels=64,
    stage_channels=(64, 128, 256, 512),
    blocks_per_stage=(2, 2, 2, 2),
    stochastic=False,
    learnable_sigma=False,
    use_dynamic_sigma=False,
    use_shortcut_lif=True,
    v_threshold=0.5,
    lif_tau=2.0,
    detach_reset=True,
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=100,
    base_lr=0.1,
    sigma_lr_scale=0.0,
    momentum=0.9,
    weight_decay=1e-4,
    warmup_epochs=5,
    use_amp=True,
    amp_dtype="float16",
    gradient_clip=1.0,
    clean_ratio_start=0.5,
    clean_ratio_end=0.1,
    entropy_lambda=0.01,
    epsilon=8/255,
    alpha=8/255,
    random_start_scale=0.1,
    use_parseval=False,
    parseval_beta=0.0,
    parseval_interval=5,
    mixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name="vanilla_resnet18_full_model.pt",
    eval_interval=5,
    selection_metric="clean",
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

selection_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=4,
    eot_iters=1,
)

adaptive_attack_cfg = AttackConfig(
    epsilon=8/255,
    alpha=2/255,
    steps=20,
    eot_iters=1,
    exit_aware=True,
    force_full_trace=True,
    attack_time_steps=model_cfg.max_time_steps,
)

compute_attack_cfg = replace(
    adaptive_attack_cfg,
    objective="compute_overuse",
    preserve_label_weight=1.0,
)

autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8/255,
    exit_aware=adaptive_attack_cfg.exit_aware,
    force_full_trace=adaptive_attack_cfg.force_full_trace,
    attack_time_steps=model_cfg.max_time_steps,
    eot_iters=1,
    autoattack_version="standard",
)

print(dataset_cfg)
print(model_cfg)
print(training_cfg)
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else 'cpu'}")
print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark if torch.cuda.is_available() else 'cpu'}")
print(f"Main-table VT compute mode: {'sample_exact' if eval_cfg.exact_vt_compute else 'batched_proxy'}")
print(f"Main-table fixed-T match metric: {eval_cfg.match_metric}")
print(f"AutoAttack version policy: {autoattack_cfg.autoattack_version}")
print(f"Selected-threshold panel size: {SELECTED_THRESHOLD_COUNT}")
print(
    f"Eval loader overrides -> calib: {CALIBRATION_BATCH_SIZE or dataset_cfg.batch_size}, "
    f"test: {TEST_BATCH_SIZE or dataset_cfg.batch_size}, "
    f"attack: {ATTACK_BATCH_SIZE or TEST_BATCH_SIZE or dataset_cfg.batch_size}, "
    f"exact_vt: {EXACT_VT_BATCH_SIZE or CALIBRATION_BATCH_SIZE or TEST_BATCH_SIZE or dataset_cfg.batch_size}"
)
print(
    f"Attack subset -> samples: {ATTACK_SUBSET_SAMPLES}, "
    f"balanced: {ATTACK_SUBSET_BALANCED}, seed: {ATTACK_SUBSET_SEED}"
)
"""
    )


def build_notebook() -> dict:
    source_cells = _load_source_cells()
    cells = _base_notebook_cells()
    cells.extend(
        [
            _markdown_cell(
                "# Experiment Driver\n\n"
                "This driver reuses the full VT training and evaluation workflow, but the defaults are "
                "specialized for a deterministic vanilla ResNet-18 SNN."
            ),
            _driver_imports_cell(),
            _config_cell(),
        ]
    )
    for cell in source_cells[3:]:
        cells.append(cell)
    return _notebook_payload(cells)


def main() -> None:
    OUTPUT_NOTEBOOK.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
