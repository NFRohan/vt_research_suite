from __future__ import annotations

import copy
import json
import re
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

# Offline Kaggle support: attach a dataset named `offline-src` containing:
# - autoattack-minimal-kaggle.zip
# - cifar10-offline-kaggle.zip
from pathlib import Path
import os
import sys
import tarfile
import zipfile

OFFLINE_SRC_CANDIDATES = [
    Path("/kaggle/input/offline-src"),
    Path("/kaggle/input/datasets/ihatethesis/offline-src"),
]
OFFLINE_SRC_ROOT = next((path for path in OFFLINE_SRC_CANDIDATES if path.exists()), None)

if OFFLINE_SRC_ROOT is not None:
    print(f"Using offline source dataset: {OFFLINE_SRC_ROOT}")
    autoattack_candidates = list(OFFLINE_SRC_ROOT.rglob("autoattack-minimal-kaggle.zip"))
    autoattack_package_candidates = list(OFFLINE_SRC_ROOT.rglob("autoattack/__init__.py"))
    if autoattack_package_candidates:
        autoattack_package_parent = autoattack_package_candidates[0].parent.parent
        sys.path.insert(0, str(autoattack_package_parent))
        try:
            from autoattack import AutoAttack
            print(f"AutoAttack import OK from extracted offline dataset: {autoattack_package_parent}")
        except Exception as exc:
            print(f"AutoAttack extracted import failed: {exc!r}")
    elif autoattack_candidates:
        autoattack_zip = autoattack_candidates[0]
        autoattack_dir = Path("/kaggle/working/autoattack-minimal")
        if not autoattack_dir.exists():
            with zipfile.ZipFile(autoattack_zip, "r") as zf:
                zf.extractall(autoattack_dir)
        sys.path.insert(0, str(autoattack_dir))
        try:
            from autoattack import AutoAttack
            print(f"AutoAttack import OK from offline dataset: {autoattack_zip}")
        except Exception as exc:
            print(f"AutoAttack offline import failed: {exc!r}")
    else:
        print("offline-src found, but no AutoAttack zip or extracted autoattack package was found.")

    cifar_candidates = list(OFFLINE_SRC_ROOT.rglob("cifar10-offline-kaggle.zip"))
    cifar_batch_candidates = list(OFFLINE_SRC_ROOT.rglob("cifar-10-batches-py"))
    if cifar_batch_candidates:
        cifar_batch_dir = cifar_batch_candidates[0]
        data_root = cifar_batch_dir.parent
        os.environ["VT_OFFLINE_CIFAR_ROOT"] = str(data_root)
        print(f"CIFAR-10 offline data found at: {data_root}")
    elif cifar_candidates:
        cifar_zip = cifar_candidates[0]
        cifar_work = Path("/kaggle/working/cifar10-offline")
        data_root = Path("/kaggle/working/data")
        if not cifar_work.exists():
            with zipfile.ZipFile(cifar_zip, "r") as zf:
                zf.extractall(cifar_work)
        tar_path = cifar_work / "cifar-10-python.tar.gz"
        if tar_path.exists() and not (data_root / "cifar-10-batches-py").exists():
            data_root.mkdir(exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(data_root)
        os.environ["VT_OFFLINE_CIFAR_ROOT"] = str(data_root)
        print(f"CIFAR-10 offline data ready at: {data_root}")
    else:
        print("offline-src found, but no CIFAR-10 zip or extracted cifar-10-batches-py folder was found.")
else:
    print("offline-src dataset not attached/found; falling back to existing Kaggle/working paths.")

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
DEFAULT_DATA_ROOT = os.environ.get(
    "VT_OFFLINE_CIFAR_ROOT",
    "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data",
)
# ---- Training regime toggle (pure ratios, NO annealing) ----
#   "clean"       -> 100% clean examples (clean_ratio 1.0)
#   "adversarial" -> 100% adversarial    (clean_ratio 0.0)
TRAIN_REGIME = "adversarial"   # or "clean"
if TRAIN_REGIME == "clean":
    REGIME_TAG, REGIME_CLEAN_RATIO = "clean", 1.0
elif TRAIN_REGIME == "adversarial":
    REGIME_TAG, REGIME_CLEAN_RATIO = "adv", 0.0
else:
    raise ValueError(f"TRAIN_REGIME must be 'clean' or 'adversarial', got {TRAIN_REGIME!r}")

# ---- Training time-step schedule toggle ----
#   "fixed_t8" -> train ONLY at T=8. This is the paper's stated methodology
#                 ("trained at fixed T=8") and matches the original checkpoints.
#   "mixed"    -> train across T=1..8. Produces a stronger backbone, but the short
#                 fixed-T controls also get strong, which shrinks VT's relative gain.
TRAIN_SCHEDULE = "fixed_t8"   # or "mixed"
if TRAIN_SCHEDULE == "fixed_t8":
    SCHED_TAG = "T8"
elif TRAIN_SCHEDULE == "mixed":
    SCHED_TAG = "mixT"
else:
    raise ValueError(f"TRAIN_SCHEDULE must be 'fixed_t8' or 'mixed', got {TRAIN_SCHEDULE!r}")

# Outputs are tagged by regime AND schedule so all four runs coexist without
# overwriting each other's CSVs.
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT / f"{REGIME_TAG}_{SCHED_TAG}"
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

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
    clean_ratio_start=REGIME_CLEAN_RATIO,
    clean_ratio_end=REGIME_CLEAN_RATIO,
    entropy_lambda=0.0,
    epsilon=8/255,
    alpha=8/255,
    random_start_scale=0.1,
    use_parseval=False,
    parseval_beta=0.0,
    parseval_interval=5,
    mixed_time_steps=((model_cfg.max_time_steps,) if TRAIN_SCHEDULE == "fixed_t8"
                      else tuple(range(1, model_cfg.max_time_steps + 1))),
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name=f"vanilla_resnet18_{REGIME_TAG}_{SCHED_TAG}_full_model.pt",
    eval_interval=5,
    selection_metric="clean",
)

print(f"Training regime : {TRAIN_REGIME} (clean_ratio={REGIME_CLEAN_RATIO})")
print(f"Time schedule   : {TRAIN_SCHEDULE} (mixed_time_steps={training_cfg.mixed_time_steps})")
print(f"Checkpoint      : {training_cfg.checkpoint_name}")
print(f"Output root     : {EXPERIMENT_ROOT}")

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


PIN_REGIME_RE = 'TRAIN_REGIME = "[a-z_]+".*'
PIN_SCHED_RE = 'TRAIN_SCHEDULE = "[a-z_0-9]+".*'


def _pin_variant(notebook: dict, regime: str, schedule: str, note: str) -> dict:
    """Copy the notebook with TRAIN_REGIME and TRAIN_SCHEDULE hard-pinned."""
    nb = copy.deepcopy(notebook)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "REGIME_CLEAN_RATIO" not in src:
            continue
        src = re.sub(PIN_REGIME_RE, f'TRAIN_REGIME = "{regime}"   # {note}', src, count=1)
        src = re.sub(PIN_SCHED_RE, f'TRAIN_SCHEDULE = "{schedule}"   # {note}', src, count=1)
        cell["source"] = src.splitlines(keepends=True)
    return nb


def main() -> None:
    base = build_notebook()
    variants = [
        ("clean", "fixed_t8", "clean_T8",
         "pinned: 100% CLEAN training, fixed T=8 only (paper methodology)"),
        ("clean", "mixed", "clean_mixT",
         "pinned: 100% CLEAN training, mixed T=1..8 schedule"),
        ("adversarial", "fixed_t8", "adv_T8",
         "pinned: 100% ADVERSARIAL training, fixed T=8 only (paper methodology)"),
        ("adversarial", "mixed", "adv_mixT",
         "pinned: 100% ADVERSARIAL training, mixed T=1..8 schedule"),
    ]
    for regime, schedule, tag, note in variants:
        nb = _pin_variant(base, regime, schedule, note)
        out_path = ROOT / f"vt_vanilla_resnet18_{tag}_kaggle.ipynb"
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
