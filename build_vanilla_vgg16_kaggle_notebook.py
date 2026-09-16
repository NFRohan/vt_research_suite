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
OUTPUT_NOTEBOOK = ROOT / "vt_vanilla_vgg16_kaggle_all_in_one.ipynb"


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
            "# VT Research Suite: Vanilla VGG16 Kaggle All-In-One\n\n"
            "This notebook is a separate deterministic VT pipeline built around a vanilla VGG16-style SNN backbone.\n\n"
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

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_vanilla_vgg16_outputs") if Path("/kaggle/working").exists() else Path("vt_vanilla_vgg16_outputs")
DEFAULT_DATA_ROOT = os.environ.get(
    "VT_OFFLINE_CIFAR_ROOT",
    "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data",
)
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

# ---- Training regime toggle (pure ratios, NO annealing) ----
# Run the notebook once per regime; checkpoints/outputs are regime-tagged so they
# never collide.
#   "clean"       -> 100% clean examples (clean_ratio 1.0)
#   "adversarial" -> 100% adversarial    (clean_ratio 0.0)
TRAIN_REGIME = "clean"   # or "adversarial"
if TRAIN_REGIME == "clean":
    REGIME_TAG, REGIME_CLEAN_RATIO = "clean", 1.0
elif TRAIN_REGIME == "adversarial":
    REGIME_TAG, REGIME_CLEAN_RATIO = "adv", 0.0
else:
    raise ValueError(f"TRAIN_REGIME must be 'clean' or 'adversarial', got {TRAIN_REGIME!r}")

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

CALIBRATION_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
ATTACK_BATCH_SIZE = 64
EXACT_VT_BATCH_SIZE = 256

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=128,
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
    architecture="vanilla_vgg16",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    input_encoding="direct",
    stem_channels=64,
    stage_channels=(64, 128, 256, 512, 512),
    blocks_per_stage=(2, 2, 3, 3, 3),
    dropout_rate=0.5,
    vgg_classifier_dims=(4096, 4096),
    stochastic=False,
    learnable_sigma=False,
    use_dynamic_sigma=False,
    use_shortcut_lif=False,
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
    mixed_time_steps=(model_cfg.max_time_steps,),
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name=f"vanilla_vgg16_{REGIME_TAG}_full_model.pt",
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


def _optional_subset_attack_cells() -> list[dict]:
    subset_config_cell = _code_cell(
        """# Optional subset-based fairness / reviewer-stopper reruns.
# Run these after the main clean evaluation + attack sections have completed.

if "selected_threshold_panel" not in globals():
    selected_threshold_panel = pd.read_csv(EXPERIMENT_ROOT / "main_selected_thresholds.csv")

if "selected_threshold" not in globals():
    selected_threshold = float(
        selected_threshold_panel.loc[selected_threshold_panel["threshold_role"] == "selected", "threshold"].iloc[0]
    )

if "test_vt_matches" not in globals():
    test_vt_matches = pd.read_csv(EXPERIMENT_ROOT / "test_vt_fixed_matches.csv")

ATTACK_SUBSET_SAMPLES = 1000
ATTACK_SUBSET_BALANCED = True
ATTACK_SUBSET_SEED = 0
ATTACK_BATCH_SIZE = 100

test_loaders = get_test_loaders(splits)

print(f"Selected threshold: {selected_threshold:.2f}")
print("attack samples:", len(test_loaders["attack"].dataset))
print("attack batch size:", test_loaders["attack"].batch_size)
display(selected_threshold_panel)
display(test_vt_matches[["threshold", "matched_fixed_t"]].drop_duplicates().sort_values("threshold"))
"""
    )

    consistent_selected_cell = _code_cell(
        """subset_rows = []

selected_threshold_only = float(selected_threshold)
matched_fixed_t = int(
    test_vt_matches.loc[test_vt_matches["threshold"] == selected_threshold_only, "matched_fixed_t"].iloc[0]
)

vt_pgd_cfg = replace(
    adaptive_attack_cfg,
    objective="untargeted",
    eot_iters=1,
    exit_aware=True,
)

vt_autoattack_cfg = replace(
    autoattack_cfg,
    eot_iters=1,
    autoattack_version="standard",
    exit_aware=True,
)

fixed_pgd_cfg = replace(vt_pgd_cfg, exit_aware=False)
fixed_autoattack_cfg = replace(vt_autoattack_cfg, exit_aware=False)

fixed_t8_pgd = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="fixed",
    time_steps=model_cfg.max_time_steps,
    attack_config=fixed_pgd_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
fixed_t8_pgd["comparison_family"] = "fixed_T8_subset"
fixed_t8_pgd["matched_fixed_t"] = model_cfg.max_time_steps
fixed_t8_pgd["matched_to_threshold"] = selected_threshold_only
fixed_t8_pgd["threshold_role"] = "fixed_T8_baseline"
subset_rows.append(fixed_t8_pgd)

fixed_t8_auto = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="fixed",
    time_steps=model_cfg.max_time_steps,
    attack_config=fixed_autoattack_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
fixed_t8_auto["comparison_family"] = "fixed_T8_subset"
fixed_t8_auto["matched_fixed_t"] = model_cfg.max_time_steps
fixed_t8_auto["matched_to_threshold"] = selected_threshold_only
fixed_t8_auto["threshold_role"] = "fixed_T8_baseline"
subset_rows.append(fixed_t8_auto)

matched_fixed_pgd = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="fixed",
    time_steps=matched_fixed_t,
    attack_config=fixed_pgd_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
matched_fixed_pgd["comparison_family"] = "matched_fixed_control"
matched_fixed_pgd["matched_fixed_t"] = matched_fixed_t
matched_fixed_pgd["matched_to_threshold"] = selected_threshold_only
matched_fixed_pgd["threshold_role"] = "selected"
subset_rows.append(matched_fixed_pgd)

matched_fixed_auto = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="fixed",
    time_steps=matched_fixed_t,
    attack_config=fixed_autoattack_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
matched_fixed_auto["comparison_family"] = "matched_fixed_control"
matched_fixed_auto["matched_fixed_t"] = matched_fixed_t
matched_fixed_auto["matched_to_threshold"] = selected_threshold_only
matched_fixed_auto["threshold_role"] = "selected"
subset_rows.append(matched_fixed_auto)

vt_selected_pgd = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="vt",
    time_steps=model_cfg.max_time_steps,
    stop_rule=vt_rule,
    threshold=selected_threshold_only,
    attack_config=vt_pgd_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
vt_selected_pgd["comparison_family"] = "vt_selected_subset"
vt_selected_pgd["matched_fixed_t"] = matched_fixed_t
vt_selected_pgd["matched_to_threshold"] = selected_threshold_only
vt_selected_pgd["threshold_role"] = "selected"
subset_rows.append(vt_selected_pgd)

vt_selected_auto = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="vt",
    time_steps=model_cfg.max_time_steps,
    stop_rule=vt_rule,
    threshold=selected_threshold_only,
    attack_config=vt_autoattack_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
vt_selected_auto["comparison_family"] = "vt_selected_subset"
vt_selected_auto["matched_fixed_t"] = matched_fixed_t
vt_selected_auto["matched_to_threshold"] = selected_threshold_only
vt_selected_auto["threshold_role"] = "selected"
subset_rows.append(vt_selected_auto)

subset_compare_df = pd.DataFrame(subset_rows)
persist_frame(subset_compare_df, "subset_consistent_selected_attack_comparison")

display(
    subset_compare_df[
        [
            "comparison_family",
            "mode",
            "time_steps",
            "threshold",
            "matched_fixed_t",
            "threshold_role",
            "matched_to_threshold",
            "attack_name",
            "autoattack_version",
            "attack_objective",
            "accuracy",
            "avg_stopping_depth",
            "avg_executed_steps",
            "attack_eot",
            "exit_aware_attack",
        ]
    ].sort_values(by=["attack_name", "comparison_family"])
)
"""
    )

    sweep_cell = _code_cell(
        """subset_attack_rows = []

vt_pgd_cfg = replace(
    adaptive_attack_cfg,
    objective="untargeted",
    eot_iters=1,
    exit_aware=True,
)
vt_autoattack_cfg = replace(
    autoattack_cfg,
    eot_iters=1,
    autoattack_version="standard",
    exit_aware=True,
)
fixed_pgd_cfg = replace(vt_pgd_cfg, exit_aware=False)
fixed_autoattack_cfg = replace(vt_autoattack_cfg, exit_aware=False)

for t in range(1, model_cfg.max_time_steps + 1):
    fixed_pgd = evaluate_policy(
        model,
        test_loaders["attack"],
        mode="fixed",
        time_steps=t,
        attack_config=fixed_pgd_cfg,
        compute_metrics=False,
        exact_vt_compute=False,
        device=training_cfg.device,
        max_batches=eval_cfg.max_batches,
    )
    fixed_pgd["comparison_family"] = "fixed_t_sweep_subset"
    fixed_pgd["matched_fixed_t"] = t
    fixed_pgd["matched_to_threshold"] = None
    fixed_pgd["threshold_role"] = f"fixed_T{t}"
    subset_attack_rows.append(fixed_pgd)

    fixed_auto = evaluate_policy(
        model,
        test_loaders["attack"],
        mode="fixed",
        time_steps=t,
        attack_config=fixed_autoattack_cfg,
        compute_metrics=False,
        exact_vt_compute=False,
        device=training_cfg.device,
        max_batches=eval_cfg.max_batches,
    )
    fixed_auto["comparison_family"] = "fixed_t_sweep_subset"
    fixed_auto["matched_fixed_t"] = t
    fixed_auto["matched_to_threshold"] = None
    fixed_auto["threshold_role"] = f"fixed_T{t}"
    subset_attack_rows.append(fixed_auto)

for _, threshold_row in selected_threshold_panel.iterrows():
    threshold = float(threshold_row["threshold"])
    matched_t = int(test_vt_matches.loc[test_vt_matches["threshold"] == threshold, "matched_fixed_t"].iloc[0])

    vt_pgd = evaluate_policy(
        model,
        test_loaders["attack"],
        mode="vt",
        time_steps=model_cfg.max_time_steps,
        stop_rule=vt_rule,
        threshold=threshold,
        attack_config=vt_pgd_cfg,
        compute_metrics=False,
        exact_vt_compute=False,
        device=training_cfg.device,
        max_batches=eval_cfg.max_batches,
    )
    vt_pgd["comparison_family"] = "vt_threshold_subset"
    vt_pgd["matched_fixed_t"] = matched_t
    vt_pgd["matched_to_threshold"] = threshold
    vt_pgd["threshold_role"] = threshold_row["threshold_role"]
    subset_attack_rows.append(vt_pgd)

    vt_auto = evaluate_policy(
        model,
        test_loaders["attack"],
        mode="vt",
        time_steps=model_cfg.max_time_steps,
        stop_rule=vt_rule,
        threshold=threshold,
        attack_config=vt_autoattack_cfg,
        compute_metrics=False,
        exact_vt_compute=False,
        device=training_cfg.device,
        max_batches=eval_cfg.max_batches,
    )
    vt_auto["comparison_family"] = "vt_threshold_subset"
    vt_auto["matched_fixed_t"] = matched_t
    vt_auto["matched_to_threshold"] = threshold
    vt_auto["threshold_role"] = threshold_row["threshold_role"]
    subset_attack_rows.append(vt_auto)

subset_attack_df = pd.DataFrame(subset_attack_rows)
persist_frame(subset_attack_df, "subset_consistent_fixed_t_and_vt_attack_sweep")

display(
    subset_attack_df[
        [
            "comparison_family",
            "mode",
            "time_steps",
            "threshold",
            "matched_fixed_t",
            "threshold_role",
            "attack_name",
            "autoattack_version",
            "attack_objective",
            "accuracy",
            "avg_stopping_depth",
            "avg_executed_steps",
            "attack_eot",
            "exit_aware_attack",
        ]
    ].sort_values(
        by=["attack_name", "comparison_family", "time_steps", "threshold"],
        na_position="last",
    )
)
"""
    )

    return [
        _markdown_cell(
            "## Optional Fairness Checks\n\n"
            "These cells reproduce the chat-added robustness checks:\n"
            "- a consistent selected-point subset comparison (`T=8`, matched fixed-$T$, selected VT)\n"
            "- a full fixed-$T$ and VT threshold sweep on the same subset to answer the reviewer's `wrong T` objection"
        ),
        subset_config_cell,
        _markdown_cell("### Selected-Point Fair Subset Comparison"),
        consistent_selected_cell,
        _markdown_cell("### Reviewer-Stopper Fixed-T / VT Sweep"),
        sweep_cell,
    ]


def build_notebook() -> dict:
    source_cells = _load_source_cells()
    cells = _base_notebook_cells()
    cells.extend(
        [
            _markdown_cell(
                "# Experiment Driver\n\n"
                "This driver reuses the full VT training and evaluation workflow, but the defaults are "
                "specialized for a deterministic vanilla VGG16 SNN."
            ),
            _driver_imports_cell(),
            _config_cell(),
        ]
    )
    for cell in source_cells[3:]:
        cells.append(cell)
    cells.extend(_optional_subset_attack_cells())
    return _notebook_payload(cells)


def main() -> None:
    OUTPUT_NOTEBOOK.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
