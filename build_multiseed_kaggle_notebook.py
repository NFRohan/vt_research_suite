from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODULE_ORDER = [
    "configs",
    "data",
    "stopping",
    "model",
    "metrics",
    "attacks",
    "training",
    "experiments",
]

# One multi-seed notebook per backbone. Each `model_cfg` is copied verbatim from
# that backbone's vanilla all-in-one builder so load_state_dict matches the paper
# checkpoints. `batch_size` is sized to the backbone's memory footprint.
ARCH_SPECS = {
    "resnet18": {
        "arch": "vanilla_resnet18",
        "label": "vanilla ResNet-18",
        "batch_size": 128,
        "model_cfg": '''model_cfg = ModelConfig(
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
)''',
    },
    "vgg16": {
        "arch": "vanilla_vgg16",
        "label": "vanilla VGG16",
        "batch_size": 128,
        "model_cfg": '''model_cfg = ModelConfig(
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
)''',
    },
    "mobilenetv2": {
        "arch": "vanilla_mobilenetv2",
        "label": "vanilla MobileNetV2",
        "batch_size": 256,
        "model_cfg": '''model_cfg = ModelConfig(
    architecture="vanilla_mobilenetv2",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    input_encoding="direct",
    stem_channels=32,
    stage_channels=(16, 24, 32, 64, 96, 160, 320),
    blocks_per_stage=(1, 2, 3, 4, 3, 3, 1),
    dropout_rate=0.0,
    stochastic=False,
    learnable_sigma=False,
    use_dynamic_sigma=False,
    use_shortcut_lif=False,
    v_threshold=0.5,
    lif_tau=2.0,
    detach_reset=True,
    sigma_init_space="raw",
)''',
    },
}


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _strip_relative_imports(source: str) -> str:
    patterns = [rf"^from \.{name} import .*\n" for name in MODULE_ORDER]
    for pattern in patterns:
        source = re.sub(pattern, "", source, flags=re.MULTILINE)
    source = source.replace("    from .experiments import evaluate_policy\n", "")
    return source


def _read_module(name: str) -> str:
    source = (ROOT / f"{name}.py").read_text(encoding="utf-8")
    return _strip_relative_imports(source).strip() + "\n"


# ---------------------------------------------------------------------------
# Offline setup: stage CIFAR-10 into the torchvision layout and make the
# AutoAttack package importable from an attached Kaggle dataset (internet off).
# This cell MUST run before the inlined `attacks` module cell, because
# attacks.py resolves `from autoattack import AutoAttack` at import time.
# ---------------------------------------------------------------------------
OFFLINE_SETUP = r'''# Offline Kaggle support (mirrors the vanilla all-in-one notebooks): attach a
# dataset named `offline-src` containing:
#   - autoattack-minimal-kaggle.zip  (or an already-extracted autoattack/ package)
#   - cifar10-offline-kaggle.zip     (or an already-extracted cifar-10-batches-py/)
# This cell stages both so the inlined modules run with internet OFF. It MUST run
# before the inlined `attacks` cell, which resolves `from autoattack import ...`.
import os
import sys
import tarfile
import zipfile
from pathlib import Path

RUN_OPTIONAL_INSTALLS = False  # internet is off; keep the git-install path disabled
HAS_AUTOATTACK = False

OFFLINE_SRC_CANDIDATES = [
    Path("/kaggle/input/offline-src"),
    Path("/kaggle/input/datasets/ihatethesis/offline-src"),
]
OFFLINE_SRC_ROOT = next((path for path in OFFLINE_SRC_CANDIDATES if path.exists()), None)

if OFFLINE_SRC_ROOT is not None:
    print(f"Using offline source dataset: {OFFLINE_SRC_ROOT}")

    # --- AutoAttack ---
    autoattack_package_candidates = list(OFFLINE_SRC_ROOT.rglob("autoattack/__init__.py"))
    autoattack_zip_candidates = list(OFFLINE_SRC_ROOT.rglob("autoattack-minimal-kaggle.zip"))
    if autoattack_package_candidates:
        autoattack_parent = autoattack_package_candidates[0].parent.parent
        sys.path.insert(0, str(autoattack_parent))
        try:
            from autoattack import AutoAttack  # noqa: F401
            HAS_AUTOATTACK = True
            print(f"AutoAttack import OK from extracted offline dataset: {autoattack_parent}")
        except Exception as exc:  # noqa: BLE001
            print(f"AutoAttack extracted import failed: {exc!r}")
    elif autoattack_zip_candidates:
        autoattack_dir = Path("/kaggle/working/autoattack-minimal")
        if not autoattack_dir.exists():
            with zipfile.ZipFile(autoattack_zip_candidates[0], "r") as zf:
                zf.extractall(autoattack_dir)
        sys.path.insert(0, str(autoattack_dir))
        try:
            from autoattack import AutoAttack  # noqa: F401
            HAS_AUTOATTACK = True
            print(f"AutoAttack import OK from offline dataset: {autoattack_zip_candidates[0]}")
        except Exception as exc:  # noqa: BLE001
            print(f"AutoAttack offline import failed: {exc!r}")
    else:
        print("offline-src found, but no AutoAttack zip or extracted autoattack package was found.")

    # --- CIFAR-10 (sets VT_OFFLINE_CIFAR_ROOT read by the config cell) ---
    cifar_batch_candidates = list(OFFLINE_SRC_ROOT.rglob("cifar-10-batches-py"))
    cifar_zip_candidates = list(OFFLINE_SRC_ROOT.rglob("cifar10-offline-kaggle.zip"))
    if cifar_batch_candidates:
        data_root = cifar_batch_candidates[0].parent
        os.environ["VT_OFFLINE_CIFAR_ROOT"] = str(data_root)
        print(f"CIFAR-10 offline data found at: {data_root}")
    elif cifar_zip_candidates:
        cifar_work = Path("/kaggle/working/cifar10-offline")
        data_root = Path("/kaggle/working/data")
        if not cifar_work.exists():
            with zipfile.ZipFile(cifar_zip_candidates[0], "r") as zf:
                zf.extractall(cifar_work)
        tar_path = cifar_work / "cifar-10-python.tar.gz"
        if tar_path.exists() and not (data_root / "cifar-10-batches-py").exists():
            data_root.mkdir(exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(data_root)
        os.environ["VT_OFFLINE_CIFAR_ROOT"] = str(data_root)
        print(f"CIFAR-10 offline data ready at: {data_root}")
    else:
        print("offline-src found, but no CIFAR-10 zip or extracted cifar-10-batches-py was found.")
else:
    print("offline-src dataset not attached/found; falling back to existing Kaggle/working paths.")

print(f"HAS_AUTOATTACK = {HAS_AUTOATTACK}")
'''


DRIVER_IMPORTS = r'''import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 200)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DEFAULT_OUTPUT_ROOT = (
    Path("/kaggle/working/vt_research_suite_outputs")
    if Path("/kaggle/working").exists()
    else Path("vt_research_suite_outputs")
)
# Honors VT_OFFLINE_CIFAR_ROOT exported by the offline-setup cell (same as the
# vanilla all-in-one notebooks); falls back to the standard working path.
DEFAULT_DATA_ROOT = os.environ.get(
    "VT_OFFLINE_CIFAR_ROOT",
    "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data",
)
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | data root: {DEFAULT_DATA_ROOT}")
'''


# The config cell is deliberately the FIRST thing to edit. Defaults match the
# adversarially-trained vanilla ResNet-18 case study from the paper.
DRIVER_CONFIG = r'''# ======================= EDIT THIS CELL =======================

# --- Training (Job 0). You have no checkpoint yet, so train at least seed 0. ---
# Resumable across Kaggle sessions: attach a previous session's output (or list
# paths in CHECKPOINTS below) and only the missing seeds are trained.
RUN_TRAINING = True
TRAIN_SEEDS = (0, 1, 2)   # 3 seeds: needed to quantify training run-to-run variance.
                     # (0, 1, 2) -> three checkpoints, also enables Job 2.
                     # NOTE: ~100 training epochs per seed. Training 3 seeds may
                     # exceed one Kaggle session; run seeds across sessions and
                     # re-attach the outputs (missing-only training makes this safe).

# Training regime. The paper compares BOTH, each at a pure ratio (no annealing):
#   "adversarial" -> 100% RFGSM examples (clean_ratio 0.0); the robustness case study.
#   "clean"       -> 100% clean examples  (clean_ratio 1.0); collapses under strong attack.
# Checkpoints and output CSVs are tagged by regime so the two never collide.
# Run once per regime (compute permitting), or set TRAIN_REGIME per session.
TRAIN_REGIME = "adversarial"   # or "clean"

# --- Attack-seed variance (CHEAP: no retraining). This is the primary job. ---
# Inference is deterministic, so clean metrics have zero eval-seed variance by
# construction. The variance we can measure cheaply comes from attack randomness
# (PGD random start + AutoAttack's Square component), so we re-run the attacks
# across several attack seeds on ONE checkpoint.
RUN_ATTACK_SEED_VARIANCE = True
ATTACK_SEEDS = (0, 1, 2)
INCLUDE_AUTOATTACK = True

# --- Training-seed variance (EXPENSIVE: needs retrained checkpoints). ---
# Only runs if you attach >= 2 checkpoints (retrain the case study under seeds
# 0/1/2 offline, then attach them). Leave off if you are compute constrained.
RUN_TRAIN_SEED_VARIANCE = True

# Map label -> checkpoint path. Attach checkpoints as a Kaggle dataset.
# If left empty, a single .pt is auto-discovered under /kaggle/input for the
# attack-seed job.
CHECKPOINTS = {
    # "seed0": "/kaggle/input/vt-checkpoints/vanilla_resnet18_adv_seed0.pt",
    # "seed1": "/kaggle/input/vt-checkpoints/vanilla_resnet18_adv_seed1.pt",
    # "seed2": "/kaggle/input/vt-checkpoints/vanilla_resnet18_adv_seed2.pt",
}

# Class-balanced attack subset (paper used 1000). Drop to 500 to save compute.
ATTACK_SUBSET_SAMPLES = 1000
ATTACK_SUBSET_BALANCED = True
ATTACK_SUBSET_SEED = 0

# ---- Backbone / data / attack definitions (__ARCH_LABEL__) ----
dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=__ARCH_BATCH_SIZE__,
    num_workers=2,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
)

# Must match the checkpoint's training config exactly (deterministic vanilla
# backbone), or load_state_dict will mismatch. Copied verbatim from the matching
# vanilla all-in-one notebook so this notebook's checkpoints reproduce the paper's.
__MODEL_CFG_BLOCK__

# Regime -> (clean_ratio, filename tag). Pure ratios, matching the project's
# vanilla builders: clean=1.0 (RFGSM branch never taken), adversarial=0.0
# (every example adversarial). No annealing.
# NOTE: do not name this `_clean_ratio` - training.py defines a function by that
# name and the notebook shares one global namespace (shadowing it breaks training).
if TRAIN_REGIME == "clean":
    REGIME_TAG, REGIME_CLEAN_RATIO = "clean", 1.0
elif TRAIN_REGIME == "adversarial":
    REGIME_TAG, REGIME_CLEAN_RATIO = "adv", 0.0
else:
    raise ValueError(f"TRAIN_REGIME must be 'clean' or 'adversarial', got {TRAIN_REGIME!r}")

training_cfg = TrainingConfig(
    epochs=100,
    base_lr=0.1,
    sigma_lr_scale=0.0,
    weight_decay=1e-4,
    use_parseval=False,
    parseval_beta=0.0,
    entropy_lambda=0.0,
    clean_ratio_start=REGIME_CLEAN_RATIO,
    clean_ratio_end=REGIME_CLEAN_RATIO,
    seed=0,
    device=DEVICE,
    output_dir=str(EXPERIMENT_ROOT / "checkpoints"),
    checkpoint_name=f"{model_cfg.architecture}_{REGIME_TAG}_full_model.pt",
    mixed_time_steps=(model_cfg.max_time_steps,),  # paper methodology: train at fixed T=8 ONLY (NOT mixed 1..8)
    selection_metric="clean",
    eval_interval=5,
    epsilon=8 / 255,
    alpha=8 / 255,
)

eval_cfg = EvaluationConfig(
    vt_thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    fixed_time_steps=tuple(range(1, model_cfg.max_time_steps + 1)),
    compute_metrics=True,
    exact_vt_compute=True,
    match_metric="avg_synops",
    max_batches=None,
)

# Weak PGD used only for robust checkpoint selection during training.
selection_attack_cfg = AttackConfig(epsilon=8 / 255, alpha=2 / 255, steps=4, eot_iters=1)

# PGD-20 (eot=1 is correct: these backbones are deterministic at inference).
pgd_cfg = AttackConfig(
    attack_name="pgd",
    objective="untargeted",
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=20,
    eot_iters=1,
    random_start=True,
    exit_aware=True,
    force_full_trace=True,
)

# Standard AutoAttack (matches the vanilla paper config).
autoattack_cfg = AttackConfig(
    attack_name="autoattack",
    epsilon=8 / 255,
    autoattack_version="standard",
    exit_aware=True,
    force_full_trace=True,
    eot_iters=1,
)

vt_rule = build_stop_rule("max_softmax")

print(model_cfg.architecture, "| ckpt:", training_cfg.checkpoint_name)
print("Attack seeds:", ATTACK_SEEDS, "| AutoAttack:", INCLUDE_AUTOATTACK and HAS_AUTOATTACK)
print("Attack subset:", ATTACK_SUBSET_SAMPLES, "balanced" if ATTACK_SUBSET_BALANCED else "random")
'''


DRIVER_HELPERS = r'''import glob


def discover_checkpoint() -> str:
    hits = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
    if not hits:
        raise FileNotFoundError(
            "No .pt checkpoint found under /kaggle/input. Attach your checkpoint dataset."
        )
    print(f"Auto-discovered checkpoint: {hits[0]}")
    return hits[0]


def load_checkpoint_model(ckpt_path: str):
    seed_everything(0)
    model = build_model(model_cfg, seed=0).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    return model


def calibrate_operating_points(model, splits):
    """Deterministic: selects the VT threshold + its SynOps-matched fixed-T control."""
    calib = select_compute_matched_threshold(
        model,
        splits.calib_loader,
        thresholds=eval_cfg.vt_thresholds,
        fixed_time_steps=eval_cfg.fixed_time_steps,
        stop_rule=vt_rule,
        time_steps=model_cfg.max_time_steps,
        compute_metrics=True,
        exact_vt_compute=eval_cfg.exact_vt_compute,
        match_metric=eval_cfg.match_metric,
        device=DEVICE,
        max_batches=eval_cfg.max_batches,
        vt_dataloader=splits.calib_loader,
        fixed_dataloader=splits.calib_loader,
    )
    threshold = float(calib["selected_threshold"])
    matched_t = int(calib["selected_match"]["matched_fixed_t"])
    print(f"  selected VT threshold={threshold} | SynOps-matched fixed-T={matched_t}")
    return threshold, matched_t


def build_operating_points(threshold: float, matched_t: int):
    return [
        {"name": "fixed_T8", "mode": "fixed", "time_steps": model_cfg.max_time_steps,
         "threshold": None, "attack_time_steps": model_cfg.max_time_steps},
        {"name": "fixed_matched", "mode": "fixed", "time_steps": matched_t,
         "threshold": None, "attack_time_steps": matched_t},
        {"name": "vt_selected", "mode": "vt", "time_steps": model_cfg.max_time_steps,
         "threshold": threshold, "attack_time_steps": model_cfg.max_time_steps},
    ]


def make_attack_loader(splits):
    return build_subset_loader(
        splits.test_loader,
        ATTACK_SUBSET_SAMPLES,
        ATTACK_SUBSET_SEED,
        balanced=ATTACK_SUBSET_BALANCED,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
    ) if ATTACK_SUBSET_SAMPLES is not None else splits.test_loader


def eval_operating_point(model, loader, op, attack_config):
    kwargs = dict(
        model=model,
        dataloader=loader,
        mode=op["mode"],
        time_steps=op["time_steps"],
        device=DEVICE,
        max_batches=eval_cfg.max_batches,
    )
    if op["mode"] == "vt":
        kwargs.update(stop_rule=vt_rule, threshold=op["threshold"])
    if attack_config is None:
        # clean pass: capture the compute context (synops / stopping depth)
        kwargs.update(compute_metrics=True, exact_vt_compute=eval_cfg.exact_vt_compute)
    else:
        # attack rows: accuracy is what varies; skip exact compute for speed
        kwargs.update(
            attack_config=replace(attack_config, attack_time_steps=op["attack_time_steps"]),
            compute_metrics=False,
            exact_vt_compute=False,
        )
    return evaluate_policy(**kwargs)


def run_operating_points_for_checkpoint(label, ckpt_path, attack_seeds):
    """Clean pass (deterministic) + attack passes across the given attack seeds."""
    print(f"[{label}] loading {ckpt_path}")
    splits = build_data_splits(dataset_cfg)
    model = load_checkpoint_model(ckpt_path)
    threshold, matched_t = calibrate_operating_points(model, splits)
    ops = build_operating_points(threshold, matched_t)
    attack_loader = make_attack_loader(splits)

    rows = []
    for op in ops:
        clean = eval_operating_point(model, attack_loader, op, None)
        rows.append({
            "checkpoint": label, "attack": "clean", "attack_seed": -1,
            "operating_point": op["name"], "accuracy": clean["accuracy"],
            "avg_synops": clean.get("avg_synops"),
            "avg_stopping_depth": clean.get("avg_stopping_depth"),
            "selected_threshold": threshold, "matched_fixed_t": matched_t,
        })

    attacks = [("pgd", pgd_cfg)]
    if INCLUDE_AUTOATTACK and HAS_AUTOATTACK:
        attacks.append(("autoattack", autoattack_cfg))

    for seed in attack_seeds:
        for attack_name, attack_cfg in attacks:
            seed_everything(seed)
            seeded_cfg = replace(attack_cfg, autoattack_seed=seed)
            for op in ops:
                result = eval_operating_point(model, attack_loader, op, seeded_cfg)
                rows.append({
                    "checkpoint": label, "attack": attack_name, "attack_seed": seed,
                    "operating_point": op["name"], "accuracy": result["accuracy"],
                    "avg_synops": None,
                    "avg_stopping_depth": result.get("avg_stopping_depth"),
                    "selected_threshold": threshold, "matched_fixed_t": matched_t,
                })
            print(f"  [{label}] seed={seed} {attack_name}: "
                  + ", ".join(f"{o['name']}={r:.2f}" for o, r in zip(
                      ops, [row['accuracy'] for row in rows[-len(ops):]])))
    return rows


def summarize(raw_df, across):
    """Aggregate accuracy mean +/- std, grouping the attack rows by `across`."""
    attack_rows = raw_df[raw_df["attack"] != "clean"]
    summary = (
        attack_rows.groupby(["operating_point", "attack"])["accuracy"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["report"] = summary.apply(
        lambda r: f"{r['mean']:.2f} +/- {r['std']:.2f} (n={int(r['count'])})", axis=1)
    summary.insert(0, "variance_source", across)

    clean = raw_df[raw_df["attack"] == "clean"][
        ["operating_point", "accuracy", "avg_synops", "avg_stopping_depth"]
    ].copy()
    clean = clean.rename(columns={"accuracy": "clean_accuracy"})
    return summary, clean
'''


DRIVER_TRAINING = r'''import gc

if RUN_TRAINING:
    print(f"=== Job 0: training regime='{TRAIN_REGIME}' seed(s) {TRAIN_SEEDS} "
          f"(epochs={training_cfg.epochs} each; lower training_cfg.epochs for a quick test) ===")
    for seed in TRAIN_SEEDS:
        label = f"{REGIME_TAG}_seed{seed}"
        if label in CHECKPOINTS and Path(CHECKPOINTS[label]).exists():
            print(f"{label}: using attached checkpoint {CHECKPOINTS[label]}")
            continue
        ckpt_name = f"{model_cfg.architecture}_{REGIME_TAG}_seed{seed}.pt"
        ckpt_path = Path(training_cfg.output_dir) / ckpt_name
        if ckpt_path.exists():
            CHECKPOINTS[label] = str(ckpt_path)
            print(f"{label}: found existing checkpoint {ckpt_path}")
            continue

        # Vary the train/val/calib partition by seed. The CIFAR-10 *test* set is
        # fixed, so training-seed variance is measured against identical test data.
        seed_dataset_cfg = replace(dataset_cfg, seed=seed)
        seed_splits = build_data_splits(seed_dataset_cfg)
        seed_train_cfg = replace(training_cfg, seed=seed, checkpoint_name=ckpt_name)
        seed_everything(seed)
        seed_model = build_model(model_cfg, seed=seed).to(seed_train_cfg.device)
        print(f"{label}: training -> {ckpt_path}")
        history, saved_path = fit_model(
            seed_model,
            seed_splits.train_loader,
            seed_splits.val_loader,
            seed_train_cfg,
            selection_attack_config=selection_attack_cfg,
        )
        CHECKPOINTS[label] = str(saved_path)
        pd.DataFrame(history).to_csv(EXPERIMENT_ROOT / f"train_history_{label}.csv", index=False)
        del seed_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{label}: done -> {saved_path}")
    print("Checkpoints available:", dict(CHECKPOINTS))
else:
    print("RUN_TRAINING = False; expecting checkpoints in CHECKPOINTS or under /kaggle/input.")
'''


DRIVER_ATTACK_SEED = r'''if RUN_ATTACK_SEED_VARIANCE:
    if CHECKPOINTS:
        primary_label, primary_path = next(iter(CHECKPOINTS.items()))
    else:
        primary_label, primary_path = "case_study", discover_checkpoint()

    print(f"=== Attack-seed variance on '{primary_label}' (no retraining) ===")
    attack_rows = run_operating_points_for_checkpoint(
        primary_label, primary_path, ATTACK_SEEDS
    )
    attack_raw_df = pd.DataFrame(attack_rows)
    attack_summary, attack_clean = summarize(attack_raw_df, across="attack_seed")

    raw_path = EXPERIMENT_ROOT / f"multiseed_attack_seed_raw_{REGIME_TAG}.csv"
    summary_path = EXPERIMENT_ROOT / f"multiseed_attack_seed_summary_{REGIME_TAG}.csv"
    attack_raw_df.to_csv(raw_path, index=False)
    attack_summary.to_csv(summary_path, index=False)

    print(f"\n[regime={TRAIN_REGIME}] Clean operating points (deterministic; std = 0 by construction):")
    display(attack_clean)
    print("\nAttacked accuracy across attack seeds (mean +/- std):")
    display(attack_summary[["operating_point", "attack", "report"]])
    print(f"\nSaved -> {summary_path}")
else:
    print("RUN_ATTACK_SEED_VARIANCE = False; skipping the attack-seed job.")
'''


DRIVER_TRAIN_SEED = r'''if RUN_TRAIN_SEED_VARIANCE and len(CHECKPOINTS) >= 2:
    print(f"=== Training-seed variance across {len(CHECKPOINTS)} checkpoints ===")
    # One attack seed per checkpoint isolates the training-seed effect on the
    # clean frontier and on attacked accuracy.
    train_rows = []
    for label, path in CHECKPOINTS.items():
        train_rows.extend(
            run_operating_points_for_checkpoint(label, path, attack_seeds=(ATTACK_SEEDS[0],))
        )
    train_raw_df = pd.DataFrame(train_rows)

    # Clean-frontier variance across training seeds (the expensive claim).
    clean_across = (
        train_raw_df[train_raw_df["attack"] == "clean"]
        .groupby("operating_point")["accuracy"].agg(["mean", "std", "count"]).reset_index()
    )
    clean_across["std"] = clean_across["std"].fillna(0.0)
    clean_across["report"] = clean_across.apply(
        lambda r: f"{r['mean']:.2f} +/- {r['std']:.2f} (n={int(r['count'])})", axis=1)

    train_summary, _ = summarize(train_raw_df, across="train_seed")

    train_raw_df.to_csv(EXPERIMENT_ROOT / f"multiseed_train_seed_raw_{REGIME_TAG}.csv", index=False)
    train_summary.to_csv(EXPERIMENT_ROOT / f"multiseed_train_seed_summary_{REGIME_TAG}.csv", index=False)
    clean_across.to_csv(EXPERIMENT_ROOT / f"multiseed_train_seed_clean_{REGIME_TAG}.csv", index=False)

    print("\nClean accuracy across training seeds (mean +/- std):")
    display(clean_across[["operating_point", "report"]])
    print("\nAttacked accuracy across training seeds (mean +/- std):")
    display(train_summary[["operating_point", "attack", "report"]])
elif RUN_TRAIN_SEED_VARIANCE:
    print("RUN_TRAIN_SEED_VARIANCE = True but < 2 checkpoints attached; "
          "retrain the case study under 3 seeds and add them to CHECKPOINTS.")
else:
    print("RUN_TRAIN_SEED_VARIANCE = False; skipping the (expensive) retrain-based job.")
'''


def _config_cell(spec: dict) -> str:
    return (
        DRIVER_CONFIG
        .replace("__MODEL_CFG_BLOCK__", spec["model_cfg"])
        .replace("__ARCH_BATCH_SIZE__", str(spec["batch_size"]))
        .replace("__ARCH_LABEL__", spec["label"])
    )


def _driver_cells(spec: dict) -> list[dict]:
    return [
        _markdown_cell(
            f"# VT Research Suite: Multi-Seed Variance Driver ({spec['label']})\n\n"
            "Self-contained, **internet-off** Kaggle notebook that adds variance bars to the "
            "paper's headline operating points: the full-horizon fixed baseline (T=8), the "
            "calibration-selected VT point, and its SynOps-matched fixed-T control.\n\n"
            "Two jobs:\n"
            "1. **Attack-seed variance (cheap, default):** re-runs PGD-20 and AutoAttack across "
            "several attack seeds on one checkpoint. No retraining. Hardens the "
            "PGD-vs-AutoAttack robustness ranking.\n"
            "2. **Training-seed variance (expensive, opt-in):** if you attach >=2 retrained "
            "checkpoints, reports clean-frontier mean +/- std across training seeds.\n\n"
            "Because inference is deterministic, clean metrics have no eval-seed variance; only "
            "attacked accuracy varies without retraining."
        ),
        _code_cell(DRIVER_IMPORTS),
        _markdown_cell("## Configuration"),
        _code_cell(_config_cell(spec)),
        _markdown_cell("## Helpers"),
        _code_cell(DRIVER_HELPERS),
        _markdown_cell(
            "## Job 0 - Training\n\n"
            "Trains one checkpoint per seed in `TRAIN_SEEDS` (missing seeds only, so it is "
            "resumable across sessions). `TRAIN_SEEDS=(0,)` is enough for Job 1; use `(0,1,2)` "
            "to also enable Job 2. Skip this cell if you attach pre-trained checkpoints."
        ),
        _code_cell(DRIVER_TRAINING),
        _markdown_cell("## Job 1 - Attack-seed variance (no retraining)"),
        _code_cell(DRIVER_ATTACK_SEED),
        _markdown_cell("## Job 2 - Training-seed variance (opt-in, needs retrained checkpoints)"),
        _code_cell(DRIVER_TRAIN_SEED),
    ]


def build_notebook(spec: dict) -> dict:
    cells = [
        _markdown_cell(
            f"# VT Research Suite: Multi-Seed - {spec['label']} (Kaggle, offline)\n\n"
            f"Backbone: `{spec['arch']}`. Packaged for Kaggle with **internet off**. Attach the "
            "`offline-src` dataset (CIFAR-10 + AutoAttack) and your checkpoint dataset; the setup "
            "cell stages them so the inlined modules run without network access."
        ),
        _markdown_cell("## Offline setup (CIFAR-10 staging + AutoAttack on sys.path)"),
        _code_cell(OFFLINE_SETUP),
    ]
    for module_name in MODULE_ORDER:
        cells.append(_markdown_cell(f"## Module: {module_name}"))
        cells.append(_code_cell(_read_module(module_name)))
    cells.extend(_driver_cells(spec))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for short_name, spec in ARCH_SPECS.items():
        notebook = build_notebook(spec)
        out_path = ROOT / f"vt_research_multiseed_{short_name}_kaggle.ipynb"
        out_path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
