from __future__ import annotations

import json
import textwrap
from pathlib import Path

from build_split_kaggle_notebooks import (
    _code_cell,
    _driver_imports_cell,
    _load_source_cells,
    _markdown_cell,
    _notebook_payload,
    _read_module,
)
from build_vanilla_vgg16_kaggle_notebook import _optional_subset_attack_cells


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "vt_vanilla_mobilenetv2_kaggle_all_in_one.ipynb"


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
            "# VT Research Suite: Vanilla MobileNetV2 Kaggle All-In-One\n\n"
            "This notebook is a separate deterministic VT pipeline built around a MobileNetV2-style SNN backbone.\n\n"
            "Key choices:\n"
            "- canonical MobileNetV2-1.0 inverted-residual stage schedule\n"
            "- CIFAR-appropriate stride-1 input stem\n"
            "- deterministic LIF neurons in place of MobileNetV2 nonlinearities\n"
            "- linear bottleneck projection layers are left non-spiking\n"
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
import os

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_vanilla_mobilenetv2_outputs") if Path("/kaggle/working").exists() else Path("vt_vanilla_mobilenetv2_outputs")
DEFAULT_DATA_ROOT = os.environ.get(
    "VT_OFFLINE_CIFAR_ROOT",
    "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data",
)
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
USE_MOBILENETV2_CLEAN_RESTORE = False
MOBILENETV2_RESTORED_SELECTED_THRESHOLD = 0.95

RUN_FULL_TEST_THRESHOLD_SWEEP = True
RUN_MULTI_THRESHOLD_ATTACKS = True
RUN_AUTOATTACK_ON_ALL_SELECTED_THRESHOLDS = False
SELECTED_THRESHOLD_COUNT = 3

ATTACK_SUBSET_SAMPLES = 2000
ATTACK_SUBSET_BALANCED = True
ATTACK_SUBSET_SEED = 0

# RTX PRO 6000 Blackwell-friendly defaults. The card has a large memory pool,
# but the stateful SNN trace still runs as a single-device workflow.
CALIBRATION_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 2048
ATTACK_BATCH_SIZE = 256
EXACT_VT_BATCH_SIZE = 1024

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=512,
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
    clean_ratio_start=0.0,
    clean_ratio_end=0.0,
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
    checkpoint_name="vanilla_mobilenetv2_adv_full_model.pt",
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
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Active CUDA device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if torch.cuda.device_count() > 1:
        print("Note: this notebook uses one GPU by design; stateful SNN membrane traces are not DataParallel-wrapped.")
        print("If one RTX PRO 6000 Blackwell is visible, that is expected and enough for the large-batch profile.")
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


def _restore_clean_eval_cell() -> dict:
    return _code_cell(
        r'''import textwrap
from io import StringIO


def _read_restored_csv(source):
    return pd.read_csv(StringIO(textwrap.dedent(source).strip()))


def _select_restored_threshold_panel(vt_results, selected_threshold, count=3):
    thresholds = sorted(float(value) for value in vt_results["threshold"].dropna().unique())
    anchor_idx = min(range(len(thresholds)), key=lambda idx: abs(thresholds[idx] - float(selected_threshold)))
    chosen = [anchor_idx]
    step = 1
    while len(chosen) < min(count, len(thresholds)):
        left = anchor_idx - step
        right = anchor_idx + step
        if left >= 0:
            chosen.append(left)
            if len(chosen) >= count:
                break
        if right < len(thresholds):
            chosen.append(right)
            if len(chosen) >= count:
                break
        step += 1

    rows = []
    for rank, idx in enumerate(sorted(chosen, key=lambda value: thresholds[value])):
        threshold = thresholds[idx]
        if abs(threshold - float(selected_threshold)) < 1e-12:
            role = "selected"
        elif threshold < float(selected_threshold):
            role = "lower_neighbor"
        else:
            role = "upper_neighbor"
        rows.append(
            {
                "threshold": threshold,
                "threshold_role": role,
                "distance_from_selected": abs(threshold - float(selected_threshold)),
                "selection_rank": rank,
            }
        )
    return pd.DataFrame(rows)


def restore_mobilenetv2_clean_eval_state(selected_threshold_override=None):
    """Restore completed MobileNetV2 clean-eval CSVs from the partial run.

    This lets us retrain/load the checkpoint, skip expensive clean sweeps, and
    jump directly into attack/fairness cells. The raw pipeline selected theta=0.50
    by best matched delta, but that is a degenerate low-accuracy operating point.
    By default we restore the practical high-accuracy operating point theta=0.95.
    """
    global vt_rule, main_calibration, calib_vt, calib_fixed, calib_matches
    global selected_threshold, selected_synops, selected_threshold_panel, selected_threshold_values
    global test_loaders, test_fixed_sweep, test_fixed_full, test_fixed_matched
    global fixed_baseline_df, test_vt_sweep, test_vt_matches, selected_threshold_test_panel
    global test_vt, test_results, rule_calibration, rule_test_df

    vt_rule = build_stop_rule("max_softmax")
    selected_threshold_value = (
        float(MOBILENETV2_RESTORED_SELECTED_THRESHOLD)
        if selected_threshold_override is None
        else float(selected_threshold_override)
    )

    calib_fixed = _read_restored_csv(
        """
        time_steps,accuracy,avg_executed_steps,avg_synops
        1,10.64,1.0,45329390.0
        2,17.78,2.0,92279090.0
        3,38.54,3.0,139436500.0
        4,61.04,4.0,187738300.0
        5,64.92,5.0,234005700.0
        6,74.34,6.0,282619700.0
        7,74.84,7.0,328731300.0
        8,77.58,8.0,377030200.0
        """
    )
    calib_fixed["mode"] = "fixed"
    calib_fixed["stop_rule"] = "fixed"
    calib_fixed["threshold"] = None
    calib_fixed["avg_stopping_depth"] = calib_fixed["avg_executed_steps"]
    calib_fixed["compute_mode"] = "batched_proxy"

    calib_vt = _read_restored_csv(
        """
        threshold,accuracy,avg_stopping_depth,avg_executed_steps,avg_synops,compute_mode
        0.50,28.82,2.3732,2.3732,110003300.0,sample_exact
        0.60,38.82,2.9076,2.9076,135394500.0,sample_exact
        0.70,49.76,3.6276,3.6276,169592200.0,sample_exact
        0.80,60.68,4.5598,4.5598,213852200.0,sample_exact
        0.85,65.72,5.0906,5.0906,239041300.0,sample_exact
        0.90,70.60,5.7342,5.7342,269577500.0,sample_exact
        0.95,75.04,6.5082,6.5082,306290600.0,sample_exact
        0.99,77.52,7.4050,7.4050,348820100.0,sample_exact
        """
    )
    calib_vt["mode"] = "vt"
    calib_vt["time_steps"] = model_cfg.max_time_steps
    calib_vt["stop_rule"] = "max_softmax"

    calib_matches = _read_restored_csv(
        """
        match_metric,stop_rule,threshold,vt_accuracy,vt_avg_stopping_depth,vt_avg_executed_steps,matched_fixed_t,fixed_accuracy,fixed_avg_executed_steps,metric_gap,accuracy_delta,vt_avg_synops,fixed_avg_synops,synops_delta
        avg_synops,max_softmax,0.50,28.82,2.3732,2.3732,2,17.78,2.0,17724210.0,11.04,110003300.0,92279090.0,17724210.0
        avg_synops,max_softmax,0.60,38.82,2.9076,2.9076,3,38.54,3.0,4041963.0,0.28,135394500.0,139436500.0,-4041963.0
        avg_synops,max_softmax,0.70,49.76,3.6276,3.6276,4,61.04,4.0,18146040.0,-11.28,169592200.0,187738300.0,-18146040.0
        avg_synops,max_softmax,0.80,60.68,4.5598,4.5598,5,64.92,5.0,20153480.0,-4.24,213852200.0,234005700.0,-20153480.0
        avg_synops,max_softmax,0.85,65.72,5.0906,5.0906,5,64.92,5.0,5035605.0,0.80,239041300.0,234005700.0,5035605.0
        avg_synops,max_softmax,0.90,70.60,5.7342,5.7342,6,74.34,6.0,13042220.0,-3.74,269577500.0,282619700.0,-13042220.0
        avg_synops,max_softmax,0.95,75.04,6.5082,6.5082,7,74.84,7.0,22440760.0,0.20,306290600.0,328731300.0,-22440760.0
        avg_synops,max_softmax,0.99,77.52,7.4050,7.4050,7,74.84,7.0,20088790.0,2.68,348820100.0,328731300.0,20088790.0
        """
    )

    selected_match = calib_matches.loc[calib_matches["threshold"].eq(selected_threshold_value)].iloc[0].to_dict()
    selected_threshold = selected_threshold_value
    selected_synops = float(selected_match["vt_avg_synops"])
    selected_threshold_panel = _select_restored_threshold_panel(
        calib_vt,
        selected_threshold,
        count=SELECTED_THRESHOLD_COUNT,
    )
    selected_threshold_values = selected_threshold_panel["threshold"].tolist()

    main_calibration = {
        "selected_threshold": selected_threshold,
        "selected_metric_value": selected_synops,
        "selected_match": selected_match,
        "vt_results": calib_vt,
        "fixed_results": calib_fixed,
        "matches": calib_matches,
    }

    test_fixed_sweep = _read_restored_csv(
        """
        time_steps,accuracy,avg_executed_steps,avg_synops
        1,10.00,1.0,45330410.0
        2,17.60,2.0,92281990.0
        3,37.72,3.0,139441700.0
        4,61.06,4.0,187746300.0
        5,64.91,5.0,234018200.0
        6,73.92,6.0,282631400.0
        7,74.71,7.0,328746900.0
        8,77.66,8.0,377047400.0
        """
    )
    test_fixed_sweep["mode"] = "fixed"
    test_fixed_sweep["stop_rule"] = "fixed"
    test_fixed_sweep["threshold"] = None
    test_fixed_sweep["avg_stopping_depth"] = test_fixed_sweep["avg_executed_steps"]
    test_fixed_sweep["compute_mode"] = "batched_proxy"

    test_vt_sweep = _read_restored_csv(
        """
        threshold,accuracy,avg_stopping_depth,avg_executed_steps,avg_synops,compute_mode
        0.50,28.96,2.3807,2.3807,110367100.0,sample_exact
        0.60,39.32,2.9548,2.9548,137646300.0,sample_exact
        0.70,49.96,3.6710,3.6710,171671900.0,sample_exact
        0.80,60.53,4.6057,4.6057,216047300.0,sample_exact
        0.85,66.50,5.1709,5.1709,242871900.0,sample_exact
        0.90,71.33,5.8001,5.8001,272723000.0,sample_exact
        0.95,75.68,6.5558,6.5558,308567400.0,sample_exact
        0.99,78.08,7.4130,7.4130,349209500.0,sample_exact
        """
    )
    test_vt_sweep["mode"] = "vt"
    test_vt_sweep["time_steps"] = model_cfg.max_time_steps
    test_vt_sweep["stop_rule"] = "max_softmax"

    test_vt_matches = _read_restored_csv(
        """
        match_metric,stop_rule,threshold,vt_accuracy,vt_avg_stopping_depth,vt_avg_executed_steps,matched_fixed_t,fixed_accuracy,fixed_avg_executed_steps,metric_gap,accuracy_delta,vt_avg_synops,fixed_avg_synops,synops_delta
        avg_synops,max_softmax,0.50,28.96,2.3807,2.3807,2,17.60,2.0,18085080.0,11.36,110367100.0,92281990.0,18085080.0
        avg_synops,max_softmax,0.60,39.32,2.9548,2.9548,3,37.72,3.0,1795326.0,1.60,137646300.0,139441700.0,-1795326.0
        avg_synops,max_softmax,0.70,49.96,3.6710,3.6710,4,61.06,4.0,16074380.0,-11.10,171671900.0,187746300.0,-16074380.0
        avg_synops,max_softmax,0.80,60.53,4.6057,4.6057,5,64.91,5.0,17970820.0,-4.38,216047300.0,234018200.0,-17970820.0
        avg_synops,max_softmax,0.85,66.50,5.1709,5.1709,5,64.91,5.0,8853781.0,1.59,242871900.0,234018200.0,8853781.0
        avg_synops,max_softmax,0.90,71.33,5.8001,5.8001,6,73.92,6.0,9908459.0,-2.59,272723000.0,282631400.0,-9908459.0
        avg_synops,max_softmax,0.95,75.68,6.5558,6.5558,7,74.71,7.0,20179520.0,0.97,308567400.0,328746900.0,-20179520.0
        avg_synops,max_softmax,0.99,78.08,7.4130,7.4130,7,74.71,7.0,20462580.0,3.37,349209500.0,328746900.0,20462580.0
        """
    )

    selected_fixed_t = int(selected_match["matched_fixed_t"])
    test_fixed_full = test_fixed_sweep.loc[test_fixed_sweep["time_steps"].eq(model_cfg.max_time_steps)].iloc[0].to_dict()
    test_fixed_full["report_mode"] = "fixed_full_horizon"
    test_fixed_matched = test_fixed_sweep.loc[test_fixed_sweep["time_steps"].eq(selected_fixed_t)].iloc[0].to_dict()
    test_fixed_matched["report_mode"] = "fixed_compute_matched"
    fixed_baseline_df = pd.DataFrame([test_fixed_full, test_fixed_matched])

    selected_threshold_test_panel = selected_threshold_panel.merge(test_vt_sweep, on="threshold", how="left")
    test_vt = selected_threshold_test_panel.loc[selected_threshold_test_panel["threshold_role"].eq("selected")].iloc[0].to_dict()
    test_vt["report_mode"] = "vt_selected_threshold"
    test_results = pd.DataFrame([test_fixed_full, test_fixed_matched, test_vt])

    rule_calibration = _read_restored_csv(
        """
        stop_rule,threshold,accuracy,avg_synops,metric_gap
        max_softmax,0.5,28.82,110003300.0,0.0
        margin,0.5,48.42,165287900.0,55284590.0
        entropy,0.5,29.62,111584300.0,1581000.0
        temperature_softmax,0.5,56.92,199252300.0,89249000.0
        evidential_surrogate,0.5,18.48,92565030.0,17438270.0
        """
    )
    fixed_rule_anchor = test_fixed_full.copy()
    fixed_rule_anchor["stop_rule"] = "fixed_T8_baseline"
    fixed_rule_anchor["threshold"] = None
    rule_test_df = pd.DataFrame(
        [
            fixed_rule_anchor,
            {
                "mode": "vt",
                "time_steps": model_cfg.max_time_steps,
                "stop_rule": "max_softmax",
                "threshold": 0.5,
                "accuracy": 28.96,
                "avg_stopping_depth": 2.3807,
                "avg_executed_steps": 2.3807,
                "avg_synops": 110367100.0,
            },
            {
                "mode": "vt",
                "time_steps": model_cfg.max_time_steps,
                "stop_rule": "margin",
                "threshold": 0.5,
                "accuracy": 48.82,
                "avg_stopping_depth": 3.5832,
                "avg_executed_steps": 3.5832,
                "avg_synops": 167491300.0,
            },
            {
                "mode": "vt",
                "time_steps": model_cfg.max_time_steps,
                "stop_rule": "entropy",
                "threshold": 0.5,
                "accuracy": 29.92,
                "avg_stopping_depth": 2.4451,
                "avg_executed_steps": 2.4451,
                "avg_synops": 113452000.0,
            },
            {
                "mode": "vt",
                "time_steps": model_cfg.max_time_steps,
                "stop_rule": "temperature_softmax",
                "threshold": 0.5,
                "accuracy": 57.05,
                "avg_stopping_depth": 4.2912,
                "avg_executed_steps": 4.2912,
                "avg_synops": 201143600.0,
            },
            {
                "mode": "vt",
                "time_steps": model_cfg.max_time_steps,
                "stop_rule": "evidential_surrogate",
                "threshold": 0.5,
                "accuracy": 18.10,
                "avg_stopping_depth": 2.0050,
                "avg_executed_steps": 2.0050,
                "avg_synops": 92520350.0,
            },
        ]
    )

    if PERSIST_OUTPUTS:
        persist_frame(selected_threshold_panel, "main_selected_thresholds")
        persist_frame(fixed_baseline_df, "test_fixed_baseline_results")
        persist_frame(test_vt_sweep, "test_vt_threshold_sweep")
        persist_frame(test_vt_matches, "test_vt_fixed_matches")
        persist_frame(selected_threshold_test_panel, "test_selected_threshold_results")
        persist_frame(test_results, "main_test_results")
        persist_frame(rule_calibration, "stopping_rule_calibration")
        persist_frame(rule_test_df, "stopping_rule_test_results")

    print("Restored MobileNetV2 clean-eval state from partial-run CSVs.")
    print(f"Restored selected threshold for attacks: theta={selected_threshold:.2f}")
    print(f"Matched fixed-T for restored threshold: T={selected_fixed_t}")
    print("Note: restored stopping-rule CSVs remain the original theta=0.50-target summaries from the partial run.")
    display(selected_threshold_panel)
    display(test_results[["report_mode", "mode", "time_steps", "threshold", "accuracy", "avg_stopping_depth", "avg_executed_steps", "avg_synops"]])
    return main_calibration
'''
    )


def _wrap_restore_skip(source: str, restored_message: str, display_source: str = "") -> str:
    return (
        "if USE_MOBILENETV2_CLEAN_RESTORE:\n"
        "    restore_mobilenetv2_clean_eval_state()\n"
        f"    print({restored_message!r})\n"
        + textwrap.indent(display_source.strip() + "\n" if display_source.strip() else "", "    ")
        + "else:\n"
        + textwrap.indent(source, "    ")
    )


def _fast_attack_cell() -> dict:
    return _code_cell(
        """attack_rows = []
require_autoattack_for_final_runs()
test_loaders = get_test_loaders(splits)

attack_threshold_panel = (
    selected_threshold_panel
    if RUN_MULTI_THRESHOLD_ATTACKS
    else selected_threshold_panel[selected_threshold_panel["threshold_role"] == "selected"]
)
attack_objectives = ["untargeted", "wrong_early_exit", "compute_overuse"]

# Fast deterministic attacks for the restored MobileNetV2 continuation run.
fixed_attack_cfg = replace(
    adaptive_attack_cfg,
    objective="untargeted",
    exit_aware=False,
    eot_iters=1,
)

print("Running fixed-T baseline attacks first...")
fixed_attack = evaluate_policy(
    model,
    test_loaders["attack"],
    mode="fixed",
    time_steps=model_cfg.max_time_steps,
    attack_config=fixed_attack_cfg,
    compute_metrics=False,
    exact_vt_compute=False,
    device=training_cfg.device,
    max_batches=eval_cfg.max_batches,
)
fixed_attack["threshold_role"] = "fixed_T8_baseline"
attack_rows.append(fixed_attack)

print("Running VT attacks against the restored threshold panel...")
for objective in progress_iter(attack_objectives, desc="Attack objectives", total=len(attack_objectives), leave=False):
    current_attack = replace(
        adaptive_attack_cfg if objective != "compute_overuse" else compute_attack_cfg,
        objective=objective,
        eot_iters=1,
        exit_aware=True,
    )
    for _, threshold_row in attack_threshold_panel.iterrows():
        result = evaluate_policy(
            model,
            test_loaders["attack"],
            mode="vt",
            time_steps=model_cfg.max_time_steps,
            stop_rule=vt_rule,
            threshold=float(threshold_row["threshold"]),
            attack_config=current_attack,
            compute_metrics=False,
            exact_vt_compute=False,
            device=training_cfg.device,
            max_batches=eval_cfg.max_batches,
        )
        result["threshold_role"] = threshold_row["threshold_role"]
        attack_rows.append(result)

try:
    autoattack_fixed_cfg = replace(
        autoattack_cfg,
        eot_iters=1,
        exit_aware=False,
        autoattack_version="standard",
    )
    autoattack_fixed = evaluate_policy(
        model,
        test_loaders["attack"],
        mode="fixed",
        time_steps=model_cfg.max_time_steps,
        attack_config=autoattack_fixed_cfg,
        compute_metrics=False,
        exact_vt_compute=False,
        device=training_cfg.device,
        max_batches=eval_cfg.max_batches,
    )
    autoattack_fixed["threshold_role"] = "fixed_T8_baseline"
    attack_rows.append(autoattack_fixed)

    vt_autoattack_panel = (
        attack_threshold_panel
        if RUN_AUTOATTACK_ON_ALL_SELECTED_THRESHOLDS
        else attack_threshold_panel[attack_threshold_panel["threshold_role"] == "selected"]
    )
    autoattack_vt_cfg = replace(
        autoattack_cfg,
        eot_iters=1,
        exit_aware=True,
        autoattack_version="standard",
    )
    for _, threshold_row in vt_autoattack_panel.iterrows():
        autoattack_vt = evaluate_policy(
            model,
            test_loaders["attack"],
            mode="vt",
            time_steps=model_cfg.max_time_steps,
            stop_rule=vt_rule,
            threshold=float(threshold_row["threshold"]),
            attack_config=autoattack_vt_cfg,
            compute_metrics=False,
            exact_vt_compute=False,
            device=training_cfg.device,
            max_batches=eval_cfg.max_batches,
        )
        autoattack_vt["threshold_role"] = threshold_row["threshold_role"]
        attack_rows.append(autoattack_vt)
except ImportError as exc:
    print(f"AutoAttack skipped: {exc}")

attack_df = pd.DataFrame(attack_rows)
persist_frame(attack_df, "attack_results")
display(
    attack_df[
        [
            "mode",
            "threshold_role",
            "threshold",
            "attack_name",
            "autoattack_version",
            "attack_objective",
            "accuracy",
            "avg_stopping_depth",
            "avg_executed_steps",
            "attack_eot",
            "exit_aware_attack",
        ]
    ]
)

print("Skipped temporal diagnostic plots in the fast MobileNetV2 continuation path.")
"""
    )


def _maybe_wrap_source_cell(cell: dict) -> dict:
    if cell.get("cell_type") != "code":
        return cell
    source = "".join(cell["source"])
    if "attack_rows = []" in source and "Running fixed-T baseline attacks first" in source:
        return _fast_attack_cell()
    if "main_calibration = calibrate_compute_matched_threshold" in source:
        wrapped = _wrap_restore_skip(
            source,
            "Skipped expensive calibration sweeps; using restored MobileNetV2 clean-eval state.",
        )
        return _code_cell(wrapped)
    if "print(\"Running fixed-T baseline suite on test first...\")" in source:
        wrapped = _wrap_restore_skip(
            source,
            "Skipped expensive fixed/VT test sweeps; using restored MobileNetV2 clean-eval state.",
            'display(test_results[["report_mode", "mode", "time_steps", "threshold", "accuracy", "avg_stopping_depth", "avg_executed_steps", "avg_synops"]])',
        )
        return _code_cell(wrapped)
    if "rule_calibration = compare_stopping_rules" in source:
        wrapped = _wrap_restore_skip(
            source,
            "Skipped stopping-rule rerun; using restored MobileNetV2 stopping-rule summaries.",
            'display(rule_calibration[["stop_rule", "threshold", "accuracy", "avg_synops", "metric_gap"]])',
        )
        return _code_cell(wrapped)
    return cell


def _adjust_optional_subset_cells_for_large_gpu(cells: list[dict]) -> list[dict]:
    adjusted = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            adjusted.append(cell)
            continue
        source = "".join(cell["source"]).replace("ATTACK_BATCH_SIZE = 100", "ATTACK_BATCH_SIZE = 256")
        adjusted.append(_code_cell(source))
    return adjusted


def build_notebook() -> dict:
    source_cells = _load_source_cells()
    cells = _base_notebook_cells()
    cells.extend(
        [
            _markdown_cell(
                "# Experiment Driver\n\n"
                "This driver reuses the full VT training and evaluation workflow, but the defaults are "
                "specialized for a deterministic vanilla MobileNetV2 SNN."
            ),
            _driver_imports_cell(),
            _config_cell(),
        ]
    )
    for source_index, cell in enumerate(source_cells[3:], start=3):
        cells.append(_maybe_wrap_source_cell(cell))
        if source_index == 4:
            cells.append(
                _markdown_cell(
                    "## MobileNetV2 Clean-Eval Restore\n\n"
                    "When `USE_MOBILENETV2_CLEAN_RESTORE=True`, the notebook skips the already completed clean "
                    "calibration/test sweeps and restores those tables from the partial run. This is intended for "
                    "retraining/loading the checkpoint and then running the remaining attack/fairness cells."
                )
            )
            cells.append(_restore_clean_eval_cell())
    cells.extend(_adjust_optional_subset_cells_for_large_gpu(_optional_subset_attack_cells()))
    return _notebook_payload(cells)


def main() -> None:
    OUTPUT_NOTEBOOK.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
