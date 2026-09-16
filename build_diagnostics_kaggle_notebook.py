"""Build an eval-only Kaggle notebook that produces per-timestep trajectories.

No training. Loads an existing checkpoint and runs collect_temporal_diagnostics
under clean / PGD / AutoAttack at two thresholds, then writes trajectories.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_multiseed_kaggle_notebook import (
    MODULE_ORDER, OFFLINE_SETUP, _code_cell, _markdown_cell, _read_module,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "vt_diagnostics_kaggle.ipynb"

DRIVER = '''import os, glob
from dataclasses import replace
from pathlib import Path
import pandas as pd
import torch
from IPython.display import display

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = Path("/kaggle/working/vt_diagnostics") if Path("/kaggle/working").exists() else Path("vt_diagnostics")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = os.environ.get("VT_OFFLINE_CIFAR_ROOT",
    "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data")

# ===================== EDIT =====================
N_IMAGES   = 1000          # class-balanced subset
BATCH      = 128
THRESHOLDS = (0.60, 0.50)  # 0.60 = this checkpoint's selected point; 0.50 = shallow-halting probe
RUN_AUTOATTACK = True
CKPT = None                # None -> auto-discover a .pt under /kaggle/input
# ================================================

model_cfg = ModelConfig(
    architecture="vanilla_resnet18", num_classes=10, in_channels=3,
    max_time_steps=8, input_encoding="direct", stem_channels=64,
    stage_channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2),
    stochastic=False, learnable_sigma=False, use_dynamic_sigma=False,
    use_shortcut_lif=True, v_threshold=0.5, lif_tau=2.0,
    detach_reset=True, sigma_init_space="raw")

dataset_cfg = DatasetConfig(name="cifar10", root=DATA_ROOT, batch_size=BATCH,
    num_workers=2, seed=0, augment=True, val_fraction=0.1, calibration_fraction=0.1)

if CKPT is None:
    hits = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
    if not hits:
        raise FileNotFoundError("Attach a checkpoint dataset containing a .pt file.")
    CKPT = hits[0]
print("checkpoint:", CKPT)

seed_everything(0)
model = build_model(model_cfg, seed=0).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()
tc = state.get("training_config", {})
if not isinstance(tc, dict):
    tc = vars(tc)
print("ckpt config -> clean_ratio:", tc.get("clean_ratio_start"), "/", tc.get("clean_ratio_end"),
      "| mixed_time_steps:", tc.get("mixed_time_steps"))

splits = build_data_splits(dataset_cfg)
loader = build_subset_loader(splits.test_loader, N_IMAGES, 0, balanced=True,
                             batch_size=BATCH, shuffle=False)
print("eval images:", len(loader.dataset))

pgd_cfg = AttackConfig(attack_name="pgd", objective="untargeted", epsilon=8/255,
    alpha=2/255, steps=20, eot_iters=1, random_start=True,
    exit_aware=True, force_full_trace=True, attack_time_steps=8)
aa_cfg = AttackConfig(attack_name="autoattack", epsilon=8/255,
    autoattack_version="standard", exit_aware=True, force_full_trace=True,
    eot_iters=1, attack_time_steps=8)

conditions = [("clean", None), ("PGD-20", pgd_cfg)]
if RUN_AUTOATTACK and HAS_AUTOATTACK:
    conditions.append(("AutoAttack", aa_cfg))
elif RUN_AUTOATTACK:
    print("!! AutoAttack unavailable (offline-src missing autoattack) -- skipping")

rule = build_stop_rule("max_softmax")
rows = []
for theta in THRESHOLDS:
    for name, acfg in conditions:
        print(f"theta={theta}  {name} ...", flush=True)
        seed_everything(0)
        d = collect_temporal_diagnostics(model, loader, time_steps=8, stop_rule=rule,
                                         threshold=theta, attack_config=acfg,
                                         device=DEVICE, max_batches=None)
        for i, (c, a) in enumerate(zip(d["avg_confidence_by_step"],
                                       d["avg_accuracy_by_step"]), start=1):
            rows.append(dict(threshold=theta, condition=name, step=i,
                             confidence=c, accuracy=a))
        print("   exit histogram:", d["step_histogram"])
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

df = pd.DataFrame(rows)
path = OUT_ROOT / "trajectories.csv"
df.to_csv(path, index=False)
print("saved", path)
display(df.pivot_table(index="step", columns=["threshold", "condition"],
                       values=["confidence", "accuracy"]).round(3))
'''


def build():
    cells = [
        _markdown_cell(
            "# VT Temporal Diagnostics (Kaggle, offline)\n\n"
            "**Eval only - no training.** Loads an existing checkpoint and records "
            "per-timestep confidence and accuracy under clean / PGD / AutoAttack.\n\n"
            "Attach: `offline-src` (CIFAR-10 + AutoAttack) and a dataset containing your `.pt`.\n\n"
            "Output: `trajectories.csv` -> feeds the *why VT collapses* figure."),
        _markdown_cell("## Offline setup"),
        _code_cell(OFFLINE_SETUP),
    ]
    for m in MODULE_ORDER:
        cells.append(_markdown_cell("## Module: " + m))
        cells.append(_code_cell(_read_module(m)))
    cells.append(_markdown_cell("## Diagnostics driver"))
    cells.append(_code_cell(DRIVER))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUT)
