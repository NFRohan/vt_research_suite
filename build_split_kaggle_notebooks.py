from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "vt_research_direction.ipynb"
TRAINER_NOTEBOOK = ROOT / "vt_research_train_kaggle.ipynb"
EVALUATOR_NOTEBOOK = ROOT / "vt_research_evaluate_kaggle.ipynb"


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
    patterns = [
        r"^from \.configs import .*\n",
        r"^from \.data import .*\n",
        r"^from \.stopping import .*\n",
        r"^from \.model import .*\n",
        r"^from \.metrics import .*\n",
        r"^from \.attacks import .*\n",
        r"^from \.training import .*\n",
        r"^from \.experiments import .*\n",
    ]
    for pattern in patterns:
        source = re.sub(pattern, "", source, flags=re.MULTILINE)
    source = source.replace("    from .experiments import evaluate_policy\n", "")
    return source


def _read_module(name: str) -> tuple[str, str]:
    path = ROOT / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    source = _strip_relative_imports(source).strip() + "\n"
    return name, source


def _optional_install_cell() -> dict:
    return _code_cell(
        """RUN_OPTIONAL_INSTALLS = False

OPTIONAL_PACKAGES = {
    "torchattacks": "torchattacks",
    "autoattack": "git+https://github.com/fra31/auto-attack.git",
    "tonic": "tonic",
}

if RUN_OPTIONAL_INSTALLS:
    import subprocess
    import sys

    for module_name, install_target in OPTIONAL_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing {install_target} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", install_target])
"""
    )


def _driver_imports_cell() -> dict:
    return _code_cell(
        """from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
"""
    )


def _base_notebook_cells(title: str, description: str) -> list[dict]:
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
        _markdown_cell(f"# {title}\n\n{description}"),
        _optional_install_cell(),
    ]

    for module_name in module_order:
        pretty_name = module_name.replace("_", " ").title()
        cells.append(_markdown_cell(f"## {pretty_name}"))
        _, source = _read_module(module_name)
        cells.append(_code_cell(source))

    return cells


def _load_source_cells() -> list[dict]:
    return json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _kaggleize_config_cell(source: str, overrides: dict[str, str]) -> str:
    source = source.replace(
        'EXPERIMENT_ROOT = Path("vt_research_suite_outputs")',
        'DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")\n'
        'DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"\n'
        'EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT',
    )
    source = source.replace('    root="./data",', '    root=DEFAULT_DATA_ROOT,')
    for name, value in overrides.items():
        source = re.sub(rf"^{name} = .*$", f"{name} = {value}", source, flags=re.MULTILINE)
    return source


def _trainer_driver_cells(source_cells: list[dict]) -> list[dict]:
    config_source = "".join(source_cells[2]["source"])
    config_source = _kaggleize_config_cell(
        config_source,
        {
            "RUN_TRAINING": "True",
            "RUN_ABLATION_TRAINING": "False",
            "RUN_ARCHITECTURE_TRAINING": "False",
            "RUN_MULTI_SEED": "False",
            "RUN_EXTRA_DATASETS": "False",
            "PERSIST_OUTPUTS": "True",
            "REQUIRE_AUTOATTACK_FOR_FINAL": "False",
            "SHOW_NOTEBOOK_PROGRESS": "True",
        },
    )

    history_cell = """history_csv = EXPERIMENT_ROOT / f"{Path(training_cfg.checkpoint_name).stem}_history.csv"
if history is None:
    print("No training history found in memory. If you loaded a checkpoint without history, only the checkpoint path is available.")
else:
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv, index=False)
    print(f"Saved {history_csv}")
    display(history_df.tail())

print(f"Checkpoint path: {checkpoint_path}")
print("Attach this checkpoint to the evaluator notebook for calibration, matched fixed-T controls, and final attacks.")
"""

    return [
        _markdown_cell(
            "# Trainer Driver\n\n"
            "Use this notebook to train one backbone/seed at a time and save checkpoints plus training history."
        ),
        _driver_imports_cell(),
        _code_cell(config_source),
        source_cells[3],
        _markdown_cell("## Training Summary"),
        _code_cell(history_cell),
    ]


def _evaluator_multiseed_cell() -> dict:
    return _code_cell(
        """if RUN_MULTI_SEED:
    multiseed_rows = []
    seed_values = list(eval_cfg.seeds)
    for seed in progress_iter(seed_values, desc="Seeds", total=len(seed_values), leave=False):
        seed_dataset_cfg = replace(dataset_cfg, seed=seed)
        seed_splits = build_data_splits(seed_dataset_cfg)
        seed_train_cfg = replace(training_cfg, seed=seed, checkpoint_name=f"{model_cfg.architecture}_seed{seed}.pt")
        seed_everything(seed)
        seed_model = build_model(model_cfg, seed=seed).to(seed_train_cfg.device)
        seed_checkpoint = Path(seed_train_cfg.output_dir) / seed_train_cfg.checkpoint_name
        if not seed_checkpoint.exists():
            print(f"Skipping seed {seed}; checkpoint not found: {seed_checkpoint}")
            continue

        checkpoint = torch.load(seed_checkpoint, map_location=seed_train_cfg.device)
        seed_model.load_state_dict(checkpoint["model_state_dict"])

        seed_calibration = calibrate_compute_matched_threshold(
            seed_model,
            seed_splits,
            model_cfg,
            seed_train_cfg,
            vt_rule,
            f"multiseed_seed{seed}",
        )
        seed_threshold = float(seed_calibration["selected_threshold"])
        seed_test_loaders = get_test_loaders(seed_splits)
        seed_eval = evaluate_policy(
            seed_model,
            seed_test_loaders["vt"],
            mode="vt",
            time_steps=model_cfg.max_time_steps,
            stop_rule=vt_rule,
            threshold=seed_threshold,
            compute_metrics=True,
            exact_vt_compute=eval_cfg.exact_vt_compute,
            device=seed_train_cfg.device,
        )
        seed_eval["selected_threshold"] = seed_threshold
        seed_eval["target_synops"] = float(seed_calibration["selected_metric_value"])
        seed_eval["seed"] = seed
        seed_eval["architecture"] = model_cfg.architecture
        multiseed_rows.append(seed_eval)

    if not multiseed_rows:
        print("No seed checkpoints were found. Attach checkpoints like resnet_seed0.pt, resnet_seed1.pt, and resnet_seed2.pt first.")
    else:
        multiseed_df = pd.DataFrame(multiseed_rows)
        multiseed_summary = summarize_multiseed(multiseed_rows, group_cols=["architecture"])
        persist_frame(multiseed_df, "multiseed_results")
        persist_frame(multiseed_summary, "multiseed_summary")
        display(multiseed_df)
        display(multiseed_summary)
else:
    print("Set RUN_MULTI_SEED = True to evaluate the multi-seed summary from existing checkpoints.")
"""
    )


def _evaluator_driver_cells(source_cells: list[dict]) -> list[dict]:
    config_source = "".join(source_cells[2]["source"])
    config_source = _kaggleize_config_cell(
        config_source,
        {
            "RUN_TRAINING": "False",
            "RUN_ABLATION_TRAINING": "False",
            "RUN_ARCHITECTURE_TRAINING": "False",
            "RUN_MULTI_SEED": "False",
            "RUN_EXTRA_DATASETS": "False",
            "PERSIST_OUTPUTS": "True",
            "REQUIRE_AUTOATTACK_FOR_FINAL": "False",
            "SHOW_NOTEBOOK_PROGRESS": "True",
        },
    )

    cells: list[dict] = [
        _markdown_cell(
            "# Evaluator Driver\n\n"
            "Use this notebook to load saved checkpoints, calibrate VT on the calibration split, and produce final paper tables."
        ),
        _driver_imports_cell(),
        _code_cell(config_source),
        source_cells[3],
        source_cells[4],
    ]

    for idx in range(5, len(source_cells) - 1):
        cells.append(source_cells[idx])
    cells.append(source_cells[17])
    cells.append(_evaluator_multiseed_cell())
    return cells


def _notebook_payload(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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


def build_trainer_notebook() -> dict:
    source_cells = _load_source_cells()
    cells = _base_notebook_cells(
        title="VT Research Suite: Kaggle Trainer",
        description=(
            "This notebook packages the VT training workflow into a single Kaggle-ready file. "
            "Train one backbone or seed at a time, save checkpoints, and export training history."
        ),
    )
    cells.extend(_trainer_driver_cells(source_cells))
    return _notebook_payload(cells)


def build_evaluator_notebook() -> dict:
    source_cells = _load_source_cells()
    cells = _base_notebook_cells(
        title="VT Research Suite: Kaggle Evaluator",
        description=(
            "This notebook packages the VT evaluation workflow into a single Kaggle-ready file. "
            "Load saved checkpoints, calibrate VT, match fixed-T controls, and run final attacks and summaries."
        ),
    )
    cells.extend(_evaluator_driver_cells(source_cells))
    return _notebook_payload(cells)


def _write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    _write_notebook(TRAINER_NOTEBOOK, build_trainer_notebook())
    _write_notebook(EVALUATOR_NOTEBOOK, build_evaluator_notebook())


if __name__ == "__main__":
    main()
