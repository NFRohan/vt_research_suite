from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SOURCE_NOTEBOOK = ROOT / "vt_research_direction.ipynb"
OUTPUT_NOTEBOOK = ROOT / "vt_research_direction_kaggle_all_in_one.ipynb"


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


def _driver_cells() -> list[dict]:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    driver_cells: list[dict] = []

    driver_cells.append(
        _markdown_cell(
            "# Experiment Driver\n\n"
            "The cells below mirror the paper pipeline from the modular notebook, "
            "but they run entirely inside this one Kaggle notebook."
        )
    )

    driver_imports = """from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
"""
    driver_cells.append(_code_cell(driver_imports))

    config_cell = "".join(cells[2]["source"])
    config_cell = config_cell.replace(
        'EXPERIMENT_ROOT = Path("vt_research_suite_outputs")',
        'DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")\n'
        'DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"\n'
        'EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT'
    )
    config_cell = config_cell.replace('    root="./data",', '    root=DEFAULT_DATA_ROOT,')
    driver_cells.append(_code_cell(config_cell))

    for cell in cells[3:]:
        driver_cells.append(cell)
    return driver_cells


def build_notebook() -> dict:
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
            "# VT Research Suite: Kaggle All-In-One\n\n"
            "This notebook packages the full VT research pipeline into a single file for Kaggle deployment.\n\n"
            "It contains the configuration, data pipeline, model definitions, attacks, training loop, evaluation helpers, and the paper-run driver cells in one place."
        ),
        _code_cell(
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
        ),
    ]

    for module_name in module_order:
        pretty_name = module_name.replace("_", " ").title()
        cells.append(_markdown_cell(f"## {pretty_name}"))
        _, source = _read_module(module_name)
        cells.append(_code_cell(source))

    cells.extend(_driver_cells())

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


def main() -> None:
    notebook = build_notebook()
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
