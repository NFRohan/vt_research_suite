from __future__ import annotations

import json
from pathlib import Path

from build_split_kaggle_notebooks import (
    _base_notebook_cells,
    _code_cell,
    _driver_imports_cell,
    _markdown_cell,
    _notebook_payload,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "vt_research_train_deterministic_kaggle.ipynb"


def _config_cell() -> dict:
    return _code_cell(
        """from dataclasses import asdict, replace
import json

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vt_research_suite_outputs") if Path("/kaggle/working").exists() else Path("vt_research_suite_outputs")
DEFAULT_DATA_ROOT = "/kaggle/working/data" if Path("/kaggle/working").exists() else "./data"
EXPERIMENT_ROOT = DEFAULT_OUTPUT_ROOT
EXPERIMENT_ROOT.mkdir(exist_ok=True)

SHOW_NOTEBOOK_PROGRESS = True
FORCE_RETRAIN = False
TRAIN_ARCHITECTURES = ("resnet", "vgg", "wrn")

dataset_cfg = DatasetConfig(
    name="cifar10",
    root=DEFAULT_DATA_ROOT,
    batch_size=256,
    num_workers=2,
    seed=0,
    augment=True,
    val_fraction=0.1,
    calibration_fraction=0.1,
    val_subset=None,
    calib_subset=None,
    test_subset=None,
)

base_model_cfg = ModelConfig(
    architecture="resnet",
    num_classes=10,
    in_channels=3,
    max_time_steps=8,
    stochastic=False,
    learnable_sigma=False,
    use_dynamic_sigma=False,
    sigma_init_space="raw",
)

training_cfg = TrainingConfig(
    epochs=100,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir=str(EXPERIMENT_ROOT / "deterministic_checkpoints"),
    checkpoint_name="deterministic_placeholder.pt",
    mixed_time_steps=tuple(range(1, base_model_cfg.max_time_steps + 1)),
    selection_metric="clean",
    use_amp=True,
    eval_interval=5,
    epsilon=8/255,
    alpha=8/255,
)

architecture_presets = {
    "resnet": replace(
        base_model_cfg,
        architecture="resnet",
        stem_channels=128,
        stage_channels=(128, 256, 512),
        blocks_per_stage=(3, 3, 2),
    ),
    "vgg": replace(
        base_model_cfg,
        architecture="vgg",
        stem_channels=64,
        stage_channels=(64, 128, 256, 512, 512),
        blocks_per_stage=(1, 1, 2, 2, 2),
        dropout_rate=0.5,
    ),
    "wrn": replace(
        base_model_cfg,
        architecture="wrn",
        stem_channels=16,
        widen_factor=10,
        wrn_depth=28,
        stage_channels=(160, 320, 640),
        blocks_per_stage=(4, 4, 4),
    ),
}

print(dataset_cfg)
print(base_model_cfg)
print(training_cfg)
print("Deterministic training architectures:", TRAIN_ARCHITECTURES)
"""
    )


def _driver_cell() -> dict:
    return _code_cell(
        """splits = build_data_splits(dataset_cfg)
summary_rows = []

for architecture_name in TRAIN_ARCHITECTURES:
    if architecture_name not in architecture_presets:
        raise ValueError(f"Unknown architecture preset: {architecture_name}")

    model_cfg = architecture_presets[architecture_name]
    run_training_cfg = replace(
        training_cfg,
        checkpoint_name=f"{architecture_name}_deterministic_full_model.pt",
    )
    checkpoint_path = Path(run_training_cfg.output_dir) / run_training_cfg.checkpoint_name
    history_csv = EXPERIMENT_ROOT / f"{Path(run_training_cfg.checkpoint_name).stem}_history.csv"
    config_json = EXPERIMENT_ROOT / f"{Path(run_training_cfg.checkpoint_name).stem}_config.json"

    print(f"\\n=== Training {architecture_name} deterministic baseline ===")
    print(model_cfg)
    print(f"Checkpoint: {checkpoint_path}")

    if checkpoint_path.exists() and not FORCE_RETRAIN:
        print("Checkpoint already exists; skipping training.")
        history = None
    else:
        seed_everything(run_training_cfg.seed)
        model = build_model(model_cfg, seed=run_training_cfg.seed).to(run_training_cfg.device)
        history, checkpoint_path = fit_model(
            model,
            splits.train_loader,
            splits.val_loader,
            run_training_cfg,
            selection_attack_config=None,
        )
        model.reset()

        if history is not None:
            history_df = pd.DataFrame(history)
            history_df.to_csv(history_csv, index=False)
            print(f"Saved {history_csv}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint_path}")

    if history_csv.exists():
        history_df = pd.read_csv(history_csv)
        best_val_clean = float(history_df["val_clean_acc"].dropna().max()) if history_df["val_clean_acc"].notna().any() else float("nan")
        final_val_clean = float(history_df["val_clean_acc"].dropna().iloc[-1]) if history_df["val_clean_acc"].notna().any() else float("nan")
        final_train_clean = float(history_df["train_clean_acc"].iloc[-1])
        final_train_adv = float(history_df["train_adv_acc"].iloc[-1])
    else:
        best_val_clean = float("nan")
        final_val_clean = float("nan")
        final_train_clean = float("nan")
        final_train_adv = float("nan")

    config_payload = {
        "dataset_cfg": asdict(dataset_cfg),
        "model_cfg": asdict(model_cfg),
        "training_cfg": asdict(run_training_cfg),
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
    }
    config_json.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Saved {config_json}")

    summary_rows.append(
        {
            "architecture": architecture_name,
            "checkpoint_path": str(checkpoint_path),
            "history_csv": str(history_csv),
            "best_val_clean_acc": best_val_clean,
            "final_val_clean_acc": final_val_clean,
            "final_train_clean_acc": final_train_clean,
            "final_train_adv_acc": final_train_adv,
            "stochastic": model_cfg.stochastic,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_csv = EXPERIMENT_ROOT / "deterministic_training_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved {summary_csv}")
display(summary_df)
"""
    )


def build_notebook() -> dict:
    cells = _base_notebook_cells(
        title="VT Research Suite: Deterministic Kaggle Trainer",
        description=(
            "This notebook trains deterministic control baselines for ResNet, VGG11-style, "
            "and WideResNet-28-10 SNN backbones sequentially. Upload to Kaggle, set the "
            "dataset/output paths if needed, and run all cells."
        ),
    )
    cells.extend(
        [
            _markdown_cell(
                "# Deterministic Training Driver\n\n"
                "This driver disables stochastic noise in the neuron model, trains all three backbones "
                "sequentially, and saves checkpoints plus history CSVs for later evaluator runs."
            ),
            _driver_imports_cell(),
            _config_cell(),
            _driver_cell(),
        ]
    )
    return _notebook_payload(cells)


def main() -> None:
    OUTPUT_NOTEBOOK.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
