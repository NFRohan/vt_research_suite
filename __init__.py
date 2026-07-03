from __future__ import annotations

from importlib import import_module

from .configs import (
    AblationSpec,
    AttackConfig,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    TrainingConfig,
    apply_ablation,
    default_ablation_specs,
)
from .stopping import build_stop_rule

__all__ = [
    "AblationSpec",
    "AttackConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "TrainingConfig",
    "apply_ablation",
    "autoattack_attack",
    "build_stop_rule",
    "default_ablation_specs",
    "build_data_splits",
    "build_dataloaders",
    "rebuild_loader",
    "build_model",
    "build_optimizer",
    "calibrate_threshold",
    "collect_temporal_diagnostics",
    "compare_stopping_rules",
    "evaluate_policy",
    "fit_model",
    "fixed_time_sweep",
    "get_dataset_metadata",
    "match_fixed_time_controls",
    "select_threshold_neighbors",
    "select_compute_matched_threshold",
    "pgd_attack",
    "seed_everything",
    "summarize_multiseed",
    "threshold_sweep",
    "train_one_epoch",
    "TraceableSpikingModel",
    "VTDynRandHetSNN",
]

_LAZY_ATTRS = {
    "build_data_splits": ("vt_research_suite.data", "build_data_splits"),
    "build_dataloaders": ("vt_research_suite.data", "build_dataloaders"),
    "get_dataset_metadata": ("vt_research_suite.data", "get_dataset_metadata"),
    "rebuild_loader": ("vt_research_suite.data", "rebuild_loader"),
    "build_model": ("vt_research_suite.model", "build_model"),
    "TraceableSpikingModel": ("vt_research_suite.model", "TraceableSpikingModel"),
    "VTDynRandHetSNN": ("vt_research_suite.model", "VTDynRandHetSNN"),
    "autoattack_attack": ("vt_research_suite.attacks", "autoattack_attack"),
    "pgd_attack": ("vt_research_suite.attacks", "pgd_attack"),
    "calibrate_threshold": ("vt_research_suite.experiments", "calibrate_threshold"),
    "collect_temporal_diagnostics": ("vt_research_suite.experiments", "collect_temporal_diagnostics"),
    "compare_stopping_rules": ("vt_research_suite.experiments", "compare_stopping_rules"),
    "evaluate_policy": ("vt_research_suite.experiments", "evaluate_policy"),
    "fixed_time_sweep": ("vt_research_suite.experiments", "fixed_time_sweep"),
    "match_fixed_time_controls": ("vt_research_suite.experiments", "match_fixed_time_controls"),
    "select_threshold_neighbors": ("vt_research_suite.experiments", "select_threshold_neighbors"),
    "select_compute_matched_threshold": ("vt_research_suite.experiments", "select_compute_matched_threshold"),
    "summarize_multiseed": ("vt_research_suite.experiments", "summarize_multiseed"),
    "threshold_sweep": ("vt_research_suite.experiments", "threshold_sweep"),
    "build_optimizer": ("vt_research_suite.training", "build_optimizer"),
    "fit_model": ("vt_research_suite.training", "fit_model"),
    "seed_everything": ("vt_research_suite.training", "seed_everything"),
    "train_one_epoch": ("vt_research_suite.training", "train_one_epoch"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
