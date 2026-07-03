from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Iterable

import pandas as pd
import torch

from .attacks import autoattack_attack, pgd_attack
from .configs import AttackConfig
from .metrics import ComputeMonitor
from .model import TraceResult, TraceableSpikingModel
from .stopping import StopRule, build_stop_rule

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _experiments_maybe_len(iterable) -> int | None:
    try:
        return len(iterable)
    except TypeError:
        return None


def _experiments_progress(iterable, desc: str, total: int | None = None, leave: bool = False):
    if tqdm is None:
        return iterable
    resolved_total = _experiments_maybe_len(iterable) if total is None else total
    return tqdm(iterable, total=resolved_total, desc=desc, leave=leave, dynamic_ncols=True)


def _iter_batches(dataloader, max_batches: int | None, desc: str, leave: bool = False):
    total = _experiments_maybe_len(dataloader)
    if total is not None and max_batches is not None:
        total = min(total, max_batches)
    iterator = _experiments_progress(dataloader, desc=desc, total=total, leave=leave)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch


def _run_trace(
    model: TraceableSpikingModel,
    inputs: torch.Tensor,
    mode: str,
    time_steps: int,
    stop_rule: StopRule | None,
    threshold: float,
    force_full: bool = False,
) -> TraceResult:
    if mode == "fixed":
        return model.forward_trace(inputs, time_steps=time_steps, force_full=force_full)
    return model.forward_trace(
        inputs,
        time_steps=time_steps,
        stop_rule=stop_rule,
        threshold=threshold,
        force_full=force_full,
    )


def _empty_result(mode: str, time_steps: int, stop_rule: StopRule | None, threshold: float, attack_config: AttackConfig | None, compute_mode: str) -> dict[str, object]:
    autoattack_version = None
    if attack_config is not None and attack_config.attack_name == "autoattack":
        autoattack_version = attack_config.autoattack_version or ("rand" if attack_config.eot_iters > 1 else "standard")
    return {
        "mode": mode,
        "time_steps": time_steps,
        "stop_rule": stop_rule.name if stop_rule is not None else "fixed",
        "threshold": threshold if mode == "vt" else None,
        "accuracy": 0.0,
        "avg_steps": 0.0,
        "avg_stopping_depth": 0.0,
        "avg_executed_steps": 0.0,
        "avg_confidence": 0.0,
        "step_histogram": {},
        "compute_mode": compute_mode,
        "attack_name": attack_config.attack_name if attack_config is not None else "clean",
        "attack_objective": attack_config.objective if attack_config is not None else "clean",
        "attack_steps": attack_config.steps if attack_config is not None else 0,
        "attack_eot": attack_config.eot_iters if attack_config is not None else 0,
        "exit_aware_attack": bool(attack_config.exit_aware) if attack_config is not None else False,
        "autoattack_version": autoattack_version,
    }


def _make_attack_inputs(
    model: TraceableSpikingModel,
    data: torch.Tensor,
    labels: torch.Tensor,
    attack_config: AttackConfig | None,
    time_steps: int,
    stop_rule: StopRule | None,
    threshold: float,
) -> torch.Tensor:
    if attack_config is None:
        return data
    if attack_config.attack_name == "pgd":
        return pgd_attack(model, data, labels, attack_config, time_steps, stop_rule=stop_rule, threshold=threshold)
    if attack_config.attack_name == "autoattack":
        return autoattack_attack(model, data, labels, attack_config, time_steps, stop_rule=stop_rule, threshold=threshold)
    raise ValueError(f"Unsupported attack_name: {attack_config.attack_name}")


def _update_metrics(totals: defaultdict[float, float], summary, weight: int) -> None:
    totals["synops"] += summary.avg_synops * weight
    totals["dense_ops"] += summary.avg_dense_ops * weight
    totals["spike_count"] += summary.avg_spike_count * weight
    totals["spike_density"] += summary.avg_spike_density * weight
    totals["activation_sparsity"] += summary.activation_sparsity * weight


def evaluate_policy(
    model: TraceableSpikingModel,
    dataloader,
    mode: str,
    time_steps: int,
    stop_rule: StopRule | None = None,
    threshold: float = 0.95,
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    exact_vt_compute: bool = True,
    device: str = "cuda",
    max_batches: int | None = None,
    force_full_trace: bool = False,
    progress_desc: str | None = None,
) -> dict[str, object]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    step_histogram: Counter[int] = Counter()
    confidence_rule = stop_rule or build_stop_rule("max_softmax")
    compute_mode = "sample_exact" if compute_metrics and mode == "vt" and exact_vt_compute else "batched_proxy"
    attack_name = attack_config.attack_name if attack_config is not None else "clean"
    desc = progress_desc or f"Eval {mode} [{attack_name}]"

    for data, labels in _iter_batches(dataloader, max_batches, desc=desc, leave=False):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if compute_mode == "sample_exact":
            sample_iterator = _experiments_progress(range(data.size(0)), desc=f"{desc} samples", total=data.size(0), leave=False)
            for sample_idx in sample_iterator:
                single_data = data[sample_idx:sample_idx + 1]
                single_label = labels[sample_idx:sample_idx + 1]
                inputs = (
                    _make_attack_inputs(model, single_data, single_label, attack_config, time_steps, stop_rule, threshold)
                    if attack_config is not None
                    else single_data
                )
                with ComputeMonitor(model) as monitor:
                    with torch.no_grad():
                        trace = _run_trace(
                            model,
                            inputs,
                            mode=mode,
                            time_steps=time_steps,
                            stop_rule=stop_rule,
                            threshold=threshold,
                            force_full=force_full_trace,
                        )
                summary = monitor.summary(1)
                _update_metrics(totals, summary, 1)
                prediction = trace.final_logits.argmax(dim=1)
                totals["correct"] += prediction.eq(single_label).sum().item()
                totals["count"] += 1
                totals["stopping_depth"] += trace.steps_taken.item()
                totals["executed_steps"] += trace.executed_steps
                totals["confidence"] += confidence_rule.confidence(trace.final_logits).sum().item()
                step_histogram.update(trace.steps_taken.detach().cpu().tolist())
            continue

        inputs = _make_attack_inputs(model, data, labels, attack_config, time_steps, stop_rule, threshold)

        if compute_metrics:
            with ComputeMonitor(model) as monitor:
                with torch.no_grad():
                    trace = _run_trace(
                        model,
                        inputs,
                        mode=mode,
                        time_steps=time_steps,
                        stop_rule=stop_rule,
                        threshold=threshold,
                        force_full=force_full_trace,
                    )
            summary = monitor.summary(labels.size(0))
            _update_metrics(totals, summary, labels.size(0))
        else:
            with torch.no_grad():
                trace = _run_trace(
                    model,
                    inputs,
                    mode=mode,
                    time_steps=time_steps,
                    stop_rule=stop_rule,
                    threshold=threshold,
                    force_full=force_full_trace,
                )

        predictions = trace.final_logits.argmax(dim=1)
        totals["correct"] += predictions.eq(labels).sum().item()
        totals["count"] += labels.size(0)
        totals["stopping_depth"] += trace.steps_taken.sum().item()
        totals["executed_steps"] += trace.executed_steps * labels.size(0)
        totals["confidence"] += confidence_rule.confidence(trace.final_logits).sum().item()
        step_histogram.update(trace.steps_taken.detach().cpu().tolist())

    count = max(totals["count"], 1.0)
    result = _empty_result(mode, time_steps, stop_rule, threshold, attack_config, compute_mode)
    result.update(
        {
            "accuracy": 100.0 * totals["correct"] / count,
            "avg_steps": totals["stopping_depth"] / count,
            "avg_stopping_depth": totals["stopping_depth"] / count,
            "avg_executed_steps": totals["executed_steps"] / count,
            "avg_confidence": totals["confidence"] / count,
            "step_histogram": dict(sorted(step_histogram.items())),
        }
    )
    if compute_metrics:
        result.update(
            {
                "avg_synops": totals["synops"] / count,
                "avg_dense_ops": totals["dense_ops"] / count,
                "avg_spike_count": totals["spike_count"] / count,
                "avg_spike_density": totals["spike_density"] / count,
                "activation_sparsity": totals["activation_sparsity"] / count,
            }
        )
    model.reset()
    return result


def threshold_sweep(
    model: TraceableSpikingModel,
    dataloader,
    thresholds: Iterable[float],
    stop_rule: StopRule,
    time_steps: int,
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    exact_vt_compute: bool = True,
    device: str = "cuda",
    max_batches: int | None = None,
) -> pd.DataFrame:
    threshold_values = list(thresholds)
    rows = []
    for threshold in _experiments_progress(threshold_values, desc=f"VT threshold sweep [{stop_rule.name}]", total=len(threshold_values), leave=False):
        rows.append(
            evaluate_policy(
            model=model,
            dataloader=dataloader,
            mode="vt",
            time_steps=time_steps,
            stop_rule=stop_rule,
            threshold=threshold,
            attack_config=attack_config,
            compute_metrics=compute_metrics,
            exact_vt_compute=exact_vt_compute,
            device=device,
            max_batches=max_batches,
            progress_desc=f"VT eval [{stop_rule.name}] theta={threshold:.2f}",
        )
        )
    return pd.DataFrame(rows)


def fixed_time_sweep(
    model: TraceableSpikingModel,
    dataloader,
    time_step_values: Iterable[int],
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    device: str = "cuda",
    max_batches: int | None = None,
) -> pd.DataFrame:
    step_values = list(time_step_values)
    rows = []
    for time_steps in _experiments_progress(step_values, desc="Fixed-T sweep", total=len(step_values), leave=False):
        rows.append(
            evaluate_policy(
            model=model,
            dataloader=dataloader,
            mode="fixed",
            time_steps=time_steps,
            attack_config=attack_config,
            compute_metrics=compute_metrics,
            exact_vt_compute=False,
            device=device,
            max_batches=max_batches,
            progress_desc=f"Fixed-T eval T={time_steps}",
        )
        )
    return pd.DataFrame(rows)


def _pick_match_column(vt_results: pd.DataFrame, fixed_results: pd.DataFrame, match_metric: str | None) -> str:
    if match_metric is not None:
        return match_metric
    if "avg_synops" in vt_results.columns and "avg_synops" in fixed_results.columns:
        return "avg_synops"
    if "avg_executed_steps" in vt_results.columns and "avg_executed_steps" in fixed_results.columns:
        return "avg_executed_steps"
    return "avg_stopping_depth"


def match_fixed_time_controls(
    vt_results: pd.DataFrame,
    fixed_results: pd.DataFrame,
    match_metric: str | None = None,
) -> pd.DataFrame:
    matches = []
    metric = _pick_match_column(vt_results, fixed_results, match_metric)
    for _, row in vt_results.iterrows():
        distances = (fixed_results[metric] - row[metric]).abs()
        fixed_row = fixed_results.loc[distances.idxmin()]
        match = {
            "match_metric": metric,
            "stop_rule": row["stop_rule"],
            "threshold": row["threshold"],
            "vt_accuracy": row["accuracy"],
            "vt_avg_stopping_depth": row["avg_stopping_depth"],
            "vt_avg_executed_steps": row.get("avg_executed_steps"),
            "matched_fixed_t": fixed_row["time_steps"],
            "fixed_accuracy": fixed_row["accuracy"],
            "fixed_avg_executed_steps": fixed_row.get("avg_executed_steps"),
            "metric_gap": abs(fixed_row[metric] - row[metric]),
            "accuracy_delta": row["accuracy"] - fixed_row["accuracy"],
        }
        if "avg_synops" in row and "avg_synops" in fixed_row:
            match["vt_avg_synops"] = row["avg_synops"]
            match["fixed_avg_synops"] = fixed_row["avg_synops"]
            match["synops_delta"] = row["avg_synops"] - fixed_row["avg_synops"]
        matches.append(match)
    return pd.DataFrame(matches)


def select_threshold_neighbors(
    vt_results: pd.DataFrame,
    selected_threshold: float,
    count: int = 3,
) -> pd.DataFrame:
    if "threshold" not in vt_results.columns:
        raise ValueError("VT results must include a `threshold` column.")
    unique_thresholds = sorted({float(value) for value in vt_results["threshold"].dropna().tolist()})
    if not unique_thresholds:
        return pd.DataFrame(columns=["threshold", "threshold_role", "distance_from_selected", "selection_rank"])

    target_count = max(1, min(count, len(unique_thresholds)))
    anchor_idx = min(
        range(len(unique_thresholds)),
        key=lambda idx: abs(unique_thresholds[idx] - float(selected_threshold)),
    )

    chosen_indices = [anchor_idx]
    step = 1
    while len(chosen_indices) < target_count:
        left_idx = anchor_idx - step
        right_idx = anchor_idx + step
        if left_idx >= 0:
            chosen_indices.append(left_idx)
            if len(chosen_indices) >= target_count:
                break
        if right_idx < len(unique_thresholds):
            chosen_indices.append(right_idx)
            if len(chosen_indices) >= target_count:
                break
        step += 1

    rows = []
    for rank, idx in enumerate(sorted(chosen_indices, key=lambda value: unique_thresholds[value])):
        threshold = unique_thresholds[idx]
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


def select_compute_matched_threshold(
    model: TraceableSpikingModel,
    dataloader,
    thresholds: Iterable[float],
    fixed_time_steps: Iterable[int],
    stop_rule: StopRule,
    time_steps: int,
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    exact_vt_compute: bool = True,
    match_metric: str | None = None,
    device: str = "cuda",
    max_batches: int | None = None,
    vt_dataloader=None,
    fixed_dataloader=None,
) -> dict[str, object]:
    vt_loader = dataloader if vt_dataloader is None else vt_dataloader
    fixed_loader = dataloader if fixed_dataloader is None else fixed_dataloader
    if vt_loader is None or fixed_loader is None:
        raise ValueError("select_compute_matched_threshold requires at least one dataloader.")
    vt_results = threshold_sweep(
        model,
        vt_loader,
        thresholds=thresholds,
        stop_rule=stop_rule,
        time_steps=time_steps,
        attack_config=attack_config,
        compute_metrics=compute_metrics,
        exact_vt_compute=exact_vt_compute,
        device=device,
        max_batches=max_batches,
    )
    fixed_results = fixed_time_sweep(
        model,
        fixed_loader,
        time_step_values=fixed_time_steps,
        attack_config=attack_config,
        compute_metrics=compute_metrics,
        device=device,
        max_batches=max_batches,
    )
    matches = match_fixed_time_controls(vt_results, fixed_results, match_metric=match_metric)
    if matches.empty:
        raise RuntimeError("No VT/fixed-T matches were produced during threshold calibration.")
    sort_cols = [column for column in ["accuracy_delta", "synops_delta", "metric_gap"] if column in matches.columns]
    if not sort_cols:
        raise RuntimeError("Threshold calibration produced matches without sortable comparison columns.")
    ascending = [False if column == "accuracy_delta" else True for column in sort_cols]
    selected_match = matches.sort_values(sort_cols, ascending=ascending).iloc[0]
    effective_match_metric = selected_match["match_metric"]
    selected_threshold = float(selected_match["threshold"])
    metric_value = float(vt_results.loc[vt_results["threshold"] == selected_threshold, effective_match_metric].iloc[0])
    return {
        "selected_threshold": selected_threshold,
        "selected_metric_value": metric_value,
        "selected_match": selected_match.to_dict(),
        "vt_results": vt_results,
        "fixed_results": fixed_results,
        "matches": matches,
    }


def calibrate_threshold(
    model: TraceableSpikingModel,
    dataloader,
    stop_rule: StopRule,
    candidate_thresholds: Iterable[float],
    target_value: float,
    time_steps: int,
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    exact_vt_compute: bool = True,
    target_metric: str = "avg_stopping_depth",
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict[str, object]:
    best_result = None
    best_gap = None
    threshold_values = list(candidate_thresholds)
    for threshold in _experiments_progress(threshold_values, desc=f"Calibrate [{stop_rule.name}]", total=len(threshold_values), leave=False):
        result = evaluate_policy(
            model=model,
            dataloader=dataloader,
            mode="vt",
            time_steps=time_steps,
            stop_rule=stop_rule,
            threshold=threshold,
            attack_config=attack_config,
            compute_metrics=compute_metrics,
            exact_vt_compute=exact_vt_compute,
            device=device,
            max_batches=max_batches,
            progress_desc=f"Calib eval [{stop_rule.name}] theta={threshold:.2f}",
        )
        gap = abs(result[target_metric] - target_value)
        if best_result is None or gap < best_gap:
            best_result = result
            best_gap = gap
    assert best_result is not None
    best_result["target_metric"] = target_metric
    best_result["target_value"] = target_value
    best_result["metric_gap"] = best_gap
    return best_result


def compare_stopping_rules(
    model: TraceableSpikingModel,
    dataloader,
    rule_names: Iterable[str],
    candidate_thresholds: Iterable[float],
    time_steps: int,
    target_value: float | None = None,
    attack_config: AttackConfig | None = None,
    compute_metrics: bool = False,
    exact_vt_compute: bool = True,
    target_metric: str = "avg_stopping_depth",
    device: str = "cuda",
    max_batches: int | None = None,
) -> pd.DataFrame:
    rows = []
    rule_values = list(rule_names)
    for rule_name in _experiments_progress(rule_values, desc="Stopping-rule comparison", total=len(rule_values), leave=False):
        stop_rule = build_stop_rule(rule_name)
        if target_value is None:
            sweep = threshold_sweep(
                model,
                dataloader,
                candidate_thresholds,
                stop_rule,
                time_steps,
                attack_config=attack_config,
                compute_metrics=compute_metrics,
                exact_vt_compute=exact_vt_compute,
                device=device,
                max_batches=max_batches,
            )
            rows.extend(sweep.to_dict("records"))
        else:
            rows.append(
                calibrate_threshold(
                    model,
                    dataloader,
                    stop_rule,
                    candidate_thresholds,
                    target_value,
                    time_steps,
                    attack_config=attack_config,
                    compute_metrics=compute_metrics,
                    exact_vt_compute=exact_vt_compute,
                    target_metric=target_metric,
                    device=device,
                    max_batches=max_batches,
                )
            )
    return pd.DataFrame(rows)


def collect_temporal_diagnostics(
    model: TraceableSpikingModel,
    dataloader,
    time_steps: int,
    stop_rule: StopRule | None = None,
    threshold: float = 0.95,
    attack_config: AttackConfig | None = None,
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    confidence_rule = stop_rule or build_stop_rule("max_softmax")
    confidence_sums = None
    accuracy_sums = None
    counts = None
    step_histogram: Counter[int] = Counter()

    for data, labels in _iter_batches(dataloader, max_batches, desc="Temporal diagnostics", leave=False):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = _make_attack_inputs(model, data, labels, attack_config, time_steps, stop_rule, threshold)
        with torch.no_grad():
            trace = _run_trace(
                model,
                inputs,
                mode="vt" if stop_rule is not None else "fixed",
                time_steps=time_steps,
                stop_rule=stop_rule,
                threshold=threshold,
                force_full=True,
            )

        steps = trace.avg_logits_per_step.size(0)
        if confidence_sums is None:
            confidence_sums = torch.zeros(steps, dtype=torch.float64)
            accuracy_sums = torch.zeros(steps, dtype=torch.float64)
            counts = torch.zeros(steps, dtype=torch.float64)

        for step in range(steps):
            step_logits = trace.avg_logits_per_step[step]
            confidence_sums[step] += confidence_rule.confidence(step_logits).sum().item()
            accuracy_sums[step] += step_logits.argmax(dim=1).eq(labels).sum().item()
            counts[step] += labels.size(0)
        step_histogram.update(trace.steps_taken.detach().cpu().tolist())

    model.reset()
    if confidence_sums is None or accuracy_sums is None or counts is None:
        return {"avg_confidence_by_step": [], "avg_accuracy_by_step": [], "step_histogram": {}}

    return {
        "avg_confidence_by_step": (confidence_sums / counts.clamp(min=1)).tolist(),
        "avg_accuracy_by_step": (100.0 * accuracy_sums / counts.clamp(min=1)).tolist(),
        "step_histogram": dict(sorted(step_histogram.items())),
    }


def summarize_multiseed(records: list[dict[str, object]], group_cols: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    numeric_cols = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    summaries = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        for column in numeric_cols:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95"] = ci95
        summaries.append(row)
    return pd.DataFrame(summaries)
