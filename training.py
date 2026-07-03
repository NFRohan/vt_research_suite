from __future__ import annotations

import gc
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import AttackConfig, TrainingConfig, ensure_output_dir
from .model import AdaptiveRHLIFNode, TraceableSpikingModel, get_sigma_stats

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_context(config: TrainingConfig):
    device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
    enabled = config.use_amp and device_type == "cuda"
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=getattr(torch, config.amp_dtype))


def _build_grad_scaler(enabled: bool):
    try:
        return GradScaler("cuda", enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def _clean_ratio(epoch: int, total_epochs: int, config: TrainingConfig) -> float:
    if total_epochs <= 1:
        return config.clean_ratio_end
    progress = (epoch - 1) / (total_epochs - 1)
    return config.clean_ratio_start + (config.clean_ratio_end - config.clean_ratio_start) * progress


def _sample_time_steps(config: TrainingConfig) -> int:
    return random.choice(config.mixed_time_steps)


def _training_maybe_len(iterable) -> int | None:
    try:
        return len(iterable)
    except TypeError:
        return None


def _training_progress(iterable, desc: str, total: int | None = None, leave: bool = False):
    if tqdm is None:
        return iterable
    resolved_total = _training_maybe_len(iterable) if total is None else total
    return tqdm(iterable, total=resolved_total, desc=desc, leave=leave, dynamic_ncols=True)


def _release_memory(
    device: str,
    model: TraceableSpikingModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    if model is not None:
        model.reset()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _entropy_regularization(model: nn.Module, target_sigma: float = 0.5) -> torch.Tensor:
    total = None
    count = 0
    for module in model.modules():
        if isinstance(module, AdaptiveRHLIFNode) and module.learnable_sigma:
            term = torch.clamp(target_sigma - module.sigma, min=0.0) ** 2
            total = term if total is None else total + term
            count += 1
    if total is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return total / max(count, 1)


def parseval_retraction(model: nn.Module, beta: float = 0.004) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                shape = weight.shape
                flat = weight.view(shape[0], -1)
                update = (1 + beta) * flat - beta * flat @ flat.t() @ flat
                module.weight.data.copy_(update.view(shape))


class AdversarialWeightPerturbation:
    def __init__(self, model: nn.Module, adv_param: str = "weight", adv_lr: float = 0.01, adv_eps: float = 0.01):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup: dict[str, torch.Tensor] = {}

    def perturb(self) -> None:
        eps = 1e-6
        for name, param in self.model.named_parameters():
            if self.adv_param not in name or param.grad is None or not param.requires_grad:
                continue
            if name not in self.backup:
                self.backup[name] = param.data.clone()
            grad_norm = torch.norm(param.grad)
            weight_norm = torch.norm(param.data.detach())
            if grad_norm == 0 or torch.isnan(grad_norm):
                continue
            step = self.adv_lr * param.grad / (grad_norm + eps) * (weight_norm + eps)
            step = torch.clamp(step, -self.adv_eps, self.adv_eps)
            param.data.add_(step)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def build_optimizer(model: TraceableSpikingModel, config: TrainingConfig) -> torch.optim.Optimizer:
    sigma_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "sigma" in name:
            sigma_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.SGD(
        [
            {"params": other_params, "lr": config.base_lr},
            {"params": sigma_params, "lr": config.base_lr * config.sigma_lr_scale},
        ],
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < config.warmup_epochs:
            return float(epoch_idx + 1) / max(config.warmup_epochs, 1)
        progress = (epoch_idx - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_rfgsm_batch(
    model: TraceableSpikingModel,
    data: torch.Tensor,
    labels: torch.Tensor,
    time_steps: int,
    config: TrainingConfig,
) -> torch.Tensor:
    delta = torch.zeros_like(data).uniform_(
        -config.epsilon * config.random_start_scale,
        config.epsilon * config.random_start_scale,
    )
    delta.requires_grad_(True)
    adv_inputs = torch.clamp(data + delta, 0.0, 1.0)
    with _autocast_context(config):
        logits = model(adv_inputs, time_steps=time_steps, use_vt=False)
        loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
    delta = torch.clamp(delta.detach() + config.alpha * grad.sign(), -config.epsilon, config.epsilon)
    return torch.clamp(data + delta, 0.0, 1.0)


def train_one_epoch(
    model: TraceableSpikingModel,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    epoch: int,
    awp: AdversarialWeightPerturbation | None = None,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    stats = {"loss": 0.0, "clean_correct": 0.0, "adv_correct": 0.0, "clean_total": 0.0, "adv_total": 0.0}
    clean_ratio = _clean_ratio(epoch, config.epochs, config)

    batch_iterator = _training_progress(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
    for batch_idx, (data, labels) in enumerate(batch_iterator):
        data = data.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        batch_size = data.size(0)
        time_steps = _sample_time_steps(config)

        n_clean = int(batch_size * clean_ratio)
        perm = torch.randperm(batch_size, device=config.device)
        clean_idx = perm[:n_clean]
        adv_idx = perm[n_clean:]

        clean_data = data[clean_idx]
        clean_labels = labels[clean_idx]
        adv_data = data[adv_idx]
        adv_labels = labels[adv_idx]

        adv_batch = None
        if adv_idx.numel() > 0:
            adv_batch = _build_rfgsm_batch(model, adv_data, adv_labels, time_steps, config)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(config):
            clean_loss = torch.tensor(0.0, device=config.device)
            adv_loss = torch.tensor(0.0, device=config.device)
            clean_logits = None
            adv_logits = None

            if clean_idx.numel() > 0:
                clean_logits = model(clean_data, time_steps=time_steps, use_vt=False)
                clean_loss = criterion(clean_logits, clean_labels)
            if adv_idx.numel() > 0 and adv_batch is not None:
                adv_logits = model(adv_batch, time_steps=time_steps, use_vt=False)
                adv_loss = criterion(adv_logits, adv_labels)
            main_loss = clean_ratio * clean_loss + (1.0 - clean_ratio) * adv_loss
            total_loss = main_loss + config.entropy_lambda * _entropy_regularization(model)

        scaler.scale(total_loss).backward()

        if awp is not None:
            awp.perturb()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(config):
                awp_clean_loss = torch.tensor(0.0, device=config.device)
                awp_adv_loss = torch.tensor(0.0, device=config.device)
                if clean_idx.numel() > 0:
                    awp_clean_loss = criterion(model(clean_data, time_steps=time_steps, use_vt=False), clean_labels)
                if adv_idx.numel() > 0 and adv_batch is not None:
                    awp_adv_loss = criterion(model(adv_batch, time_steps=time_steps, use_vt=False), adv_labels)
                awp_loss = clean_ratio * awp_clean_loss + (1.0 - clean_ratio) * awp_adv_loss
            scaler.scale(awp_loss).backward()
            awp.restore()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        if config.use_parseval and batch_idx % config.parseval_interval == 0:
            parseval_retraction(model, beta=config.parseval_beta)

        stats["loss"] += total_loss.item() * batch_size
        if clean_logits is not None:
            stats["clean_correct"] += clean_logits.argmax(dim=1).eq(clean_labels).sum().item()
            stats["clean_total"] += clean_labels.size(0)
        if adv_logits is not None:
            stats["adv_correct"] += adv_logits.argmax(dim=1).eq(adv_labels).sum().item()
            stats["adv_total"] += adv_labels.size(0)

        if hasattr(batch_iterator, "set_postfix"):
            batch_iterator.set_postfix(
                loss=f"{stats['loss'] / max(stats['clean_total'] + stats['adv_total'], 1.0):.4f}",
                clean=f"{100.0 * stats['clean_correct'] / max(stats['clean_total'], 1.0):.2f}",
                adv=f"{100.0 * stats['adv_correct'] / max(stats['adv_total'], 1.0):.2f}",
                T=time_steps,
            )

    total_samples = max(stats["clean_total"] + stats["adv_total"], 1.0)
    return {
        "loss": stats["loss"] / total_samples,
        "train_clean_acc": 100.0 * stats["clean_correct"] / max(stats["clean_total"], 1.0),
        "train_adv_acc": 100.0 * stats["adv_correct"] / max(stats["adv_total"], 1.0),
        "clean_ratio": clean_ratio,
    }


def _evaluate_selection_metrics(
    model: TraceableSpikingModel,
    val_loader,
    config: TrainingConfig,
    selection_attack_config: AttackConfig | None = None,
) -> tuple[float, float]:
    from .experiments import evaluate_policy

    max_time_steps = max(config.mixed_time_steps)
    clean_metrics = evaluate_policy(
        model,
        val_loader,
        mode="fixed",
        time_steps=max_time_steps,
        compute_metrics=False,
        exact_vt_compute=False,
        device=config.device,
        progress_desc="Validation clean",
    )
    robust_accuracy = 0.0
    if selection_attack_config is not None and config.selection_metric in {"robust", "clean_plus_robust"}:
        robust_metrics = evaluate_policy(
            model,
            val_loader,
            mode="fixed",
            time_steps=max_time_steps,
            attack_config=selection_attack_config,
            compute_metrics=False,
            exact_vt_compute=False,
            device=config.device,
            progress_desc="Validation robust",
        )
        robust_accuracy = robust_metrics["accuracy"]
    return clean_metrics["accuracy"], robust_accuracy


def _selection_score(clean_accuracy: float, robust_accuracy: float, metric: str) -> float:
    if metric == "clean":
        return clean_accuracy
    if metric == "robust":
        return robust_accuracy
    if metric == "clean_plus_robust":
        return clean_accuracy + robust_accuracy
    raise ValueError(f"Unsupported selection_metric: {metric}")


def fit_model(
    model: TraceableSpikingModel,
    train_loader,
    val_loader,
    config: TrainingConfig,
    use_awp: bool = False,
    selection_attack_config: AttackConfig | None = None,
) -> tuple[dict[str, list[float]], Path]:
    model.to(config.device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = _build_grad_scaler(config.use_amp and str(config.device).startswith("cuda"))
    awp = AdversarialWeightPerturbation(model) if use_awp else None

    output_dir = ensure_output_dir(config)
    checkpoint_path = output_dir / config.checkpoint_name
    history = {
        "loss": [],
        "train_clean_acc": [],
        "train_adv_acc": [],
        "val_clean_acc": [],
        "val_robust_acc": [],
        "sigma_mean": [],
        "lr": [],
    }

    best_state = None
    best_score = float("-inf")
    best_clean = 0.0
    best_robust = 0.0

    epoch_iterator = _training_progress(range(1, config.epochs + 1), desc="Training epochs", leave=True)
    for epoch in epoch_iterator:
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, config, epoch, awp=awp)
        scheduler.step()
        validation_ran = epoch % config.eval_interval == 0
        val_clean, val_robust = (float("nan"), float("nan"))
        if validation_ran:
            _release_memory(config.device, model=model, optimizer=optimizer)
            val_clean, val_robust = _evaluate_selection_metrics(
                model,
                val_loader,
                config,
                selection_attack_config=selection_attack_config,
            )
            _release_memory(config.device, model=model, optimizer=optimizer)

        sigma_stats = get_sigma_stats(model) or {"mean": 0.0}

        history["loss"].append(train_stats["loss"])
        history["train_clean_acc"].append(train_stats["train_clean_acc"])
        history["train_adv_acc"].append(train_stats["train_adv_acc"])
        history["val_clean_acc"].append(val_clean)
        history["val_robust_acc"].append(val_robust)
        history["sigma_mean"].append(sigma_stats["mean"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if validation_ran:
            score = _selection_score(val_clean, val_robust, config.selection_metric)
        else:
            score = None

        if validation_ran and score is not None and score >= best_score:
            best_score = score
            best_clean = val_clean
            best_robust = val_robust
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                loss=f"{train_stats['loss']:.4f}",
                clean=f"{train_stats['train_clean_acc']:.2f}",
                adv=f"{train_stats['train_adv_acc']:.2f}",
                val=f"{val_clean:.2f}" if validation_ran else "skip",
                robust=f"{val_robust:.2f}" if validation_ran else "skip",
                best=f"{best_score:.2f}" if best_score != float("-inf") else "n/a",
            )

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    torch.save(
        {
            "model_state_dict": best_state,
            "history": history,
            "training_config": asdict(config),
            "best_val_clean_acc": best_clean,
            "best_val_robust_acc": best_robust,
            "selection_metric": config.selection_metric,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state)
    return history, checkpoint_path
