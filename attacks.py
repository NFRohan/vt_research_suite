from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import AttackConfig
from .model import TraceResult, TraceableSpikingModel
from .stopping import StopRule

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torchattacks
except Exception:
    # Kaggle and other hosted environments occasionally ship NumPy / SciPy wheels
    # that are ABI-incompatible with the current torchattacks dependency chain.
    # We only need torchattacks opportunistically; the suite has a fallback path
    # through the official autoattack package and our own PGD implementation.
    torchattacks = None

try:
    from autoattack import AutoAttack as ExternalAutoAttack
except ImportError:
    ExternalAutoAttack = None


def _untargeted_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    max_other = masked.max(dim=1).values
    return max_other - true_logits


def _build_attack_loss(
    trace: TraceResult,
    labels: torch.Tensor,
    stop_rule: StopRule | None,
    attack_config: AttackConfig,
) -> torch.Tensor:
    if attack_config.objective == "untargeted":
        return F.cross_entropy(trace.final_logits, labels)

    if attack_config.objective == "wrong_early_exit":
        if stop_rule is None:
            raise ValueError("wrong_early_exit objective requires an exit-aware stop rule.")
        losses = []
        total_steps = trace.avg_logits_per_step.size(0)
        for step in range(total_steps):
            logits = trace.avg_logits_per_step[step]
            margin = _untargeted_margin(logits, labels)
            confidence = stop_rule.confidence(logits)
            early_weight = (total_steps - step) / total_steps
            losses.append(early_weight * (margin + attack_config.wrong_exit_weight * confidence))
        return torch.stack(losses, dim=0).mean()

    if attack_config.objective == "compute_overuse":
        if stop_rule is None:
            raise ValueError("compute_overuse objective requires an exit-aware stop rule.")
        avg_logits = trace.avg_logits_per_step
        probabilities = F.softmax(avg_logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(torch.tensor(avg_logits.size(-1), device=avg_logits.device, dtype=avg_logits.dtype))
        confidences = torch.stack([stop_rule.confidence(step_logits) for step_logits in avg_logits], dim=0)
        delay_term = entropy.mean() - confidences.mean()
        preserve_term = -F.cross_entropy(trace.final_logits, labels)
        return attack_config.compute_weight * delay_term + attack_config.preserve_label_weight * preserve_term

    raise ValueError(f"Unsupported attack objective: {attack_config.objective}")


def _attack_progress(iterable, desc: str, total: int | None = None, leave: bool = False, enabled: bool = True):
    if tqdm is None or not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def pgd_attack(
    model: TraceableSpikingModel | nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_config: AttackConfig,
    time_steps: int,
    stop_rule: StopRule | None = None,
    threshold: float = 0.95,
) -> torch.Tensor:
    model.eval()
    base = images.detach()
    if attack_config.random_start:
        delta = torch.empty_like(base).uniform_(-attack_config.epsilon, attack_config.epsilon)
        delta = torch.clamp(base + delta, 0.0, 1.0) - base
    else:
        delta = torch.zeros_like(base)

    effective_time_steps = attack_config.attack_time_steps or time_steps

    show_step_progress = attack_config.steps > 1 and images.size(0) > 1
    step_iterator = _attack_progress(
        range(attack_config.steps),
        desc=f"PGD [{attack_config.objective}]",
        total=attack_config.steps,
        leave=False,
        enabled=show_step_progress,
    )
    for _ in step_iterator:
        eot_iters = max(1, attack_config.eot_iters)
        grad_accum = torch.zeros_like(delta)
        for _ in range(eot_iters):
            delta_step = delta.detach().requires_grad_(True)
            adv_images = torch.clamp(base + delta_step, 0.0, 1.0)
            trace = model.forward_trace(
                adv_images,
                time_steps=effective_time_steps,
                stop_rule=stop_rule if attack_config.exit_aware else None,
                threshold=threshold,
                force_full=attack_config.force_full_trace,
            )
            # Accumulate the EOT gradient one trace at a time. This is mathematically
            # equivalent to differentiating the mean EOT loss, but it avoids holding
            # all stochastic forward graphs in memory simultaneously.
            loss = _build_attack_loss(
                trace,
                labels,
                stop_rule if attack_config.exit_aware else None,
                attack_config,
            ) / float(eot_iters)
            grad = torch.autograd.grad(loss, delta_step, retain_graph=False, create_graph=False)[0]
            grad_accum = grad_accum + grad.detach()

            if hasattr(model, "reset"):
                model.reset()

            del delta_step, adv_images, trace, loss, grad

        delta = delta.detach() + attack_config.alpha * grad_accum.sign()
        delta = delta.clamp(-attack_config.epsilon, attack_config.epsilon)
        delta = torch.clamp(base + delta, 0.0, 1.0) - base

    return torch.clamp(base + delta.detach(), 0.0, 1.0)


class _ForwardTraceWrapper(nn.Module):
    def __init__(
        self,
        model: TraceableSpikingModel | nn.Module,
        time_steps: int,
        stop_rule: StopRule | None,
        threshold: float,
        exit_aware: bool,
        force_full_trace: bool,
        eot_iters: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.time_steps = time_steps
        self.stop_rule = stop_rule if exit_aware else None
        self.threshold = threshold
        self.force_full_trace = force_full_trace
        self.eot_iters = max(1, eot_iters)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = None
        for _ in range(self.eot_iters):
            trace = self.model.forward_trace(
                inputs,
                time_steps=self.time_steps,
                stop_rule=self.stop_rule,
                threshold=self.threshold,
                force_full=self.force_full_trace,
            )
            if logits is None:
                logits = trace.final_logits
            else:
                logits = logits + trace.final_logits
        assert logits is not None
        return logits / float(self.eot_iters)


def _resolve_autoattack_version(attack_config: AttackConfig) -> str:
    if attack_config.autoattack_version is not None:
        return attack_config.autoattack_version
    return "rand" if attack_config.eot_iters > 1 else "standard"


def autoattack_attack(
    model: TraceableSpikingModel | nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_config: AttackConfig,
    time_steps: int,
    stop_rule: StopRule | None = None,
    threshold: float = 0.95,
) -> torch.Tensor:
    wrapper = _ForwardTraceWrapper(
        model=model,
        time_steps=attack_config.attack_time_steps or time_steps,
        stop_rule=stop_rule,
        threshold=threshold,
        exit_aware=attack_config.exit_aware,
        force_full_trace=attack_config.force_full_trace,
        eot_iters=attack_config.eot_iters,
    )
    wrapper.eval()
    version = _resolve_autoattack_version(attack_config)

    if torchattacks is not None:
        attack = torchattacks.AutoAttack(
            wrapper,
            norm="Linf",
            eps=attack_config.epsilon,
            version=version,
            n_classes=wrapper(images[:1]).size(1),
            seed=attack_config.autoattack_seed,
            verbose=attack_config.autoattack_verbose,
        )
        return attack(images, labels)

    if ExternalAutoAttack is not None:
        adversary = ExternalAutoAttack(
            wrapper,
            norm="Linf",
            eps=attack_config.epsilon,
            version=version,
            seed=attack_config.autoattack_seed,
            verbose=attack_config.autoattack_verbose,
        )
        try:
            return adversary.run_standard_evaluation(images, labels, bs=images.size(0), return_labels=False)
        except TypeError:
            return adversary.run_standard_evaluation(images, labels, bs=images.size(0))

    raise ImportError(
        "AutoAttack support requires either `torchattacks` or `autoattack` to be installed."
    )
