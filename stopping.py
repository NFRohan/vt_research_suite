from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StopRule:
    name: str

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def should_stop(self, logits: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.confidence(logits) >= threshold


class MaxSoftmaxStopRule(StopRule):
    def __init__(self) -> None:
        super().__init__(name="max_softmax")

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1).amax(dim=-1)


class MarginStopRule(StopRule):
    def __init__(self) -> None:
        super().__init__(name="margin")

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        return (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)


class EntropyStopRule(StopRule):
    def __init__(self) -> None:
        super().__init__(name="entropy")

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return (1.0 - entropy / math.log(logits.size(-1))).clamp(0.0, 1.0)


class TemperatureScaledStopRule(StopRule):
    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__(name="temperature_softmax")
        self.temperature = temperature

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1).amax(dim=-1)


class EvidentialStopRule(StopRule):
    def __init__(self) -> None:
        super().__init__(name="evidential_surrogate")

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(logits) + 1.0
        uncertainty = logits.size(-1) / alpha.sum(dim=-1)
        return (1.0 - uncertainty).clamp(0.0, 1.0)


def build_stop_rule(name: str, **kwargs) -> StopRule:
    key = name.lower()
    if key == "max_softmax":
        return MaxSoftmaxStopRule()
    if key == "margin":
        return MarginStopRule()
    if key == "entropy":
        return EntropyStopRule()
    if key in {"temperature", "temperature_softmax"}:
        return TemperatureScaledStopRule(temperature=kwargs.get("temperature", 2.0))
    if key in {"evidential", "evidential_surrogate"}:
        return EvidentialStopRule()
    raise ValueError(f"Unsupported stop rule: {name}")
