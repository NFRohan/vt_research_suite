from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class ComputeMetrics:
    avg_synops: float
    avg_dense_ops: float
    avg_spike_count: float
    avg_spike_density: float
    activation_sparsity: float


class ComputeMonitor:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.reset()

    def reset(self) -> None:
        self.synops = 0.0
        self.dense_ops = 0.0
        self.spike_count = 0.0
        self.spike_elements = 0.0
        self.zero_activations = 0.0
        self.total_activations = 0.0

    def _register(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif getattr(module, "is_spiking_neuron", False):
                self.handles.append(module.register_forward_hook(self._spike_hook))

    def _remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "ComputeMonitor":
        self.reset()
        self._register()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._remove()

    def _conv_hook(self, module: nn.Conv2d, inputs, output) -> None:
        x = inputs[0].detach()
        active_fraction = (x != 0).float().mean().item()
        output_elements = output.numel()
        kernel_mul = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        dense_ops = output_elements * kernel_mul
        self.dense_ops += dense_ops
        self.synops += dense_ops * active_fraction
        self.zero_activations += (x == 0).sum().item()
        self.total_activations += x.numel()

    def _linear_hook(self, module: nn.Linear, inputs, output) -> None:
        x = inputs[0].detach()
        active_fraction = (x != 0).float().mean().item()
        dense_ops = x.size(0) * module.in_features * module.out_features
        self.dense_ops += dense_ops
        self.synops += dense_ops * active_fraction
        self.zero_activations += (x == 0).sum().item()
        self.total_activations += x.numel()

    def _spike_hook(self, module: nn.Module, inputs, output) -> None:
        spikes = output.detach()
        self.spike_count += spikes.sum().item()
        self.spike_elements += spikes.numel()

    def summary(self, batch_size: int) -> ComputeMetrics:
        denom = max(batch_size, 1)
        return ComputeMetrics(
            avg_synops=self.synops / denom,
            avg_dense_ops=self.dense_ops / denom,
            avg_spike_count=self.spike_count / denom,
            avg_spike_density=self.spike_count / max(self.spike_elements, 1.0),
            activation_sparsity=self.zero_activations / max(self.total_activations, 1.0),
        )
