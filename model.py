from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import ModelConfig
from .stopping import StopRule


def _softplus_inverse(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class SurrogateBPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha
        return (inputs >= 0).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (inputs,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha / (2 * (1 + (alpha * inputs).abs()) ** 2)
        return grad_input, None


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            return (inputs - self.mean) / self.std
        if inputs.ndim == 5:
            batch, steps, channels, height, width = inputs.shape
            flat = inputs.view(batch * steps, channels, height, width)
            flat = (flat - self.mean) / self.std
            return flat.view(batch, steps, channels, height, width)
        raise ValueError(f"Unexpected input shape: {tuple(inputs.shape)}")


class SENoiseScaler(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, base_scale: float = 1.0) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.base_scale = base_scale
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )
        self.scale_factor = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        squeezed = self.squeeze(inputs).view(batch, channels)
        excited = self.excite(squeezed).view(batch, channels, 1, 1)
        scale = self.base_scale + F.softplus(self.scale_factor) * excited
        return scale.clamp(0.5, 2.5)


class AdaptiveRHLIFNode(nn.Module):
    is_spiking_neuron = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.v_threshold = config.v_threshold
        self.stochastic = config.stochastic
        self.learnable_sigma = config.learnable_sigma and self.stochastic
        if self.learnable_sigma:
            if config.sigma_init_space == "raw":
                raw_sigma = config.sigma_init
            elif config.sigma_init_space == "sigma":
                raw_sigma = _softplus_inverse(config.sigma_init)
            else:
                raise ValueError(f"Unsupported sigma_init_space: {config.sigma_init_space}")
        else:
            raw_sigma = config.fixed_sigma_raw
        if self.learnable_sigma:
            self.sigma_raw = nn.Parameter(torch.tensor(raw_sigma, dtype=torch.float32))
        else:
            self.register_buffer("sigma_raw", torch.tensor(raw_sigma, dtype=torch.float32))
        self.v: torch.Tensor | None = None

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw)

    def forward(self, inputs: torch.Tensor, sigma_scaler: torch.Tensor | None = None) -> torch.Tensor:
        if self.v is None or self.v.shape != inputs.shape:
            self.v = torch.zeros_like(inputs)
        if self.stochastic:
            final_sigma = self.sigma if sigma_scaler is None else self.sigma * sigma_scaler
            noise = torch.randn_like(inputs) * final_sigma
            gate = torch.sigmoid(noise)
        else:
            # Deterministic control: same membrane update structure, but without
            # stochastic noise injection. Since the stochastic gate has zero-mean
            # logits, its expected value is 0.5; we use that fixed leak factor here.
            gate = torch.full_like(inputs, 0.5)
        self.v = gate * self.v + (1 - gate) * inputs
        spike = SurrogateBPFunction.apply(self.v - self.v_threshold)
        self.v = self.v * (1 - spike)
        return spike

    def reset(self) -> None:
        self.v = None


class VanillaLIFNode(nn.Module):
    is_spiking_neuron = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.v_threshold = config.v_threshold
        self.tau = config.lif_tau
        self.detach_reset = config.detach_reset
        self.v: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != inputs.shape:
            self.v = torch.zeros_like(inputs)

        # Standard leaky integrate-and-fire update used for the vanilla deterministic path.
        self.v = self.v + (inputs - self.v) / self.tau
        spike = SurrogateBPFunction.apply(self.v - self.v_threshold)
        reset_spike = spike.detach() if self.detach_reset else spike
        self.v = self.v * (1 - reset_spike)
        return spike

    def reset(self) -> None:
        self.v = None


class AdaptiveConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaler = SENoiseScaler(out_channels) if config.use_dynamic_sigma else None
        self.lif = AdaptiveRHLIFNode(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.bn(self.conv(inputs))
        return self.lif(out, sigma_scaler=self.scaler(out) if self.scaler is not None else None)


class AdaptiveLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = AdaptiveRHLIFNode(config)
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.linear(inputs)
        out = self.lif(out)
        return self.dropout(out)


class VanillaConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = VanillaLIFNode(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lif(self.bn(self.conv(inputs)))


class VanillaLinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: ModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = VanillaLIFNode(config)
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.linear(inputs)
        out = self.lif(out)
        return self.dropout(out)


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, config: ModelConfig) -> None:
        super().__init__()
        self.has_shortcut = stride != 1 or in_planes != planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.attn1 = SENoiseScaler(planes) if config.use_dynamic_sigma else None
        self.lif1 = AdaptiveRHLIFNode(config)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.attn2 = SENoiseScaler(planes) if config.use_dynamic_sigma else None
        self.lif2 = AdaptiveRHLIFNode(config)

        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(planes)
            self.shortcut_attn = SENoiseScaler(planes) if config.use_dynamic_sigma else None
            self.shortcut_lif = AdaptiveRHLIFNode(config) if config.use_shortcut_lif else None
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
            self.shortcut_attn = None
            self.shortcut_lif = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(inputs))
        out = self.lif1(out, sigma_scaler=self.attn1(out) if self.attn1 is not None else None)

        out = self.bn2(self.conv2(out))
        out = self.lif2(out, sigma_scaler=self.attn2(out) if self.attn2 is not None else None)

        if self.has_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
            if self.shortcut_lif is not None:
                shortcut = self.shortcut_lif(
                    shortcut,
                    sigma_scaler=self.shortcut_attn(shortcut) if self.shortcut_attn is not None else None,
                )
        else:
            shortcut = inputs
        return out + shortcut


class AdaptiveWideResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_planes)
        self.scaler1 = SENoiseScaler(in_planes) if config.use_dynamic_sigma else None
        self.lif1 = AdaptiveRHLIFNode(config)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.norm2 = nn.BatchNorm2d(planes)
        self.scaler2 = SENoiseScaler(planes) if config.use_dynamic_sigma else None
        self.lif2 = AdaptiveRHLIFNode(config)
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.norm1(inputs)
        out = self.lif1(out, sigma_scaler=self.scaler1(out) if self.scaler1 is not None else None)
        shortcut = self.shortcut(out) if self.shortcut is not None else inputs
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.lif2(out, sigma_scaler=self.scaler2(out) if self.scaler2 is not None else None)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + shortcut


class VanillaResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int, config: ModelConfig) -> None:
        super().__init__()
        self.has_shortcut = stride != 1 or in_planes != planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = VanillaLIFNode(config)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = VanillaLIFNode(config)

        if self.has_shortcut:
            self.shortcut_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(planes)
            self.shortcut_lif = VanillaLIFNode(config) if config.use_shortcut_lif else None
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None
            self.shortcut_lif = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.lif1(self.bn1(self.conv1(inputs)))
        out = self.lif2(self.bn2(self.conv2(out)))

        if self.has_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
            if self.shortcut_lif is not None:
                shortcut = self.shortcut_lif(shortcut)
        else:
            shortcut = inputs

        return out + shortcut


class VanillaInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int, config: ModelConfig) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("MobileNetV2 inverted residual stride must be 1 or 2.")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        layers: list[nn.Module] = []

        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    VanillaLIFNode(config),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                VanillaLIFNode(config),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.block(inputs)
        if self.use_residual:
            out = out + inputs
        return out


@dataclass
class TraceResult:
    final_logits: torch.Tensor
    steps_taken: torch.Tensor
    avg_logits_per_step: torch.Tensor
    raw_logits_per_step: torch.Tensor
    executed_steps: int
    confidence_per_step: torch.Tensor | None = None
    stop_mask_per_step: torch.Tensor | None = None


class TraceableSpikingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.normalize = (
            NormalizeByChannelMeanStd(config.normalization_mean, config.normalization_std)
            if config.normalization_mean is not None and config.normalization_std is not None
            else None
        )

    def reset(self) -> None:
        for module in self.modules():
            if hasattr(module, "reset") and module is not self:
                module.reset()

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalize(inputs) if self.normalize is not None else inputs

    def _step_input(self, inputs: torch.Tensor, step: int) -> torch.Tensor:
        if inputs.ndim == 5:
            return inputs[:, step]
        if self.config.input_encoding.lower() == "direct":
            return inputs
        raise ValueError(f"Unsupported static-input encoding: {self.config.input_encoding}")

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_trace(
        self,
        inputs: torch.Tensor,
        time_steps: int | None = None,
        stop_rule: StopRule | None = None,
        threshold: float = 0.95,
        force_full: bool = False,
    ) -> TraceResult:
        prepared = self._prepare_inputs(inputs)
        max_steps = time_steps or self.config.max_time_steps
        if prepared.ndim == 5:
            max_steps = min(max_steps, prepared.size(1))

        self.reset()
        batch_size = prepared.size(0)
        device = prepared.device
        accumulated = torch.zeros(batch_size, self.config.num_classes, device=device, dtype=prepared.dtype)
        raw_logits = []
        avg_logits = []
        confidences = []
        stop_masks = []
        final_logits = torch.zeros(batch_size, self.config.num_classes, device=device, dtype=prepared.dtype)
        steps_taken = torch.full((batch_size,), max_steps, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for step in range(max_steps):
            logits = self._forward_one_step(self._step_input(prepared, step))
            accumulated = accumulated + logits
            averaged = accumulated / float(step + 1)
            raw_logits.append(logits)
            avg_logits.append(averaged)

            if stop_rule is not None:
                confidence = stop_rule.confidence(averaged)
                should_stop = stop_rule.should_stop(averaged, threshold)
                newly_finished = should_stop & (~finished)
                if newly_finished.any():
                    final_logits[newly_finished] = averaged[newly_finished]
                    steps_taken[newly_finished] = step + 1
                finished = finished | newly_finished
                confidences.append(confidence)
                stop_masks.append(should_stop)
                if finished.all() and not force_full:
                    break

        raw_logits_tensor = torch.stack(raw_logits, dim=0)
        avg_logits_tensor = torch.stack(avg_logits, dim=0)
        executed_steps = avg_logits_tensor.size(0)

        if stop_rule is None:
            final = avg_logits_tensor[-1]
            return TraceResult(
                final_logits=final,
                steps_taken=torch.full((batch_size,), executed_steps, device=device, dtype=torch.long),
                avg_logits_per_step=avg_logits_tensor,
                raw_logits_per_step=raw_logits_tensor,
                executed_steps=executed_steps,
            )

        if (~finished).any():
            final_logits[~finished] = avg_logits_tensor[-1][~finished]

        return TraceResult(
            final_logits=final_logits,
            steps_taken=steps_taken,
            avg_logits_per_step=avg_logits_tensor,
            raw_logits_per_step=raw_logits_tensor,
            executed_steps=executed_steps,
            confidence_per_step=torch.stack(confidences, dim=0),
            stop_mask_per_step=torch.stack(stop_masks, dim=0),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        time_steps: int | None = None,
        use_vt: bool = False,
        stop_rule: StopRule | None = None,
        threshold: float = 0.95,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        trace = self.forward_trace(
            inputs,
            time_steps=time_steps,
            stop_rule=stop_rule if use_vt else None,
            threshold=threshold,
        )
        if use_vt:
            return trace.final_logits, trace.steps_taken
        return trace.final_logits


class VTResNetSNN(TraceableSpikingModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.conv1 = nn.Conv2d(
            config.in_channels,
            config.stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.stem_channels)
        self.lif1 = AdaptiveRHLIFNode(config)
        c1, c2, c3 = config.stage_channels
        b1, b2, b3 = config.blocks_per_stage
        self.layer1 = self._make_layer(config.stem_channels, c1, b1, 1)
        self.layer2 = self._make_layer(c1, c2, b2, 2)
        self.layer3 = self._make_layer(c2, c3, b3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, config.num_classes)
        self._initialize_weights()

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        current = in_planes
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(AdaptiveResidualBlock(current, planes, current_stride, self.config))
            current = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.lif1(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class VTVanillaResNet18SNN(TraceableSpikingModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if len(config.stage_channels) != 4 or len(config.blocks_per_stage) != 4:
            raise ValueError(
                "vanilla_resnet18 expects four stages. Use stage_channels=(64, 128, 256, 512) "
                "and blocks_per_stage=(2, 2, 2, 2)."
            )

        self.conv1 = nn.Conv2d(
            config.in_channels,
            config.stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.stem_channels)
        self.lif1 = VanillaLIFNode(config)

        c1, c2, c3, c4 = config.stage_channels
        b1, b2, b3, b4 = config.blocks_per_stage
        self.layer1 = self._make_layer(config.stem_channels, c1, b1, 1)
        self.layer2 = self._make_layer(c1, c2, b2, 2)
        self.layer3 = self._make_layer(c2, c3, b3, 2)
        self.layer4 = self._make_layer(c3, c4, b4, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, config.num_classes)
        self._initialize_weights()

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        current = in_planes
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(VanillaResidualBlock(current, planes, current_stride, self.config))
            current = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.lif1(self.bn1(self.conv1(inputs)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class VTVGGSNN(TraceableSpikingModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        channels = config.stage_channels
        blocks = config.blocks_per_stage
        if len(channels) != len(blocks):
            raise ValueError("VGG stage_channels and blocks_per_stage must have the same length.")
        if len(channels) == 0:
            raise ValueError("VGG requires at least one convolutional stage.")
        current_channels = config.in_channels
        stages = []
        for stage_channels, stage_blocks in zip(channels, blocks):
            layers = []
            for _ in range(stage_blocks):
                layers.append(AdaptiveConvBlock(current_channels, stage_channels, config, stride=1))
                current_channels = stage_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier_layers: list[nn.Module] = []
        current_features = current_channels
        for hidden_dim in config.vgg_classifier_dims:
            classifier_layers.append(AdaptiveLinearBlock(current_features, hidden_dim, config))
            current_features = hidden_dim
        classifier_layers.append(nn.Linear(current_features, config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for stage in self.stages:
            out = stage(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class VTVanillaVGGSNN(TraceableSpikingModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        channels = config.stage_channels
        blocks = config.blocks_per_stage
        if len(channels) != len(blocks):
            raise ValueError("vanilla_vgg16 stage_channels and blocks_per_stage must have the same length.")
        if len(channels) != 5 or tuple(blocks) != (2, 2, 3, 3, 3):
            raise ValueError(
                "vanilla_vgg16 expects five stages with blocks_per_stage=(2, 2, 3, 3, 3), "
                "e.g. stage_channels=(64, 128, 256, 512, 512)."
            )

        current_channels = config.in_channels
        stages = []
        for stage_channels, stage_blocks in zip(channels, blocks):
            layers = []
            for _ in range(stage_blocks):
                layers.append(VanillaConvBlock(current_channels, stage_channels, config, stride=1))
                current_channels = stage_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier_layers: list[nn.Module] = []
        current_features = current_channels
        for hidden_dim in config.vgg_classifier_dims:
            classifier_layers.append(VanillaLinearBlock(current_features, hidden_dim, config))
            current_features = hidden_dim
        classifier_layers.append(nn.Linear(current_features, config.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for stage in self.stages:
            out = stage(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class VTVanillaMobileNetV2SNN(TraceableSpikingModel):
    # MobileNetV2-1.0 block schedule from Sandler et al., adapted for CIFAR.
    # The CIFAR convention keeps the width/depth but avoids the early ImageNet
    # downsampling by using stride 1 in the stem and the 24-channel stage.
    INVERTED_RESIDUAL_SETTINGS: tuple[tuple[int, int, int, int], ...] = (
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        stem_channels = config.stem_channels
        if stem_channels <= 0:
            raise ValueError("vanilla_mobilenetv2 requires a positive stem_channels value.")

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            VanillaLIFNode(config),
        )

        layers: list[nn.Module] = []
        input_channels = stem_channels
        for expand_ratio, output_channels, num_blocks, stride in self.INVERTED_RESIDUAL_SETTINGS:
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                layers.append(
                    VanillaInvertedResidualBlock(
                        input_channels,
                        output_channels,
                        block_stride,
                        expand_ratio,
                        config,
                    )
                )
                input_channels = output_channels
        self.features = nn.Sequential(*layers)

        last_channels = 1280
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels, last_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channels),
            VanillaLIFNode(config),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(last_channels, config.num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.stem(inputs)
        out = self.features(out)
        out = self.final_conv(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)


class VTWideResNetSNN(TraceableSpikingModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        base_width = 16
        if (config.wrn_depth - 4) % 6 != 0 or config.wrn_depth < 10:
            raise ValueError("WRN depth must follow 6n+4 with n >= 1, e.g. 16, 22, 28, 40.")
        depth_blocks = (config.wrn_depth - 4) // 6
        widths = (
            base_width * config.widen_factor,
            2 * base_width * config.widen_factor,
            4 * base_width * config.widen_factor,
        )
        # WRN stage widths and residual block counts are defined by (depth, widen_factor).
        # The generic stage_channels / blocks_per_stage fields are kept for config display
        # and notebook presets, but this backbone follows the canonical WRN construction.

        self.conv1 = nn.Conv2d(config.in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(base_width, widths[0], depth_blocks, stride=1)
        self.layer2 = self._make_layer(widths[0], widths[1], depth_blocks, stride=2)
        self.layer3 = self._make_layer(widths[1], widths[2], depth_blocks, stride=2)
        self.bn = nn.BatchNorm2d(widths[2])
        self.scaler = SENoiseScaler(widths[2]) if config.use_dynamic_sigma else None
        self.lif = AdaptiveRHLIFNode(config)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2], config.num_classes)
        self._initialize_weights()

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        current = in_planes
        for current_stride in [stride] + [1] * (blocks - 1):
            layers.append(AdaptiveWideResidualBlock(current, planes, current_stride, self.config))
            current = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _forward_one_step(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.lif(out, sigma_scaler=self.scaler(out) if self.scaler is not None else None)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_model(config: ModelConfig, seed: int | None = None) -> TraceableSpikingModel:
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    architecture = config.architecture.lower()
    if architecture == "resnet":
        return VTResNetSNN(config)
    if architecture == "vanilla_resnet18":
        return VTVanillaResNet18SNN(config)
    if architecture == "vanilla_vgg16":
        return VTVanillaVGGSNN(config)
    if architecture == "vanilla_mobilenetv2":
        return VTVanillaMobileNetV2SNN(config)
    if architecture == "vgg":
        return VTVGGSNN(config)
    if architecture == "wrn":
        return VTWideResNetSNN(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")


VTDynRandHetSNN = VTResNetSNN


def get_sigma_stats(model: nn.Module) -> dict[str, float] | None:
    sigmas = []
    for module in model.modules():
        if isinstance(module, AdaptiveRHLIFNode) and module.learnable_sigma:
            sigmas.append(module.sigma.item())
    if not sigmas:
        return None
    tensor = torch.tensor(sigmas, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "count": len(sigmas),
    }
