# VT Research Suite

This folder turns the monolithic `dyn-randhet-vt-snn V4.ipynb` workflow into a reusable paper pipeline.

Main pieces:

- `model.py`: a common traceable spiking interface plus ResNet, VGG, and WRN backbones.
- `training.py`: adversarial training, mixed-time-step training, Parseval support, and validation-based checkpoint selection.
- `attacks.py`: fixed-T PGD, custom exit-aware wrong-early-exit / compute-overuse objectives, plus an AutoAttack hook with EoT-aware wrapper support for final robustness tables.
- `experiments.py`: fixed-T control sweeps, stopping-rule comparisons, exact VT compute evaluation, temporal diagnostics, and multi-seed summaries.
- `data.py`: CIFAR-10/100 loaders plus explicit train / val / calib / test splits and optional MNIST-DVS / DVS128 Gesture support through `tonic`.
- `vt_research_direction.ipynb`: the clean notebook entry point for the paper experiments.

Intended workflow:

1. Build train / val / calib / test splits.
2. Seed before model construction, then train or load a checkpoint.
3. Calibrate thresholds and stopping rules on the calibration split.
4. Report final fixed-T / VT / attack results on the test split.
5. Repeat the same workflow across ResNet, VGG, and WRN backbones.

Paper-facing notes:

- Use `exact_vt_compute=True` for main VT compute tables. This triggers `compute_mode="sample_exact"` so `avg_executed_steps` and `avg_synops` reflect sample-exact VT execution rather than a batched proxy.
- Use `exact_vt_compute=False` only for quick screening or large exploratory attack sweeps, and report those numbers as batched compute proxies rather than literal saved execution.
- Match VT against fixed-`T` controls with `match_metric="avg_synops"` whenever compute metrics are enabled. Falling back to stopping depth is fine for debugging, but SynOps is the paper-safe control.
- `ComputeMonitor` reports SynOps / dense-op / spike-count proxies for algorithmic comparison, not hardware energy measurements.
- `DatasetConfig` now separates `train_subset`, `val_subset`, `calib_subset`, and `test_subset` so fast local debugging does not silently shrink calibration unless you want it to.
- `ModelConfig.sigma_init_space` defaults to `"raw"` to match the original V4 notebook semantics for learnable sigma initialization.
- AutoAttack defaults to `version="rand"` whenever `eot_iters > 1`, and the wrapper averages logits over repeated stochastic traces while honoring `force_full_trace`.
- The `temperature_softmax` and `evidential_surrogate` stop rules are intentionally named as surrogates, not as full post-hoc calibration pipelines.
