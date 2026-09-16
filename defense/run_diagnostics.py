"""Per-timestep confidence/accuracy trajectories: clean vs PGD vs AutoAttack.

Eval-only. Uses an existing checkpoint; no training. Writes a tidy CSV that
make_figs.py turns into the 'why VT collapses' figure.
"""
import argparse, os, sys, json
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vt_research_suite.configs import DatasetConfig, ModelConfig, AttackConfig
from vt_research_suite.model import build_model
from vt_research_suite.stopping import build_stop_rule
from vt_research_suite.experiments import collect_temporal_diagnostics
from vt_research_suite.data import build_data_splits, build_subset_loader
from vt_research_suite.training import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="defense/ckpt/vanilla_resnet18_adv_T8.pt")
    ap.add_argument("--n", type=int, default=512, help="images to evaluate")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thresholds", default="0.60,0.50")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="defense/figs/trajectories.csv")
    ap.add_argument("--skip-autoattack", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| n =", args.n, "| batch =", args.batch)

    model_cfg = ModelConfig(
        architecture="vanilla_resnet18", num_classes=10, in_channels=3,
        max_time_steps=8, input_encoding="direct", stem_channels=64,
        stage_channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2),
        stochastic=False, learnable_sigma=False, use_dynamic_sigma=False,
        use_shortcut_lif=True, v_threshold=0.5, lif_tau=2.0,
        detach_reset=True, sigma_init_space="raw")

    seed_everything(0)
    model = build_model(model_cfg, seed=0).to(dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print("loaded:", args.ckpt)

    ds = DatasetConfig(name="cifar10", root=args.data_root, batch_size=args.batch,
                       num_workers=0, seed=0, augment=True,
                       val_fraction=0.1, calibration_fraction=0.1)
    splits = build_data_splits(ds)
    loader = build_subset_loader(splits.test_loader, args.n, 0, balanced=True,
                                 batch_size=args.batch, shuffle=False)
    print("eval images:", len(loader.dataset))

    pgd = AttackConfig(attack_name="pgd", objective="untargeted", epsilon=8/255,
                       alpha=2/255, steps=20, eot_iters=1, random_start=True,
                       exit_aware=True, force_full_trace=True, attack_time_steps=8)
    aa = AttackConfig(attack_name="autoattack", epsilon=8/255,
                      autoattack_version="standard", exit_aware=True,
                      force_full_trace=True, eot_iters=1, attack_time_steps=8)

    conditions = [("clean", None), ("PGD-20", pgd)]
    if not args.skip_autoattack:
        try:
            import autoattack  # noqa: F401
            conditions.append(("AutoAttack", aa))
        except ImportError:
            try:
                import torchattacks  # noqa: F401
                conditions.append(("AutoAttack", aa))
            except ImportError:
                print("!! neither autoattack nor torchattacks installed -- skipping AutoAttack")

    rule = build_stop_rule("max_softmax")
    rows = []
    for theta in [float(t) for t in args.thresholds.split(",")]:
        for name, acfg in conditions:
            print(f"  theta={theta}  {name} ...", flush=True)
            seed_everything(0)
            d = collect_temporal_diagnostics(
                model, loader, time_steps=8, stop_rule=rule, threshold=theta,
                attack_config=acfg, device=dev, max_batches=None)
            for i, (c, a) in enumerate(zip(d["avg_confidence_by_step"],
                                           d["avg_accuracy_by_step"]), start=1):
                rows.append(dict(threshold=theta, condition=name, step=i,
                                 confidence=c, accuracy=a))
            print(f"     hist={d['step_histogram']}")
            if dev == "cuda":
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
