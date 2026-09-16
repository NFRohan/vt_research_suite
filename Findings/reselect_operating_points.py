"""VT-neutral re-selection of the VT operating point.

The shipped criterion in experiments.select_compute_matched_threshold sorts by
`accuracy_delta` DESCENDING -- it picks the threshold that maximises
(VT accuracy - matched fixed-T accuracy), i.e. the very quantity the paper
reports. That systematically seeks wherever the fixed baseline is weakest and
can land on degenerate, far-below-full-horizon operating points.

This script re-selects using a VT-neutral, standard early-exit criterion:

    among thresholds whose CALIBRATION accuracy is within `delta` of the
    full-horizon (T=max) calibration accuracy, take the one with the LOWEST
    avg_synops (i.e. the cheapest policy that preserves accuracy).

Selection uses calibration CSVs (main_*), reporting uses test CSVs (test_*).
No GPU required -- operates entirely on existing sweep outputs.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def _fixed_full(fixed: pd.DataFrame) -> tuple[int, float]:
    t = int(fixed["time_steps"].max())
    return t, float(fixed.loc[fixed["time_steps"] == t, "accuracy"].iloc[0])


def _match_fixed(vt_synops: float, fixed: pd.DataFrame) -> tuple[int, float, float]:
    """Nearest fixed-T control by avg_synops."""
    f = fixed.dropna(subset=["avg_synops"]).copy()
    f["gap"] = (f["avg_synops"] - vt_synops).abs()
    row = f.sort_values("gap").iloc[0]
    return int(row["time_steps"]), float(row["accuracy"]), float(row["avg_synops"])


def neutral_select(calib_vt: pd.DataFrame, calib_fixed: pd.DataFrame, delta: float):
    _, full_acc = _fixed_full(calib_fixed)
    cand = calib_vt[calib_vt["accuracy"] >= full_acc - delta]
    fallback = cand.empty
    if fallback:                      # nothing preserves accuracy -> best available
        cand = calib_vt.sort_values("accuracy", ascending=False).head(1)
    row = cand.sort_values("avg_synops").iloc[0]
    return float(row["threshold"]), full_acc, fallback


def biased_select(calib_vt: pd.DataFrame, calib_fixed: pd.DataFrame):
    """Reproduces the shipped criterion: max (VT acc - matched fixed acc)."""
    best, best_delta = None, -1e9
    for _, r in calib_vt.dropna(subset=["avg_synops"]).iterrows():
        _, facc, _ = _match_fixed(float(r["avg_synops"]), calib_fixed)
        d = float(r["accuracy"]) - facc
        if d > best_delta:
            best, best_delta = float(r["threshold"]), d
    return best


def report(name: str, root: Path, delta: float = 1.0):
    cv = pd.read_csv(root / "main_vt_threshold_sweep.csv")
    cf = pd.read_csv(root / "main_fixed_t_sweep.csv")
    tv = pd.read_csv(root / "test_vt_threshold_sweep.csv")
    tf = pd.read_csv(root / "test_fixed_t_sweep.csv")
    tf = tf[tf["mode"] == "fixed"] if "mode" in tf.columns else tf
    cf = cf[cf["mode"] == "fixed"] if "mode" in cf.columns else cf

    t_full, test_full_acc = _fixed_full(tf)
    out = {"run": name, "full_T": t_full, "full_acc": test_full_acc}

    n_thr, _, n_fallback = neutral_select(cv, cf, delta)
    for label, thr, extra in [
        ("biased", biased_select(cv, cf), {}),
        ("neutral", n_thr, {"fallback": n_fallback}),
    ]:
        row = tv.loc[(tv["threshold"] - thr).abs().idxmin()]
        vt_acc, vt_syn = float(row["accuracy"]), float(row["avg_synops"])
        mt, macc, msyn = _match_fixed(vt_syn, tf)
        out[label] = {
            "theta": thr, "vt_acc": vt_acc, "vt_synops": vt_syn,
            "depth": float(row.get("avg_stopping_depth", float("nan"))),
            "matched_T": mt, "matched_acc": macc,
            "gain": vt_acc - macc,
            "below_full": test_full_acc - vt_acc,
            **extra,
        }
    return out


RUNS = {
    # --- fixed-T8 trained (three backbones) ---
    "R18 clean  T8":  Path("Findings/ResNet Clean Trained"),
    "R18 adv    T8":  Path("Findings/ResNet Adversarial Trained"),
    "R18 adv T8 RERUN": Path("Findings/ResNet Adv T8 Rerun"),
    "VGG clean  T8":  Path("Findings/VGG Clean Trained"),
    "VGG adv    T8":  Path("Findings/VGG Adversarial Trained"),
    "MBv2 clean T8":  Path("Findings/MobileNetV2 Clean Trained"),
    "MBv2 adv   T8":  Path("Findings/MobileNetV2 Adversarial Trained"),
    # --- mixT trained (ResNet only) ---
    "R18 clean  mixT": Path("Findings/Resnet Clean MixT"),
    "R18 adv    mixT": Path("Findings/Resnet ReRun"),
}

if __name__ == "__main__":
    import sys
    delta = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"VT-neutral re-selection (delta = {delta} pts below full horizon, on calibration)\n")
    hdr = f"{'run':20s} {'crit':8s} {'theta':>6s} {'depth':>6s} {'VT acc':>7s} {'mT':>3s} {'fixed':>7s} {'gain':>7s} {'vs full':>8s}"
    print(hdr); print("-"*len(hdr))
    for name, root in RUNS.items():
        if not root.exists():
            print(f"{name:20s}  (missing: {root})"); continue
        r = report(name, root, delta)
        for crit in ("biased", "neutral"):
            c = r[crit]
            flag = "  <-- DEGENERATE" if c["below_full"] > 5 else ""
            fb = " [fallback]" if c.get("fallback") else ""
            print(f"{name:20s} {crit:8s} {c['theta']:6.2f} {c['depth']:6.2f} {c['vt_acc']:7.2f} "
                  f"{c['matched_T']:3d} {c['matched_acc']:7.2f} {c['gain']:+7.2f} {c['below_full']:8.2f}{flag}{fb}")
        print()
