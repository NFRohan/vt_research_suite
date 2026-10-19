"""Generate every data figure of the thesis from facts.json and the raw CSVs.

Run from the thesis root:  python analysis/make_figures.py
Also writes analysis/pooled.json with the pooled multiseed statistics quoted in Chapter 4.
"""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
import os
PREVIEW = Path(os.environ["FIG_PREVIEW"]) if os.environ.get("FIG_PREVIEW") else None
FINDINGS = HERE.parents[2] / "Findings"
F = json.loads((ROOT / "facts.json").read_text())
C = F["configs"]

CFIX, CVT, CATK, CGRY = "#36649E", "#CE7B1A", "#B22D2D", "#7A7A7A"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "savefig.bbox": "tight", "figure.dpi": 150,
})

LABEL = {
    "r18_clean_T8": "ResNet-18 / clean / T8", "r18_adv_T8": "ResNet-18 / adv / T8 (run 1)",
    "r18_adv_T8_rep": "ResNet-18 / adv / T8 (run 2)", "r18_clean_mixT": "ResNet-18 / clean / mixed-T",
    "r18_adv_mixT": "ResNet-18 / adv / mixed-T", "vgg_clean_T8": "VGG16 / clean / T8",
    "vgg_adv_T8": "VGG16 / adv / T8", "mbv2_clean_T8": "MobileNetV2 / clean / T8",
    "mbv2_adv_T8": "MobileNetV2 / adv / T8",
}
SHORT = {
    "r18_clean_T8": "R18 clean", "r18_adv_T8": "R18 adv (1)", "r18_adv_T8_rep": "R18 adv (2)",
    "r18_clean_mixT": "R18 clean mixT", "r18_adv_mixT": "R18 adv mixT", "vgg_clean_T8": "VGG clean",
    "vgg_adv_T8": "VGG adv", "mbv2_clean_T8": "MBv2 clean", "mbv2_adv_T8": "MBv2 adv",
}
ORDER = list(LABEL)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    if PREVIEW:
        fig.savefig(PREVIEW / f"{name}.png", dpi=110)
    plt.close(fig)


def mops(x):
    return np.asarray(x) / 1e6


# ---------------------------------------------------------------- fixed-T sweeps
def fig_fixed_sweeps():
    groups = [("ResNet-18", ["r18_clean_T8", "r18_adv_T8", "r18_adv_T8_rep", "r18_clean_mixT", "r18_adv_mixT"]),
              ("VGG16", ["vgg_clean_T8", "vgg_adv_T8"]), ("MobileNetV2", ["mbv2_clean_T8", "mbv2_adv_T8"])]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    for ax, (title, keys) in zip(axs, groups):
        for k in keys:
            s = C[k]["fixed_sweep"]
            ls = "--" if "mixT" in k else "-"
            col = CVT if "clean" in k else CFIX
            if k == "r18_adv_T8":
                col = "#8FA9CC"
            ax.plot([r["T"] for r in s], [r["acc"] for r in s], ls, marker="o", ms=3, color=col,
                    label=SHORT[k])
        ax.set_title(title)
        ax.set_xlabel("Timesteps $T$")
        ax.set_xticks(range(1, 9))
        ax.legend(loc="lower right", fontsize=6.5)
    axs[0].set_ylabel("Clean test accuracy (%)")
    save(fig, "fixed_sweeps")


# ---------------------------------------------------------------- frontier small multiples
def frontier_panel(ax, k, annotate=True):
    c = C[k]
    fx, vt = c["fixed_sweep"], c["vt_sweep"]
    ax.plot(mops([r["synops"] for r in fx]), [r["acc"] for r in fx], "-o", ms=3, color=CFIX, label="Fixed $T$")
    ax.plot(mops([r["synops"] for r in vt]), [r["acc"] for r in vt], "-s", ms=3, color=CVT, label="VT sweep")
    op = c["selection"]["neutral_1.0"]
    ax.plot(mops(op["vt_synops"]), op["vt_acc"], "*", ms=10, color=CATK, zorder=5, label="Neutral op. point")
    ax.plot(mops(op["matched_synops"]), op["matched_acc"], "D", ms=5, mfc="white", mec=CFIX, zorder=5,
            label="Matched control")
    if annotate:
        ax.annotate(f"{op['gain']:+.2f}", (mops(op["vt_synops"]), op["vt_acc"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=7, color=CATK)
    lo = min(min(r["acc"] for r in vt), [r["acc"] for r in fx][2]) - 3
    ax.set_ylim(max(lo, 0), max(r["acc"] for r in fx) + 2)


def fig_frontiers():
    fig, axs = plt.subplots(3, 3, figsize=(6.4, 6.4))
    for ax, k in zip(axs.flat, ORDER):
        frontier_panel(ax, k)
        ax.set_title(LABEL[k], fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("Mean SynOps / image ($10^6$)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "frontiers_all")


def fig_frontier_main():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    frontier_panel(ax, "r18_clean_T8", annotate=False)
    c = C["r18_clean_T8"]
    for r in c["vt_sweep"]:
        ax.annotate(f"{r['theta']:.2f}", (mops(r["synops"]), r["acc"]), xytext=(-20, 1),
                    textcoords="offset points", fontsize=6, color=CVT)
    for r in c["fixed_sweep"][1:]:
        ax.annotate(f"T={r['T']}", (mops(r["synops"]), r["acc"]), xytext=(-4, 5),
                    textcoords="offset points", fontsize=6, color=CFIX)
    op = c["selection"]["neutral_1.0"]
    ax.annotate("", xy=(mops(op["vt_synops"]), op["vt_acc"]), xytext=(mops(op["matched_synops"]), op["matched_acc"]),
                arrowprops=dict(arrowstyle="->", color=CATK, lw=1))
    ax.text(150, 86.6, f"+{op['gain']:.2f} pts over $T$=3 at equal compute\n"
            f"({100 * op['synops_saving']:.0f}% fewer SynOps than $T$=8)",
            color=CATK, fontsize=7.5)
    ax.set_ylim(82, 92.2)
    ax.set_xlabel("Mean SynOps per image ($10^6$)")
    ax.set_ylabel("Clean test accuracy (%)")
    ax.legend(loc="lower right")
    save(fig, "frontier_r18_clean")


# ---------------------------------------------------------------- selection bias & delta
def fig_selection():
    x = np.arange(len(ORDER))
    neu = [C[k]["selection"]["neutral_1.0"]["gain"] for k in ORDER]
    bia = [C[k]["selection"]["biased"]["gain"] for k in ORDER]
    bfn = [C[k]["selection"]["neutral_1.0"]["below_full"] for k in ORDER]
    bfb = [C[k]["selection"]["biased"]["below_full"] for k in ORDER]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    w = 0.38
    axs[0].bar(x - w / 2, bia, w, color=CGRY, label="Max-gain (biased)")
    axs[0].bar(x + w / 2, neu, w, color=CVT, label="VT-neutral ($\\delta$=1)")
    axs[0].set_ylabel("Gain vs matched (pts)")
    axs[1].bar(x - w / 2, bfb, w, color=CGRY)
    axs[1].bar(x + w / 2, bfn, w, color=CVT)
    axs[1].set_ylabel("Below full horizon (pts)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=55, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
    axs[0].legend()
    fig.tight_layout()
    save(fig, "selection_bias")


def fig_delta():
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    x = np.arange(len(ORDER))
    w = 0.26
    for i, (d, col) in enumerate([("0.5", "#E9B97A"), ("1.0", CVT), ("2.0", "#8A4F0E")]):
        ax.bar(x + (i - 1) * w, [C[k]["selection"][f"neutral_{d}"]["gain"] for k in ORDER], w, color=col,
               label=f"$\\delta$ = {d}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Gain vs matched (pts)")
    ax.legend(ncol=3)
    save(fig, "delta_sensitivity")


# ---------------------------------------------------------------- exit histograms
def test_hist(k, theta):
    folder = FINDINGS / {v: kk for kk, v in []}.get(k, "") if False else None
    root = FINDINGS / F["configs"][k]["folder"]
    d = pd.read_csv(root / "test_vt_threshold_sweep.csv")
    d = d[(d["mode"] == "vt") & (d["stop_rule"] == "max_softmax") & (np.isclose(d["threshold"], theta))]
    h = ast.literal_eval(d.iloc[0]["step_histogram"])
    n = sum(h.values())
    return [100 * h.get(t, 0) / n for t in range(1, 9)]


def fig_exit_hist():
    keys = ["r18_clean_T8", "r18_adv_T8_rep", "vgg_clean_T8", "vgg_adv_T8", "mbv2_clean_T8", "mbv2_adv_T8"]
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 3.6), sharex=True, sharey=True)
    for ax, k in zip(axs.T.flat, keys):
        th = C[k]["selection"]["neutral_1.0"]["theta"]
        hh = test_hist(k, th)
        ax.bar(range(1, 9), hh, color=CVT if "clean" in k else CFIX)
        ax.set_title(f"{LABEL[k].replace(' / T8', '')}\n$\\theta$={th:.2f}, mean depth {C[k]['selection']['neutral_1.0']['depth']:.2f}",
                     fontsize=7.5)
        ax.set_xticks(range(1, 9))
    for ax in axs[-1]:
        ax.set_xlabel("Exit timestep")
    for ax in axs[:, 0]:
        ax.set_ylabel("% of test images")
    fig.tight_layout()
    save(fig, "exit_histograms")


# ---------------------------------------------------------------- stopping rules
def fig_stopping():
    keys = ["r18_clean_T8", "r18_adv_T8_rep", "vgg_clean_T8", "vgg_adv_T8", "r18_adv_mixT", "mbv2_adv_T8"]
    mk = {"max_softmax": "o", "margin": "s", "entropy": "^", "temperature_softmax": "D",
          "evidential_surrogate": "v", "fixed_T8_baseline": "*"}
    cols = {"max_softmax": CVT, "margin": "#2A9D8F", "entropy": "#8E6BB5", "temperature_softmax": "#D1495B",
            "evidential_surrogate": CGRY, "fixed_T8_baseline": "k"}
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 3.9))
    for ax, k in zip(axs.flat, keys):
        for r in C[k]["stopping_rules"]:
            ax.plot(r["depth"], r["acc"], mk[r["rule"]], color=cols[r["rule"]], ms=6 if r["rule"] != "fixed_T8_baseline" else 9,
                    label=r["rule"].replace("_", " ").replace("fixed T8 baseline", "fixed T=8"))
        fs = C[k]["fixed_sweep"]
        ax.plot([r["T"] for r in fs][2:], [r["acc"] for r in fs][2:], "-", color=CFIX, lw=0.8, alpha=0.6,
                label="fixed-$T$ curve")
        ax.set_title(LABEL[k], fontsize=7.5)
    for ax in axs[-1]:
        ax.set_xlabel("Mean exit depth")
    for ax in axs[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "stopping_rules")


# ---------------------------------------------------------------- mixed-T comparison
def fig_mixT():
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, (a, b, t) in zip(axs, [("r18_clean_T8", "r18_clean_mixT", "Clean training"),
                                  ("r18_adv_T8_rep", "r18_adv_mixT", "Adversarial training")]):
        for k, ls, lab in [(a, "-", "fixed-T8 schedule"), (b, "--", "mixed-T schedule")]:
            ax.plot(mops([r["synops"] for r in C[k]["fixed_sweep"]]), [r["acc"] for r in C[k]["fixed_sweep"]], ls + "o",
                    ms=3, color=CFIX, label=f"Fixed $T$, {lab}")
            ax.plot(mops([r["synops"] for r in C[k]["vt_sweep"]]), [r["acc"] for r in C[k]["vt_sweep"]], ls + "s",
                    ms=3, color=CVT, label=f"VT, {lab}")
        ax.set_title(f"ResNet-18, {t}")
        ax.set_xlabel("Mean SynOps / image ($10^6$)")
        ax.set_ylim(35 if "Clean" in t else 40, None)
    axs[0].set_ylabel("Clean test accuracy (%)")
    axs[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "mixT_frontiers")


# ---------------------------------------------------------------- single-run robustness
def fig_single_robust():
    keys = ["r18_adv_T8_rep", "r18_adv_mixT", "vgg_adv_T8", "mbv2_adv_T8"]  # run-1 attack numbers withheld
    rows = {}
    for k in keys:
        for r in C[k]["attacks"]:
            if r["mode"] == "fixed" and r["role"] == "fixed_T8_baseline":
                rows[(k, r["attack"], "fix")] = r["acc"]
            if r["mode"] == "vt" and r["role"] == "selected" and r["objective"] == "untargeted":
                rows.setdefault((k, r["attack"], "vt"), r["acc"])
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    w = 0.38
    pg = [rows[(k, "pgd", "vt")] - rows[(k, "pgd", "fix")] for k in keys]
    aa = [rows[(k, "autoattack", "vt")] - rows[(k, "autoattack", "fix")] for k in keys]
    ax.bar(x - w / 2, pg, w, color=CFIX, label="PGD-20")
    ax.bar(x + w / 2, aa, w, color=CATK, label="AutoAttack")
    for i, (p, a) in enumerate(zip(pg, aa)):
        ax.text(i - w / 2, p + (0.15 if p >= 0 else -0.55), f"{p:+.1f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, a + (0.15 if a >= 0 else -0.55), f"{a:+.1f}", ha="center", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in keys], fontsize=7.5)
    ax.set_ylabel("VT $-$ fixed $T$=8 (pts)")
    ax.legend()
    save(fig, "pgd_aa_gap")


# ---------------------------------------------------------------- multiseed
def pooled_stats():
    out = {}
    pcs = []
    for arch in ["r18_adv", "vgg_adv"]:
        for m in ["clean", "pgd", "autoattack"]:
            for seed, v in F["multiseed"][arch][m]["per_checkpoint"].items():
                pcs.append(dict(arch=arch, metric=m, seed=seed, **v))
    df = pd.DataFrame(pcs)
    for m in ["clean", "pgd", "autoattack"]:
        d = df[df.metric == m]
        for ref in ["matched", "fixed_T8"]:
            delta = (d["vt"] - d[ref]).to_numpy()
            t, p = stats.ttest_1samp(delta, 0.0)
            pos = int((delta > 0).sum())
            sign_p = stats.binomtest(pos, len(delta), 0.5).pvalue if 0 < pos < len(delta) or True else None
            out[f"{m}_vs_{ref}"] = dict(mean=float(delta.mean()), std=float(delta.std(ddof=1)), t=float(t),
                                        p=float(p), n=len(delta), positive=pos, sign_p=float(sign_p),
                                        per_ckpt=[round(float(x), 2) for x in delta])
        for arch in ["r18_adv", "vgg_adv"]:
            dd = d[d.arch == arch]
            delta = (dd["vt"] - dd["fixed_T8"]).to_numpy()
            out[f"{arch}_{m}_vs_fixed_T8"] = dict(mean=float(delta.mean()), std=float(delta.std(ddof=1)),
                                                 per_ckpt=[round(float(x), 2) for x in delta])
    (HERE / "pooled.json").write_text(json.dumps(out, indent=1))
    return df, out


def fig_multiseed(df, pooled):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    metrics = [("clean", "Clean"), ("pgd", "PGD-20"), ("autoattack", "AutoAttack")]
    rng = np.random.default_rng(0)
    for ax, (ref, title) in zip(axs, [("matched", "VT $-$ compute-matched fixed $T$"), ("fixed_T8", "VT $-$ full horizon ($T$=8)")]):
        for i, (m, lab) in enumerate(metrics):
            for arch, mkr, col in [("r18_adv", "o", CFIX), ("vgg_adv", "s", CVT)]:
                d = df[(df.metric == m) & (df.arch == arch)]
                y = d["vt"] - d[ref]
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, marker=mkr, color=col, s=22, zorder=3,
                           label={"r18_adv": "ResNet-18 seed", "vgg_adv": "VGG16 seed"}[arch] if i == 0 else None)
            s = pooled[f"{m}_vs_{ref}"]
            ax.errorbar(i + 0.3, s["mean"], yerr=s["std"], fmt="_", color="k", ms=12, capsize=3, lw=1)
            ax.text(i + 0.36, s["mean"], f"p={s['p']:.3f}", fontsize=6.5, va="center")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels([l for _, l in metrics])
        ax.set_xlim(-0.5, 2.9)
        ax.set_title(title)
    axs[0].set_ylabel("Paired difference (pts)")
    axs[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "multiseed_paired")


# ---------------------------------------------------------------- compute overuse
def fig_overuse():
    rows = []
    for k in ORDER:
        for r in C[k]["attacks"]:
            pass
    # use facts.md numbers via compute_overuse block regenerated here from attacks
    co = {}
    for k in ORDER:
        clean_d = None
        for r in C[k]["attacks"]:
            if r.get("objective") == "compute_overuse" and r["role"] == "selected":
                co[k] = (r["depth"], r["acc"], r["theta"])
    # clean depth at same theta from VT sweep
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    keys = [k for k in ORDER if k in co]
    for i, k in enumerate(keys):
        th = co[k][2]
        cd = [r["depth"] for r in C[k]["vt_sweep"] if abs(r["theta"] - th) < 1e-6]
        cd = cd[0] if cd else np.nan
        ax.plot([cd, co[k][0]], [i, i], "-", color=CGRY, lw=1.5)
        ax.plot(cd, i, "o", color=CFIX, ms=6, label="Clean input" if i == 0 else None)
        ax.plot(co[k][0], i, "o", color=CATK, ms=6, label="Compute-overuse attack" if i == 0 else None)
        ax.text(8.15, i, f"+{co[k][0] - cd:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABEL[k] for k in keys], fontsize=7)
    ax.set_xlim(1, 8.7)
    ax.set_xlabel("Mean exit depth (timesteps)")
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.42), ncol=2, frameon=False)
    save(fig, "compute_overuse")


# ---------------------------------------------------------------- probe
def fig_probe():
    p = pd.DataFrame(F["probe"])
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4))
    for cond, col in [("clean", CFIX), ("PGD-20", CATK)]:
        d = p[p.condition == cond]
        axs[0].plot(d.step, d.acc, "-o", ms=3, color=col, label=cond)
        axs[1].plot(d.step, d.conf, "-o", ms=3, color=col, label=cond)
    axs[1].axhline(0.60, ls=":", color=CVT, label="$\\theta$ = 0.60")
    axs[0].set_ylabel("Accuracy of running prediction (%)")
    axs[1].set_ylabel("Mean max-softmax confidence")
    for ax in axs:
        ax.set_xlabel("Timestep $t$")
        ax.set_xticks(range(1, 9))
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "probe_trajectory")


# ---------------------------------------------------------------- training curves
def fig_train():
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.3), sharey=True)
    for ax, (folder, title) in zip(axs, [("Multiseed ResNet18", "ResNet-18"), ("Multiseed VGG16", "VGG16"),
                                         ("Multiseed MobileNetV2", "MobileNetV2")]):
        for i, h in enumerate(sorted((FINDINGS / folder).glob("train_history_adv_seed*.csv"))):
            d = pd.read_csv(h)
            ax.plot(np.arange(1, len(d) + 1), d["train_adv_acc"], color=CFIX, alpha=0.4 + 0.2 * i, lw=0.9,
                    label=f"train (RFGSM) s{i}")
            v = d["val_clean_acc"].dropna()
            ax.plot(v.index + 1, v.values, "o-", color=CVT, alpha=0.4 + 0.2 * i, ms=2, lw=0.9, label=f"val clean s{i}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "train_curves")


if __name__ == "__main__":
    fig_fixed_sweeps()
    fig_frontiers()
    fig_frontier_main()
    fig_selection()
    fig_delta()
    fig_exit_hist()
    fig_stopping()
    fig_mixT()
    fig_single_robust()
    df, pooled = pooled_stats()
    fig_multiseed(df, pooled)
    fig_overuse()
    fig_probe()
    fig_train()
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in pooled.items() if "_vs_" in k and not k.startswith(("r18", "vgg"))}, indent=0))
    print(sorted(p.name for p in FIG.iterdir()))
