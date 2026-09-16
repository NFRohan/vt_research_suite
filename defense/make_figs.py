"""Generate defense figures from the result CSVs. No GPU, no new experiments."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CFIX = "#36649E"   # fixed-T baseline
CVT  = "#CE7B1A"   # Variable Timing
CATK = "#B22D2D"   # under attack
OUT  = os.path.join(os.path.dirname(__file__), "figs")
ROOT = os.path.join(os.path.dirname(__file__), "..", "Findings")

CONFIGS = [
    ("ResNet-18",   "Clean", "ResNet Clean Trained",            0.90),
    ("VGG16",       "Clean", "VGG Clean Trained",               0.80),
    ("MobileNetV2", "Clean", "MobileNetV2 Clean Trained",       0.99),
    ("ResNet-18",   "Adv.",  "ResNet Adv T8 Rerun",             0.60),
    ("VGG16",       "Adv.",  "VGG Adversarial Trained",         0.50),
    ("MobileNetV2", "Adv.",  "MobileNetV2 Adversarial Trained", 0.60),
]


def load(folder):
    p = os.path.join(ROOT, folder)
    fx = pd.read_csv(os.path.join(p, "test_fixed_t_sweep.csv"))
    vt = pd.read_csv(os.path.join(p, "test_vt_threshold_sweep.csv"))
    if "mode" in fx.columns:
        fx = fx[fx["mode"] == "fixed"]
    fx = fx.dropna(subset=["avg_synops"]).sort_values("avg_synops")
    vt = vt.dropna(subset=["avg_synops"]).sort_values("avg_synops")
    return fx, vt


def matched(vt_syn, fx):
    f = fx.copy()
    f["gap"] = (f["avg_synops"] - vt_syn).abs()
    return f.sort_values("gap").iloc[0]


def panel(ax, fx, vt, theta, title=None, legend=False, norm=True):
    """norm=True plots x as a fraction of full-horizon (T=8) SynOps so panels share an axis."""
    full = fx["avg_synops"].max()
    sx = (lambda v: v / full) if norm else (lambda v: v / 1e6)
    ax.plot(sx(fx["avg_synops"]), fx["accuracy"], "o-", color=CFIX,
            lw=1.8, ms=4, label="Fixed-$T$ truncation", zorder=2)
    ax.plot(sx(vt["avg_synops"]), vt["accuracy"], "s-", color=CVT,
            lw=1.8, ms=4, label="Variable Timing", zorder=3)

    row = vt.iloc[(vt["threshold"] - theta).abs().argsort().iloc[0]]
    m = matched(float(row["avg_synops"]), fx)
    rx, ry = sx(float(row["avg_synops"])), float(row["accuracy"])
    mx, my = sx(float(m["avg_synops"])), float(m["accuracy"])
    ax.scatter([rx], [ry], s=120, facecolors="none", edgecolors=CVT, lw=2.2, zorder=5)
    ax.scatter([mx], [my], s=120, facecolors="none", edgecolors=CFIX, lw=2.2, zorder=5)
    ax.annotate("", xy=(rx, ry), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.3), zorder=4)

    # zoom y to the informative band (exclude the chance-level T=1 point)
    lo = min(float(vt["accuracy"].min()), my) - 6
    hi = max(float(fx["accuracy"].max()), float(vt["accuracy"].max())) + 4
    ax.set_ylim(lo, hi)
    # place the delta label in free space below-right of the VT marker
    xlo, xhi = ax.get_xlim()
    near_right = (rx - xlo) / max(xhi - xlo, 1e-9) > 0.68
    ax.annotate(("$+$%.2f" % (ry - my)),
                xy=(rx, ry),
                xytext=((-10, -18) if near_right else (10, -16)),
                textcoords="offset points", color=CVT, fontsize=9,
                fontweight="bold", ha=("right" if near_right else "left"))
    ax.margins(x=0.08)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    if legend:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)


def fig_representative():
    fx, vt = load("VGG Clean Trained")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    panel(ax, fx, vt, 0.80, legend=True, norm=False)
    ax.set_xlabel("Average SynOps per image (millions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("VGG16, clean-trained: accuracy vs. compute", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frontier_vgg_clean.png"), dpi=200)
    plt.close(fig)
    print("wrote frontier_vgg_clean.png")


def fig_small_multiples():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for k, (bb, reg, folder, th) in enumerate(CONFIGS):
        ax = axes[k // 3][k % 3]
        try:
            fx, vt = load(folder)
        except Exception as e:
            ax.text(0.5, 0.5, "missing", ha="center", va="center"); continue
        panel(ax, fx, vt, th, title=bb + " / " + reg, legend=(k == 0))
        if k % 3 == 0:
            ax.set_ylabel("Accuracy (%)")
        if k // 3 == 1:
            ax.set_xlabel("SynOps / full-horizon SynOps")
    fig.suptitle("VT vs. compute-matched fixed-$T$ across three backbones and two regimes",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "frontier_small_multiples.png"), dpi=200)
    plt.close(fig)
    print("wrote frontier_small_multiples.png")


def fig_pgd_aa_gap():
    rows = [("ResNet-18", 46.10, 48.70, 28.85, 24.15),
            ("VGG16",     45.90, 44.45, 10.75,  8.55),
            ("MobileNetV2", 35.60, 34.75, 27.60, 27.55)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, (idx, name) in zip(axes, [(1, "PGD-20"), (3, "AutoAttack")]):
        xs = range(len(rows))
        fixv = [r[idx] for r in rows]
        vtv = [r[idx + 1] for r in rows]
        ax.bar([x - 0.19 for x in xs], fixv, 0.38, color=CFIX, label="Fixed $T$=8")
        ax.bar([x + 0.19 for x in xs], vtv, 0.38, color=CVT, label="VT selected")
        for x, (a, b) in enumerate(zip(fixv, vtv)):
            ax.text(x + 0.19, b + 0.8, ("%+.2f" % (b - a)),
                    ha="center", fontsize=8,
                    color=(CATK if b < a else "green"), fontweight="bold")
        ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
        ax.set_title(name, fontsize=11); ax.grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].set_ylabel("Robust accuracy (%)"); axes[0].legend(fontsize=8)
    fig.suptitle("Adversarially trained: VT vs. fixed $T$=8 under two attacks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "pgd_aa_gap.png"), dpi=200)
    plt.close(fig)
    print("wrote pgd_aa_gap.png")


def fig_compute_overuse():
    rows = [("ResNet-18", "Clean", 2.393, 3.579), ("VGG16", "Clean", 3.533, 4.702),
            ("MobileNetV2", "Clean", 6.556, 7.707), ("ResNet-18", "Adv.", 5.130, 5.404),
            ("VGG16", "Adv.", 6.272, 6.311), ("MobileNetV2", "Adv.", 7.208, 7.262)]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    for i, (bb, reg, c, a) in enumerate(rows):
        ax.plot([c, a], [i, i], "-", color="0.6", lw=2, zorder=1)
        ax.scatter([c], [i], s=70, color=CFIX, zorder=3, label="clean input" if i == 0 else None)
        ax.scatter([a], [i], s=70, color=CATK, zorder=3, label="under compute-overuse attack" if i == 0 else None)
        ax.text(a + 0.09, i, ("+%.3f" % (a - c)), va="center", fontsize=8,
                color=CATK, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] + " / " + r[1] for r in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Average stopping depth (timesteps)")
    ax.set_xlim(2, 8.6); ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_title("Compute-overuse attack inflates inference depth (label preserved)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compute_overuse.png"), dpi=200)
    plt.close(fig)
    print("wrote compute_overuse.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_representative()
    fig_small_multiples()
    fig_pgd_aa_gap()
    fig_compute_overuse()
