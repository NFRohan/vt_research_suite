from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    "fixed_t8": "#4C72B0",
    "matched": "#55A868",
    "vt": "#DD8452",
    "pgd": "#4C72B0",
    "autoattack": "#C44E52",
    "vt_pgd": "#DD8452",
    "vt_auto": "#937860",
}


def load_csv(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
        }
    )


def save_current_figure(stem: str) -> None:
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#D0D0D0", linewidth=1.0, alpha=0.9)


def annotate_bars(ax, bars, values) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
        )


def plot_selected_comparison(attack_name: str, stem: str, title: str) -> None:
    df = load_csv("subset_consistent_selected_attack_comparison.csv")
    attack_df = df[df["attack_name"] == attack_name].copy()

    rows = [
        attack_df.loc[attack_df["comparison_family"] == "fixed_T8_subset"].iloc[0],
        attack_df.loc[attack_df["comparison_family"] == "matched_fixed_control"].iloc[0],
        attack_df.loc[attack_df["comparison_family"] == "vt_selected_subset"].iloc[0],
    ]

    labels = ["Fixed T=8", "Matched Fixed T=4", "VT selected\n(theta=0.5)"]
    values = [float(row["accuracy"]) for row in rows]
    colors = [COLORS["fixed_t8"], COLORS["matched"], COLORS["vt"]]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    annotate_bars(ax, bars, values)

    vt_value = values[2]
    matched_value = values[1]
    fixed_value = values[0]

    ax.text(
        0.98,
        0.98,
        f"VT - matched: {vt_value - matched_value:+.1f}\nVT - T8: {vt_value - fixed_value:+.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.95, "edgecolor": "#BBBBBB"},
    )

    ax.set_ylim(0, max(values) + 8)
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_title(title)
    style_axes(ax, grid_axis="y")
    save_current_figure(stem)


def plot_attack_sweep() -> None:
    df = load_csv("subset_consistent_fixed_t_and_vt_attack_sweep.csv")

    fixed_df = df[df["comparison_family"] == "fixed_t_sweep_subset"].copy()
    vt_df = df[df["comparison_family"] == "vt_threshold_subset"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    for attack_name, color, marker in [
        ("pgd", COLORS["pgd"], "o"),
        ("autoattack", COLORS["autoattack"], "s"),
    ]:
        attack_slice = fixed_df[fixed_df["attack_name"] == attack_name].sort_values("time_steps")
        ax.plot(
            attack_slice["time_steps"],
            attack_slice["accuracy"],
            marker=marker,
            markersize=7,
            linewidth=2.5,
            color=color,
            label=attack_name.upper(),
        )
        best_row = attack_slice.loc[attack_slice["accuracy"].idxmax()]
        ax.scatter(
            [best_row["time_steps"]],
            [best_row["accuracy"]],
            s=140,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            f"T={int(best_row['time_steps'])}, {best_row['accuracy']:.1f}",
            xy=(best_row["time_steps"], best_row["accuracy"]),
            xytext=(8, 6 if attack_name == "pgd" else -18),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Fixed-T Sweep on 1k Attack Subset")
    ax.set_xlabel("Fixed Timestep Budget")
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_xticks(sorted(fixed_df["time_steps"].unique()))
    ax.legend(loc="upper left")
    style_axes(ax, grid_axis="both")

    ax = axes[1]
    for attack_name, color, marker in [
        ("pgd", COLORS["vt_pgd"], "o"),
        ("autoattack", COLORS["vt_auto"], "s"),
    ]:
        attack_slice = vt_df[vt_df["attack_name"] == attack_name].sort_values("threshold")
        ax.plot(
            attack_slice["threshold"],
            attack_slice["accuracy"],
            marker=marker,
            markersize=7,
            linewidth=2.5,
            color=color,
            label=attack_name.upper(),
        )
        best_row = attack_slice.loc[attack_slice["accuracy"].idxmax()]
        ax.scatter(
            [best_row["threshold"]],
            [best_row["accuracy"]],
            s=140,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            f"theta={best_row['threshold']:.1f}, {best_row['accuracy']:.1f}",
            xy=(best_row["threshold"], best_row["accuracy"]),
            xytext=(8, 6 if attack_name == "pgd" else -18),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1.2, label="Selected theta")
    ax.set_title("VT Threshold Sweep on 1k Attack Subset")
    ax.set_xlabel("VT Threshold")
    ax.set_ylabel("Robust Accuracy (%)")
    ax.set_xticks(sorted(vt_df["threshold"].unique()))
    ax.legend(loc="upper right")
    style_axes(ax, grid_axis="both")

    save_current_figure("fig_attack_sweep_subset")


def main() -> None:
    configure_style()
    plot_selected_comparison(
        "pgd",
        "fig_pgd_selected_subset_comparison",
        "PGD: VT Remains Competitive on the 1k Subset",
    )
    plot_selected_comparison(
        "autoattack",
        "fig_autoattack_selected_subset_comparison",
        "AutoAttack: VT Loses to Fixed-Time Baselines",
    )
    plot_attack_sweep()


if __name__ == "__main__":
    main()
