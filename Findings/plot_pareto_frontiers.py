from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent

TARGETS = [
    ("ResNet Clean Trained", ROOT / "ResNet Clean Trained"),
    ("ResNet Adversarial Trained", ROOT / "ResNet Adversarial Trained"),
]

COLORS = {
    "fixed": "#4C72B0",
    "vt": "#DD8452",
    "selected": "#C44E52",
}


def load_csv(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
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
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
        }
    )


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="both", color="#D0D0D0", linewidth=1.0, alpha=0.9)


def save_current_figure(folder: Path, stem: str) -> None:
    figures_dir = folder / "figures"
    figures_dir.mkdir(exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def plot_frontier(folder_name: str, folder: Path, stem: str) -> None:
    fixed_df = load_csv(folder, "test_fixed_t_sweep.csv").sort_values("time_steps")
    vt_df = load_csv(folder, "test_vt_threshold_sweep.csv").sort_values("threshold")
    selected_df = load_csv(folder, "main_selected_thresholds.csv")
    selected_theta = float(selected_df.loc[selected_df["threshold_role"] == "selected", "threshold"].iloc[0])
    selected_row = vt_df.loc[vt_df["threshold"] == selected_theta].iloc[0]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    ax.plot(
        fixed_df["avg_synops"] / 1e6,
        fixed_df["accuracy"],
        marker="o",
        markersize=7,
        linewidth=2.6,
        color=COLORS["fixed"],
        label="Fixed-T",
    )
    ax.plot(
        vt_df["avg_synops"] / 1e6,
        vt_df["accuracy"],
        marker="s",
        markersize=7,
        linewidth=2.6,
        color=COLORS["vt"],
        label="VT",
    )

    for _, row in fixed_df.iterrows():
        ax.annotate(
            f"T={int(row['time_steps'])}",
            xy=(row["avg_synops"] / 1e6, row["accuracy"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color=COLORS["fixed"],
        )

    for _, row in vt_df.iterrows():
        ax.annotate(
            f"{float(row['threshold']):.2f}",
            xy=(row["avg_synops"] / 1e6, row["accuracy"]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
            color=COLORS["vt"],
        )

    ax.scatter(
        [selected_row["avg_synops"] / 1e6],
        [selected_row["accuracy"]],
        s=220,
        color=COLORS["selected"],
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label=f"Selected VT (theta={selected_theta:.2f})",
    )

    ax.set_title(f"{folder_name}: Accuracy vs SynOps")
    ax.set_xlabel("Average SynOps (Millions)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    style_axes(ax)
    save_current_figure(folder, stem)


def main() -> None:
    configure_style()
    plot_frontier("Clean-Trained", ROOT / "ResNet Clean Trained", "fig_pareto_clean_accuracy_synops")
    plot_frontier("Adversarially Trained", ROOT / "ResNet Adversarial Trained", "fig_pareto_adv_accuracy_synops")


if __name__ == "__main__":
    main()
