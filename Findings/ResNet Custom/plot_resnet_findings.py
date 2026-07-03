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
    "fixed": "#4C72B0",
    "vt": "#DD8452",
    "selected": "#C44E52",
    "green": "#55A868",
    "orange": "#DD8452",
    "brown": "#937860",
    "purple": "#8172B2",
}


def load_csv(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def save_current_figure(stem: str) -> None:
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def threshold_label(value: float) -> str:
    return rf"$\theta={value:.1f}$"


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (9, 6),
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "axes.axisbelow": True,
        }
    )


def annotate_bars(ax, bars, values, *, fmt="{:.1f}", dy=4, prefix="") -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{prefix}{fmt.format(value)}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=10,
        )


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#D0D0D0", linewidth=1.0, alpha=0.9)


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    ordered = df.sort_values([x_col, y_col], ascending=[True, False]).reset_index(drop=True)
    best_y = float("-inf")
    keep_rows = []
    for _, row in ordered.iterrows():
        current_y = float(row[y_col])
        if current_y > best_y:
            keep_rows.append(row)
            best_y = current_y
    return pd.DataFrame(keep_rows)


def plot_clean_frontier() -> None:
    fixed_df = load_csv("test_fixed_t_sweep.csv").sort_values("time_steps")
    vt_df = load_csv("test_vt_threshold_sweep.csv").sort_values("threshold")
    selected_df = load_csv("main_selected_thresholds.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        fixed_df["avg_synops"],
        fixed_df["accuracy"],
        marker="o",
        markersize=8,
        linewidth=2.8,
        color=COLORS["fixed"],
        label="Fixed-T",
    )
    ax.plot(
        vt_df["avg_synops"],
        vt_df["accuracy"],
        marker="s",
        markersize=8,
        linewidth=2.8,
        color=COLORS["vt"],
        label="VT",
    )

    selected_threshold = float(
        selected_df.loc[selected_df["threshold_role"] == "selected", "threshold"].iloc[0]
    )
    selected_row = vt_df.loc[vt_df["threshold"] == selected_threshold].iloc[0]
    ax.scatter(
        [selected_row["avg_synops"]],
        [selected_row["accuracy"]],
        s=260,
        color=COLORS["selected"],
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label=f"Selected VT ({threshold_label(selected_threshold)})",
    )
    ax.annotate(
        threshold_label(selected_threshold),
        xy=(selected_row["avg_synops"], selected_row["accuracy"]),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=12,
        color="#7A1F28",
    )

    ax.set_title("Clean Accuracy vs SynOps on CIFAR-10 Test")
    ax.set_xlabel("Average SynOps")
    ax.set_ylabel("Accuracy (%)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="upper left")
    style_axes(ax, grid_axis="both")
    save_current_figure("clean_accuracy_frontier")


def plot_pareto_frontier() -> None:
    fixed_df = load_csv("test_fixed_t_sweep.csv").copy()
    fixed_df["family"] = "Fixed-T"
    fixed_df["label"] = fixed_df["time_steps"].map(lambda t: f"T={int(t)}")

    vt_df = load_csv("test_vt_threshold_sweep.csv").copy()
    vt_df["family"] = "VT"
    vt_df["label"] = vt_df["threshold"].map(lambda x: threshold_label(float(x)))

    combined = pd.concat(
        [
            fixed_df[["family", "label", "avg_synops", "accuracy"]],
            vt_df[["family", "label", "avg_synops", "accuracy"]],
        ],
        ignore_index=True,
    )
    frontier = pareto_frontier(combined, "avg_synops", "accuracy")

    fig, ax = plt.subplots(figsize=(9, 6))

    fixed_mask = combined["family"] == "Fixed-T"
    vt_mask = combined["family"] == "VT"

    ax.scatter(
        combined.loc[fixed_mask, "avg_synops"],
        combined.loc[fixed_mask, "accuracy"],
        s=75,
        color=COLORS["fixed"],
        alpha=0.65,
        label="Fixed-T candidates",
        zorder=3,
    )
    ax.scatter(
        combined.loc[vt_mask, "avg_synops"],
        combined.loc[vt_mask, "accuracy"],
        s=75,
        color=COLORS["vt"],
        alpha=0.65,
        marker="s",
        label="VT candidates",
        zorder=3,
    )

    ax.plot(
        frontier["avg_synops"],
        frontier["accuracy"],
        color=COLORS["selected"],
        linewidth=3.0,
        marker="o",
        markersize=7,
        label="Pareto frontier",
        zorder=4,
    )

    for _, row in frontier.iterrows():
        ax.annotate(
            str(row["label"]),
            xy=(row["avg_synops"], row["accuracy"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Pareto Frontier of Clean Accuracy vs SynOps")
    ax.set_xlabel("Average SynOps")
    ax.set_ylabel("Accuracy (%)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="both")
    save_current_figure("clean_pareto_frontier")


def plot_matched_delta() -> None:
    match_df = load_csv("test_vt_fixed_matches.csv").sort_values("threshold").copy()
    labels = [threshold_label(value) for value in match_df["threshold"]]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, match_df["accuracy_delta"], color=COLORS["fixed"], width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title("Accuracy Gain of VT Over Compute-Matched Fixed-T")
    ax.set_xlabel("VT Threshold")
    ax.set_ylabel("Accuracy Delta (VT - Fixed-T, %)")

    for bar, delta in zip(bars, match_df["accuracy_delta"]):
        ax.annotate(
            f"{delta:+.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom" if delta >= 0 else "top",
            xytext=(0, 4 if delta >= 0 else -6),
            textcoords="offset points",
            fontsize=10,
        )

    save_current_figure("matched_fixed_vs_vt_accuracy_delta")


def plot_stopping_rules() -> None:
    stopping_df = load_csv("stopping_rule_test_results.csv")
    fixed_row = stopping_df.loc[stopping_df["mode"] == "fixed"].iloc[0]
    vt_rules = stopping_df.loc[stopping_df["mode"] == "vt"].copy()
    vt_rules["rule_label"] = vt_rules["stop_rule"].replace(
        {
            "max_softmax": "Max softmax",
            "margin": "Margin",
            "entropy": "Entropy",
            "temperature_softmax": "Temperature softmax",
            "evidential_surrogate": "Evidential surrogate",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(vt_rules["rule_label"], vt_rules["accuracy"], color=COLORS["green"], width=0.72)
    ax.axhline(
        float(fixed_row["accuracy"]),
        color=COLORS["selected"],
        linewidth=2.0,
        linestyle="--",
        label=f"Fixed T=8 ({fixed_row['accuracy']:.2f}%)",
    )
    ax.set_title("Stopping Rule Comparison at Matched Compute")
    ax.set_xlabel("Stopping Rule")
    ax.set_ylabel("Accuracy (%)")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")

    for bar, synops in zip(bars, vt_rules["avg_synops"]):
        ax.annotate(
            f"{synops / 1e9:.2f}B",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.legend(loc="lower right")
    style_axes(ax, grid_axis="y")
    save_current_figure("stopping_rule_comparison")


def plot_attack_accuracy() -> None:
    attack_df = load_csv("attack_results.csv").copy()

    role_order = ["fixed_T8_baseline", "lower_neighbor", "selected", "upper_neighbor"]
    role_label_map = {
        "fixed_T8_baseline": "Fixed T=8",
        "lower_neighbor": threshold_label(0.5),
        "selected": threshold_label(0.6),
        "upper_neighbor": threshold_label(0.7),
    }

    untargeted = attack_df.loc[attack_df["attack_objective"] == "untargeted"].copy()
    untargeted["threshold_role"] = pd.Categorical(untargeted["threshold_role"], role_order, ordered=True)
    untargeted = untargeted.sort_values("threshold_role")
    untargeted_labels = untargeted["threshold_role"].map(role_label_map).tolist()

    wrong_exit = attack_df.loc[attack_df["attack_objective"] == "wrong_early_exit"].copy()
    wrong_exit["threshold_role"] = pd.Categorical(
        wrong_exit["threshold_role"],
        ["lower_neighbor", "selected", "upper_neighbor"],
        ordered=True,
    )
    wrong_exit = wrong_exit.sort_values("threshold_role")
    wrong_exit_labels = wrong_exit["threshold_role"].map(role_label_map).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    bars0 = axes[0].bar(
        untargeted_labels,
        untargeted["accuracy"],
        color=["#8FB2D0", "#6E9CC2", "#567F9F", "#445C71"],
        width=0.78,
        zorder=3,
    )
    axes[0].set_title("Untargeted PGD (20 steps, EOT=8)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Accuracy (%)")
    plt.setp(axes[0].get_xticklabels(), rotation=12)
    annotate_bars(axes[0], bars0, untargeted["accuracy"])

    bars1 = axes[1].bar(
        wrong_exit_labels,
        wrong_exit["accuracy"],
        color=["#84B889", "#4E9A5D", "#447451"],
        width=0.78,
        zorder=3,
    )
    axes[1].set_title("Wrong-Early-Exit PGD")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    plt.setp(axes[1].get_xticklabels(), rotation=12)
    annotate_bars(axes[1], bars1, wrong_exit["accuracy"])
    style_axes(axes[0], grid_axis="y")
    style_axes(axes[1], grid_axis="y")

    save_current_figure("adaptive_attack_accuracy")


def plot_compute_overuse() -> None:
    overuse_df = load_csv("compute_overuse_deltas.csv").sort_values("threshold").copy()
    labels = [threshold_label(value) for value in overuse_df["threshold"]]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, overuse_df["stopping_depth_increase"], color=COLORS["orange"], width=0.72)
    ax.set_title("Compute-Overuse Attack Increases VT Stopping Depth")
    ax.set_xlabel("VT Threshold")
    ax.set_ylabel("Increase in Average Stopping Depth")
    annotate_bars(ax, bars, overuse_df["stopping_depth_increase"], fmt="{:.3f}", prefix="+")
    style_axes(ax, grid_axis="y")
    save_current_figure("compute_overuse_stopping_depth")


def plot_training_dynamics() -> None:
    history_df = load_csv("resnet_full_model_history.csv").copy()
    history_df = history_df.reset_index(names="epoch")
    history_df["epoch"] += 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        history_df["epoch"],
        history_df["train_clean_acc"],
        label="Train clean",
        color=COLORS["fixed"],
        linewidth=2.2,
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["train_adv_acc"],
        label="Train adversarial",
        color=COLORS["vt"],
        linewidth=2.2,
    )
    if history_df["val_clean_acc"].notna().any():
        val_df = history_df.loc[history_df["val_clean_acc"].notna(), ["epoch", "val_clean_acc"]]
        axes[0].plot(
            val_df["epoch"],
            val_df["val_clean_acc"],
            label="Val clean",
            color=COLORS["green"],
            linewidth=2.2,
        )
    axes[0].set_title("Training Dynamics")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right")
    style_axes(axes[0], grid_axis="y")

    axes[1].plot(
        history_df["epoch"],
        history_df["sigma_mean"],
        color=COLORS["purple"],
        linewidth=2.2,
    )
    axes[1].set_title("Mean Learnable Sigma")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sigma")
    style_axes(axes[1], grid_axis="y")

    save_current_figure("training_dynamics")


def plot_autoattack_if_present() -> None:
    autoattack_files = sorted(ROOT.glob("autoattack*.csv"))
    if not autoattack_files:
        print("No AutoAttack CSV found in Findings/ResNet; skipping AutoAttack figure.")
        return

    autoattack_path = autoattack_files[-1]
    auto_df = pd.read_csv(autoattack_path).copy()
    label_map = {
        "fixed_T8_baseline": "Fixed T=8",
        "lower_neighbor": threshold_label(0.5),
        "selected": threshold_label(0.6),
        "upper_neighbor": threshold_label(0.7),
    }
    order = ["fixed_T8_baseline", "lower_neighbor", "selected", "upper_neighbor"]
    auto_df["threshold_role"] = pd.Categorical(auto_df["threshold_role"], order, ordered=True)
    auto_df = auto_df.sort_values("threshold_role")
    labels = auto_df["threshold_role"].map(label_map).tolist()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(labels, auto_df["accuracy"], color=COLORS["brown"], width=0.72)
    eot = int(auto_df["attack_eot"].iloc[0]) if "attack_eot" in auto_df.columns else 0
    ax.set_title(f"AutoAttack Accuracy on Attack Subset (EOT={eot})")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    plt.setp(ax.get_xticklabels(), rotation=12)
    annotate_bars(ax, bars, auto_df["accuracy"])
    style_axes(ax, grid_axis="y")
    save_current_figure(f"autoattack_summary_{autoattack_path.stem}")


def main() -> None:
    configure_style()
    plot_clean_frontier()
    plot_pareto_frontier()
    plot_matched_delta()
    plot_stopping_rules()
    plot_attack_accuracy()
    plot_compute_overuse()
    plot_training_dynamics()
    plot_autoattack_if_present()


if __name__ == "__main__":
    main()
