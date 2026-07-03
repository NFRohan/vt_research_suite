from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGETS = [
    ROOT / "ResNet Clean Trained",
    ROOT / "ResNet Adversarial Trained",
]

COLORS = {
    "fixed": "#4C72B0",
    "vt": "#DD8452",
    "selected": "#C44E52",
    "green": "#55A868",
    "brown": "#937860",
    "purple": "#8172B2",
    "pgd": "#4C72B0",
    "wrong": "#55A868",
    "auto": "#C44E52",
    "compute": "#DD8452",
}


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
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
        }
    )


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#D0D0D0", linewidth=1.0, alpha=0.9)


def load_csv(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


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


def threshold_label(value: float) -> str:
    return f"{value:.2f}"


def short_stop_rule(name: str) -> str:
    return {
        "max_softmax": "Max softmax",
        "margin": "Margin",
        "entropy": "Entropy",
        "temperature_softmax": "Temp-softmax",
        "evidential_surrogate": "Evidential",
        "fixed": "Fixed T=8",
    }.get(name, name)


def annotate_bars(ax, bars, values, *, fmt="{:.1f}", dy=4) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=9,
        )


def selected_panel_df(folder: Path) -> pd.DataFrame:
    panel = load_csv(folder, "test_selected_threshold_results.csv").copy()
    return panel.sort_values("threshold")


def fixed_baseline_row(folder: Path) -> pd.Series:
    baseline = load_csv(folder, "test_fixed_baseline_results.csv")
    return baseline.iloc[0]


def plot_fig1_fixed_t_sweep(folder: Path) -> None:
    fixed_df = load_csv(folder, "test_fixed_t_sweep.csv").sort_values("time_steps")

    fig, ax1 = plt.subplots(figsize=(8.6, 5.4))
    ax2 = ax1.twinx()

    ax1.plot(
        fixed_df["time_steps"],
        fixed_df["accuracy"],
        marker="o",
        linewidth=2.5,
        color=COLORS["fixed"],
        label="Accuracy",
    )
    ax2.plot(
        fixed_df["time_steps"],
        fixed_df["avg_synops"] / 1e6,
        marker="s",
        linewidth=2.5,
        color=COLORS["vt"],
        label="SynOps",
    )

    ax1.set_title("Fixed-T Clean Sweep")
    ax1.set_xlabel("Fixed Timestep Budget")
    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Average SynOps (Millions)")
    ax1.set_xticks(fixed_df["time_steps"].tolist())

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")
    style_axes(ax1, grid_axis="both")
    save_current_figure(folder, "fig1_fixed_t_sweep")


def plot_fig2_vt_threshold_sweep(folder: Path) -> None:
    vt_df = load_csv(folder, "test_vt_threshold_sweep.csv").sort_values("threshold")
    selected_df = load_csv(folder, "main_selected_thresholds.csv")
    selected_theta = float(selected_df.loc[selected_df["threshold_role"] == "selected", "threshold"].iloc[0])

    fig, ax1 = plt.subplots(figsize=(8.6, 5.4))
    ax2 = ax1.twinx()

    ax1.plot(
        vt_df["threshold"],
        vt_df["accuracy"],
        marker="o",
        linewidth=2.5,
        color=COLORS["vt"],
        label="Accuracy",
    )
    ax2.plot(
        vt_df["threshold"],
        vt_df["avg_stopping_depth"],
        marker="s",
        linewidth=2.5,
        color=COLORS["green"],
        label="Avg. stopping depth",
    )

    ax1.axvline(selected_theta, color=COLORS["selected"], linestyle="--", linewidth=1.5, label=f"Selected theta={selected_theta:.2f}")

    ax1.set_title("VT Threshold Sweep")
    ax1.set_xlabel("VT Threshold")
    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Average Stopping Depth")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")
    style_axes(ax1, grid_axis="both")
    save_current_figure(folder, "fig2_vt_threshold_sweep")


def plot_fig_clean_compute_savings(folder: Path) -> None:
    panel = selected_panel_df(folder)
    baseline = fixed_baseline_row(folder)
    fixed_df = load_csv(folder, "test_fixed_t_sweep.csv")
    vt_df = load_csv(folder, "test_vt_threshold_sweep.csv")

    baseline_acc = float(baseline["accuracy"])
    baseline_synops = float(baseline["avg_synops"])

    panel["synops_saved_pct"] = 100.0 * (1.0 - panel["avg_synops"] / baseline_synops)
    panel["accuracy_delta"] = panel["accuracy"] - baseline_acc

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    labels = [f"{role}\n(theta={thr:.2f})" for role, thr in zip(panel["threshold_role"], panel["threshold"])]
    bars0 = axes[0].bar(labels, panel["synops_saved_pct"], color=[COLORS["green"], COLORS["vt"], COLORS["brown"]], width=0.72)
    annotate_bars(axes[0], bars0, panel["synops_saved_pct"], fmt="{:.1f}")
    axes[0].set_title("SynOps Saved vs Fixed T=8")
    axes[0].set_ylabel("SynOps Saved (%)")
    plt.setp(axes[0].get_xticklabels(), rotation=12)
    style_axes(axes[0], grid_axis="y")

    axes[1].plot(fixed_df["avg_synops"], fixed_df["accuracy"], marker="o", linewidth=2.2, color=COLORS["fixed"], label="Fixed-T")
    axes[1].plot(vt_df["avg_synops"], vt_df["accuracy"], marker="s", linewidth=2.2, color=COLORS["vt"], label="VT")
    axes[1].set_title("Clean Accuracy-Compute Frontier")
    axes[1].set_xlabel("Average SynOps")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axes[1].legend(loc="lower right")
    style_axes(axes[1], grid_axis="both")

    bars2 = axes[2].bar(labels, panel["accuracy_delta"], color=[COLORS["green"], COLORS["vt"], COLORS["brown"]], width=0.72)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    for bar, delta in zip(bars2, panel["accuracy_delta"]):
        axes[2].annotate(
            f"{delta:+.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom" if delta >= 0 else "top",
            xytext=(0, 4 if delta >= 0 else -6),
            textcoords="offset points",
            fontsize=9,
        )
    axes[2].set_title("Accuracy Relative to Fixed T=8")
    axes[2].set_ylabel("Accuracy Delta (%)")
    plt.setp(axes[2].get_xticklabels(), rotation=12)
    style_axes(axes[2], grid_axis="y")

    save_current_figure(folder, "fig_clean_compute_savings_vs_t8")


def plot_fig6_stopping_rules(folder: Path) -> None:
    stop_df = load_csv(folder, "stopping_rule_test_results.csv").copy()
    fixed_row = stop_df.loc[stop_df["mode"] == "fixed"].iloc[0]
    vt_rules = stop_df.loc[stop_df["mode"] == "vt"].copy()
    vt_rules["rule_label"] = vt_rules["stop_rule"].map(short_stop_rule)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))

    bars0 = axes[0].bar(vt_rules["rule_label"], vt_rules["accuracy"], color=COLORS["green"], width=0.72)
    axes[0].axhline(float(fixed_row["accuracy"]), color=COLORS["selected"], linestyle="--", linewidth=1.4)
    annotate_bars(axes[0], bars0, vt_rules["accuracy"])
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    plt.setp(axes[0].get_xticklabels(), rotation=18, ha="right")
    style_axes(axes[0], grid_axis="y")

    bars1 = axes[1].bar(vt_rules["rule_label"], vt_rules["avg_stopping_depth"], color=COLORS["vt"], width=0.72)
    annotate_bars(axes[1], bars1, vt_rules["avg_stopping_depth"], fmt="{:.2f}")
    axes[1].set_title("Average Stopping Depth")
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    style_axes(axes[1], grid_axis="y")

    bars2 = axes[2].bar(vt_rules["rule_label"], vt_rules["avg_synops"] / 1e6, color=COLORS["brown"], width=0.72)
    annotate_bars(axes[2], bars2, vt_rules["avg_synops"] / 1e6, fmt="{:.1f}")
    axes[2].set_title("Average SynOps")
    axes[2].set_ylabel("SynOps (Millions)")
    plt.setp(axes[2].get_xticklabels(), rotation=18, ha="right")
    style_axes(axes[2], grid_axis="y")

    save_current_figure(folder, "fig6_stopping_rules")


def plot_fig_attack_compute_savings(folder: Path) -> None:
    attack_df = load_csv(folder, "attack_results.csv").copy()
    panel = selected_panel_df(folder).copy()
    clean_depth_map = dict(zip(panel["threshold"], panel["avg_stopping_depth"]))

    untargeted = attack_df.loc[attack_df["attack_objective"] == "untargeted"].copy()
    wrong_exit = attack_df.loc[attack_df["attack_objective"] == "wrong_early_exit"].copy()
    autoattack = attack_df.loc[(attack_df["attack_name"] == "autoattack") & (attack_df["attack_objective"] == "untargeted")].copy()
    vt_only = attack_df.loc[attack_df["mode"] == "vt"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    if not untargeted.empty:
        labels = [
            "Fixed T=8" if pd.isna(row["threshold"]) else f"theta={float(row['threshold']):.2f}"
            for _, row in untargeted.sort_values(["mode", "threshold"], na_position="first").iterrows()
        ]
        values = untargeted.sort_values(["mode", "threshold"], na_position="first")["accuracy"].tolist()
        bars = axes[0, 0].bar(labels, values, color=COLORS["pgd"], width=0.72)
        annotate_bars(axes[0, 0], bars, values)
    axes[0, 0].set_title("Untargeted PGD Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    plt.setp(axes[0, 0].get_xticklabels(), rotation=12)
    style_axes(axes[0, 0], grid_axis="y")

    if not wrong_exit.empty:
        wrong_exit = wrong_exit.sort_values("threshold")
        labels = [f"theta={float(v):.2f}" for v in wrong_exit["threshold"]]
        values = wrong_exit["accuracy"].tolist()
        bars = axes[0, 1].bar(labels, values, color=COLORS["green"], width=0.72)
        annotate_bars(axes[0, 1], bars, values)
    axes[0, 1].set_title("Wrong-Early-Exit PGD")
    plt.setp(axes[0, 1].get_xticklabels(), rotation=12)
    style_axes(axes[0, 1], grid_axis="y")

    depth_conditions = []
    for label, objective, color in [
        ("Clean", None, COLORS["fixed"]),
        ("Untargeted", "untargeted", COLORS["pgd"]),
        ("Wrong exit", "wrong_early_exit", COLORS["green"]),
    ]:
        if objective is None:
            temp = panel[["threshold", "avg_stopping_depth"]].copy()
        else:
            temp = vt_only.loc[vt_only["attack_objective"] == objective, ["threshold", "avg_stopping_depth"]].copy()
        temp["condition"] = label
        temp["color"] = color
        depth_conditions.append(temp)
    depth_df = pd.concat(depth_conditions, ignore_index=True)

    for label, color in [("Clean", COLORS["fixed"]), ("Untargeted", COLORS["pgd"]), ("Wrong exit", COLORS["green"])]:
        view = depth_df.loc[depth_df["condition"] == label].sort_values("threshold")
        if not view.empty:
            axes[1, 0].plot(
                view["threshold"],
                view["avg_stopping_depth"],
                marker="o",
                linewidth=2.2,
                color=color,
                label=label,
            )
    axes[1, 0].set_title("Stopping Depth Under Attack")
    axes[1, 0].set_xlabel("VT Threshold")
    axes[1, 0].set_ylabel("Average Stopping Depth")
    axes[1, 0].legend(loc="upper left")
    style_axes(axes[1, 0], grid_axis="both")

    if not autoattack.empty:
        autoattack = autoattack.sort_values(["mode", "threshold"], na_position="first")
        labels = [
            "Fixed T=8" if pd.isna(row["threshold"]) else f"theta={float(row['threshold']):.2f}"
            for _, row in autoattack.iterrows()
        ]
        values = autoattack["accuracy"].tolist()
        colors = [COLORS["auto"] if pd.isna(row["threshold"]) else COLORS["brown"] for _, row in autoattack.iterrows()]
        bars = axes[1, 1].bar(labels, values, color=colors, width=0.72)
        annotate_bars(axes[1, 1], bars, values)
    axes[1, 1].set_title("AutoAttack Accuracy")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Accuracy (%)")
    plt.setp(axes[1, 1].get_xticklabels(), rotation=12)
    style_axes(axes[1, 1], grid_axis="y")

    save_current_figure(folder, "fig_attack_compute_savings")


def plot_fig5_compute_overuse(folder: Path) -> None:
    attack_df = load_csv(folder, "attack_results.csv").copy()
    panel = selected_panel_df(folder).copy()
    clean_lookup = panel.set_index("threshold")[["accuracy", "avg_stopping_depth"]].to_dict("index")

    overuse = attack_df.loc[attack_df["attack_objective"] == "compute_overuse"].copy().sort_values("threshold")
    overuse["clean_accuracy"] = overuse["threshold"].map(lambda t: clean_lookup[float(t)]["accuracy"])
    overuse["clean_depth"] = overuse["threshold"].map(lambda t: clean_lookup[float(t)]["avg_stopping_depth"])
    overuse["depth_increase"] = overuse["avg_stopping_depth"] - overuse["clean_depth"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    labels = [f"theta={float(t):.2f}" for t in overuse["threshold"]]
    x = range(len(labels))
    width = 0.38

    bars0 = axes[0].bar([i - width / 2 for i in x], overuse["clean_accuracy"], width=width, color=COLORS["fixed"], label="Clean")
    bars1 = axes[0].bar([i + width / 2 for i in x], overuse["accuracy"], width=width, color=COLORS["compute"], label="Compute-overuse")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=12)
    axes[0].set_title("Accuracy Under Compute-Overuse Attack")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper left")
    style_axes(axes[0], grid_axis="y")

    bars2 = axes[1].bar(labels, overuse["depth_increase"], color=COLORS["compute"], width=0.72)
    annotate_bars(axes[1], bars2, overuse["depth_increase"], fmt="+{:.2f}")
    axes[1].set_title("Increase in Stopping Depth")
    axes[1].set_ylabel("Delta Depth")
    plt.setp(axes[1].get_xticklabels(), rotation=12)
    style_axes(axes[1], grid_axis="y")

    save_current_figure(folder, "fig5_compute_overuse")


def generate_for_folder(folder: Path) -> None:
    print(f"Generating figures for {folder.name}")
    plot_fig1_fixed_t_sweep(folder)
    plot_fig2_vt_threshold_sweep(folder)
    plot_fig_clean_compute_savings(folder)
    plot_fig6_stopping_rules(folder)
    plot_fig_attack_compute_savings(folder)
    plot_fig5_compute_overuse(folder)


def main() -> None:
    configure_style()
    for folder in TARGETS:
        generate_for_folder(folder)


if __name__ == "__main__":
    main()
