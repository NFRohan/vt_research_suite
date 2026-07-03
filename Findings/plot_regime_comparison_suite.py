from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "ResNet Clean Trained"
ADV_DIR = ROOT / "ResNet Adversarial Trained"
OUT_DIR = ROOT / "Comparison Figures"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "clean": "#4C72B0",
    "adv": "#C44E52",
    "fixed": "#4C72B0",
    "matched": "#55A868",
    "vt": "#DD8452",
    "pgd": "#4C72B0",
    "auto": "#C44E52",
    "brown": "#937860",
    "purple": "#8172B2",
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


def save_current_figure(stem: str) -> None:
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


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


def short_rule(name: str) -> str:
    return {
        "max_softmax": "Max softmax",
        "margin": "Margin",
        "entropy": "Entropy",
        "temperature_softmax": "Temp-softmax",
        "evidential_surrogate": "Evidential",
    }.get(name, name)


def selected_threshold(folder: Path) -> float:
    selected_df = load_csv(folder, "main_selected_thresholds.csv")
    return float(selected_df.loc[selected_df["threshold_role"] == "selected", "threshold"].iloc[0])


def panel_rows(folder: Path) -> pd.DataFrame:
    return load_csv(folder, "test_selected_threshold_results.csv").sort_values("threshold")


def plot_fig1_regime_fixed_t_sweep() -> None:
    clean_df = load_csv(CLEAN_DIR, "test_fixed_t_sweep.csv").sort_values("time_steps")
    adv_df = load_csv(ADV_DIR, "test_fixed_t_sweep.csv").sort_values("time_steps")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    axes[0].plot(clean_df["time_steps"], clean_df["accuracy"], marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[0].plot(adv_df["time_steps"], adv_df["accuracy"], marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[0].set_title("Fixed-T Clean Accuracy")
    axes[0].set_xlabel("Fixed Timestep Budget")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(clean_df["time_steps"].tolist())
    axes[0].legend(loc="lower right")
    style_axes(axes[0], grid_axis="both")

    axes[1].plot(clean_df["time_steps"], clean_df["avg_synops"] / 1e6, marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[1].plot(adv_df["time_steps"], adv_df["avg_synops"] / 1e6, marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[1].set_title("Fixed-T Compute Growth")
    axes[1].set_xlabel("Fixed Timestep Budget")
    axes[1].set_ylabel("Average SynOps (Millions)")
    axes[1].set_xticks(clean_df["time_steps"].tolist())
    axes[1].legend(loc="upper left")
    style_axes(axes[1], grid_axis="both")

    save_current_figure("fig1_regime_fixed_t_sweep")


def plot_fig2_regime_vt_threshold_sweep() -> None:
    clean_df = load_csv(CLEAN_DIR, "test_vt_threshold_sweep.csv").sort_values("threshold")
    adv_df = load_csv(ADV_DIR, "test_vt_threshold_sweep.csv").sort_values("threshold")
    clean_theta = selected_threshold(CLEAN_DIR)
    adv_theta = selected_threshold(ADV_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    axes[0].plot(clean_df["threshold"], clean_df["accuracy"], marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[0].plot(adv_df["threshold"], adv_df["accuracy"], marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[0].axvline(clean_theta, color=COLORS["clean"], linestyle="--", linewidth=1.2)
    axes[0].axvline(adv_theta, color=COLORS["adv"], linestyle="--", linewidth=1.2)
    axes[0].set_title("VT Accuracy Frontier")
    axes[0].set_xlabel("VT Threshold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right")
    style_axes(axes[0], grid_axis="both")

    axes[1].plot(clean_df["threshold"], clean_df["avg_stopping_depth"], marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[1].plot(adv_df["threshold"], adv_df["avg_stopping_depth"], marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[1].axvline(clean_theta, color=COLORS["clean"], linestyle="--", linewidth=1.2)
    axes[1].axvline(adv_theta, color=COLORS["adv"], linestyle="--", linewidth=1.2)
    axes[1].set_title("VT Stopping Depth Frontier")
    axes[1].set_xlabel("VT Threshold")
    axes[1].set_ylabel("Average Stopping Depth")
    axes[1].legend(loc="upper left")
    style_axes(axes[1], grid_axis="both")

    save_current_figure("fig2_regime_vt_threshold_sweep")


def plot_fig3_regime_selected_points() -> None:
    clean_main = load_csv(CLEAN_DIR, "main_test_results.csv")
    adv_main = load_csv(ADV_DIR, "main_test_results.csv")

    def extract_triplet(df: pd.DataFrame) -> tuple[list[float], list[float]]:
        fixed_full = df.loc[df["report_mode"] == "fixed_full_horizon"].iloc[0]
        fixed_match = df.loc[df["report_mode"] == "fixed_compute_matched"].iloc[0]
        vt_sel = df.loc[df["report_mode"] == "vt_selected_threshold"].iloc[0]
        return (
            [float(fixed_full["accuracy"]), float(fixed_match["accuracy"]), float(vt_sel["accuracy"])],
            [float(fixed_full["avg_synops"]) / 1e6, float(fixed_match["avg_synops"]) / 1e6, float(vt_sel["avg_synops"]) / 1e6],
        )

    clean_acc, clean_syn = extract_triplet(clean_main)
    adv_acc, adv_syn = extract_triplet(adv_main)

    labels = ["Fixed T=8", "Matched Fixed", "VT selected"]
    x = range(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    bars0 = axes[0].bar([i - width / 2 for i in x], clean_acc, width=width, color=COLORS["clean"], label="Clean-trained")
    bars1 = axes[0].bar([i + width / 2 for i in x], adv_acc, width=width, color=COLORS["adv"], label="Adv-trained")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Selected Operating-Point Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper right")
    annotate_bars(axes[0], bars0, clean_acc)
    annotate_bars(axes[0], bars1, adv_acc)
    style_axes(axes[0], grid_axis="y")

    bars2 = axes[1].bar([i - width / 2 for i in x], clean_syn, width=width, color=COLORS["clean"], label="Clean-trained")
    bars3 = axes[1].bar([i + width / 2 for i in x], adv_syn, width=width, color=COLORS["adv"], label="Adv-trained")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Selected Operating-Point Compute")
    axes[1].set_ylabel("Average SynOps (Millions)")
    axes[1].legend(loc="upper right")
    annotate_bars(axes[1], bars2, clean_syn, fmt="{:.1f}")
    annotate_bars(axes[1], bars3, adv_syn, fmt="{:.1f}")
    style_axes(axes[1], grid_axis="y")

    save_current_figure("fig3_regime_selected_points")


def plot_fig4_regime_stopping_rules() -> None:
    clean_df = load_csv(CLEAN_DIR, "stopping_rule_test_results.csv")
    adv_df = load_csv(ADV_DIR, "stopping_rule_test_results.csv")

    clean_rules = clean_df.loc[clean_df["mode"] == "vt"].copy()
    adv_rules = adv_df.loc[adv_df["mode"] == "vt"].copy()
    clean_rules["rule_label"] = clean_rules["stop_rule"].map(short_rule)
    adv_rules["rule_label"] = adv_rules["stop_rule"].map(short_rule)

    merged = clean_rules[["rule_label", "accuracy", "avg_stopping_depth"]].merge(
        adv_rules[["rule_label", "accuracy", "avg_stopping_depth"]],
        on="rule_label",
        suffixes=("_clean", "_adv"),
    )

    labels = merged["rule_label"].tolist()
    x = range(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    bars0 = axes[0].bar([i - width / 2 for i in x], merged["accuracy_clean"], width=width, color=COLORS["clean"], label="Clean-trained")
    bars1 = axes[0].bar([i + width / 2 for i in x], merged["accuracy_adv"], width=width, color=COLORS["adv"], label="Adv-trained")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_title("Stopping Rules: Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper right")
    style_axes(axes[0], grid_axis="y")

    bars2 = axes[1].bar([i - width / 2 for i in x], merged["avg_stopping_depth_clean"], width=width, color=COLORS["clean"], label="Clean-trained")
    bars3 = axes[1].bar([i + width / 2 for i in x], merged["avg_stopping_depth_adv"], width=width, color=COLORS["adv"], label="Adv-trained")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_title("Stopping Rules: Avg. Depth")
    axes[1].set_ylabel("Average Stopping Depth")
    axes[1].legend(loc="upper right")
    style_axes(axes[1], grid_axis="y")

    save_current_figure("fig4_regime_stopping_rules")


def plot_fig5_regime_robustness() -> None:
    clean_attack = load_csv(CLEAN_DIR, "attack_results.csv")
    adv_attack = load_csv(ADV_DIR, "attack_results.csv")
    adv_subset = load_csv(ADV_DIR, "subset_consistent_selected_attack_comparison.csv")
    adv_sweep = load_csv(ADV_DIR, "subset_consistent_fixed_t_and_vt_attack_sweep.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # Clean-trained full-test selected comparison
    clean_view = clean_attack.loc[
        (clean_attack["attack_objective"] == "untargeted")
        & (clean_attack["attack_name"].isin(["pgd", "autoattack"]))
        & (
            ((clean_attack["mode"] == "fixed") & (clean_attack["threshold_role"] == "fixed_T8_baseline"))
            | ((clean_attack["mode"] == "vt") & (clean_attack["threshold_role"] == "selected"))
        )
    ].copy()
    clean_summary = (
        clean_view.pivot(index="attack_name", columns="mode", values="accuracy")
        .rename(index={"pgd": "PGD", "autoattack": "AutoAttack"}, columns={"fixed": "Fixed T=8", "vt": "VT selected"})
        .loc[["PGD", "AutoAttack"]]
    )
    x = range(len(clean_summary.index))
    width = 0.36
    bars0 = axes[0, 0].bar([i - width / 2 for i in x], clean_summary["Fixed T=8"], width=width, color=COLORS["fixed"], label="Fixed T=8")
    bars1 = axes[0, 0].bar([i + width / 2 for i in x], clean_summary["VT selected"], width=width, color=COLORS["vt"], label="VT selected")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(clean_summary.index.tolist())
    axes[0, 0].set_title("Clean-Trained: Full-Test Robustness")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(loc="upper right")
    style_axes(axes[0, 0], grid_axis="y")

    # Adv-trained full-test selected comparison
    adv_view = adv_attack.loc[
        (adv_attack["attack_objective"] == "untargeted")
        & (adv_attack["attack_name"].isin(["pgd", "autoattack"]))
        & (
            ((adv_attack["mode"] == "fixed") & (adv_attack["threshold_role"] == "fixed_T8_baseline"))
            | ((adv_attack["mode"] == "vt") & (adv_attack["threshold_role"] == "selected"))
        )
    ].copy()
    adv_summary = (
        adv_view.pivot(index="attack_name", columns="mode", values="accuracy")
        .rename(index={"pgd": "PGD", "autoattack": "AutoAttack"}, columns={"fixed": "Fixed T=8", "vt": "VT selected"})
        .loc[["PGD", "AutoAttack"]]
    )
    bars2 = axes[0, 1].bar([i - width / 2 for i in x], adv_summary["Fixed T=8"], width=width, color=COLORS["fixed"], label="Fixed T=8")
    bars3 = axes[0, 1].bar([i + width / 2 for i in x], adv_summary["VT selected"], width=width, color=COLORS["vt"], label="VT selected")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(adv_summary.index.tolist())
    axes[0, 1].set_title("Adv-Trained: Full-Test Robustness")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend(loc="upper right")
    style_axes(axes[0, 1], grid_axis="y")

    # Adv-trained 1k subset fair selected comparison
    subset_summary = (
        adv_subset.pivot(index="attack_name", columns="comparison_family", values="accuracy")
        .rename(
            index={"pgd": "PGD", "autoattack": "AutoAttack"},
            columns={
                "fixed_T8_subset": "Fixed T=8",
                "matched_fixed_control": "Matched Fixed T=4",
                "vt_selected_subset": "VT selected",
            },
        )
        .loc[["PGD", "AutoAttack"]]
    )
    x2 = range(len(subset_summary.index))
    width2 = 0.24
    bars4 = axes[1, 0].bar([i - width2 for i in x2], subset_summary["Fixed T=8"], width=width2, color=COLORS["fixed"], label="Fixed T=8")
    bars5 = axes[1, 0].bar([i for i in x2], subset_summary["Matched Fixed T=4"], width=width2, color=COLORS["matched"], label="Matched Fixed T=4")
    bars6 = axes[1, 0].bar([i + width2 for i in x2], subset_summary["VT selected"], width=width2, color=COLORS["vt"], label="VT selected")
    axes[1, 0].set_xticks(list(x2))
    axes[1, 0].set_xticklabels(subset_summary.index.tolist())
    axes[1, 0].set_title("Adv-Trained: Fair 1k Subset Comparison")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend(loc="upper right")
    style_axes(axes[1, 0], grid_axis="y")

    # Adv-trained 1k subset sweep summary
    fixed_sweep = adv_sweep.loc[adv_sweep["comparison_family"] == "fixed_t_sweep_subset"].copy()
    vt_sweep = adv_sweep.loc[adv_sweep["comparison_family"] == "vt_threshold_subset"].copy()
    sweep_rows = []
    for attack_name, attack_label in [("pgd", "PGD"), ("autoattack", "AutoAttack")]:
        fixed_slice = fixed_sweep.loc[fixed_sweep["attack_name"] == attack_name]
        vt_slice = vt_sweep.loc[vt_sweep["attack_name"] == attack_name]
        best_fixed = fixed_slice.loc[fixed_slice["accuracy"].idxmax()]
        best_vt = vt_slice.loc[vt_slice["accuracy"].idxmax()]
        selected_vt = vt_slice.loc[vt_slice["threshold_role"] == "selected"].iloc[0]
        sweep_rows.append(
            {
                "Attack": attack_label,
                "Best fixed": float(best_fixed["accuracy"]),
                "Best VT": float(best_vt["accuracy"]),
                "Selected VT": float(selected_vt["accuracy"]),
            }
        )
    sweep_summary = pd.DataFrame(sweep_rows).set_index("Attack")
    x3 = range(len(sweep_summary.index))
    bars7 = axes[1, 1].bar([i - width2 for i in x3], sweep_summary["Best fixed"], width=width2, color=COLORS["fixed"], label="Best fixed-T")
    bars8 = axes[1, 1].bar([i for i in x3], sweep_summary["Best VT"], width=width2, color=COLORS["brown"], label="Best VT")
    bars9 = axes[1, 1].bar([i + width2 for i in x3], sweep_summary["Selected VT"], width=width2, color=COLORS["vt"], label="Selected VT")
    axes[1, 1].set_xticks(list(x3))
    axes[1, 1].set_xticklabels(sweep_summary.index.tolist())
    axes[1, 1].set_title("Adv-Trained: 1k Sweep Summary")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend(loc="upper right")
    style_axes(axes[1, 1], grid_axis="y")

    save_current_figure("fig5_regime_robustness")


def plot_fig6_regime_compute_overuse() -> None:
    clean_attack = load_csv(CLEAN_DIR, "attack_results.csv")
    adv_attack = load_csv(ADV_DIR, "attack_results.csv")
    clean_panel = panel_rows(CLEAN_DIR)
    adv_panel = panel_rows(ADV_DIR)

    def prepare(folder_attack: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        overuse = folder_attack.loc[folder_attack["attack_objective"] == "compute_overuse"].copy().sort_values("threshold")
        clean_lookup = panel.set_index("threshold")[["accuracy", "avg_stopping_depth"]].to_dict("index")
        overuse["clean_accuracy"] = overuse["threshold"].map(lambda t: clean_lookup[float(t)]["accuracy"])
        overuse["depth_increase"] = overuse["avg_stopping_depth"] - overuse["threshold"].map(
            lambda t: clean_lookup[float(t)]["avg_stopping_depth"]
        )
        return overuse

    clean_over = prepare(clean_attack, clean_panel)
    adv_over = prepare(adv_attack, adv_panel)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    axes[0].plot(clean_over["threshold"], clean_over["depth_increase"], marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[0].plot(adv_over["threshold"], adv_over["depth_increase"], marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[0].set_title("Compute-Overuse: Depth Increase")
    axes[0].set_xlabel("VT Threshold")
    axes[0].set_ylabel("Increase in Avg. Stopping Depth")
    axes[0].legend(loc="upper left")
    style_axes(axes[0], grid_axis="both")

    axes[1].plot(clean_over["threshold"], clean_over["accuracy"], marker="o", linewidth=2.4, color=COLORS["clean"], label="Clean-trained")
    axes[1].plot(adv_over["threshold"], adv_over["accuracy"], marker="s", linewidth=2.4, color=COLORS["adv"], label="Adv-trained")
    axes[1].set_title("Compute-Overuse: Attacked Accuracy")
    axes[1].set_xlabel("VT Threshold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(loc="upper left")
    style_axes(axes[1], grid_axis="both")

    save_current_figure("fig6_regime_compute_overuse")


def main() -> None:
    configure_style()
    plot_fig1_regime_fixed_t_sweep()
    plot_fig2_regime_vt_threshold_sweep()
    plot_fig3_regime_selected_points()
    plot_fig4_regime_stopping_rules()
    plot_fig5_regime_robustness()
    plot_fig6_regime_compute_overuse()


if __name__ == "__main__":
    main()
