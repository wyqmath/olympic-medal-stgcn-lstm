import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from evaluate_recent_editions import GEN, main as draw_recent_editions
from evaluate_tokyo_roc import REPORTED_CODE, YEAR
from score_period_forecasts import MATCHED_SUMMARY, write_period_summary


OUTPUT = GEN / "linked_backtest"
FIGURES = GEN


def draw_grid():
    predictions = pd.read_csv(GEN / "sensitivity" / "grid_predictions.csv")
    renamed = predictions["year"].eq(YEAR) & predictions["ioc"].eq(REPORTED_CODE)
    if renamed.sum() != 25:
        raise ValueError("The grid must have exactly one Tokyo linkage case per cell")
    stable = predictions.loc[~renamed].copy()
    stable["weighted_error"] = (
        stable["actual"] * 200 * (stable["actual"] - stable["predicted"]).abs()
        / (stable["actual"] + stable["predicted"])
    )
    grouped = stable.groupby(["param_lambda", "param_rho"])
    result = grouped.agg(n=("actual", "size"), medals=("actual", "sum"), weighted_error=("weighted_error", "sum"))
    if len(result) != 25 or not result["n"].eq(915).all():
        raise ValueError("The sensitivity grid does not share the same 915 observations")
    result["weighted_smape"] = result["weighted_error"] / result["medals"]
    result[["n", "weighted_smape"]].reset_index().to_csv(OUTPUT / "sensitivity_stable_codes.csv", index=False)
    heat = result["weighted_smape"].unstack("param_lambda")
    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    artist = ax.imshow(heat.to_numpy(), origin="lower", cmap="viridis", aspect="equal")
    ax.set_xticks(range(heat.shape[1]), [f"{value:g}" for value in heat.columns])
    ax.set_yticks(range(heat.shape[0]), [rf"$10^{{{round(math.log10(value))}}}$" for value in heat.index])
    ax.set_xlabel(r"Time-decay rate $\lambda$")
    ax.set_ylabel(r"Weight decay $\rho$")
    low, high = result["weighted_smape"].min(), result["weighted_smape"].max()
    for j, (_, row) in enumerate(heat.iterrows()):
        for i, value in enumerate(row):
            ax.text(i, j, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < low + 0.63 * (high - low) else "#151515", fontsize=9)
    x, y = heat.columns.get_loc(0.05), heat.index.get_loc(0.0001)
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                           edgecolor="black", linestyle="--", linewidth=1.6))
    fig.colorbar(artist, ax=ax, label="Pooled medal-weighted SMAPE (%)", shrink=0.88)
    fig.savefig(FIGURES / "fig_sensitivity_heatmap.png", dpi=300, facecolor="white")
    plt.close(fig)


def draw_period_figure():
    summary = pd.read_csv(MATCHED_SUMMARY).sort_values("start_year")
    labels = [f"{start}\u2013{end}" for start, end in zip(summary["start_year"], summary["end_year"])]
    positions = list(range(len(labels)))
    panels = (
        ("Macro-average SMAPE", (
            ("smape_macro", "smape_macro_ci95"),
            ("persistence_macro", "persistence_ci95"),
            ("hgbr_macro", "hgbr_macro_ci95"),
        )),
        ("Medal-weighted SMAPE", (
            ("smape_weighted", "smape_weighted_ci95"),
            ("persistence_weighted", "persistence_weighted_ci95"),
            ("hgbr_weighted", "hgbr_weighted_ci95"),
        )),
    )
    methods = (
        ("STGCN-LSTM", "#0072B2"),
        ("Persistence", "#E69F00"),
        ("HGBR", "#9EAFBF"),
    )

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]}):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
        for ax, (title, columns) in zip(axes, panels):
            for offset, (method, color), (point, interval) in zip((-0.245, 0, 0.245), methods, columns):
                values = summary[point]
                lower = summary[f"{interval}_low"]
                upper = summary[f"{interval}_high"]
                centers = [position + offset for position in positions]
                ax.bar(
                    centers, values, width=0.23, color=color, label=method,
                    yerr=[values - lower, upper - values], capsize=2.5,
                    error_kw={"elinewidth": 0.85, "capthick": 0.85},
                )
                for center, value, limit in zip(centers, values, upper):
                    ax.annotate(
                        f"{value:.1f}", (center, limit), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    )
            ax.set_title(title, fontsize=10.5, pad=7)
            ax.set_xticks(positions, labels, fontsize=9)
            ax.set_yticks(range(0, 71, 10))
            ax.set_ylim(0, 70)
            ax.set_ylabel("SMAPE (%)", fontsize=9.5)
            ax.tick_params(axis="y", labelleft=True, labelsize=9)
            ax.grid(axis="y", color="#D9DDE1", linewidth=0.55)
            ax.set_axisbelow(True)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
            ncol=3, frameon=False, fontsize=9.5, handlelength=1.4,
        )
        fig.subplots_adjust(left=0.075, right=0.986, bottom=0.13, top=0.82, wspace=0.26)
        fig.savefig(FIGURES / "average_smape_values.png", dpi=300, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw the backtest and sensitivity figures from the generated forecasts.")
    parser.add_argument("--period-only", action="store_true", help="draw the period comparison from saved scores")
    args = parser.parse_args()
    if not args.period_only:
        write_period_summary()
    draw_period_figure()
    if args.period_only:
        return
    draw_recent_editions([])
    draw_grid()


if __name__ == "__main__":
    main()
