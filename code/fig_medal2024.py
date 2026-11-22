import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def save(fig, path, *, crop=True):
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight" if crop else None)
    plt.close(fig)
    print(path)


def draw_paris_medals(panel, path):
    medals = (
        panel.loc[panel["year"].eq(2024)]
        .sort_values(["gold", "silver", "bronze"], ascending=False)
        .head(8)
    )
    series = (
        ("Gold", "gold", "#C9A227"),
        ("Silver", "silver", "#9EA0A1"),
        ("Bronze", "bronze", "#8C564B"),
        ("Total", "total", "#0072B2"),
    )
    positions = np.arange(len(medals))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, (label, column, color) in enumerate(series):
        bars = ax.bar(
            positions + (i - 1.5) * width, medals[column], width, label=label,
            color=color, edgecolor="white", linewidth=0.5,
        )
        ax.bar_label(bars, fontsize=7.2, padding=1.2, color="#333333")
    ax.set_xticks(positions, medals["ioc"], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.set_ylabel("Medals won")
    ax.set_ylim(0, 142)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(ncol=4, loc="upper right", frameon=False, fontsize=8, handlelength=1.2)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    fig.subplots_adjust(bottom=0.11, left=0.055, right=0.99, top=0.97)
    save(fig, path, crop=False)


def main():
    parser = argparse.ArgumentParser(description="Plot the Paris 2024 medal totals.")
    parser.add_argument("--out", type=Path, default=HERE / "gen" / "fig_medal2024.png")
    output = parser.parse_args().out
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(HERE.parent / "data" / "panel_clean.csv")
    with plt.rc_context(STYLE):
        draw_paris_medals(panel, output)


if __name__ == "__main__":
    main()
