import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def draw_event_study(events, path):
    event_times = [event["k"] for event in events]
    estimates = [event["delta"] for event in events]
    lower = [event["ci_lo"] for event in events]
    upper = [event["ci_hi"] for event in events]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.fill_between(event_times, lower, upper, color="#56B4E9", alpha=0.20, linewidth=0)
    for event in events:
        k, estimate = event["k"], event["delta"]
        if event["is_reference"]:
            ax.plot(k, estimate, "o", ms=7, mfc="white", mec="#999999", mew=1.6, zorder=3)
            continue
        lo, hi = event["ci_lo"], event["ci_hi"]
        ax.errorbar(
            k, estimate, yerr=[[estimate - lo], [hi - estimate]], fmt="o", ms=5.5,
            color="#0072B2", elinewidth=1.4, capsize=3.5, zorder=3,
        )
        p = event["p"]
        if p < 0.05:
            ax.annotate(
                "**" if p < 0.01 else "*", (k, hi), xytext=(4, 2),
                textcoords="offset points", fontsize=9, color="#0072B2",
            )
    ax.axhline(0, color="#444444", linewidth=0.9, zorder=2)
    ax.axvline(0, color="#D55E00", linewidth=1.4, linestyle="--", zorder=2)
    labels = [f"{k}\n({4 * k:+d} yr)".replace("-", "\N{MINUS SIGN}") for k in event_times]
    labels[event_times.index(0)] = "0\n(0 yr)"
    ax.set_xticks(event_times, labels, fontsize=8.5)
    ax.set_xlabel("Event time in Olympic editions relative to hosting")
    ax.set_ylabel(r"Hosting coefficient $\delta_k$ on $\log(1+\mathrm{medals})$")
    ax.set_xlim(-3.7, 5.7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description="Plot the hosting event-study coefficients.")
    parser.add_argument("--out", type=Path, default=HERE / "gen" / "fig_hosting_dynamic.png")
    output = parser.parse_args().out
    output.parent.mkdir(parents=True, exist_ok=True)
    events = json.loads((HERE / "gen" / "econometrics_host_dynamic_results.json").read_text(encoding="utf-8"))
    with plt.rc_context(STYLE):
        draw_event_study(events["path"], output)


if __name__ == "__main__":
    main()
