import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SRC = HERE / "gen" / "econometrics_var_decomp_results.json"
OUT = HERE / "gen" / "fig_var_decomp.png"

COLORS = {"Econ": "#0072B2", "Inst": "#E69F00", "Rand": "#999999"}
LABELS = {"Econ": "Economic block (log GDP, log population)",
          "Inst": "Institutional block (hosting)",
          "Rand": "Residual / random"}


def main():
    data = json.loads(SRC.read_text())
    sh = data["D1_shares_excl_FE"]["shares"]
    order = ["Econ", "Inst", "Rand"]
    vals = [sh[k] for k in order]

    fig = plt.figure(figsize=(7.2, 2.5))
    ax = fig.add_axes([0.07, 0.62, 0.88, 0.28])
    left = 0.0
    for key, v in zip(order, vals):
        ax.barh([0], [v], left=left, color=COLORS[key], edgecolor="white",
                linewidth=1.2, height=0.55)
        txt_color = "white" if key == "Econ" else "#222222"
        ax.annotate(f"{v*100:.1f}%", (left + v / 2, 0), ha="center", va="center",
                    fontsize=10.5, color=txt_color, fontweight="bold")
        left += v

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_xlabel("Share of explained + residual variation in log(1 + medals)",
                  fontsize=9.5, labelpad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.legend(handles=[Patch(facecolor=COLORS[k], label=LABELS[k]) for k in order],
               loc="center", bbox_to_anchor=(0.5, 0.34), ncol=1, frameon=False,
               fontsize=8.8, handlelength=1.4)

    s = data["sample"]
    note = (f"OLS on {s['n_rows_estimated']} country-Games ({s['n_games']} Games, "
            f"{s['n_countries']} countries), country + Games FE, SE clustered by country;"
            f" fixed-effect variance excluded from the denominator")
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=7.0, color="#555555",
             wrap=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
