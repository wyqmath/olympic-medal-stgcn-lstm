from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import os as _os
import glob as _glob
import matplotlib.font_manager as _fm
for _d in (_os.environ.get("OLYMPIC_FONT_DIR", ""),
           "/usr/share/fonts/truetype/msttcorefonts", "/usr/share/fonts",
           "/Library/Fonts", "C:/Windows/Fonts"):
    if _d and _os.path.isdir(_d):
        for _p in sorted(_glob.glob(_os.path.join(_d, "[Aa]rial*.ttf"))):
            try:
                _fm.fontManager.addfont(_p)
            except Exception:
                pass
try:
    _fm.findfont("Arial", fallback_to_default=False)
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Arial"
    matplotlib.rcParams["mathtext.it"] = "Arial:italic"
    matplotlib.rcParams["mathtext.bf"] = "Arial:bold"
except ValueError:
    pass
import matplotlib.pyplot as plt
import numpy as np

COUNTRIES = ["USA", "CHN", "JPN", "AUS", "FRA", "NED", "UK", "KOR"]
GOLD = [40, 40, 20, 18, 16, 15, 14, 13]
SILVER = [44, 27, 12, 19, 26, 7, 22, 9]
BRONZE = [42, 24, 13, 16, 22, 12, 29, 10]

ROWS = [
    ("GOLD", GOLD, "#C9A227"),
    ("SILVER", SILVER, "#9EA0A1"),
    ("BRONZE", BRONZE, "#8C564B"),
    ("TOTAL", None, "#0072B2"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(here, "gen/fig_medal2024.png"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    total = [g + s + b for g, s, b in zip(GOLD, SILVER, BRONZE)]
    assert total == [126, 91, 45, 53, 64, 34, 65, 32], "medal table checksum failed"

    x = np.arange(len(COUNTRIES))
    w = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i, (name, vals, color) in enumerate(ROWS):
        v = total if vals is None else vals
        bars = ax.bar(x + (i - 1.5) * w, v, w, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fontsize=7.2, padding=1.2, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(COUNTRIES, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.set_ylabel("Medals won", fontsize=9)
    ax.set_ylim(0, 142)
    ax.yaxis.grid(True, color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False,
              handlelength=1.2, handleheight=1.0, columnspacing=1.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")

    fig.subplots_adjust(bottom=0.11, left=0.055, right=0.99, top=0.97)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
