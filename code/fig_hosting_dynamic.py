import json
from pathlib import Path

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

HERE = Path(__file__).resolve().parent
SRC = HERE / "gen" / "econometrics_host_dynamic_results.json"
OUT = HERE / "gen" / "fig_hosting_dynamic.png"

C_POINT = "#0072B2"
C_REF = "#999999"
C_BAND = "#56B4E9"
C_ACCENT = "#D55E00"


def main():
    data = json.loads(SRC.read_text())
    path = data["path"]
    k = [d["k"] for d in path]
    delta = [d["delta"] for d in path]
    lo = [d["ci_lo"] if d["ci_lo"] is not None else d["delta"] for d in path]
    hi = [d["ci_hi"] if d["ci_hi"] is not None else d["delta"] for d in path]
    is_ref = [d.get("is_reference", False) for d in path]
    p = [d.get("p") for d in path]

    err_lo = [d - l for d, l in zip(delta, lo)]
    err_hi = [h - d for d, h in zip(delta, hi)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.fill_between(k, lo, hi, color=C_BAND, alpha=0.20, linewidth=0, zorder=1)

    for ki, di, ei_lo, ei_hi, ref, pi in zip(k, delta, err_lo, err_hi, is_ref, p):
        if ref:
            ax.plot(ki, di, marker="o", ms=7, mfc="white", mec=C_REF, mew=1.6,
                    zorder=3, linestyle="none")
        else:
            star = "**" if (pi is not None and pi < 0.01) else ("*" if (pi is not None and pi < 0.05) else "")
            ax.errorbar(ki, di, yerr=[[ei_lo], [ei_hi]], fmt="o", ms=5.5,
                        color=C_POINT, ecolor=C_POINT, elinewidth=1.4, capsize=3.5,
                        zorder=3)
            if star:
                ax.annotate(star, (ki, hi[k.index(ki)]), textcoords="offset points",
                            xytext=(4, 2), fontsize=9, color=C_POINT)

    ax.axhline(0, color="#444444", linewidth=0.9, linestyle="-", zorder=2)
    ax.axvline(0, color=C_ACCENT, linewidth=1.4, linestyle="--", zorder=2)

    ax.set_xticks(k)
    ax.set_xticklabels([("$_{%d}$" % ki if ki < 0 else "%d" % ki) + "\n(%+d yr)" % (4 * ki)
                        for ki in k], fontsize=8.5)
    ax.set_xlabel("Event time $k$ relative to the hosting Games (1 unit = 4 years)")
    ax.set_ylabel(r"Hosting effect $\delta_k$ on log$(1+$medals$)$")
    ax.set_xlim(-3.7, 5.7)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)


    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
