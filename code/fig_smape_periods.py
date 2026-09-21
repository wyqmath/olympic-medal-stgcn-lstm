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
import pandas as pd

PERIOD_LABELS = [
    ("Cold War", "1976-88\neditions"),
    ("Globalization", "1992-2016\neditions"),
    ("COVID-19", "2020-24\neditions"),
]
C_MODEL = "#0072B2"
C_PERSIST = "#E69F00"


def read_weighted(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).set_index("period")
    out = {}
    for p in df.index:
        row = df.loc[p]
        out[p] = (float(row["persistence_weighted"]),
                  float(row["persistence_weighted_ci95_low"]),
                  float(row["persistence_weighted_ci95_high"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gen/nested/backtest/period_summary.csv"))
    ap.add_argument("--persist-weighted", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gen/nested/backtest/persistence_weighted.csv"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gen/fig_smape_periods.png"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    s = pd.read_csv(args.summary).set_index("period")
    pw = read_weighted(args.persist_weighted)

    periods = [p for p, _ in PERIOD_LABELS if p in s.index]
    labels = [lab for p, lab in PERIOD_LABELS if p in s.index]

    model = {
        "macro": s.loc[periods, "smape_macro"].to_numpy(float),
        "macro_ci": s.loc[periods, ["smape_macro_ci95_low", "smape_macro_ci95_high"]].to_numpy(float).T,
        "weighted": s.loc[periods, "smape_weighted"].to_numpy(float),
        "weighted_ci": s.loc[periods, ["smape_weighted_ci95_low", "smape_weighted_ci95_high"]].to_numpy(float).T,
    }
    persist = {
        "macro": s.loc[periods, "persistence_macro"].to_numpy(float),
        "macro_ci": s.loc[periods, ["persistence_ci95_low", "persistence_ci95_high"]].to_numpy(float).T,
        "weighted": np.array([pw.get(p, (np.nan,) * 3)[0] for p in periods]),
        "weighted_ci": np.array([[pw.get(p, (np.nan,) * 3)[1], pw.get(p, (np.nan,) * 3)[2]]
                                 for p in periods]).T,
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    x = np.arange(len(periods))
    w = 0.36
    for ax, reading in zip(axes, ("macro", "weighted")):
        title = "SMAPE (macro average over Games)" if reading == "macro" \
            else "SMAPE (medal-weighted)"
        m, mci = model[reading], model[f"{reading}_ci"]
        p, pci = persist[reading], persist[f"{reading}_ci"]
        err_m = np.vstack([m - mci[0], mci[1] - m])
        err_p = np.vstack([p - pci[0], pci[1] - p])
        b1 = ax.bar(x - w / 2, m, w, yerr=err_m, capsize=3, color=C_MODEL,
                    label="STGCN-LSTM (nested)", error_kw=dict(lw=0.9))
        b2 = ax.bar(x + w / 2, p, w, yerr=err_p, capsize=3, color=C_PERSIST,
                    label="Persistence", error_kw=dict(lw=0.9))
        for bars, vals, cis in ((b1, m, mci), (b2, p, pci)):
            for rect, v, (lo, hi) in zip(bars, vals, cis.T):
                ax.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, max(v, hi)),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 3),
                            textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("SMAPE (%)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(np.nanmax(mci[1]), np.nanmax(pci[1])) * 1.22)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
