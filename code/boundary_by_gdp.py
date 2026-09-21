import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions",
                    default=os.path.join(HERE, "gen", "nested", "predictions_nested.csv"))
    ap.add_argument("--panel",
                    default=os.path.join(HERE, os.pardir, "data", "panel_clean.csv"))
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    panel = pd.read_csv(args.panel)
    m = pred.merge(panel[["ioc", "year", "gdp_total_2011usd_million"]],
                   on=["ioc", "year"], how="left")
    n_all, n_missing = len(m), int(m["gdp_total_2011usd_million"].isna().sum())
    m = m.dropna(subset=["gdp_total_2011usd_million"]).copy()
    m["smape_row"] = 200.0 * (m["actual"] - m["predicted"]).abs() / (m["actual"] + m["predicted"])
    m["tercile"] = pd.qcut(m["gdp_total_2011usd_million"], 3,
                           labels=["low", "mid", "high"])

    rows = []
    for t in ["low", "mid", "high"]:
        sub = m[m["tercile"] == t]
        rows.append({
            "gdp_tercile": t,
            "n": len(sub),
            "macro_smape": round(float(sub["smape_row"].mean()), 1),
            "weighted_smape": round(float((sub["actual"] * sub["smape_row"]).sum()
                                          / sub["actual"].sum()), 1),
        })
    rows.append({
        "gdp_tercile": "all",
        "n": len(m),
        "macro_smape": round(float(m["smape_row"].mean()), 1),
        "weighted_smape": round(float((m["actual"] * m["smape_row"]).sum()
                                      / m["actual"].sum()), 1),
    })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"rows: {n_all} total, {n_missing} without macro coverage (excluded)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
