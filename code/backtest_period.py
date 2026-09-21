from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

WINDOW_HALF_WIDTH = 5
DEFAULT_PERIODS = OrderedDict(
    [
        ("Cold War", (1976, 1991)),
        ("Globalization", (1992, 2016)),
        ("COVID-19", (2017, 2024)),
    ]
)
PAPER_REPORTED = {"Cold War": 21.4, "Globalization": 17.8, "COVID-19": 23.1}

COVID_ALTERNATIVES = [
    ("current_COVID-19_2017-2024", 2017, 2024,
     "paper's current definition (renaming candidate A: keep the number, "
     "relabel as 'Recent era' -- identical numerics, different label)"),
    ("true_pandemic_editions_2021-2024", 2021, 2024,
     "only editions held in/after the pandemic; with Tokyo recorded under its "
     "nominal year 2020 this retains Paris 2024 only, i.e. it tests whether "
     "the COVID-era figure is driven by the Tokyo edition alone"),
    ("recent_era_2017-2024", 2017, 2024,
     "renaming candidate B: same edition set and same value as the current "
     "definition, reported under a non-pandemic label ('Recent era')"),
]

DEFAULT_COLUMNS = dict(country="country", year="year",
                       actual="actual", predicted="predicted")


def smape_obs(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual) + np.abs(predicted)
    out = np.full(denom.shape, np.nan, dtype=float)
    ok = denom > 0
    out[ok] = 200.0 * np.abs(actual[ok] - predicted[ok]) / denom[ok]
    return out


def window_mask(years: np.ndarray, t: int,
                half_width: int = WINDOW_HALF_WIDTH) -> np.ndarray:
    return np.abs(years - t) <= half_width


def build_windows(editions: list[int], all_years: np.ndarray) -> dict[int, np.ndarray]:
    return {t: window_mask(all_years, t) for t in editions}


def macro_over_games(values: np.ndarray, masks: list[np.ndarray]) -> float:
    per_game = []
    for m in masks:
        v = values[m]
        v = v[np.isfinite(v)]
        if v.size:
            per_game.append(v.mean())
    return float(np.mean(per_game)) if per_game else float("nan")


def weighted_over_games(err: np.ndarray, weight: np.ndarray,
                        masks: list[np.ndarray]) -> float:
    per_game = []
    for m in masks:
        mm = m & np.isfinite(err) & (weight > 0)
        tot = weight[mm].sum()
        if tot > 0:
            per_game.append(float((weight[mm] * err[mm]).sum() / tot))
    return float(np.mean(per_game)) if per_game else float("nan")


def load_predictions(path: str, cols: dict[str, str] | None = None) -> pd.DataFrame:
    cols = {**DEFAULT_COLUMNS, **(cols or {})}
    df = pd.read_csv(path)
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {path} is missing required column(s): {missing}; "
                         f"found {list(df.columns)}")
    out = df.rename(columns={v: k for k, v in cols.items()})[
        ["country", "year", "actual", "predicted"]].copy()
    for c in ("year", "actual", "predicted"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    n_nan = int(out[["year", "actual", "predicted"]].isna().any(axis=1).sum())
    if n_nan:
        print(f"[warn] dropping {n_nan} row(s) with non-numeric/missing values")
        out = out.dropna(subset=["year", "actual", "predicted"])
    out["year"] = out["year"].astype(int)
    neg = int((out[["actual", "predicted"]] < 0).any(axis=1).sum())
    if neg:
        raise SystemExit(f"ERROR: {neg} row(s) have negative medal counts; "
                         "SMAPE on this protocol requires non-negative values.")
    out["country"] = out["country"].astype(str).str.strip()
    dup = int(out.duplicated(["country", "year"]).sum())
    if dup:
        raise SystemExit(f"ERROR: {dup} duplicated (country, year) row(s); the "
                         "protocol needs exactly one observation per NOC x Games.")
    return out.sort_values(["year", "country"]).reset_index(drop=True)


def add_persistence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["country", "year"]).copy()
    out["predicted_persist"] = (out.groupby("country")["actual"]
                                  .shift(1))
    out["persist_missing"] = out["predicted_persist"].isna()
    return out.sort_values(["year", "country"]).reset_index(drop=True)


def count_skips(df: pd.DataFrame) -> dict[str, int]:
    undefined = ~np.isfinite(smape_obs(df["actual"], df["predicted"]))
    return dict(
        n_rows=int(df.shape[0]),
        n_undefined_smape=int(undefined.sum()),
        n_persistence_missing=int(df["persist_missing"].sum()),
    )


def compute_games_table(df: pd.DataFrame) -> pd.DataFrame:
    years = df["year"].to_numpy()
    err = smape_obs(df["actual"].to_numpy(), df["predicted"].to_numpy())
    err_p = smape_obs(df["actual"].to_numpy(),
                      df["predicted_persist"].to_numpy())
    weight = df["actual"].to_numpy(dtype=float)
    p_missing = df["persist_missing"].to_numpy()
    editions = sorted(df["year"].unique().tolist())
    windows = build_windows(editions, years)
    period_of = {t: name for name, (lo, hi) in DEFAULT_PERIODS.items()
                 for t in editions if lo <= t <= hi}

    rows = []
    for t in editions:
        m = windows[t]
        valid = m & np.isfinite(err)
        wmask = valid & (weight > 0)
        w_total = float(weight[wmask].sum())
        ep_ok = m & np.isfinite(err_p) & ~p_missing
        rows.append(dict(
            year=t,
            window_games="|".join(str(w) for w in editions
                                  if abs(w - t) <= WINDOW_HALF_WIDTH),
            n_t=int(valid.sum()),
            n_skipped=int(m.sum()) - int(valid.sum()),
            smape=float(err[valid].mean()) if valid.any() else np.nan,
            weighted_smape=(float((weight[wmask] * err[wmask]).sum() / w_total)
                            if w_total > 0 else np.nan),
            weighted_total_medals=float(weight[m].sum()),
            persistence_smape=float(err_p[ep_ok].mean()) if ep_ok.any() else np.nan,
            persistence_n=int(ep_ok.sum()),
            persistence_missing=int(p_missing[m].sum()),
            period=period_of.get(t, ""),
        ))
    return pd.DataFrame(rows)


def period_games(editions: list[int], lo: int, hi: int) -> list[int]:
    return [t for t in editions if lo <= t <= hi]


def _period_statistic(df: pd.DataFrame, windows: dict[int, np.ndarray],
                      t_list: list[int], metric: str,
                      row_idx: np.ndarray | None = None) -> float:
    masks = ([windows[t] for t in t_list] if row_idx is None
             else [windows[t][row_idx] for t in t_list])
    actual = df["actual"].to_numpy(dtype=float)
    pred = df["predicted"].to_numpy(dtype=float)
    if row_idx is not None:
        actual, pred = actual[row_idx], pred[row_idx]
    if metric == "macro":
        return macro_over_games(smape_obs(actual, pred), masks)
    if metric == "weighted":
        return weighted_over_games(smape_obs(actual, pred), actual, masks)
    if metric == "persistence":
        pp = df["predicted_persist"].to_numpy(dtype=float)
        miss = df["persist_missing"].to_numpy()
        if row_idx is not None:
            pp, miss = pp[row_idx], miss[row_idx]
        return macro_over_games(np.where(miss, np.nan, smape_obs(actual, pp)),
                                masks)
    raise ValueError(f"unknown metric: {metric}")


def bootstrap_period(df: pd.DataFrame, windows: dict[int, np.ndarray],
                     t_list: list[int], n_boot: int, seed: int,
                     stream: int = 0) -> dict[str, tuple[float, float, int]]:
    countries = df["country"].unique()
    rows_by_country = [np.flatnonzero(df["country"].to_numpy() == c)
                       for c in countries]
    rng = np.random.default_rng([seed, stream])
    draws = {"macro": [], "weighted": [], "persistence": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_country), len(rows_by_country))
        idx = np.concatenate([rows_by_country[i] for i in pick])
        for metric in draws:
            draws[metric].append(
                _period_statistic(df, windows, t_list, metric, row_idx=idx))
    out = {}
    for metric, vals in draws.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[metric] = (float(np.percentile(vals, 2.5)),
                           float(np.percentile(vals, 97.5)), int(vals.size))
        else:
            out[metric] = (float("nan"), float("nan"), 0)
    return out


def summarise_periods(df: pd.DataFrame, periods: "OrderedDict[str, tuple[int, int]]",
                      n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    editions = sorted(df["year"].unique().tolist())
    windows = build_windows(editions, df["year"].to_numpy())
    records, payload = [], {}
    for p_i, (name, (lo, hi)) in enumerate(periods.items()):
        t_list = period_games(editions, lo, hi)
        macro = _period_statistic(df, windows, t_list, "macro")
        weighted = _period_statistic(df, windows, t_list, "weighted")
        persist = _period_statistic(df, windows, t_list, "persistence")
        if n_boot > 0:
            ci = bootstrap_period(df, windows, t_list, n_boot, seed, stream=p_i)
        else:
            ci = {k: (float("nan"), float("nan"), 0) for k in
                  ("macro", "weighted", "persistence")}
        in_period = df[df["year"].between(lo, hi)]
        skipped = int(np.sum(~np.isfinite(
            smape_obs(in_period["actual"], in_period["predicted"]))))
        rec = dict(
            period=name,
            start_year=lo,
            end_year=hi,
            games_years="|".join(str(t) for t in t_list),
            n_games=len(t_list),
            n_games_with_estimate=int(sum(
                np.isfinite(_period_statistic(df, windows, [t], "macro"))
                for t in t_list)),
            n_obs=int(in_period.shape[0]),
            n_skipped_zero_denominator=skipped,
            smape_macro=macro,
            smape_macro_ci95_low=ci["macro"][0],
            smape_macro_ci95_high=ci["macro"][1],
            smape_weighted=weighted,
            smape_weighted_ci95_low=ci["weighted"][0],
            smape_weighted_ci95_high=ci["weighted"][1],
            total_medals_in_period=float(in_period["actual"].sum()),
            persistence_macro=persist,
            persistence_ci95_low=ci["persistence"][0],
            persistence_ci95_high=ci["persistence"][1],
            persistence_n_missing=int(in_period["persist_missing"].sum()),
            n_bootstrap_replicates=ci["macro"][2],
        )
        records.append(rec)
        payload[name] = rec
    return pd.DataFrame(records), payload


def covid_alternatives_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    editions = sorted(df["year"].unique().tolist())
    windows = build_windows(editions, df["year"].to_numpy())
    rows = []
    for i, (label, lo, hi, note) in enumerate(COVID_ALTERNATIVES):
        t_list = period_games(editions, lo, hi)
        macro = _period_statistic(df, windows, t_list, "macro")
        weighted = _period_statistic(df, windows, t_list, "weighted")
        ci = bootstrap_period(df, windows, t_list, n_boot, seed, stream=100 + i)
        rows.append(dict(
            definition=label,
            start_year=lo,
            end_year=hi,
            games_in_definition="|".join(str(t) for t in t_list),
            n_games=len(t_list),
            smape_macro=macro,
            smape_macro_ci95_low=ci["macro"][0],
            smape_macro_ci95_high=ci["macro"][1],
            smape_weighted=weighted,
            note=note,
        ))
    return pd.DataFrame(rows)


def make_synthetic(path: str, seed: int = 42) -> pd.DataFrame:
    editions = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936,
                1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984,
                1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
    tiers = [
        ("USA", 105.0), ("URS", 95.0), ("CHN", 88.0), ("GER", 70.0),
        ("GBR", 65.0), ("FRA", 55.0), ("ITA", 50.0), ("AUS", 48.0),
        ("JPN", 42.0), ("KOR", 30.0), ("NED", 28.0), ("HUN", 25.0),
        ("CUB", 22.0), ("CAN", 22.0), ("POL", 20.0), ("ESP", 20.0),
        ("BRA", 18.0), ("ROU", 18.0), ("NOR", 15.0), ("SWE", 14.0),
        ("BUL", 12.0), ("FIN", 12.0), ("KEN", 12.0), ("DEN", 10.0),
        ("JAM", 10.0), ("BEL", 10.0), ("SUI", 8.0), ("AUT", 8.0),
        ("TUR", 8.0), ("MEX", 8.0), ("ARG", 8.0), ("GRE", 6.0),
        ("UKR", 22.0), ("BLR", 12.0), ("KAZ", 8.0), ("CRO", 6.0),
        ("ETH", 8.0), ("IRN", 6.0), ("ZAF", 7.0), ("COL", 5.0),
        ("TPE", 8.0), ("THA", 5.0), ("NGR", 5.0), ("SVK", 6.0),
        ("AZE", 4.0), ("LTU", 4.0), ("GEO", 4.0), ("MAR", 4.0),
        ("POR", 4.0), ("IRL", 4.0), ("ISR", 4.0), ("EGY", 3.0),
        ("INA", 3.0), ("MAS", 3.0), ("LAT", 3.0), ("EST", 3.0),
        ("TUN", 2.0), ("SIN", 2.0), ("URU", 2.0), ("CHI", 2.0),
        ("MDA", 2.0), ("ALG", 2.0), ("FIJ", 0.5), ("PER", 0.5),
        ("ERI", 0.5), ("BOT", 1.0), ("GRN", 0.5), ("UAE", 1.0),
    ]
    waves = [1896, 1920, 1952, 1980]
    join_year = {name: waves[i % 4] for i, (name, _) in enumerate(tiers)}
    boycott = {("USA", 1980), ("URS", 1984)}
    rng = np.random.default_rng(seed)
    base = {name: float(b) for name, b in tiers}

    def period_scale(y: int) -> float:
        if 1976 <= y <= 1991:
            return 0.214
        if 1992 <= y <= 2016:
            return 0.178
        if 2017 <= y <= 2024:
            return 0.231
        return 0.26

    def size_factor(actual: int) -> float:
        if actual <= 2:
            return 1.45
        if actual <= 6:
            return 1.25
        if actual <= 20:
            return 1.05
        if actual <= 50:
            return 0.85
        return 0.70

    rows = []
    for k, t in enumerate(editions):
        for name, _ in tiers:
            if t < join_year[name] or (name, t) in boycott:
                continue
            drift = 1.0 + 0.012 * (k - editions.index(join_year[name]))
            host_boost = 1.45 if (name, t) in {("USA", 1984), ("CHN", 2008),
                                               ("GBR", 2012), ("JPN", 2020)} else 1.0
            shock = float(rng.lognormal(mean=0.0, sigma=0.22))
            actual = max(0, int(round(base[name] * drift * host_boost * shock)))
            eps = rng.normal(0.0, period_scale(t)) * size_factor(actual)
            pred = max(0, int(round(actual * (1.0 + eps))))
            rows.append((name, t, actual, pred))
    df = pd.DataFrame(rows, columns=["country", "year", "actual", "predicted"])

    for _ in range(3):
        err = smape_obs(df["actual"].to_numpy(), df["predicted"].to_numpy())
        for name, (lo, hi) in DEFAULT_PERIODS.items():
            in_p = df["year"].between(lo, hi)
            sel = in_p.to_numpy() & np.isfinite(err)
            if not sel.any():
                continue
            achieved = float(np.nanmean(err[sel])) / 100.0
            if achieved <= 0:
                continue
            factor = period_scale(lo) / achieved
            eps = (df.loc[in_p, "predicted"] /
                   df.loc[in_p, "actual"].replace(0, np.nan) - 1.0)
            eps = (eps * factor).fillna(0.0)
            df.loc[in_p, "predicted"] = np.maximum(
                0, np.round(df.loc[in_p, "actual"] * (1.0 + eps))).astype(int)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [("A", 2012, 10, 8), ("B", 2012, 4, 4), ("C", 2012, 0, 2),
         ("A", 2016, 12, 12), ("B", 2016, 2, 4), ("C", 2016, 6, 3),
         ("A", 2020, 8, 10), ("B", 2020, 5, 5), ("C", 2020, 0, 0)],
        columns=["country", "year", "actual", "predicted"])


def run_selftest() -> int:
    failures = []

    def check(label, got, want, tol=1e-6):
        ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        if not ok:
            failures.append(f"{label}: got {got!r}, expected {want!r}")
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got!r} (expected {want!r})")

    df = add_persistence(toy_frame())
    tab = compute_games_table(df)

    print("[selftest] window reading Omega_t = editions with |t'-t| <= 5")
    for t, want in ((2012, "2012|2016"), (2016, "2012|2016|2020"),
                    (2020, "2016|2020")):
        check(f"window_games[{t}]", str(tab.loc[tab.year == t, "window_games"].iloc[0]), want)
    for t, want in ((2012, 6.0), (2016, 8.0), (2020, 5.0)):
        check(f"n_t[{t}]", float(tab.loc[tab.year == t, "n_t"].iloc[0]), want)
    check("per-window skip counts sum (row C@2020 is in 2 windows)",
          float(tab["n_skipped"].sum()), 2.0)
    check("distinct rows with |y|+|yhat| = 0", float(count_skips(df)["n_undefined_smape"]), 1.0)
    check("per-window persistence-missing sum (3 rows x 2 windows)",
          float(tab["persistence_missing"].sum()), 6.0)
    check("distinct rows without a previous edition",
          float(count_skips(df)["n_persistence_missing"]), 3.0)

    print("[selftest] SMAPE(t), macro paper reading")
    for t, want in ((2012, 59.259259259259), (2016, 47.222222222222),
                    (2020, 31.111111111111)):
        check(f"smape[{t}]", float(tab.loc[tab.year == t, "smape"].iloc[0]), want)

    print("[selftest] medal-weighted SMAPE(t)")
    for t, want in ((2012, 200.0 / 9.0), (2016, 8400.0 / 423.0),
                    (2020, 6400.0 / 297.0)):
        check(f"weighted_smape[{t}]",
              float(tab.loc[tab.year == t, "weighted_smape"].iloc[0]), want)

    print("[selftest] naive persistence baseline")
    for t, want in ((2012, 9400.0 / 99.0), (2016, 101.760461760462),
                    (2020, 101.760461760462)):
        check(f"persistence_smape[{t}]",
              float(tab.loc[tab.year == t, "persistence_smape"].iloc[0]), want)

    print("[selftest] period aggregation (mean over Games of SMAPE(t))")
    periods = OrderedDict([("P1", (2012, 2012)), ("P2", (2016, 2016)),
                           ("P3", (2020, 2020)), ("ALL", (2012, 2020))])
    summ, _ = summarise_periods(df, periods, n_boot=200, seed=42)
    s = summ.set_index("period")
    for p, want in (("P1", 59.259259259259), ("P2", 47.222222222222),
                    ("P3", 31.111111111111), ("ALL", 45.864197530864)):
        check(f"smape_macro[{p}]", float(s.loc[p, "smape_macro"]), want)
    for p, want in (("P1", 200.0 / 9.0), ("P2", 8400.0 / 423.0),
                    ("P3", 6400.0 / 297.0), ("ALL", 21.209733266471)):
        check(f"smape_weighted[{p}]", float(s.loc[p, "smape_weighted"]), want)
    for p, want in (("P1", 9400.0 / 99.0), ("P2", 101.760461760462),
                    ("P3", 101.760461760462), ("ALL", 99.490139490140)):
        check(f"persistence_macro[{p}]", float(s.loc[p, "persistence_macro"]), want)

    print("[selftest] bootstrap CI sanity (brackets point estimate, deterministic)")
    for p in ("P1", "ALL"):
        lo, hi = float(s.loc[p, "smape_macro_ci95_low"]), float(s.loc[p, "smape_macro_ci95_high"])
        pt = float(s.loc[p, "smape_macro"])
        check(f"ci_low[{p}] <= point", True, bool(lo <= pt))
        check(f"point <= ci_high[{p}]", True, bool(pt <= hi))
        check(f"ci width[{p}] > 0", True, bool(hi > lo))
    _, payload_a = summarise_periods(df, periods, n_boot=50, seed=42)
    _, payload_b = summarise_periods(df, periods, n_boot=50, seed=42)
    check("bootstrap deterministic under fixed seed",
          payload_a["P1"]["smape_macro_ci95_low"], payload_b["P1"]["smape_macro_ci95_low"])

    if failures:
        print(f"\n[selftest] {len(failures)} FAILURE(S):")
        for f in failures:
            print("  - " + f)
        return 1
    print("\n[selftest] all assertions passed")
    return 0


def write_outputs(df: pd.DataFrame, outdir: str, n_boot: int, seed: int,
                  source: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    tab = compute_games_table(df)
    tab_out = tab.copy()
    for c in tab_out.columns:
        if tab_out[c].dtype.kind == "f":
            tab_out[c] = tab_out[c].round(6)
    tab_out.to_csv(os.path.join(outdir, "smape_by_games.csv"), index=False)

    summary, payload = summarise_periods(df, DEFAULT_PERIODS, n_boot, seed)
    summ_out = summary.copy()
    for c in summ_out.columns:
        if summ_out[c].dtype.kind == "f":
            summ_out[c] = summ_out[c].round(6)
    summ_out.to_csv(os.path.join(outdir, "period_summary.csv"), index=False)

    meta = dict(
        source_file=os.path.abspath(source),
        protocol=dict(
            smape="SMAPE(t) = (100/n_t) * sum_{i in Omega_t} 2|y_i-yhat_i|/(|y_i|+|yhat_i|)",
            omega_t="all country-Games observations of editions with |t'-t| <= 5 years; "
                    "at the 4-year Games cadence this is the editions {t-4, t, t+4}",
            period_aggregation="unweighted mean over the Games years t inside the period "
                               "(macro-average, the paper's 'average SMAPE values')",
            weighted="SMAPE_w(t) = sum_i w_i e_i / sum_i w_i with w_i = actual medals, "
                     "then the same mean over Games; controls the small-nation right skew",
            persistence="yhat(c,t) = actual(c, previous edition present in the data); "
                        "countries with no earlier record are skipped and counted",
            bootstrap=f"cluster (country-level) bootstrap, B={n_boot}, seed={seed}, "
                      "percentile 95% CI",
            skip_rule="rows with |y|+|yhat| = 0 are excluded from all statistics and counted",
        ),
        n_observations=int(df.shape[0]),
        n_countries=int(df["country"].nunique()),
        n_games=int(df["year"].nunique()),
        year_range=[int(df["year"].min()), int(df["year"].max())],
        skips=count_skips(df),
        paper_reported_percent=PAPER_REPORTED,
    )
    payload = {k: {kk: (vv if isinstance(vv, str) else
                        (list(vv) if isinstance(vv, tuple) else vv))
                   for kk, vv in v.items()} for k, v in payload.items()}
    with open(os.path.join(outdir, "period_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "periods": payload}, fh,
                  indent=2, ensure_ascii=False)

    cols = ["period", "games_years", "n_obs", "smape_macro",
            "smape_macro_ci95_low", "smape_macro_ci95_high", "smape_weighted",
            "persistence_macro"]
    print("\n=== period summary (percent) ===")
    print(summ_out[cols].to_string(index=False))
    print("\n=== paper comparison (macro SMAPE) ===")
    for _, r in summary.iterrows():
        paper = PAPER_REPORTED.get(r["period"])
        if paper is not None:
            print(f"  {r['period']:<14} implemented {r['smape_macro']:6.2f}%  "
                  f"paper {paper:5.1f}%  delta {r['smape_macro'] - paper:+6.2f} pp")
    print(f"\n  skips (|y|+|yhat|=0): {meta['skips']['n_undefined_smape']}   "
          f"persistence-missing rows: {meta['skips']['n_persistence_missing']}   "
          f"observations: {meta['n_observations']} over {meta['n_games']} Games, "
          f"{meta['n_countries']} countries")
    print(f"\noutputs written to {os.path.abspath(outdir)}/"
          "(smape_by_games.csv, period_summary.csv, period_summary.json)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Section 5.1 period backtest: elastic-window SMAPE by Games, "
                    "three-period macro/medal-weighted values, naive persistence "
                    "baseline and country-cluster bootstrap CIs.")
    ap.add_argument("--predictions", help="CSV with country/year/actual/predicted")
    ap.add_argument("--outdir", default="out_backtest",
                    help="output directory (default: out_backtest)")
    ap.add_argument("--country-col", default=DEFAULT_COLUMNS["country"])
    ap.add_argument("--year-col", default=DEFAULT_COLUMNS["year"])
    ap.add_argument("--actual-col", default=DEFAULT_COLUMNS["actual"])
    ap.add_argument("--predicted-col", default=DEFAULT_COLUMNS["predicted"])
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap replicates (0 disables CI); default 2000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--covid-alternatives", action="store_true",
                    help="also evaluate three competing COVID-period definitions")
    ap.add_argument("--make-synthetic", metavar="PATH",
                    help="generate a synthetic prediction CSV at PATH; used as "
                         "input when --predictions is not given")
    ap.add_argument("--selftest", action="store_true",
                    help="run the hand-computed toy assertions and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.make_synthetic:
        make_synthetic(args.make_synthetic, seed=args.seed)
        print(f"[synthetic] wrote {args.make_synthetic}")
    if not args.predictions and not args.make_synthetic:
        print("ERROR: provide --predictions CSV (or --make-synthetic / --selftest)")
        return 2
    source = args.predictions or args.make_synthetic
    df = add_persistence(load_predictions(source, dict(
        country=args.country_col, year=args.year_col,
        actual=args.actual_col, predicted=args.predicted_col)))
    write_outputs(df, args.outdir, args.bootstrap, args.seed, source)
    if args.covid_alternatives:
        alt = covid_alternatives_table(df, args.bootstrap, args.seed)
        for c in alt.columns:
            if alt[c].dtype.kind == "f":
                alt[c] = alt[c].round(6)
        path = os.path.join(args.outdir, "covid_alternatives.csv")
        alt.to_csv(path, index=False)
        print("\n=== COVID-period definition alternatives ===")
        print(alt.drop(columns=["note"]).to_string(index=False))
        for _, r in alt.iterrows():
            print(f"  {r['definition']}: games={r['games_in_definition']}"
                  f"  macro={r['smape_macro']:.2f}%")
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
