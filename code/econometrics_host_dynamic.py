import json
from pathlib import Path

import numpy as np
import pandas as pd

import feols

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "data" / "panel_clean.csv"
OUT = HERE / "gen" / "econometrics_host_dynamic_results.json"
SENSITIVITY = OUT.with_name("host_sample_sensitivity.csv")

YEAR_MIN, YEAR_MAX = 1976, 2024
KMIN, KMAX, KREF = -3, 5, -1


def dummies(df, kref=KREF):
    cols = {}
    for k in range(KMIN, KMAX + 1):
        if k == kref:
            continue
        name = f"D_m{-k}" if k < 0 else f"D_{k}"
        cols[name] = (df["k"] == k).astype(float)
    return pd.DataFrame(cols, index=df.index)


def run(df, controls=True, kref=KREF):
    df, n_single = feols.drop_singletons(df)
    D = dummies(df, kref)
    dfa = pd.concat([df, D], axis=1)
    extra = list(D.columns) + (["log_gdp", "log_pop"] if controls else [])
    res, Z, info = feols.fe_ols(dfa, "log1p_total", extra=extra, interest=list(D.columns))
    return res, D, Z, info, n_single


def pre_trend_test(res):
    lead_names = [f"D_m{-k}" for k in range(KMIN, KREF)]
    test = res.wald_test(", ".join(f"{name} = 0" for name in lead_names),
                         use_f=True, scalar=True)
    return dict(F=float(test.statistic), p=float(test.pvalue),
                df_num=float(test.df_num), df_den=float(test.df_denom),
                leads_tested=lead_names)


def sample_sensitivity(df, full_dynamic):
    samples = (
        ("Full modern window", df),
        ("Excluding 1980 and 1984", df.loc[~df["year"].isin([1980, 1984])]),
        ("Excluding United States", df.loc[df["ioc"] != "USA"]),
    )
    rows = []
    for label, sample in samples:
        static_df, _ = feols.drop_singletons(sample)
        static, _, _ = feols.fe_ols(static_df, "log1p_total",
                                    extra=("host_flag",), interest=("host_flag",))
        dynamic = full_dynamic if sample is df else run(sample)[0]
        row = dict(sample=label, static_n=int(static.nobs), dynamic_n=int(dynamic.nobs))
        for prefix, model, term in (
            ("static_host", static, "host_flag"),
            ("hosting_year", dynamic, "D_0"),
            ("fifth_edition", dynamic, "D_5"),
        ):
            estimate = feols.summary_row(model, term)
            row.update({f"{prefix}_{key}": estimate[key]
                        for key in ("beta", "ci_lo", "ci_hi", "p")})
        row["pre_trend_p"] = pre_trend_test(dynamic)["p"]
        rows.append(row)
    return pd.DataFrame(rows)


def shares_of(delta_by_k):
    ks = pd.Series(delta_by_k)
    lead = float(ks[ks.index <= KREF].sum())
    curr = float(ks.loc[0])
    leg = float(ks[ks.index >= 1].sum())
    tot = lead + curr + leg
    return dict(leading=lead, current=curr, legacy=leg, total=tot,
                share_leading=lead / tot, share_current=curr / tot,
                share_legacy=leg / tot)


def main():
    df, meta = feols.load_panel(PANEL, drop_missing_gdp=False)
    df = feols.add_logvars(df)
    n_panel = len(df)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    n_window = len(df)

    df["k"] = np.nan
    n_multi_host = 0
    for ioc, g in df[df.host_flag == 1].groupby("ioc"):
        host_years = sorted(g["year"].unique())
        n_multi_host += int(len(host_years) > 1)
        m = df["ioc"] == ioc
        yrs = df.loc[m, "year"].to_numpy()
        ks = np.stack([(yrs - h) / 4.0 for h in host_years], axis=1)
        df.loc[m, "k"] = ks[np.arange(len(yrs)), np.abs(ks).argmin(axis=1)]

    n_missing_gdp = int((df["gdp_gapfill"] == "missing").sum())
    n_missing_gdp_host = int(((df["gdp_gapfill"] == "missing") & (df["host_flag"] == 1)).sum())

    res, D, Z, info, n_single = run(df, controls=True)
    res_raw, _, Zr, _, _ = run(df, controls=False)

    pretest = pre_trend_test(res)

    rows = []
    for k in range(KMIN, KMAX + 1):
        if k == KREF:
            rows.append(dict(k=k, delta=0.0, se=None, p=None, ci_lo=0.0, ci_hi=0.0,
                             games=None, is_reference=True))
            continue
        name = f"D_m{-k}" if k < 0 else f"D_{k}"
        ci = res.conf_int().loc[name]
        rows.append(dict(k=k, delta=float(res.params[name]), se=float(res.bse[name]),
                         p=float(res.pvalues[name]), ci_lo=float(ci.iloc[0]),
                         ci_hi=float(ci.iloc[1]), is_reference=False,
                         games=int(4 * k) if k else 0))
    path = pd.DataFrame(rows)
    mean_total = float(Z["total"].mean())
    path["delta_medals_at_sample_mean"] = np.expm1(path["delta"]) * (mean_total + 1)

    d_by_k = {int(r.k): float(r.delta) for r in path.itertuples()}
    signed = shares_of(d_by_k)
    absolute = shares_of({k: abs(v) for k, v in d_by_k.items()})
    raw_by_k = {}
    for k in range(KMIN, KMAX + 1):
        if k == KREF:
            raw_by_k[k] = 0.0
        else:
            raw_by_k[k] = float(res_raw.params[f"D_m{-k}" if k < 0 else f"D_{k}"])
    nocontrols = shares_of(raw_by_k)

    res_alt, _, _, info_alt, _ = run(df, controls=True, kref=KMIN)
    alt_by_k = {}
    for k in range(KMIN, KMAX + 1):
        if k == KMIN:
            alt_by_k[k] = 0.0
        else:
            alt_by_k[k] = float(res_alt.params[f"D_m{-k}" if k < 0 else f"D_{k}"])
    signed_kref_m3 = shares_of(alt_by_k)

    out = dict(
        seed=SEED,
        sample=dict(period=f"Games {YEAR_MIN}-{YEAR_MAX}", n_rows_full_panel=n_panel,
                    n_rows_in_window=n_window,
                    n_rows_estimated=int(info["n_obs"]),
                    rows_dropped_singleton_countries=int(n_single),
                    rows_dropped_missing_gdp=n_missing_gdp,
                    rows_dropped_missing_gdp_host=int(n_missing_gdp_host),
                    n_countries=int(Z["ioc"].nunique()), n_games=int(df["year"].nunique()),
                    multi_host_countries_in_window=int(n_multi_host)),
        k_units="Games units (1 unit = 4 years); k = 0 is the hosting edition",
        reference_period=KREF, reference_note="delta_{-1} := 0 (omitted)",
        pre_trend_test=pretest, path=json.loads(path.to_json(orient="records")),
        shares_signed_sum=signed, shares_abs=absolute,
        shares_nocontrols=nocontrols,
        shares_signed_kref_minus3=signed_kref_m3,
        path_kref_minus3=json.loads(pd.DataFrame(
            [{"k": k, "delta": v} for k, v in alt_by_k.items()]).to_json(orient="records")),
        mean_total_sample=mean_total,
        identification=info)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    sample_sensitivity(df, res).to_csv(SENSITIVITY, index=False)

    print(out["sample"])
    print(path.to_string(index=False, formatters={
        "delta": "{: .4f}".format, "se": "{:.4f}".format, "p": "{:.4g}".format,
        "ci_lo": "{: .4f}".format, "ci_hi": "{: .4f}".format,
        "delta_medals_at_sample_mean": "{: .2f}".format}))
    print(f"pre-trend F({pretest['df_num']:.0f},{pretest['df_den']:.0f})="
          f"{pretest['F']:.3f}, p={pretest['p']:.4g}")
    print("identification:", info)
    print("signed shares:", {k: round(v, 4) for k, v in signed.items()})
    print("|delta| shares:", {k: round(v, 4) for k, v in absolute.items()})
    print("no-control shares:", {k: round(v, 4) for k, v in nocontrols.items()})
    print("kref=-3 shares:", {k: round(v, 4) for k, v in signed_kref_m3.items()})
    print("kref=-3 path:", {k: round(v, 4) for k, v in alt_by_k.items()})
    print("wrote", OUT)
    print("wrote", SENSITIVITY)


if __name__ == "__main__":
    main()
