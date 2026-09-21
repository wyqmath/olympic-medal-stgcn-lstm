import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import ruptures as rpt

import feols

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "data" / "panel_clean.csv"
OUT = HERE / "gen" / "econometrics_gdp_break_results.json"

MIN_SEG = 5
MAX_BKP = 3
MIN_EDITION_N = 10


def pooled_split(df, break_year, country_fe):
    d = df.copy()
    d["post"] = (d["year"] > break_year).astype(float)
    d["log_gdp_x_post"] = d["log_gdp"] * d["post"]
    if country_fe:
        res, Z, info = feols.fe_ols(d, "log1p_total",
                                    extra=["log_gdp", "log_gdp_x_post"],
                                    interest=["log_gdp", "log_gdp_x_post"])
    else:
        X = pd.concat([d[["log_gdp", "log_gdp_x_post"]],
                       pd.get_dummies(d["year"].astype(str), prefix="yr",
                                      drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X, has_constant="add")
        Z = pd.concat([d[["ioc", "total", "log1p_total"]], X], axis=1).dropna(subset=["log_gdp"])
        Z = Z.loc[:, ~Z.columns.duplicated()]
        res = sm.OLS(Z["log1p_total"], Z[X.columns]).fit(
            cov_type="cluster", cov_kwds={"groups": Z["ioc"]}, use_t=True)
        info = {"design_cols": int(X.shape[1])}
    b = float(res.params["log_gdp"])
    dlt = float(res.params["log_gdp_x_post"])
    ci_b = res.conf_int().loc["log_gdp"]
    ci_d = res.conf_int().loc["log_gdp_x_post"]
    return dict(break_year=int(break_year), country_fe=bool(country_fe),
                beta_pre=b, se_pre=float(res.bse["log_gdp"]),
                p_pre=float(res.pvalues["log_gdp"]),
                ci_pre=[float(ci_b.iloc[0]), float(ci_b.iloc[1])],
                beta_post=b + dlt, delta_post=dlt,
                se_delta=float(res.bse["log_gdp_x_post"]),
                p_delta=float(res.pvalues["log_gdp_x_post"]),
                ci_delta=[float(ci_d.iloc[0]), float(ci_d.iloc[1])],
                n_obs=int(res.nobs), identification=info)


def chow_at(sig, year):
    pre = sig[sig.index <= year].to_numpy()
    post = sig[sig.index > year].to_numpy()
    ssr_r = float(((sig.to_numpy() - sig.mean()) ** 2).sum())
    ssr_u = float(((pre - pre.mean()) ** 2).sum() + ((post - post.mean()) ** 2).sum())
    k, n1, n2 = 1, len(pre), len(post)
    F = (ssr_r - ssr_u) / k / (ssr_u / (n1 + n2 - 2 * k))
    from scipy import stats
    return dict(break_year=int(year), n_pre=n1, n_post=n2,
                mean_beta_pre=float(pre.mean()), se_pre=float(pre.std(ddof=1)),
                mean_beta_post=float(post.mean()), se_post=float(post.std(ddof=1)),
                F=float(F), df=[k, n1 + n2 - 2 * k],
                p=float(1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)))


def main():
    df, meta = feols.load_panel(PANEL, drop_missing_gdp=True)
    df = feols.add_logvars(df)

    rows = []
    for yr, g in df.groupby("year"):
        g = g[g["log_gdp"].notna()]
        if len(g) < MIN_EDITION_N:
            rows.append(dict(year=int(yr), n=len(g), beta=np.nan, se=np.nan,
                             r2=np.nan, kept=False))
            continue
        X = sm.add_constant(g[["log_gdp"]], has_constant="add")
        r = sm.OLS(g["log1p_total"], X).fit()
        ci = r.conf_int().loc["log_gdp"]
        rows.append(dict(year=int(yr), n=int(len(g)), beta=float(r.params["log_gdp"]),
                         se=float(r.bse["log_gdp"]), p=float(r.pvalues["log_gdp"]),
                         ci_lo=float(ci.iloc[0]), ci_hi=float(ci.iloc[1]),
                         r2=float(r.rsquared), kept=True))
    ser = pd.DataFrame(rows)
    kept = ser[ser.kept].reset_index(drop=True)
    sig = kept.set_index("year")["beta"]

    signal = kept["beta"].to_numpy().reshape(-1, 1)
    n = len(signal)
    cands = []
    for nb in range(0, MAX_BKP + 1):
        algo = rpt.Dynp(model="l2", min_size=MIN_SEG, jump=1).fit(signal)
        end_idx = algo.predict(n_bkps=nb)
        ssr, betas_seg = 0.0, []
        start = 0
        for e in end_idx:
            seg = signal[start:e, 0]
            ssr += float(((seg - seg.mean()) ** 2).sum())
            betas_seg.append((int(kept["year"].iloc[int(start)]), int(kept["year"].iloc[int(e) - 1]),
                              float(seg.mean()), int(len(seg))))
            start = e
        p = (nb + 1) + nb
        bic = n * np.log(max(ssr, 1e-12) / n) + p * np.log(n)
        cands.append(dict(n_bkps=int(nb), ssr=float(ssr), bic=float(bic), p=int(p),
                          breaks=[int(kept["year"].iloc[int(e) - 1]) for e in end_idx[:-1]],
                          segments=[dict(y_from=s, y_to=t, mean_beta=b, n_editions=k)
                                    for s, t, b, k in betas_seg]))
    best = min(cands, key=lambda c: c["bic"])

    b_year = best["breaks"][0] if best["breaks"] else None
    chow = chow_at(sig, b_year) if b_year is not None else None
    chow_2008 = chow_at(sig, 2008) if 2008 in list(sig.index) else None

    pooled_sel = [pooled_split(df, b_year, country_fe) for country_fe in (False, True)]
    pooled_2008 = [pooled_split(df, 2008, country_fe) for country_fe in (False, True)]

    rows2 = []
    for yr, g in df.groupby("year"):
        g = g[g["log_gdp"].notna() & g["log_pop"].notna()]
        if len(g) < MIN_EDITION_N:
            continue
        X = sm.add_constant(g[["log_gdp", "log_pop"]], has_constant="add")
        r = sm.OLS(g["log1p_total"], X).fit()
        rows2.append(dict(year=int(yr), n=int(len(g)), beta=float(r.params["log_gdp"]),
                          se=float(r.bse["log_gdp"])))
    ser2 = pd.DataFrame(rows2)

    out = dict(seed=SEED, sample=dict(
        rows_raw=meta["rows_raw"], rows_mixed_team_dropped=meta["rows_mixed_team_dropped"],
        rows_gdp_missing_dropped=meta["rows_gdp_missing_dropped"],
        n_rows_estimated=int(pooled_sel[0]["n_obs"]),
        n_countries=int(df["ioc"].nunique()),
        n_games=int(df["year"].nunique()), min_rows_per_edition=MIN_EDITION_N),
        editions=json.loads(ser.to_json(orient="records")),
        editions_with_pop=json.loads(ser2.to_json(orient="records")),
        segmentation_candidates=cands, selected_segmentation=best,
        chow_test_detected_break=chow,
        chow_test_forced_2008=chow_2008,
        pooled_at_detected_break=dict(
            model="log1p(total) ~ log_gdp + post + log_gdp x post + Games FE "
                  "(+/- country FE), cluster(country)",
            no_country_fe=pooled_sel[0], with_country_fe=pooled_sel[1]),
        pooled_at_2008=dict(no_country_fe=pooled_2008[0], with_country_fe=pooled_2008[1]),
        paper_old_values=dict(beta_pre=0.47, se_pre=0.09, beta_post=0.32, se_post=0.11,
                              p=0.013, break_year=2008, source="main.tex L505-512 (eq:gdp_change)"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(out["sample"])
    print(ser.to_string(index=False, formatters={
        c: ("{: .4f}".format if c in ("beta", "se", "ci_lo", "ci_hi", "r2") else "{:.4g}".format)
        for c in ["beta", "se", "p", "ci_lo", "ci_hi", "r2", "n"]}))
    print("\nsegmentation candidates:")
    for c in cands:
        print(f"  nbkp={c['n_bkps']} SSR={c['ssr']:.4f} BIC={c['bic']:.3f} "
              f"breaks={c['breaks']} seg_means={[round(s['mean_beta'],4) for s in c['segments']]}")
    print("selected:", best["n_bkps"], "break(s)", best["breaks"])
    for nm, ch in [("Chow@detected", chow), ("Chow@2008", chow_2008)]:
        if ch:
            print(f"{nm}: pre {ch['mean_beta_pre']:.4f}+-{ch['se_pre']:.4f} -> "
                  f"post {ch['mean_beta_post']:.4f}+-{ch['se_post']:.4f} "
                  f"F={ch['F']:.3f}, p={ch['p']:.4g}")
    for nm, ps in [("pooled@detected", pooled_sel), ("pooled@2008", pooled_2008)]:
        for tag, p in zip(("yearFE", "yearFE+countryFE"), ps):
            print(f"{nm} [{tag}]: beta_pre={p['beta_pre']:.4f} (se {p['se_pre']:.4f}) -> "
                  f"beta_post={p['beta_post']:.4f} (delta {p['delta_post']:.4f} "
                  f"se {p['se_delta']:.4f}, p={p['p_delta']:.4g})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
