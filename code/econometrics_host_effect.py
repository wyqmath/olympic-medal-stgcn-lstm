import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import feols

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "data" / "panel_clean.csv"
OUT = HERE / "gen" / "econometrics_host_effect_results.json"


def fit_poisson(df, extra=("host_flag",)):
    X = feols.design(df, list(extra))
    Z = pd.concat([df[["total", "ioc"]], X], axis=1).dropna(subset=["total"])
    Z = Z.loc[:, ~Z.columns.duplicated()]
    fit_poisson.last_singletons = 0
    res = sm.GLM(Z["total"], Z[X.columns], family=sm.families.Poisson()).fit(
        method="lbfgs", maxiter=5000, cov_type="cluster",
        cov_kwds={"groups": Z["ioc"]})
    try:
        chk = sm.Poisson(Z["total"], Z[X.columns]).fit(method="newton", maxiter=500,
                                                       disp=0)
        xchk = float(np.squeeze(chk.params[extra[0]]))
    except Exception:
        xchk = None
    return res, Z, xchk


def main():
    df, meta = feols.load_panel(PANEL)
    df = feols.add_logvars(df)
    out = {"seed": SEED, "sample": meta, "specs": {}}

    def ols(key, data, y, extra=("host_flag",)):
        data, n_single = feols.drop_singletons(data)
        res, Z, info = feols.fe_ols(data, y, extra=extra, interest=list(extra) + ["const"])
        row = feols.summary_row(res, extra[0] if extra else "const")
        row.update(n_singletons_dropped=n_single, info=info,
                   mean_total_sample=float(Z["total"].mean()),
                   mean_total_host_rows=float(Z.loc[Z["host_flag"] == 1, "total"].mean()))
        if y == "log1p_total":
            pct = float(np.expm1(row["beta"]))
            host = Z["host_flag"].to_numpy()
            yhat0 = res.fittedvalues - row["beta"] * host
            row.update(pct_effect=pct,
                       medals_per_typical_country=pct * (row["mean_total_sample"] + 1),
                       medals_att_hosted=float(np.mean(
                           np.expm1(yhat0[host == 1] + row["beta"])
                           - np.expm1(yhat0[host == 1]))))
        if "log_gdp" in extra:
            row["controls"] = {c: feols.summary_row(res, c) for c in ("log_gdp", "log_pop")}
        out["specs"][key] = row
        return row

    dfg = df[df["population"].notna()].copy()
    out["specs"]["A1_level_1896_2024"] = ols("A1_level_1896_2024", df, "total")
    out["specs"]["A2_log1p_1896_2024"] = ols("A2_log1p_1896_2024", df, "log1p_total")

    res, Z, glm_b = fit_poisson(df)
    b = float(res.params["host_flag"])
    ci = res.conf_int().loc["host_flag"]
    Xd = Z[res.model.exog_names].to_numpy(float)
    mu = np.exp(Xd @ np.asarray(res.params, float))
    mu0 = mu / np.exp(b * Z["host_flag"].to_numpy())
    mu_gap = float(abs(mu.mean() - Z["total"].mean()) / Z["total"].mean())
    assert mu_gap < 1e-3
    host_mask = Z["host_flag"].to_numpy() == 1
    out["specs"]["A3_poisson_1896_2024"] = dict(
        n=int(res.nobs), rows_dropped_missing=0, info=None,
        rows_dropped_singleton_countries=int(fit_poisson.last_singletons),
        beta=b, se=float(res.bse["host_flag"]),
        p=float(res.pvalues["host_flag"]),
        ci_lo=float(ci.iloc[0]), ci_hi=float(ci.iloc[1]),
        pct_effect=float(np.exp(b) - 1.0),
        medals_per_typical_country=float((np.exp(b) - 1.0) * Z["total"].mean()),
        medals_att_hosted=float(np.mean((mu - mu0)[host_mask])),
        mean_total_sample=float(Z["total"].mean()),
        mean_total_host_rows=float(Z.loc[Z["host_flag"] == 1, "total"].mean()),
        discrete_mle_beta_crosscheck=glm_b, mu_mean_relative_gap=mu_gap,
        glm_deviance=float(res.deviance), glm_pearson_chi2=float(res.pearson_chi2))

    out["specs"]["A4_log1p_econ_gdpcomplete"] = ols(
        "A4_log1p_econ_gdpcomplete", dfg, "log1p_total",
        extra=("host_flag", "log_gdp", "log_pop"))

    dfm = df[df["year"] >= 1976].copy()
    out["specs"]["A5_level_1976_2024"] = ols("A5_level_1976_2024", dfm, "total")
    out["specs"]["A6_log1p_1976_2024"] = ols("A6_log1p_1976_2024", dfm, "log1p_total")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("sample:", meta)
    for k, s in out["specs"].items():
        msg = (f"{k:30s} b={s['beta']:8.3f} (se {s['se']:.3f}, p={s['p']:.3g}) "
               f"95%CI [{s['ci_lo']:8.2f},{s['ci_hi']:8.2f}] N={s.get('n') or s['info']['n_obs']}")
        if "medals_per_typical_country" in s:
            msg += (f" | pct={s['pct_effect']*100:6.1f}%  medals/typical country="
                    f"{s['medals_per_typical_country']:6.2f}"
                    f"  medals(ATT hosted)={s.get('medals_att_hosted', float('nan')):7.2f}")
        if s.get("info") and s["info"].get("interest_identified") is False:
            msg += "  !! INTEREST NOT IDENTIFIED"
        print(msg)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
