import numpy as np
import pandas as pd
import statsmodels.api as sm

PANEL_COLUMNS = ["country", "ioc", "year", "gold", "silver", "bronze", "total",
                 "host_flag", "gdp_total_2011usd_million", "gdp_per_capita_2011usd",
                 "population", "gdp_gapfill"]


def load_panel(path, drop_missing_gdp=False):
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df[df["ioc"].notna()].copy()
    n_mixed = n_raw - len(df)
    n_missing = 0
    if drop_missing_gdp:
        n_missing = int((df["gdp_gapfill"] == "missing").sum())
        df = df[df["gdp_gapfill"] != "missing"].copy()
    return df, dict(rows_raw=int(n_raw), rows_mixed_team_dropped=int(n_mixed),
                    rows_gdp_missing_dropped=int(n_missing))


def add_logvars(df):
    df = df.copy()
    df["log1p_total"] = np.log1p(df["total"])
    df["log_gdp"] = np.log(df["gdp_total_2011usd_million"])
    df["log_pop"] = np.log(df["population"])
    return df


def design(df, extra=()):
    X = pd.get_dummies(df[["country", "year"]].astype({"year": str}),
                       columns=["country", "year"], drop_first=True, dtype=float)
    extra = list(extra)
    if extra:
        X = pd.concat([df[extra].astype(float), X], axis=1)
    return sm.add_constant(X, has_constant="add")


def drop_singletons(df, cluster="ioc"):
    sizes = df[cluster].value_counts()
    keep = df[cluster].isin(sizes[sizes > 1].index)
    return df[keep].copy(), int((~keep).sum())


def fe_ols(df, y, extra=(), cluster="ioc", interest=("const",)):
    X = design(df, extra)
    keepcols = sorted({y, "total", cluster, "year"})
    Z = pd.concat([df[keepcols], X], axis=1)
    n_before = len(Z)
    Z = Z.dropna()
    Z = Z.loc[:, ~Z.columns.duplicated()]
    n_dropped = int(n_before - len(Z))
    res = sm.OLS(Z[y], Z[X.columns]).fit(cov_type="cluster",
                                         cov_kwds={"groups": Z[cluster]}, use_t=True)
    A = Z[X.columns].to_numpy(float)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    null = vt[s < 1e-9 * max(s[0], 1.0)]
    info = dict(n_obs=int(res.nobs), rows_dropped_missing=int(n_dropped),
                design_cols=int(A.shape[1]),
                design_rank=int((s > 1e-9 * max(s[0], 1.0)).sum()),
                n_null_directions=int(null.shape[0]))
    if null.shape[0] and interest:
        j = [X.columns.get_loc(c) for c in interest if c in X.columns]
        if j:
            info["interest_max_abs_null"] = float(np.abs(null[:, j]).max())
            info["interest_identified"] = bool(info["interest_max_abs_null"] < 1e-8)
    return res, Z, info


def summary_row(res, name):
    ci = res.conf_int().loc[name]
    return dict(beta=float(res.params[name]), se=float(res.bse[name]),
                t=float(res.tvalues[name]), p=float(res.pvalues[name]),
                ci_lo=float(ci.iloc[0]), ci_hi=float(ci.iloc[1]))
