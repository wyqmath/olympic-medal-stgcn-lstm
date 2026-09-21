import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from collections import Counter
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd

import feols
import stgcn_predict

HERE = Path(__file__).resolve().parent
YEAR_MIN, YEAR_MAX = 1976, 2024
COVS = ["log_gdp", "log_pop", "lag1_total"]
N_SPLITS_DML = 5
CALIPER_MULT = 0.2
STREAMS = {"treatment_effect": 0, "mediation": 1, "moderation": 2}
BLUES = ["#08519c", "#4292c6", "#9ecae1", "#c6dbef"]


def build_samples(panel_path):
    df, meta = feols.load_panel(panel_path)
    df = stgcn_predict.normalise_panel(df, log=lambda *_: None)
    df = feols.add_logvars(df)

    df = df.sort_values(["ioc", "year"], kind="mergesort").copy()
    df["lag1_total"] = df.groupby("ioc")["total"].shift(1)
    eds = sorted(df["year"].unique().tolist())
    prev_ed = {eds[i]: eds[i - 1] for i in range(1, len(eds))}
    df["prev_edition"] = df["year"].map(prev_ed)

    modern = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    n_modern = len(modern)
    fin = np.ones(n_modern, dtype=bool)
    for c in COVS:
        fin &= np.isfinite(modern[c].to_numpy(float))
    cc = modern[fin].copy()
    cc["lag_from_previous_edition"] = (cc["prev_edition"] == cc["year"] - 4)

    meta = dict(meta)
    meta.update(
        period=f"Games {YEAR_MIN}-{YEAR_MAX}",
        n_rows_modern_window=n_modern,
        n_rows_covariate_complete=len(cc),
        n_rows_dropped_incomplete_covariates=int(n_modern - len(cc)),
        n_rows_missing_by_covariate={
            c: int(n_modern - int(np.isfinite(modern[c].to_numpy(float)).sum()))
            for c in COVS},
        n_treated=int(cc["host_flag"].sum()), n_control=int((1 - cc["host_flag"]).sum()),
        n_countries=int(cc["ioc"].nunique()), n_games=int(cc["year"].nunique()),
        n_treated_countries=int(cc.loc[cc["host_flag"] == 1, "ioc"].nunique()),
        lag_non_adjacent_rows=int((~cc["lag_from_previous_edition"]).sum()),
        mean_total_sample=float(cc["total"].mean()),
        mean_total_treated=float(cc.loc[cc["host_flag"] == 1, "total"].mean()),
        treated_rows=cc.loc[cc["host_flag"] == 1, ["ioc", "year", "total"]]
        .to_dict("records"))
    return cc, meta


def fit_propensity(X, T, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1, max_iter=1000, random_state=seed))])
    pipe.fit(X, T)
    return pipe.predict_proba(X)[:, 1]


def tau_ipw(y, T, e):
    w0 = e[T == 0] / (1.0 - e[T == 0])
    y1 = float(y[T == 1].mean())
    y0 = float((w0 * y[T == 0]).sum() / w0.sum())
    return y1 - y0, dict(mean_treated=y1, mean_control_weighted=y0,
                         control_weight_ess=float(w0.sum() ** 2 / (w0 ** 2).sum()),
                         max_control_weight=float(w0.max()), sum_control_weights=float(w0.sum()))


def match_pairs(T, e):
    logit = np.log(e / (1.0 - e))
    caliper = CALIPER_MULT * float(logit.std())
    ti, ci = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    a, b, dropped = [], [], 0
    for i in ti:
        j = int(ci[np.argmin(np.abs(logit[ci] - logit[i]))])
        if abs(logit[j] - logit[i]) > caliper:
            dropped += 1
            continue
        a.append(int(i)), b.append(j)
    diag = dict(caliper=caliper, n_treated=int(len(ti)), n_matched=len(a),
                n_treated_dropped_by_caliper=dropped,
                n_unique_controls_used=len(set(b)) if b else 0,
                mean_abs_logit_distance=float(np.mean(np.abs(
                    logit[np.array(b, int)] - logit[np.array(a, int)]))) if b else np.nan)
    return np.array(a, int), np.array(b, int), diag


def tau_dml(y, T, X, groups, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    mhat, ghat = np.empty(len(y)), np.empty(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS_DML).split(X, y, groups):
        mhat[te] = HistGradientBoostingRegressor(random_state=seed).fit(
            X[tr], T[tr]).predict(X[te])
        ghat[te] = HistGradientBoostingRegressor(random_state=seed).fit(
            X[tr], y[tr]).predict(X[te])
    rt, ry = T - mhat, y - ghat
    denom = float(rt @ rt)
    return (float(rt @ ry / denom) if denom > 1e-12 else np.nan), dict(
        denominator=denom, n_folds=N_SPLITS_DML, fold_unit="ioc (GroupKFold)")


def fe_fit(df, y, extra, terms):
    data, n_single = feols.drop_singletons(df)
    res, Z, info = feols.fe_ols(data, y, extra=extra, interest=list(extra) + ["const"])
    return res, Z, info, {t: feols.summary_row(res, t) for t in terms}, n_single


FE_SKIPS = Counter()


def reset_fe_skips():
    FE_SKIPS.clear()


def fe_betas(df, y, extra, terms, cluster="ioc", prepare=None):
    sizes = df[cluster].value_counts()
    df = df[df[cluster].isin(sizes[sizes > 1].index)]
    if len(df) == 0 or df["year"].nunique() < 2:
        FE_SKIPS["empty_or_single_edition"] += 1
        return None
    if prepare is not None:
        df = prepare(df)
    Xd = feols.design(df, extra)
    cols = list(Xd.columns)
    if not all(c in cols for c in terms):
        FE_SKIPS["interest_column_missing"] += 1
        return None
    A = Xd.to_numpy(float)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    tol = 1e-9 * max(s[0], 1.0)
    j = [cols.index(c) for c in terms]
    null = vt[s < tol]
    if null.shape[0] and np.abs(null[:, j]).max() > 1e-8:
        FE_SKIPS["interest_in_null_space"] += 1
        return None
    beta, *_ = np.linalg.lstsq(A, df[y].to_numpy(float), rcond=None)
    return {c: float(beta[cols.index(c)]) for c in terms}


def draw_indices(ioc_arr, countries, rows_by_ioc, rng):
    pick = rng.integers(0, len(rows_by_ioc), len(rows_by_ioc))
    return pick, np.concatenate([rows_by_ioc[i] for i in pick])


def smd(x_t, x_c, w_c=None):
    if w_c is None:
        m_c, v_c = x_c.mean(), x_c.var(ddof=1)
    else:
        w = np.asarray(w_c, float) / np.asarray(w_c, float).mean()
        m_c = float((w * x_c).mean())
        v_c = float((w * (x_c - m_c) ** 2).sum() / max(w.sum() - 1.0, 1e-12))
    v = 0.5 * (x_t.var(ddof=1) + v_c)
    return float((x_t.mean() - m_c) / np.sqrt(v)) if v > 0 else 0.0


def pct_ci(draws):
    v = np.asarray(draws, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(ci_low=np.nan, ci_high=np.nan, boot_reps=0, boot_mean=np.nan,
                    boot_se=np.nan)
    return dict(ci_low=float(np.percentile(v, 2.5)), ci_high=float(np.percentile(v, 97.5)),
                boot_reps=int(v.size), boot_mean=float(v.mean()),
                boot_se=float(v.std(ddof=1)) if v.size > 1 else np.nan)


def point_estimates(cc, seed):
    y_lvl = cc["total"].to_numpy(float)
    y_log = np.log1p(y_lvl)
    T = cc["host_flag"].to_numpy(float)
    X = cc[COVS].to_numpy(float)
    e = fit_propensity(X, T, seed)
    a, b, md = match_pairs(T, e)
    out = {"propensity": e, "match": (a, b, md), "outcomes": {}}
    for tag, y in (("total", y_lvl), ("log1p_total", y_log)):
        ipw, ipw_d = tau_ipw(y, T, e)
        mtau = float(np.mean(y[a] - y[b])) if len(a) else np.nan
        dml, dml_d = tau_dml(y, T, X, cc["ioc"].to_numpy(), seed)
        out["outcomes"][tag] = dict(IPW=(ipw, ipw_d), Matching=(mtau, dict(md)),
                                    DML=(dml, dml_d))
    return out


def bootstrap_tau(cc, n_boot, seed, methods):
    y_lvl = cc["total"].to_numpy(float)
    y_log = np.log1p(y_lvl)
    T = cc["host_flag"].to_numpy(float)
    X = cc[COVS].to_numpy(float)
    ioc_arr = cc["ioc"].to_numpy()
    countries = list(pd.unique(ioc_arr))
    rows_by_ioc = [np.flatnonzero(ioc_arr == c) for c in countries]
    rng = np.random.default_rng([seed, STREAMS["treatment_effect"]])
    draws = {m: {"total": [], "log1p_total": []} for m in methods}
    skip = Counter()
    t0 = time.time()
    for b_i in range(n_boot):
        pick, idx = draw_indices(ioc_arr, countries, rows_by_ioc, rng)
        Tb, Xb = T[idx], X[idx]
        if Tb.sum() == 0 or (1 - Tb).sum() == 0 or len(set(pick.tolist())) < N_SPLITS_DML:
            skip["too_few_clusters_or_no_treated"] += 1
            continue
        try:
            e = fit_propensity(Xb, Tb, seed)
        except Exception:
            skip["propensity_fit_failed"] += 1
            continue
        a, b, _ = match_pairs(Tb, e)
        for tag, y in (("total", y_lvl[idx]), ("log1p_total", y_log[idx])):
            if "IPW" in methods:
                draws["IPW"][tag].append(tau_ipw(y, Tb, e)[0])
            if "Matching" in methods:
                draws["Matching"][tag].append(float(np.mean(y[a] - y[b])) if len(a) else np.nan)
            if "DML" in methods:
                draws["DML"][tag].append(tau_dml(y, Tb, Xb, ioc_arr[idx], seed)[0])
        if (b_i + 1) % 200 == 0:
            print("  bootstrap %d/%d (%.1f min)" % (b_i + 1, n_boot, (time.time() - t0) / 60.0))
    print("bootstrap: %.1f min, skipped %s" % ((time.time() - t0) / 60.0, dict(skip)))
    return {m: {tag: np.asarray(v, float) for tag, v in d.items()}
            for m, d in draws.items()}, dict(skip)


def treatment_effect_block(cc, pt, reps, meta, methods):
    y_lvl = cc["total"].to_numpy(float)
    T = cc["host_flag"].to_numpy(float)
    X = cc[COVS].to_numpy(float)
    e_hat = pt["propensity"]
    a, b, md = pt["match"]
    notes = {"IPW": "ATT, Hajek weights (treated 1, control e/(1-e)); e = "
                    "LogisticRegression(C=1) on standardised X",
             "Matching": "1:1 NN with replacement on logit(e), caliper=0.2*SD(logit(e))",
             "DML": "partially linear DML, HistGBR nuisances, 5-fold GroupKFold(ioc)"}
    rows, effects = [], {}
    for method in methods:
        for outcome in ("total", "log1p_total"):
            tau_pt, diag = pt["outcomes"][outcome][method]
            st = pct_ci(reps[method][outcome])
            row = dict(method=method, outcome=outcome, tau=tau_pt,
                       ci_low=st["ci_low"], ci_high=st["ci_high"],
                       n_treated=int(T.sum()),
                       n_control=int((1 - T).sum()) if method != "Matching"
                       else md["n_unique_controls_used"],
                       note=notes[method], boot_reps_used=st["boot_reps"])
            if method == "Matching":
                row["note"] += ("; %d/%d treated matched, %d dropped by caliper; "
                                "n_control = unique controls used (with replacement)"
                                % (md["n_matched"], md["n_treated"],
                                   md["n_treated_dropped_by_caliper"]))
            if outcome == "log1p_total":
                pct = float(np.exp(tau_pt) - 1.0)
                row.update(pct_effect=pct, multiplier_exp_tau=float(np.exp(tau_pt)),
                           medals_per_typical_country=pct * meta["mean_total_sample"],
                           medals_att_treated_rows=pct * meta["mean_total_treated"])
            rows.append(row)
            effects.setdefault(method, {})[outcome] = dict(
                tau=tau_pt, ci_low=st["ci_low"], ci_high=st["ci_high"],
                boot_mean=st["boot_mean"], boot_se=st["boot_se"],
                boot_reps=st["boot_reps"],
                **{k: v for k, v in row.items()
                   if k.startswith(("pct_", "multiplier", "medals"))})
            if method == "IPW":
                effects[method][outcome]["propensity_diagnostics"] = diag
    table = pd.DataFrame(rows)
    front = ["method", "outcome", "tau", "ci_low", "ci_high", "n_treated",
             "n_control", "note"]
    table = table[front + [c for c in table.columns if c not in front]]
    return table, effects


def balance_block(cc, pt):
    T = cc["host_flag"].to_numpy(float)
    X = cc[COVS].to_numpy(float)
    e_hat = pt["propensity"]
    _, b, md = pt["match"]
    w = e_hat[T == 0] / (1.0 - e_hat[T == 0])
    bal = []
    for k, c in enumerate(COVS):
        xt = X[T == 1, k]
        bal.append(dict(
            covariate=c,
            mean_treated=float(xt.mean()), mean_control=float(X[T == 0, k].mean()),
            smd_unmatched=smd(xt, X[T == 0, k]),
            smd_matched_nn=smd(xt, X[b, k]) if len(b) else np.nan,
            smd_ipw_att=smd(xt, X[T == 0, k], w_c=w),
            mean_control_matched=float(X[b, k].mean()) if len(b) else np.nan))
    return bal


def mediation_point(df, y):
    a1 = fe_betas(df, "log_gdp", ["host_flag", "lag1_total"], ["host_flag"])
    a2 = fe_betas(df, "log_pop", ["host_flag", "lag1_total"], ["host_flag"])
    b = fe_betas(df, y, ["host_flag", "log_gdp", "log_pop", "lag1_total"],
                 ["host_flag", "log_gdp", "log_pop"])
    c = fe_betas(df, y, ["host_flag", "lag1_total"], ["host_flag"])
    if None in (a1, a2, b, c):
        return None
    return dict(a1=a1["host_flag"], a2=a2["host_flag"], direct=b["host_flag"],
                b1=b["log_gdp"], b2=b["log_pop"], total_effect_c=c["host_flag"])


def mediation(cc, n_boot, seed, outdir):
    df = cc.copy()
    res_b, Z_b, info_b, srows_b, n_single = fe_fit(
        df, "log1p_total", ["host_flag", "log_gdp", "log_pop", "lag1_total"],
        ["host_flag", "log_gdp", "log_pop"])
    _, _, _, srows_a1, _ = fe_fit(df, "log_gdp", ["host_flag", "lag1_total"], ["host_flag"])
    _, _, _, srows_a2, _ = fe_fit(df, "log_pop", ["host_flag", "lag1_total"], ["host_flag"])
    _, _, _, srows_c, _ = fe_fit(df, "log1p_total", ["host_flag", "lag1_total"], ["host_flag"])

    pt = mediation_point(feols.drop_singletons(df)[0], "log1p_total")
    ind1, ind2 = pt["a1"] * pt["b1"], pt["a2"] * pt["b2"]
    total_dm = pt["direct"] + ind1 + ind2

    rng = np.random.default_rng([seed, STREAMS["mediation"]])
    data = feols.drop_singletons(df)[0]
    ioc_arr = data["ioc"].to_numpy()
    countries = list(pd.unique(ioc_arr))
    n_countries_est = len(countries)
    rows_by_ioc = [np.flatnonzero(ioc_arr == c) for c in countries]
    reset_fe_skips()
    keys = ["direct", "indirect_gdp", "indirect_pop", "total_dm", "total_effect_c"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick, idx = draw_indices(ioc_arr, countries, rows_by_ioc, rng)
        rep = data.iloc[idx].reset_index(drop=True)
        r = mediation_point(rep, "log1p_total")
        if r is None:
            continue
        draws["direct"].append(r["direct"])
        draws["indirect_gdp"].append(r["a1"] * r["b1"])
        draws["indirect_pop"].append(r["a2"] * r["b2"])
        draws["total_dm"].append(r["direct"] + r["a1"] * r["b1"] + r["a2"] * r["b2"])
        draws["total_effect_c"].append(r["total_effect_c"])
    med_skips = dict(FE_SKIPS)
    boot = {k: pct_ci(v) for k, v in draws.items()}

    paths = [("total_effect_c", pt["total_effect_c"], "total_effect_c"),
             ("direct", pt["direct"], "direct"),
             ("indirect_via_log_gdp", ind1, "indirect_gdp"),
             ("indirect_via_log_pop", ind2, "indirect_pop"),
             ("total_direct_plus_indirect", total_dm, "total_dm")]
    table = [dict(path=p, coef=v, boot_ci_low=boot[k]["ci_low"],
                  boot_ci_high=boot[k]["ci_high"],
                  share_of_total=(v / total_dm if abs(total_dm) > 1e-12 else np.nan))
             for p, v, k in paths]
    pd.DataFrame(table).to_csv(outdir / "mediation_table.csv", index=False)

    block = dict(
        outcome_primary="log1p_total", n_obs=int(res_b.nobs),
        n_countries_estimated=n_countries_est,
        singleton_countries_dropped=int(n_single), identification=info_b,
        replicate_skips=med_skips,
        equations=dict(a1_log_gdp_on_T=srows_a1, a2_log_pop_on_T=srows_a2,
                       b_outcome_on_T_and_mediators=srows_b,
                       c_total_effect=srows_c),
        log1p_total=dict(paths=table, summary=dict(
            total_effect_c=pt["total_effect_c"], direct=pt["direct"],
            indirect_via_log_gdp=ind1, indirect_via_log_pop=ind2,
            total_direct_plus_indirect=total_dm,
            share_direct=pt["direct"] / total_dm,
            share_indirect_gdp=ind1 / total_dm, share_indirect_pop=ind2 / total_dm,
            identity_note="c = direct + sum_k a_k x b_k holds by OLS algebra in "
                          "this nested specification (the c regressors are a subset "
                          "of the b regressors); it is a bookkeeping identity, not "
                          "an empirical test -- the part-1 tau estimates are the "
                          "independent cross-check",
            note="shares are ratios of log-point coefficients (approximate; log "
                 "coefficients do not add in levels)")),
        level_total_secondary=mediation_level(df),
        bootstrap=dict(reps_requested=n_boot, ci=boot,
                       skipped_reps=n_boot - boot["direct"]["boot_reps"]),
        caveat="log_gdp/log_pop are contemporaneous with hosting: the "
               "decomposition is descriptive, not a causal mediation claim")
    print("mediation (log1p):", json.dumps(block["log1p_total"]["summary"], indent=2))
    return block


def mediation_level(df):
    pt = mediation_point(feols.drop_singletons(df)[0], "total")
    ind1, ind2 = pt["a1"] * pt["b1"], pt["a2"] * pt["b2"]
    return dict(total_effect_c=pt["total_effect_c"], direct=pt["direct"],
                indirect_via_log_gdp=ind1, indirect_via_log_pop=ind2,
                total_direct_plus_indirect=pt["direct"] + ind1 + ind2,
                note="level outcome (medals); mediator coefficients are log-GDP / "
                     "log-pop elasticities, so shares are not reported")


MOD_EXTRA = ["host_flag", "T_x_mid", "T_x_high", "gdp_pc_mid", "gdp_pc_high",
             "log_gdp", "log_pop"]


def add_terciles(df):
    df = df.copy()
    df["gdp_pc_tercile"] = df.groupby("year")["gdp_per_capita_2011usd"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]))
    df["gdp_pc_mid"] = (df["gdp_pc_tercile"] == "mid").astype(float)
    df["gdp_pc_high"] = (df["gdp_pc_tercile"] == "high").astype(float)
    df["T_x_mid"] = df["host_flag"] * df["gdp_pc_mid"]
    df["T_x_high"] = df["host_flag"] * df["gdp_pc_high"]
    return df


def moderation_betas(df):
    return fe_betas(df, "log1p_total", MOD_EXTRA, ["host_flag", "T_x_mid", "T_x_high"],
                    prepare=add_terciles)


def moderation(cc, n_boot, seed, outdir):
    cols = ["ioc", "year", "country", "total", "host_flag", "log_gdp", "log_pop",
            "gdp_per_capita_2011usd", "lag1_total", "log1p_total"]
    df = cc[cols].copy()
    n0 = len(df)
    df = df[np.isfinite(df["gdp_per_capita_2011usd"].to_numpy(float))].copy()
    n_drop_gdppc = n0 - len(df)
    data, n_single = feols.drop_singletons(df)
    d = add_terciles(data)
    res, Z, info = feols.fe_ols(d, "log1p_total", extra=MOD_EXTRA, interest=MOD_EXTRA)
    srows = {t: feols.summary_row(res, t) for t in MOD_EXTRA}
    coef = {t: srows[t]["beta"] for t in MOD_EXTRA}

    rng = np.random.default_rng([seed, STREAMS["moderation"]])
    ioc_arr = data["ioc"].to_numpy()
    countries = list(pd.unique(ioc_arr))
    n_countries_est = len(countries)
    rows_by_ioc = [np.flatnonzero(ioc_arr == c) for c in countries]
    reset_fe_skips()
    draws = {"low": [], "mid": [], "high": []}
    for _ in range(n_boot):
        pick, idx = draw_indices(ioc_arr, countries, rows_by_ioc, rng)
        try:
            r = moderation_betas(data.iloc[idx].reset_index(drop=True))
        except Exception:
            FE_SKIPS["exception"] += 1
            continue
        if r is None:
            continue
        draws["low"].append(r["host_flag"])
        draws["mid"].append(r["host_flag"] + r["T_x_mid"])
        draws["high"].append(r["host_flag"] + r["T_x_high"])
    mod_skips = dict(FE_SKIPS)
    boot = {g: pct_ci(v) for g, v in draws.items()}
    mean_total = float(Z["total"].mean())

    mfx = {}
    for g, val in (("low", coef["host_flag"]),
                   ("mid", coef["host_flag"] + coef["T_x_mid"]),
                   ("high", coef["host_flag"] + coef["T_x_high"])):
        st = boot[g]
        z = val / st["boot_se"] if st["boot_se"] else np.nan
        mfx[g] = dict(marginal_effect=val, boot_ci_low=st["ci_low"],
                      boot_ci_high=st["ci_high"], boot_se=st["boot_se"],
                      boot_reps=st["boot_reps"], boot_z=z,
                      boot_p=float(2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))) if z == z else np.nan,
                      medals_per_typical_country=float((np.exp(val) - 1.0) * mean_total))

    src_fe, src_boot = "FE cluster-robust SE (analytic)", "country-cluster bootstrap"
    table = [dict(term=t, coef=srows[t]["beta"], se=srows[t]["se"], p=srows[t]["p"],
                  se_p_source=src_fe, ci_low=srows[t]["ci_lo"], ci_high=srows[t]["ci_hi"])
             for t in MOD_EXTRA]
    for g in ("low", "mid", "high"):
        m = mfx[g]
        table.append(dict(term=f"mfx_host_tercile_{g}", coef=m["marginal_effect"],
                          se=m["boot_se"], p=m["boot_p"], se_p_source=src_boot,
                          ci_low=m["boot_ci_low"], ci_high=m["boot_ci_high"]))
    pd.DataFrame(table).to_csv(outdir / "moderation_table.csv", index=False)

    block = dict(model="log1p(total) ~ T + T x mid + T x high + mid + high + "
                       "log_gdp + log_pop + country FE + Games FE (cluster SE)",
                 grouping="GDP-per-capita terciles within each Games edition "
                          "(per-edition rank; low = omitted category); terciles are "
                          "computed AFTER singleton countries are dropped, in the "
                          "point estimate and in every bootstrap replicate",
                 order_note="drop_singletons -> add_terciles -> design",
                 n_rows_estimated=int(res.nobs),
                 n_countries_estimated=n_countries_est,
                 n_rows_dropped_missing_gdppc=int(n_drop_gdppc),
                 singleton_countries_dropped=int(n_single), identification=info,
                 replicate_skips=mod_skips,
                 replicate_skip_note="replicates whose design is rank-deficient with "
                     "an interest coefficient (host_flag / T_x_mid / T_x_high) in the "
                     "null space are DROPPED, not silently estimated: a resample that "
                     "keeps host rows in only two terciles, or whose sparse country x "
                     "edition structure makes the interactions collinear with the FE "
                     "block, cannot identify them (checked by SVD as in feols.fe_ols; "
                     "null-space mass on the interest columns in the dropped "
                     "replicates is macroscopic, 0.58-1.0). Percentile CIs use the "
                     "remaining replicates (see marginal_effects[g].boot_reps); these "
                     "replicates would have contributed arbitrary min-norm draws.",
                 se_p_source=dict(regression_terms=src_fe,
                                  marginal_effects=src_boot + " (percentile CI; "
                                  "boot_p = two-sided normal approx from boot_se)"),
                 regression=table, marginal_effects=mfx)
    print("moderation:", pd.DataFrame(table).to_string(index=False))
    return block


def read_anchors():
    src = HERE / "gen" / "econometrics_host_effect_results.json"
    out = {"source_file": str(src)}
    try:
        specs = json.loads(src.read_text())["specs"]
    except Exception:
        return dict(out, available=False,
                    note="anchor file missing; re-run econometrics_host_effect.py")
    out["available"] = True
    out["A5_level_1976_2024_host_coef_medals"] = float(specs["A5_level_1976_2024"]["beta"])
    out["A6_log1p_1976_2024_medals_per_typical_country"] = float(
        specs["A6_log1p_1976_2024"]["medals_per_typical_country"])
    out["A6_log1p_1976_2024_medals_att_hosted"] = float(
        specs["A6_log1p_1976_2024"]["medals_att_hosted"])
    out["A2_log1p_1896_2024_medals_per_typical_country"] = float(
        specs["A2_log1p_1896_2024"]["medals_per_typical_country"])
    return out


def consistency_check(effects, anchors):
    if not anchors.get("available"):
        return dict(available=False, note=anchors.get("note", "anchors unavailable"))
    a6_typ = anchors["A6_log1p_1976_2024_medals_per_typical_country"]
    a6_att = anchors["A6_log1p_1976_2024_medals_att_hosted"]
    a5 = anchors["A5_level_1976_2024_host_coef_medals"]
    comp = {}
    for meth, d in effects.items():
        lvl = d["total"]["tau"]
        log_typ = d["log1p_total"].get("medals_per_typical_country")
        log_att = d["log1p_total"].get("medals_att_treated_rows")
        r_typ, r_lvl = (log_typ / a6_typ if log_typ else np.nan), lvl / a5
        comp[meth] = dict(
            tau_level_medals=lvl, tau_log_as_medals_typical_country=log_typ,
            tau_log_as_medals_att_treated=log_att, ratio_to_A6_typical=r_typ,
            ratio_to_A5_level=r_lvl,
            same_order_of_magnitude=bool(0.1 <= r_typ <= 10 and 0.1 <= r_lvl <= 10))
    lines = [f"{m}: level tau {c['tau_level_medals']:.2f} medals vs A5 FE {a5:.2f} "
             f"(ratio {c['ratio_to_A5_level']:.2f}); log tau -> "
             f"{c['tau_log_as_medals_typical_country']:.2f} medals/typical country vs "
             f"A6 FE {a6_typ:.2f} (ratio {c['ratio_to_A6_typical']:.2f}); ATT-hosted "
             f"conversion {c['tau_log_as_medals_att_treated']:.2f} vs A6 {a6_att:.2f}; "
             f"{'same order of magnitude' if c['same_order_of_magnitude'] else 'differs by an order of magnitude'}."
             for m, c in comp.items()]
    return dict(anchors=anchors, comparison=comp, verdict_text=" ".join(lines),
                note="A5/A6 are two-way (country + Games) FE associations over the "
                     "whole modern window without the lag1 covariate; the three "
                     "treatment-effect estimators use NO fixed effects "
                     "(X = {log_gdp, log_pop, lag1_total}) on the covariate-complete "
                     "subsample. The two layers are structurally different "
                     "specifications, so this is a magnitude comparison only, not a "
                     "same-specification replication -- point equality is neither "
                     "expected nor required.")


def fig_decomposition(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    import glob as _glob
    for _d in (os.environ.get("OLYMPIC_FONT_DIR", ""),
               "/usr/share/fonts/truetype/msttcorefonts", "/usr/share/fonts",
               "/Library/Fonts", "C:/Windows/Fonts"):
        if _d and os.path.isdir(_d):
            for _p in sorted(_glob.glob(os.path.join(_d, "[Aa]rial*.ttf"))):
                try:
                    _fm.fontManager.addfont(_p)
                except Exception:
                    pass
    _base = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5,
             "axes.linewidth": 0.8}
    try:
        _fm.findfont("Arial", fallback_to_default=False)
        plt.rcParams.update(dict(_base, **{"font.family": "Arial",
                                           "mathtext.fontset": "custom",
                                           "mathtext.rm": "Arial",
                                           "mathtext.it": "Arial:italic",
                                           "mathtext.bf": "Arial:bold"}))
    except ValueError:
        plt.rcParams.update(_base)
    med = results["mediation"]["log1p_total"]["summary"]
    boot = results["mediation"]["bootstrap"]["ci"]
    mean_total = results["sample"]["mean_total_sample"]
    items = [("Total effect $c$", med["total_effect_c"], "total_effect_c"),
             ("Direct effect $c'$", med["direct"], "direct"),
             ("Indirect via log GDP\n$a_1 b_{M1}$", med["indirect_via_log_gdp"],
              "indirect_gdp"),
             ("Indirect via log population\n$a_2 b_{M2}$",
              med["indirect_via_log_pop"], "indirect_pop")]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ypos = np.arange(len(items))[::-1].astype(float)
    for y, (label, val, key), shade in zip(ypos, items, BLUES):
        lo, hi = boot[key]["ci_low"], boot[key]["ci_high"]
        ax.barh(y, val, height=0.58, color=shade, edgecolor="black", linewidth=0.5,
                zorder=2)
        ax.errorbar(val, y, xerr=[[max(val - lo, 0.0)], [max(hi - val, 0.0)]],
                    fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=4)
        conv = np.expm1(val) * mean_total
        ax.annotate(f"{val:+.3f}  ({conv:+.1f} medals)", (max(hi, val), y),
                    textcoords="offset points", xytext=(5, -3), fontsize=8)
    ax.axvline(0, color="#444444", linewidth=0.9, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in items], fontsize=8.5)
    ax.set_xlabel(r"Effect on $\log(1+\mathrm{medals})$ per Games "
                  r"(parentheses: $\times$ sample-mean medal count)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_moderation(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    import glob as _glob
    for _d in (os.environ.get("OLYMPIC_FONT_DIR", ""),
               "/usr/share/fonts/truetype/msttcorefonts", "/usr/share/fonts",
               "/Library/Fonts", "C:/Windows/Fonts"):
        if _d and os.path.isdir(_d):
            for _p in sorted(_glob.glob(os.path.join(_d, "[Aa]rial*.ttf"))):
                try:
                    _fm.fontManager.addfont(_p)
                except Exception:
                    pass
    _base = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5,
             "axes.linewidth": 0.8}
    try:
        _fm.findfont("Arial", fallback_to_default=False)
        plt.rcParams.update(dict(_base, **{"font.family": "Arial",
                                           "mathtext.fontset": "custom",
                                           "mathtext.rm": "Arial",
                                           "mathtext.it": "Arial:italic",
                                           "mathtext.bf": "Arial:bold"}))
    except ValueError:
        plt.rcParams.update(_base)
    mfx = results["moderation"]["marginal_effects"]
    groups = [("Low", "low"), ("Middle", "mid"), ("High", "high")]
    tercile_colors = ["#6baed6", "#2171b5", "#08519c"]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    xpos = np.arange(len(groups)).astype(float)
    for x, (label, key), col in zip(xpos, groups, tercile_colors):
        m = mfx[key]
        est = m["marginal_effect"]
        lo, hi = m["boot_ci_low"], m["boot_ci_high"]
        ax.errorbar(x, est, yerr=[[max(est - lo, 0.0)], [max(hi - est, 0.0)]],
                    fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=5,
                    capthick=2.2, markersize=7.5, markeredgecolor="white",
                    markeredgewidth=1.1, zorder=4)
        ax.annotate("%.2f" % est, (x, est), textcoords="offset points",
                    xytext=(10, -3), ha="left", fontsize=8.5, color="#222222")
    ax.axhline(0, color="#444444", linewidth=0.9, zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{l}\nincome tercile" for l, _ in groups], fontsize=8.5)
    ax.set_ylabel(r"Hosting effect on $\log(1+\mathrm{medals})$")
    ax.set_xlabel("GDP per capita tercile within the same Games edition")
    ax.set_ylim(-0.05, 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run(panel, outdir, n_boot, seed, make_figures, selftest):
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    cc, meta = build_samples(panel)
    print("sample:", json.dumps({k: v for k, v in meta.items() if k != "treated_rows"}))
    methods = ("IPW",) if selftest else ("IPW", "Matching", "DML")

    pt = point_estimates(cc, seed)
    print("point estimates:", {f"{m}/{o}": round(v[0], 4)
                               for o in pt["outcomes"] for m, v in
                               pt["outcomes"][o].items()})

    reps, skips = bootstrap_tau(cc, n_boot, seed, methods)
    tau_table, effects = treatment_effect_block(cc, pt, reps, meta, methods)
    tau_table.to_csv(outdir / "tau_table.csv", index=False)
    print(tau_table.to_string(index=False))

    balance = balance_block(cc, pt)
    pd.DataFrame(balance).to_csv(outdir / "balance_smd.csv", index=False)
    print(pd.DataFrame(balance).to_string(index=False))

    e_hat = pt["propensity"]
    T = cc["host_flag"].to_numpy(float)
    prop = dict(
        estimator="Pipeline(StandardScaler, LogisticRegression(C=1, max_iter=1000))",
        min=float(e_hat.min()), max=float(e_hat.max()),
        treated_quantiles={str(q): float(np.quantile(e_hat[T == 1], q))
                           for q in (0, .25, .5, .75, 1)},
        control_quantiles={str(q): float(np.quantile(e_hat[T == 0], q))
                           for q in (0, .25, .5, .75, .9, .99, 1)},
        n_control_e_below_0p01=int((e_hat[T == 0] < 0.01).sum()),
        n_control_e_above_0p99=int((e_hat[T == 0] > 0.99).sum()),
        n_treated_e_below_0p01=int((e_hat[T == 1] < 0.01).sum()),
        n_treated_e_above_0p99=int((e_hat[T == 1] > 0.99).sum()))

    results = dict(
        script=Path(__file__).name, seed=seed, selftest=bool(selftest),
        panel=str(panel), generated_by="python; statsmodels FE via feols.py",
        sample=meta, methods=methods,
        bootstrap=dict(B=n_boot, unit="country (ioc), resampled with replacement",
                       rng="np.random.default_rng([seed, stream])", streams=STREAMS,
                       convention="backtest_period.bootstrap_period",
                       reps_used={m: effects[m]["total"]["boot_reps"] for m in methods},
                       reps_skipped=skips),
        treatment_effect=effects,
        matching_diagnostics=dict(
            estimator="1:1 NN with replacement on logit(e)",
            caliper=pt["outcomes"]["total"]["Matching"][1]["caliper"],
            caliper_rule="0.2 x SD(logit(e)) over the analysis sample",
            mean_abs_logit_distance=pt["outcomes"]["total"]["Matching"][1][
                "mean_abs_logit_distance"],
            n_treated=pt["outcomes"]["total"]["Matching"][1]["n_treated"],
            n_matched=pt["outcomes"]["total"]["Matching"][1]["n_matched"],
            n_treated_dropped_by_caliper=pt["outcomes"]["total"]["Matching"][1][
                "n_treated_dropped_by_caliper"],
            n_unique_controls_used=pt["outcomes"]["total"]["Matching"][1][
                "n_unique_controls_used"],
            note="matching depends only on (T, e), so one matched-pair set serves "
                 "both outcomes"),
        balance=balance,
        overlap=dict(
            propensity=prop,
            n_covariates_smd_gt_0p25=dict(
                unmatched=int(sum(abs(b["smd_unmatched"]) > 0.25 for b in balance)),
                matched_nn=int(sum(abs(b["smd_matched_nn"]) > 0.25 for b in balance)),
                ipw_att=int(sum(abs(b["smd_ipw_att"]) > 0.25 for b in balance))),
            max_control_att_weight=pt["outcomes"]["total"]["IPW"][1]["max_control_weight"],
            verdict="no extreme propensity scores (max e = %.3f) and bounded control "
                    "ATT weights (max %.3f): common support across the covariate "
                    "range is adequate; see balance_smd.csv for post-matching SMDs"
                    % (prop["max"], pt["outcomes"]["total"]["IPW"][1]["max_control_weight"])))

    results["mediation"] = mediation(cc, min(n_boot, 50) if selftest else n_boot,
                                     seed, outdir)
    results["moderation"] = moderation(cc, min(n_boot, 50) if selftest else n_boot,
                                       seed, outdir)
    results["consistency_check"] = consistency_check(results["treatment_effect"],
                                                     read_anchors())
    (outdir / "causal_results.json").write_text(json.dumps(results, indent=2))
    print("consistency check:", results["consistency_check"]["verdict_text"])

    if make_figures and not selftest:
        fig_decomposition(results, outdir / "fig_decomposition.pdf")
        fig_moderation(results, outdir / "fig_moderation.pdf")
    elif selftest:
        print("selftest: figures skipped")
    print("done in %.1f min -> %s" % ((time.time() - t0) / 60.0, outdir))
    return results


def main():
    ap = argparse.ArgumentParser(description="causal host-country effect: IPW / "
                                             "matching / DML + mediation + moderation")
    ap.add_argument("--panel", type=Path,
                    default=HERE.parent / "data" / "panel_clean.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "gen" / "causal")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw both PDFs from outdir/causal_results.json and exit "
                         "(no estimation)")
    ap.add_argument("--selftest", action="store_true",
                    help="B<=50, treatment-effect part IPW only, output to "
                         "gen/_smoke/causal/")
    args = ap.parse_args()
    if args.figures_only:
        results = json.loads((args.outdir / "causal_results.json").read_text())
        fig_decomposition(results, args.outdir / "fig_decomposition.pdf")
        fig_moderation(results, args.outdir / "fig_moderation.pdf")
        return
    if args.selftest:
        args.outdir = HERE / "gen" / "_smoke" / "causal"
        args.n_boot = min(args.n_boot, 50)
    run(args.panel, args.outdir, args.n_boot, args.seed,
        make_figures=not args.no_figures, selftest=args.selftest)


if __name__ == "__main__":
    main()
