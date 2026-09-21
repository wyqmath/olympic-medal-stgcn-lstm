import json
from pathlib import Path

import numpy as np
import pandas as pd

import feols

SEED = 42
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "data" / "panel_clean.csv"
OUT = HERE / "gen" / "econometrics_var_decomp_results.json"

ECON = ["log_gdp", "log_pop"]
INST = ["host_flag"]


def block_fit(res, Z, cols):
    beta = np.asarray(res.params, float)
    X = Z[res.model.exog_names].to_numpy(float)
    out = np.zeros(len(Z))
    for c in cols:
        j = res.model.exog_names.index(c)
        out += X[:, j] * beta[j]
    return out


def shares(y, blocks, resid):
    y = np.asarray(y, float)
    vy = float(np.var(y, ddof=0))
    covs = {k: float(np.cov(v, y, ddof=0)[0, 1]) for k, v in blocks.items()}
    covs["Resid"] = float(np.var(resid, ddof=0))
    total = sum(covs.values())
    return dict(var_y=vy, cov_contributions=covs, sum_of_parts=total,
                shares={k: v / total for k, v in covs.items()})


def main():
    df, meta = feols.load_panel(PANEL, drop_missing_gdp=True)
    df = feols.add_logvars(df)
    df = df[df["population"].notna()].copy()
    df, n_single = feols.drop_singletons(df)

    extra = ECON + INST
    res, Z, info = feols.fe_ols(df, "log1p_total", extra=extra,
                                interest=ECON + INST)
    y = Z["log1p_total"].to_numpy(float)
    resid = np.asarray(res.resid, float)
    yhat = np.asarray(res.fittedvalues, float)
    fe_contrib = yhat - (block_fit(res, Z, ECON) + block_fit(res, Z, INST))

    coefs = {c: feols.summary_row(res, c) for c in ECON + INST}

    parts = dict(Econ=block_fit(res, Z, ECON), Inst=block_fit(res, Z, INST))
    cov_y = {k: float(np.cov(v, y, ddof=0)[0, 1]) for k, v in parts.items()}
    resid_var = float(np.var(resid, ddof=0))
    denom = cov_y["Econ"] + cov_y["Inst"] + resid_var
    d1 = dict(denominator="cov(yhat_Econ,y) + cov(yhat_Inst,y) + var(e)",
              var_y=float(np.var(y, ddof=0)),
              cov_econ=cov_y["Econ"], cov_inst=cov_y["Inst"], resid_var=resid_var,
              fe_cov_excluded=float(np.cov(fe_contrib, y, ddof=0)[0, 1]),
              shares=dict(Econ=cov_y["Econ"] / denom, Inst=cov_y["Inst"] / denom,
                          Rand=resid_var / denom))
    d1_total = shares(y, dict(Econ=parts["Econ"], Inst=parts["Inst"],
                              FE=fe_contrib), resid)

    ioc = Z["ioc"].to_numpy()
    yr = Z["year"].to_numpy()

    def demean_two_way(col):
        s = pd.Series(np.asarray(col, float), index=Z.index)
        cm = pd.Series(s.to_numpy()).groupby(ioc).transform("mean")
        ym = pd.Series(s.to_numpy()).groupby(yr).transform("mean")
        return s.to_numpy() - cm.to_numpy() - ym.to_numpy() + s.mean()

    y_w = demean_two_way(Z["log1p_total"])
    e_w = demean_two_way(resid)
    blocks_w = {k: demean_two_way(v) for k, v in parts.items()}
    vy_w = float(np.var(y_w, ddof=0))
    cov_w = {k: float(np.cov(v, y_w, ddof=0)[0, 1]) for k, v in blocks_w.items()}
    var_e_w = float(np.var(e_w, ddof=0))
    denom_w = cov_w["Econ"] + cov_w["Inst"] + var_e_w
    d2 = dict(var_y_within=vy_w, cov_econ=cov_w["Econ"], cov_inst=cov_w["Inst"],
              resid_var=var_e_w,
              shares=dict(Econ=cov_w["Econ"] / denom_w, Inst=cov_w["Inst"] / denom_w,
                          Rand=var_e_w / denom_w))

    out = dict(seed=SEED, sample=dict(
        rows_raw=meta["rows_raw"], rows_mixed_team_dropped=meta["rows_mixed_team_dropped"],
        rows_gdp_missing_dropped=meta["rows_gdp_missing_dropped"],
        n_rows_estimated=int(info["n_obs"]),
        rows_dropped_singleton_countries=int(n_single),
        n_countries=int(Z["ioc"].nunique()), n_games=int(Z["year"].nunique())),
        specification=dict(outcome="log1p(total)", econ_block=ECON, inst_block=INST,
                           fe="country FE + Games FE", cluster="country",
                           r2=float(1 - resid.var() / np.var(y))),
        coefficients=coefs,
        identification=info,
        D1_shares_excl_FE=d1, D1_shares_of_total_variance=d1_total, D2_within_shares=d2,
        paper_old_values=dict(econ=0.68, inst=0.22, rand=0.10,
                              source="main.tex L516-523 (eq:error_decomp)"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(out["sample"])
    for c, s in coefs.items():
        print(f"  {c:9s} b={s['beta']: .4f} (se {s['se']:.4f}, p={s['p']:.4g})")
    print(f"R2={out['specification']['r2']:.4f}  n={info['n_obs']}")
    print("D1 (excl FE):", {k: round(v, 4) for k, v in d1["shares"].items()})
    print("D1 of var(y):", {k: round(v, 4) for k, v in d1_total["shares"].items()})
    print("D2 within   :", {k: round(v, 4) for k, v in d2["shares"].items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
