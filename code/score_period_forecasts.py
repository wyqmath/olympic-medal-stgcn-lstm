import numpy as np
import pandas as pd

from evaluate_recent_editions import GEN
from evaluate_tokyo_roc import HISTORICAL_CODE, REPORTED_CODE, YEAR


OUTPUT = GEN / "linked_backtest"
MATCHED_SUMMARY = OUTPUT / "matched_period_summary.csv"
FORECASTS = {
    "model": OUTPUT / "predictions_nested.csv",
    "persistence": OUTPUT / "predictions_persistence.csv",
    "hgbr": GEN / "tokyo_linked" / "hgbr.csv",
}
PERIODS = [
    ("Cold War", 1976, 1988),
    ("Globalization", 1992, 2016),
    ("COVID-19", 2020, 2024),
]
N_RESAMPLES = 2000
SEED = 42
KEYS = ["year", "ioc"]


def load_common_forecasts():
    sources = {
        method: pd.read_csv(path).set_index(KEYS, verify_integrity=True).sort_index()
        for method, path in FORECASTS.items()
    }
    full = sources["model"]
    persistence = sources["persistence"]
    hgbr = sources["hgbr"]
    if len(full) != 916 or len(persistence) != 838:
        raise ValueError("Unexpected model or persistence sample size")
    if not hgbr.index.equals(full.index) or not hgbr["actual"].equals(full["actual"]):
        raise ValueError("The model and HGBR use different observations or actual counts")
    if not persistence.index.isin(full.index).all():
        raise ValueError("Persistence has rows outside the backtest")
    if not persistence["actual"].equals(full.loc[persistence.index, "actual"]):
        raise ValueError("Persistence and model samples have different observed counts")
    if persistence.loc[(YEAR, REPORTED_CODE), "predicted"] != 56:
        raise ValueError("The Tokyo benchmark did not use the Russian delegation's Rio count")

    common = persistence[["actual"]].copy()
    for method, source in sources.items():
        common[method] = source.loc[common.index, "predicted"]
    if not np.isfinite(common.to_numpy()).all() or common.lt(0).any().any():
        raise ValueError("The common forecast rows contain invalid counts")
    if common["actual"].le(0).any():
        raise ValueError("A row is outside the medal-winning evaluation population")
    return common


def shared_resample_weights(forecasts, period_index):
    years = forecasts.index.get_level_values("year").to_numpy()
    ioc = forecasts.index.get_level_values("ioc").to_numpy()
    delegates = np.where((years == YEAR) & (ioc == REPORTED_CODE), HISTORICAL_CODE, ioc)
    clusters, inverse = np.unique(delegates, return_inverse=True)
    rng = np.random.default_rng([SEED, period_index])
    selected = rng.integers(len(clusters), size=(N_RESAMPLES, len(clusters)))
    draws = np.stack([np.bincount(draw, minlength=len(clusters)) for draw in selected])
    return np.vstack((np.ones((1, len(clusters)), dtype=int), draws))[:, inverse]


def summarize_period(group, weights):
    actual = group["actual"].to_numpy(dtype=float)
    predicted = group[list(FORECASTS)].to_numpy(dtype=float)
    errors = 200 * np.abs(actual[:, None] - predicted) / (actual[:, None] + predicted)
    edition_means = []
    years = group.index.get_level_values("year")
    for year in years.unique():
        use = years == year
        edition_weights = weights[:, use]
        size = edition_weights.sum(axis=1, keepdims=True)
        totals = edition_weights @ errors[use]
        edition_means.append(
            np.divide(totals, size, out=np.full_like(totals, np.nan), where=size > 0)
        )
    macro = np.nanmean(edition_means, axis=0)
    medals = weights @ actual
    weighted = (weights @ (actual[:, None] * errors)) / medals[:, None]
    if not np.isfinite(macro).all() or not np.isfinite(weighted).all():
        raise ValueError("A bootstrap draw has no usable forecasts in this period")
    return macro, weighted


def add_estimate(row, scores, point_key, interval_key):
    row[point_key] = scores[0]
    row[f"{interval_key}_low"], row[f"{interval_key}_high"] = np.quantile(scores[1:], (0.025, 0.975))


def ranking_summary(group):
    estimates = []
    for _, edition in group.groupby(level="year", sort=False):
        observed_top = set(edition["actual"].nlargest(3).index)
        model_top = set(edition["model"].nlargest(3).index)
        rho = edition["actual"].rank().corr(edition["model"].rank())
        estimates.append((len(observed_top & model_top) / 3, rho))
    return np.mean(estimates, axis=0)


def write_period_summary():
    forecasts = load_common_forecasts()
    years = forecasts.index.get_level_values("year")
    rows = []
    for period_index, (period, lower, upper) in enumerate(PERIODS):
        weights = shared_resample_weights(forecasts, period_index)
        use = (years >= lower) & (years <= upper)
        group = forecasts.loc[use]
        macro, weighted = summarize_period(group, weights[:, use])
        editions = group.index.get_level_values("year").unique()
        top_three, spearman = ranking_summary(group)
        row = {
            "period": period,
            "start_year": lower,
            "end_year": upper,
            "games_years": "|".join(map(str, editions)),
            "n_games": len(editions),
            "n_obs": len(group),
            "n_bootstrap_replicates": N_RESAMPLES,
            "bootstrap_seed": SEED,
            "model_top3_hit_mean": top_three,
            "model_spearman_mean": spearman,
        }
        add_estimate(row, macro[:, 0], "smape_macro", "smape_macro_ci95")
        add_estimate(row, weighted[:, 0], "smape_weighted", "smape_weighted_ci95")
        add_estimate(row, macro[:, 1], "persistence_macro", "persistence_ci95")
        add_estimate(row, weighted[:, 1], "persistence_weighted", "persistence_weighted_ci95")
        add_estimate(row, macro[:, 2], "hgbr_macro", "hgbr_macro_ci95")
        add_estimate(row, weighted[:, 2], "hgbr_weighted", "hgbr_weighted_ci95")
        add_estimate(row, macro[:, 0] - macro[:, 1], "macro_delta", "macro_delta_ci95")
        add_estimate(row, weighted[:, 0] - weighted[:, 1], "weighted_delta", "weighted_delta_ci95")
        rows.append(row)
    if sum(row["n_obs"] for row in rows) != len(forecasts) or sum(row["n_games"] for row in rows) != 13:
        raise ValueError("Period definitions do not cover the common set of target editions")
    pd.DataFrame(rows).to_csv(MATCHED_SUMMARY, index=False, float_format="%.8f")


if __name__ == "__main__":
    write_period_summary()
