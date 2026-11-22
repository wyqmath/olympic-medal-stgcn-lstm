import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_recent_editions import GEN
from run_ablation import ARMS, CALLER_DIRECTORY, HISTORICAL_CODE, OUTDIR, REPORTED_CODE, YEAR
from run_ablation import baselines, nested


KEYS = ["year", "ioc"]
LINKED = GEN / "linked_backtest"
METRICS = ("rmse", "r2", "macro_smape", "weighted_smape", "top3_hit_mean", "spearman_mean")
BENCHMARK_METHODS = ("persistence", "arima", "xgboost", "hgbr", "pure_lstm", "backbone", "tt_ensemble")
COMPARISONS = (
    ("full", "no_rame"),
    ("full", "no_pagerank"),
    ("full", "self_loops"),
    ("self_loops", "no_graph"),
    ("full", "no_graph"),
)


def load_forecasts(directory=None):
    directory = OUTDIR if directory is None else directory
    reference = pd.read_csv(GEN / "baselines" / "predictions_backbone.csv")
    reference = reference.set_index(KEYS, verify_integrity=True).sort_index()
    common = pd.read_csv(LINKED / "predictions_persistence.csv")
    common = common.set_index(KEYS, verify_integrity=True).sort_index()
    if len(reference) != 916 or len(common) != 838:
        raise ValueError("The published evaluation population has changed")
    if not common["actual"].eq(reference.loc[common.index, "actual"]).all():
        raise ValueError("The published model and persistence counts disagree")

    protocol, parameter_counts, frames = None, {}, {}
    records, splits = [], {}
    for arm in ARMS:
        forecasts = []
        parameters = set()
        for year in nested.TARGET_YEARS:
            path = directory / "fits" / arm / f"{year}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["arm"] != arm or record["year"] != year:
                raise ValueError(f"Unexpected fold identity: {path}")
            if protocol is None:
                protocol = record["protocol"]
            if protocol != record["protocol"]:
                raise ValueError(f"Training settings differ in {path}")
            split = (record["train_editions"], record["fit"]["val_editions"])
            if splits.setdefault(year, split) != split:
                raise ValueError(f"The training or validation editions differ in {path}")
            earlier = split[0] + split[1]
            if not earlier or max(earlier) >= year or len(set(earlier)) != len(earlier):
                raise ValueError(f"Training/validation are not strictly pre-target in {path}")
            if set(record["intervention"]) != {"training", "validation", "target"}:
                raise ValueError(f"Missing input verification for {path}")
            forecasts.extend(record["predictions"])
            parameters.add(record["parameters"])
            records.append({
                "arm": arm, "year": year, **record["fit"],
                "blend_theta": record["blend_theta"], "parameters": record["parameters"],
            })
        if len(parameters) != 1:
            raise ValueError(f"Parameter count varies across {arm} fits")
        parameter_counts[arm] = parameters.pop()
        frame = pd.DataFrame(forecasts).set_index(KEYS, verify_integrity=True).sort_index()
        if not frame.index.equals(reference.index) or not frame["actual"].eq(reference["actual"]).all():
            raise ValueError(f"Prediction identity or actual medal count changed for {arm}")
        if not np.isfinite(frame["predicted"]).all() or frame["predicted"].lt(0).any():
            raise ValueError(f"Invalid predictions for {arm}")
        frame["country"] = reference["country"]
        frames[arm] = frame

    if any(parameter_counts[arm] != parameter_counts["full"] for arm in ARMS[:-1]):
        raise ValueError("The feature or self-loop intervention altered the parameter count")
    if parameter_counts["no_graph"] >= parameter_counts["full"]:
        raise ValueError("The no-graph reference still has the full graph encoder size")
    return frames, common.index, protocol, records


def score_frames(frames, common):
    metrics, by_edition = [], []
    for arm, forecasts in frames.items():
        matched = forecasts.loc[common].reset_index()
        scores, _ = baselines.evaluate_method(arm, matched)
        metrics.append({name: scores[name] for name in ("method", "n_rows", "n_editions", *METRICS)})
        for year, group in matched.groupby("year"):
            scores, _ = baselines.evaluate_method(arm, group)
            by_edition.append({"arm": arm, "year": year, "n": len(group),
                               **{name: scores[name] for name in METRICS}})
    return pd.DataFrame(metrics), pd.DataFrame(by_edition)


def paired_diagnostics(frames, common):
    actual = frames["full"].loc[common, "actual"].to_numpy(dtype=float)
    years = common.get_level_values("year").to_numpy()
    nodes = common.get_level_values("ioc").to_numpy()
    grouping = np.where((years == YEAR) & (nodes == REPORTED_CODE), HISTORICAL_CODE, nodes)
    nations, cluster = np.unique(grouping, return_inverse=True)
    editions, yi = np.unique(years, return_inverse=True)
    n_groups, n_editions = len(nations), len(editions)
    n = np.zeros((n_editions, n_groups))
    np.add.at(n, (yi, cluster), 1)
    medals = np.bincount(cluster, weights=actual, minlength=n_groups)
    score = {}
    for arm, frame in frames.items():
        forecast = frame.loc[common, "predicted"].to_numpy()
        score[arm] = 200 * np.abs(actual - forecast) / (actual + forecast)

    rng = np.random.default_rng(42)
    copies = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=2000)
    edition_sizes = n.sum(axis=1)
    boot_sizes = copies @ n.T
    boot_medals = copies @ medals
    results, contributions = [], []
    for reference, reduced in COMPARISONS:
        diff = score[reduced] - score[reference]
        sums = np.zeros((n_editions, n_groups))
        np.add.at(sums, (yi, cluster), diff)
        weighted_sums = np.bincount(cluster, weights=actual * diff, minlength=n_groups)
        edition_deltas = sums.sum(axis=1) / edition_sizes
        boot_sums = copies @ sums.T
        boot_means = np.divide(boot_sums, boot_sizes,
                               out=np.zeros_like(boot_sums), where=boot_sizes > 0)
        macro_boot = boot_means.sum(axis=1) / (boot_sizes > 0).sum(axis=1)
        weighted_boot = copies @ weighted_sums / boot_medals
        macro_ci = np.quantile(macro_boot, [0.025, 0.975])
        weighted_ci = np.quantile(weighted_boot, [0.025, 0.975])
        without_edition = (edition_deltas.sum() - edition_deltas) / (n_editions - 1)
        without_delegation = (weighted_sums.sum() - weighted_sums) / (medals.sum() - medals)
        macro_deltas = (sums / edition_sizes[:, None]).sum(axis=0) / n_editions
        weighted_deltas = weighted_sums / medals.sum()
        worst = int(np.argmax(np.abs(weighted_deltas)))
        results.append({
            "reference": reference, "reduced": reduced,
            "macro_delta": edition_deltas.mean(), "macro_ci_low": macro_ci[0], "macro_ci_high": macro_ci[1],
            "weighted_delta": weighted_deltas.sum(), "weighted_ci_low": weighted_ci[0],
            "weighted_ci_high": weighted_ci[1], "editions_higher_macro": int((edition_deltas > 0).sum()),
            "macro_without_one_edition_min": without_edition.min(),
            "macro_without_one_edition_max": without_edition.max(),
            "weighted_without_one_delegation_min": without_delegation.min(),
            "weighted_without_one_delegation_max": without_delegation.max(),
            "largest_weighted_contribution_ioc": nations[worst],
            "largest_weighted_contribution": weighted_deltas[worst],
        })
        contributions.extend(
            {"reference": reference, "reduced": reduced, "ioc": ioc,
             "macro_contribution": macro, "weighted_contribution": weighted}
            for ioc, macro, weighted in zip(nations, macro_deltas, weighted_deltas)
        )
    return pd.DataFrame(results), pd.DataFrame(contributions)


def update_benchmark_scores(metrics, reference):
    baseline_table = pd.read_csv(LINKED / "metrics_table.csv").set_index("method", verify_integrity=True)
    hgbr = pd.read_csv(GEN / "tokyo_linked" / "hgbr.csv").set_index(KEYS, verify_integrity=True)
    hgbr = hgbr.loc[reference.index]
    if not hgbr["actual"].eq(reference["actual"]).all():
        raise ValueError("HGBR actual medal counts disagree with the benchmark")
    if not np.isfinite(hgbr["predicted"]).all() or hgbr["predicted"].lt(0).any():
        raise ValueError("Invalid HGBR forecasts for the benchmark")
    hgbr = hgbr.reset_index()
    scores, _ = baselines.evaluate_method("hgbr", hgbr)
    scores.update(baselines.ranking_means(hgbr))
    baseline_table.loc["hgbr", list(METRICS)] = [scores[key] for key in METRICS]
    lookup = metrics.set_index("method")
    for entry, arm in (("backbone", "full"), ("pure_lstm", "no_graph")):
        baseline_table.loc[entry, list(METRICS)] = lookup.loc[arm, list(METRICS)]
    return baseline_table.loc[list(BENCHMARK_METHODS), list(METRICS)].reset_index()


def main():
    parser = argparse.ArgumentParser(description="Score matched ablations and benchmark models.")
    parser.add_argument("--results-dir", type=Path, default=OUTDIR)
    directory = (CALLER_DIRECTORY / parser.parse_args().results_dir).resolve()
    frames, common, protocol, fits = load_forecasts(directory)
    metrics, by_edition = score_frames(frames, common)
    paired, contributions = paired_diagnostics(frames, common)
    benchmark = update_benchmark_scores(metrics, frames["full"].loc[common])
    predictions = pd.concat(frames, names=["arm"]).reset_index()
    predictions.to_csv(directory / "predictions.csv", index=False)
    by_edition.to_csv(directory / "metrics_by_edition.csv", index=False)
    metrics.to_csv(directory / "metrics.csv", index=False)
    paired.to_csv(directory / "paired_smape.csv", index=False)
    contributions.to_csv(directory / "paired_delegation_contributions.csv", index=False)
    benchmark.to_csv(directory / "benchmark_metrics.csv", index=False)
    (directory / "protocol.json").write_text(
        json.dumps(
            {"training": protocol, "scored_rows": len(common),
             "bootstrap": {"paired": True, "cluster": "delegation", "draws": 2000,
                           "seed": 42, "delta": "reduced minus reference"},
             "fits": fits},
            indent=2, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
