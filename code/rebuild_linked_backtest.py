import argparse
import json

import pandas as pd

from evaluate_recent_editions import GEN
from evaluate_tokyo_roc import PANEL, YEAR
import backtest_period as periods
import baselines_compare as baselines
import nested_backtest as nested


TOKYO = GEN / "tokyo_linked"
OUTPUT = GEN / "linked_backtest"
BASELINES = GEN / "baselines"
COLS = ["country", "ioc", "year", "actual", "predicted"]


def replace_tokyo(unlinked, tokyo, expected_extra=0):
    previous = unlinked.loc[unlinked["year"].eq(YEAR)]
    previous_actuals = previous.set_index("ioc")["actual"]
    updated_actuals = tokyo.set_index("ioc")["actual"]
    if not updated_actuals.index.is_unique or not updated_actuals.reindex(previous_actuals.index).equals(previous_actuals):
        raise ValueError("Tokyo actual medal counts do not align")
    if len(updated_actuals) - len(previous_actuals) != expected_extra:
        raise ValueError("Unexpected change in Tokyo delegation coverage")
    labels = tokyo["ioc"].map(previous.set_index("ioc")["country"])
    tokyo = tokyo.assign(country=labels.fillna(tokyo["country"]))
    combined = pd.concat([unlinked.loc[unlinked["year"].ne(YEAR)], tokyo[COLS]], ignore_index=True)
    return combined.sort_values(["year", "actual", "ioc"], ascending=[True, False, True]).reset_index(drop=True)


def get_candidates():
    official = nested.load_variant_csvs(str(GEN / "nested" / "variants"), log=lambda *_: None)
    v1 = pd.read_csv(GEN / "tokyo_roc_sensitivity.csv").rename(
        columns={"linked_refit": "predicted"},
    )
    reference = official["V1"].loc[official["V1"]["year"].eq(YEAR), ["year", "ioc", "country"]]
    v1 = v1.merge(reference, on=["year", "ioc"], how="left", validate="one_to_one")
    if v1["country"].isna().any():
        raise ValueError("The Tokyo host-interaction refit differs in coverage")
    corrected = {}
    for tag, frame in official.items():
        updated = v1 if tag == "V1" else pd.read_csv(TOKYO / f"{tag}.csv")
        corrected[tag] = replace_tokyo(frame, updated)
    nested.assert_cross_variant(corrected)
    return corrected


def write_benchmark_results(backbone):
    frames = {"backbone": backbone}
    for method in baselines.METHOD_ORDER:
        if method == "backbone":
            continue
        frames[method] = replace_tokyo(
            pd.read_csv(BASELINES / f"predictions_{method}.csv"),
            pd.read_csv(TOKYO / f"{method}.csv"),
            expected_extra=int(method == "persistence"),
        )
    persistence = frames["persistence"]
    persistence.to_csv(OUTPUT / "predictions_persistence.csv", index=False)
    common = set.intersection(*[
        set(zip(frame["ioc"].astype(str), frame["year"].astype(int))) for frame in frames.values()
    ])
    if len(common) != 838:
        raise ValueError(f"Expected 838 matched benchmark rows, found {len(common)}")
    metrics, support = [], []
    for method in baselines.METHOD_ORDER:
        subset = baselines.restrict(frames[method], common)
        row, _ = baselines.evaluate_method(method, subset)
        metrics.append(row)
        uncommon = [(ioc, year) not in common for ioc, year in zip(frames[method]["ioc"], frames[method]["year"])]
        forecast_without_baseline = frames[method].loc[uncommon, "predicted"]
        support.append({
            "method": method, "n": len(forecast_without_baseline),
            "share_at_least_one": float(forecast_without_baseline.ge(1).mean()),
        })
    pd.DataFrame(metrics).to_csv(OUTPUT / "metrics_table.csv", index=False)
    pd.DataFrame(support).to_csv(OUTPUT / "forecast_availability.csv", index=False)


def write_boundary(forecasts):
    panel = pd.read_csv(PANEL, usecols=["ioc", "year", "gdp_total_2011usd_million"])
    merged = forecasts.merge(panel, on=["ioc", "year"], validate="one_to_one")
    gdp = "gdp_total_2011usd_million"
    merged = merged.dropna(subset=[gdp])
    merged["error"] = periods.smape_obs(merged["actual"].to_numpy(), merged["predicted"].to_numpy())
    merged["tercile"] = pd.qcut(merged[gdp], 3, labels=["low", "mid", "high"])
    rows = []
    for name, group in [*merged.groupby("tercile", observed=True), ("all", merged)]:
        rows.append({
            "tercile": name, "n": len(group), "macro": group["error"].mean(),
            "weighted": (group["error"] * group["actual"]).sum() / group["actual"].sum(),
        })
    pd.DataFrame(rows).to_csv(OUTPUT / "gdp_terciles.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Rebuild the linked backtest, benchmark inputs, and 2028 configuration selection.")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    frames = get_candidates()
    raw = nested.inner_scores(frames)
    selections = [
        nested.select_target(year, raw, nested.DEFAULT_TOL_PP)
        for year in [*nested.TARGET_YEARS, 2028]
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nested.selection_table(selections).to_csv(OUTPUT / "selection_table.csv", index=False)
    (OUTPUT / "forward_selection.json").write_text(json.dumps(selections[-1], indent=2) + "\n", encoding="utf-8")
    print(f"Selected specifications for 2020, 2024, 2028: {[item['winner'] for item in selections[-3:]]}")
    if args.selection_only:
        return
    forecasts = nested.nested_predictions(frames, selections[:-1])
    forecasts.to_csv(OUTPUT / "predictions_nested.csv", index=False)
    write_benchmark_results(frames["base"])
    write_boundary(forecasts)


if __name__ == "__main__":
    main()
