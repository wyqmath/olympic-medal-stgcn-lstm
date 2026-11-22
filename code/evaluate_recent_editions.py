import argparse
import csv
import math
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "code" / "gen"
EDITIONS = {2016: "Rio 2016", 2020: "Tokyo 2020", 2024: "Paris 2024"}


class MedalPrediction(NamedTuple):
    actual: int
    predicted: float


class Scores(NamedTuple):
    kl: float
    mae: float
    actual_total: int
    predicted_total: float
    largest_kl_ioc: str
    largest_kl_term: float


def read_predictions(path):
    by_year = {year: {} for year in EDITIONS}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["year"])
            if year not in EDITIONS:
                continue
            ioc = row["ioc"].strip()
            if not ioc or ioc in by_year[year]:
                raise ValueError(f"{path}: missing or duplicate IOC code for {year}: {ioc!r}")
            actual, predicted = int(row["actual"]), float(row["predicted"])
            if actual <= 0 or not math.isfinite(predicted) or predicted <= 0:
                raise ValueError(
                    f"{path}: expected positive finite medal counts for {year} {ioc}, "
                    f"got actual={actual}, predicted={predicted}"
                )
            by_year[year][ioc] = MedalPrediction(actual, predicted)
    return by_year


def score(predictions):
    actual_total = sum(row.actual for row in predictions.values())
    predicted_total = math.fsum(row.predicted for row in predictions.values())
    if actual_total <= 0 or predicted_total <= 0:
        raise ValueError("Medal shares require positive observed and predicted totals")
    terms = {}
    for ioc, row in predictions.items():
        observed_share = row.actual / actual_total
        predicted_share = row.predicted / predicted_total
        terms[ioc] = (
            math.inf if predicted_share == 0
            else observed_share * math.log(observed_share / predicted_share)
        )
    largest_ioc = max(terms, key=terms.get)
    mae = math.fsum(abs(row.actual - row.predicted) for row in predictions.values())
    return Scores(
        math.fsum(terms.values()),
        mae / len(predictions),
        actual_total,
        predicted_total,
        largest_ioc,
        terms[largest_ioc],
    )


def summarize(year, model, persistence):
    if len(model) < 10 or not persistence:
        raise ValueError(f"{year}: expected at least ten model rows and a nonempty comparison")
    for ioc, row in persistence.items():
        if ioc not in model or row.actual != model[ioc].actual:
            raise ValueError(f"{year} {ioc}: observation differs between the two inputs")
    model_common = {ioc: model[ioc] for ioc in persistence}
    full = score(model)
    matched = score(model_common)
    baseline = score(persistence)
    return {
        "year": year,
        "n_awarded": len(model),
        "n_common": len(model_common),
        "kl_model_all": full.kl,
        "kl_model_common": matched.kl,
        "kl_persistence_common": baseline.kl,
        "mae_model_common": matched.mae,
        "mae_persistence_common": baseline.mae,
        "actual_total_all": full.actual_total,
        "model_total_all": full.predicted_total,
        "actual_total_common": matched.actual_total,
        "model_total_common": matched.predicted_total,
        "persistence_total_common": baseline.predicted_total,
        "largest_kl_ioc_model_all": full.largest_kl_ioc,
        "largest_kl_term_model_all": full.largest_kl_term,
    }


def top_ten(year, predictions):
    ordered = sorted(predictions, key=lambda ioc: (-predictions[ioc].actual, ioc))
    return [
        {"year": year, "rank": rank, "ioc": ioc, **predictions[ioc]._asdict()}
        for rank, ioc in enumerate(ordered[:10], start=1)
    ]


def draw_figure(path, panels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(3, 1, sharey=True, figsize=(7.2, 5.5))
    largest = max(max(row["actual"], row["predicted"]) for panel in panels for row in panel)
    ymax = 25 * math.ceil(1.04 * largest / 25)
    for ax, edition, rows in zip(axes, EDITIONS.values(), panels):
        positions = range(len(rows))
        ax.bar(
            [x - 0.19 for x in positions], [row["actual"] for row in rows],
            width=0.34, color="#b7c2ce", edgecolor="#566775", linewidth=0.6, label="Actual",
        )
        ax.bar(
            [x + 0.19 for x in positions], [row["predicted"] for row in rows],
            width=0.34, color="#205671", edgecolor="#205671", linewidth=0.6, label="Predicted",
        )
        ax.set_title(edition, loc="left", fontsize=11, fontweight="semibold", pad=5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.55, len(rows) - 0.45)
        ax.set_xticks(list(positions), [row["ioc"] for row in rows])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.tick_params(axis="both", length=0, labelsize=9, pad=4)
        ax.grid(axis="y", color="#d9dee3", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#657583")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.988, 0.995),
        ncol=2, frameon=False, fontsize=10, handlelength=1.5,
    )
    fig.supylabel("Total medals", x=0.022, fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.895, bottom=0.065, hspace=0.58)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the 2016 to 2024 medal forecasts and draw their comparison.")
    linked = GEN / "linked_backtest"
    parser.add_argument("--model", type=Path, default=linked / "predictions_nested.csv")
    parser.add_argument(
        "--persistence", type=Path, default=linked / "predictions_persistence.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=GEN)
    parser.add_argument("--figure", type=Path, default=GEN / "fig_recent_editions.png")
    args = parser.parse_args(argv)
    model = read_predictions(args.model)
    persistence = read_predictions(args.persistence)
    metrics = [summarize(year, model[year], persistence[year]) for year in EDITIONS]
    panels = [top_ten(year, model[year]) for year in EDITIONS]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "recent_editions_metrics.csv", metrics)
    write_csv(args.out_dir / "recent_editions_top10.csv", [row for panel in panels for row in panel])
    draw_figure(args.figure, panels)


if __name__ == "__main__":
    main()
