import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

from evaluate_recent_editions import GEN, ROOT, MedalPrediction, read_predictions, score


YEAR = 2020
HISTORICAL_CODE = "RUS"
REPORTED_CODE = "ROC"
PANEL = ROOT / "data" / "panel_clean.csv"
NESTED = GEN / "nested"
BASELINE = GEN / "baselines" / "predictions_persistence.csv"
IOC_GUIDANCE = (
    "https://stillmed.olympics.com/media/Document%20Library/OlympicOrg/News/"
    "2021/02/Principles-implementation-CAS-WADA-Rusada.pdf"
)


def selected_tag():
    with (NESTED / "selection_table.csv").open(encoding="utf-8", newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if int(row["year"]) == YEAR)
    return row["winner"]


def fit_tokyo(tag, seed, threads):
    import numpy as np
    import pandas as pd
    import torch
    import torch_geometric

    import stgcn_variants as variants

    sp = variants.sp
    cfg = variants.VARIANTS[tag]
    if tag != "V1" or not cfg["aug"] or cfg["host"] or cfg.get("hgbr"):
        raise ValueError(f"Expected the host-interaction configuration, got {tag}")

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    panel, info = variants.load_panel(str(PANEL))
    nodes = {ioc: idx for idx, ioc in enumerate(sorted(panel["ioc"].unique()))}
    editions = [year for year in info["years"] if year <= YEAR]
    earlier = [year for year in editions if year < YEAR]
    history = panel.loc[panel["year"] <= YEAR]
    samples = sp.build_samples(history, nodes, editions)
    aug = variants.augmented_statics(history, nodes, samples, editions)
    samples = variants.with_static(samples, aug)
    if any(hist_year >= year for year, s in samples.items() for hist_year in s["hist_years"]):
        raise ValueError("A feature window includes its own target edition or later data")
    if (history.loc[history["ioc"] == REPORTED_CODE, "year"] < YEAR).any():
        raise ValueError("The reported code unexpectedly has a pre-Tokyo record")

    device = torch.device("cpu")
    sp.set_seed(seed)
    model, meta = variants.train_target_v(
        samples, YEAR, earlier, device, samples[YEAR]["static"].shape[1],
        variants.make_loss(cfg["loss"]),
    )
    if not meta["val_editions"] or max(meta["val_editions"]) >= YEAR:
        raise ValueError("The blending validation window is invalid")
    theta, _ = variants.select_mix(
        model, samples, meta["val_editions"], device,
        grid=variants.THETA_GRIDS[cfg["theta"]], tol_pp=variants.THETA_TOL_PP[cfg["theta"]],
    )[0]
    raw = sp.predict_target(model, samples, YEAR, device)
    anchor = np.expm1(samples[YEAR]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
    forecasts = np.maximum(theta * raw + (1.0 - theta) * anchor, 0.0)
    if not np.isfinite(forecasts).all():
        raise ValueError("The refit produced a non-finite forecast")
    prior = history.loc[(history["ioc"] == HISTORICAL_CODE) & (history["year"] < YEAR)]
    last = prior.sort_values("year").iloc[-1]
    if anchor[nodes[REPORTED_CODE]] != 0 or not math.isclose(
        anchor[nodes[HISTORICAL_CODE]], last["total"], rel_tol=1e-6
    ):
        raise ValueError("The Tokyo inputs do not reflect the expected code-specific histories")
    meta.update(
        blend_weight=theta,
        seed=seed,
        threads=threads,
        torch=torch.__version__,
        torch_geometric=torch_geometric.__version__,
        pandas=pd.__version__,
        numpy=np.__version__,
        platform=platform.system(),
        python=platform.python_version(),
        training_editions=sp.split_train_val(earlier)[0],
        last_rus_edition=int(last["year"]),
        last_rus_total=int(last["total"]),
    )
    return {ioc: float(forecasts[idx]) for ioc, idx in nodes.items()}, meta


def metric_row(predictions):
    metrics = score(predictions)
    return {
        "kl": metrics.kl if math.isfinite(metrics.kl) else "inf",
        "mae": metrics.mae,
        "actual_total": metrics.actual_total,
        "predicted_total": metrics.predicted_total,
    }


def evaluate(forecasts, meta, tag):
    unlinked = read_predictions(NESTED / "predictions_nested.csv")[YEAR]
    persistence = read_predictions(BASELINE)[YEAR]
    roc_actual = unlinked[REPORTED_CODE].actual
    if REPORTED_CODE in persistence or HISTORICAL_CODE in unlinked:
        raise ValueError("The original Tokyo samples do not match the case being tested")
    if any(ioc not in unlinked or unlinked[ioc].actual != row.actual
           for ioc, row in persistence.items()):
        raise ValueError("Persistence and model results have different observed counts")
    original = {
        ioc: MedalPrediction(row.actual, forecasts[ioc]) for ioc, row in unlinked.items()
    }
    linked = {
        **original, REPORTED_CODE: MedalPrediction(roc_actual, forecasts[HISTORICAL_CODE])
    }
    persistence[REPORTED_CODE] = MedalPrediction(roc_actual, meta["last_rus_total"])
    common = list(persistence)
    results = {
        "year": YEAR,
        "variant": tag,
        "ioc_guidance": IOC_GUIDANCE,
        "training": meta,
        "source_sha256": {
            path.relative_to(ROOT).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                PANEL, NESTED / "predictions_nested.csv", NESTED / "selection_table.csv",
                BASELINE, ROOT / "code" / "stgcn_predict.py",
                ROOT / "code" / "stgcn_variants.py",
            )
        },
        "roc": {
            "actual": roc_actual,
            "unlinked_forecast": unlinked[REPORTED_CODE].predicted,
            "exact_code_refit": original[REPORTED_CODE].predicted,
            "linked_refit": linked[REPORTED_CODE].predicted,
            "linked_persistence": persistence[REPORTED_CODE].predicted,
        },
        "full": {
            "n": len(original),
            "exact_code_refit": metric_row(original),
            "linked_refit": metric_row(linked),
        },
        "common": {
            "n": len(common),
            "exact_code_refit": metric_row({ioc: original[ioc] for ioc in common}),
            "linked_refit": metric_row({ioc: linked[ioc] for ioc in common}),
            "linked_persistence": metric_row(persistence),
        },
    }
    rows = [
        {
            "year": YEAR,
            "ioc": ioc,
            "actual": row.actual,
            "exact_code_refit": original[ioc].predicted,
            "linked_refit": row.predicted,
            "linked_persistence": persistence[ioc].predicted if ioc in persistence else None,
        }
        for ioc, row in linked.items()
    ]
    return results, rows


def main():
    parser = argparse.ArgumentParser(description="Refit the host-interaction model for Tokyo 2020 and link the ROC record.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=GEN)
    args = parser.parse_args()
    output = args.out_dir.absolute()
    tag = selected_tag()
    forecasts, meta = fit_tokyo(tag, args.seed, args.threads)
    results, rows = evaluate(forecasts, meta, tag)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tokyo_roc_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    (output / "tokyo_roc_sensitivity.json").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
