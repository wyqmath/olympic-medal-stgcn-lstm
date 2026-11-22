import argparse
import json

import numpy as np
import pandas as pd
import torch

from evaluate_recent_editions import GEN
from evaluate_tokyo_roc import HISTORICAL_CODE, PANEL, REPORTED_CODE, YEAR
import baselines_compare as baselines
import stgcn_variants as variants


sp = variants.sp
OUTDIR = GEN / "tokyo_linked"
GROUPS = {
    "plain": ("base", "V3", "V4", "V5"),
    "huber": ("V2",),
    "host_huber": ("V6",),
}


def match_tokyo_codes(frame):
    return frame.assign(ioc=frame["ioc"].replace({REPORTED_CODE: HISTORICAL_CODE}))


def original_codes(frame):
    return frame.assign(ioc=frame["ioc"].replace({HISTORICAL_CODE: REPORTED_CODE}))


def fit_candidates(group, panel, nodes, editions, samples, target):
    tags = GROUPS[group]
    setup = variants.VARIANTS[tags[0]]
    used = samples
    if setup["aug"]:
        used = variants.with_static(
            samples, variants.augmented_statics(panel, nodes, samples, editions),
        )
    device = torch.device("cpu")
    earlier = [edition for edition in editions if edition < YEAR]
    sp.set_seed(sp.SEED)
    model, meta = variants.train_target_v(
        used, YEAR, earlier, device, used[YEAR]["static"].shape[1],
        variants.make_loss(setup["loss"]),
    )
    raw = sp.predict_target(model, used, YEAR, device)
    anchor = np.expm1(used[YEAR]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
    for tag in tags:
        config = variants.VARIANTS[tag]
        hgbr = (
            variants.hgbr_member(panel, nodes, samples, editions)
            if config.get("hgbr") else None
        )
        (a, b), _ = variants.select_mix(
            model, used, meta["val_editions"], device, hgbr=hgbr,
            grid=variants.THETA_GRIDS[config["theta"]], tol_pp=variants.THETA_TOL_PP[config["theta"]],
        )
        pred = a * raw + (1.0 - a - b) * anchor
        if hgbr is not None:
            pred = pred + b * hgbr[YEAR]
        if config["host"]:
            pred, _ = variants.apply_host_multiplier(pred, panel, YEAR, nodes)
        out = target.assign(predicted=[max(float(pred[nodes[ioc]]), 0.0) for ioc in target["ioc"]])
        yield tag, original_codes(out), dict(meta, blend_a=a, blend_b=b)


def fit_benchmarks(group, panel, nodes, editions, samples, target):
    device = torch.device("cpu")
    if group == "pure_lstm":
        frame, meta = baselines.pure_lstm_predictions(panel, samples, editions, [YEAR], nodes, target, device)
        yield group, original_codes(frame), meta[str(YEAR)]
    elif group == "tt_ensemble":
        frame, meta = baselines.tt_ensemble_predictions(panel, samples, editions, [YEAR], nodes, target, device)
        yield group, original_codes(frame), meta[str(YEAR)]
    elif group == "xgboost":
        xgb = baselines.xgboost_predictions(samples, editions, [YEAR], nodes, target)
        yield group, original_codes(xgb), {}
    else:
        fit_panel = panel.copy()
        tokyo = fit_panel["year"].eq(YEAR)
        fit_panel.loc[tokyo, "ioc"] = match_tokyo_codes(fit_panel.loc[tokyo])["ioc"]
        persistence = baselines.persistence_predictions(fit_panel, target)
        yield "persistence", original_codes(persistence), {}
        lookup = {(r.ioc, YEAR): r.predicted for r in persistence.itertuples()}
        arima, detail = baselines.arima_predictions(panel, target, lookup)
        yield "arima", original_codes(arima), detail


def fit_standalone_hgbr(panel, nodes, editions, samples, targets):
    from sklearn.ensemble import HistGradientBoostingRegressor

    design = variants.hgbr_design(samples, editions)
    for year in targets:
        prior = [edition for edition in editions if edition < year]
        x = np.concatenate([design[e][samples[e]["mask"]] for e in prior])
        y = np.concatenate([samples[e]["y"][samples[e]["mask"]] for e in prior])
        reg = HistGradientBoostingRegressor(**variants.HGBR_PARAMS).fit(x, y)
        forecasts = np.expm1(np.maximum(0.0, reg.predict(design[year])))
        target = panel.loc[panel["year"].eq(year), ["country", "ioc", "year", "total"]].rename(columns={"total": "actual"})
        lookup = match_tokyo_codes(target) if year == YEAR else target
        yield target.assign(predicted=[float(forecasts[nodes[ioc]]) for ioc in lookup["ioc"]])


def main():
    parser = argparse.ArgumentParser(description="Fit Tokyo forecasts and the standalone histogram gradient boosting benchmark.")
    parser.add_argument("group", choices=[*GROUPS, "pure_lstm", "tt_ensemble", "tabular", "xgboost", "hgbr"])
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    panel, info = variants.load_panel(str(PANEL))
    nodes = {ioc: idx for idx, ioc in enumerate(sorted(panel["ioc"].unique()))}
    editions = info["years"] if args.group == "hgbr" else [e for e in info["years"] if e <= YEAR]
    history = panel.loc[panel["year"].isin(editions)]
    raw_target = panel.loc[panel["year"].eq(YEAR), ["country", "ioc", "year", "total"]].rename(columns={"total": "actual"})
    target = match_tokyo_codes(raw_target)
    if len(raw_target) != 93 or raw_target["ioc"].eq(REPORTED_CODE).sum() != 1 or not target["ioc"].is_unique:
        raise ValueError("Unexpected Tokyo delegation records")
    samples = sp.build_samples(history, nodes, editions)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.group == "hgbr":
        pd.concat(fit_standalone_hgbr(panel, nodes, editions, samples, info["targets"])).to_csv(
            OUTDIR / "hgbr.csv", index=False
        )
        return
    fit = fit_candidates if args.group in GROUPS else fit_benchmarks
    for tag, frame, meta in fit(args.group, history, nodes, editions, samples, target):
        if frame.duplicated(["year", "ioc"]).any():
            raise ValueError(f"Duplicate predictions for {tag}")
        if not np.isfinite(frame["predicted"]).all() or (frame["predicted"] < 0).any():
            raise ValueError(f"Invalid predicted counts for {tag}")
        frame.to_csv(OUTDIR / f"{tag}.csv", index=False)
        (OUTDIR / f"{tag}.json").write_text(
            json.dumps(dict(meta, threads=args.threads, torch=torch.__version__), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{tag}: {len(frame)} Tokyo rows, ROC={frame.loc[frame['ioc'].eq('ROC'), 'predicted'].iloc[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
