from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_ORIG_CWD = os.getcwd()

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import stgcn_predict as sp
import stgcn_variants as sv
import nested_backtest as nb

GEN = os.path.join(HERE, "gen")
DATA_DIR = os.path.join(HERE, os.pardir, "data")
DEFAULT_PANEL = os.path.join(DATA_DIR, "panel_clean.csv")
DEFAULT_VARIANTS_DIR = os.path.join(GEN, "nested", "variants")
DEFAULT_OUTDIR = os.path.join(GEN, "forward2028")

FORWARD_YEAR = 2028
FORWARD_HOST_IOC = "USA"
CARRY_GDFILL = "carry2y"
B_REPS = 64
RNG_SEED = 42
CI_PERCENTILES = (2.5, 97.5)
METALS = ("gold", "silver", "bronze")


def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ORIG_CWD, p)


def _quiet(*a, **k) -> None:
    pass


def build_forward_panel(panel: pd.DataFrame, log=print) -> tuple:
    assert FORWARD_YEAR > int(panel["year"].max()), "panel already contains 2028"
    hist = panel[panel["ioc"] == FORWARD_HOST_IOC].sort_values("year")
    assert len(hist), f"no panel rows for the host {FORWARD_HOST_IOC}"
    last = hist.iloc[-1]
    row = {
        "country": str(last["country"]),
        "ioc": FORWARD_HOST_IOC,
        "year": FORWARD_YEAR,
        "gold": 0.0, "silver": 0.0, "bronze": 0.0, "total": 0.0,
        "host_flag": 1,
        "gdp_total_2011usd_million": float(last["gdp_total_2011usd_million"]),
        "gdp_per_capita_2011usd": float(last["gdp_per_capita_2011usd"]),
        "population": float(last["population"]),
        "gdp_gapfill": CARRY_GDFILL,
    }
    assert int(last["year"]) < FORWARD_YEAR
    assert row["gdp_gapfill"] in sp.GAPFILL_VALUES, "gapfill label not in the panel contract"
    out = pd.concat([panel, pd.DataFrame([row])], ignore_index=True)
    assert len(out) == len(panel) + 1
    assert int(out["year"].max()) == FORWARD_YEAR
    assert (out["year"] < FORWARD_YEAR).sum() == len(panel)
    rec = dict(row=row, carried_from_year=int(last["year"]),
               note=("medal columns are placeholders (unknown in 2028); the 2028 sample is "
                     "predict-only, so the placeholder cannot enter any feature, graph, "
                     "training loss or validation score"))
    log(f"[panel] appended 1 row: {row['ioc']} {FORWARD_YEAR} host=1 covariates carried from "
        f"{int(last['year'])} ({len(out)} rows)")
    return out, rec


def select_variant_2028(variants_dir: str, log=print) -> dict:
    frames = nb.load_variant_csvs(variants_dir, log=log)
    nb.assert_cross_variant(frames)
    raw = nb.inner_scores(frames)
    sel = nb.select_target(FORWARD_YEAR, raw, nb.DEFAULT_TOL_PP)
    assert not sel["fallback"], "no eligible candidate before 2028 -- protocol broken"
    log(f"[select] v*({FORWARD_YEAR}) = {sel['winner']} (argmin {sel['argmin']}, "
        f"pool {sel['pool']}, tie_broken {sel['tie_broken']})")
    for tag in nb.VARIANT_TAGS:
        log(f"[select]   {tag:<4} S(<{FORWARD_YEAR}) = {sel['scores'][tag]:.6f}")
    return sel


def forecast_winner(tag: str, panel: pd.DataFrame, samples: dict, node_index: dict,
                    editions: list, statics: dict, hgbr: dict, device,
                    max_epochs=None, log=print) -> tuple:
    cfg = sv.VARIANTS[tag]
    loss_fn = sv.make_loss(cfg["loss"])
    static_dim = sp.N_STATIC_FEAT + (sv.N_AUG if cfg["aug"] else 0)
    if cfg["aug"]:
        assert statics is not None, f"{tag} needs the augmented static block"
        used = sv.with_static(samples, statics)
    else:
        assert statics is None, f"{tag} must keep the baseline static block"
        used = samples
    grid = sv.THETA_GRIDS[cfg["theta"]]
    tol = sv.THETA_TOL_PP[cfg["theta"]]
    hgbr_use = hgbr if cfg.get("hgbr") else None
    assert (hgbr_use is not None) == bool(cfg.get("hgbr"))

    editions_before = [e for e in editions if e < FORWARD_YEAR]
    assert max(editions_before) < FORWARD_YEAR, "leakage: training window reaches 2028"
    model, meta = sv.train_target_v(used, FORWARD_YEAR, editions_before, device,
                                    static_dim, loss_fn, max_epochs=max_epochs, log=log)
    raw = sp.predict_target(model, used, FORWARD_YEAR, device)
    anchor = np.expm1(used[FORWARD_YEAR]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
    if hgbr_use is not None:
        (a, b), _scores = sv.select_mix(model, used, meta["val_editions"], device,
                                        hgbr=hgbr_use, grid=grid, tol_pp=tol, log=log)
        pred = a * raw + b * hgbr_use[FORWARD_YEAR] + (1.0 - a - b) * anchor
        meta["blend_a"], meta["blend_b"] = a, b
    else:
        (th, _b), _scores = sv.select_mix(model, used, meta["val_editions"], device,
                                          grid=grid, tol_pp=tol, log=log)
        pred = float(th) * raw + (1.0 - float(th)) * anchor
        meta["blend_a"], meta["blend_b"] = float(th), 0.0
    if cfg["host"]:
        pred, r = sv.apply_host_multiplier(pred, panel, FORWARD_YEAR, node_index, log=log)
        meta["host_multiplier"] = r
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    assert np.isfinite(pred).all() and len(pred) == len(node_index)
    del model
    return pred, meta


def masked_samples(samples: dict, statics: dict, indicator: np.ndarray) -> dict:
    out = {}
    for y, s in samples.items():
        out[y] = {**s, "mask": np.logical_and(s["mask"], indicator)}
    return sv.with_static(out, statics) if statics is not None else out


def bootstrap_reps(tag: str, panel: pd.DataFrame, samples: dict, node_index: dict,
                   editions: list, statics: dict, device, reps: int,
                   max_epochs=None, log=print) -> tuple:
    n = len(node_index)
    rng = np.random.default_rng(RNG_SEED)
    draws = np.empty((reps, n), dtype=float)
    seconds, n_in_loss = [], []
    cfg = sv.VARIANTS[tag]
    use_statics = statics if cfg["aug"] else None
    for b in range(reps):
        t0 = time.time()
        pick = rng.integers(0, n, n)
        indicator = np.bincount(pick, minlength=n) > 0
        sp.set_seed(sp.SEED)
        used = masked_samples(samples, use_statics, indicator)
        hgbr_rep = None
        if cfg.get("hgbr"):
            hgbr_rep = sv.hgbr_member(panel, node_index, used, editions, log=_quiet)
        draws[b], _meta = forecast_winner(tag, panel, used, node_index, editions,
                                        use_statics, hgbr_rep, device,
                                        max_epochs=max_epochs, log=_quiet)
        seconds.append(time.time() - t0)
        n_in_loss.append(int(indicator.sum()))
        if (b + 1) % 10 == 0 or b + 1 == reps:
            log(f"[boot] rep {b + 1}/{reps}: {n_in_loss[-1]}/{n} countries in the loss, "
                f"{seconds[-1]:.2f}s")
    info = dict(reps=reps, seed=RNG_SEED,
                rng="np.random.default_rng(42), one stream, "
                    "pick = rng.integers(0, n_nodes, n_nodes) per replicate",
                mask_rule="rep_mask = sample['mask'] AND sampled-country indicator; "
                          "duplicates enter the loss once, absent countries are predicted "
                          "but contribute no loss and no blend score",
                seconds_total=round(float(sum(seconds)), 2),
                seconds_mean=round(float(np.mean(seconds)), 2),
                countries_in_loss_mean=round(float(np.mean(n_in_loss)), 2),
                countries_in_loss_min=int(min(n_in_loss)),
                countries_in_loss_max=int(max(n_in_loss)))
    return draws, info


def medal_shares(panel: pd.DataFrame, target_year: int, node_index: dict,
                 log=print) -> tuple:
    hist = panel[(panel["year"] < target_year) & (panel["total"] > 0)]
    own = hist.groupby("ioc").apply(
        lambda g: pd.Series({
            "gold": float((g["gold"] / g["total"]).mean()),
            "silver": float((g["silver"] / g["total"]).mean()),
            "bronze": float((g["bronze"] / g["total"]).mean()),
        }), include_groups=False)
    pooled = [float((hist[m] / hist["total"]).mean()) for m in METALS]
    n_own = 0
    shares = np.zeros((len(node_index), 3), dtype=float)
    had = {}
    for ioc, j in node_index.items():
        if ioc in own.index:
            vec = own.loc[ioc, list(METALS)].to_numpy(dtype=float)
            had[ioc] = True
            n_own += 1
        else:
            vec = np.array(pooled, dtype=float)
            had[ioc] = False
        vec = np.clip(vec, 0.0, None)
        shares[j] = vec / vec.sum() if vec.sum() > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
    log(f"[shares] {n_own}/{len(node_index)} countries decomposed on their own history, "
        f"{len(node_index) - n_own} on the all-country mean shares "
        f"(g/s/b = {pooled[0]:.4f}/{pooled[1]:.4f}/{pooled[2]:.4f})")
    return shares, had


def movers_table(panel: pd.DataFrame, node_index: dict, point: np.ndarray,
                 active_iocs: set, log=print) -> pd.DataFrame:
    base_year = max(int(y) for y in panel["year"] if y < FORWARD_YEAR)
    last = panel[panel["year"] == base_year].set_index("ioc")
    iocs = [c for c in node_index if c in last.index and c in active_iocs]
    jdx = [node_index[c] for c in iocs]
    log(f"[movers] base year {base_year}: {len(iocs)} of {len(node_index)} forecast "
        f"countries have a row to diff against")
    shares, _had = medal_shares(panel, FORWARD_YEAR, node_index, log=_quiet)
    parts = []
    for m in ("total",) + METALS:
        pred = point[jdx] if m == "total" else point[jdx] * shares[jdx, METALS.index(m)]
        actual = last.loc[iocs, m].to_numpy(dtype=float)
        parts.append(pd.DataFrame(dict(ioc=iocs, country=last.loc[iocs, "country"].to_numpy(),
                                       metal=m, actual_2024=actual, pred_2028=pred,
                                       delta=pred - actual)))
    allm = pd.concat(parts, ignore_index=True)
    blocks = []
    for _m, g in allm.groupby("metal"):
        blocks.append(g.sort_values("delta", ascending=False).head(10))
        blocks.append(g.sort_values("delta", ascending=True).head(10))
    cols = ["ioc", "country", "metal", "actual_2024", "pred_2028", "delta"]
    return pd.concat(blocks, ignore_index=True)[cols]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--variants-dir", default=DEFAULT_VARIANTS_DIR,
                    help="directory with the seven nested predictions_{base,V1..V6}.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reps", type=int, default=B_REPS,
                    help=f"country-cluster bootstrap replicates (frozen: {B_REPS})")
    ap.add_argument("--max-epochs", type=int, default=0, help="smoke run: cap the epochs")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    log = print
    assert args.reps > 0, "--reps must be > 0"
    device = torch.device(args.device)
    max_epochs = args.max_epochs or None
    log(f"[run] v*({FORWARD_YEAR}) pipeline, reps={args.reps} device={device} "
        f"max_epochs={args.max_epochs or sp.MAX_EPOCHS} F7 deterministic flags on")

    panel_raw = pd.read_csv(_abs_path(args.panel))
    panel = sp.normalise_panel(panel_raw, log=log)
    info = sp.assert_panel_contract(panel)
    panel_f, row_rec = build_forward_panel(panel, log=log)
    node_index = {c: i for i, c in enumerate(sorted(panel_f["ioc"].unique()))}
    assert len(node_index) == info["n_countries"], "the appended host row must not add a node"
    editions = sorted(int(y) for y in panel_f["year"].unique())
    country_of = (panel_f[panel_f["year"] < FORWARD_YEAR].sort_values("year")
                  .groupby("ioc")["country"].last().to_dict())
    base_year = max(int(y) for y in panel_f["year"] if y < FORWARD_YEAR)
    active_iocs = set(panel_f.loc[(panel_f["year"] == base_year)
                                 & (panel_f["total"] > 0), "ioc"])
    log(f"[protocol] reporting rule: {FORWARD_YEAR} tables default to the {len(active_iocs)} "
        f"NOCs with medals in {base_year}; defunct/discontinued NOCs stay in "
        f"predictions_point.csv (active_2024 column) for completeness")

    sel = select_variant_2028(_abs_path(args.variants_dir), log=log)
    tag = sel["winner"]

    t0 = time.time()
    samples = sp.build_samples(panel_f, node_index, editions, log=log)
    assert max(samples[FORWARD_YEAR]["hist_years"]) < FORWARD_YEAR, "leakage: 2028 history"
    assert all((not samples[e]["hist_years"]) or max(samples[e]["hist_years"]) < e
               for e in editions), "leakage in a sample"
    statics = sv.augmented_statics(panel_f, node_index, samples, editions, log=log)
    log(f"[samples] {len(samples)} edition-samples in {time.time() - t0:.1f}s")
    hgbr = None
    if sv.VARIANTS[tag].get("hgbr"):
        t0 = time.time()
        hgbr = sv.hgbr_member(panel_f, node_index, samples, editions, log=log)
        log(f"[hgbr] member built in {time.time() - t0:.1f}s (V4 needs it for the blend)")
    editions_before = [e for e in editions if e < FORWARD_YEAR]
    assert max(editions_before) < FORWARD_YEAR
    train_ed, val_ed = sp.split_train_val(editions_before)
    assert FORWARD_YEAR not in train_ed + val_ed
    log(f"[protocol] target {FORWARD_YEAR}: train {train_ed[0]}..{train_ed[-1]} "
        f"({len(train_ed)}), internal validation {val_ed}")

    sp.set_seed(sp.SEED)
    t0 = time.time()
    point, meta = forecast_winner(tag, panel_f, samples, node_index, editions,
                                statics if sv.VARIANTS[tag]["aug"] else None, hgbr,
                                device, max_epochs=max_epochs, log=log)
    point_seconds = time.time() - t0
    log(f"[point] {tag} at {FORWARD_YEAR}: top5 "
        f"{[country_of[i] for i in sorted(node_index, key=lambda c: -point[node_index[c]])[:5]]} "
        f"| epoch {meta['best_epoch']}/{meta['epochs_run']} "
        f"blend a={meta['blend_a']} host_r={meta.get('host_multiplier')} "
        f"{point_seconds:.2f}s")

    t0 = time.time()
    draws, boot_info = bootstrap_reps(tag, panel_f, samples, node_index, editions,
                                      statics, device, args.reps,
                                      max_epochs=max_epochs, log=log)
    boot_info["seconds_total"] = round(time.time() - t0, 2)
    boot_info["n_nodes"] = len(node_index)
    lo_q, hi_q = CI_PERCENTILES
    ci_low = np.percentile(draws, lo_q, axis=0)
    ci_high = np.percentile(draws, hi_q, axis=0)
    ci_mean = draws.mean(axis=0)
    ci_median = np.median(draws, axis=0)
    log(f"[boot] {args.reps} replicates in {boot_info['seconds_total']:.1f}s "
        f"(mean {boot_info['seconds_mean']:.2f}s/rep)")

    shares, had_history = medal_shares(panel_f, FORWARD_YEAR, node_index, log=log)
    iocs = [c for c, _ in sorted(node_index.items(), key=lambda kv: kv[1])]
    active_col = np.array([c in active_iocs for c in iocs])
    point_df = pd.DataFrame(dict(
        ioc=iocs,
        country=[country_of[i] for i in iocs],
        active_2024=active_col,
        total_pred=point,
        gold_pred=point * shares[:, 0],
        silver_pred=point * shares[:, 1],
        bronze_pred=point * shares[:, 2],
    )).sort_values("total_pred", ascending=False).reset_index(drop=True)
    ci_all = pd.DataFrame(dict(
        ioc=iocs,
        country=[country_of[i] for i in iocs],
        active_2024=active_col,
        total_pred=point,
        total_ci_low=ci_low, total_ci_high=ci_high,
        total_mean=ci_mean, total_median=ci_median,
        gold_ci_low=ci_low * shares[:, 0], gold_ci_high=ci_high * shares[:, 0],
        silver_ci_low=ci_low * shares[:, 1], silver_ci_high=ci_high * shares[:, 1],
        bronze_ci_low=ci_low * shares[:, 2], bronze_ci_high=ci_high * shares[:, 2],
    )).sort_values("total_pred", ascending=False).reset_index(drop=True)
    ci_df = ci_all[ci_all["active_2024"]].reset_index(drop=True)
    movers = movers_table(panel_f, node_index, point, log=log, active_iocs=active_iocs)
    aggregate_check = dict(
        base_year=int(base_year),
        n_active_nocs=int(active_col.sum()),
        n_all_nocs=int(len(iocs)),
        sum_pred_active_2028=round(float(point[active_col].sum()), 2),
        sum_pred_all_2028=round(float(point.sum()), 2),
        actual_total_base_year=int(panel_f.loc[panel_f["year"] == base_year, "total"].sum()),
        note=("2028 forecasts for defunct/discontinued NOCs keep their historical "
              "persistence level, so the all-NOC total is not comparable with any "
              "single Games' actual total; the paper tables quote the active-NOC sum"),
    )
    log(f"[aggregate] active {aggregate_check['n_active_nocs']} NOCs: "
        f"{aggregate_check['sum_pred_active_2028']:.1f} predicted vs "
        f"{aggregate_check['actual_total_base_year']} actual {base_year}; "
        f"all {aggregate_check['n_all_nocs']} NOCs: {aggregate_check['sum_pred_all_2028']:.1f}")

    outdir = _abs_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    p_point = os.path.join(outdir, "predictions_point.csv")
    p_ci = os.path.join(outdir, "ci_table.csv")
    p_mov = os.path.join(outdir, "movers.csv")
    p_sel = os.path.join(outdir, "selection_2028.json")
    point_df.round(4).to_csv(p_point, index=False)
    ci_df.round(4).to_csv(p_ci, index=False)
    movers.round(4).to_csv(p_mov, index=False)
    log(f"[write] {p_point} ({len(point_df)} rows, all NOCs, active_2024 flag)")
    log(f"[write] {p_ci} ({len(ci_df)} rows, active NOCs only)")
    log(f"[write] {p_mov} ({len(movers)} rows, active NOCs only)")

    summary = dict(
        protocol=dict(
            document="nested selection rules F2/F3 (see nested_backtest.py); "
                     "forward procedure frozen in this file's docstring",
            forward_year=FORWARD_YEAR,
            selection=dict(tol_pp=nb.DEFAULT_TOL_PP,
                           parsimony_order=list(nb.PARSIMONY_ORDER),
                           candidates=list(nb.VARIANT_TAGS),
                           n_inner_editions=sel["n_inner_editions"],
                           scores={k: round(v, 6) for k, v in sel["scores"].items()},
                           n_inner_defined=sel["n_inner_defined"],
                           eligible=sel["eligible"], argmin=sel["argmin"],
                           winner=sel["winner"], pool=sel["pool"],
                           pool_size=sel["pool_size"], tie_broken=bool(sel["tie_broken"]),
                           margin_pp=sel["margin_pp"],
                           used_as_selected=True),
            point_forecast=dict(variant=tag, device=str(device),
                                max_epochs=int(args.max_epochs or sp.MAX_EPOCHS),
                                train_editions=[int(e) for e in train_ed],
                                val_editions=[int(e) for e in val_ed],
                                blend_a=meta["blend_a"], blend_b=meta["blend_b"],
                                host_multiplier=meta.get("host_multiplier"),
                                epochs_run=meta["epochs_run"], best_epoch=meta["best_epoch"],
                                seconds=round(point_seconds, 2)),
            forward_row=row_rec,
            reporting_rule=("2028 reporting restricted to NOCs with medals in "
                            f"{base_year}; defunct/discontinued NOCs retained in full "
                            "predictions_point.csv (active_2024 column) for completeness "
                            "but excluded from paper tables"),
            uncertainty=dict(method="country-cluster bootstrap of the winner pipeline",
                             reps=int(args.reps), seed=RNG_SEED, rng=boot_info["rng"],
                             mask_rule=boot_info["mask_rule"],
                             percentiles=list(CI_PERCENTILES),
                             note="B=64 resolves a 95% interval coarsely; reported as is",
                             semantics="intervals reflect training-resampling (cluster "
                                       "bootstrap) uncertainty, not calibrated prediction "
                                       "intervals",
                             seconds_total=boot_info["seconds_total"],
                             seconds_mean=boot_info["seconds_mean"]),
            aggregate_check=aggregate_check,
            decomposition=dict(method="share decomposition: metal forecast = total forecast "
                                     "x country mean historical share over its own <2028 "
                                     "medal-winning editions; all-country mean shares for "
                                     "countries without history; metal intervals use the "
                                     "same shares on the total interval bounds",
                               n_own_history=int(sum(had_history.values())),
                               n_pooled=int(len(had_history) - sum(had_history.values()))),
            determinism="CUBLAS_WORKSPACE_CONFIG=:4096:8, use_deterministic_algorithms(True), "
                        "cudnn.benchmark=False, sp.set_seed(42) before the point forecast "
                        "and before every replicate",
            panel=dict(source=_abs_path(args.panel), rows=int(len(panel)),
                       rows_with_forward_row=int(len(panel_f)),
                       n_nodes=len(node_index), editions=[int(e) for e in editions]),
        ),
        outputs=dict(predictions_point=p_point, ci_table=p_ci, movers=p_mov, summary=p_sel),
    )
    with open(p_sel, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[write] {p_sel}")
    log(f"[done] v*({FORWARD_YEAR})={tag} USA={float(point[node_index['USA']]):.1f} "
        f"[{float(ci_low[node_index['USA']]):.1f}, {float(ci_high[node_index['USA']]):.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
