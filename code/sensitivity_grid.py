import argparse
import copy
import json
import os
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import stgcn_predict as sp
from backtest_period import (DEFAULT_PERIODS, build_windows, macro_over_games,
                             period_games, smape_obs)

SEED = sp.SEED
LAMBDA0 = sp.LAMBDA_DECAY
RHO0 = sp.WEIGHT_DECAY
DEFAULT_LAMBDA_GRID = (0.0125, 0.025, 0.05, 0.1, 0.2)
DEFAULT_RHO_GRID = (1e-05, 1e-04, 1e-03, 1e-02, 1e-01)
BASE_PREDICTIONS_PATH = os.path.join(sp.GEN_DIR, "variants", "predictions_base.csv")
NESTED_SMAPE_PATH = os.path.join(sp.GEN_DIR, "nested", "backtest", "smape_by_games.csv")
CONSISTENCY_TOL_PP = 0.5
PERIOD_COL = {"Cold War": "coldwar", "Globalization": "globalization",
              "COVID-19": "recent"}
DIAG_PAIRS = ((1, 3), (3, 0), (4, 4))


def step_features_lam(ed: pd.DataFrame, state: dict, pr: np.ndarray,
                      node_index: dict, lam: float) -> np.ndarray:
    n = len(node_index)
    f = np.zeros((n, sp.N_STEP_FEAT), dtype=np.float32)
    if ed is None or len(ed) == 0:
        return f
    ed_year = int(ed["year"].iloc[0])
    ed_total = float(ed["total"].sum())
    raw = np.stack([
        np.log1p(ed["gdp_total_2011usd_million"].to_numpy(dtype=float)),
        np.log1p(ed["gdp_per_capita_2011usd"].to_numpy(dtype=float)),
        np.log1p(ed["population"].to_numpy(dtype=float)),
    ], axis=1)
    ez = sp._zcols(np.where(np.isfinite(raw), raw, np.nan))
    pr_z = sp._zcols(pr.reshape(-1, 1)).ravel()

    lt = np.log1p(ed["total"].to_numpy(dtype=float))
    w_all = np.maximum(sp.dynamic_weight(
        lt, ed["gdp_total_2011usd_million"].to_numpy(dtype=float),
        ed["population"].to_numpy(dtype=float),
        np.full(len(ed), ed_year), ed_year, lam=lam), 0.0)
    w_sum = max(float(w_all.sum()), 1e-12)
    cum, lastp, ma3 = state["cum"], state["last_part"], state["ma3"]

    for pos, (ioc, r) in enumerate(ed.iterrows()):
        j = node_index.get(ioc)
        if j is None:
            continue
        tot = float(r["total"])
        gdp_m = pd.isna(r["gdp_total_2011usd_million"])
        gpc_m = pd.isna(r["gdp_per_capita_2011usd"])
        pop_m = pd.isna(r["population"])
        f[j] = (
            1.0,
            float(np.log1p(tot)),
            float(np.log1p(r["gold"])),
            float(np.log1p(r["silver"])),
            float(np.log1p(r["bronze"])),
            tot / ed_total if ed_total > 0 else 0.0,
            (float(r["gold"]) / tot) if tot > 0 else 0.0,
            0.0 if gdp_m else float(ez[pos, 0]),
            1.0 if gdp_m else 0.0,
            0.0 if gpc_m else float(ez[pos, 1]),
            1.0 if gpc_m else 0.0,
            0.0 if pop_m else float(ez[pos, 2]),
            1.0 if pop_m else 0.0,
            float(r["host_flag"]),
            float(np.log1p(w_all[pos])),
            float(w_all[pos] / w_sum),
            min(ed_year - lastp[ioc], 16) / 16.0 if ioc in lastp else 1.0,
            float(ma3.get(ioc, 0.0)),
            float(pr_z[j]),
            float(np.log1p(cum.get(ioc, 0.0))),
        )
    return f


def static_features_lam(target_year: int, panel: pd.DataFrame, node_index: dict,
                        state: dict, last_records, lam: float) -> np.ndarray:
    n = len(node_index)
    s = np.zeros((n, sp.N_STATIC_FEAT), dtype=np.float32)
    if last_records is None or len(last_records) == 0:
        return s
    lr = last_records
    raw = np.stack([
        np.log1p(lr["gdp_total_2011usd_million"].to_numpy(dtype=float)),
        np.log1p(lr["gdp_per_capita_2011usd"].to_numpy(dtype=float)),
        np.log1p(lr["population"].to_numpy(dtype=float)),
    ], axis=1)
    ez = sp._zcols(np.where(np.isfinite(raw), raw, np.nan))
    years = lr["year"].to_numpy(dtype=float)
    w_all = sp.dynamic_weight(np.log1p(lr["total"].to_numpy(dtype=float)),
                              lr["gdp_total_2011usd_million"].to_numpy(dtype=float),
                              lr["population"].to_numpy(dtype=float),
                              years, int(target_year), lam=lam)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    w_sum = max(float(np.nansum(np.maximum(w_all, 0.0))), 1e-12)
    hosts = set(panel.loc[(panel["year"] == target_year) & (panel["host_flag"] == 1), "ioc"])
    cum, lastp, ma3, med = state["cum"], state["last_part"], state["ma3"], state["median"]
    totals = lr["total"].to_numpy(dtype=float)

    for pos, (ioc, _r) in enumerate(lr.iterrows()):
        j = node_index.get(ioc)
        if j is None:
            continue
        gdp_m = pd.isna(lr["gdp_total_2011usd_million"].iloc[pos])
        gpc_m = pd.isna(lr["gdp_per_capita_2011usd"].iloc[pos])
        pop_m = pd.isna(lr["population"].iloc[pos])
        wv = float(w_all[pos])
        last_year = int(years[pos])
        s[j] = (
            1.0 if ioc in hosts else 0.0,
            0.0 if gdp_m else float(ez[pos, 0]),
            1.0 if gdp_m else 0.0,
            0.0 if gpc_m else float(ez[pos, 1]),
            1.0 if gpc_m else 0.0,
            0.0 if pop_m else float(ez[pos, 2]),
            1.0 if pop_m else 0.0,
            float(np.log1p(max(wv, 0.0))),
            max(wv, 0.0) / w_sum,
            min(target_year - lastp[ioc], 16) / 16.0 if ioc in lastp else 1.0,
            float(np.log1p(cum.get(ioc, 0.0))),
            float(ma3.get(ioc, 0.0)),
            float(np.log1p(totals[pos])),
            float(np.log1p(med.get(ioc, float("nan")))) if ioc in med else 0.0,
        )
    return s


def build_samples_lam(panel: pd.DataFrame, node_index: dict, editions: list,
                      lam: float, log=print) -> dict:
    hp = panel.sort_values(["year", "ioc"])
    years = sorted(int(y) for y in hp["year"].unique())
    state = {"cum": {}, "last_part": {}, "ma3": {}, "ma_hist": {}, "median": {}}
    graph_cache, pr_cache, step_cache, static_cache, years_before_cache = {}, {}, {}, {}, {}

    for year in years:
        hist = hp[hp["year"] < year]
        years_before_cache[year] = sorted(int(y) for y in hist["year"].unique())
        ei, ew, share = sp.build_graph(hist, year, node_index)
        graph_cache[year] = (ei, ew)
        pr_cache[year] = sp.walkforward_pagerank(share, node_index)
        ed = hp[hp["year"] == year].sort_values("ioc").set_index("ioc")
        ed = ed.assign(year=year)
        step_cache[year] = step_features_lam(ed, state, pr_cache[year], node_index, lam)
        last_records = None
        if years_before_cache[year]:
            last_records = (hist.sort_values("year").groupby("ioc").tail(1)
                            .set_index("ioc"))
        static_cache[year] = static_features_lam(year, panel, node_index, state,
                                                 last_records, lam)
        sp.advance_state(ed, year, state)

    samples = {}
    for year in editions:
        ei, ew = graph_cache[year]
        years_before = years_before_cache[year]
        hyears = years_before[-sp.SEQ_LEN:]
        steps = np.zeros((sp.SEQ_LEN, len(node_index), sp.N_STEP_FEAT), dtype=np.float32)
        for k, y in enumerate(hyears):
            steps[sp.SEQ_LEN - len(hyears) + k] = step_cache[y]

        mask = np.zeros(len(node_index), dtype=bool)
        yv = np.zeros(len(node_index), dtype=np.float32)
        tgt = hp[hp["year"] == year]
        for ioc, tot in zip(tgt["ioc"], tgt["total"]):
            j = node_index.get(ioc)
            if j is not None:
                mask[j] = True
                yv[j] = float(np.log1p(tot))

        samples[year] = {
            "edge_index": ei,
            "edge_weight": ew,
            "steps": steps,
            "static": static_cache[year],
            "y": yv,
            "mask": mask,
            "hist_years": hyears,
            "n_edges": int(ei.shape[1]),
        }
    log(f"[samples] built {len(samples)} edition-samples at lambda={lam} "
        f"(avg edges/sample {np.mean([s['n_edges'] for s in samples.values()]):.0f})")
    return samples


def train_target_rho(samples, target_year, editions_before, device, weight_decay: float,
                     max_epochs=None, log=print):
    max_epochs = max_epochs or sp.MAX_EPOCHS
    train_ed, val_ed = sp.split_train_val(editions_before)

    model = sp.STGCN_LSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sp.LR, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sp.lr_lambda)

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = sp.run_epoch(model, samples, train_ed, opt, device)
        sched.step()
        if val_ed:
            va = sp.eval_epoch(model, samples, val_ed, device)
            if va < best_val - 1e-6:
                best_val, best_state, best_epoch, bad = va, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= sp.PATIENCE:
                    break
        elif tr < best_val - 1e-6:
            best_val, best_state, best_epoch, bad = tr, copy.deepcopy(model.state_dict()), epoch, 0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": epoch, "best_epoch": best_epoch,
                   "val_editions": list(val_ed), "best_val_log_mse": best_val,
                   "seconds": time.time() - t0, "n_train_samples": len(train_ed),
                   "n_val_samples": len(val_ed)}


def run_forecast_rho(panel, target_year, device, samples, node_index, editions,
                     weight_decay: float, max_epochs=None, log=print):
    editions_before = [e for e in editions if e < target_year]
    model, meta = train_target_rho(samples, target_year, editions_before, device,
                                   weight_decay, max_epochs=max_epochs, log=log)
    raw_pred = sp.predict_target(model, samples, target_year, device)
    theta, _scores = sp.select_blend_theta(model, samples, meta["val_editions"], device)
    anchor_full = np.expm1(samples[target_year]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
    pred = float(theta) * raw_pred + (1.0 - float(theta)) * anchor_full
    meta["blend_theta"] = theta
    del model
    return pred, meta


def assert_samples_identical(a: dict, b: dict, tag: str) -> None:
    assert set(a) == set(b), f"{tag}: edition sets differ {set(a) ^ set(b)}"
    for y in a:
        for k in ("edge_index", "edge_weight", "steps", "static", "y", "mask", "hist_years"):
            x, z = a[y][k], b[y][k]
            if isinstance(x, np.ndarray):
                assert np.array_equal(x, z), f"{tag} {y}: column {k} differs"
            else:
                assert x == z, f"{tag} {y}: field {k} differs ({x} != {z})"


def assert_no_future_leak(samples: dict, targets: list) -> None:
    for t in targets:
        hyears = samples[t]["hist_years"]
        assert len(hyears) == 0 or max(hyears) < t, (
            f"leak: target {t} sees history year {max(hyears)} >= {t}")


def parse_float_grid(text: str, name: str) -> list:
    grid = tuple(float(v) for v in str(text).split(",") if str(v).strip())
    assert len(grid) >= 1, f"--{name} must hold at least one value"
    assert all(v > 0 for v in grid), f"--{name} values must be positive"
    return sorted(set(grid))


def select_cells(lambda_grid: list, rho_grid: list, mode: str) -> list:
    if mode == "full":
        return [(l, r) for l in lambda_grid for r in rho_grid]
    l0 = _nearest(lambda_grid, LAMBDA0)
    r0 = _nearest(rho_grid, RHO0)
    cells = {(l0, r) for r in rho_grid} | {(l, r0) for l in lambda_grid}
    for li, ri in DIAG_PAIRS:
        if li < len(lambda_grid) and ri < len(rho_grid):
            cells.add((lambda_grid[li], rho_grid[ri]))
    return sorted(cells)


def _nearest(grid: list, value: float) -> float:
    return min(grid, key=lambda g: (abs(np.log(g / value)), g))


def window_weighted_by_games(rows: list) -> dict:
    d = pd.DataFrame(rows)
    eds = sorted(int(y) for y in d["year"].unique())
    win = build_windows(eds, d["year"].to_numpy())
    act = d["actual"].to_numpy(dtype=float)
    pred = d["predicted"].to_numpy(dtype=float)
    err = smape_obs(act, pred)
    out = {}
    for t in eds:
        m = win[t] & np.isfinite(err) & (act > 0)
        out[t] = float((act[m] * err[m]).sum() / act[m].sum()) if act[m].sum() > 0 else float("nan")
    return out


def evaluate_cells(rows: list, periods=DEFAULT_PERIODS) -> dict:
    assert set(periods) == set(PERIOD_COL), (
        f"backtest_period periods changed: {sorted(periods)} vs {sorted(PERIOD_COL)}")
    d = pd.DataFrame(rows)
    eds = sorted(int(y) for y in d["year"].unique())
    win = build_windows(eds, d["year"].to_numpy())
    act = d["actual"].to_numpy(dtype=float)
    pred = d["predicted"].to_numpy(dtype=float)
    err = smape_obs(act, pred)
    out = {"weighted_smape_all": sp.weighted_smape(act, pred)}
    by_game = window_weighted_by_games(rows)
    vals = [v for v in by_game.values() if np.isfinite(v)]
    out["weighted_smape_window"] = float(np.mean(vals)) if vals else float("nan")
    for name, (lo, hi) in periods.items():
        tl = period_games(eds, lo, hi)
        out[f"macro_smape_{PERIOD_COL[name]}"] = macro_over_games(err, [win[t] for t in tl])
    return out


def base_reference_by_games(log=print):
    if not os.path.exists(BASE_PREDICTIONS_PATH):
        log(f"[check] base reference not found ({BASE_PREDICTIONS_PATH}); "
            f"incumbent consistency check skipped")
        return None, None
    ref = pd.read_csv(BASE_PREDICTIONS_PATH)[["year", "actual", "predicted"]]
    nested = None
    if os.path.exists(NESTED_SMAPE_PATH):
        n = pd.read_csv(NESTED_SMAPE_PATH)
        nested = dict(zip(n["year"].astype(int), n["weighted_smape"].astype(float)))
    return window_weighted_by_games(ref.to_dict("records")), nested


def check_incumbent_consistency(cell_rows: list, log=print) -> dict:
    ref, nested = base_reference_by_games(log)
    if ref is None:
        return {"status": "skipped", "reason": "reference file missing"}
    got = window_weighted_by_games(cell_rows)
    years = sorted(set(ref) & set(got))
    assert years, "no shared Games between the incumbent cell and the reference"
    per_game = {int(y): round(got[y] - ref[y], 4) for y in years}
    d_mean = float(np.mean([got[y] for y in years]) - np.mean([ref[y] for y in years]))
    d_max = float(np.max(np.abs(list(per_game.values()))))
    out = {"status": "ok", "reference": os.path.relpath(BASE_PREDICTIONS_PATH, HERE),
           "tolerance_pp": CONSISTENCY_TOL_PP,
           "mean_cell": round(float(np.mean([got[y] for y in years])), 4),
           "mean_reference": round(float(np.mean([ref[y] for y in years])), 4),
           "mean_diff_pp": round(d_mean, 4), "max_abs_per_game_diff_pp": round(d_max, 4),
           "per_game_diff_pp": per_game}
    if nested is not None:
        common = [y for y in years if y in nested]
        out["nested_mix_mean_diff_pp"] = round(
            float(np.mean([got[y] for y in common]) - np.mean([nested[y] for y in common])), 4)
        out["nested_mix_note"] = ("information only: that file backs the nested "
                                  "selection mix, not the base protocol")
    level = "warn" if abs(d_mean) > CONSISTENCY_TOL_PP else "ok"
    out["verdict"] = level
    log(f"[check] incumbent cell vs base reference ({out['reference']}): mean window "
        f"weighted SMAPE {out['mean_cell']:.3f}% vs {out['mean_reference']:.3f}% "
        f"-> diff {d_mean:+.3f} pp (tolerance {CONSISTENCY_TOL_PP} pp), "
        f"max per-Games diff {d_max:.3f} pp -> {level}")
    if level == "warn":
        log(f"[check] WARNING: the incumbent cell deviates from the base run by more than "
            f"{CONSISTENCY_TOL_PP} pp on the pooled window reading; not a gate, but inspect "
            f"the per-Games diffs in grid_meta.json before quoting the grid")
    return out


def write_heatmap(df: pd.DataFrame, lambda_grid: list, rho_grid: list,
                  path: str, lambda0: float, rho0: float) -> None:
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
    try:
        _fm.findfont("Arial", fallback_to_default=False)
        _family = "Arial"
    except ValueError:
        _family = None
    from matplotlib.patches import Rectangle

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    if _family is not None:
        matplotlib.rcParams["font.family"] = _family
        matplotlib.rcParams["mathtext.fontset"] = "custom"
        matplotlib.rcParams["mathtext.rm"] = _family
        matplotlib.rcParams["mathtext.it"] = _family + ":italic"
        matplotlib.rcParams["mathtext.bf"] = _family + ":bold"

    def fmt(v: float) -> str:
        return f"{v:.0e}" if v < 1e-3 else f"{v:g}"

    grid = (df.pivot(index="param_rho", columns="param_lambda",
                     values="weighted_smape_all")
            .reindex(index=sorted(rho_grid, reverse=True), columns=sorted(lambda_grid)))
    data = grid.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#E8E8E8")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pooled weighted SMAPE (%)")

    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isfinite(data[i, j]):
                continue
            bright = (data[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if bright else "white")

    li = sorted(lambda_grid).index(_nearest(lambda_grid, lambda0))
    ri = sorted(rho_grid, reverse=True).index(_nearest(rho_grid, rho0))
    ax.add_patch(Rectangle((li - 0.5, ri - 0.5), 1.0, 1.0, fill=False,
                           ls="--", lw=1.6, ec="black"))
    ax.set_xticks(range(len(lambda_grid)))
    ax.set_xticklabels([fmt(v) for v in sorted(lambda_grid)])
    ax.set_yticks(range(len(rho_grid)))
    ax.set_yticklabels([r"$10^{%d}$" % round(__import__("math").log10(v))
                        for v in sorted(rho_grid, reverse=True)])
    ax.set_xlabel(r"time-decay rate $\lambda$")
    ax.set_ylabel(r"weight decay $\rho$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=sp.PANEL_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambda-grid", default=",".join(f"{v:g}" for v in DEFAULT_LAMBDA_GRID),
                    help="comma-separated time-decay rates (default: incumbent x {1/4,1/2,1,2,4})")
    ap.add_argument("--rho-grid", default=",".join(f"{v:g}" for v in DEFAULT_RHO_GRID),
                    help="comma-separated Adam weight-decay values")
    ap.add_argument("--max-targets", type=int, default=0, help="smoke: only the first K editions")
    ap.add_argument("--max-epochs", type=int, default=sp.MAX_EPOCHS)
    ap.add_argument("--outdir", default=os.path.join(sp.GEN_DIR, "sensitivity"))
    ap.add_argument("--mode", choices=("full", "reduced"), default="full",
                    help="full = 5x5 grid; reduced = two sweeps + 3 diagonal cells "
                         "(pre-registered fallback if the full grid exceeds the time budget)")
    args = ap.parse_args()

    log = print
    sp.set_seed(SEED)
    device = torch.device(args.device)

    assert os.path.exists(args.panel), f"panel not found: {args.panel}"
    panel = sp.normalise_panel(pd.read_csv(args.panel), log=log)
    info = sp.assert_panel_contract(panel)
    log(f"[panel] {args.panel}: {info['n_rows']} rows, {info['n_countries']} countries, "
        f"{info['n_editions']} editions")

    lambda_grid = parse_float_grid(args.lambda_grid, "lambda-grid")
    rho_grid = parse_float_grid(args.rho_grid, "rho-grid")
    cells = select_cells(lambda_grid, rho_grid, args.mode)
    targets = info["targets"][: args.max_targets] if args.max_targets else info["targets"]
    editions = info["years"]
    log(f"[grid] mode={args.mode}: {len(cells)} cells "
        f"(lambda {lambda_grid} x rho {rho_grid}), targets {targets[0]}..{targets[-1]} "
        f"({len(targets)} editions), device={device}")

    node_index = {c: i for i, c in enumerate(sorted(panel["ioc"].unique()))}
    t0 = time.time()
    base_samples = sp.build_samples(panel, node_index, editions, log=lambda *a, **k: None)
    check_samples = build_samples_lam(panel, node_index, editions, LAMBDA0,
                                      log=lambda *a, **k: None)
    assert_samples_identical(base_samples, check_samples,
                             f"build_samples_lam(lambda={LAMBDA0})")
    log(f"[check] build_samples_lam at the incumbent lambda reproduces "
        f"stgcn_predict.build_samples bit-for-bit ({time.time() - t0:.1f}s)")
    del base_samples, check_samples

    samples_by_lambda = {}
    rows_out, per_cell_seconds, pred_pool, cell_detail = [], [], [], {}
    t_all = time.time()
    for c_i, (lam, rho) in enumerate(cells, start=1):
        if lam not in samples_by_lambda:
            samples_by_lambda[lam] = build_samples_lam(panel, node_index, editions, lam, log=log)
        samples = samples_by_lambda[lam]
        assert_no_future_leak(samples, targets)
        t_cell = time.time()
        rows = []
        best_epochs = []
        per_target_meta = {}
        for t in targets:
            sp.set_seed(SEED)
            pred, meta = run_forecast_rho(panel, t, device, samples, node_index,
                                          editions, rho, max_epochs=args.max_epochs,
                                          log=lambda *a, **k: None)
            best_epochs.append(int(meta["best_epoch"]))
            per_target_meta[int(t)] = {
                "best_epoch": int(meta["best_epoch"]),
                "epochs_run": int(meta["epochs_run"]),
                "blend_theta": float(meta["blend_theta"]),
                "n_train_samples": int(meta["n_train_samples"]),
                "val_editions": [int(v) for v in meta["val_editions"]],
                "seconds": round(float(meta["seconds"]), 1),
            }
            tgt = panel[panel["year"] == t]
            for ioc, actual in zip(tgt["ioc"], tgt["total"]):
                j = node_index.get(ioc)
                if j is None:
                    continue
                rows.append({"param_lambda": lam, "param_rho": rho, "ioc": ioc,
                             "year": int(t), "actual": int(actual),
                             "predicted": float(max(pred[j], 0.0))})
        secs = time.time() - t_cell
        per_cell_seconds.append(secs)
        m = evaluate_cells(rows)
        rec = {"param_lambda": lam, "param_rho": rho, "n_configs": len(cells),
               "n_targets": len(targets), "weighted_smape_all": round(m["weighted_smape_all"], 4),
               "weighted_smape_window": round(m["weighted_smape_window"], 4),
               "macro_smape_coldwar": round(m["macro_smape_coldwar"], 4),
               "macro_smape_globalization": round(m["macro_smape_globalization"], 4),
               "macro_smape_recent": round(m["macro_smape_recent"], 4),
               "best_epoch_mean": round(float(np.mean(best_epochs)), 2),
               "seconds": round(secs, 1)}
        rows_out.append(rec)
        pred_pool.extend(rows)
        cell_detail[f"lambda={lam:g};rho={rho:g}"] = per_target_meta
        log(f"[cell {c_i}/{len(cells)}] lambda={lam:g} rho={rho:g}: "
            f"weighted SMAPE {rec['weighted_smape_all']:.3f}% "
            f"(window {rec['weighted_smape_window']:.3f}%), "
            f"macro {rec['macro_smape_coldwar']:.2f}/{rec['macro_smape_globalization']:.2f}/"
            f"{rec['macro_smape_recent']:.2f}, {secs:.1f}s")
        est_left = float(np.mean(per_cell_seconds)) * (len(cells) - c_i)
        log(f"[grid] elapsed {time.time() - t_all:.1f}s, projected remaining {est_left / 60:.1f} min")

    os.makedirs(args.outdir, exist_ok=True)
    out = pd.DataFrame(rows_out).sort_values(["param_lambda", "param_rho"])
    csv_path = os.path.join(args.outdir, "grid_results.csv")
    out.to_csv(csv_path, index=False)
    log(f"[done] {len(out)} cells -> {csv_path}")

    pred_path = os.path.join(args.outdir, "grid_predictions.csv")
    pd.DataFrame(pred_pool).sort_values(
        ["param_lambda", "param_rho", "year", "ioc"]).to_csv(pred_path, index=False)
    log(f"[done] {len(pred_pool)} prediction rows -> {pred_path}")

    pdf_path = os.path.join(args.outdir, "heatmap.pdf")
    write_heatmap(out, lambda_grid, rho_grid, pdf_path, LAMBDA0, RHO0)
    log(f"[done] heat map -> {pdf_path}")

    lam_c, rho_c = _nearest(lambda_grid, LAMBDA0), _nearest(rho_grid, RHO0)
    cell_rows = [r for r in pred_pool
                 if r["param_lambda"] == lam_c and r["param_rho"] == rho_c]
    incumbent_check = (check_incumbent_consistency(cell_rows, log=log) if cell_rows
                       else {"status": "skipped", "reason": "incumbent cell not in this run"})

    best = out.sort_values("weighted_smape_all").iloc[0]
    inc = out[(out["param_lambda"] == _nearest(lambda_grid, LAMBDA0))
              & (out["param_rho"] == _nearest(rho_grid, RHO0))]
    meta = {
        "script": "sensitivity_grid.py",
        "purpose": "sensitivity of the STGCN-LSTM forecasts to the "
                   "time-decay rate lambda and the regularisation strength rho",
        "mode": args.mode,
        "mode_definition": {
            "full": "|lambda| x |rho| joint grid (every cell of the two grids)",
            "reduced": "two single-parameter sweeps through the incumbent cell "
                       "plus 3 off-axis diagonal cells (pre-registered fallback)",
        },
        "seed": SEED,
        "device": str(device),
        "determinism": {"CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                        "torch_use_deterministic_algorithms": True,
                        "cudnn_benchmark": False},
        "panel": {"path": args.panel, "rows": info["n_rows"], "countries": info["n_countries"],
                  "editions": info["n_editions"]},
        "targets": [int(t) for t in targets],
        "protocol": {
            "variant": "base (log-space MSE, no augmented statics, coarse persistence "
                       "blend theta, no host multiplier, no HGBR) == stgcn_predict.run_forecast",
            "walk_forward": "target t is predicted by a model fit only on editions < t; "
                            "internal validation = last 2 pre-target editions",
            "seed_reset": "seed 42 reset before every target training in every cell, so "
                          "all cells share one initialisation/dropout stream; the incumbent "
                          "run drew a single stream over the whole walk-forward, hence the "
                          "incumbent cell is not bit-identical to it",
            "consistency_check": "incumbent-cell vs gen/variants/predictions_base.csv, "
                                 "mean window weighted SMAPE, warn only (see incumbent_check)",
        },
        "parameters": {
            "lambda": {
                "role": "time decay of the dynamic national power weight (paper Sec 3.3)",
                "formula": "W_t^c = ln(M_c(t)+1) / (ln(G_c(t)) * sqrt(P_c(t))) "
                           "* exp(-lambda * (T - t))",
                "incumbent": LAMBDA0,
                "incumbent_half_life_years": round(float(np.log(2.0) / LAMBDA0), 3),
                "grid": lambda_grid,
                "grid_factors": [round(v / LAMBDA0, 4) for v in lambda_grid],
                "code": "stgcn_predict.py:60 (LAMBDA_DECAY), :220-237 (dynamic_weight), "
                        ":403 (step_features), :474 (static_features)",
                "features_affected": ["log1p_dynweight", "dynweight_share",
                                      "log1p_dynweight_at_target", "dynweight_share_at_target"],
            },
            "rho": {
                "role": "L2 regularisation of the network (Adam weight_decay)",
                "formula": "log-MSE loss + rho * ||theta||_2^2 (torch.optim.Adam weight_decay)",
                "incumbent": RHO0,
                "grid": rho_grid,
                "code": "stgcn_predict.py:71 (WEIGHT_DECAY), :788 (Adam in train_target)",
                "note": "the pipeline declares no separate L1 penalty; dropout 0.2 and "
                        "gradient clipping 5.0 belong to the architecture and are held fixed",
            },
        },
        "metrics": {
            "weighted_smape_all": "medal-weighted SMAPE over all country-Games rows of the "
                                  "13 editions (stgcn_predict.weighted_smape, actual > 0 rows)",
            "weighted_smape_window": "same pool under the paper's elastic window "
                                     "(|t'-t| <= 5), mean over Games of SMAPE_w(t)",
            "macro_smape_*": "unweighted mean over Games of SMAPE(t) per reporting period "
                             "(backtest_period.DEFAULT_PERIODS: Cold War 1976-1991, "
                             "Globalization 1992-2016, COVID-19/recent 2017-2024)",
            "best_epoch_mean": "mean early-stopping epoch over the target editions",
        },
        "n_configs": len(cells),
        "cells": [[l, r] for l, r in cells],
        "per_cell_target_detail": cell_detail,
        "best_cell": {"param_lambda": float(best["param_lambda"]),
                      "param_rho": float(best["param_rho"]),
                      "weighted_smape_all": float(best["weighted_smape_all"])},
        "incumbent_cell": (None if inc.empty else
                           {"param_lambda": float(inc.iloc[0]["param_lambda"]),
                            "param_rho": float(inc.iloc[0]["param_rho"]),
                            "weighted_smape_all": float(inc.iloc[0]["weighted_smape_all"])}),
        "timing": {"per_cell_seconds": [round(s, 1) for s in per_cell_seconds],
                   "mean_cell_seconds": round(float(np.mean(per_cell_seconds)), 1),
                   "total_seconds": round(time.time() - t_all, 1),
                   "max_epochs": args.max_epochs,
                   "time_budget_hours": 20},
        "checks": {
            "copy_bit_identical_at_incumbent_lambda": True,
            "anti_leak_assertion": "samples[t]['hist_years'] max < t asserted for every cell",
            "incumbent_check": incumbent_check,
        },
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    meta_path = os.path.join(args.outdir, "grid_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"[done] meta -> {meta_path} (total {meta['timing']['total_seconds']:.1f}s, "
        f"best cell lambda={best['param_lambda']:g} rho={best['param_rho']:g} "
        f"{best['weighted_smape_all']:.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
