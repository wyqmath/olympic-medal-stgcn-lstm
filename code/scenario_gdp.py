import argparse
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

SEED = sp.SEED
DEFAULT_DELTAS = (-0.30, -0.15, -0.05, 0.05, 0.15, 0.30)
DEFAULT_TARGETS = (2028, 2024)
GDP_COLS = ("gdp_total_2011usd_million", "gdp_per_capita_2011usd")
HOSTS_PATH = os.path.join(sp.HERE, os.pardir, "data", "summerOly_hosts_with_ioc.csv")
DEFAULT_HOST_IOC = "USA"
TOP_MOVERS_K = 10


def last_records(panel: pd.DataFrame, target_year: int) -> pd.DataFrame:
    hist = panel[panel["year"] < target_year]
    return hist.sort_values("year").groupby("ioc").tail(1).set_index("ioc")


def replay_state(panel: pd.DataFrame, target_year: int) -> dict:
    hp = panel.sort_values(["year", "ioc"])
    state = {"cum": {}, "last_part": {}, "ma3": {}, "ma_hist": {}, "median": {}}
    for year in sorted(int(y) for y in hp["year"].unique()):
        if year >= target_year:
            break
        ed = hp[hp["year"] == year].sort_values("ioc").set_index("ioc")
        sp.advance_state(ed.assign(year=year), year, state)
    return state


def shocked_static(panel: pd.DataFrame, node_index: dict, samples: dict,
                   target_year: int, delta: float, scope: str = "uniform",
                   host_iocs=()) -> np.ndarray:
    lr = last_records(panel, target_year)
    if delta != 0.0:
        lr = lr.copy()
        if scope == "host":
            sel = lr.index.isin(set(host_iocs))
            assert sel.any(), f"scope=host: no host row among the last records ({host_iocs})"
        else:
            sel = np.ones(len(lr), dtype=bool)
        for c in GDP_COLS:
            if c in lr.columns:
                lr.loc[sel, c] = lr.loc[sel, c] * (1.0 + delta)
    return sp.static_features(target_year, panel, node_index,
                              replay_state(panel, target_year), lr)


def shocked_samples(samples: dict, target_year: int, static: np.ndarray) -> dict:
    out = dict(samples)
    out[target_year] = {**samples[target_year], "static": static}
    return out


def assert_shock_scope(panel: pd.DataFrame, node_index: dict, samples: dict,
                       target_year: int, scope: str = "uniform", host_iocs=()) -> dict:
    base = samples[target_year]
    zero = shocked_static(panel, node_index, samples, target_year, 0.0, scope, host_iocs)
    assert np.array_equal(zero, base["static"]), (
        f"target {target_year}: delta=0 rebuild is not bit-identical to the pipeline")
    hit = shocked_static(panel, node_index, samples, target_year, 0.30, scope, host_iocs)
    moved = float(np.abs(hit - base["static"]).max())
    assert moved > 0.0, f"target {target_year}: +/-30% shock left the static block unchanged"
    assert np.array_equal(hit[:, sp.IDX_LOG1P_LAST_TOTAL], base["static"][:, sp.IDX_LOG1P_LAST_TOTAL]), (
        f"target {target_year}: the persistence anchor must not move with the shock")
    return {"static_max_abs_delta_at_plus30": moved}


def forward_host(host_path: str, log=print) -> str:
    if os.path.exists(host_path):
        h = pd.read_csv(host_path)
        row = h[h["Year"] == 2028]
        if len(row):
            ioc = str(row["IOC_Code"].iloc[-1]).strip()
            log(f"[forward] 2028 host from {os.path.basename(host_path)}: {ioc}")
            return ioc
    log(f"[forward] hosts file unusable ({host_path}); defaulting to {DEFAULT_HOST_IOC}")
    return DEFAULT_HOST_IOC


def extend_panel(panel: pd.DataFrame, target_year: int, host_ioc: str, log=print) -> pd.DataFrame:
    last_year = int(panel["year"].max())
    base = panel[panel["year"] == last_year].copy()
    base["year"] = int(target_year)
    base["host_flag"] = (base["ioc"] == host_ioc).astype(int)
    for c in ("gold", "silver", "bronze", "total"):
        base[c] = 0
    base["gdp_gapfill"] = "carry2y"
    ext = pd.concat([panel, base], ignore_index=True)
    log(f"[forward] extended panel {panel.shape[0]} -> {ext.shape[0]} rows "
        f"(edition {target_year}: {len(base)} countries, host {host_ioc}, "
        f"GDP/pop carried forward from {last_year})")
    return ext


def run_target(panel_use: pd.DataFrame, target_year: int, device, deltas: list,
               max_epochs, realised: bool, scope: str = "uniform", log=print) -> dict:
    node_index = {c: i for i, c in enumerate(sorted(panel_use["ioc"].unique()))}
    editions = sorted(int(y) for y in panel_use["year"].unique())
    samples = sp.build_samples(panel_use, node_index, editions, log=log)
    assert max(samples[target_year]["hist_years"]) < target_year, "anti-leak red line"

    hosts = panel_use.loc[(panel_use["year"] == target_year)
                          & (panel_use["host_flag"] == 1), "ioc"].tolist()
    checks = assert_shock_scope(panel_use, node_index, samples, target_year, scope, hosts)
    log(f"[scope] {target_year}: delta=0 static block bit-identical to the pipeline; "
        f"a +30% ({scope}) shock moves the target statics by up to "
        f"{checks['static_max_abs_delta_at_plus30']:.4f} (anchor column untouched)")

    tgt = panel_use[panel_use["year"] == target_year]
    keep = [node_index[ioc] for ioc in tgt["ioc"] if ioc in node_index]
    actuals = ({ioc: float(v) for ioc, v in zip(tgt["ioc"], tgt["total"])}
               if realised else {})

    preds, recs, footprint = {}, [], {}
    for delta in [0.0] + list(deltas):
        static = shocked_static(panel_use, node_index, samples, target_year,
                                float(delta), scope, hosts)
        cols = [sp.STATIC_FEAT_NAMES[k] for k in range(static.shape[1])
                if not np.array_equal(static[:, k], samples[target_year]["static"][:, k])]
        footprint[f"{float(delta):+.2f}"] = {
            "static_max_abs_delta": float(np.abs(static - samples[target_year]["static"]).max()),
            "static_columns_moved": cols,
        }
        if delta == 0.0:
            assert np.array_equal(static, samples[target_year]["static"])
        sp.set_seed(SEED)
        t0 = time.time()
        pred, meta, _ = sp.run_forecast(panel_use, target_year, device,
                                        samples=shocked_samples(samples, target_year, static),
                                        node_index=node_index, editions=editions,
                                        max_epochs=max_epochs, log=lambda *a, **k: None)
        preds[float(delta)] = pred
        recs.append({"target": int(target_year), "delta": float(delta),
                     "meta": meta, "seconds": time.time() - t0, "keep": keep,
                     "actuals": actuals, "node_index": node_index})
        log(f"[run] {target_year} delta={delta:+.2f}: epoch {meta['best_epoch']}/"
            f"{meta['epochs_run']} theta={meta['blend_theta']} "
            f"sum(pred)={float(np.sum(pred[keep])):.1f} {recs[-1]['seconds']:.1f}s")
    return {"records": recs, "preds": preds, "keep": keep, "actuals": actuals,
            "node_index": node_index, "checks": checks, "samples": samples,
            "panel": panel_use, "hosts": hosts, "footprint": footprint}


def summarise(res: dict, deltas: list) -> list:
    target = res["records"][0]["target"]
    keep, actuals = res["keep"], res["actuals"]
    node_index = res["node_index"]
    rows = []
    base_sum = None
    base_smape = None
    for rec, delta in zip(res["records"], [0.0] + list(deltas)):
        pred = res["preds"][float(delta)]
        total = float(np.sum(pred[keep]))
        if base_sum is None:
            base_sum = total
        row = {"target": int(target), "delta": float(delta),
               "total_pred_sum": round(total, 3),
               "pct_change_vs_0": round(100.0 * (total / base_sum - 1.0), 4),
               "weighted_smape": np.nan, "wsmape_change_pp": np.nan,
               "n_rows": len(keep)}
        if actuals:
            act = np.array([actuals[ioc] for ioc in actuals], dtype=float)
            pr = np.array([pred[node_index[ioc]] for ioc in actuals], dtype=float)
            s = sp.weighted_smape(act, pr)
            if base_smape is None:
                base_smape = s
            row["weighted_smape"] = round(s, 4)
            row["wsmape_change_pp"] = round(s - base_smape, 4)
        rows.append(row)
    return rows


def top_movers(res: dict, deltas: list, country_of: dict, k: int = TOP_MOVERS_K) -> list:
    target = res["records"][0]["target"]
    keep, node_index, actuals = res["keep"], res["node_index"], res["actuals"]
    idx_to_ioc = {j: ioc for ioc, j in node_index.items()}
    base = res["preds"][0.0]
    out = []
    for delta in deltas:
        pred = res["preds"][float(delta)]
        chg = [(float(pred[j] - base[j]), j) for j in keep]
        chg.sort(key=lambda t: (-abs(t[0]), t[1]))
        for d, j in chg[:k]:
            ioc = idx_to_ioc[j]
            out.append({"target": int(target), "delta": float(delta), "ioc": ioc,
                        "country": country_of.get(ioc, ioc),
                        "pred_delta0": round(float(base[j]), 3),
                        "pred_delta": round(float(pred[j]), 3),
                        "abs_change": round(d, 3),
                        "pct_change": round(100.0 * d / base[j], 2) if base[j] > 0 else np.nan,
                        "actual_or_placeholder": actuals.get(ioc, np.nan)})
    return out


def write_curve(rows: pd.DataFrame, path: str, log=print) -> bool:
    fwd = rows[rows["weighted_smape"].isna() & rows["pct_change_vs_0"].notna()]
    real = rows[rows["wsmape_change_pp"].notna() & (rows["delta"] != 0.0)]
    if not len(fwd) and not len(real):
        log("[curve] no curve to draw (no forward and no realised arm); skipped")
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    handles = []
    if len(fwd):
        d = fwd.sort_values("delta")
        h, = ax1.plot(d["delta"].abs(), d["pct_change_vs_0"], marker="o", ls="-",
                      color="black", label="forecast total change (%)")
        handles.append(h)
    ax1.set_xlabel(r"|$\delta$| : GDP shock magnitude")
    ax1.set_ylabel("Change of summed medal forecast (%)", color="black")
    ax1.axhline(0.0, lw=0.6, color="0.6")
    if len(real):
        ax2 = ax1.twinx()
        d = real.sort_values("delta")
        h2, = ax2.plot(d["delta"].abs(), d["wsmape_change_pp"], marker="s", ls="--",
                       color="0.35", label="weighted SMAPE change (pp)")
        handles.append(h2)
        ax2.set_ylabel("Change of weighted SMAPE (pp)", color="0.35")
        ax2.tick_params(axis="y", colors="0.35")
    if handles:
        ax1.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=sp.PANEL_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deltas", default=",".join(f"{d:g}" for d in DEFAULT_DELTAS),
                    help="comma-separated GDP shock fractions (multiplier = 1 + delta)")
    ap.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS),
                    help="comma-separated target editions; years beyond the panel are "
                         "forward scenarios (default 2028,2024)")
    ap.add_argument("--max-epochs", type=int, default=sp.MAX_EPOCHS)
    ap.add_argument("--outdir", default=os.path.join(sp.GEN_DIR, "scenarios"))
    ap.add_argument("--hosts", default=HOSTS_PATH)
    ap.add_argument("--scope", choices=("uniform", "host"), default="uniform",
                    help="uniform (default, pre-registered): every country's GDP is scaled; "
                         "host: only the target edition's host NOC -- a country-specific "
                         "shock, because a uniform shock nearly cancels in the z-scored "
                         "economy covariates (log1p nonlinearity residual)")
    args = ap.parse_args()

    log = print
    sp.set_seed(SEED)
    device = torch.device(args.device)

    assert os.path.exists(args.panel), f"panel not found: {args.panel}"
    panel = sp.normalise_panel(pd.read_csv(args.panel), log=log)
    info = sp.assert_panel_contract(panel)
    deltas = [float(v) for v in str(args.deltas).split(",") if str(v).strip()]
    targets = [int(v) for v in str(args.targets).split(",") if str(v).strip()]
    assert len(deltas) >= 1, "--deltas must hold at least one value"
    assert all(-0.99 < d for d in deltas), "delta must stay above -1 (GDP stays positive)"
    log(f"[scen] deltas={deltas}, targets={targets}, scope={args.scope}, "
        f"max_epochs={args.max_epochs}, device={device}")
    if args.scope == "uniform":
        log("[scen] note: the GDP covariates are within-edition z-scores of log1p GDP, so a "
            "uniform all-country shock nearly cancels there (log1p nonlinearity residual); "
            "the forecast response is expected to be small regardless of delta")
    log(f"[panel] {args.panel}: {info['n_rows']} rows, {info['n_countries']} countries, "
        f"editions {info['years'][0]}..{info['years'][-1]}")

    country_of = (panel.drop_duplicates("ioc").set_index("ioc")["country"].to_dict())
    rows, movers, per_target = [], [], {}
    t_all = time.time()
    for target in targets:
        is_forward = target > int(panel["year"].max())
        if is_forward:
            panel_use = extend_panel(panel, target, forward_host(args.hosts, log=log), log=log)
        else:
            assert target in info["targets"], (
                f"target {target} is not an edition of the panel ({info['targets']})")
            panel_use = panel
        t0 = time.time()
        res = run_target(panel_use, target, device, deltas, args.max_epochs,
                         realised=not is_forward, scope=args.scope, log=log)
        recs = summarise(res, deltas)
        rows.extend(recs)
        if is_forward:
            movers.extend(top_movers(res, deltas, country_of))
        per_target[str(target)] = {
            "arm": "forward" if is_forward else "realised",
            "scope": args.scope,
            "seconds": round(time.time() - t0, 1),
            "scope_checks": res["checks"],
            "n_countries": len(res["keep"]),
            "host_ioc": res["hosts"],
            "shock_input_footprint": res["footprint"],
            "per_delta_seconds": [round(r["seconds"], 1) for r in res["records"]],
            "best_epochs": {f"{r['delta']:+.2f}": int(r["meta"]["best_epoch"])
                            for r in res["records"]},
        }
        del res
        log(f"[done] {target}: {len(recs)} delta rows in {per_target[str(target)]['seconds']}s")

    os.makedirs(args.outdir, exist_ok=True)
    table = pd.DataFrame(rows)[["target", "delta", "total_pred_sum", "pct_change_vs_0",
                                "weighted_smape", "wsmape_change_pp", "n_rows"]]
    csv_path = os.path.join(args.outdir, "scenario_table.csv")
    table.to_csv(csv_path, index=False)
    log(f"[done] {len(table)} scenario rows -> {csv_path}")

    movers_cols = ["target", "delta", "ioc", "country", "pred_delta0", "pred_delta",
                   "abs_change", "pct_change", "actual_or_placeholder"]
    movers_path = os.path.join(args.outdir, "top_movers.csv")
    movers_df = (pd.DataFrame(movers)[movers_cols] if movers
                 else pd.DataFrame(columns=movers_cols))
    movers_df.to_csv(movers_path, index=False)
    log(f"[done] {len(movers_df)} top-mover rows -> {movers_path}")

    curve_path = os.path.join(args.outdir, "scenario_curve.pdf")
    if write_curve(table, curve_path, log=log):
        log(f"[done] curve -> {curve_path}")

    fwd = table[table["weighted_smape"].isna() & (table["delta"] != 0.0)]
    real = table[table["wsmape_change_pp"].notna() & (table["delta"] != 0.0)]
    meta = {
        "script": "scenario_gdp.py",
        "purpose": "paper Sec 5.2 'Extreme Scenario Testing': GDP shocks of +-5/15/30% "
                   "pushed through the accepted STGCN-LSTM pipeline",
        "seed": SEED,
        "device": str(device),
        "determinism": {"CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                        "torch_use_deterministic_algorithms": True,
                        "cudnn_benchmark": False},
        "panel": {"path": args.panel, "rows": info["n_rows"], "countries": info["n_countries"],
                  "editions": info["years"]},
        "deltas": deltas,
        "targets": targets,
        "shock_scope": {
            "selected": args.scope,
            "uniform": "every country's GDP columns are scaled by (1 + delta) "
                       "(pre-registered default)",
            "host": "only the target edition's host NOC is scaled (optional secondary "
                    "reading: a uniform shock nearly cancels in the within-edition z-scored "
                    "economy covariates -- log1p nonlinearity residual, see "
                    "shock_input_footprint per target -- so the country-specific reading "
                    "gives the shock more grip)",
        },
        "perturbation_semantics": {
            "definition": "gdp_total_2011usd_million and gdp_per_capita_2011usd of the "
                          "affected countries are multiplied by (1 + delta); population is "
                          "untouched",
            "edition_scope": "target edition only -- the shock is applied to the GDP columns "
                             "of each country's latest pre-target record, i.e. the values the "
                             "target static block carries as 'the economy at the target "
                             "Games'; earlier editions' step features, the graph, the labels, "
                             "the internal validation editions and the persistence anchor are "
                             "identical across deltas",
            "why_this_level": "the forecaster never observes the target edition, so its "
                              "economy enters the model only through the latest observed "
                              "(carry-forward) record; scaling the target-edition panel rows "
                              "instead would change nothing in this pipeline",
            "correspondence_to_paper": "Sec 5.2 sudden policy changes / external shocks and "
                                       "the mild (+-5%), moderate (+-15%) and extreme "
                                       "(+-30%) stress levels",
            "implementation": "stgcn_predict.static_features re-run on a shocked "
                              "last_records frame (no feature code copied); delta = 0 is "
                              "asserted bit-identical to the pipeline's own static block",
            "training_invariance": "the trained weights cannot depend on delta by "
                                   "construction; every delta still runs the full "
                                   "train + blend + predict path so baseline and scenarios "
                                   "share one code path",
        },
        "protocol": {
            "variant": "base == stgcn_predict.run_forecast (log-space MSE, coarse persistence "
                       "blend theta on the internal validation editions, no augmented "
                       "statics / host multiplier / HGBR)",
            "walk_forward": "target t is predicted by a model fit only on editions < t",
            "n_trainings": len(rows),
            "forward_arm": {
                "target": [t for t in targets if t > int(panel["year"].max())],
                "participant_set": "participants of the latest observed edition",
                "host": "from the hosts file (2028 -> USA)",
                "medal_rows": "zero-filled placeholders: a future edition has no medals and "
                              "target-edition rows are never used as labels or features",
                "gdp_population": "carried forward from the latest observed edition "
                                  "(gdp_gapfill = carry2y)",
            },
            "real_arm_metric": "medal-weighted SMAPE over the target edition's participants "
                               "(stgcn_predict.weighted_smape)",
        },
        "headline": {
            "forward_total_change_pct_by_delta": {
                f"{r['delta']:+.2f}": r["pct_change_vs_0"] for _, r in fwd.iterrows()},
            "real_wsmape_change_pp_by_delta": {
                f"{r['delta']:+.2f}": r["wsmape_change_pp"] for _, r in real.iterrows()},
        },
        "per_target": per_target,
        "timing": {"total_seconds": round(time.time() - t_all, 1),
                   "max_epochs": args.max_epochs},
        "checks": {
            "delta0_static_bit_identical": True,
            "persistence_anchor_invariant": True,
            "anti_leak_assertion": "samples[target]['hist_years'] max < target",
            "contract_assertions": "run on the observed panel; the forward extension is "
                                   "outside the contract (year > 2024, medal placeholders)",
        },
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    meta_path = os.path.join(args.outdir, "scenario_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"[done] meta -> {meta_path} (total {meta['timing']['total_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
