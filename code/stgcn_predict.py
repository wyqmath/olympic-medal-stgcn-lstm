from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(HERE, "gen")
PANEL_PATH = os.path.join(HERE, os.pardir, "data", "panel_clean.csv")
PAGERANK_PATH = os.path.join(GEN_DIR, "sport_pagerank_scores.csv")
PRED_PATH = os.path.join(GEN_DIR, "predictions_stgcn.csv")
SYNTH_PATH = os.path.join(GEN_DIR, "synthetic_panel_dev.csv")
RUNLOG_PATH = os.path.join(GEN_DIR, "stgcn_run_log.json")

PANEL_COLUMNS = [
    "country", "ioc", "year", "gold", "silver", "bronze", "total", "host_flag",
    "gdp_total_2011usd_million", "gdp_per_capita_2011usd", "population", "gdp_gapfill",
]
GAPFILL_VALUES = {"raw", "proxy_deu", "proxy_sum", "proxy_hist", "carry2y", "missing"}

SEED = 42
LAMBDA_DECAY = 0.05
HALF_LIFE = float(np.log(2.0) / LAMBDA_DECAY)
SEQ_LEN = 8
KNN_EDGES = 8
MEDAL_SIM_THRESH = 0.2
MIN_TRAIN_EDITION = 1920
N_VAL_EDITIONS = 2
MAX_EPOCHS = 400
PATIENCE = 40
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP = 10
STEP_SIZE = 80
STEP_GAMMA = 0.5
HIDDEN = 64
HEADS = 4
LSTM_HIDDEN = 128
DROPOUT = 0.2
N_STEP_FEAT = 20
N_STATIC_FEAT = 14
IDX_LOG1P_LAST_TOTAL = 12
STEP_FEAT_NAMES = [
    "participates", "log1p_total", "log1p_gold", "log1p_silver", "log1p_bronze",
    "medal_share", "gold_ratio", "z_log_gdp", "gdp_missing", "z_log_gdppc",
    "gdppc_missing", "z_log_pop", "pop_missing", "host_flag", "log1p_dynweight",
    "dynweight_share", "years_since_participation", "ma3_log1p_total",
    "z_walkforward_pagerank", "log1p_cumulative_total",
]
STATIC_FEAT_NAMES = [
    "target_host_flag", "z_log_gdp", "gdp_missing", "z_log_gdppc", "gdppc_missing",
    "z_log_pop", "pop_missing", "log1p_dynweight_at_target", "dynweight_share_at_target",
    "years_since_participation", "log1p_cumulative_total", "ma3_log1p_total",
    "log1p_last_total", "log1p_past_median_total",
]


def normalise_panel(df: pd.DataFrame, log=print) -> pd.DataFrame:
    df = df.copy()
    for c in ["country", "ioc", "gdp_gapfill"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    n0 = len(df)
    dropped = df[df["ioc"].isna() | df["ioc"].eq("")]
    if len(dropped):
        log(f"[panel] dropping {len(dropped)} rows without an IOC code: "
            f"{sorted(set(dropped['country'].tolist()))} "
            f"(years {sorted(set(int(y) for y in dropped['year']))})")
    df = df[df["ioc"].notna() & ~df["ioc"].eq("")].copy()
    log(f"[panel] {n0} -> {len(df)} rows after non-NOC removal")
    return df


def assert_panel_contract(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in PANEL_COLUMNS if c not in df.columns]
    assert not missing_cols, f"panel missing columns: {missing_cols}"
    for c in ["country", "ioc", "year", "gold", "silver", "bronze", "total", "host_flag"]:
        assert not df[c].isna().any(), f"NaN in required column {c}"

    df = df.copy()
    for c in ["year", "gold", "silver", "bronze", "total", "host_flag"]:
        df[c] = df[c].astype(int)

    dup = int(df.duplicated(["ioc", "year"]).sum())
    assert dup == 0, f"{dup} duplicate (ioc,year) rows -- many-to-one mapping bug"
    bad_sum = int((df["gold"] + df["silver"] + df["bronze"] != df["total"]).sum())
    assert bad_sum == 0, f"{bad_sum} rows where total != gold+silver+bronze"
    for c in ["gold", "silver", "bronze", "total"]:
        assert (df[c] >= 0).all(), f"negative values in {c}"
    assert df["ioc"].str.fullmatch(r"[A-Z0-9]{3}").all(), "ioc codes must be 3-char uppercase"
    assert df["host_flag"].isin([0, 1]).all(), "host_flag must be 0/1"

    gf = df["gdp_gapfill"].astype(str).str.strip()
    bad_gf = set(gf.unique()) - GAPFILL_VALUES
    assert not bad_gf, f"unexpected gdp_gapfill values: {bad_gf}"
    gdp = df["gdp_total_2011usd_million"]
    flagged = gf.eq("missing")
    assert int((gdp.notna() & flagged).sum()) == 0, "GDP value present but gapfill='missing'"
    assert int((gdp.isna() & ~flagged).sum()) == 0, "NaN GDP must be flagged gapfill='missing'"

    years = sorted(int(y) for y in df["year"].unique())
    assert 1896 <= years[0] and years[-1] <= 2024, f"year range {years[0]}..{years[-1]}"
    targets = [y for y in years if 1976 <= y <= 2024]
    assert len(targets) >= 13, f"only {len(targets)} editions in 1976-2024: {targets}"

    return {
        "n_rows": int(len(df)),
        "n_countries": int(df["ioc"].nunique()),
        "n_editions": len(years),
        "years": years,
        "targets": targets,
        "gdp_missing_rows": int((gdp.isna() | flagged).sum()),
        "gapfill_counts": {k: int(v) for k, v in gf.value_counts().items()},
        "medal_sum": int(df["total"].sum()),
    }


def make_synthetic_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    editions = [y for y in range(1896, 2025, 4) if y not in (1916, 1940, 1944)]
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ioc = sorted({"".join(letters[rng.integers(0, 26, 3)]) for _ in range(150)})
    n_c = len(ioc)
    strength0 = rng.gamma(1.2, 1.0, n_c)
    gdp0 = np.exp(rng.normal(4, 2.0, n_c)) * 1000.0
    pop0 = np.exp(rng.normal(3, 2.2, n_c)) * 1e6
    host_idx = rng.integers(0, n_c, len(editions))

    rows = []
    for e_i, year in enumerate(editions):
        host = ioc[host_idx[e_i]]
        s = strength0 * np.exp(0.05 * (year - 1896) / 32.0 + rng.normal(0, 0.25, n_c))
        gdp = gdp0 * np.exp(0.02 * (year - 1896) / 4.0 + rng.normal(0, 0.15, n_c))
        pop = pop0 * np.exp(0.01 * (year - 1896) / 4.0 + rng.normal(0, 0.05, n_c))
        gdp_pc = gdp * 1e6 / pop
        lam = 6.0 * s * (1 + rng.normal(0, 0.05, n_c))
        total = rng.poisson(np.clip(lam, 0, 200))
        if year in (1980, 1984):
            total = np.where(rng.random(n_c) < 0.18, 0, total)
        for j in range(n_c):
            if total[j] == 0 and rng.random() < 0.45:
                continue
            g = int(min(total[j], rng.binomial(total[j], 0.35 + 0.2 * rng.random())))
            left = total[j] - g
            sv = int(min(left, rng.binomial(left, 0.5)))
            b = left - sv
            u = rng.random()
            if u < 0.10:
                gdp_v, gf = np.nan, "missing"
            elif u < 0.16:
                gdp_v, gf = gdp[j], str(rng.choice(["proxy_hist", "carry2y", "proxy_deu"]))
            else:
                gdp_v, gf = gdp[j], "raw"
            rows.append({
                "country": f"Ctry_{ioc[j]}", "ioc": ioc[j], "year": int(year),
                "gold": g, "silver": sv, "bronze": b, "total": int(g + sv + b),
                "host_flag": int(ioc[j] == host),
                "gdp_total_2011usd_million": gdp_v,
                "gdp_per_capita_2011usd": gdp_pc[j],
                "population": pop[j],
                "gdp_gapfill": gf,
            })
    return pd.DataFrame(rows)


def dynamic_weight(log_total, gdp_million, population, edition_year, target_year,
                   lam: float = LAMBDA_DECAY) -> np.ndarray:
    gdp_h = np.asarray(gdp_million, dtype=float) / 100.0
    pop_m = np.asarray(population, dtype=float) / 1e6
    ok_g = np.isfinite(gdp_h) & (gdp_h > 0)
    ok_p = np.isfinite(pop_m) & (pop_m > 0)
    g_term = np.where(ok_g, np.log(np.where(ok_g, gdp_h, 1.0)), 0.5)
    g_term = np.maximum(g_term, 0.5)
    p_term = np.maximum(np.where(ok_p, np.sqrt(np.where(ok_p, pop_m, 1.0)), 1.0), 1.0)
    decay = np.exp(-lam * (float(target_year) - np.asarray(edition_year, dtype=float)))
    return np.asarray(log_total, dtype=float) / (g_term * p_term + 1e-10) * decay


def walkforward_pagerank(share_by_year: dict, node_index: dict, d: float = 0.85,
                         iters: int = 60) -> np.ndarray:
    n = len(node_index)
    w = np.zeros((n, n), dtype=float)
    for sh in share_by_year.values():
        if not sh:
            continue
        idx = np.array([node_index[c] for c in sh if c in node_index])
        val = np.array([sh[c] for c in sh if c in node_index], dtype=float)
        w[np.ix_(idx, idx)] += np.clip(val[:, None] - val[None, :], 0, None)
    out = w.sum(axis=1, keepdims=True)
    norm = np.where(out > 0, out, 1.0)
    p = np.full(n, 1.0 / max(n, 1))
    for _ in range(iters):
        pn = (1 - d) / n + d * ((w / norm).T @ p)
        pn /= max(pn.sum(), 1e-12)
        if np.abs(pn - p).max() < 1e-10:
            return pn
        p = pn
    return p


def _zcols(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return np.where(np.isfinite(x), (x - mu) / sd, 0.0)


def _knn_edges(feat: np.ndarray, k: int, gaussian: bool = True):
    n = feat.shape[0]
    d2 = ((feat[:, None, :] - feat[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    k = max(1, min(k, n - 1))
    idx = np.argsort(d2, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    dist = d2[rows, cols]
    if gaussian:
        finite = dist[np.isfinite(dist)]
        sigma = max(float(np.median(finite)) if finite.size else 1.0, 1e-6)
        w = np.exp(-dist / (2 * sigma ** 2))
    else:
        w = np.ones_like(dist)
    return rows.astype(np.int64), cols.astype(np.int64), w.astype(float)


def _topk_edges(sim: np.ndarray, k: int, thresh: float):
    n = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    k = max(1, min(k, n - 1))
    idx = np.argsort(-s, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    w = s[rows, cols]
    keep = np.isfinite(w) & (w > thresh)
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64), np.clip(w[keep], 0, None)


def _merge_undirected(families, n: int):
    best: dict = {}
    for rows, cols, w in families:
        for r, c, wt in zip(rows, cols, w):
            if r == c:
                continue
            key = (min(int(r), int(c)), max(int(r), int(c)))
            cur = best.get(key)
            if cur is None or wt > cur[2]:
                best[key] = (int(r), int(c), float(wt))
    src = []
    for (r, c, wt) in best.values():
        src.append((r, c, wt))
        src.append((c, r, wt))
    src.extend((i, i, 1.0) for i in range(n))
    arr = np.array(src, dtype=float)
    return np.stack([arr[:, 0], arr[:, 1]]).astype(np.int64), arr[:, 2].astype(np.float32)


def build_graph(hist: pd.DataFrame, cutoff_year: int, node_index: dict):
    n = len(node_index)
    past = hist[hist["year"] < cutoff_year]
    years = sorted(int(y) for y in past["year"].unique())

    latest = past.sort_values("year").groupby("ioc").tail(1).set_index("ioc")
    econ = np.full((n, 3), np.nan)
    for ioc, r in latest.iterrows():
        j = node_index.get(ioc)
        if j is None:
            continue
        econ[j] = [
            np.log1p(r["gdp_total_2011usd_million"]) if pd.notna(r["gdp_total_2011usd_million"]) else np.nan,
            np.log1p(r["gdp_per_capita_2011usd"]) if pd.notna(r["gdp_per_capita_2011usd"]) else np.nan,
            np.log1p(r["population"]) if pd.notna(r["population"]) else np.nan,
        ]
    fam_a = _knn_edges(_zcols(econ), KNN_EDGES, gaussian=True)

    tot_year = past.groupby("year")["total"].sum()
    mat = np.zeros((n, max(len(years), 1)))
    share: dict = {}
    for k_i, y in enumerate(years):
        sub = past[past["year"] == y]
        d = float(tot_year.get(y, 0))
        for ioc, tot in zip(sub["ioc"], sub["total"]):
            j = node_index.get(ioc)
            if j is None:
                continue
            s = (float(tot) / d) if d > 0 else 0.0
            mat[j, k_i] = s
            share.setdefault(y, {})[ioc] = s
    nrm = np.linalg.norm(mat, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    matn = mat / nrm
    fam_b = _topk_edges(matn @ matn.T, KNN_EDGES, thresh=MEDAL_SIM_THRESH)

    ei, ew = _merge_undirected([fam_a, fam_b], n)
    return ei, ew, share


def step_features(ed: pd.DataFrame, state: dict, pr: np.ndarray, node_index: dict) -> np.ndarray:
    n = len(node_index)
    f = np.zeros((n, N_STEP_FEAT), dtype=np.float32)
    if ed is None or len(ed) == 0:
        return f
    ed_year = int(ed["year"].iloc[0])
    ed_total = float(ed["total"].sum())
    raw = np.stack([
        np.log1p(ed["gdp_total_2011usd_million"].to_numpy(dtype=float)),
        np.log1p(ed["gdp_per_capita_2011usd"].to_numpy(dtype=float)),
        np.log1p(ed["population"].to_numpy(dtype=float)),
    ], axis=1)
    ez = _zcols(np.where(np.isfinite(raw), raw, np.nan))
    pr_z = _zcols(pr.reshape(-1, 1)).ravel()

    lt = np.log1p(ed["total"].to_numpy(dtype=float))
    w_all = np.maximum(dynamic_weight(lt, ed["gdp_total_2011usd_million"].to_numpy(dtype=float),
                                      ed["population"].to_numpy(dtype=float),
                                      np.full(len(ed), ed_year), ed_year), 0.0)
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


def advance_state(ed: pd.DataFrame, year: int, state: dict) -> None:
    for ioc, r in ed.iterrows():
        state["cum"][ioc] = state["cum"].get(ioc, 0.0) + float(r["total"])
        h = state["ma_hist"].setdefault(ioc, [])
        h.append(float(np.log1p(float(r["total"]))))
        state["ma3"][ioc] = float(np.mean(h[-3:]))
        state["median"][ioc] = float(np.median(h))
        state["last_part"][ioc] = int(year)


def static_features(target_year: int, panel: pd.DataFrame, node_index: dict,
                    state: dict, last_records: pd.DataFrame | None) -> np.ndarray:
    n = len(node_index)
    s = np.zeros((n, N_STATIC_FEAT), dtype=np.float32)
    if last_records is None or len(last_records) == 0:
        return s
    lr = last_records
    raw = np.stack([
        np.log1p(lr["gdp_total_2011usd_million"].to_numpy(dtype=float)),
        np.log1p(lr["gdp_per_capita_2011usd"].to_numpy(dtype=float)),
        np.log1p(lr["population"].to_numpy(dtype=float)),
    ], axis=1)
    ez = _zcols(np.where(np.isfinite(raw), raw, np.nan))
    years = lr["year"].to_numpy(dtype=float)
    w_all = dynamic_weight(np.log1p(lr["total"].to_numpy(dtype=float)),
                           lr["gdp_total_2011usd_million"].to_numpy(dtype=float),
                           lr["population"].to_numpy(dtype=float),
                           years, int(target_year))
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


def build_samples(panel: pd.DataFrame, node_index: dict, editions: list, log=print) -> dict:
    hp = panel.sort_values(["year", "ioc"])
    years = sorted(int(y) for y in hp["year"].unique())
    state = {"cum": {}, "last_part": {}, "ma3": {}, "ma_hist": {}, "median": {}}
    graph_cache, pr_cache, step_cache, static_cache, years_before_cache = {}, {}, {}, {}, {}

    for year in years:
        hist = hp[hp["year"] < year]
        years_before_cache[year] = sorted(int(y) for y in hist["year"].unique())
        ei, ew, share = build_graph(hist, year, node_index)
        graph_cache[year] = (ei, ew)
        pr_cache[year] = walkforward_pagerank(share, node_index)
        ed = hp[hp["year"] == year].sort_values("ioc").set_index("ioc")
        ed = ed.assign(year=year)
        step_cache[year] = step_features(ed, state, pr_cache[year], node_index)
        last_records = None
        if years_before_cache[year]:
            last_records = (hist.sort_values("year").groupby("ioc").tail(1)
                            .set_index("ioc"))
        static_cache[year] = static_features(year, panel, node_index, state, last_records)
        advance_state(ed, year, state)

    samples = {}
    for year in editions:
        ei, ew = graph_cache[year]
        years_before = years_before_cache[year]
        hyears = years_before[-SEQ_LEN:]
        steps = np.zeros((SEQ_LEN, len(node_index), N_STEP_FEAT), dtype=np.float32)
        for k, y in enumerate(hyears):
            steps[SEQ_LEN - len(hyears) + k] = step_cache[y]

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
    return samples


def collate(sample_list):
    B = len(sample_list)
    T, N, _ = sample_list[0]["steps"].shape
    xs, eis, ews, ys, ms, sts = [], [], [], [], [], []
    for b, s in enumerate(sample_list):
        steps = torch.from_numpy(s["steps"])
        xs.append(steps.reshape(T * N, -1))
        off = b * T * N
        ei = torch.from_numpy(s["edge_index"])
        ew = torch.from_numpy(s["edge_weight"])
        for k in range(T):
            eis.append(ei + off + k * N)
            ews.append(ew)
        ys.append(torch.from_numpy(s["y"]))
        ms.append(torch.from_numpy(s["mask"]))
        sts.append(torch.from_numpy(s["static"]))
    x = torch.cat(xs, 0)
    edge_index = torch.cat(eis, 1)
    edge_weight = torch.cat(ews, 0)
    y = torch.cat(ys, 0)
    mask = torch.cat(ms, 0)
    static = torch.cat(sts, 0)
    return x, edge_index, edge_weight, static, y, mask, B, T, N


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class STGCN_LSTM(nn.Module):

    def __init__(self, in_dim: int = N_STEP_FEAT, static_dim: int = N_STATIC_FEAT,
                 hidden: int = HIDDEN, heads: int = HEADS, lstm_hidden: int = LSTM_HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.static_dim = static_dim
        h = hidden * heads
        self.in_proj = nn.Linear(in_dim, h)
        self.conv1 = GATConv(in_dim, hidden, heads=heads, edge_dim=1, concat=True)
        self.norm1 = nn.LayerNorm(h)
        self.conv2 = GATConv(h, hidden, heads=heads, edge_dim=1, concat=True)
        self.norm2 = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)
        self.act = Swish()
        self.lstm = nn.LSTM(h, lstm_hidden // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(2 * (lstm_hidden // 2), num_heads=4,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(2 * (lstm_hidden // 2))
        self.ln2 = nn.LayerNorm(2 * (lstm_hidden // 2))
        d = 2 * (lstm_hidden // 2)
        self.highway = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(d + static_dim, hidden), Swish(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_weight, static, B: int, T: int, N: int):
        res = self.in_proj(x)
        h = self.drop(self.act(self.norm1(self.conv1(x, edge_index, edge_weight.view(-1, 1))))) + res
        h = self.drop(self.act(self.norm2(self.conv2(h, edge_index, edge_weight.view(-1, 1))))) + h
        h = h.view(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        lstm_out, _ = self.lstm(h)
        att, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.ln1(lstm_out + att)
        last = out[:, -1, :]
        pooled = out.mean(dim=1)
        hw = self.highway(torch.cat([last, pooled], dim=-1))
        g = self.gate(torch.cat([last, pooled], dim=-1))
        fused = self.ln2(g * hw + (1.0 - g) * last)
        z = torch.cat([fused, static], dim=-1)
        raw = self.head(z).view(B, N)
        anchor = static[:, IDX_LOG1P_LAST_TOTAL].view(B, N)
        return F.relu(anchor + raw)


def loss_fn(log_pred, y, mask):
    return F.mse_loss(log_pred.reshape(-1)[mask], y[mask])


def set_seed(seed: int = SEED, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def lr_lambda(step):
    if step < WARMUP:
        return (step + 1) / WARMUP
    return STEP_GAMMA ** (((step - WARMUP) // STEP_SIZE) + 1)


def run_epoch(model, samples, editions, opt, device):
    model.train()
    total, nb = 0.0, 0
    order = list(editions)
    for i in range(0, len(order), BATCH_SIZE):
        batch = [samples[e] for e in order[i:i + BATCH_SIZE]]
        x, ei, ew, static, y, mask, B, T, N = collate(batch)
        x, ei, ew = x.to(device), ei.to(device), ew.to(device)
        static, y, mask = static.to(device), y.to(device), mask.to(device)
        pred = model(x, ei, ew, static, B, T, N)
        loss = loss_fn(pred, y, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += float(loss.item()) * B
        nb += B
    return total / max(nb, 1)


@torch.no_grad()
def eval_epoch(model, samples, editions, device):
    model.eval()
    total, nb = 0.0, 0
    for i in range(0, len(editions), BATCH_SIZE):
        batch = [samples[e] for e in editions[i:i + BATCH_SIZE]]
        x, ei, ew, static, y, mask, B, T, N = collate(batch)
        x, ei, ew = x.to(device), ei.to(device), ew.to(device)
        static, y, mask = static.to(device), y.to(device), mask.to(device)
        loss = loss_fn(model(x, ei, ew, static, B, T, N), y, mask)
        total += float(loss.item()) * B
        nb += B
    return total / max(nb, 1)


@torch.no_grad()
def predict_target(model, samples, year, device):
    model.eval()
    x, ei, ew, static, _y, mask, B, T, N = collate([samples[year]])
    x, ei, ew, static = x.to(device), ei.to(device), ew.to(device), static.to(device)
    log_pred = model(x, ei, ew, static, B, T, N)[0]
    return np.expm1(log_pred.clamp(min=0.0).cpu().numpy())


def weighted_smape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    e = 200.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))
    ok = np.isfinite(e) & (actual > 0)
    tot = actual[ok].sum()
    return float((actual[ok] * e[ok]).sum() / tot) if tot > 0 else float("inf")


def split_train_val(editions_before):
    pool = [e for e in editions_before if e >= MIN_TRAIN_EDITION]
    if len(pool) >= 8:
        return pool[:-N_VAL_EDITIONS], pool[-N_VAL_EDITIONS:]
    return list(pool), []


@torch.no_grad()
def select_blend_theta(model, samples, val_editions, device,
                       grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if not val_editions:
        return 0.5, None
    ys, ps, as_ = [], [], []
    for e in val_editions:
        pred = predict_target(model, samples, e, device)
        m = samples[e]["mask"]
        ys.append(np.expm1(samples[e]["y"][m]))
        ps.append(pred[m])
        as_.append(np.expm1(samples[e]["static"][m, IDX_LOG1P_LAST_TOTAL]))
    ys, ps, as_ = np.concatenate(ys), np.concatenate(ps), np.concatenate(as_)
    scores = {th: weighted_smape(ys, th * ps + (1 - th) * as_) for th in grid}
    best = min(scores, key=lambda k: (scores[k], k))
    return float(best), scores


def train_target(samples, target_year, editions_before, device, max_epochs=None, log=print):
    max_epochs = max_epochs or MAX_EPOCHS
    train_ed, val_ed = split_train_val(editions_before)

    model = STGCN_LSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = run_epoch(model, samples, train_ed, opt, device)
        sched.step()
        if val_ed:
            va = eval_epoch(model, samples, val_ed, device)
            if va < best_val - 1e-6:
                best_val, best_state, best_epoch, bad = va, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        elif tr < best_val - 1e-6:
            best_val, best_state, best_epoch, bad = tr, copy.deepcopy(model.state_dict()), epoch, 0
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"epochs_run": epoch, "best_epoch": best_epoch, "val_editions": list(val_ed),
                   "best_val_log_mse": best_val, "seconds": time.time() - t0,
                   "n_train_samples": len(train_ed), "n_val_samples": len(val_ed)}


def run_forecast(panel: pd.DataFrame, target_year: int, device,
                 samples=None, node_index=None, editions=None, max_epochs=None, log=print):
    if node_index is None:
        node_index = {c: i for i, c in enumerate(sorted(panel["ioc"].unique()))}
    if editions is None:
        editions = sorted(int(y) for y in panel["year"].unique())
    if samples is None:
        samples = build_samples(panel, node_index, [e for e in editions if e <= target_year], log=log)
    editions_before = [e for e in editions if e < target_year]
    model, meta = train_target(samples, target_year, editions_before, device,
                               max_epochs=max_epochs, log=log)
    raw_pred = predict_target(model, samples, target_year, device)
    theta, scores = select_blend_theta(model, samples, meta_val_editions(meta), device)
    anchor_full = np.expm1(samples[target_year]["static"][:, IDX_LOG1P_LAST_TOTAL])
    pred = float(theta) * raw_pred + (1.0 - float(theta)) * anchor_full
    meta["blend_theta"] = theta
    meta["val_blend_scores"] = None if scores is None else {str(k): round(v, 4) for k, v in scores.items()}
    del model
    return pred, meta, node_index


def meta_val_editions(meta):
    return meta.get("val_editions", [])


def perturb(panel: pd.DataFrame, since_year: int, medals: float = 1.0,
            gdp: float = 1.0, pop: float = 1.0, host_shift: bool = False) -> pd.DataFrame:
    p = panel.copy()
    m = p["year"] >= since_year
    if medals != 1.0:
        for c in ["gold", "silver", "bronze", "total"]:
            p.loc[m, c] = (p.loc[m, c] * medals).round().astype(int)
    if gdp != 1.0:
        p.loc[m, "gdp_total_2011usd_million"] = p.loc[m, "gdp_total_2011usd_million"] * gdp
        p.loc[m, "gdp_per_capita_2011usd"] = p.loc[m, "gdp_per_capita_2011usd"] * gdp
    if pop != 1.0:
        p.loc[m, "population"] = p.loc[m, "population"] * pop
    if host_shift:
        h = p["host_flag"] == 1
        p.loc[h & (p["year"] >= since_year), "host_flag"] = 0
    return p


def leak_selftest(panel: pd.DataFrame, targets, device, max_epochs=None, log=print) -> list:
    quiet = lambda *a, **k: None
    results = []
    for t in targets:
        log(f"[leak] {t}: tier 1 data-path (exact)")
        node_index = {c: i for i, c in enumerate(sorted(panel["ioc"].unique()))}
        base_samples = build_samples(panel, node_index, [t], log=quiet)[t]

        p_bad = perturb(panel, t + 1, medals=10.0, host_shift=True)
        p_bad = perturb(p_bad, t, gdp=3.0, pop=2.0)
        bad_samples = build_samples(p_bad, node_index, [t], log=quiet)[t]

        p_ctl = perturb(panel, t - 8, medals=10.0)
        ctl_samples = build_samples(p_ctl, node_index, [t], log=quiet)[t]

        def d(a, b):
            return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())

        checks = {
            "target": int(t),
            "tier1_graph_max_abs_delta": d(base_samples["edge_index"], bad_samples["edge_index"]),
            "tier1_graph_weight_max_abs_delta": d(base_samples["edge_weight"], bad_samples["edge_weight"]),
            "tier1_steps_max_abs_delta": d(base_samples["steps"], bad_samples["steps"]),
            "tier1_static_max_abs_delta": d(base_samples["static"], bad_samples["static"]),
            "tier1_labels_max_abs_delta": d(base_samples["y"], bad_samples["y"]),
            "tier1_mask_delta": int((base_samples["mask"] != bad_samples["mask"]).sum()),
        }
        checks["tier1_pass"] = bool(
            checks["tier1_graph_max_abs_delta"] == 0.0
            and checks["tier1_graph_weight_max_abs_delta"] == 0.0
            and checks["tier1_steps_max_abs_delta"] == 0.0
            and checks["tier1_static_max_abs_delta"] == 0.0
            and checks["tier1_labels_max_abs_delta"] == 0.0
            and checks["tier1_mask_delta"] == 0)
        checks["tier1_control_steps_max_abs_delta"] = d(base_samples["steps"], ctl_samples["steps"])
        checks["tier1_control_moves"] = bool(checks["tier1_control_steps_max_abs_delta"] > 0)

        log(f"[leak] {t}: tier 2 end-to-end on {device}")
        set_seed()
        base_pred, meta, _ = run_forecast(panel, t, device, max_epochs=max_epochs, log=log)
        set_seed()
        again, meta_again, _ = run_forecast(panel, t, device, max_epochs=max_epochs, log=quiet)
        noise = float(np.abs(again - base_pred).max())
        checks["tier2_noise_floor_max_abs_delta"] = noise
        checks["epochs"] = {"base": meta["best_epoch"], "rerun": meta_again["best_epoch"]}

        set_seed()
        predA, metaA, _ = run_forecast(p_bad, t, device, max_epochs=max_epochs, log=quiet)
        checks["tier2_future_perturbed_max_abs_delta"] = float(np.abs(predA - base_pred).max())
        checks["epochs"]["future_perturbed"] = metaA["best_epoch"]

        set_seed()
        predC, _, _ = run_forecast(p_ctl, t, device, max_epochs=max_epochs, log=quiet)
        checks["tier2_control_max_abs_delta"] = float(np.abs(predC - base_pred).max())

        tol = max(3.0 * noise, 1e-3)
        checks["tier2_tolerance"] = tol
        checks["tier2_future_over_control_ratio"] = (
            checks["tier2_future_perturbed_max_abs_delta"]
            / max(checks["tier2_control_max_abs_delta"], 1e-12))
        checks["tier2_pass"] = bool(
            (checks["tier2_future_perturbed_max_abs_delta"] <= tol
             or checks["tier2_future_over_control_ratio"] <= 1e-3)
            and checks["tier2_control_max_abs_delta"] > 10 * tol)
        checks["pass"] = bool(checks["tier1_pass"] and checks["tier2_pass"])
        checks["epochs"] = meta["best_epoch"]
        results.append(checks)
        log(f"[leak] {t}: {json.dumps(checks)}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="run on the synthetic dev panel")
    ap.add_argument("--check-panel", action="store_true", help="contract assertions only")
    ap.add_argument("--panel", default=PANEL_PATH)
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--leak-selftest", nargs="*", metavar="YEAR",
                    help="perturbation test: forecasts at these years must not move "
                         "when editions >= t are perturbed")
    args = ap.parse_args()

    set_seed()
    device = torch.device(args.device)
    log = print

    if args.synthetic:
        panel = make_synthetic_panel()
        panel_path = SYNTH_PATH
        os.makedirs(GEN_DIR, exist_ok=True)
        panel.to_csv(SYNTH_PATH, index=False)
        log = print
        log(f"[dev] synthetic panel written -> {SYNTH_PATH}")
    else:
        panel_path = args.panel
        assert os.path.exists(panel_path), (
            f"panel not found: {panel_path}. The clean-panel task has not landed yet; "
            f"run with --synthetic for the dev pipeline, or re-run once gen/panel_clean.csv exists.")
        panel = pd.read_csv(panel_path)

    panel = normalise_panel(panel, log=log)
    info = assert_panel_contract(panel)
    log(f"[panel] {panel_path}")
    for k, v in info.items():
        log(f"  {k}: {v}")
    if args.check_panel:
        log("[ok] contract assertions passed")
        return

    node_list = sorted(panel["ioc"].unique())
    node_index = {c: i for i, c in enumerate(node_list)}
    n = len(node_list)
    editions = info["years"]
    targets = info["targets"]
    if args.max_targets:
        targets = targets[:args.max_targets]
    log(f"[graph] {n} nodes, {len(editions)} editions, {len(targets)} targets "
        f"({targets[0]}..{targets[-1]})")

    t_build = time.time()
    samples = build_samples(panel, node_index, editions, log=log)
    log(f"[samples] built {len(samples)} edition-samples in {time.time() - t_build:.1f}s "
        f"(avg edges/sample {np.mean([s['n_edges'] for s in samples.values()]):.0f})")

    if args.leak_selftest is not None:
        st_targets = [int(y) for y in args.leak_selftest] or [targets[0], targets[len(targets) // 2]]
        st = leak_selftest(panel, st_targets, device, max_epochs=args.max_epochs, log=log)
        runlog_stub = {"seed": SEED, "device": str(device), "panel": panel_path,
                       "leak_selftest": st}
        with open(RUNLOG_PATH, "w") as fh:
            json.dump(runlog_stub, fh, indent=2)
        log(f"[leak] all pass: {all(r['pass'] for r in st)} -> {RUNLOG_PATH}")
        return

    country_of = panel.drop_duplicates("ioc").set_index("ioc")["country"].to_dict()
    rows, runlog = [], {"seed": SEED, "device": str(device), "panel": panel_path,
                        "n_nodes": n, "targets": []}
    total_t0 = time.time()
    for t_i, year in enumerate(targets):
        pred, meta, _ = run_forecast(panel, year, device, samples=samples,
                                     node_index=node_index, editions=editions,
                                     max_epochs=args.max_epochs, log=log)
        tgt = panel[panel["year"] == year]
        n_pred = 0
        for ioc, actual in zip(tgt["ioc"], tgt["total"]):
            j = node_index.get(ioc)
            if j is None:
                continue
            rows.append({
                "country": country_of.get(ioc, ioc),
                "ioc": ioc,
                "year": int(year),
                "actual": int(actual),
                "predicted": float(max(pred[j], 0.0)),
            })
            n_pred += 1
        rec = {"year": int(year), "n_predicted": n_pred, **meta}
        runlog["targets"].append(rec)
        log(f"[{t_i + 1}/{len(targets)}] {year}: {n_pred} countries, "
            f"epoch {meta['best_epoch']}/{meta['epochs_run']}, "
            f"val_logMSE {meta['best_val_log_mse']:.4f}, {meta['seconds']:.1f}s")

    os.makedirs(GEN_DIR, exist_ok=True)
    out = pd.DataFrame(rows).sort_values(["year", "actual"], ascending=[True, False])
    out.to_csv(PRED_PATH, index=False)
    runlog["total_seconds"] = time.time() - total_t0
    runlog["n_prediction_rows"] = len(out)
    with open(RUNLOG_PATH, "w") as fh:
        json.dump(runlog, fh, indent=2)
    log(f"[done] {len(out)} prediction rows -> {PRED_PATH}")
    log(f"[done] run log -> {RUNLOG_PATH} (total {runlog['total_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
