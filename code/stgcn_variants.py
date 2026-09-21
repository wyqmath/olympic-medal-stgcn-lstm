from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import stgcn_predict as sp
from backtest_period import (DEFAULT_PERIODS, _period_statistic, add_persistence,
                             build_windows, period_games)

GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen")
VAR_DIR = os.path.join(GEN, "variants")
SEED = sp.SEED

AUG_NAMES = [
    "host_x_log1p_last_total",
    "host_x_log1p_dynweight",
    "host_x_z_log_gdp",
    "host_x_log1p_cumulative",
    "host_experience_count",
    "years_since_hosting",
]
N_AUG = len(AUG_NAMES)
HOST_EXP_CAP = 3.0
YEARS_SINCE_CAP = 16.0


def augmented_statics(panel: pd.DataFrame, node_index: dict, samples: dict,
                      editions: list, log=print) -> dict:
    hosts_by_year = (panel[panel["host_flag"] == 1]
                     .drop_duplicates("year").set_index("year")["ioc"].to_dict())
    out = {}
    for year in editions:
        s = samples[year]["static"]
        host_col = s[:, 0]
        extra = np.zeros((s.shape[0], N_AUG), dtype=np.float32)
        extra[:, 0] = host_col * s[:, sp.IDX_LOG1P_LAST_TOTAL]
        extra[:, 1] = host_col * s[:, 7]
        extra[:, 2] = host_col * s[:, 1]
        extra[:, 3] = host_col * s[:, 10]
        aug = np.concatenate([s, extra], axis=1).astype(np.float32)
        for ioc, j in node_index.items():
            hosted = [int(y) for y, h in hosts_by_year.items() if h == ioc and y < year]
            if hosted:
                aug[j, 14 + 4] = min(len(hosted), HOST_EXP_CAP) / HOST_EXP_CAP
                aug[j, 14 + 5] = min(year - max(hosted), YEARS_SINCE_CAP) / YEARS_SINCE_CAP
            else:
                aug[j, 14 + 5] = 1.0
        out[year] = aug
    log(f"[V1] augmented statics: {N_AUG} extra host channels -> "
        f"{samples[editions[0]]['static'].shape[1]} + {N_AUG} columns")
    return out


def with_static(samples: dict, statics: dict) -> dict:
    return {y: {**s, "static": statics[y]} for y, s in samples.items()}


def make_loss(kind: str):
    kind = kind.lower()
    if kind in ("mse", "logmse"):
        return lambda p, y, m: F.mse_loss(p.reshape(-1)[m], y[m])
    if kind == "huber":
        return lambda p, y, m: F.huber_loss(p.reshape(-1)[m], y[m], delta=1.0)
    if kind in ("l1", "mae"):
        return lambda p, y, m: F.l1_loss(p.reshape(-1)[m], y[m])
    if kind.startswith("pinball"):
        q = float(kind.split("q")[-1])

        def _pin(p, y, m):
            e = y[m] - p.reshape(-1)[m]
            return torch.mean(torch.maximum(q * e, (q - 1.0) * e))
        return _pin
    raise ValueError(f"unknown loss {kind}")


def theta_candidates(grid):
    return [round(float(g), 4) for g in grid]


def select_mix(model, samples, val_editions, device, hgbr=None,
               grid=(0.0, 0.25, 0.5, 0.75, 1.0), tol_pp=0.0, log=print):
    if not val_editions:
        return (0.5, 0.0), None
    ys, ps, as_ = [], [], []
    hs = [] if hgbr is not None else None
    for e in val_editions:
        pred = sp.predict_target(model, samples, e, device)
        m = samples[e]["mask"]
        ys.append(np.expm1(samples[e]["y"][m]))
        ps.append(pred[m])
        as_.append(np.expm1(samples[e]["static"][m, sp.IDX_LOG1P_LAST_TOTAL]))
        if hgbr is not None:
            hs.append(hgbr[e][m])
    ys, ps, as_ = np.concatenate(ys), np.concatenate(ps), np.concatenate(as_)
    if hs is not None:
        hs = np.concatenate(hs)

    scores = {}
    if hs is None:
        for th in theta_candidates(grid):
            scores[(round(th, 4), 0.0)] = sp.weighted_smape(ys, th * ps + (1 - th) * as_)
    else:
        for a in theta_candidates(grid):
            for b in theta_candidates(grid):
                if a + b > 1.0 + 1e-9:
                    continue
                scores[(a, b)] = sp.weighted_smape(ys, a * ps + b * hs + (1 - a - b) * as_)
    best = min(scores.values())
    pool = [k for k, v in scores.items() if v <= best + tol_pp]
    pick = min(pool, key=lambda k: (round(k[0] + k[1], 6), k[0], k[1]))
    return pick, scores


HGBR_PARAMS = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
                   min_samples_leaf=10, l2_regularization=1.0,
                   early_stopping=False, random_state=SEED)
HGBR_LAST_STEP_COLS = (5, 6, 13)


def hgbr_design(samples: dict, editions: list) -> dict:
    F_ = sp.N_STATIC_FEAT + len(HGBR_LAST_STEP_COLS) + 1
    pos = {y: i / max(len(editions) - 1, 1) for i, y in enumerate(editions)}
    out = {}
    for y in editions:
        s = samples[y]["static"]
        last = samples[y]["steps"][-1][:, list(HGBR_LAST_STEP_COLS)]
        out[y] = np.concatenate([s, last,
                                 np.full((s.shape[0], 1), pos[y], dtype=np.float32)],
                                axis=1).astype(np.float32)
    assert out[editions[0]].shape[1] == F_
    return out


def hgbr_member(panel: pd.DataFrame, node_index: dict, samples: dict, editions: list,
                log=print) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = hgbr_design(samples, editions)
    preds = {}
    for t in editions:
        if t < min(sp.MIN_TRAIN_EDITION, 1950):
            preds[t] = np.zeros(len(node_index), dtype=float)
            continue
        train_ed, val_ed = sp.split_train_val([e for e in editions if e < t])
        rows_x, rows_y = [], []
        for tau in train_ed:
            m = samples[tau]["mask"]
            if m.any():
                rows_x.append(X[tau][m])
                rows_y.append(samples[tau]["y"][m])
        if not rows_x:
            preds[t] = np.zeros(len(node_index), dtype=float)
            continue
        Xtr = np.concatenate(rows_x)
        ytr = np.concatenate(rows_y)
        reg = HistGradientBoostingRegressor(**HGBR_PARAMS).fit(Xtr, ytr)
        p = np.expm1(np.clip(reg.predict(X[t]), 0.0, None))
        p[~np.isfinite(p)] = 0.0
        preds[t] = p
        log(f"[V4] hgbr fit {t}: n_train={len(ytr)} (val {[v for v in val_ed]})")
    return preds


HOST_RATIO_FLOOR_MEDALS = 5.0
HOST_RATIO_SINCE = 1948
HOST_RATIO_CLIP = (1.0, 2.5)


def host_multiplier(panel: pd.DataFrame, target_year: int) -> float:
    p = panel.sort_values(["ioc", "year"])
    prev = p.groupby("ioc")["total"].shift(1)
    fr = pd.DataFrame({"ioc": p["ioc"].to_numpy(), "year": p["year"].to_numpy(),
                       "total": p["total"].to_numpy(), "host": p["host_flag"].to_numpy(),
                       "prev": prev.to_numpy()})
    h = fr[(fr["host"] == 1) & (fr["year"] < target_year)
           & (fr["year"] >= HOST_RATIO_SINCE) & (fr["prev"] >= HOST_RATIO_FLOOR_MEDALS)]
    if not len(h):
        return 1.0
    r = float(np.median(h["total"].to_numpy() / h["prev"].to_numpy()))
    return float(np.clip(r, *HOST_RATIO_CLIP))


def apply_host_multiplier(pred: np.ndarray, panel: pd.DataFrame, target_year: int,
                          node_index: dict, log=print):
    r = host_multiplier(panel, target_year)
    hosts = set(panel.loc[(panel["host_flag"] == 1) & (panel["year"] == target_year), "ioc"])
    out = pred.copy()
    for ioc in hosts:
        j = node_index.get(ioc)
        if j is not None:
            out[j] = out[j] * r
    return out, r


def train_epoch_v(model, samples, editions, opt, device, loss_fn):
    model.train()
    total, nb = 0.0, 0
    order = list(editions)
    for i in range(0, len(order), sp.BATCH_SIZE):
        batch = [samples[e] for e in order[i:i + sp.BATCH_SIZE]]
        x, ei, ew, static, y, mask, B, T, N = sp.collate(batch)
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
def eval_epoch_v(model, samples, editions, device, loss_fn):
    model.eval()
    total, nb = 0.0, 0
    for i in range(0, len(editions), sp.BATCH_SIZE):
        batch = [samples[e] for e in editions[i:i + sp.BATCH_SIZE]]
        x, ei, ew, static, y, mask, B, T, N = sp.collate(batch)
        x, ei, ew = x.to(device), ei.to(device), ew.to(device)
        static, y, mask = static.to(device), y.to(device), mask.to(device)
        loss = loss_fn(model(x, ei, ew, static, B, T, N), y, mask)
        total += float(loss.item()) * B
        nb += B
    return total / max(nb, 1)


def train_target_v(samples, target_year, editions_before, device, static_dim,
                   loss_fn, max_epochs=None, log=print):
    max_epochs = max_epochs or sp.MAX_EPOCHS
    train_ed, val_ed = sp.split_train_val(editions_before)
    model = sp.STGCN_LSTM(static_dim=static_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sp.LR, weight_decay=sp.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sp.lr_lambda)
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = train_epoch_v(model, samples, train_ed, opt, device, loss_fn)
        sched.step()
        if val_ed:
            va = eval_epoch_v(model, samples, val_ed, device, loss_fn)
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
    meta = {"epochs_run": epoch, "best_epoch": best_epoch, "val_editions": list(val_ed),
            "best_val_loss": best_val, "seconds": time.time() - t0,
            "n_train_samples": len(train_ed), "n_val_samples": len(val_ed)}
    return model, meta


VARIANTS = {
    "base": dict(loss="mse",    aug=False, theta="coarse", host=False),
    "V1":   dict(loss="mse",    aug=True,  theta="coarse", host=False),
    "V2":   dict(loss="huber",  aug=False, theta="coarse", host=False),
    "V3":   dict(loss="mse",    aug=False, theta="fine",   host=False),
    "V4":   dict(loss="mse",    aug=False, theta="coarse", host=False, hgbr=True),
    "V5":   dict(loss="mse",    aug=False, theta="coarse", host=True),
    "V6":   dict(loss="huber",  aug=True,  theta="fine",   host=True),
}
THETA_GRIDS = {
    "coarse": (0.0, 0.25, 0.5, 0.75, 1.0),
    "fine": tuple(np.round(np.arange(0.0, 1.0001, 0.05), 4)),
}
THETA_TOL_PP = {"coarse": 0.0, "fine": 0.25}


def run_variant(tag: str, panel: pd.DataFrame, device, samples: dict,
                node_index: dict, editions: list, targets: list,
                statics: dict | None, hgbr: dict | None, max_epochs=None,
                log=print, country_of=None):
    cfg = VARIANTS[tag]
    loss_fn = make_loss(cfg["loss"])
    static_dim = sp.N_STATIC_FEAT + (N_AUG if cfg["aug"] else 0)
    used = with_static(samples, statics) if cfg["aug"] else samples
    grid = THETA_GRIDS[cfg["theta"]]
    tol = THETA_TOL_PP[cfg["theta"]]
    hgbr_use = hgbr if cfg.get("hgbr") else None
    if hgbr_use is not None and cfg["aug"]:
        hgbr_use = {y: v for y, v in hgbr.items()}

    rows, metas = [], {}
    for t_i, year in enumerate(targets):
        editions_before = [e for e in editions if e < year]
        model, meta = train_target_v(used, year, editions_before, device, static_dim,
                                     loss_fn, max_epochs=max_epochs, log=log)
        raw = sp.predict_target(model, used, year, device)
        if hgbr_use is not None:
            (a, b), scores = select_mix(model, used, meta["val_editions"], device,
                                        hgbr=hgbr_use, grid=grid, tol_pp=tol, log=log)
            anchor = np.expm1(used[year]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
            pred = a * raw + b * hgbr_use[year] + (1.0 - a - b) * anchor
            meta["blend_a"], meta["blend_b"] = a, b
            meta["val_blend_scores"] = None if scores is None else {
                f"{k[0]}+{k[1]}": round(v, 4) for k, v in scores.items()}
        else:
            (th, _b), scores = select_mix(model, used, meta["val_editions"], device,
                                          grid=grid, tol_pp=tol, log=log)
            anchor = np.expm1(used[year]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
            pred = th * raw + (1.0 - th) * anchor
            meta["blend_a"], meta["blend_b"] = th, 0.0
            meta["val_blend_scores"] = None if scores is None else {
                f"{k[0]}+{k[1]}": round(v, 4) for k, v in scores.items()}
        if cfg["host"]:
            pred, r = apply_host_multiplier(pred, panel, year, node_index, log=log)
            meta["host_multiplier"] = r
        del model
        metas[year] = meta
        tgt = panel[panel["year"] == year]
        for ioc, actual in zip(tgt["ioc"], tgt["total"]):
            j = node_index.get(ioc)
            if j is None:
                continue
            rows.append({
                "country": (country_of or {}).get(ioc, ioc),
                "ioc": ioc, "year": int(year), "actual": int(actual),
                "predicted": float(max(pred[j], 0.0)),
            })
        log(f"[{tag}] [{t_i + 1}/{len(targets)}] {year}: epoch {meta['best_epoch']}/"
            f"{meta['epochs_run']} blend a={meta['blend_a']} b={meta['blend_b']} "
            f"r={meta.get('host_multiplier')} {meta['seconds']:.1f}s")
    return rows, metas


def evaluate_predictions(df: pd.DataFrame, periods=DEFAULT_PERIODS) -> dict:
    d = add_persistence(df[["country", "year", "actual", "predicted"]].copy())
    eds = sorted(d["year"].unique().tolist())
    win = build_windows(eds, d["year"].to_numpy())
    act = d["actual"].to_numpy(float)
    pp, miss = d["predicted_persist"].to_numpy(float), d["persist_missing"].to_numpy()
    err_p = np.where(miss, np.nan,
                     200.0 * np.abs(act - pp) / np.maximum(np.abs(act) + np.abs(pp), 1e-9))
    out = {}
    for name, (lo, hi) in periods.items():
        tl = period_games(eds, lo, hi)
        per_pw = []
        for t in tl:
            m = win[t]
            mm = m & np.isfinite(err_p) & (act > 0)
            if act[mm].sum() > 0:
                per_pw.append(float((act[mm] * err_p[mm]).sum() / act[mm].sum()))
        out[name] = dict(
            games=tl,
            macro=_period_statistic(d, win, tl, "macro"),
            weighted=_period_statistic(d, win, tl, "weighted"),
            persist_macro=_period_statistic(d, win, tl, "persistence"),
            persist_weighted=float(np.mean(per_pw)) if per_pw else float("nan"),
        )
    out["_mean_weighted"] = float(np.nanmean([out[k]["weighted"] for k in periods]))
    out["_mean_macro"] = float(np.nanmean([out[k]["macro"] for k in periods]))
    return out


def deficits(ev: dict, periods=DEFAULT_PERIODS) -> dict:
    out = {}
    for name in periods:
        for m, pk in (("macro", "persist_macro"), ("weighted", "persist_weighted")):
            out[f"{name}_{m}"] = float(ev[name][m] - ev[name][pk])
    return out


def adoption_check(ev: dict, base_ev: dict, periods=DEFAULT_PERIODS,
                   margin_pp: float = 1.0) -> dict:
    m = float(ev["_mean_weighted"])
    mb = float(base_ev["_mean_weighted"])
    d_v, d_b = deficits(ev, periods), deficits(base_ev, periods)
    guard, worst = [], -99.0
    for k in d_v:
        slack = float(d_v[k] - max(d_b[k], margin_pp))
        worst = max(worst, slack)
        guard.append(dict(metric=k, variant_deficit=round(d_v[k], 3),
                          baseline_deficit=round(d_b[k], 3),
                          allowed=max(d_b[k], margin_pp), slack=round(slack, 3)))
    return dict(mean_weighted=round(m, 4), baseline_mean_weighted=round(mb, 4),
                improvement_pp=round(mb - m, 4), passes_primary=bool(m <= mb - margin_pp),
                guard=guard, worst_guard_slack=round(worst, 4),
                passes=bool(m <= mb - margin_pp and worst <= 1e-9))


def variant_table(directory: str, baseline_csv: str,
                  periods=DEFAULT_PERIODS, baseline_label: str = "base") -> pd.DataFrame:
    rows = []
    base_ev = evaluate_predictions(pd.read_csv(baseline_csv))
    rows.append(_table_row(baseline_label, baseline_csv, base_ev, base_ev, periods))
    for fn in sorted(os.listdir(directory)):
        if not fn.startswith("predictions_") or not fn.endswith(".csv") \
                or fn == os.path.basename(baseline_csv):
            continue
        path = os.path.join(directory, fn)
        try:
            ev = evaluate_predictions(pd.read_csv(path))
        except Exception as exc:
            rows.append(dict(variant=fn, error=str(exc)))
            continue
        rows.append(_table_row(os.path.splitext(fn)[0].replace("predictions_", ""),
                               path, ev, base_ev, periods))
    return pd.DataFrame(rows)


def _table_row(tag: str, path: str, ev: dict, base_ev: dict,
               periods=DEFAULT_PERIODS) -> dict:
    d = deficits(ev, periods)
    row = dict(variant=tag, source=os.path.basename(path))
    for name in periods:
        row[f"{name}_macro"] = round(ev[name]["macro"], 2)
        row[f"{name}_wght"] = round(ev[name]["weighted"], 2)
        row[f"{name}_per_macro"] = round(ev[name]["persist_macro"], 2)
        row[f"{name}_per_wght"] = round(ev[name]["persist_weighted"], 2)
    row["MEAN_macro"] = round(ev["_mean_macro"], 3)
    row["MEAN_weighted"] = round(ev["_mean_weighted"], 3)
    chk = adoption_check(ev, base_ev, periods)
    row["d_MEAN_wght_vs_base"] = round(-chk["improvement_pp"], 3)
    row["primary_ok"] = chk["passes_primary"]
    row["guard_ok"] = bool(chk["worst_guard_slack"] <= 1e-9)
    row["ADOPT"] = chk["passes"]
    return row


def tier1_selftest(panel: pd.DataFrame, targets: list, log=print) -> list:
    quiet = lambda *a, **k: None
    res = []
    for t in targets:
        node_index = {c: i for i, c in enumerate(sorted(panel["ioc"].unique()))}
        base = sp.build_samples(panel, node_index, [t], log=quiet)[t]
        base_aug = augmented_statics(panel, node_index, {t: base}, [t], log=quiet)[t]
        r_base = host_multiplier(panel, int(t))

        p_bad = sp.perturb(panel, t + 1, medals=10.0, host_shift=True)
        p_bad = sp.perturb(p_bad, t, gdp=3.0, pop=2.0)
        bad = sp.build_samples(p_bad, node_index, [t], log=quiet)[t]
        bad_aug = augmented_statics(p_bad, node_index, {t: bad}, [t], log=quiet)[t]
        r_bad = host_multiplier(p_bad, int(t))

        p_ctl = sp.perturb(panel, t - 8, medals=10.0)
        ctl = sp.build_samples(p_ctl, node_index, [t], log=quiet)[t]

        def d(a, b):
            return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())

        chk = {
            "target": int(t),
            "graph_delta": d(base["edge_index"], bad["edge_index"]),
            "graph_weight_delta": d(base["edge_weight"], bad["edge_weight"]),
            "steps_delta": d(base["steps"], bad["steps"]),
            "static_delta": d(base["static"], bad["static"]),
            "aug_static_delta": d(base_aug[:, 14:], bad_aug[:, 14:]),
            "labels_delta": d(base["y"], bad["y"]),
            "mask_delta": int((base["mask"] != bad["mask"]).sum()),
            "host_mult_base": round(r_base, 6), "host_mult_perturbed": round(r_bad, 6),
            "control_steps_delta": d(base["steps"], ctl["steps"]),
        }
        chk["pass"] = bool(
            max(chk["graph_delta"], chk["graph_weight_delta"], chk["steps_delta"],
                chk["static_delta"], chk["aug_static_delta"], chk["labels_delta"]) == 0.0
            and chk["mask_delta"] == 0
            and chk["host_mult_base"] == chk["host_mult_perturbed"]
            and chk["control_steps_delta"] > 0)
        res.append(chk)
        log(f"[tier1] {t}: pass={chk['pass']} {json.dumps({k: v for k, v in chk.items() if k != 'pass'})}")
    return res


def ranking_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = add_persistence(df[["country", "year", "actual", "predicted"]].copy())
    rows = []
    for y, g in d.groupby("year"):
        n = len(g)
        act_rank = g["actual"].rank(ascending=False, method="first")
        mod_rank = g["predicted"].rank(ascending=False, method="first")
        per_rank = g["predicted_persist"].rank(ascending=False, method="first")
        act_v = g["actual"].to_numpy(float)
        rho_mod = _spearman(act_v, g["predicted"].to_numpy(float))
        rho_per = _spearman(act_v, g["predicted_persist"].to_numpy(float))
        rec = dict(year=int(y), n_countries=n,
                   n_persist_missing=int((~np.isfinite(g["predicted_persist"].to_numpy(float))).sum()))
        per_ok = rec["n_persist_missing"] < n
        for k in (10, 3):
            kk = min(k, n)
            top_act = set(g.index[act_rank <= kk])
            rec[f"top{kk}_hit_model"] = float(len(top_act & set(g.index[mod_rank <= kk])) / kk)
            rec[f"top{kk}_hit_persist"] = (
                float(len(top_act & set(g.index[per_rank <= kk])) / kk) if per_ok else float("nan"))
        rec["spearman_model"] = rho_mod
        rec["spearman_persist"] = rho_per
        rows.append(rec)
    return pd.DataFrame(rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def add_period_rows(rank: pd.DataFrame, periods=DEFAULT_PERIODS) -> pd.DataFrame:
    rank = rank.copy()
    rank["period"] = [next((n for n, (lo, hi) in periods.items() if lo <= int(y) <= hi), "")
                      for y in rank["year"]]
    blocks = [rank]
    for name in periods:
        sel = rank["period"] == name
        if not sel.any():
            continue
        agg = {"year": f"{name} (mean)", "n_countries": int(rank.loc[sel, "n_countries"].sum()),
               "period": name}
        for c in rank.columns:
            if c not in ("year", "n_countries", "period"):
                agg[c] = float(rank.loc[sel, c].mean())
        blocks.append(pd.DataFrame([agg]))
    out = pd.concat(blocks, ignore_index=True)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    return out


def persistence_weighted_table(pred_csv: str, n_boot: int = 2000, seed: int = 42,
                               periods=DEFAULT_PERIODS) -> pd.DataFrame:
    d = add_persistence(pd.read_csv(pred_csv)[["country", "year", "actual", "predicted"]])
    eds = sorted(d["year"].unique().tolist())
    win = build_windows(eds, d["year"].to_numpy())
    act = d["actual"].to_numpy(float)
    pp = d["predicted_persist"].to_numpy(float)
    miss = d["persist_missing"].to_numpy()
    err_p = np.where(miss, np.nan,
                     200.0 * np.abs(act - pp) / np.maximum(np.abs(act) + np.abs(pp), 1e-9))
    countries = d["country"].unique()
    rows_by_country = [np.flatnonzero(d["country"].to_numpy() == c) for c in countries]

    def stat(t_list, row_idx=None):
        e, a = (err_p, act) if row_idx is None else (err_p[row_idx], act[row_idx])
        vals = []
        for t in t_list:
            m = win[t] if row_idx is None else win[t][row_idx]
            mm = m & np.isfinite(e) & (a > 0)
            if a[mm].sum() > 0:
                vals.append(float((a[mm] * e[mm]).sum() / a[mm].sum()))
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for p_i, (name, (lo, hi)) in enumerate(periods.items()):
        tl = period_games(eds, lo, hi)
        point = stat(tl)
        rng = np.random.default_rng([seed, p_i])
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(rows_by_country), len(rows_by_country))
            idx = np.concatenate([rows_by_country[i] for i in pick])
            v = stat(tl, idx)
            if np.isfinite(v):
                draws.append(v)
        rows.append(dict(
            period=name, start_year=lo, end_year=hi, games_years="|".join(map(str, tl)),
            persistence_weighted=round(point, 6),
            persistence_weighted_ci95_low=round(float(np.percentile(draws, 2.5)), 6),
            persistence_weighted_ci95_high=round(float(np.percentile(draws, 97.5)), 6),
            n_bootstrap_replicates=len(draws),
        ))
    return pd.DataFrame(rows)


def final_leak_check(panel: pd.DataFrame, targets: list, device, tag: str = "V5",
                     seed: int = SEED, log=print) -> list:
    quiet = lambda *a, **k: None
    res = []
    for t in targets:
        def forecast(df):
            node_index = {c: i for i, c in enumerate(sorted(df["ioc"].unique()))}
            editions = sorted(int(y) for y in df["year"].unique())
            samples = sp.build_samples(df, node_index, editions, log=quiet)
            statics = augmented_statics(df, node_index, samples, editions, log=quiet)
            sp.set_seed(seed)
            rows, _ = run_variant(tag, df, device, samples, node_index, editions, [int(t)],
                                  statics, None, log=quiet)
            return {r["ioc"]: r["predicted"] for r in rows}

        clean = forecast(panel)
        p_bad = sp.perturb(panel, int(t) + 1, medals=10.0, host_shift=True)
        p_bad = sp.perturb(p_bad, int(t), gdp=3.0, pop=2.0)
        bad = forecast(p_bad)
        p_ctl = sp.perturb(panel, int(t) - 8, medals=10.0)
        ctl = forecast(p_ctl)
        common = sorted(set(clean) & set(bad))
        delta = max(abs(clean[k] - bad[k]) for k in common)
        delta_ctl = max(abs(clean[k] - ctl[k]) for k in sorted(set(clean) & set(ctl)))
        res.append(dict(target=int(t), variant=tag, n_common=len(common),
                        future_perturbed_max_abs_delta=float(delta),
                        control_max_abs_delta=float(delta_ctl),
                        pass_=bool(delta == 0.0 and delta_ctl > 0)))
        log(f"[final-leak] {t}: future-perturbed max |delta| = {delta:.3g} "
            f"(control {delta_ctl:.3g}) -> {'PASS' if delta == 0.0 else 'FAIL'}")
    return res


def load_panel(panel_path: str):
    panel = sp.normalise_panel(pd.read_csv(panel_path), log=lambda *a, **k: None)
    info = sp.assert_panel_contract(panel)
    return panel, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="base,V1,V2,V3,V4,V5,V6")
    ap.add_argument("--panel",
                    default=os.path.join(HERE, os.pardir, "data", "panel_clean.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = default)")
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=0)
    ap.add_argument("--outdir", default=VAR_DIR)
    ap.add_argument("--out", default="", help="write this variant's predictions to this CSV")
    ap.add_argument("--table", action="store_true",
                    help="score every predictions CSV in --dir and print the variant table")
    ap.add_argument("--dir", default=VAR_DIR)
    ap.add_argument("--tier1-selftest", nargs="*", metavar="YEAR")
    ap.add_argument("--final-leak-check", nargs="*", metavar="YEAR",
                    help="end-to-end future-perturbation check of the shipped variant")
    ap.add_argument("--leak-tag", default="V5", help="variant used by --final-leak-check")
    ap.add_argument("--rank-metrics", metavar="PRED_CSV",
                    help="write per-Games Spearman/top-k hit rates (model vs persistence) "
                         "for this prediction CSV to --rank-out")
    ap.add_argument("--rank-out", default=os.path.join(GEN, "ranking_metrics_final.csv"))
    ap.add_argument("--persist-weighted-csv", dest="persist_weighted_csv", default="",
                    help="write the persistence *weighted* per-period table (with cluster "
                         "bootstrap CIs) for this prediction CSV to this path")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    log = print

    if args.persist_weighted_csv:
        src = args.rank_metrics or os.path.join(GEN, "predictions_stgcn_final.csv")
        tab = persistence_weighted_table(src, n_boot=2000, seed=args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.persist_weighted_csv)), exist_ok=True)
        tab.to_csv(args.persist_weighted_csv, index=False)
        log(f"[persist-w] {src} -> {args.persist_weighted_csv}")
        print(tab.to_string(index=False))
        return

    if args.rank_metrics:
        tab = add_period_rows(ranking_metrics(pd.read_csv(args.rank_metrics)))
        os.makedirs(os.path.dirname(os.path.abspath(args.rank_out)), exist_ok=True)
        tab.to_csv(args.rank_out, index=False)
        log(f"[rank] {args.rank_metrics} -> {args.rank_out}")
        print(tab.to_string(index=False))
        return

    if args.tier1_selftest is not None:
        panel, _ = load_panel(args.panel)
        targets = [int(y) for y in args.tier1_selftest] or [1976, 2000, 2024]
        res = tier1_selftest(panel, targets, log=log)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "tier1_selftest.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        log(f"[tier1] all pass: {all(r['pass'] for r in res)}")
        return

    if args.final_leak_check is not None:
        panel, _ = load_panel(args.panel)
        targets = [int(y) for y in args.final_leak_check] or [1976, 2000]
        res = final_leak_check(panel, targets, device, tag=args.leak_tag, seed=args.seed, log=log)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "final_leak_check.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        log(f"[final-leak] all pass: {all(r['pass_'] for r in res)}")
        return

    if args.table:
        print(variant_table(args.dir, os.path.join(GEN, "predictions_stgcn.csv"),
                            baseline_label="base(incumbent)").to_string(index=False))
        return

    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    unknown = [t for t in tags if t not in VARIANTS]
    assert not unknown, f"unknown variants: {unknown} (known: {list(VARIANTS)})"
    panel, info = load_panel(args.panel)
    node_list = sorted(panel["ioc"].unique())
    node_index = {c: i for i, c in enumerate(node_list)}
    editions = info["years"]
    targets = info["targets"][:args.max_targets] if args.max_targets else info["targets"]
    country_of = panel.drop_duplicates("ioc").set_index("ioc")["country"].to_dict()
    log(f"[run] variants={tags} device={device} targets={targets[0]}..{targets[-1]} "
        f"({len(targets)}), seed={args.seed}")

    t0 = time.time()
    samples = sp.build_samples(panel, node_index, editions, log=log)
    log(f"[run] samples built in {time.time() - t0:.1f}s")
    statics = augmented_statics(panel, node_index, samples, editions, log=log)
    hgbr = None
    if any(VARIANTS[t].get("hgbr") for t in tags):
        t0 = time.time()
        hgbr = hgbr_member(panel, node_index, samples, editions, log=log)
        log(f"[run] hgbr member built in {time.time() - t0:.1f}s")

    os.makedirs(args.outdir, exist_ok=True)
    baseline_path = os.path.join(GEN, "predictions_stgcn.csv")
    base_ev = None
    if os.path.exists(baseline_path):
        base_ev = evaluate_predictions(pd.read_csv(baseline_path))
        log(f"[base] incumbent baseline mean weighted {base_ev['_mean_weighted']:.4f} "
            f"mean macro {base_ev['_mean_macro']:.4f}  <- {baseline_path}")
    summary = {"baseline": {"predictions": baseline_path,
                            "mean_macro": round(base_ev["_mean_macro"], 4),
                            "mean_weighted": round(base_ev["_mean_weighted"], 4)}
               if base_ev is not None else None}
    for tag in tags:
        sp.set_seed(args.seed)
        rows, metas = run_variant(tag, panel, device, samples, node_index, editions,
                                  targets, statics, hgbr,
                                  max_epochs=args.max_epochs or None, log=log,
                                  country_of=country_of)
        out = pd.DataFrame(rows).sort_values(["year", "actual"],
                                             ascending=[True, False]).reset_index(drop=True)
        path = os.path.join(args.outdir, f"predictions_{tag}.csv")
        out.to_csv(path, index=False)
        ev = evaluate_predictions(out)
        chk = adoption_check(ev, base_ev) if base_ev is not None else {}
        summary[tag] = dict(
            predictions=path,
            periods={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                         for kk, vv in v.items()} for k, v in ev.items()
                     if not k.startswith("_")},
            mean_macro=round(ev["_mean_macro"], 4),
            mean_weighted=round(ev["_mean_weighted"], 4),
            adoption=chk,
            meta={str(y): {k: v for k, v in m.items() if k != "val_blend_scores"}
                  for y, m in metas.items()},
        )
        log(f"[{tag}] macro {ev['_mean_macro']:.3f} weighted {ev['_mean_weighted']:.3f} "
            f"| vs base {chk.get('improvement_pp', float('nan')):+.3f} pp, "
            f"adopt={chk.get('passes')} -> {path}")
        if args.out and tag == tags[-1]:
            out.to_csv(args.out, index=False)
            log(f"[{tag}] copy -> {args.out}")
        with open(os.path.join(args.outdir, f"run_log_{tag}.json"), "w") as fh:
            json.dump(summary[tag], fh, indent=2)
    with open(os.path.join(args.outdir, "variant_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[done] summary -> {os.path.join(args.outdir, 'variant_summary.json')}")


if __name__ == "__main__":
    main()
