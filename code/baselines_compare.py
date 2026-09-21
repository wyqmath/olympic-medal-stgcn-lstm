from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
_ORIG_CWD = os.getcwd()

import argparse
import copy
import json
import math
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import torch.nn as nn
import torch.nn.functional as F

import stgcn_predict as sp
import stgcn_variants as sv
from backtest_period import (DEFAULT_PERIODS, add_persistence,
                             smape_obs)

GEN = os.path.join(HERE, "gen")
DATA_DIR = os.path.join(HERE, os.pardir, "data")
DEFAULT_PANEL = os.path.join(DATA_DIR, "panel_clean.csv")
DEFAULT_OUTDIR = os.path.join(GEN, "baselines")
REFERENCE_CSV = os.path.join(GEN, "nested", "variants", "predictions_base.csv")
COLUMNS = ("country", "ioc", "year", "actual", "predicted")

METHOD_ORDER = ("persistence", "arima", "xgboost", "pure_lstm", "backbone", "tt_ensemble")
GPU_METHODS = ("pure_lstm", "tt_ensemble")
CPU_METHODS = ("arima", "xgboost")
ALWAYS_METHODS = ("persistence", "backbone")
DEFAULT_METHODS = ("arima", "xgboost", "pure_lstm", "tt_ensemble")

ARIMA_ORDERS = ((1, 0, 0), (0, 1, 0), (1, 1, 0))
ARIMA_MIN_HISTORY = 4
XGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.06, max_depth=6, subsample=0.9,
    colsample_bytree=0.9, reg_lambda=1.0, n_jobs=1,
    random_state=sp.SEED, tree_method="hist", objective="reg:squarederror",
)
TT_LAYERS = 2
TT_D_MODEL = sp.HIDDEN
TT_HEADS = sp.HEADS
TT_FF = 128
TT_DROPOUT = 0.1
THETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ORIG_CWD, p)


def _pkey(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum())


def load_reference(path: str, targets: list, panel: pd.DataFrame, log=print) -> pd.DataFrame:
    assert os.path.exists(path), f"reference predictions missing: {path}"
    df = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in COLUMNS if c not in df.columns]
    assert not missing, f"{path}: missing column(s) {missing}"
    df = df[list(COLUMNS)].copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"].isin(targets)].reset_index(drop=True)
    assert not df.duplicated(["ioc", "year"]).any(), f"{path}: duplicate (ioc,year)"
    assert (df["actual"] > 0).all(), f"{path}: non-winning rows (actual <= 0)"
    panel_tot = {(int(y), str(i)): float(t)
                 for y, i, t in zip(panel["year"], panel["ioc"], panel["total"])}
    bad = [ (r.ioc, r.year) for r in df.itertuples()
            if panel_tot.get((int(r.year), str(r.ioc))) != float(r.actual) ]
    assert not bad, f"{path}: {len(bad)} actual values disagree with the panel"
    log(f"[ref] {path}: {len(df)} rows over {df['year'].nunique()} editions "
        f"({min(df['year'])}..{max(df['year'])})")
    return df


def restrict(df: pd.DataFrame, common: set) -> pd.DataFrame:
    keep = [(str(i), int(y)) in common for i, y in zip(df["ioc"], df["year"])]
    return (df[keep].sort_values(["year", "ioc"]).reset_index(drop=True))


def _jsonable(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _jsonable(obj)


def write_predictions(df: pd.DataFrame, path: str, log=print) -> None:
    out = df[list(COLUMNS)].sort_values(["year", "actual"], ascending=[True, False])
    out.to_csv(path, index=False)
    log(f"[write] {path} ({len(out)} rows)")


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def r_squared(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_res = float(((actual - predicted) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def macro_smape_over_editions(years, actual, predicted) -> float:
    years = np.asarray(years, dtype=int)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    per_game = []
    for t in sorted(set(years.tolist())):
        m = years == t
        e = smape_obs(actual[m], predicted[m])
        e = e[np.isfinite(e)]
        if e.size:
            per_game.append(float(e.mean()))
    return float(np.mean(per_game)) if per_game else float("nan")


def ranking_means(df: pd.DataFrame) -> dict:
    rk = sv.ranking_metrics(df[["country", "year", "actual", "predicted"]].copy().reset_index(drop=True))
    return dict(
        top3_hit_mean=float(rk["top3_hit_model"].mean()),
        top10_hit_mean=float(rk["top10_hit_model"].mean()),
        spearman_mean=float(rk["spearman_model"].mean()),
    )


def evaluate_method(name: str, df: pd.DataFrame, periods=DEFAULT_PERIODS) -> tuple:
    actual = df["actual"].to_numpy(dtype=float)
    predicted = df["predicted"].to_numpy(dtype=float)
    years = df["year"].to_numpy(dtype=int)
    row = dict(
        method=name, n_rows=int(len(df)), n_editions=int(df["year"].nunique()),
        rmse=round(rmse(actual, predicted), 4),
        r2=round(r_squared(actual, predicted), 4),
        macro_smape=round(macro_smape_over_editions(years, actual, predicted), 4),
        weighted_smape=round(sp.weighted_smape(actual, predicted), 4),
    )
    detail_periods, per_edition = {}, []
    for pname, (lo, hi) in periods.items():
        m = (years >= lo) & (years <= hi)
        key = _pkey(pname)
        vals = dict(n_rows=int(m.sum()), n_editions=int(df.loc[m, "year"].nunique()))
        if m.any():
            vals.update(
                rmse=round(rmse(actual[m], predicted[m]), 4),
                r2=round(r_squared(actual[m], predicted[m]), 4),
                macro_smape=round(macro_smape_over_editions(years[m], actual[m], predicted[m]), 4),
                weighted_smape=round(sp.weighted_smape(actual[m], predicted[m]), 4),
            )
        else:
            vals.update(rmse=float("nan"), r2=float("nan"),
                        macro_smape=float("nan"), weighted_smape=float("nan"))
        row.update({f"{key}_{k}": v for k, v in vals.items() if k not in ("n_rows", "n_editions")})
        detail_periods[pname] = vals
    for t in sorted(set(years.tolist())):
        m = years == t
        per_edition.append(dict(
            year=int(t), n_rows=int(m.sum()),
            rmse=round(rmse(actual[m], predicted[m]), 4),
            macro_smape=round(macro_smape_over_editions(years[m], actual[m], predicted[m]), 4),
            weighted_smape=round(sp.weighted_smape(actual[m], predicted[m]), 4),
        ))
    row.update({k: round(v, 4) for k, v in ranking_means(df).items()})
    detail = dict(periods=detail_periods, per_edition=per_edition)
    return row, detail


def lowbase_mask(panel: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    q = panel.sort_values(["ioc", "year"]).copy()
    q["prev_total"] = q.groupby("ioc")["total"].shift(1)
    prev = {(str(r.ioc), int(r.year)): float(r.prev_total) if pd.notna(r.prev_total) else float("nan")
            for r in q.itertuples()}
    out = np.zeros(len(df), dtype=bool)
    for pos, r in enumerate(df.itertuples()):
        v = prev.get((str(r.ioc), int(r.year)), float("nan"))
        out[pos] = (not np.isfinite(v)) or v == 0.0
    return out


def persistence_predictions(panel: pd.DataFrame, ref: pd.DataFrame, log=print) -> pd.DataFrame:
    frame = pd.DataFrame({
        "country": panel["ioc"].to_numpy(),
        "year": panel["year"].to_numpy(dtype=int),
        "actual": panel["total"].to_numpy(dtype=float),
    })
    pers = add_persistence(frame)
    lut = {(str(r.country), int(r.year)): float(r.predicted_persist) for r in pers.itertuples()}
    rows = []
    for r in ref.itertuples():
        v = lut.get((str(r.ioc), int(r.year)), float("nan"))
        if not np.isfinite(v):
            continue
        rows.append(dict(country=r.country, ioc=r.ioc, year=int(r.year),
                         actual=int(r.actual), predicted=float(max(v, 0.0))))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    log(f"[persistence] {len(out)}/{len(ref)} reference rows covered "
        f"({len(ref) - len(out)} dropped: no earlier record)")
    return out


def arima_predictions(panel: pd.DataFrame, ref: pd.DataFrame, persist_lut: dict,
                      log=print) -> tuple:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", ValueWarning)

    series = {}
    q = panel.sort_values(["ioc", "year"])
    for ioc, g in q.groupby("ioc", sort=False):
        series[str(ioc)] = [(int(y), float(t))
                            for y, t in zip(g["year"], g["total"])]
    fallbacks = {"too_short": 0, "fit_failed": 0}
    fallback_rows, rows = [], []
    t0 = time.time()
    n_fit = 0
    for ioc, g in ref.groupby("ioc", sort=False):
        hist_all = series.get(str(ioc), [])
        for r in g.itertuples():
            t = int(r.year)
            hist = [v for (y, v) in hist_all if y < t]
            pred = float("nan")
            if len(hist) < ARIMA_MIN_HISTORY:
                fallbacks["too_short"] += 1
            else:
                y = np.asarray(hist, dtype=float)
                best_aic, best_pred = float("inf"), float("nan")
                for order in ARIMA_ORDERS:
                    trend = "c" if order[1] == 0 else "n"
                    try:
                        res = ARIMA(y, order=order, trend=trend,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
                        n_fit += 1
                        if np.isfinite(res.aic) and res.aic < best_aic:
                            best_aic, best_pred = float(res.aic), float(res.forecast(1)[0])
                    except Exception:
                        continue
                if np.isfinite(best_pred):
                    pred = max(best_pred, 0.0)
                else:
                    fallbacks["fit_failed"] += 1
            if not np.isfinite(pred):
                reason = "too_short" if len(hist) < ARIMA_MIN_HISTORY else "fit_failed"
                pred = persist_lut.get((str(ioc), t), 0.0)
                fallback_rows.append(dict(ioc=str(ioc), year=t, reason=reason,
                                          n_history=len(hist),
                                          fallback_value=round(float(pred), 4)))
            rows.append(dict(country=r.country, ioc=str(ioc), year=t,
                             actual=int(r.actual), predicted=float(pred)))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    log(f"[arima] {len(out)} rows, {n_fit} fits, fallbacks {fallbacks}, "
        f"{time.time() - t0:.1f}s")
    return out, {"counts": fallbacks, "n_fits": n_fit, "rows": fallback_rows,
                 "seconds": round(time.time() - t0, 2)}


def xgboost_predictions(samples: dict, editions: list, targets: list,
                        node_index: dict, ref: pd.DataFrame, log=print) -> pd.DataFrame:
    from xgboost import XGBRegressor

    X = sv.hgbr_design(samples, editions)
    rows = []
    t0 = time.time()
    for t_i, t in enumerate(targets):
        assert max(samples[t]["hist_years"]) < t, "leakage: sample history reaches the target"
        train_ed = [e for e in editions if e < t]
        rows_x, rows_y = [], []
        for tau in train_ed:
            m = samples[tau]["mask"]
            if m.any():
                rows_x.append(X[tau][m])
                rows_y.append(samples[tau]["y"][m])
        reg = XGBRegressor(**XGB_PARAMS)
        reg.fit(np.concatenate(rows_x), np.concatenate(rows_y))
        p = np.expm1(np.clip(reg.predict(X[t]), 0.0, None))
        p[~np.isfinite(p)] = 0.0
        tgt = ref[ref["year"] == t]
        for r in tgt.itertuples():
            j = node_index[r.ioc]
            rows.append(dict(country=r.country, ioc=r.ioc, year=int(t),
                             actual=int(r.actual), predicted=float(p[j])))
        log(f"[xgboost] [{t_i + 1}/{len(targets)}] {t}: n_train={sum(len(a) for a in rows_y)} "
            f"({time.time() - t0:.1f}s)")
    log(f"[xgboost] total {time.time() - t0:.1f}s")
    return pd.DataFrame(rows, columns=list(COLUMNS))


class PureLSTMNet(nn.Module):

    def __init__(self, in_dim: int = sp.N_STEP_FEAT, static_dim: int = sp.N_STATIC_FEAT,
                 lstm_hidden: int = sp.LSTM_HIDDEN, heads: int = sp.HEADS,
                 hidden: int = sp.HIDDEN, dropout: float = sp.DROPOUT):
        super().__init__()
        self.static_dim = static_dim
        d = 2 * (lstm_hidden // 2)
        self.lstm = nn.LSTM(in_dim, lstm_hidden // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(d, num_heads=heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.highway = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(d + static_dim, hidden), sp.Swish(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_weight, static, B: int, T: int, N: int):
        h = x.view(B, T, N, -1).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        lstm_out, _ = self.lstm(h)
        att, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.ln1(lstm_out + att)
        last = out[:, -1, :]
        pooled = out.mean(dim=1)
        hw = self.highway(torch.cat([last, pooled], dim=-1))
        g = self.gate(torch.cat([last, pooled], dim=-1))
        fused = self.ln2(g * hw + (1.0 - g) * last)
        raw = self.head(torch.cat([fused, static], dim=-1)).view(B, N)
        anchor = static[:, sp.IDX_LOG1P_LAST_TOTAL].view(B, N)
        return F.relu(anchor + raw)


def train_pure_lstm(samples: dict, target_year: int, editions_before: list, device,
                    max_epochs=None, log=print):
    max_epochs = max_epochs or sp.MAX_EPOCHS
    train_ed, val_ed = sp.split_train_val(editions_before)
    model = PureLSTMNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sp.LR, weight_decay=sp.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, sp.lr_lambda)
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = sv.train_epoch_v(model, samples, train_ed, opt, device, sp.loss_fn)
        sched.step()
        if val_ed:
            va = sv.eval_epoch_v(model, samples, val_ed, device, sp.loss_fn)
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
    meta = dict(epochs_run=epoch, best_epoch=best_epoch, val_editions=list(val_ed),
                best_val_log_mse=best_val, seconds=time.time() - t0,
                n_train_samples=len(train_ed), n_val_samples=len(val_ed))
    return model, meta


def pure_lstm_predictions(panel: pd.DataFrame, samples: dict, editions: list,
                          targets: list, node_index: dict, ref: pd.DataFrame,
                          device, max_epochs=None, log=print) -> tuple:
    rows, detail = [], {}
    for t_i, t in enumerate(targets):
        assert max(samples[t]["hist_years"]) < t, "leakage: sample history reaches the target"
        sp.set_seed(sp.SEED)
        model, meta = train_pure_lstm(samples, t, [e for e in editions if e < t],
                                      device, max_epochs=max_epochs, log=log)
        raw = sp.predict_target(model, samples, t, device)
        (th, _b), scores = sv.select_mix(model, samples, meta["val_editions"], device,
                                         grid=THETA_GRID, tol_pp=0.0, log=log)
        anchor = np.expm1(samples[t]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
        pred = float(th) * raw + (1.0 - float(th)) * anchor
        tgt = ref[ref["year"] == t]
        for r in tgt.itertuples():
            j = node_index[r.ioc]
            rows.append(dict(country=r.country, ioc=r.ioc, year=int(t),
                             actual=int(r.actual), predicted=float(max(pred[j], 0.0))))
        detail[str(t)] = dict(**meta, blend_theta=float(th),
                              val_theta_scores=None if scores is None else
                              {str(k[0]): round(v, 4) for k, v in scores.items()})
        log(f"[pure_lstm] [{t_i + 1}/{len(targets)}] {t}: epoch {meta['best_epoch']}/"
            f"{meta['epochs_run']} theta={th} {meta['seconds']:.2f}s")
    return pd.DataFrame(rows, columns=list(COLUMNS)), detail


class TemporalTransformerMember(nn.Module):

    def __init__(self, in_dim: int = sp.N_STEP_FEAT, static_dim: int = sp.N_STATIC_FEAT,
                 d_model: int = TT_D_MODEL, nhead: int = TT_HEADS, ff: int = TT_FF,
                 layers: int = TT_LAYERS, dropout: float = TT_DROPOUT,
                 seq_len: int = sp.SEQ_LEN, hidden: int = sp.HIDDEN):
        super().__init__()
        self.static_dim = static_dim
        self.seq_len = seq_len
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=ff, dropout=dropout,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(d_model + static_dim, hidden), sp.Swish(), nn.Dropout(sp.DROPOUT),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_weight, static, B: int, T: int, N: int):
        h = self.in_proj(x.view(B, T, N, -1)).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h = self.encoder(h + self.pos[:, :T, :])
        pooled = h.mean(dim=1)
        raw = self.head(torch.cat([pooled, static], dim=-1)).view(B, N)
        anchor = static[:, sp.IDX_LOG1P_LAST_TOTAL].view(B, N)
        return F.relu(anchor + raw)


def _pair_epoch(bb, tt, samples, editions, device, opt_bb=None, opt_tt=None):
    if opt_bb is not None:
        bb.train(); tt.train()
    else:
        bb.eval(); tt.eval()
    total, nb = 0.0, 0
    with torch.set_grad_enabled(opt_bb is not None):
        for i in range(0, len(editions), sp.BATCH_SIZE):
            batch = [samples[e] for e in editions[i:i + sp.BATCH_SIZE]]
            x, ei, ew, static, y, mask, B, T, N = sp.collate(batch)
            x, ei, ew = x.to(device), ei.to(device), ew.to(device)
            static, y, mask = static.to(device), y.to(device), mask.to(device)
            loss = sp.loss_fn(bb(x, ei, ew, static, B, T, N), y, mask) \
                 + sp.loss_fn(tt(x, ei, ew, static, B, T, N), y, mask)
            if opt_bb is not None:
                opt_bb.zero_grad(); opt_tt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bb.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(tt.parameters(), 5.0)
                opt_bb.step(); opt_tt.step()
            total += float(loss.item()) * B
            nb += B
    return total / max(nb, 1)


def train_tt_ensemble(samples: dict, target_year: int, editions_before: list, device,
                      max_epochs=None, log=print):
    max_epochs = max_epochs or sp.MAX_EPOCHS
    train_ed, val_ed = sp.split_train_val(editions_before)
    bb = sp.STGCN_LSTM().to(device)
    tt = TemporalTransformerMember().to(device)
    opt_bb = torch.optim.Adam(bb.parameters(), lr=sp.LR, weight_decay=sp.WEIGHT_DECAY)
    opt_tt = torch.optim.Adam(tt.parameters(), lr=sp.LR, weight_decay=sp.WEIGHT_DECAY)
    sched_bb = torch.optim.lr_scheduler.LambdaLR(opt_bb, sp.lr_lambda)
    sched_tt = torch.optim.lr_scheduler.LambdaLR(opt_tt, sp.lr_lambda)
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = _pair_epoch(bb, tt, samples, train_ed, device, opt_bb, opt_tt)
        sched_bb.step(); sched_tt.step()
        if val_ed:
            va = _pair_epoch(bb, tt, samples, val_ed, device)
            if va < best_val - 1e-6:
                best_val, best_state, best_epoch, bad = va, (
                    copy.deepcopy(bb.state_dict()), copy.deepcopy(tt.state_dict())), epoch, 0
            else:
                bad += 1
                if bad >= sp.PATIENCE:
                    break
        elif tr < best_val - 1e-6:
            best_val, best_state, best_epoch, bad = tr, (
                copy.deepcopy(bb.state_dict()), copy.deepcopy(tt.state_dict())), epoch, 0
    if best_state is not None:
        bb.load_state_dict(best_state[0]); tt.load_state_dict(best_state[1])
    meta = dict(epochs_run=epoch, best_epoch=best_epoch, val_editions=list(val_ed),
                best_val_log_mse=best_val, seconds=time.time() - t0,
                n_train_samples=len(train_ed), n_val_samples=len(val_ed))
    return bb, tt, meta


def select_two_member_theta(model_a, model_b, samples, val_editions, device,
                            grid=THETA_GRID, log=print):
    if not val_editions:
        return 0.5, None
    ys, pa, pb = [], [], []
    for e in val_editions:
        m = samples[e]["mask"]
        ys.append(np.expm1(samples[e]["y"][m]))
        pa.append(sp.predict_target(model_a, samples, e, device)[m])
        pb.append(sp.predict_target(model_b, samples, e, device)[m])
    ys, pa, pb = np.concatenate(ys), np.concatenate(pa), np.concatenate(pb)
    scores = {float(th): sp.weighted_smape(ys, th * pa + (1.0 - th) * pb) for th in grid}
    best = float(min(scores.values()))
    pick = max(k for k, v in scores.items() if v <= best + 1e-12)
    log(f"[tt_ensemble] theta2 scores {({k: round(v, 3) for k, v in scores.items()})}")
    return float(pick), scores


def tt_ensemble_predictions(panel: pd.DataFrame, samples: dict, editions: list,
                            targets: list, node_index: dict, ref: pd.DataFrame,
                            device, max_epochs=None, log=print) -> tuple:
    rows, detail = [], {}
    for t_i, t in enumerate(targets):
        assert max(samples[t]["hist_years"]) < t, "leakage: sample history reaches the target"
        sp.set_seed(sp.SEED)
        bb, tt, meta = train_tt_ensemble(samples, t, [e for e in editions if e < t],
                                         device, max_epochs=max_epochs, log=log)
        th2, scores = select_two_member_theta(bb, tt, samples, meta["val_editions"],
                                              device, log=log)
        raw_bb = sp.predict_target(bb, samples, t, device)
        raw_tt = sp.predict_target(tt, samples, t, device)
        pred = float(th2) * raw_bb + (1.0 - float(th2)) * raw_tt
        tgt = ref[ref["year"] == t]
        for r in tgt.itertuples():
            j = node_index[r.ioc]
            rows.append(dict(country=r.country, ioc=r.ioc, year=int(t),
                             actual=int(r.actual), predicted=float(max(pred[j], 0.0))))
        detail[str(t)] = dict(**meta, theta2=float(th2),
                              val_theta_scores=None if scores is None else
                              {str(k): round(v, 4) for k, v in scores.items()})
        log(f"[tt_ensemble] [{t_i + 1}/{len(targets)}] {t}: epoch {meta['best_epoch']}/"
            f"{meta['epochs_run']} theta2={th2} {meta['seconds']:.2f}s")
        del bb, tt
    return pd.DataFrame(rows, columns=list(COLUMNS)), detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="new methods to compute (comma separated); persistence and "
                         "backbone are always included (artifact / panel arithmetic)")
    ap.add_argument("--max-targets", type=int, default=0, help="smoke run: first N targets")
    ap.add_argument("--max-epochs", type=int, default=0, help="smoke run: cap the epochs")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--reference", default=REFERENCE_CSV,
                    help="backbone predictions CSV used as the alignment standard")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="only the CPU methods (persistence, arima, xgboost, backbone)")
    args = ap.parse_args()
    log = print

    todo = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in todo if m not in METHOD_ORDER or m in ALWAYS_METHODS]
    assert not unknown, f"unknown/implicit methods: {unknown} (known: {list(METHOD_ORDER)})"
    if args.skip_gpu:
        banned = [m for m in todo if m in GPU_METHODS]
        assert not banned, f"--skip-gpu forbids {banned}"
    device = torch.device(args.device)
    log(f"[run] methods={ALWAYS_METHODS + tuple(todo)} device={device} "
        f"max_epochs={args.max_epochs or sp.MAX_EPOCHS} F7 deterministic flags on")

    panel = sp.normalise_panel(pd.read_csv(_abs_path(args.panel)), log=log)
    info = sp.assert_panel_contract(panel)
    node_index = {c: i for i, c in enumerate(sorted(panel["ioc"].unique()))}
    editions = info["years"]
    targets = info["targets"][:args.max_targets] if args.max_targets else info["targets"]
    log(f"[panel] {info['n_rows']} rows, {info['n_countries']} countries, "
        f"{len(editions)} editions, targets {targets[0]}..{targets[-1]}")

    ref = load_reference(_abs_path(args.reference), targets, panel, log=log)
    max_epochs = args.max_epochs or None

    pers_frame = pd.DataFrame({
        "country": panel["ioc"].to_numpy(),
        "year": panel["year"].to_numpy(dtype=int),
        "actual": panel["total"].to_numpy(dtype=float),
    })
    persist_lut = {(str(r.country), int(r.year)): float(max(r.predicted_persist, 0.0))
                   for r in add_persistence(pers_frame).itertuples()
                   if np.isfinite(r.predicted_persist)}

    frames, details = {}, {}
    frames["persistence"] = persistence_predictions(panel, ref, log=log)
    frames["backbone"] = ref[list(COLUMNS)].copy()
    log(f"[backbone] read from the nested artifact ({len(ref)} rows), not retrained")

    if "arima" in todo:
        frames["arima"], details["arima"] = arima_predictions(panel, ref, persist_lut, log=log)
    if "xgboost" in todo:
        t0 = time.time()
        samples = sp.build_samples(panel, node_index, editions, log=log)
        frames["xgboost"] = xgboost_predictions(samples, editions, targets, node_index,
                                                ref, log=log)
        details["xgboost"] = {"seconds": round(time.time() - t0, 2)}

    need_gpu = [m for m in todo if m in GPU_METHODS]
    if need_gpu:
        t0 = time.time()
        samples = sp.build_samples(panel, node_index, editions, log=log)
        log(f"[samples] built in {time.time() - t0:.1f}s")
        if "pure_lstm" in need_gpu:
            t0 = time.time()
            frames["pure_lstm"], details["pure_lstm"] = pure_lstm_predictions(
                panel, samples, editions, targets, node_index, ref, device,
                max_epochs=max_epochs, log=log)
            details["pure_lstm"]["seconds_total"] = round(time.time() - t0, 2)
        if "tt_ensemble" in need_gpu:
            t0 = time.time()
            frames["tt_ensemble"], details["tt_ensemble"] = tt_ensemble_predictions(
                panel, samples, editions, targets, node_index, ref, device,
                max_epochs=max_epochs, log=log)
            details["tt_ensemble"]["seconds_total"] = round(time.time() - t0, 2)

    keysets = [set(zip(f["ioc"].astype(str), f["year"].astype(int))) for f in frames.values()]
    common = set.intersection(*keysets)
    log(f"[protocol] common evaluation set: {len(common)} of {len(ref)} reference rows "
        f"(persistence availability is the binding constraint)")

    lb_ref = lowbase_mask(panel, ref)
    subgroup = set(zip(ref["ioc"].astype(str)[lb_ref], ref["year"].astype(int)[lb_ref]))
    log(f"[protocol] low-base subgroup: {len(subgroup)} of {len(ref)} reference rows "
        f"(no/zero previous record)")

    table, per_method_detail = [], {}
    for name in METHOD_ORDER:
        if name not in frames:
            continue
        d = restrict(frames[name], common)
        row, det = evaluate_method(name, d)
        key = list(zip(frames[name]["ioc"].astype(str), frames[name]["year"].astype(int)))
        m = np.array([k in subgroup for k in key])
        sg = frames[name][m].sort_values(["year", "ioc"]).reset_index(drop=True)
        ya = sg["actual"].to_numpy(float) if len(sg) else np.array([])

        def sg_col(col):
            return sg[col].to_numpy(float) if len(sg) else np.array([])
        row["lowbase_n"] = int(len(sg))
        row["lowbase_rmse"] = (round(rmse(ya, sg_col("predicted")), 4)
                              if len(sg) else float("nan"))
        row["lowbase_weighted_smape"] = (round(sp.weighted_smape(ya, sg_col("predicted")), 4)
                                        if len(sg) else float("nan"))
        row["lowbase_share_pred_ge1"] = (round(float((sg_col("predicted") >= 1.0).mean()), 4)
                                        if len(sg) else float("nan"))
        det["lowbase"] = dict(
            definition="previous panel record has total == 0 or does not exist",
            n_rows=int(len(sg)),
            years=sorted(int(y) for y in sg["year"]) if len(sg) else [],
            rows=[dict(ioc=r.ioc, year=int(r.year), actual=int(r.actual),
                       predicted=round(float(r.predicted), 4)) for r in sg.itertuples()],
        )
        table.append(row)
        per_method_detail[name] = det
        log(f"[{name}] rmse {row['rmse']:.3f} r2 {row['r2']:.3f} "
            f"macro {row['macro_smape']:.3f} weighted {row['weighted_smape']:.3f} "
            f"top10 {row['top10_hit_mean']:.3f} rho {row['spearman_mean']:.3f} "
            f"| lowbase n={row['lowbase_n']} rmse {row['lowbase_rmse']} "
            f"share>=1 {row['lowbase_share_pred_ge1']}")

    outdir = _abs_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    for name, f in frames.items():
        write_predictions(f, os.path.join(outdir, f"predictions_{name}.csv"), log=log)
    table_path = os.path.join(outdir, "metrics_table.csv")
    pd.DataFrame(table).to_csv(table_path, index=False)
    log(f"[write] {table_path}")

    summary = dict(
        protocol=dict(
            targets=[int(t) for t in targets],
            anti_leakage="every fit/training/calibration for target t uses editions < t only; "
                         "max(sample['hist_years']) < t asserted before each training run",
            scored_rows="medal-winning rows (actual > 0) of the Section 5.1 set, restricted "
                        "to the rows every compared method produces; the low-base subgroup "
                        "is scored on each method's own available rows instead (it is empty "
                        "in the common set by construction)",
            n_reference_rows=int(len(ref)), n_common_rows=int(len(common)),
            periods={k: list(map(int, v)) for k, v in DEFAULT_PERIODS.items()},
            lowbase_definition="previous panel record has total == 0 or does not exist",
            disposition=("report as it is, no tuning to close the gap; if a baseline is "
                         "stronger, decompose the low-base subgroup and position the "
                         "neural model as an ablation"),
            determinism="CUBLAS_WORKSPACE_CONFIG=:4096:8, use_deterministic_algorithms(True), "
                        "cudnn.benchmark=False, seed 42 before each training run",
            max_epochs=int(args.max_epochs or sp.MAX_EPOCHS),
            device=str(device),
            hyper_parameters=dict(arima_orders=[list(o) for o in ARIMA_ORDERS],
                                  arima_min_history=ARIMA_MIN_HISTORY,
                                  xgboost=XGB_PARAMS,
                                  theta_grid=list(THETA_GRID),
                                  transformer=dict(layers=TT_LAYERS, d_model=TT_D_MODEL,
                                                   heads=TT_HEADS, ff=TT_FF,
                                                   dropout=TT_DROPOUT)),
        ),
        methods={k: v for k, v in per_method_detail.items()},
        training_detail=details,
        timings={k: (round(v.get("seconds_total"), 2) if isinstance(v, dict)
                     and v.get("seconds_total") is not None else
                     (v.get("seconds") if isinstance(v, dict) else None))
                 for k, v in details.items()},
    )
    detail_path = os.path.join(outdir, "metrics_detail.json")
    with open(detail_path, "w") as fh:
        json.dump(_sanitize(summary), fh, indent=2)
    log(f"[write] {detail_path}")
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
