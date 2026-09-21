from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from backtest_period import smape_obs

GEN = os.path.join(_HERE, "gen")
DEFAULT_VARIANTS_DIR = os.path.join(GEN, "nested", "variants")
DEFAULT_OUT_DIR = os.path.join(GEN, "nested")
REFERENCE_DIR = os.path.join(GEN, "variants_cpu")

VARIANT_TAGS = ("base", "V1", "V2", "V3", "V4", "V5", "V6")
PARSIMONY_ORDER = ("base", "V3", "V2", "V1", "V5", "V4", "V6")
TARGET_YEARS = (1976, 1980, 1984, 1988, 1992, 1996,
                2000, 2004, 2008, 2012, 2016, 2020, 2024)
N_EXPECTED_ROWS = 916
DEFAULT_TOL_PP = 0.25
REFERENCE_TAGS = ("base", "V1", "V5")
COLUMNS = ("country", "ioc", "year", "actual", "predicted")
SCORE_DP = 6


def smape_w_raw(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype=float)
    ok = y > 0
    tot = float(y[ok].sum())
    if tot <= 0:
        return float("nan")
    e = smape_obs(y, predicted)
    return float((y[ok] * e[ok]).sum() / tot)


def inner_scores(frames: dict) -> dict:
    out = {int(t): {} for t in TARGET_YEARS}
    for tag, df in frames.items():
        for tau, g in df.groupby("year"):
            out[int(tau)][tag] = smape_w_raw(g["actual"].to_numpy(float),
                                             g["predicted"].to_numpy(float))
    return out


def select_target(t: int, raw: dict, tol_pp: float) -> dict:
    scores, n_inner_defined, eligible = {}, {}, []
    for tag in VARIANT_TAGS:
        vals = [raw[tau][tag] for tau in TARGET_YEARS
                if tau < t and math.isfinite(raw[tau][tag])]
        n_inner_defined[tag] = len(vals)
        scores[tag] = float(np.mean(vals)) if vals else float("nan")
        if vals:
            eligible.append(tag)
    n_inner_editions = sum(1 for tau in TARGET_YEARS if tau < t)

    if not eligible:
        return dict(year=t, n_inner_editions=n_inner_editions, scores=scores,
                    n_inner_defined=n_inner_defined, eligible=[],
                    argmin=None, winner="base", pool=[], pool_size=0,
                    tie_broken=False, margin_pp=None, fallback=True)

    best = min(scores[v] for v in eligible)
    pool = [v for v in PARSIMONY_ORDER
            if v in eligible and scores[v] <= best + tol_pp]
    winner = pool[0]
    argmin = min(eligible, key=lambda v: (scores[v], PARSIMONY_ORDER.index(v)))
    outside = [v for v in eligible if v not in pool]
    margin = float(min(scores[v] for v in outside) - scores[winner]) if outside else None
    return dict(year=t, n_inner_editions=n_inner_editions, scores=scores,
                n_inner_defined=n_inner_defined, eligible=eligible,
                argmin=argmin, winner=winner, pool=pool, pool_size=len(pool),
                tie_broken=len(pool) > 1, margin_pp=margin, fallback=False)


def prediction_path(directory: str, tag: str) -> str:
    return os.path.join(directory, f"predictions_{tag}.csv")


def load_variant_csvs(directory: str, log=print) -> dict:
    frames = {}
    for tag in VARIANT_TAGS:
        path = prediction_path(directory, tag)
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: missing {path}; produce the seven "
                             "configuration CSVs first "
                             "(run_nested_variants.py)")
        df = pd.read_csv(path)
        missing = [c for c in COLUMNS if c not in df.columns]
        assert not missing, f"{path}: missing column(s) {missing}"
        df = df[list(COLUMNS)].copy()
        df["year"] = df["year"].astype(int)
        df["actual"] = df["actual"].astype(int)
        pred = df["predicted"].to_numpy(float)
        assert np.isfinite(pred).all(), f"{path}: non-finite predicted value(s)"
        assert (pred >= 0).all(), (f"{path}: negative predicted value(s) -- the "
                                   "producer clips at 0, so this is a broken input")
        frames[tag] = df
        log(f"[load] {tag:<4} {os.path.basename(path)}: {len(df)} rows, "
            f"{df['year'].nunique()} editions")
    return frames


def assert_cross_variant(frames: dict) -> None:
    ref = frames["base"].sort_values(["ioc", "year"]).reset_index(drop=True)
    assert not ref.duplicated(["ioc", "year"]).any(), "base: duplicate (ioc,year) rows"
    assert len(ref) == N_EXPECTED_ROWS, \
        f"base: {len(ref)} rows, expected {N_EXPECTED_ROWS}"
    years = sorted(int(y) for y in ref["year"].unique())
    assert years == list(TARGET_YEARS), \
        f"base: target years {years} != {list(TARGET_YEARS)}"
    for tag in VARIANT_TAGS[1:]:
        d = frames[tag].sort_values(["ioc", "year"]).reset_index(drop=True)
        assert len(d) == len(ref), f"{tag}: {len(d)} rows vs base {len(ref)}"
        for col in ("ioc", "year", "country", "actual"):
            same = (d[col].to_numpy() == ref[col].to_numpy()).all()
            assert same, (f"{tag}: column '{col}' differs from base when aligned "
                          "on (ioc,year) -- the seven CSVs are not the same "
                          "observation set, nested stitching would be ill-defined")


def selection_table(selections: list) -> pd.DataFrame:
    rows = []
    for sel in selections:
        row = {"year": sel["year"]}
        for tag in VARIANT_TAGS:
            s = sel["scores"][tag]
            row[f"score_{tag}"] = np.nan if not math.isfinite(s) else round(s, SCORE_DP)
        row["winner"] = sel["winner"]
        row["pool_size"] = sel["pool_size"]
        row["tie_broken"] = bool(sel["tie_broken"])
        rows.append(row)
    return pd.DataFrame(rows)


def nested_predictions(frames: dict, selections: list) -> pd.DataFrame:
    blocks = [frames[sel["winner"]][frames[sel["winner"]]["year"] == sel["year"]]
              for sel in selections]
    out = pd.concat(blocks, ignore_index=True)[list(COLUMNS)]
    assert len(out) == N_EXPECTED_ROWS, \
        f"nested predictions: {len(out)} rows != {N_EXPECTED_ROWS}"
    assert set(int(y) for y in out["year"]) == set(TARGET_YEARS), \
        "nested predictions: unexpected edition set"
    years, actuals = out["year"].to_numpy(), out["actual"].to_numpy()
    assert (np.diff(years) >= 0).all(), "nested predictions: years not ascending"
    assert ((np.diff(years) > 0) | (np.diff(actuals) <= 0)).all(), \
        "nested predictions: actual not descending within an edition"
    return out


def equivalence_check(frames: dict, reference_dir: str, log=print) -> dict:
    out = {"reference_dir": reference_dir, "note": "report only, no gate (F7)",
           "variants": {}}
    for tag in REFERENCE_TAGS:
        path = prediction_path(reference_dir, tag)
        entry = {"reference_csv": path}
        if not os.path.exists(path):
            entry["status"] = "skipped: reference file missing"
            out["variants"][tag] = entry
            log(f"[equiv] {tag}: skipped ({path} missing)")
            continue
        ref = pd.read_csv(path)
        missing = [c for c in COLUMNS if c not in ref.columns]
        assert not missing, f"{path}: missing column(s) {missing}"
        ref["year"] = ref["year"].astype(int)
        ref["actual"] = ref["actual"].astype(int)
        cand = frames[tag][["ioc", "year", "actual", "predicted"]]
        m = cand.merge(ref, on=["ioc", "year"], how="outer",
                       suffixes=("", "_ref"), indicator=True)
        both = m[m["_merge"] == "both"]
        delta = np.abs(both["predicted"].to_numpy(float)
                       - both["predicted_ref"].to_numpy(float))
        entry.update(
            status="ok",
            n_keys_candidate=int(len(cand)), n_keys_reference=int(len(ref)),
            n_matched=int(len(both)),
            n_only_in_candidate=int((m["_merge"] == "left_only").sum()),
            n_only_in_reference=int((m["_merge"] == "right_only").sum()),
            n_actual_mismatch=int((both["actual"].to_numpy()
                                   != both["actual_ref"].to_numpy()).sum()),
            max_abs_delta_predicted=float(delta.max()) if delta.size else None,
        )
        out["variants"][tag] = entry
        log(f"[equiv] {tag}: matched {entry['n_matched']}/{entry['n_keys_reference']} "
            f"keys, only_in_candidate={entry['n_only_in_candidate']}, "
            f"only_in_reference={entry['n_only_in_reference']}, "
            f"actual_mismatch={entry['n_actual_mismatch']}, "
            f"max|dpred|={entry['max_abs_delta_predicted']}")
    return out


def _jsonable(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def build_summary(args, selections: list, equivalence: dict) -> dict:
    wins = {tag: 0 for tag in VARIANT_TAGS}
    editions = []
    for sel in selections:
        wins[sel["winner"]] += 1
        editions.append(dict(
            year=sel["year"], n_inner_editions=sel["n_inner_editions"],
            scores={tag: _jsonable(sel["scores"][tag]) for tag in VARIANT_TAGS},
            n_inner_defined=sel["n_inner_defined"], eligible=sel["eligible"],
            argmin_variant=sel["argmin"], winner=sel["winner"], pool=sel["pool"],
            pool_size=sel["pool_size"], tie_broken=bool(sel["tie_broken"]),
            margin_pp=_jsonable(sel["margin_pp"]),
            fallback_to_base=bool(sel["fallback"]),
        ))
    return dict(
        protocol=dict(
            document="this module", frozen="before the nested rerun",
            rule="F2 SMAPE_w_raw inner statistic averaged over strictly pre-target "
                 "editions; F3 argmin + tolerance pool + frozen parsimony order; "
                 "F4 per-edition winner rows; base fallback when nothing is eligible",
            tol_pp=args.tol_pp, parsimony_order=list(PARSIMONY_ORDER),
            candidate_tags=list(VARIANT_TAGS), target_years=list(TARGET_YEARS),
            note="no elastic-window smoothing in the selection statistic "
                 "(a window at tau=t-4 would reach target t); window statistics "
                 "are for reporting only (backtest_period.py)",
        ),
        inputs=dict(variants_dir=args.variants_dir,
                    files={tag: prediction_path(args.variants_dir, tag)
                           for tag in VARIANT_TAGS}),
        outputs=dict(selection_table=os.path.join(args.out_dir, "selection_table.csv"),
                     predictions_nested=os.path.join(args.out_dir,
                                                     "predictions_nested.csv"),
                     summary=os.path.join(args.out_dir, "selection_summary.json")),
        wins=wins, editions=editions, equivalence_check=equivalence,
    )


def parse_args(argv: list | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-dir", default=DEFAULT_VARIANTS_DIR,
                    help="directory holding predictions_{base,V1..V6}.csv")
    ap.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--tol-pp", dest="tol_pp", type=float, default=DEFAULT_TOL_PP,
                    help=f"F3 tie tolerance in pp (frozen: {DEFAULT_TOL_PP})")
    ap.add_argument("--equivalence-check", dest="equivalence_check",
                    action="store_true", default=True,
                    help="cross-environment reference check vs gen/variants_cpu "
                         "(report only, default on)")
    ap.add_argument("--no-equivalence-check", dest="equivalence_check",
                    action="store_false")
    return ap.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    log = print
    assert args.tol_pp > 0, \
        f"--tol-pp must be > 0 (F3 freezes {DEFAULT_TOL_PP} pp)"

    frames = load_variant_csvs(args.variants_dir, log=log)
    assert_cross_variant(frames)
    raw = inner_scores(frames)

    selections = [select_target(t, raw, args.tol_pp) for t in TARGET_YEARS]
    for sel in selections:
        margin = "n/a" if sel["margin_pp"] is None else f"{sel['margin_pp']:.3f}pp"
        note = "  [F3 fallback: no eligible candidate]" if sel["fallback"] else ""
        log(f"[{sel['year']}] inner={sel['n_inner_editions']:2d} "
            f"winner={sel['winner']:<4} pool={sel['pool_size']} "
            f"tie_broken={int(sel['tie_broken'])} argmin={str(sel['argmin']):<4} "
            f"margin={margin}{note}")

    table = selection_table(selections)
    nested = nested_predictions(frames, selections)
    equivalence = (equivalence_check(frames, REFERENCE_DIR, log=log)
                   if args.equivalence_check else
                   {"status": "disabled (--no-equivalence-check)"})
    summary = build_summary(args, selections, equivalence)

    os.makedirs(args.out_dir, exist_ok=True)
    table_path = summary["outputs"]["selection_table"]
    nested_path = summary["outputs"]["predictions_nested"]
    summary_path = summary["outputs"]["summary"]
    table.to_csv(table_path, index=False)
    nested.to_csv(nested_path, index=False)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[write] {table_path}")
    log(f"[write] {nested_path} ({len(nested)} rows)")
    log(f"[write] {summary_path}")
    log("[wins] " + "  ".join(f"{tag}={summary['wins'][tag]}"
                              for tag in VARIANT_TAGS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
