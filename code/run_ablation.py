import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch_geometric

from evaluate_recent_editions import GEN
from evaluate_tokyo_roc import HISTORICAL_CODE, PANEL, REPORTED_CODE, YEAR

CALLER_DIRECTORY = Path.cwd()
sys.dont_write_bytecode = True
import baselines_compare as baselines
import nested_backtest as nested
import stgcn_variants as variants


sp = variants.sp
OUTDIR = GEN / "ablations"
ARMS = ("full", "no_rame", "no_pagerank", "self_loops", "no_graph")
REMOVED_FIELDS = {
    "no_rame": {
        "steps": ("log1p_dynweight", "dynweight_share"),
        "static": ("log1p_dynweight_at_target", "dynweight_share_at_target"),
    },
    "no_pagerank": {"steps": ("z_walkforward_pagerank",)},
}
FEATURES = {"steps": sp.STEP_FEAT_NAMES, "static": sp.STATIC_FEAT_NAMES}


def ablated_samples(samples, arm):
    result = {}
    fields = REMOVED_FIELDS.get(arm, {})
    for year, sample in samples.items():
        modified = dict(sample)
        for block, names in fields.items():
            columns = [FEATURES[block].index(name) for name in names]
            values = sample[block].copy()
            values[..., columns] = 0
            modified[block] = values
        if arm == "self_loops":
            edges = sample["edge_index"]
            keep = edges[0] == edges[1]
            modified.update(
                edge_index=edges[:, keep],
                edge_weight=sample["edge_weight"][keep],
                n_edges=int(keep.sum()),
            )
        result[year] = modified
    return result


def check_inputs(original, ablated, arm, years, n_nodes):
    fields = REMOVED_FIELDS.get(arm, {})
    changes = {name: 0 for names in fields.values() for name in names}
    edge_counts = {}
    for year in years:
        before, after = original[year], ablated[year]
        if max(after["hist_years"]) >= year:
            raise ValueError(f"Sample {year} contains results at or after its target")
        for block, names in FEATURES.items():
            changed = [i for i, name in enumerate(names) if name in fields.get(block, ())]
            other = [i for i in range(len(names)) if i not in changed]
            if np.any(after[block][..., changed]) or not np.array_equal(
                before[block][..., other], after[block][..., other]
            ):
                raise ValueError(f"Unexpected {block} input change in {arm}, {year}")
            for i in changed:
                changes[names[i]] += int(np.count_nonzero(before[block][..., i]))
        if not np.array_equal(before["static"][:, sp.IDX_LOG1P_LAST_TOTAL],
                              after["static"][:, sp.IDX_LOG1P_LAST_TOTAL]):
            raise ValueError(f"Persistence anchor changed in {arm}, {year}")
        for key in ("mask", "y"):
            if not np.array_equal(before[key], after[key]):
                raise ValueError(f"Labels or award coverage changed in {arm}, {year}")
        for key in ("edge_index", "edge_weight"):
            if arm != "self_loops" and not np.array_equal(before[key], after[key]):
                raise ValueError(f"Unintended graph change in {arm}, {year}")
        if arm == "self_loops":
            idx = after["edge_index"]
            if not (np.array_equal(idx[0], idx[1]) and
                    np.array_equal(np.sort(idx[0]), np.arange(n_nodes)) and
                    np.all(after["edge_weight"] == 1) and
                    before["n_edges"] > after["n_edges"]):
                raise ValueError(f"Self-loop intervention is invalid in {year}")
        edge_counts[year] = {"original": before["n_edges"], "used": after["n_edges"]}
    if any(value == 0 for value in changes.values()):
        raise ValueError(f"Inactive ablated covariates in {arm}: {changes}")
    return {"removed_nonzero_values": changes, "training_sample_edges": edge_counts}


def protocol(args):
    paths = [PANEL, Path(__file__), Path(sp.__file__), Path(variants.__file__), Path(baselines.__file__)]
    return {
        "target_rng_seed": sp.SEED,
        "loss": "masked log-count MSE",
        "max_epochs": sp.MAX_EPOCHS,
        "patience": sp.PATIENCE,
        "batch_size": sp.BATCH_SIZE,
        "grid": list(baselines.THETA_GRID),
        "device": args.device,
        "threads": args.threads,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def save_result(path, result):
    path.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    Path(handle.name).replace(path)


def fit_one(arm, samples, year, earlier, device):
    sp.set_seed(sp.SEED)
    if arm == "no_graph":
        return baselines.train_pure_lstm(samples, year, earlier, device)
    return variants.train_target_v(samples, year, earlier, device, sp.N_STATIC_FEAT, sp.loss_fn)


def predict_year(model, samples, reference, nodes, year, metadata, device):
    theta, _ = variants.select_mix(
        model, samples, metadata["val_editions"], device, grid=baselines.THETA_GRID, tol_pp=0.0,
    )
    a, b = theta
    if b != 0.0:
        raise ValueError("The reference blend should use only the network and persistence")
    raw = sp.predict_target(model, samples, year, device)
    anchor = np.expm1(samples[year]["static"][:, sp.IDX_LOG1P_LAST_TOTAL])
    forecasts = a * raw + (1.0 - a) * anchor
    if not np.all(np.isfinite(forecasts)) or np.any(forecasts < 0):
        raise ValueError(f"Invalid medal forecasts for {year}")
    rows = []
    for row in reference.itertuples(index=False):
        lookup = HISTORICAL_CODE if year == YEAR and row.ioc == REPORTED_CODE else row.ioc
        rows.append({
            "country": row.country, "ioc": row.ioc, "year": year,
            "actual": int(row.total), "predicted": float(forecasts[nodes[lookup]]),
        })
    return rows, a


def main():
    parser = argparse.ArgumentParser(description="Fit matched component ablations.")
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--years", type=int, nargs="+", choices=nested.TARGET_YEARS,
                        default=list(nested.TARGET_YEARS))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=Path, default=OUTDIR)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    directory = (CALLER_DIRECTORY / args.results_dir).resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    panel, info = variants.load_panel(str(PANEL))
    editions = info["years"]
    if tuple(info["targets"]) != nested.TARGET_YEARS:
        raise ValueError("Ablation targets differ from the published benchmark")
    nodes = {ioc: i for i, ioc in enumerate(sorted(panel["ioc"].unique()))}
    samples = sp.build_samples(panel, nodes, editions)
    metadata = protocol(args)
    for arm in dict.fromkeys(args.arm):
        ablated = ablated_samples(samples, arm)

        for year in sorted(set(args.years)):
            earlier = [e for e in editions if e < year]
            train, val = sp.split_train_val(earlier)
            phases = {
                phase: check_inputs(samples, ablated, arm, phase_years, len(nodes))
                for phase, phase_years in (("training", train), ("validation", val), ("target", [year]))
            }
            if args.check_only:
                print(f"{arm} {year}: feature, graph, target and label checks passed", flush=True)
                continue
            path = directory / "fits" / arm / f"{year}.json"
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["protocol"] != metadata or previous["arm"] != arm or previous["year"] != year:
                    raise ValueError(f"Existing fold has a different configuration: {path}")
                print(f"{arm} {year}: restored completed fold", flush=True)
                continue
            print(f"{arm} {year}: fitting {len(train)} training and {len(val)} validation editions", flush=True)
            model, fitted = fit_one(arm, ablated, year, earlier, device)
            ref = panel.loc[panel["year"].eq(year)]
            rows, theta = predict_year(model, ablated, ref, nodes, year, fitted, device)
            n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            del model
            save_result(path, {
                "arm": arm, "year": year, "protocol": metadata,
                "intervention": phases, "parameters": n_params,
                "train_editions": train, "fit": fitted, "blend_theta": theta, "predictions": rows,
            })
            print(f"{arm} {year}: {len(rows)} medalists, epoch {fitted['best_epoch']}, "
                  f"blend {theta:g}, {fitted['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
