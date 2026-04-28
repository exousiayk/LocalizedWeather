#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


CHANNELS = ["u", "v", "temp", "dewpoint"]
REPO_PREFIX = Path("/projects3/home/flag0220/LocalizedWeather")


class _DummyMadisNetwork:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _MadisNetworkSafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("Network") or name == "MadisNetwork" or module.endswith("MadisNetwork"):
            return _DummyMadisNetwork
        if module.startswith("scipy") or module.startswith("torch"):
            return _DummyMadisNetwork
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_madis_network(path):
    with open(path, "rb") as f:
        return _MadisNetworkSafeUnpickler(f).load()


def resolve_input_dir(input_dir):
    input_dir = Path(input_dir)
    if input_dir.exists():
        return input_dir

    if not input_dir.is_absolute():
        candidate = REPO_ROOT / input_dir
        if candidate.exists():
            return candidate

    try:
        relative = input_dir.relative_to(REPO_PREFIX)
        candidate = REPO_ROOT / relative
        if candidate.exists():
            return candidate
    except ValueError:
        pass

    return input_dir


def load_station_split(input_dir):
    split_path = Path(input_dir) / "station_split.json"
    if not split_path.exists():
        return None
    with open(split_path, "r") as f:
        return json.load(f)


def build_station_split(n_stations, ghost_holdout_ratio, ghost_split_seed):
    n_stations = int(n_stations)
    n_ghost = int(np.floor(n_stations * ghost_holdout_ratio))
    n_ghost = max(1, n_ghost)
    n_ghost = min(n_ghost, n_stations - 1)

    rng = np.random.RandomState(ghost_split_seed)
    ghost_indices = np.sort(rng.choice(np.arange(n_stations), size=n_ghost, replace=False))
    seen_mask = np.ones(n_stations, dtype=bool)
    seen_mask[ghost_indices] = False
    ghost_mask = ~seen_mask

    return {
        "n_stations": n_stations,
        "ghost_holdout_ratio": float(ghost_holdout_ratio),
        "ghost_split_seed": int(ghost_split_seed),
        "ghost_indices": ghost_indices.astype(np.int64),
        "seen_indices": np.where(seen_mask)[0].astype(np.int64),
        "ghost_station_mask": ghost_mask,
        "seen_station_mask": seen_mask,
    }


def resolve_output_file(input_dir, prefix, metric, experiment_tag=None):
    input_dir = Path(input_dir)
    if experiment_tag:
        tagged = input_dir / f"{prefix}_{metric}_{experiment_tag}_min.pkl"
        if tagged.exists():
            return tagged
    legacy = input_dir / f"{prefix}_{metric}_min.pkl"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Missing file: prefix={prefix}, metric={metric}, tag={experiment_tag}")


def to_time_station_feature(arr, n_stations_hint=None):
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        if n_stations_hint is None or n_stations_hint <= 0:
            raise ValueError(
                f"Unsupported 2D shape without station hint: {arr.shape}."
            )
        if arr.shape[0] % n_stations_hint != 0:
            raise ValueError(
                f"Cannot reshape 2D array {arr.shape} with n_stations={n_stations_hint}"
            )
        return arr.reshape(-1, n_stations_hint, arr.shape[-1])
    raise ValueError(f"Unsupported shape: {arr.shape}")


def wmape_percent(pred, target, eps=1e-6):
    num = np.nansum(np.abs(pred - target))
    den = max(float(np.nansum(np.abs(target))), eps)
    return float(num / den * 100.0)


def summarize_channel(pred, target, channel_name):
    err = pred - target
    abs_err = np.abs(err)
    return {
        "channel": channel_name,
        "mae": float(np.nanmean(abs_err)),
        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "wmape_percent": wmape_percent(pred, target),
        "bias": float(np.nanmean(err)),
        "pred_mean": float(np.nanmean(pred)),
        "target_mean": float(np.nanmean(target)),
        "pred_std": float(np.nanstd(pred)),
        "target_std": float(np.nanstd(target)),
        "max_abs_error": float(np.nanmax(abs_err)),
    }


def build_neighbor_indices(lons, lats, ghost_indices, seen_indices, k=4, method="mean"):
    seen_coords = np.column_stack([lons[seen_indices], lats[seen_indices]]).astype(np.float32)
    ghost_coords = np.column_stack([lons[ghost_indices], lats[ghost_indices]]).astype(np.float32)

    diff = ghost_coords[:, None, :] - seen_coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    n_k = min(k, len(seen_indices))
    nn_local = np.argsort(dist, axis=1)[:, :n_k]
    nn_global = seen_indices[nn_local]

    if method == "nearest":
        return nn_global[:, :1], None

    if method == "mean":
        return nn_global, None

    if method == "idw":
        nn_dist = np.take_along_axis(dist, nn_local, axis=1)
        weights = 1.0 / np.maximum(nn_dist, 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return nn_global, weights.astype(np.float32)

    raise ValueError(f"Unsupported interpolation method: {method}")


def interpolate_ghost_values(targets, ghost_neighbor_idx, ghost_weights, ghost_indices, method):
    if method == "nearest":
        return targets[:, ghost_neighbor_idx[:, 0], :]

    if method == "mean":
        neighbor_values = targets[:, ghost_neighbor_idx, :]
        return np.mean(neighbor_values, axis=2)

    if method == "idw":
        neighbor_values = targets[:, ghost_neighbor_idx, :]
        weights = ghost_weights[None, :, :, None]
        return np.sum(neighbor_values * weights, axis=2)

    raise ValueError(f"Unsupported interpolation method: {method}")


def evaluate_baseline(args):
    input_dir = resolve_input_dir(args.input_dir)
    params_path = input_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json in {input_dir}")
    with open(params_path, "r") as f:
        params = json.load(f)

    split = load_station_split(input_dir)

    madis_network = load_madis_network(input_dir / "madis_network.pkl")
    lons = np.asarray(madis_network.stat_lons)
    lats = np.asarray(madis_network.stat_lats)

    if split is None:
        split = build_station_split(
            madis_network.n_stations,
            params.get("ghost_holdout_ratio", 0.1),
            params.get("ghost_split_seed", 42),
        )

    ghost_indices = np.asarray(split["ghost_indices"], dtype=np.int64)
    seen_indices = np.asarray(split["seen_indices"], dtype=np.int64)

    targets = load_pickle(resolve_output_file(input_dir, "Targets", args.metric, args.experiment_tag))
    y = to_time_station_feature(targets, n_stations_hint=split.get("n_stations"))

    if y.shape[1] != split.get("n_stations"):
        raise ValueError(f"Station count mismatch: targets={y.shape[1]}, split={split.get('n_stations')}")

    ghost_neighbor_idx, ghost_weights = build_neighbor_indices(
        lons,
        lats,
        ghost_indices,
        seen_indices,
        k=args.k,
        method=args.method,
    )

    y_ghost = y[:, ghost_indices, :]
    pred_ghost = interpolate_ghost_values(y, ghost_neighbor_idx, ghost_weights, ghost_indices, args.method)

    rows = []
    for channel_idx, channel_name in enumerate(CHANNELS):
        rows.append(summarize_channel(pred_ghost[..., channel_idx], y_ghost[..., channel_idx], channel_name))

    overall = summarize_channel(pred_ghost.reshape(-1), y_ghost.reshape(-1), "overall")
    overall["ghost_count"] = int(len(ghost_indices))
    overall["seen_count"] = int(len(seen_indices))
    overall["frames"] = int(y.shape[0])
    overall["method"] = args.method
    overall["k"] = int(args.k)

    tag_part = f"_{args.experiment_tag}" if args.experiment_tag else ""
    out_prefix = args.output_prefix or f"interp_{args.method}"
    summary_csv = input_dir / f"{out_prefix}_{args.metric}{tag_part}_ghost_channel_summary.csv"
    summary_json = input_dir / f"{out_prefix}_{args.metric}{tag_part}_ghost_summary.json"
    pred_pkl = input_dir / f"{out_prefix}_{args.metric}{tag_part}_Preds_ghost.pkl"
    target_pkl = input_dir / f"{out_prefix}_{args.metric}{tag_part}_Targets_ghost.pkl"

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump({"overall": overall, "channels": rows}, f, indent=2)

    with open(pred_pkl, "wb") as f:
        pickle.dump(pred_ghost, f)
    with open(target_pkl, "wb") as f:
        pickle.dump(y_ghost, f)

    print("Interpolation baseline (ghost nodes only)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"overall: {overall}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {pred_pkl}")
    print(f"saved: {target_pkl}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate interpolation-only ghost baseline on test outputs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="CUSTOM")
    parser.add_argument("--experiment_tag", type=str, default=None)
    parser.add_argument("--method", type=str, choices=["mean", "nearest", "idw"], default="mean")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--output_prefix", type=str, default="interp_baseline")
    args = parser.parse_args()

    evaluate_baseline(args)


if __name__ == "__main__":
    main()