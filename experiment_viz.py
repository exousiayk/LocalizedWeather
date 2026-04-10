#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

CHANNELS = ["u", "v", "temp", "dewpoint"]


def load_station_split(input_dir):
    split_path = Path(input_dir) / "station_split.json"
    if not split_path.exists():
        return None
    with open(split_path, "r") as f:
        return json.load(f)


def get_scope_station_indices(input_dir, node_scope):
    split = load_station_split(input_dir)
    if split is None:
        return None
    if node_scope == "ghost":
        return split.get("ghost_indices")
    if node_scope == "seen":
        return split.get("seen_indices")
    return list(range(split.get("n_stations", 0)))

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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
                f"Unsupported 2D shape without station hint: {arr.shape}. "
                "Provide station_split.json or use full-scope outputs."
            )
        if arr.shape[0] % n_stations_hint != 0:
            raise ValueError(
                f"Cannot reshape 2D array {arr.shape} with n_stations={n_stations_hint}"
            )
        return arr.reshape(-1, n_stations_hint, arr.shape[-1])
    raise ValueError(f"Unsupported shape: {arr.shape}")

def load_pred_target(input_dir, metric, experiment_tag=None, node_scope="all"):
    target_prefix = "Targets"
    pred_prefix = "Preds"
    if node_scope == "seen":
        target_prefix = "Targets_seen"
        pred_prefix = "Preds_seen"
    elif node_scope == "ghost":
        target_prefix = "Targets_ghost"
        pred_prefix = "Preds_ghost"

    scope_indices = get_scope_station_indices(input_dir, node_scope)
    n_stations_hint = len(scope_indices) if scope_indices else None

    targets = load_pickle(resolve_output_file(input_dir, target_prefix, metric, experiment_tag))
    preds = load_pickle(resolve_output_file(input_dir, pred_prefix, metric, experiment_tag))
    y = to_time_station_feature(targets, n_stations_hint=n_stations_hint)
    p = to_time_station_feature(preds, n_stations_hint=n_stations_hint)
    if y.shape != p.shape:
        raise ValueError(f"Shape mismatch: targets={y.shape}, preds={p.shape}")
    return y, p

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

def run_summary(args):
    y, p = load_pred_target(args.input_dir, args.metric, args.experiment_tag, args.node_scope)
    if y.shape[-1] != len(CHANNELS):
        raise ValueError(f"Expected {len(CHANNELS)} channels, got {y.shape[-1]}")

    rows = []
    for i, ch in enumerate(CHANNELS):
        rows.append(summarize_channel(p[..., i], y[..., i], ch))

    input_dir = Path(args.input_dir)
    tag_part = f"_{args.experiment_tag}" if args.experiment_tag else ""
    scope_part = f"_{args.node_scope}"
    summary_csv = input_dir / f"{args.output_prefix}_{args.metric}{tag_part}{scope_part}_channel_summary.csv"
    summary_json = input_dir / f"{args.output_prefix}_{args.metric}{tag_part}{scope_part}_summary.json"

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)

    print("Channel summary:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")

    if args.save_stationwise:
        station_rows = []
        n_stations = y.shape[1]
        for s in range(n_stations):
            row = {"station": s}
            for c, ch in enumerate(CHANNELS):
                ps = p[:, s, c]
                ys = y[:, s, c]
                row[f"{ch}_mae"] = float(np.nanmean(np.abs(ps - ys)))
                row[f"{ch}_rmse"] = float(np.sqrt(np.nanmean((ps - ys) ** 2)))
                row[f"{ch}_wmape_percent"] = wmape_percent(ps, ys)
            station_rows.append(row)
        station_csv = input_dir / f"{args.output_prefix}_{args.metric}{tag_part}{scope_part}_stationwise.csv"
        pd.DataFrame(station_rows).to_csv(station_csv, index=False)
        print(f"saved: {station_csv}")

def run_gif(args):
    channel_map = {"u": 0, "v": 1, "temp": 2, "dewpoint": 3}
    channel_idx = channel_map[args.channel]

    y_all, p_all = load_pred_target(args.input_dir, args.metric, args.experiment_tag, args.node_scope)
    y = y_all[..., channel_idx]
    p = p_all[..., channel_idx]

    if args.time_end is not None:
        y = y[args.time_start:args.time_end]
        p = p[args.time_start:args.time_end]
    else:
        y = y[args.time_start:]
        p = p[args.time_start:]

    frame_indices = np.arange(0, len(y), max(1, args.frame_step))
    y = y[frame_indices]
    p = p[frame_indices]

    input_dir = Path(args.input_dir)
    data_root = input_dir.parents[1]
    shapefile_path = Path(args.shapefile_path) if args.shapefile_path else data_root / "Shapefiles/Regions/northeastern_buffered.shp"
    if not shapefile_path.exists():
        shapefile_path = Path(__file__).resolve().parent / "Shapefiles/Regions/northeastern_buffered.shp"

    madis_network = load_pickle(input_dir / "madis_network.pkl")
    lons = np.asarray(madis_network.stat_lons)
    lats = np.asarray(madis_network.stat_lats)

    scope_indices = get_scope_station_indices(input_dir, args.node_scope)
    if scope_indices:
        scope_indices = np.asarray(scope_indices, dtype=int)
        lons = lons[scope_indices]
        lats = lats[scope_indices]

    roi = gpd.read_file(shapefile_path).dissolve() if shapefile_path.exists() else None

    if len(lons) != y.shape[1]:
        raise ValueError(f"Station count mismatch: coords={len(lons)}, data={y.shape[1]}")

    vmin = min(np.nanmin(y), np.nanmin(p))
    vmax = max(np.nanmax(y), np.nanmax(p))
    abs_error = np.abs(p - y)
    diff_limit = np.nanmax(abs_error)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    def draw_panel(ax, values, title, cmap, vmin_value=None, vmax_value=None):
        if roi is not None:
            roi.boundary.plot(ax=ax, color="0.35", linewidth=1.0, zorder=0)
        sc = ax.scatter(lons, lats, c=values, s=35, cmap=cmap,
                        vmin=vmin_value, vmax=vmax_value, edgecolors="black", linewidths=0.2)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lons.min() - 0.5, lons.max() + 0.5)
        ax.set_ylim(lats.min() - 0.5, lats.max() + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        return sc

    sc_gt = draw_panel(axes[0], y[0], "GT", "coolwarm", vmin, vmax)
    sc_pred = draw_panel(axes[1], p[0], "Pred", "coolwarm", vmin, vmax)
    sc_diff = draw_panel(axes[2], abs_error[0], "Abs Error", "gray_r", 0, diff_limit)

    cbar0 = fig.colorbar(sc_gt, ax=axes[:2], shrink=0.9)
    cbar0.set_label(args.channel)
    cbar1 = fig.colorbar(sc_diff, ax=axes[2], shrink=0.9)
    cbar1.set_label(f"{args.channel} abs error")

    time_text = fig.suptitle(f"{args.channel} at time index {args.time_start}", y=1.02)

    def update(frame_idx):
        sc_gt.set_array(y[frame_idx])
        sc_pred.set_array(p[frame_idx])
        sc_diff.set_array(np.abs(p[frame_idx] - y[frame_idx]))
        time_text.set_text(f"{args.channel} at sampled frame {frame_idx + 1}/{len(y)} (step={args.frame_step})")
        return sc_gt, sc_pred, sc_diff, time_text

    anim = FuncAnimation(fig, update, frames=len(y), interval=1000 // max(args.fps, 1), blit=False)

    out_path = input_dir / args.output
    anim.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified summary + gif tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--input_dir", type=str, required=True)
    p_sum.add_argument("--metric", type=str, default="CUSTOM")
    p_sum.add_argument("--experiment_tag", type=str, default=None)
    p_sum.add_argument("--node_scope", type=str, choices=["all", "seen", "ghost"], default="all")
    p_sum.add_argument("--output_prefix", type=str, default="summary")
    p_sum.add_argument("--save_stationwise", action="store_true")
    p_sum.set_defaults(func=run_summary)

    p_gif = sub.add_parser("gif")
    p_gif.add_argument("--input_dir", type=str, required=True)
    p_gif.add_argument("--metric", type=str, default="CUSTOM")
    p_gif.add_argument("--experiment_tag", type=str, default=None)
    p_gif.add_argument("--node_scope", type=str, choices=["all", "seen", "ghost"], default="all")
    p_gif.add_argument("--channel", type=str, default="v", choices=["u", "v", "temp", "dewpoint"])
    p_gif.add_argument("--output", type=str, default="gt_pred_map.gif")
    p_gif.add_argument("--fps", type=int, default=4)
    p_gif.add_argument("--frame_step", type=int, default=2)
    p_gif.add_argument("--dpi", type=int, default=120)
    p_gif.add_argument("--time_start", type=int, default=0)
    p_gif.add_argument("--time_end", type=int, default=None)
    p_gif.add_argument("--shapefile_path", type=str, default=None)
    p_gif.set_defaults(func=run_gif)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
