#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import geopandas as gpd


sys.path.insert(0, str(Path(__file__).resolve().parent / "Source"))

CHANNEL_MAP = {"u": 0, "v": 1, "temp": 2, "dewpoint": 3}
CHANNEL_ORDER = ["u", "v", "dewpoint", "temp"]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
        if module == "shapely.io" and name == "from_wkb":
            return lambda *args, **kwargs: None
        # Avoid importing fragile scipy classes during unpickling.
        if module.startswith("scipy") or module.startswith("torch"):
            return _DummyMadisNetwork
        return super().find_class(module, name)


def load_madis_network(path):
    with open(path, "rb") as f:
        return _MadisNetworkSafeUnpickler(f).load()


def load_station_coords(station_file):
    stations = gpd.read_file(station_file)
    if "geometry" not in stations.columns:
        raise ValueError(f"Missing geometry column in {station_file}")
    lons = np.asarray([geom.x for geom in stations.geometry], dtype=np.float32)
    lats = np.asarray([geom.y for geom in stations.geometry], dtype=np.float32)
    return lons, lats


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


def to_time_station_feature(arr, n_stations_hint=None):
    arr = np.asarray(arr)

    # Common shapes:
    # [time, station, feature]
    # [time, batch, station, feature]
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


def resolve_output_file(input_dir, prefix, metric, experiment_tag, node_scope="all"):
    input_dir = Path(input_dir)
    candidate_prefixes = []
    if node_scope in {"seen", "ghost"}:
        candidate_prefixes.append(f"{prefix}_{node_scope}")
    candidate_prefixes.append(prefix)

    for candidate_prefix in candidate_prefixes:
        if experiment_tag:
            tagged = input_dir / f"{candidate_prefix}_{metric}_{experiment_tag}_min.pkl"
            if tagged.exists():
                return tagged
        legacy = input_dir / f"{candidate_prefix}_{metric}_min.pkl"
        if legacy.exists():
            return legacy

    searched = ", ".join(candidate_prefixes)
    raise FileNotFoundError(f"Missing file for prefix={prefix}, metric={metric}, tag={experiment_tag}, scopes={searched}")


def make_gif_file(y, p, lons, lats, roi, channel_name, scope_name, time_start, frame_step, fps, dpi, out_path, diff_limit=None):
    if len(lons) != y.shape[1]:
        raise ValueError(f"Station count mismatch: coords={len(lons)}, data={y.shape[1]}")

    pred_vmin = np.nanmin(p)
    pred_vmax = np.nanmax(p)
    abs_error = np.abs(p - y)
    if diff_limit is None:
        diff_limit = np.nanmax(abs_error)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    def draw_panel(ax, values, title, cmap="coolwarm", vmin_value=None, vmax_value=None):
        if roi is not None:
            roi.boundary.plot(ax=ax, color="0.35", linewidth=1.0, zorder=0)
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=35,
            cmap=cmap,
            vmin=vmin_value,
            vmax=vmax_value,
            edgecolors="black",
            linewidths=0.2,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lons.min() - 0.5, lons.max() + 0.5)
        ax.set_ylim(lats.min() - 0.5, lats.max() + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        return sc

    sc_pred = draw_panel(axes[0], p[0], f"Pred ({scope_name})", cmap="coolwarm", vmin_value=pred_vmin, vmax_value=pred_vmax)
    sc_diff = draw_panel(axes[1], abs_error[0], "Abs Error", cmap="gray_r", vmin_value=0, vmax_value=diff_limit)

    cbar0 = fig.colorbar(sc_pred, ax=axes[0], shrink=0.9)
    cbar0.set_label(channel_name)
    cbar1 = fig.colorbar(sc_diff, ax=axes[1], shrink=0.9)
    cbar1.set_label(f"{channel_name} abs error")

    time_text = fig.suptitle(f"{channel_name} | {scope_name} | sampled frame 1/{len(y)} (step={frame_step})", y=1.02)

    def update(frame_idx):
        sc_pred.set_array(p[frame_idx])
        sc_diff.set_array(np.abs(p[frame_idx] - y[frame_idx]))
        time_text.set_text(
            f"{channel_name} | {scope_name} | sampled frame {frame_idx + 1}/{len(y)} (step={frame_step})"
        )
        return sc_pred, sc_diff, time_text

    anim = FuncAnimation(fig, update, frames=len(y), interval=1000 // max(fps, 1), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    print(f"saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='/projects3/home/flag0220/LocalizedWeather/WindDataNE-US/ModelOutputs/best_ghost_100', type=str)
    parser.add_argument("--metric", type=str, default="CUSTOM")
    parser.add_argument("--experiment_tag", type=str, default="")
    parser.add_argument("--channel", type=str, default=None, choices=["u", "v", "temp", "dewpoint"])
    parser.add_argument("--output", type=str, default="pred_error_map.gif")
    parser.add_argument("--holdout_output", type=str, default="pred_error_map_holdout.gif")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--time_start", type=int, default=0)
    parser.add_argument("--time_end", type=int, default=128)
    parser.add_argument("--shapefile_path", type=str, default=None)
    args = parser.parse_args()

    channels = [args.channel] if args.channel is not None else CHANNEL_ORDER

    input_dir = Path(args.input_dir)
    data_root = input_dir.parents[1]
    shapefile_path = Path(args.shapefile_path) if args.shapefile_path is not None else data_root / "Shapefiles/Regions/northeastern_buffered.shp"
    if not shapefile_path.exists():
        shapefile_path = Path(__file__).resolve().parent / "Shapefiles/Regions/northeastern_buffered.shp"

    targets = load_pickle(resolve_output_file(input_dir, "Targets", args.metric, args.experiment_tag))
    preds = load_pickle(resolve_output_file(input_dir, "Preds", args.metric, args.experiment_tag))
    madis_network = load_madis_network(input_dir / "madis_network.pkl")

    y_all = to_time_station_feature(targets)
    p_all = to_time_station_feature(preds)

    if y_all.shape != p_all.shape:
        raise ValueError(f"Shape mismatch: y={y_all.shape}, p={p_all.shape}")

    if args.time_end is not None:
        y_all = y_all[args.time_start:args.time_end]
        p_all = p_all[args.time_start:args.time_end]
    else:
        y_all = y_all[args.time_start:]
        p_all = p_all[args.time_start:]

    frame_indices = np.arange(0, len(y_all), max(1, args.frame_step))
    y_all = y_all[frame_indices]
    p_all = p_all[frame_indices]

    lons = np.asarray(madis_network.stat_lons)
    lats = np.asarray(madis_network.stat_lats)
    if shapefile_path.exists():
        roi = gpd.read_file(shapefile_path).dissolve()
    else:
        roi = None

    split_path = input_dir / "station_split.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        ghost_indices = split.get("ghost_indices", [])
        if len(ghost_indices) > 0:
            ghost_indices = np.asarray(ghost_indices, dtype=int)
        else:
            ghost_indices = None
    else:
        ghost_indices = None

    for channel_name in channels:
        channel_idx = CHANNEL_MAP[channel_name]
        y = y_all[..., channel_idx]
        p = p_all[..., channel_idx]

        if ghost_indices is not None:
            y_ghost = y[:, ghost_indices]
            p_ghost = p[:, ghost_indices]
            lons_ghost = lons[ghost_indices]
            lats_ghost = lats[ghost_indices]
        else:
            y_ghost = None
            p_ghost = None
            lons_ghost = None
            lats_ghost = None

        if len(channels) == 1:
            output_path = input_dir / args.output
            holdout_output_path = input_dir / args.holdout_output
        else:
            output_path = input_dir / f"{channel_name}_{Path(args.output).stem}{Path(args.output).suffix}"
            holdout_output_path = input_dir / f"{channel_name}_{Path(args.holdout_output).stem}{Path(args.holdout_output).suffix}"

        make_gif_file(
            y,
            p,
            lons,
            lats,
            roi,
            channel_name,
            "all stations",
            args.time_start,
            args.frame_step,
            args.fps,
            args.dpi,
            output_path,
        )

        if ghost_indices is not None:
            make_gif_file(
                y_ghost,
                p_ghost,
                lons_ghost,
                lats_ghost,
                roi,
                channel_name,
                "ghost stations",
                args.time_start,
                args.frame_step,
                args.fps,
                args.dpi,
                holdout_output_path,
            )
        else:
            print(f"station_split.json not found or missing ghost_indices in {input_dir}. Skipping holdout GIF for {channel_name}.")


if __name__ == "__main__":
    main()