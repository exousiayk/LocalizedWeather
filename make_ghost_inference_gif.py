#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Point
from torch.utils.data import ConcatDataset, DataLoader


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "Source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from Dataloader.MetaStation import MetaStation
from Network.MadisNetwork import MadisNetwork
from Normalization import NormalizerBuilder
from Normalization.Normalizers import MinMaxNormalizer
from Settings.Settings import (
    EnvVariables,
    GhostInitMode,
    InterpolationType,
    LossFunctionType,
    ModelType,
    NetworkConstructionMethod,
)
from Main import Main


CHANNEL_MAP = {"u": 0, "v": 1, "temp": 2, "dewpoint": 3}
CHANNEL_ORDER = ["u", "v", "dewpoint", "temp"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def convert_params(params, data_path, output_dir):
    args = SimpleNamespace(**params)
    args.data_path = str(data_path)
    try:
        args.output_saving_path = str(output_dir.relative_to(data_path))
    except ValueError:
        args.output_saving_path = os.path.relpath(output_dir, data_path)

    args.model_type = ModelType(params["model_type"])
    args.network_construction_method = NetworkConstructionMethod(params["network_construction_method"])
    args.loss_function_type = LossFunctionType(params["loss_function_type"])
    args.ghost_init_mode = GhostInitMode(params["ghost_init_mode"])
    args.interpolation_type = InterpolationType(params.get("interpolation_type", -1))
    args.madis_vars_i = [EnvVariables(value) for value in params["madis_vars_i"]]
    args.madis_vars_o = [EnvVariables(value) for value in params["madis_vars_o"]]
    args.external_vars = [EnvVariables(value) for value in params["external_vars"]]
    args.past_only = bool(params.get("past_only", False))
    args.use_wb = bool(params.get("use_wb", 0))
    args.show_progress_bar = bool(params.get("show_progress_bar", False))
    return args


def build_all_seen_split(n_stations):
    seen_mask = np.ones(n_stations, dtype=bool)
    ghost_mask = np.zeros(n_stations, dtype=bool)
    return {
        "n_stations": int(n_stations),
        "ghost_holdout_ratio": 0.0,
        "ghost_split_seed": 0,
        "ghost_indices": np.array([], dtype=np.int64),
        "seen_indices": np.arange(n_stations, dtype=np.int64),
        "ghost_station_mask": ghost_mask,
        "seen_station_mask": seen_mask,
    }


def load_model_state(model, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model.load_state_dict(stripped)


def get_roi_geometry(shapefile_path):
    if shapefile_path is None:
        return None

    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        return None

    roi = gpd.read_file(shapefile_path).dissolve()
    if len(roi) == 0:
        return None

    return roi.geometry.iloc[0]


def sample_random_points(roi_geometry, count, rng, fallback_bounds):
    lon_low, lat_low, lon_up, lat_up = fallback_bounds
    if roi_geometry is not None:
        lon_low, lat_low, lon_up, lat_up = roi_geometry.bounds

    points = []
    while len(points) < count:
        batch_size = max(32, (count - len(points)) * 4)
        cand_lons = rng.uniform(lon_low, lon_up, size=batch_size)
        cand_lats = rng.uniform(lat_low, lat_up, size=batch_size)
        if roi_geometry is None:
            points.extend(zip(cand_lons.tolist(), cand_lats.tolist()))
            continue

        for lon, lat in zip(cand_lons, cand_lats):
            if roi_geometry.contains(Point(float(lon), float(lat))):
                points.append((float(lon), float(lat)))
            if len(points) >= count:
                break

    return np.asarray(points[:count], dtype=np.float32)


def filter_points_near_stations(points, real_lons, real_lats, min_distance_deg):
    if points is None or len(points) == 0:
        return np.asarray(points, dtype=np.float32), 0
    if min_distance_deg is None or float(min_distance_deg) <= 0:
        return np.asarray(points, dtype=np.float32), 0

    real_points = np.column_stack([real_lons, real_lats]).astype(np.float32)
    candidate_points = np.asarray(points, dtype=np.float32)
    diff = candidate_points[:, None, :] - real_points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    keep_mask = np.min(dist2, axis=1) > float(min_distance_deg) ** 2
    filtered = candidate_points[keep_mask]
    removed = int(len(candidate_points) - len(filtered))
    return filtered, removed


def build_grid_points(roi_geometry, count, fallback_bounds, real_lons=None, real_lats=None,
                     spacing_deg=None, station_exclusion_radius_deg=None):
    lon_low, lat_low, lon_up, lat_up = fallback_bounds
    if roi_geometry is not None:
        lon_low, lat_low, lon_up, lat_up = roi_geometry.bounds

    if spacing_deg is not None:
        spacing_deg = float(spacing_deg)
        if spacing_deg <= 0:
            raise ValueError("grid spacing must be positive")

        lon_values = np.arange(lon_low, lon_up + spacing_deg * 0.5, spacing_deg, dtype=np.float32)
        lat_values = np.arange(lat_low, lat_up + spacing_deg * 0.5, spacing_deg, dtype=np.float32)
        if len(lon_values) == 0:
            lon_values = np.asarray([lon_low], dtype=np.float32)
        if len(lat_values) == 0:
            lat_values = np.asarray([lat_low], dtype=np.float32)

        grid_lon, grid_lat = np.meshgrid(lon_values, lat_values)
        points = np.column_stack([grid_lon.reshape(-1), grid_lat.reshape(-1)]).astype(np.float32)

        if roi_geometry is not None:
            mask = np.fromiter(
                (
                    roi_geometry.contains(Point(float(lon), float(lat)))
                    or roi_geometry.touches(Point(float(lon), float(lat)))
                    for lon, lat in points
                ),
                dtype=bool,
                count=len(points),
            )
            points = points[mask]

        points, removed_by_station = filter_points_near_stations(
            points,
            real_lons,
            real_lats,
            station_exclusion_radius_deg,
        )

        if len(points) == 0:
            raise ValueError("Unable to generate any grid sensor points with the requested spacing.")

        return np.asarray(points, dtype=np.float32), {
            "grid_rows": int(len(lat_values)),
            "grid_cols": int(len(lon_values)),
            "grid_candidates": int(len(points)),
            "grid_spacing_deg": float(spacing_deg),
            "station_exclusion_radius_deg": float(station_exclusion_radius_deg) if station_exclusion_radius_deg is not None else None,
            "station_exclusion_removed": int(removed_by_station),
        }

    lon_span = max(float(lon_up - lon_low), 1e-6)
    lat_span = max(float(lat_up - lat_low), 1e-6)
    aspect_ratio = lon_span / lat_span

    n_cols = max(1, int(np.ceil(np.sqrt(count * aspect_ratio))))
    n_rows = max(1, int(np.ceil(count / n_cols)))

    def make_points(rows, cols):
        lon_values = np.linspace(lon_low, lon_up, cols, dtype=np.float32)
        lat_values = np.linspace(lat_low, lat_up, rows, dtype=np.float32)
        grid_lon, grid_lat = np.meshgrid(lon_values, lat_values)
        points = np.column_stack([grid_lon.reshape(-1), grid_lat.reshape(-1)]).astype(np.float32)

        if roi_geometry is not None:
            mask = np.fromiter(
                (
                    roi_geometry.contains(Point(float(lon), float(lat)))
                    or roi_geometry.touches(Point(float(lon), float(lat)))
                    for lon, lat in points
                ),
                dtype=bool,
                count=len(points),
            )
            points = points[mask]

        points, removed_by_station = filter_points_near_stations(
            points,
            real_lons,
            real_lats,
            station_exclusion_radius_deg,
        )

        if len(points) > 0:
            order = np.lexsort((points[:, 0], points[:, 1]))
            points = points[order]

        return points, removed_by_station

    grid_points, removed_total = make_points(n_rows, n_cols)
    expansion_steps = 0
    while len(grid_points) < count and expansion_steps < 8:
        expansion_steps += 1
        n_cols = max(n_cols + 1, int(np.ceil(n_cols * 1.25)))
        n_rows = max(n_rows + 1, int(np.ceil(n_rows * 1.25)))
        grid_points, removed_by_station = make_points(n_rows, n_cols)
        removed_total += removed_by_station

    if len(grid_points) == 0:
        raise ValueError("Unable to generate grid sensor points within the requested bounds.")

    return np.asarray(grid_points[:count], dtype=np.float32), {
        "grid_rows": int(n_rows),
        "grid_cols": int(n_cols),
        "grid_candidates": int(len(grid_points)),
        "grid_spacing_deg": None,
        "station_exclusion_radius_deg": float(station_exclusion_radius_deg) if station_exclusion_radius_deg is not None else None,
        "station_exclusion_removed": int(removed_total),
    }


def build_augmented_station_geometries(real_lons, real_lats, ghost_lons, ghost_lats):
    all_lons = np.concatenate([real_lons, ghost_lons]).astype(np.float32)
    all_lats = np.concatenate([real_lats, ghost_lats]).astype(np.float32)
    stations = gpd.GeoDataFrame(geometry=gpd.points_from_xy(all_lons, all_lats), crs="EPSG:4326")
    return stations, all_lons, all_lats


def precompute_ghost_neighbors(real_lons, real_lats, ghost_lons, ghost_lats, k=4):
    real = np.column_stack([real_lons, real_lats]).astype(np.float32)
    ghost = np.column_stack([ghost_lons, ghost_lats]).astype(np.float32)

    if ghost.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)

    diff = ghost[:, None, :] - real[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    nn_idx = np.argsort(dist2, axis=1)[:, : min(k, real.shape[0])]
    nn_dist = np.take_along_axis(dist2, nn_idx, axis=1)
    weights = 1.0 / np.maximum(np.sqrt(nn_dist), 1e-6)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return nn_idx.astype(np.int64), weights.astype(np.float32)


def move_sample_to_device(sample, device):
    moved = {}
    for key, value in sample.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def get_sample_field(sample, field, fallback_name=None):
    if field in sample:
        return sample[field]
    if fallback_name is not None and fallback_name in sample:
        return sample[fallback_name]
    raise KeyError(f"Missing sample field for {field!r}")


def build_encoded_history(sample, madis_vars_i, madis_norm_dict, lead_hrs, past_only, device):
    encoded_vars = []
    for madis_var in madis_vars_i:
        value = get_sample_field(sample, madis_var, madis_var.name).to(device).float()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        encoded = madis_norm_dict[madis_var].encode(value).unsqueeze(-1)
        if not past_only:
            encoded = encoded[:, :, : encoded.shape[2] - lead_hrs, :]
        encoded_vars.append(encoded)

    return torch.cat(encoded_vars, dim=-1)


def build_external_tensor(time_sel, year, external_data_object, external_vars, external_norm_dict, device):
    if external_norm_dict is None or external_data_object is None or len(external_vars) == 0:
        return None

    time_sel = pd.to_datetime(np.asarray(time_sel).reshape(-1).astype(np.int64))

    encoded_vars = []
    for external_var in external_vars:
        value = external_data_object.getSample(time_sel, external_var.name, None, len(time_sel), 0)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.unsqueeze(0).to(device).float()
        encoded = external_norm_dict[external_var].encode(value).unsqueeze(-1)
        encoded_vars.append(encoded)

    return torch.cat(encoded_vars, dim=-1)


def build_seen_target(sample, madis_vars_o, past_only, device):
    targets = []
    for madis_var in madis_vars_o:
        if past_only:
            value = get_sample_field(sample, f"target_{madis_var.name}", f"target_{madis_var.name}").to(device).float()
            if value.ndim == 1:
                value = value.unsqueeze(0)
        else:
            value = get_sample_field(sample, madis_var, madis_var.name).to(device).float()
            if value.ndim == 2:
                value = value.unsqueeze(0)
            value = value[:, :, -1]
        targets.append(value.unsqueeze(-1))

    return torch.cat(targets, dim=-1)


def interpolate_ghost_features(real_history, neighbor_idx, neighbor_weights):
    if neighbor_idx.size == 0:
        return real_history[:, :0, :, :]

    ghost_chunks = []
    for row_idx, row_weights in zip(neighbor_idx, neighbor_weights):
        neighbor_features = real_history[:, row_idx, :, :]
        weight_tensor = torch.as_tensor(row_weights, dtype=real_history.dtype, device=real_history.device)
        ghost_value = (neighbor_features * weight_tensor.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        ghost_chunks.append(ghost_value)

    return torch.cat(ghost_chunks, dim=1)


def decode_predictions(prediction, madis_vars_o, madis_norm_dict):
    decoded = prediction.clone()
    for channel_idx, madis_var in enumerate(madis_vars_o):
        decoded[..., channel_idx] = madis_norm_dict[madis_var].decode(decoded[..., channel_idx])
    return decoded


def summarize_channel(pred, target):
    error = pred - target
    abs_error = np.abs(error)
    return {
        "mae": float(np.nanmean(abs_error)),
        "rmse": float(np.sqrt(np.nanmean(error ** 2))),
        "bias": float(np.nanmean(error)),
        "max_abs_error": float(np.nanmax(abs_error)),
    }


def normalize_gif_output_path(output_path):
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".gif":
        return output_path

    clean_name = output_path.name.rstrip(".")
    if not clean_name:
        clean_name = "output"
    return output_path.with_name(f"{clean_name}.gif")


def render_gif(frames, channel_name, roi_geometry, real_lons, real_lats, ghost_lons, ghost_lats,
               output_path, fps, dpi, ratio, n_ghost, extra_mode):
    if len(frames) == 0:
        raise ValueError(f"No frames were collected for {channel_name}")

    output_path = normalize_gif_output_path(output_path)

    pred_samples = [frame["pred_seen"] for frame in frames]
    if n_ghost > 0:
        pred_samples.extend(frame["pred_ghost"] for frame in frames)
    pred_values = np.concatenate([values.reshape(-1) for values in pred_samples])
    gt_values = np.concatenate([frame["target_seen"].reshape(-1) for frame in frames])

    shared_vmin = float(np.nanmin(np.concatenate([gt_values, pred_values])))
    shared_vmax = float(np.nanmax(np.concatenate([gt_values, pred_values])))

    all_lons = np.concatenate([real_lons, ghost_lons]) if len(ghost_lons) > 0 else real_lons
    all_lats = np.concatenate([real_lats, ghost_lats]) if len(ghost_lats) > 0 else real_lats

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    def setup_panel(ax, title):
        if roi_geometry is not None:
            gpd.GeoSeries([roi_geometry]).boundary.plot(ax=ax, color="0.35", linewidth=1.0, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(all_lons.min() - 0.5, all_lons.max() + 0.5)
        ax.set_ylim(all_lats.min() - 0.5, all_lats.max() + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    setup_panel(axes[0], f"Pred (seen only) | {channel_name}")
    setup_panel(axes[1], f"Pred (ghost only) | {channel_name}")
    setup_panel(axes[2], f"GT seen + Pred ghost | {channel_name}")

    gt_real = axes[0].scatter(
        real_lons,
        real_lats,
        c=frames[0]["pred_seen"].reshape(-1),
        s=35,
        cmap="coolwarm",
        vmin=shared_vmin,
        vmax=shared_vmax,
        edgecolors="black",
        linewidths=0.2,
    )

    ghost_only = axes[1].scatter(
        ghost_lons,
        ghost_lats,
        c=frames[0]["pred_ghost"].reshape(-1),
        s=35,
        cmap="coolwarm",
        vmin=shared_vmin,
        vmax=shared_vmax,
        edgecolors="black",
        linewidths=0.2,
    )

    pred_real = axes[2].scatter(
        real_lons,
        real_lats,
        c=frames[0]["pred_seen"].reshape(-1),
        s=35,
        cmap="coolwarm",
        vmin=shared_vmin,
        vmax=shared_vmax,
        edgecolors="black",
        linewidths=0.2,
    )
    pred_ghost = axes[2].scatter(
        ghost_lons,
        ghost_lats,
        c=frames[0]["pred_ghost"].reshape(-1),
        s=42,
        cmap="coolwarm",
        vmin=shared_vmin,
        vmax=shared_vmax,
        edgecolors="black",
        linewidths=0.2,
    )

    cbar0 = fig.colorbar(gt_real, ax=axes[0], shrink=0.9)
    cbar0.set_label(channel_name)
    cbar1 = fig.colorbar(ghost_only, ax=axes[1], shrink=0.9)
    cbar1.set_label(channel_name)
    cbar2 = fig.colorbar(pred_real, ax=axes[2], shrink=0.9)
    cbar2.set_label(channel_name)

    time_text = fig.suptitle(
        f"{channel_name} | ghost ratio={ratio:.2f} | added sensors={n_ghost} | mode={extra_mode} | frame 1/{len(frames)}",
        y=1.02,
    )

    def update(frame_idx):
        frame = frames[frame_idx]
        gt_real.set_array(frame["pred_seen"].reshape(-1))
        pred_real.set_array(frame["pred_seen"].reshape(-1))
        pred_ghost.set_array(frame["pred_ghost"].reshape(-1))
        time_text.set_text(
            f"{channel_name} | ghost ratio={ratio:.2f} | added sensors={n_ghost} | mode={extra_mode} | "
            f"frame {frame_idx + 1}/{len(frames)} | {frame['time_label']}"
        )
        ghost_only.set_array(frame["pred_ghost"].reshape(-1))
        return gt_real, pred_real, pred_ghost, ghost_only, time_text

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // max(fps, 1), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved: {output_path}")


def collect_frames(loader, model, device, madis_vars_i, madis_vars_o, external_vars, madis_norm_dict,
                   external_norm_dict, external_data_objects, external_network, real_station_count, ghost_count,
                   ghost_neighbor_idx, ghost_neighbor_weights, ghost_lons_enc, ghost_lats_enc,
                   aug_internal_edge_index, past_only, lead_hrs, max_frames, frame_start, frame_end, frame_step):
    frames_by_channel = {channel: [] for channel in CHANNEL_ORDER}
    metrics_by_channel = {channel: [] for channel in CHANNEL_ORDER}

    model.eval()
    with torch.no_grad():
        for sample_idx, sample in enumerate(loader):
            if sample_idx < frame_start:
                continue
            if frame_end is not None and sample_idx >= frame_end:
                break
            if (sample_idx - frame_start) % max(frame_step, 1) != 0:
                continue

            sample = move_sample_to_device(sample, device)
            time_values = sample["time"].detach().cpu().numpy()
            sample_year = pd.to_datetime(np.asarray(time_values).reshape(-1)[0]).year

            real_history = build_encoded_history(sample, madis_vars_i, madis_norm_dict, lead_hrs, past_only, device)
            ghost_history = interpolate_ghost_features(real_history, ghost_neighbor_idx, ghost_neighbor_weights)
            madis_x = torch.cat([real_history, ghost_history], dim=1)

            madis_lon = sample["madis_lon"].float()
            madis_lat = sample["madis_lat"].float()
            if madis_lon.ndim == 2:
                madis_lon = madis_lon.unsqueeze(0)
            if madis_lat.ndim == 2:
                madis_lat = madis_lat.unsqueeze(0)
            madis_lon = torch.cat([madis_lon, ghost_lons_enc], dim=1)
            madis_lat = torch.cat([madis_lat, ghost_lats_enc], dim=1)

            external_data_object = external_data_objects.get(sample_year) if external_data_objects is not None else None
            external_x = build_external_tensor(time_values, sample_year, external_data_object, external_vars,
                                               external_norm_dict, device)
            edge_index_m2m = aug_internal_edge_index.to(device)
            edge_index_e2m = external_network.ex2m_edge_index.to(device) if external_x is not None else None
            external_lon = None
            external_lat = None
            if external_network is not None:
                external_lon = external_network.lons.to(device).float().view(1, -1)
                external_lat = external_network.lats.to(device).float().view(1, -1)

            model_output = model(
                madis_x,
                madis_lon,
                madis_lat,
                edge_index_m2m,
                external_lon,
                external_lat,
                external_x,
                edge_index_e2m,
            )
            prediction = model_output[0] if isinstance(model_output, tuple) else model_output

            prediction = decode_predictions(prediction, madis_vars_o, madis_norm_dict).detach().cpu().numpy()[0]
            target_seen = build_seen_target(sample, madis_vars_o, past_only, device).detach().cpu().numpy()[0]

            time_label = pd.to_datetime(time_values[0, -1]).strftime("%Y-%m-%d %H:%M")

            pred_seen = prediction[:real_station_count]
            pred_ghost = prediction[real_station_count:] if ghost_count > 0 else np.zeros((0, prediction.shape[-1]), dtype=prediction.dtype)

            for channel_name in CHANNEL_ORDER:
                channel_idx = CHANNEL_MAP[channel_name]
                pred_channel_seen = pred_seen[:, channel_idx]
                pred_channel_ghost = pred_ghost[:, channel_idx] if ghost_count > 0 else np.zeros((0,), dtype=prediction.dtype)
                target_channel_seen = target_seen[:, channel_idx]
                abs_error_seen = np.abs(pred_channel_seen - target_channel_seen)

                frames_by_channel[channel_name].append(
                    {
                        "pred_seen": np.asarray(pred_channel_seen).reshape(-1),
                        "pred_ghost": np.asarray(pred_channel_ghost).reshape(-1),
                        "target_seen": np.asarray(target_channel_seen).reshape(-1),
                        "abs_error_seen": np.asarray(abs_error_seen).reshape(-1),
                        "time_label": time_label,
                    }
                )
                metrics_by_channel[channel_name].append(summarize_channel(pred_channel_seen, target_channel_seen))

            if max_frames is not None and len(frames_by_channel[CHANNEL_ORDER[0]]) >= max_frames:
                break

    return frames_by_channel, metrics_by_channel


def main():
    parser = argparse.ArgumentParser(description="Inference GIF with random ghost nodes")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "WindDataNE-US/ModelOutputs/exp2/best_ghost_100"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metric", type=str, default="CUSTOM")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--channel", type=str, default=None, choices=["u", "v", "temp", "dewpoint"])
    parser.add_argument("--output", type=str, default="ghost_pred_error_map.gif")
    parser.add_argument("--extra_ratio", type=float, default=0.2)
    parser.add_argument("--extra_count", type=int, default=None)
    parser.add_argument("--extra_mode", type=str, choices=["grid", "random"], default="grid")
    parser.add_argument("--grid_spacing_deg", type=float, default=None)
    parser.add_argument("--station_exclusion_radius_deg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=72)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--shapefile_path", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    params_path = output_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json in {output_dir}")

    params = load_json(params_path)
    data_path = Path(params.get("data_path", output_dir.parents[2])).resolve()
    runtime_args = convert_params(params, data_path, output_dir)
    if args.shapefile_path is not None:
        runtime_args.shapefile_path = args.shapefile_path

    main_runtime = Main(runtime_args)
    if main_runtime.model_type != ModelType.GNN:
        raise NotImplementedError("Random ghost-node augmentation is currently supported for GNN checkpoints only.")

    real_meta_station = MetaStation(
        main_runtime.lat_low,
        main_runtime.lat_up,
        main_runtime.lon_low,
        main_runtime.lon_up,
        main_runtime.n_years,
        main_runtime.madis_control_ratio,
        shapefile_path=main_runtime.shapefile_path,
        data_path=main_runtime.data_path,
    )
    real_madis_network = main_runtime.BuildMadisNetwork(
        real_meta_station,
        main_runtime.n_neighbors_m2m,
        main_runtime.network_construction_method,
    )
    main_runtime.madis_network = real_madis_network

    shapefile_path = Path(args.shapefile_path) if args.shapefile_path else main_runtime.shapefile_path
    if shapefile_path is not None and not Path(shapefile_path).is_absolute():
        shapefile_path = data_path / shapefile_path
    roi_geometry = get_roi_geometry(shapefile_path)

    real_lons = np.asarray(real_madis_network.stat_lons, dtype=np.float32)
    real_lats = np.asarray(real_madis_network.stat_lats, dtype=np.float32)
    n_real = len(real_lons)

    if args.extra_count is None:
        requested_extra_count = max(1, int(round(n_real * args.extra_ratio)))
    else:
        requested_extra_count = max(1, int(args.extra_count))

    rng = np.random.default_rng(args.seed)
    fallback_bounds = (main_runtime.lon_low, main_runtime.lat_low, main_runtime.lon_up, main_runtime.lat_up)
    if args.extra_mode == "random":
        ghost_points = sample_random_points(roi_geometry, requested_extra_count, rng, fallback_bounds)
        ghost_points, removed_by_station = filter_points_near_stations(
            ghost_points,
            real_lons,
            real_lats,
            args.station_exclusion_radius_deg,
        )
        layout_info = {
            "grid_rows": None,
            "grid_cols": None,
            "grid_candidates": None,
            "grid_spacing_deg": None,
            "station_exclusion_radius_deg": float(args.station_exclusion_radius_deg),
            "station_exclusion_removed": int(removed_by_station),
        }
    else:
        ghost_points, layout_info = build_grid_points(
            roi_geometry,
            requested_extra_count,
            fallback_bounds,
            real_lons=real_lons,
            real_lats=real_lats,
            spacing_deg=args.grid_spacing_deg,
            station_exclusion_radius_deg=args.station_exclusion_radius_deg,
        )

    n_ghost = int(len(ghost_points))
    ghost_lons = ghost_points[:, 0]
    ghost_lats = ghost_points[:, 1]
    ghost_nn_idx, ghost_nn_weights = precompute_ghost_neighbors(real_lons, real_lats, ghost_lons, ghost_lats, k=4)

    aug_stations, _, _ = build_augmented_station_geometries(real_lons, real_lats, ghost_lons, ghost_lats)
    aug_network_stub = SimpleNamespace(stations=aug_stations)
    aug_internal_network = MadisNetwork(
        aug_network_stub,
        main_runtime.n_neighbors_m2m,
        main_runtime.network_construction_method,
    )

    aug_meta_station = copy.copy(real_meta_station)
    aug_meta_station.filtered_file_name = (
        f"{real_meta_station.filtered_file_name}_ghostextra_{n_ghost}_{args.extra_mode}"
        f"_spacing_{args.grid_spacing_deg if args.grid_spacing_deg is not None else 'auto'}"
        f"_seed_{args.seed}"
    )

    years = list(range(2024 - main_runtime.n_years, 2024))
    external_data_objects, external_len, external_network, _ = main_runtime.BuildExternalNetwork(
        main_runtime.data_path,
        main_runtime.external_len,
        main_runtime.hrrr_analysis_only,
        main_runtime.interpolation_type,
        main_runtime.lead_hrs,
        aug_internal_network,
        aug_meta_station,
        main_runtime.n_neighbors_e2m,
        main_runtime.n_neighbors_h2m,
        main_runtime.whole_len,
        years,
        main_runtime.past_only,
    )

    if external_data_objects is not None:
        for year in years:
            external_data_object = external_data_objects.get(year)
            if external_data_object is None:
                continue
            if getattr(external_data_object, "data", None) is None:
                continue
            if hasattr(external_data_object, "LoadDataToMemory"):
                external_data_object.LoadDataToMemory()

    all_seen_split = build_all_seen_split(n_real)
    data_list = main_runtime.GetDataList(
        main_runtime.back_hrs,
        main_runtime.data_path,
        None,
        None,
        main_runtime.external_vars,
        main_runtime.lead_hrs,
        main_runtime.madis_vars,
        real_meta_station,
        years,
        all_seen_split,
        GhostInitMode.ZERO,
        False,
        0.0,
        main_runtime.past_only,
    )

    madis_norm_dict = load_pickle(output_dir / "madis_norm_dict.pkl")
    external_norm_dict = load_pickle(output_dir / "external_norm_dict.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = main_runtime.GetModel(
        main_runtime.Madis_len,
        external_len,
        external_data_objects is not None,
        main_runtime.external_vars,
        main_runtime.hidden_dim,
        main_runtime.lead_hrs,
        main_runtime.madis_vars_i,
        main_runtime.madis_vars_o,
        main_runtime.model_type,
        main_runtime.n_passing,
        n_real + n_ghost,
        aug_internal_network,
    ).to(device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else output_dir / f"model_{args.metric}_min.pt"
    load_model_state(model, checkpoint_path, device)
    model.eval()

    loaders = main_runtime.CreateDataLoaders(data_list, batch_size=1, n_years=main_runtime.n_years)
    if args.split == "all":
        dataset = ConcatDataset(data_list)
    else:
        dataset = loaders[args.split].dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    coord_lon_norm = MinMaxNormalizer(
        external_data_objects[years[0]].lon_low if external_data_objects is not None else real_meta_station.lon_low,
        external_data_objects[years[0]].lon_up if external_data_objects is not None else real_meta_station.lon_up,
    )
    coord_lat_norm = MinMaxNormalizer(
        external_data_objects[years[0]].lat_low if external_data_objects is not None else real_meta_station.lat_low,
        external_data_objects[years[0]].lat_up if external_data_objects is not None else real_meta_station.lat_up,
    )
    ghost_lons_enc = coord_lon_norm.encode(torch.as_tensor(ghost_lons, dtype=torch.float32)).view(1, -1, 1).to(device)
    ghost_lats_enc = coord_lat_norm.encode(torch.as_tensor(ghost_lats, dtype=torch.float32)).view(1, -1, 1).to(device)

    frames_by_channel, metrics_by_channel = collect_frames(
        loader,
        model,
        device,
        main_runtime.madis_vars_i,
        main_runtime.madis_vars_o,
        main_runtime.external_vars,
        madis_norm_dict,
        external_norm_dict,
        external_data_objects,
        external_network,
        n_real,
        n_ghost,
        ghost_nn_idx,
        ghost_nn_weights,
        ghost_lons_enc,
        ghost_lats_enc,
        aug_internal_network.k_edge_index,
        main_runtime.past_only,
        main_runtime.lead_hrs,
        args.max_frames,
        args.frame_start,
        args.frame_end,
        args.frame_step,
    )

    channels = [args.channel] if args.channel is not None else CHANNEL_ORDER
    output_dir.mkdir(exist_ok=True, parents=True)

    summary = {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "extra_mode": args.extra_mode,
        "extra_ratio": float(args.extra_ratio),
        "requested_extra_count": int(requested_extra_count),
        "extra_count": int(n_ghost),
        "real_nodes": int(n_real),
        "ghost_nodes": int(n_ghost),
        "seed": int(args.seed),
        "grid_rows": layout_info["grid_rows"],
        "grid_cols": layout_info["grid_cols"],
        "grid_candidates": layout_info["grid_candidates"],
        "grid_spacing_deg": layout_info["grid_spacing_deg"],
        "station_exclusion_radius_deg": layout_info["station_exclusion_radius_deg"],
        "station_exclusion_removed": layout_info["station_exclusion_removed"],
        "channels": {},
    }

    for channel_name in channels:
        frames = frames_by_channel[channel_name]
        channel_metrics = metrics_by_channel[channel_name]
        summary["channels"][channel_name] = {
            "frames": len(frames),
            "mean_mae": float(np.mean([entry["mae"] for entry in channel_metrics])) if channel_metrics else np.nan,
            "mean_rmse": float(np.mean([entry["rmse"] for entry in channel_metrics])) if channel_metrics else np.nan,
            "mean_bias": float(np.mean([entry["bias"] for entry in channel_metrics])) if channel_metrics else np.nan,
            "max_abs_error": float(np.max([entry["max_abs_error"] for entry in channel_metrics])) if channel_metrics else np.nan,
        }

        if args.channel is not None:
            gif_name = args.output
        else:
            gif_name = f"{channel_name}_{Path(args.output).stem}{Path(args.output).suffix}"

        render_gif(
            frames,
            channel_name,
            roi_geometry,
            real_lons,
            real_lats,
            ghost_lons,
            ghost_lats,
            output_dir / gif_name,
            args.fps,
            args.dpi,
            args.extra_ratio,
            n_ghost,
            args.extra_mode,
        )

    summary_path = output_dir / f"ghost_inference_summary_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved: {summary_path}")
    print(
        f"added sensors: {n_ghost} (requested {requested_extra_count}, mode={args.extra_mode}, "
        f"grid={layout_info['grid_rows']}x{layout_info['grid_cols']}, candidates={layout_info['grid_candidates']}, "
        f"spacing={layout_info['grid_spacing_deg']}, exclusion_radius={layout_info['station_exclusion_radius_deg']}, "
        f"removed_near_stations={layout_info['station_exclusion_removed']})"
    )


if __name__ == "__main__":
    main()





# '''
# python make_ghost_inference_gif.py --output_dir WindDataNE-US/ModelOutputs/exp2/best_ghost_100 --extra_ratio 0.2 --max_frames 2 --frame_step 1 --split test --channel u --output ghost_20pct_u.gif
# '''