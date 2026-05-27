#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_blackout_ranges(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    ranges = []
    for item in values:
        if not item:
            continue
        if "," not in item:
            raise ValueError("Blackout range must be 'start,end'")
        start_str, end_str = [part.strip() for part in item.split(",", 1)]
        start = pd.to_datetime(start_str, errors="raise")
        end = pd.to_datetime(end_str, errors="raise")
        if end < start:
            raise ValueError(f"Blackout end before start: {item}")
        ranges.append((start, end))
    return ranges


def load_counts_csv(path, missing_value=None):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    if missing_value is not None:
        df = df.replace(missing_value, np.nan)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_location_csv(path):
    df = pd.read_csv(path)
    cols = {col.lower().strip(): col for col in df.columns}
    lon_col = cols.get("lon") or cols.get("longitude") or cols.get("lng") or cols.get("x")
    lat_col = cols.get("lat") or cols.get("latitude") or cols.get("y")
    if lon_col is None or lat_col is None:
        raise ValueError(f"Missing lon/lat columns in {path}")

    id_candidates = [col for col in df.columns if col not in {lon_col, lat_col}]
    if not id_candidates:
        raise ValueError(f"Missing node id column in {path}")
    id_col = id_candidates[0]

    out = pd.DataFrame(
        {
            "node_id": df[id_col].astype(str).str.strip(),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["node_id", "lon", "lat"]).drop_duplicates(subset=["node_id"])
    return out


def coerce_station_ids(ids):
    numeric = pd.to_numeric(ids, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()
    return np.asarray(ids, dtype=str)


def align_time_axes(cctv_df, skt_df, freq):
    start = max(cctv_df.index.min(), skt_df.index.min())
    end = min(cctv_df.index.max(), skt_df.index.max())
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("No overlapping time range between CCTV and SKT")
    full_index = pd.date_range(start=start, end=end, freq=freq)
    cctv_df = cctv_df.reindex(full_index)
    skt_df = skt_df.reindex(full_index)
    return cctv_df, skt_df, full_index


def apply_blackout(df, blackout_ranges):
    if not blackout_ranges:
        return df, pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for start, end in blackout_ranges:
        mask |= (df.index >= start) & (df.index <= end)
    df = df.copy()
    df.loc[mask, :] = np.nan
    return df, mask


def build_all_missing_mask(df):
    if df.empty:
        return pd.Series(False, index=df.index)
    return df.isna().all(axis=1)


def filter_by_missing_rate(df, max_missing_rate, mask_exclude=None):
    if max_missing_rate is None:
        return df, df.columns
    if mask_exclude is None:
        mask_exclude = pd.Series(False, index=df.index)
    valid_df = df.loc[~mask_exclude]
    missing_rate = valid_df.isna().mean(axis=0)
    keep_cols = missing_rate[missing_rate <= max_missing_rate].index.tolist()
    return df[keep_cols], keep_cols


def fill_short_gaps(series, max_gap_steps):
    if max_gap_steps <= 0:
        return series
    is_na = series.isna()
    if not is_na.any():
        return series
    groups = (is_na != is_na.shift()).cumsum()
    group_sizes = is_na.groupby(groups).transform("sum")
    long_gap_mask = is_na & (group_sizes > max_gap_steps)
    filled = series.interpolate(method="time", limit=max_gap_steps, limit_direction="both")
    filled[long_gap_mask] = np.nan
    return filled


def fill_short_gaps_df(df, max_gap_steps):
    filled = df.copy()
    for col in filled.columns:
        filled[col] = fill_short_gaps(filled[col], max_gap_steps)
    return filled


def build_knn_indices(lons, lats, k):
    n_nodes = len(lons)
    if n_nodes <= 1:
        return np.zeros((n_nodes, 0), dtype=np.int64)
    coords = np.column_stack([lons, lats]).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)
    k = min(int(k), n_nodes - 1)
    return order[:, :k].astype(np.int64)


def spatial_fill(values, knn_idx, block_mask=None):
    filled = values.copy()
    n_time, n_nodes = filled.shape
    for t in range(n_time):
        if block_mask is not None and block_mask.iloc[t]:
            continue
        row = filled[t]
        missing = np.isnan(row)
        if not missing.any():
            continue
        missing_idx = np.where(missing)[0]
        for node_idx in missing_idx:
            neighbors = knn_idx[node_idx]
            if neighbors.size == 0:
                continue
            neighbor_vals = row[neighbors]
            neighbor_vals = neighbor_vals[~np.isnan(neighbor_vals)]
            if neighbor_vals.size > 0:
                filled[t, node_idx] = float(np.mean(neighbor_vals))
    return filled


def apply_fallback_fill(values, mode, blackout_mask=None):
    if mode == "none":
        return values
    if blackout_mask is not None:
        keep_mask = ~blackout_mask.to_numpy()
    else:
        keep_mask = np.ones(values.shape[0], dtype=bool)

    if not keep_mask.any():
        return values

    work = values[keep_mask].copy()

    if mode in {"station_mean", "station_then_global"}:
        station_means = np.nanmean(work, axis=0)
        missing = np.isnan(work)
        if missing.any():
            work = np.where(missing, station_means.reshape(1, -1), work)

    if mode in {"global_mean", "station_then_global"}:
        if np.isnan(work).any():
            global_mean = np.nanmean(work)
            if not np.isnan(global_mean):
                work = np.where(np.isnan(work), global_mean, work)

    filled = values.copy()
    filled[keep_mask] = work
    return filled


def to_xarray(count_values, is_real_mask, time_index, locations, source_name):
    station_ids = coerce_station_ids(locations["node_id"])
    ds = xr.Dataset(
        {
            "count": (("stations", "time"), count_values.T.astype(np.float32)),
            "count_is_real": (("stations", "time"), is_real_mask.T.astype(bool)),
        },
        coords={
            "stations": station_ids,
            "time": time_index.values,
            "lon": ("stations", locations["lon"].to_numpy(dtype=np.float32)),
            "lat": ("stations", locations["lat"].to_numpy(dtype=np.float32)),
        },
        attrs={
            "source": source_name,
            "missing_mask": "count_is_real==false indicates originally missing or blackout",
        },
    )
    return ds


def format_station_list(values, max_items=80):
    values = [str(v) for v in values]
    if len(values) <= max_items:
        return ", ".join(values)
    head = ", ".join(values[:max_items])
    return f"{head} ... (+{len(values) - max_items} more)"


def report_missing_rates(label, df, blackout_mask, threshold, kept_ids, top_k=5):
    if df.empty:
        print(f"{label}: no data")
        return
    valid_df = df.loc[~blackout_mask]
    missing_rate = valid_df.isna().mean(axis=0)
    kept_count = len(kept_ids)
    total_count = len(missing_rate)
    print(f"{label} missing-rate threshold={threshold:.2f} -> kept {kept_count}/{total_count}")
    print(f"{label} kept stations: {format_station_list(kept_ids)}")
    best = missing_rate.sort_values().head(top_k)
    worst = missing_rate.sort_values(ascending=False).head(top_k)
    best_str = ", ".join([f"{idx}:{rate:.3f}" for idx, rate in best.items()])
    worst_str = ", ".join([f"{idx}:{rate:.3f}" for idx, rate in worst.items()])
    # print(f"{label} best {top_k} missing rates: {best_str}")
    # print(f"{label} worst {top_k} missing rates: {worst_str}")


def report_output_stats(label, values, is_real_mask, time_index, blackout_mask=None):
    total = values.size
    nan_count = int(np.isnan(values).sum()) if total else 0
    nan_ratio = (nan_count / total) if total else 0.0
    is_real_ratio = float(is_real_mask.mean()) if is_real_mask.size else 0.0
    print(
        f"{label} output: stations={values.shape[1]}, time_steps={len(time_index)}, "
        f"nan_ratio={nan_ratio:.4f}, is_real_ratio={is_real_ratio:.4f}"
    )
    if blackout_mask is not None:
        blackout_steps = int(blackout_mask.sum())
        blackout_ratio = blackout_steps / max(len(time_index), 1)
        print(f"{label} blackout steps: {blackout_steps} ({blackout_ratio:.4f})")
        keep_mask = ~blackout_mask.to_numpy()
        if keep_mask.any():
            kept_values = values[keep_mask, :]
            kept_total = kept_values.size
            kept_nan = int(np.isnan(kept_values).sum()) if kept_total else 0
            kept_ratio = (kept_nan / kept_total) if kept_total else 0.0
            print(f"{label} nan_ratio (non-blackout): {kept_ratio:.4f}")
        else:
            print(f"{label} nan_ratio (non-blackout): n/a (no non-blackout steps)")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CCTV/SKT CSVs into NetCDF")
    parser.add_argument("--cctv_counts", type=str, default="sample/cctv_total.csv")
    parser.add_argument("--cctv_locations", type=str, default="sample/cctv_location.csv")
    parser.add_argument("--skt_counts", type=str, default="sample/skt_total.csv")
    parser.add_argument("--skt_locations", type=str, default="sample/skt_location.csv")
    parser.add_argument("--output_dir", type=str, default="sample/processed")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument("--short_gap_max_minutes", type=int, default=30)
    parser.add_argument("--cctv_missing_value", type=float, default=-1)
    parser.add_argument("--cctv_missing_rate_threshold", type=float, default=0.4)
    parser.add_argument("--knn_k", type=int, default=4)
    parser.add_argument(
        "--fallback_fill",
        type=str,
        default="station_then_global",
        choices=["none", "station_mean", "global_mean", "station_then_global"],
        help="Fallback fill for remaining NaNs (non-blackout only).",
    )
    parser.add_argument(
        "--auto_blackout_all_missing",
        action="store_true",
        default=True,
        help="Treat timestamps where all CCTV sensors are missing as blackout.",
    )
    parser.add_argument(
        "--blackout",
        action="append",
        default=["2026-04-04 04:15:00,2026-04-10 10:20:00"],
        help="Blackout range as 'start,end' (inclusive). Can be provided multiple times.",
    )
    args = parser.parse_args()

    cctv_counts_path = Path(args.cctv_counts)
    cctv_locations_path = Path(args.cctv_locations)
    skt_counts_path = Path(args.skt_counts)
    skt_locations_path = Path(args.skt_locations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blackout_ranges = parse_blackout_ranges(args.blackout)

    cctv_counts = load_counts_csv(cctv_counts_path, missing_value=args.cctv_missing_value)
    skt_counts = load_counts_csv(skt_counts_path, missing_value=None)

    cctv_locations = load_location_csv(cctv_locations_path)
    skt_locations = load_location_csv(skt_locations_path)

    cctv_counts, skt_counts, aligned_index = align_time_axes(cctv_counts, skt_counts, args.freq)

    auto_blackout_mask = build_all_missing_mask(cctv_counts) if args.auto_blackout_all_missing else pd.Series(
        False, index=cctv_counts.index
    )

    cctv_counts, manual_blackout_mask = apply_blackout(cctv_counts, blackout_ranges)
    blackout_mask = manual_blackout_mask | auto_blackout_mask
    if auto_blackout_mask.any():
        cctv_counts.loc[auto_blackout_mask, :] = np.nan

    cctv_counts_raw = cctv_counts.copy()

    cctv_counts, kept_cctv_ids = filter_by_missing_rate(
        cctv_counts,
        args.cctv_missing_rate_threshold,
        mask_exclude=blackout_mask,
    )
    cctv_locations = cctv_locations[cctv_locations["node_id"].isin([str(i) for i in kept_cctv_ids])].copy()

    cctv_counts = cctv_counts.reindex(columns=[str(i) for i in cctv_locations["node_id"]])
    skt_counts = skt_counts.reindex(columns=[str(i) for i in skt_locations["node_id"]])

    cctv_is_real = ~cctv_counts.isna()
    skt_is_real = ~skt_counts.isna()

    freq_minutes = int(pd.Timedelta(args.freq) / pd.Timedelta(minutes=1))
    max_gap_steps = max(1, int(args.short_gap_max_minutes / max(freq_minutes, 1)))

    cctv_filled = fill_short_gaps_df(cctv_counts, max_gap_steps)
    skt_filled = fill_short_gaps_df(skt_counts, max_gap_steps)

    cctv_knn = build_knn_indices(cctv_locations["lon"].to_numpy(), cctv_locations["lat"].to_numpy(), args.knn_k)
    skt_knn = build_knn_indices(skt_locations["lon"].to_numpy(), skt_locations["lat"].to_numpy(), args.knn_k)

    cctv_values = spatial_fill(cctv_filled.to_numpy(dtype=np.float32), cctv_knn, block_mask=blackout_mask)
    skt_values = spatial_fill(skt_filled.to_numpy(dtype=np.float32), skt_knn, block_mask=None)

    cctv_values = apply_fallback_fill(cctv_values, args.fallback_fill, blackout_mask=blackout_mask)
    skt_values = apply_fallback_fill(skt_values, args.fallback_fill, blackout_mask=None)

    if blackout_mask.any():
        cctv_values[blackout_mask.to_numpy(), :] = np.nan

    cctv_ds = to_xarray(cctv_values, cctv_is_real.to_numpy(dtype=bool), aligned_index, cctv_locations, "cctv")
    skt_ds = to_xarray(skt_values, skt_is_real.to_numpy(dtype=bool), aligned_index, skt_locations, "skt")

    cctv_out = output_dir / "cctv_processed.nc"
    skt_out = output_dir / "skt_processed.nc"
    cctv_ds.to_netcdf(cctv_out)
    skt_ds.to_netcdf(skt_out)

    manual_blackout_steps = int(manual_blackout_mask.sum())
    auto_blackout_steps = int(auto_blackout_mask.sum())
    total_blackout_steps = int(blackout_mask.sum())
    print(
        "CCTV blackout steps: "
        f"manual={manual_blackout_steps}, auto={auto_blackout_steps}, total={total_blackout_steps}"
    )
    report_missing_rates(
        "CCTV",
        cctv_counts_raw,
        blackout_mask,
        args.cctv_missing_rate_threshold,
        kept_cctv_ids,
    )
    report_output_stats("CCTV", cctv_values, cctv_is_real.to_numpy(dtype=bool), aligned_index, blackout_mask)
    report_output_stats("SKT", skt_values, skt_is_real.to_numpy(dtype=bool), aligned_index)

    print(f"saved: {cctv_out}")
    print(f"saved: {skt_out}")


if __name__ == "__main__":
    main()
