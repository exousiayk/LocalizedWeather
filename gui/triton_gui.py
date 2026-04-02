#!/usr/bin/env python3
from __future__ import annotations

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import xarray as xr
from scipy.spatial import cKDTree


HOST = "127.0.0.1"
PORT = 8080
TRITON_URL = "http://localhost:18000/v2/models/weather_model/infer"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "WindDataNE-US"
MADIS_PATH = DATA_ROOT / "madis" / "processed" / "Meta-2019-2023" / "madis_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc"
ERA5_PATH = DATA_ROOT / "ERA5" / "Processed" / "era5_2023_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc"
BOUNDARY_PATH = DATA_ROOT / "Shapefiles" / "Regions" / "northeastern_buffered.shp"

CHANNELS = ["u", "v", "temp", "dewpoint"]
ERA5_CHANNELS = ["u10", "v10", "t2m"]
COORD_BOUNDS = {"minLon": -83.0, "maxLon": -65.0, "minLat": 37.0, "maxLat": 49.0}
SAMPLE_TIME_INDEX = 5000
M2M_NEIGHBORS = 4
E2M_NEIGHBORS = 8


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LocalizedWeather Triton GUI</title>
  <style>
    :root {
      --bg: #07101f;
      --panel: rgba(12, 20, 37, 0.92);
      --panel-2: rgba(20, 30, 54, 0.95);
      --text: #eef4ff;
      --muted: #a9b5d6;
      --border: rgba(142, 172, 255, 0.18);
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
      --accent: #6fe7c8;
      --accent-2: #7db8ff;
      --danger: #ff8a8a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 14%, rgba(111, 231, 200, 0.18), transparent 22%),
        radial-gradient(circle at 88% 12%, rgba(125, 184, 255, 0.20), transparent 26%),
        linear-gradient(135deg, #06101f 0%, #101a33 52%, #081122 100%);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .wrap {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 18px 36px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.45fr 0.75fr;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(32px, 4.5vw, 60px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }

    .subtitle {
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 74ch;
      font-size: 15px;
      line-height: 1.55;
    }

    .status {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(111, 231, 200, 0.10), rgba(125, 184, 255, 0.06));
      box-shadow: var(--shadow);
    }

    .status strong { display: block; margin-bottom: 6px; }

    .grid {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 16px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(18px);
    }

    .panel-head {
      padding: 16px 18px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent);
    }

    .panel-head h2 {
      margin: 0;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .panel-body { padding: 16px 18px 18px; }

    label {
      display: block;
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
    }

    select, button {
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      color: var(--text);
      background: rgba(255, 255, 255, 0.06);
      padding: 12px 14px;
      font-size: 15px;
      outline: none;
    }

    button {
      cursor: pointer;
      margin-top: 10px;
      background: linear-gradient(135deg, rgba(111, 231, 200, 0.20), rgba(125, 184, 255, 0.16));
      font-weight: 700;
    }

    button:hover { border-color: rgba(111, 231, 200, 0.5); }

    .meta {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }

    .meta-card {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .meta-card span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .meta-card strong { font-size: 14px; }

    .map-panel { min-height: 860px; }

    .map-shell {
      display: grid;
      grid-template-columns: 1fr 260px;
      gap: 0;
      min-height: 760px;
    }

    .map-area {
      position: relative;
      padding: 18px;
      min-height: 760px;
      background:
        radial-gradient(circle at 20% 15%, rgba(255,255,255,0.05), transparent 25%),
        radial-gradient(circle at 80% 0%, rgba(125,184,255,0.08), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    }

    svg { width: 100%; height: 100%; display: block; }

    .side {
      padding: 18px;
      background: rgba(6, 12, 24, 0.48);
      border-left: 1px solid rgba(255, 255, 255, 0.07);
    }

    .legend {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .swatch {
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.3);
      flex: 0 0 auto;
    }

    .station-list {
      margin-top: 14px;
      max-height: 520px;
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 4px;
    }

    .station-item {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      font-size: 12px;
    }

    .station-item strong { display: block; font-size: 13px; margin-bottom: 4px; }

    .footer-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    @media (max-width: 1080px) {
      .hero, .grid, .map-shell { grid-template-columns: 1fr; }
      .map-panel { min-height: auto; }
      .map-area { min-height: 640px; }
      .side { border-left: 0; border-top: 1px solid rgba(255,255,255,0.07); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>LocalizedWeather Triton GUI</h1>
        <p class="subtitle">Single-channel weather forecast visualization backed by the 2023 processed MADIS and ERA5 inputs. Choose one channel, run inference against Triton, and inspect the predicted station map.</p>
      </div>
      <div class="status">
        <strong>Triton</strong>
        <div>Ready: <span id="ready">checking</span></div>
        <div>Sample time: <span id="sample-time">-</span></div>
      </div>
    </div>

    <div class="grid">
      <aside class="panel">
        <div class="panel-head"><h2>Controls</h2></div>
        <div class="panel-body">
          <label for="channel">Channel</label>
          <select id="channel">
            <option value="u">u</option>
            <option value="v" selected>v</option>
            <option value="temp">temp</option>
            <option value="dewpoint">dewpoint</option>
          </select>
          <button id="run">Run inference</button>
          <button id="sample" style="margin-top: 8px;">Reset map</button>

          <div class="meta">
            <div class="meta-card">
              <span>Status</span>
              <strong id="summary">Awaiting request.</strong>
            </div>
            <div class="meta-card">
              <span>Model version</span>
              <strong id="model-version">-</strong>
            </div>
            <div class="meta-card">
              <span>Channel</span>
              <strong id="current-channel">v</strong>
            </div>
          </div>
        </div>
      </aside>

      <section class="panel map-panel">
        <div class="panel-head"><h2>Prediction map</h2></div>
        <div class="map-shell">
          <div class="map-area">
            <svg id="map" viewBox="0 0 1000 760" aria-label="Forecast map"></svg>
          </div>
          <aside class="side">
            <div>
              <div style="font-size: 13px; color: var(--muted);">Legend</div>
              <div id="legend" class="legend"></div>
            </div>
            <div class="station-list" id="station-list"></div>
            <div class="footer-note" id="map-meta">The map is based on the Northeastern buffered region boundary and station locations from the processed dataset.</div>
          </aside>
        </div>
      </section>
    </div>
  </div>

  <script>
    const COLOR_STOPS = [
      [0.0, [37, 54, 139]],
      [0.5, [113, 198, 255]],
      [1.0, [122, 233, 183]],
    ];

    const CHANNEL_LABELS = {
      u: "U wind",
      v: "V wind",
      temp: "Temperature",
      dewpoint: "Dew point",
    };

    const state = {
      boundary: [],
      bounds: null,
      stations: [],
      values: [],
      predicted: [],
      sampleTime: "-",
      channel: "v",
      outputShape: null,
    };

    function clearSvg(svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function makeSvgEl(name, attrs) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      return el;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function rgba(r, g, b, a) {
      return `rgba(${r}, ${g}, ${b}, ${a})`;
    }

    function colorForRatio(ratio) {
      const value = clamp(ratio, 0, 1);
      let left = COLOR_STOPS[0];
      let right = COLOR_STOPS[COLOR_STOPS.length - 1];
      for (let index = 0; index < COLOR_STOPS.length - 1; index += 1) {
        const a = COLOR_STOPS[index];
        const b = COLOR_STOPS[index + 1];
        if (value >= a[0] && value <= b[0]) {
          left = a;
          right = b;
          break;
        }
      }
      const span = right[0] - left[0] || 1;
      const t = (value - left[0]) / span;
      const rgb = left[1].map((component, index) => Math.round(component + (right[1][index] - component) * t));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }

    function project(bounds, lon, lat, width, height, pad) {
      const x = pad + ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * (width - pad * 2);
      const y = height - pad - ((lat - bounds.minLat) / (bounds.maxLat - bounds.minLat)) * (height - pad * 2);
      return [x, y];
    }

    function updateLegend(channel, minValue, maxValue) {
      const legend = document.getElementById("legend");
      const label = CHANNEL_LABELS[channel] || channel;
      legend.innerHTML = `
        <div class="legend-item"><span class="swatch" style="background:${colorForRatio(0)}"></span>${label} low</div>
        <div class="legend-item"><span class="swatch" style="background:${colorForRatio(0.5)}"></span>mid</div>
        <div class="legend-item"><span class="swatch" style="background:${colorForRatio(1)}"></span>high</div>
        <div style="margin-top: 8px; color: var(--text);">Range: ${minValue.toFixed(2)} to ${maxValue.toFixed(2)}</div>
      `;
    }

    function renderBoundary() {
      const svg = document.getElementById("map");
      clearSvg(svg);
      const width = 1000;
      const height = 760;
      const pad = 56;
      const bounds = state.bounds || { minLon: -83, maxLon: -65, minLat: 37, maxLat: 49 };

      svg.appendChild(makeSvgEl("rect", {
        x: 0, y: 0, width, height, rx: 20,
        fill: "rgba(255,255,255,0.02)", stroke: "rgba(155,180,255,0.14)"
      }));

      state.boundary.forEach((ring) => {
        if (!ring.length) return;
        const points = ring.map(([lon, lat]) => {
          const [x, y] = project(bounds, lon, lat, width, height, pad);
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        }).join(" ");
        svg.appendChild(makeSvgEl("polyline", {
          points,
          fill: "none",
          stroke: "rgba(255,255,255,0.46)",
          "stroke-width": 1.6,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }));
      });

      svg.appendChild(makeSvgEl("text", {
        x: 24, y: 28, fill: "#eff4ff", "font-size": 14, "font-weight": 700,
      })).textContent = "Region map";
      drawStations();
    }

    function drawStations() {
      const svg = document.getElementById("map");
      const width = 1000;
      const height = 760;
      const pad = 56;
      const bounds = state.bounds || { minLon: -83, maxLon: -65, minLat: 37, maxLat: 49 };
      const values = state.predicted.length ? state.predicted : state.values;
      if (!state.stations.length || !values.length) return;

      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      updateLegend(state.channel, minValue, maxValue);

      const list = document.getElementById("station-list");
      list.innerHTML = "";

      state.stations.forEach((station, index) => {
        const value = values[index];
        const ratio = (value - minValue) / ((maxValue - minValue) || 1);
        const color = colorForRatio(ratio);
        const [x, y] = project(bounds, station.lon, station.lat, width, height, pad);

        svg.appendChild(makeSvgEl("circle", {
          cx: x, cy: y, r: 4.2,
          fill: color,
          stroke: "rgba(255,255,255,0.86)",
          "stroke-width": 0.7,
        }));

        const tooltip = `${station.lon.toFixed(2)}, ${station.lat.toFixed(2)} -> ${value.toFixed(3)}`;
        svg.appendChild(makeSvgEl("title", {})).textContent = tooltip;

        const item = document.createElement("div");
        item.className = "station-item";
        item.innerHTML = `<strong>#${index + 1}</strong><div>${station.lon.toFixed(2)}, ${station.lat.toFixed(2)}</div><div>${value.toFixed(3)}</div>`;
        list.appendChild(item);
      });

      document.getElementById("map-meta").textContent = `Showing ${state.stations.length} stations for ${CHANNEL_LABELS[state.channel] || state.channel}.`;
    }

    async function loadBoundary() {
      const response = await fetch("/api/boundary");
      const data = await response.json();
      state.boundary = data.boundary || [];
      state.bounds = data.bounds || null;
      renderBoundary();
    }

    async function infer() {
      const channel = document.getElementById("channel").value;
      state.channel = channel;
      document.getElementById("current-channel").textContent = channel;
      document.getElementById("summary").textContent = "Running inference...";

      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel }),
      });

      const data = await response.json();
      if (!response.ok) {
        document.getElementById("summary").textContent = data.error || "Inference failed";
        return;
      }

      state.sampleTime = data.sample_time || "-";
      state.stations = (data.stations?.lons || []).map((lon, index) => ({ lon, lat: data.stations.lats[index] }));
      const output = data.outputs?.[0] || {};
      state.outputShape = output.shape || null;
      state.predicted = output.decoded_data || [];
      state.values = output.decoded_data || [];

      document.getElementById("sample-time").textContent = state.sampleTime;
      document.getElementById("summary").textContent = "Inference complete.";
      document.getElementById("model-version").textContent = data.model_version || "-";
      renderBoundary();
    }

    async function resetDefaults() {
      document.getElementById("channel").value = "v";
      document.getElementById("current-channel").textContent = "v";
      document.getElementById("summary").textContent = "Defaults loaded.";
      document.getElementById("sample-time").textContent = "-";
      document.getElementById("model-version").textContent = "-";
      state.channel = "v";
      state.predicted = [];
      state.values = [];
      state.stations = [];
      document.getElementById("station-list").innerHTML = "";
      document.getElementById("legend").innerHTML = "";
      await loadBoundary();
    }

    fetch("/api/ready").then((res) => res.json()).then((data) => {
      document.getElementById("ready").textContent = data.ready ? "yes" : "no";
    }).catch(() => {
      document.getElementById("ready").textContent = "unreachable";
    });

    document.getElementById("run").addEventListener("click", infer);
    document.getElementById("sample").addEventListener("click", resetDefaults);
    loadBoundary();
  </script>
</body>
</html>
"""


@lru_cache(maxsize=1)
def load_boundary() -> dict:
  try:
    import geopandas as gpd
  except ImportError:
    return {"boundary": [], "bounds": COORD_BOUNDS}

  if not BOUNDARY_PATH.exists():
    return {"boundary": [], "bounds": COORD_BOUNDS}

  gdf = gpd.read_file(BOUNDARY_PATH)
  if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

  geom = gdf.dissolve().geometry.iloc[0]
  rings: list[list[list[float]]] = []
  if geom.geom_type == "Polygon":
    rings.append([[float(x), float(y)] for x, y in geom.exterior.coords])
  elif geom.geom_type == "MultiPolygon":
    for poly in geom.geoms:
      rings.append([[float(x), float(y)] for x, y in poly.exterior.coords])

  min_lon, min_lat, max_lon, max_lat = map(float, gdf.total_bounds)
  bounds = {
    "minLon": min_lon,
    "maxLon": max_lon,
    "minLat": min_lat - 1.5,
    "maxLat": max_lat + 1.5,
  }
  return {"boundary": rings, "bounds": bounds}


@lru_cache(maxsize=1)
def load_context() -> dict:
    if not MADIS_PATH.exists():
        raise FileNotFoundError(f"Missing MADIS file: {MADIS_PATH}")
    if not ERA5_PATH.exists():
        raise FileNotFoundError(f"Missing ERA5 file: {ERA5_PATH}")

    madis = xr.load_dataset(MADIS_PATH)
    era5 = xr.load_dataset(ERA5_PATH)

    stats = {
        "madis": {
            var: {"mean": float(madis[var].mean()), "std": float(madis[var].std())}
            for var in CHANNELS
        },
        "era5": {
            var: {"mean": float(era5[var].mean()), "std": float(era5[var].std())}
            for var in ERA5_CHANNELS
        },
    }

    station_lons = madis["lon"].values.astype(np.float32)
    station_lats = madis["lat"].values.astype(np.float32)
    external_lons = era5["longitude"].values.astype(np.float32)
    external_lats = era5["latitude"].values.astype(np.float32)
    sample_index = min(SAMPLE_TIME_INDEX, int(madis.sizes["time"]) - 2)

    return {
        "madis": madis,
        "era5": era5,
        "stats": stats,
        "station_lons": station_lons,
        "station_lats": station_lats,
        "external_lons": external_lons,
        "external_lats": external_lats,
        "sample_index": sample_index,
        "sample_time": np.datetime_as_string(np.asarray(madis["time"].values[sample_index]), unit="h"),
        "boundary": load_boundary(),
    }


def _normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / (std + 1e-6)


def _denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * (std + 1e-6) + mean


def _make_input(name: str, array: np.ndarray, datatype: str = "FP32") -> dict:
    return {"name": name, "shape": list(array.shape), "datatype": datatype, "data": array.reshape(-1).tolist()}


def _build_edges(target_lons: np.ndarray, target_lats: np.ndarray, source_lons: np.ndarray, source_lats: np.ndarray, k: int) -> np.ndarray:
    source_coords = np.column_stack([source_lons, source_lats])
    target_coords = np.column_stack([target_lons, target_lats])
    tree = cKDTree(source_coords)
    query_k = min(k, len(source_coords))
    _, neighbor_idx = tree.query(target_coords, k=query_k)
    if query_k == 1:
        neighbor_idx = neighbor_idx[:, None]

    src = np.repeat(np.arange(len(target_coords), dtype=np.int64), query_k)
    dst = neighbor_idx.reshape(-1).astype(np.int64)
    return np.stack([dst, src], axis=0).astype(np.int64)


def build_real_request() -> dict:
    ctx = load_context()
    madis = ctx["madis"]
    era5 = ctx["era5"]
    stats = ctx["stats"]
    t = ctx["sample_index"]

    madis_slice = slice(t - 48, t + 1)
    era5_slice = slice(t - 48, t + 2)

    madis_arrays = []
    for var in CHANNELS:
        values = madis[var].isel(time=madis_slice).values.astype(np.float32)
        values = _normalize(values, stats["madis"][var]["mean"], stats["madis"][var]["std"])
        madis_arrays.append(values)
    madis_x = np.stack(madis_arrays, axis=-1)[None, ...].astype(np.float32)

    external_arrays = []
    for var in ERA5_CHANNELS:
        values = era5[var].isel(time=era5_slice).values.astype(np.float32).T
        values = _normalize(values, stats["era5"][var]["mean"], stats["era5"][var]["std"])
        external_arrays.append(values)
    ex_x = np.stack(external_arrays, axis=-1)[None, ...].astype(np.float32)

    station_lons = ctx["station_lons"]
    station_lats = ctx["station_lats"]
    external_lons = ctx["external_lons"]
    external_lats = ctx["external_lats"]

    lon_span = COORD_BOUNDS["maxLon"] - COORD_BOUNDS["minLon"] + 1e-6
    lat_span = COORD_BOUNDS["maxLat"] - COORD_BOUNDS["minLat"] + 1e-6

    madis_lon = ((station_lons - COORD_BOUNDS["minLon"]) / lon_span).astype(np.float32)
    madis_lat = ((station_lats - COORD_BOUNDS["minLat"]) / lat_span).astype(np.float32)
    ex_lon = ((external_lons - COORD_BOUNDS["minLon"]) / lon_span).astype(np.float32)
    ex_lat = ((external_lats - COORD_BOUNDS["minLat"]) / lat_span).astype(np.float32)

    edge_index = _build_edges(station_lons, station_lats, station_lons, station_lats, M2M_NEIGHBORS)
    edge_index_e2m = _build_edges(station_lons, station_lats, external_lons, external_lats, E2M_NEIGHBORS)

    return {
        "inputs": [
            _make_input("madis_x", madis_x),
            _make_input("madis_lon", madis_lon[None, :, None]),
            _make_input("madis_lat", madis_lat[None, :, None]),
            _make_input("edge_index", edge_index[None, :, :], datatype="INT64"),
            _make_input("ex_lon", ex_lon[None, :, None]),
            _make_input("ex_lat", ex_lat[None, :, None]),
            _make_input("ex_x", ex_x),
            _make_input("edge_index_e2m", edge_index_e2m[None, :, :], datatype="INT64"),
        ],
        "outputs": [{"name": "pred"}],
        "meta": {
            "sample_time": ctx["sample_time"],
            "station_lons": station_lons.tolist(),
            "station_lats": station_lats.tolist(),
            "boundary": ctx["boundary"]["boundary"],
            "boundary_bounds": ctx["boundary"]["bounds"],
        },
    }


def infer_actual(channel: str) -> dict:
    request = build_real_request()
    response = requests.post(TRITON_URL, json=request, timeout=120)
    response.raise_for_status()
    triton_data = response.json()

    ctx = load_context()
    stats = ctx["stats"]["madis"]
    output = triton_data["outputs"][0]
    values = np.asarray(output["data"], dtype=np.float32).reshape(output["shape"])
    decoded = np.empty_like(values)
    for idx, var in enumerate(CHANNELS):
        decoded[..., idx] = _denormalize(values[..., idx], stats[var]["mean"], stats[var]["std"])

    channel_index = CHANNELS.index(channel)
    selected = decoded[0, :, channel_index].astype(float)

    return {
        "model_name": triton_data.get("model_name"),
        "model_version": triton_data.get("model_version"),
        "sample_time": request["meta"]["sample_time"],
        "boundary": request["meta"]["boundary"],
        "boundary_bounds": request["meta"]["boundary_bounds"],
        "stations": {
            "lons": request["meta"]["station_lons"],
            "lats": request["meta"]["station_lats"],
        },
        "outputs": [
            {
                "name": output["name"],
                "datatype": output["datatype"],
                "shape": output["shape"],
                "data": output["data"],
                "decoded_data": decoded.reshape(-1).tolist(),
                "selected_channel": selected.tolist(),
            }
        ],
        "channel": channel,
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route == "/api/ready":
            try:
                response = requests.get("http://localhost:18000/v2/health/ready", timeout=5)
                ready = response.status_code == 200
            except requests.RequestException:
                ready = False
            self._send_json(200, {"ready": ready})
            return

        if route == "/api/boundary":
            self._send_json(200, load_boundary())
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        route = urlparse(self.path).path
        if route != "/api/infer":
            self._send_json(404, {"error": "Not Found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            params = json.loads(raw.decode("utf-8")) if raw else {}
            channel = params.get("channel", "v")
            data = infer_actual(channel)
            self._send_json(200, data)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Triton GUI running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
