from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests


TRITON_URL = "http://localhost:18000/v2/models/weather_model/infer"


def make_input(name: str, array: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(array.shape),
        "datatype": "FP32" if array.dtype == np.float32 else "INT64",
        "data": array.reshape(-1).tolist(),
    }


def main() -> None:
    batch_size = 1
    n_stations = 3
    n_external_nodes = 4
    madis_time_steps = 49
    external_time_steps = 50
    madis_vars = 4
    external_vars = 3

    rng = np.random.default_rng(42)

    madis_x = rng.normal(size=(batch_size, n_stations, madis_time_steps, madis_vars)).astype(np.float32)
    madis_lon = np.linspace(-80.0, -79.0, n_stations, dtype=np.float32).reshape(batch_size, n_stations, 1)
    madis_lat = np.linspace(40.0, 41.0, n_stations, dtype=np.float32).reshape(batch_size, n_stations, 1)

    edge_index = np.array([[[0, 1, 1, 2], [1, 0, 2, 1]]], dtype=np.int64)

    ex_lon = np.linspace(-80.2, -79.8, n_external_nodes, dtype=np.float32).reshape(batch_size, n_external_nodes, 1)
    ex_lat = np.linspace(39.8, 41.2, n_external_nodes, dtype=np.float32).reshape(batch_size, n_external_nodes, 1)
    ex_x = rng.normal(size=(batch_size, n_external_nodes, external_time_steps, external_vars)).astype(np.float32)

    edge_index_e2m = np.array([[[0, 1, 2, 3, 0, 2], [0, 1, 1, 2, 2, 0]]], dtype=np.int64)

    payload = {
        "inputs": [
            make_input("madis_x", madis_x),
            make_input("madis_lon", madis_lon),
            make_input("madis_lat", madis_lat),
            make_input("edge_index", edge_index),
            make_input("ex_lon", ex_lon),
            make_input("ex_lat", ex_lat),
            make_input("ex_x", ex_x),
            make_input("edge_index_e2m", edge_index_e2m),
        ],
        "outputs": [{"name": "pred"}],
    }

    response = requests.post(TRITON_URL, json=payload, timeout=30)
    response.raise_for_status()

    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()