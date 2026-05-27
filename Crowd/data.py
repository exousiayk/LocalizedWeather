from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

try:
    from .graph import build_bipartite_edges, build_ghost_neighbors, build_knn_edges
except ImportError:  # pragma: no cover
    from graph import build_bipartite_edges, build_ghost_neighbors, build_knn_edges


@dataclass
class CrowdMeta:
    cctv_lon: np.ndarray
    cctv_lat: np.ndarray
    skt_lon: np.ndarray
    skt_lat: np.ndarray
    edge_index_cctv: torch.Tensor
    edge_index_skt2cctv: torch.Tensor
    seen_mask: np.ndarray
    ghost_mask: np.ndarray
    ghost_neighbors: np.ndarray


class CrowdDataset(Dataset):
    def __init__(
        self,
        cctv_nc_path: str | Path,
        skt_nc_path: str | Path,
        back_steps: int = 48,
        lead_steps: int = 12,
        ghost_holdout_ratio: float = 0.2,
        ghost_init_mode: str = "interp",
        ghost_seed: int = 42,
        knn_cctv: int = 4,
        knn_skt: int = 4,
    ):
        self.cctv_nc_path = Path(cctv_nc_path)
        self.skt_nc_path = Path(skt_nc_path)
        self.back_steps = int(back_steps)
        self.lead_steps = int(lead_steps)
        self.ghost_holdout_ratio = float(ghost_holdout_ratio)
        self.ghost_init_mode = ghost_init_mode
        self.ghost_seed = int(ghost_seed)
        self.knn_cctv = int(knn_cctv)
        self.knn_skt = int(knn_skt)

        self._load_data()
        self._build_graphs()
        self._build_valid_indices()

    def _load_data(self) -> None:
        cctv_ds = xr.open_dataset(self.cctv_nc_path)
        skt_ds = xr.open_dataset(self.skt_nc_path)

        cctv_time = cctv_ds["time"].values
        skt_time = skt_ds["time"].values
        if not np.array_equal(cctv_time, skt_time):
            common = np.intersect1d(cctv_time, skt_time)
            cctv_ds = cctv_ds.sel(time=common)
            skt_ds = skt_ds.sel(time=common)
            cctv_time = cctv_ds["time"].values
            skt_time = skt_ds["time"].values

        self.time = cctv_time
        self.cctv_counts = cctv_ds["count"].values.astype(np.float32)
        self.cctv_is_real = cctv_ds["count_is_real"].values.astype(bool)
        self.skt_counts = skt_ds["count"].values.astype(np.float32)
        self.skt_is_real = skt_ds["count_is_real"].values.astype(bool)

        self.cctv_lon = cctv_ds["lon"].values.astype(np.float32)
        self.cctv_lat = cctv_ds["lat"].values.astype(np.float32)
        self.skt_lon = skt_ds["lon"].values.astype(np.float32)
        self.skt_lat = skt_ds["lat"].values.astype(np.float32)

        cctv_ds.close()
        skt_ds.close()

        self.n_cctv = self.cctv_counts.shape[0]
        self.n_skt = self.skt_counts.shape[0]
        self.total_time = self.cctv_counts.shape[1]
        self.hist_len = self.back_steps

    def _build_valid_indices(self) -> None:
        max_start = self.total_time - self.hist_len - self.lead_steps + 1
        if max_start <= 0:
            self.valid_indices = np.zeros((0,), dtype=np.int64)
            return

        cctv_valid = self.cctv_is_real.any(axis=0)
        skt_valid = self.skt_is_real.any(axis=0)
        time_valid = cctv_valid & skt_valid

        indices = []
        for idx in range(max_start):
            hist_start = idx
            hist_end = idx + self.hist_len
            target_t = idx + self.hist_len - 1 + self.lead_steps
            if target_t >= self.total_time:
                break
            if time_valid[hist_start:hist_end].all() and time_valid[target_t]:
                indices.append(idx)

        self.valid_indices = np.asarray(indices, dtype=np.int64)

    def _build_graphs(self) -> None:
        rng = np.random.RandomState(self.ghost_seed)
        n_ghost = int(np.floor(self.n_cctv * self.ghost_holdout_ratio))
        n_ghost = max(1, min(n_ghost, self.n_cctv - 1))
        ghost_idx = np.sort(rng.choice(np.arange(self.n_cctv), size=n_ghost, replace=False))
        seen_mask = np.ones(self.n_cctv, dtype=bool)
        seen_mask[ghost_idx] = False
        ghost_mask = ~seen_mask

        edge_index_cctv = build_knn_edges(self.cctv_lon, self.cctv_lat, self.knn_cctv)
        edge_index_skt2cctv = build_bipartite_edges(self.skt_lon, self.skt_lat, self.cctv_lon, self.cctv_lat, self.knn_skt)
        ghost_neighbors = build_ghost_neighbors(self.cctv_lon, self.cctv_lat, ghost_idx, np.where(seen_mask)[0], self.knn_cctv)

        self.meta = CrowdMeta(
            cctv_lon=self.cctv_lon,
            cctv_lat=self.cctv_lat,
            skt_lon=self.skt_lon,
            skt_lat=self.skt_lat,
            edge_index_cctv=edge_index_cctv,
            edge_index_skt2cctv=edge_index_skt2cctv,
            seen_mask=seen_mask,
            ghost_mask=ghost_mask,
            ghost_neighbors=ghost_neighbors,
        )

    def __len__(self) -> int:
        return int(self.valid_indices.size)

    def _apply_ghost_init(self, history: np.ndarray) -> np.ndarray:
        if not self.meta.ghost_mask.any():
            return history

        out = history.copy()
        ghost_idx = np.where(self.meta.ghost_mask)[0]
        if self.ghost_init_mode.lower() == "zero":
            out[ghost_idx, :] = 0.0
            return out

        if self.ghost_init_mode.lower() == "interp":
            for local_idx, gidx in enumerate(ghost_idx):
                neighbors = self.meta.ghost_neighbors[local_idx]
                if neighbors.size == 0:
                    out[gidx, :] = 0.0
                    continue
                neighbor_block = out[neighbors, :]
                if neighbor_block.size == 0:
                    out[gidx, :] = 0.0
                    continue
                finite_mask = np.isfinite(neighbor_block)
                if not finite_mask.any():
                    out[gidx, :] = 0.0
                    continue
                safe_vals = np.where(finite_mask, neighbor_block, 0.0)
                counts = finite_mask.sum(axis=0)
                mean_vals = safe_vals.sum(axis=0) / np.maximum(counts, 1)
                out[gidx, :] = mean_vals
            return out

        raise ValueError(f"Unsupported ghost_init_mode: {self.ghost_init_mode}")

    def __getitem__(self, idx: int) -> dict:
        if idx >= self.valid_indices.size:
            raise IndexError("Index out of range")
        hist_start = int(self.valid_indices[idx])
        hist_end = hist_start + self.hist_len
        target_t = hist_start + self.hist_len - 1 + self.lead_steps

        cctv_hist = self.cctv_counts[:, hist_start:hist_end]
        skt_hist = self.skt_counts[:, hist_start:hist_end]
        cctv_hist = self._apply_ghost_init(cctv_hist)

        cctv_hist = np.nan_to_num(cctv_hist, nan=0.0)
        skt_hist = np.nan_to_num(skt_hist, nan=0.0)

        target = np.nan_to_num(self.cctv_counts[:, target_t], nan=0.0)
        target_is_real = self.cctv_is_real[:, target_t]

        sample = {
            "cctv_x": torch.from_numpy(cctv_hist).unsqueeze(-1),
            "skt_x": torch.from_numpy(skt_hist).unsqueeze(-1),
            "target": torch.from_numpy(target).unsqueeze(-1),
            "target_is_real": torch.from_numpy(target_is_real.astype(np.float32)).unsqueeze(-1),
            "seen_mask": torch.from_numpy(self.meta.seen_mask.astype(np.float32)).unsqueeze(-1),
            "ghost_mask": torch.from_numpy(self.meta.ghost_mask.astype(np.float32)).unsqueeze(-1),
            "time": torch.tensor(np.int64(self.time[target_t].astype("datetime64[ns]").astype("int64"))),
        }
        return sample

    def get_meta(self) -> CrowdMeta:
        return self.meta

    def get_scaler_values(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.cctv_counts, self.skt_counts
