from __future__ import annotations

import numpy as np
import torch


def build_knn_edges(lons: np.ndarray, lats: np.ndarray, k: int) -> torch.Tensor:
    coords = np.column_stack([lons, lats]).astype(np.float32)
    n_nodes = len(coords)
    if n_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    k = min(int(k), n_nodes - 1)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    neighbors = np.argsort(dist2, axis=1)[:, :k]
    src = neighbors.reshape(-1)
    dst = np.repeat(np.arange(n_nodes), k)
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


def build_bipartite_edges(
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    dst_lons: np.ndarray,
    dst_lats: np.ndarray,
    k: int,
) -> torch.Tensor:
    src_coords = np.column_stack([src_lons, src_lats]).astype(np.float32)
    dst_coords = np.column_stack([dst_lons, dst_lats]).astype(np.float32)
    n_src = len(src_coords)
    n_dst = len(dst_coords)
    if n_src == 0 or n_dst == 0:
        return torch.empty((2, 0), dtype=torch.long)

    diff = dst_coords[:, None, :] - src_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    k = min(int(k), n_src)
    neighbors = np.argsort(dist2, axis=1)[:, :k]
    src = neighbors.reshape(-1)
    dst = np.repeat(np.arange(n_dst), k)
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


def build_ghost_neighbors(lons: np.ndarray, lats: np.ndarray, ghost_idx: np.ndarray, seen_idx: np.ndarray, k: int) -> np.ndarray:
    if ghost_idx.size == 0 or seen_idx.size == 0:
        return np.zeros((len(ghost_idx), 0), dtype=np.int64)

    seen_coords = np.column_stack([lons[seen_idx], lats[seen_idx]]).astype(np.float32)
    ghost_coords = np.column_stack([lons[ghost_idx], lats[ghost_idx]]).astype(np.float32)
    diff = ghost_coords[:, None, :] - seen_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    k = min(int(k), seen_coords.shape[0])
    neighbors = np.argsort(dist2, axis=1)[:, :k]
    return seen_idx[neighbors]
