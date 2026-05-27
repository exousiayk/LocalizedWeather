from __future__ import annotations

import argparse
import os
from random import sample
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import contextily as cx
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from data import CrowdDataset
from model import CrowdMPNN
from normalizer import StandardScaler

# 거리 기반 KNN 엣지 생성 함수 (가상 노드 연결용)
def build_knn_edges_dynamic(lons: np.ndarray, lats: np.ndarray, k: int) -> torch.Tensor:
    coords = np.column_stack([lons, lats]).astype(np.float32)
    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    k = min(int(k), n_nodes - 1)
    neighbors = np.argsort(dist2, axis=1)[:, :k]
    src = neighbors.reshape(-1)
    dst = np.repeat(np.arange(n_nodes), k)
    return torch.from_numpy(np.stack([src, dst], axis=0)).long()

def main():
    parser = argparse.ArgumentParser(description="Generate Dense Heatmap using Extra Grid Nodes")
    parser.add_argument("--cctv_nc", type=str, default='../sample/processed/cctv_processed.nc')
    parser.add_argument("--skt_nc", type=str, default='../sample/processed/skt_processed.nc')
    parser.add_argument("--checkpoint", type=str, default='outputs/exp_1/best.pt', help="Path to the trained .pt file")
    parser.add_argument("--grid_resolution", type=int, default=100, help="Resolution of the extra nodes grid (e.g., 30x30)")
    parser.add_argument("--test_index", type=int, default=0, help="Which timestep to visualize")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 원본 데이터셋 로드
    dataset = CrowdDataset(
        args.cctv_nc, 
        args.skt_nc, 
        ghost_holdout_ratio=0.0,
        back_steps=24,   # <--- 추가된 부분!
        lead_steps=24    # <--- 추가된 부분!
    ) # 평가가 아니므로 Ghost 0으로 설정
    cctv_vals, skt_vals = dataset.get_scaler_values()
    cctv_scaler = StandardScaler.fit(cctv_vals).to(device)
    skt_scaler = StandardScaler.fit(skt_vals).to(device)

    # 타겟 샘플 하나 가져오기 (시간대 1개)
    test_idx = int(len(dataset) * 0.9) + args.test_index # Test 셋의 특정 시점
    sample = dataset[test_idx]
    
    real_cctv_x = sample["cctv_x"].unsqueeze(0).to(device) # (1, N_real, hist_len, 1)
    skt_x = sample["skt_x"].unsqueeze(0).to(device)
    target_real = sample["target"].numpy()
    
    # 수정된 부분: int64 (나노초 timestamp)를 pandas datetime으로 안전하게 변환 후 문자열 포맷팅
    time_val = sample["time"].numpy()
    time_label = pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")

    meta = dataset.get_meta()
    real_lons = meta.cctv_lon
    real_lats = meta.cctv_lat
    n_real = len(real_lons)

    # ---------------------------------------------------------
    # 🔥 2. 추론용 Extra 가상 노드(Grid) 촘촘하게 생성
    # ---------------------------------------------------------
    lon_min, lon_max = real_lons.min() - 0.005, real_lons.max() + 0.005
    lat_min, lat_max = real_lats.min() - 0.005, real_lats.max() + 0.005

    grid_lons_1d = np.linspace(lon_min, lon_max, args.grid_resolution)
    grid_lats_1d = np.linspace(lat_min, lat_max, args.grid_resolution)
    glon, glat = np.meshgrid(grid_lons_1d, grid_lats_1d)
    
    extra_lons = glon.reshape(-1)
    extra_lats = glat.reshape(-1)
    n_extra = len(extra_lons)
    print(f"[*] Generated {n_extra} extra grid nodes for spatial inference.")

    # 전체 좌표 병합 (Real + Extra)
    all_lons = np.concatenate([real_lons, extra_lons])
    all_lats = np.concatenate([real_lats, extra_lats])
    all_pos = torch.from_numpy(np.column_stack([all_lons, all_lats]).astype(np.float32)).to(device)
    skt_pos = torch.from_numpy(np.column_stack([meta.skt_lon, meta.skt_lat]).astype(np.float32)).to(device)

    # ---------------------------------------------------------
    # 🔥 3. 그래프 구조 재구축 (Rebuild Edge Index) & 가상 입력 데이터 생성
    # ---------------------------------------------------------
    # 가상 노드가 포함된 거대한 새 인접 행렬 생성
    new_edge_index_cctv = build_knn_edges_dynamic(all_lons, all_lats, k=4).to(device)
    
    # 가상 노드의 과거 데이터는 가장 가까운 Real 노드의 값으로 보간(복사)하여 초기화
    extra_x = torch.zeros(1, n_extra, real_cctv_x.shape[2], 1, device=device)
    real_coords = np.column_stack([real_lons, real_lats])
    extra_coords = np.column_stack([extra_lons, extra_lats])
    for i in range(n_extra):
        dist = np.sum((real_coords - extra_coords[i])**2, axis=1)
        nearest_idx = np.argmin(dist)
        extra_x[0, i] = real_cctv_x[0, nearest_idx] # 가장 가까운 CCTV의 과거 흐름 복사

    # 모델에 넣을 최종 입력 데이터 병합
    all_cctv_x = torch.cat([real_cctv_x, extra_x], dim=1) # (1, N_real + N_extra, hist_len, 1)

    # 4. 모델 로드 및 추론
    model = CrowdMPNN(
        back_steps=24,   # <--- dataset.hist_len 대신 명시적 숫자로 고정
        hidden_dim=128, 
        n_passing=4
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        cctv_x_scaled = cctv_scaler.transform(all_cctv_x)
        skt_x_scaled = skt_scaler.transform(skt_x)
        pred_scaled = model(cctv_x_scaled, skt_x_scaled, all_pos, skt_pos, new_edge_index_cctv, meta.edge_index_skt2cctv.to(device))
        pred_real = cctv_scaler.inverse(pred_scaled).detach().cpu().numpy()[0, :, 0]

    # 예측값을 Real과 Extra로 분리
    pred_extra = pred_real[n_real:]

    # ---------------------------------------------------------
    # 🔥 5. 히트맵 시각화 (실제 지도 위 Grid Overlay)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 가상 노드 예측값 (배경 히트맵 역할)
    sc_grid = ax.scatter(extra_lons, extra_lats, c=pred_extra, cmap="YlOrRd", 
                         s=180, marker='s', alpha=0.55, vmin=0, vmax=pred_real.max())
    
    # 실제 노드 정답값 (위에 겹쳐 그리기)
    sc_real = ax.scatter(real_lons, real_lats, c=target_real, cmap="YlOrRd", 
                         s=250, edgecolors="black", linewidths=2.0, zorder=5, 
                         vmin=0, vmax=pred_real.max(), label="Real CCTV")

    ax.set_title(f"High-Density Crowd Inference Heatmap | Time: {time_label}", fontsize=16, pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper right', fontsize=12)
    plt.colorbar(sc_grid, ax=ax, label="Crowd Count")

    # 지도 배경 깔기
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, alpha=0.8)
    except Exception as e:
        print(f"[!] Map failed to load: {e}")

    plt.tight_layout()
    save_path = f"heatmap_inference_{time_label.replace(':', '')}.png"
    plt.savefig(save_path, dpi=200)
    print(f"[*] Dense inference heatmap saved to {save_path}")

if __name__ == "__main__":
    main()