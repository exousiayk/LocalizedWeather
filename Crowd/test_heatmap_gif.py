from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import contextily as cx
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from data import CrowdDataset
from model import CrowdMPNN
from normalizer import StandardScaler

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

# SKT -> Ghost 연결을 위한 Bipartite KNN 생성 함수
def build_bipartite_knn_edges(src_lons, src_lats, dst_lons, dst_lats, k=1):
    src_coords = np.column_stack([src_lons, src_lats]).astype(np.float32)
    dst_coords = np.column_stack([dst_lons, dst_lats]).astype(np.float32)
    diff = dst_coords[:, None, :] - src_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    
    k = min(k, len(src_coords))
    neighbors = np.argsort(dist2, axis=1)[:, :k]
    
    dst_indices = np.repeat(np.arange(len(dst_coords)), k)
    src_indices = neighbors.reshape(-1)
    return torch.from_numpy(np.stack([src_indices, dst_indices], axis=0)).long()


def main():
    parser = argparse.ArgumentParser(description="Ghost Nodes Inference GIF over Time")
    parser.add_argument("--cctv_nc", type=str, default='../sample/processed/cctv_processed.nc')
    parser.add_argument("--skt_nc", type=str, default='../sample/processed/skt_processed.nc')
    parser.add_argument("--checkpoint", type=str, default='outputs/exp_2/best.pt', help="Path to the trained .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--grid_resolution", type=int, default=30, help="Resolution for Ghost nodes across SKT bounds")
    parser.add_argument("--max_frames", type=int, default=256, help="Number of time steps to animate")
    parser.add_argument("--from_end", type=int, default=1, help="1이면 뒤에서부터, 0이면 앞에서부터 프레임 선택")
    parser.add_argument("--fps", type=int, default=8, help="GIF frames per second")
    args = parser.parse_args()
    args.from_end = args.from_end == 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 원본 데이터셋 로드 (back_steps=24 고정)
    dataset = CrowdDataset(args.cctv_nc, args.skt_nc, ghost_holdout_ratio=0.0, back_steps=24, lead_steps=24)
    cctv_vals, skt_vals = dataset.get_scaler_values()
    cctv_scaler = StandardScaler.fit(cctv_vals).to(device)
    skt_scaler = StandardScaler.fit(skt_vals).to(device)

    # Test 셋 분할
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    test_idx = list(range(n_train + n_val, n_total))
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    meta = dataset.get_meta()
    real_lons, real_lats = meta.cctv_lon, meta.cctv_lat
    skt_lons, skt_lats = meta.skt_lon, meta.skt_lat
    skt_pos = torch.from_numpy(np.column_stack([skt_lons, skt_lats]).astype(np.float32)).to(device)
    n_real = len(real_lons)
    n_skt = len(skt_lons)

    # ---------------------------------------------------------
    # 🔥 2. SKT 영역 기준 Ghost 노드 생성 & Decay 완화
    # ---------------------------------------------------------
    lon_min, lon_max = skt_lons.min(), skt_lons.max()
    lat_min, lat_max = skt_lats.min(), skt_lats.max()

    glon, glat = np.meshgrid(np.linspace(lon_min, lon_max, args.grid_resolution), 
                             np.linspace(lat_min, lat_max, args.grid_resolution))
    ghost_lons, ghost_lats = glon.reshape(-1), glat.reshape(-1)
    n_ghost = len(ghost_lons)
    print(f"[*] Generated {n_ghost} Ghost nodes spanning the SKT boundaries.")

    all_lons = np.concatenate([real_lons, ghost_lons])
    all_lats = np.concatenate([real_lats, ghost_lats])
    all_pos = torch.from_numpy(np.column_stack([all_lons, all_lats]).astype(np.float32)).to(device)
    
    # CCTV <-> CCTV (Real + Ghost 통합) 엣지
    new_edge_index_cctv = build_knn_edges_dynamic(all_lons, all_lats, k=4).to(device)

    # 🔥 외곽 Ghost 노드 살리기 1: SKT -> Ghost 다이렉트 엣지 추가
    # 기존 skt2cctv 엣지에 skt2ghost 엣지를 이어붙임
    edge_index_skt2ghost = build_bipartite_knn_edges(skt_lons, skt_lats, ghost_lons, ghost_lats, k=1)
    # Ghost 인덱스는 Real 노드 개수(n_real)만큼 뒤로 밀려있으므로 오프셋 더해주기
    edge_index_skt2ghost[1] += n_real 
    
    original_skt2cctv = meta.edge_index_skt2cctv
    new_edge_index_skt2cctv = torch.cat([original_skt2cctv, edge_index_skt2ghost], dim=1).to(device)

    # 🔥 외곽 Ghost 노드 살리기 2: Decay 하한선 설정
    real_coords = np.column_stack([real_lons, real_lats])
    ghost_coords = np.column_stack([ghost_lons, ghost_lats])
    ghost_nn_indices = []
    ghost_nn_weights = []
    ghost_decays = []
    
    for i in range(n_ghost):
        dist2 = np.sum((real_coords - ghost_coords[i])**2, axis=1)
        dist2 = np.maximum(dist2, 1e-8)
        nn_idx = np.argsort(dist2)[:4]
        nearest_dist2 = dist2[nn_idx]
        
        w = 1.0 / np.sqrt(nearest_dist2)
        w = w / np.sum(w)
        
        ghost_nn_indices.append(nn_idx)
        ghost_nn_weights.append(w)
        
        # 감쇄율(Decay) 계산 후, 최소 20%의 볼륨은 유지하도록 하한선(Clipping) 설정
        min_dist_deg = np.sqrt(nearest_dist2[0])
        raw_decay = np.exp(-min_dist_deg / 0.005) # 감쇄 반경도 500m로 넓힘
        decay_val = max(raw_decay, 0.2) 
        ghost_decays.append(decay_val) 

    # 3. 모델 로드
    model = CrowdMPNN(back_steps=24, hidden_dim=128, n_passing=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # 4. 추론 루프
    frames = []
    with torch.no_grad():
        total_test_frames = len(test_loader)
        if args.from_end:
            selected_indices = set(range(max(total_test_frames - args.max_frames, 0), total_test_frames))
            print(f"[*] Selecting last {min(args.max_frames, total_test_frames)} frames from the test split.")
        else:
            selected_indices = set(range(0, min(args.max_frames, total_test_frames)))
            print(f"[*] Selecting first {min(args.max_frames, total_test_frames)} frames from the test split.")

        for i, batch in enumerate(test_loader):
            if i not in selected_indices:
                continue
                
            real_cctv_x = batch["cctv_x"].to(device)
            skt_x = batch["skt_x"].to(device)
            time_val = batch["time"].detach().cpu().numpy()[0]
            
            ghost_x = torch.zeros(1, n_ghost, real_cctv_x.shape[2], 1, device=device)
            for j in range(n_ghost):
                blended = torch.zeros_like(real_cctv_x[0, 0])
                for k, r_idx in enumerate(ghost_nn_indices[j]):
                    blended += real_cctv_x[0, r_idx] * ghost_nn_weights[j][k]
                ghost_x[0, j] = blended * ghost_decays[j]

            all_cctv_x = torch.cat([real_cctv_x, ghost_x], dim=1)
            
            real_x_scaled = cctv_scaler.transform(real_cctv_x) # 원래의 학습 기준 적용
            ghost_x_scaled = cctv_scaler.transform(ghost_x)    # 같은 스케일러로 변환

            # 다시 합침
            cctv_x_scaled = torch.cat([real_x_scaled, ghost_x_scaled], dim=1)
            skt_x_scaled = skt_scaler.transform(skt_x)
            
            # 🔥 업데이트된 SKT->통합 엣지 인덱스 사용
            pred_scaled = model(cctv_x_scaled, skt_x_scaled, all_pos, skt_pos, new_edge_index_cctv, new_edge_index_skt2cctv)
            pred_real = cctv_scaler.inverse(pred_scaled).detach().cpu().numpy()[0, :, 0]
            
            target_real = batch["target"].detach().cpu().numpy()[0, :, 0]
            time_label = pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")
            
            frames.append({
                "pred_ghost": pred_real[n_real:],
                "pred_real": pred_real[:n_real],
                "target_real": target_real,
                "time_label": time_label
            })
            print(f"[*] Processed frame {i+1}/{total_test_frames} ({time_label})")

    # ---------------------------------------------------------
    # 🔥 5. GIF 렌더링
    # ---------------------------------------------------------
    print("[*] Rendering GIF animation...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    
    all_target_vals = np.concatenate([f["target_real"] for f in frames])
    all_pred_vals = np.concatenate([f["pred_ghost"] for f in frames])
    vmax = float(np.nanmax(np.concatenate([all_target_vals, all_pred_vals])))
    
    lon_pad, lat_pad = (lon_max - lon_min) * 0.05, (lat_max - lat_min) * 0.05

    def setup_panel(ax, title):
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, alpha=0.85)
        except Exception:
            pass

    setup_panel(axes[0], "Prediction on Real CCTV Nodes")
    setup_panel(axes[1], f"Prediction on Real CCTV Nodes + {n_ghost} Ghost Nodes")

    scat_real_pred = axes[0].scatter(real_lons, real_lats, c=frames[0]["pred_real"], cmap="YlOrRd",
                                     s=150, marker='o', edgecolors="black", linewidths=1.0, vmin=0, vmax=vmax, zorder=5)
    
    scat_ghost = axes[1].scatter(ghost_lons, ghost_lats, c=frames[0]["pred_ghost"], cmap="YlOrRd", 
                                 s=150, marker='o', edgecolors="black", linewidths=0.5, vmin=0, vmax=vmax, zorder=4, label="Ghost Nodes")
    
    scat_real = axes[1].scatter(real_lons, real_lats, c=frames[0]["pred_real"], cmap="YlOrRd",
                                s=150, marker='o', edgecolors="black", linewidths=1.0, vmin=0, vmax=vmax, zorder=5, label="Real CCTV")
    axes[1].legend(loc="upper right", fontsize=12)

    fig.colorbar(scat_real_pred, ax=axes[0], label="Crowd Count", shrink=0.8)
    fig.colorbar(scat_ghost, ax=axes[1], label="Crowd Count", shrink=0.8)
    
    time_text = fig.suptitle(f"Crowd Network Prediction | Time: {frames[0]['time_label']}", fontsize=22, fontweight="bold")

    def update(frame_idx):
        frame = frames[frame_idx]
        scat_real_pred.set_array(frame["pred_real"])
        scat_ghost.set_array(frame["pred_ghost"])
        scat_real.set_array(frame["pred_real"])
        time_text.set_text(f"Crowd Network Prediction | Time: {frame['time_label']}")
        return scat_real_pred, scat_ghost, scat_real, time_text

    gif_path = output_dir / f"ghost_network_animation.gif"
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    
    print(f"[*] Success! Ghost Network GIF saved at: {gif_path}")

if __name__ == "__main__":
    main()