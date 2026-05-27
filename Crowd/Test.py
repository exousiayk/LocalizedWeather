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
from torch.utils.data import DataLoader, Subset

# 지도 배경을 위한 라이브러리 추가
import contextily as cx

sys.path.append(str(Path(__file__).resolve().parent))
from data import CrowdDataset
from model import CrowdMPNN
from normalizer import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Crowd Inference and GIF Visualization")
    parser.add_argument("--cctv_nc", type=str, default='../sample/processed/cctv_processed.nc')
    parser.add_argument("--skt_nc", type=str, default='../sample/processed/skt_processed.nc')
    parser.add_argument("--checkpoint", type=str, default='outputs/exp_1/best.pt', help="Path to the trained .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/inference")
    parser.add_argument("--ghost_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--back_steps", type=int, default=24)
    parser.add_argument("--lead_steps", type=int, default=24)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_passing", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=72)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrowdDataset(
        args.cctv_nc,
        args.skt_nc,
        back_steps=args.back_steps,
        lead_steps=args.lead_steps,
        ghost_holdout_ratio=args.ghost_holdout_ratio,
    )

    cctv_vals, skt_vals = dataset.get_scaler_values()
    cctv_scaler = StandardScaler.fit(cctv_vals)
    skt_scaler = StandardScaler.fit(skt_vals)

    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    test_idx = list(range(n_train + n_val, n_total))
    
    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cctv_scaler.to(device)
    skt_scaler.to(device)

    meta = dataset.get_meta()
    cctv_pos = torch.from_numpy(np.column_stack([meta.cctv_lon, meta.cctv_lat]).astype(np.float32)).to(device)
    skt_pos = torch.from_numpy(np.column_stack([meta.skt_lon, meta.skt_lat]).astype(np.float32)).to(device)
    edge_index_cctv = meta.edge_index_cctv.to(device)
    edge_index_skt2cctv = meta.edge_index_skt2cctv.to(device)

    seen_mask_np = meta.seen_mask
    ghost_mask_np = meta.ghost_mask
    lons = meta.cctv_lon
    lats = meta.cctv_lat
    skt_lons = meta.skt_lon
    skt_lats = meta.skt_lat

    model = CrowdMPNN(
        back_steps=args.back_steps,
        hidden_dim=args.hidden_dim,
        n_passing=args.n_passing,
    ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[*] Successfully loaded checkpoint: {args.checkpoint}")

    t_seen_abs, t_seen_sq, t_seen_t_sum, t_seen_cnt = 0.0, 0.0, 0.0, 0.0
    t_ghost_abs, t_ghost_sq, t_ghost_t_sum, t_ghost_cnt = 0.0, 0.0, 0.0, 0.0
    frames = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            cctv_x = batch["cctv_x"].to(device)
            skt_x = batch["skt_x"].to(device)
            target = batch["target"].to(device)
            target_is_real = batch["target_is_real"].to(device)
            seen_mask = batch["seen_mask"].to(device)
            ghost_mask = batch["ghost_mask"].to(device)
            time_val = batch["time"].detach().cpu().numpy()[0]

            cctv_x_scaled = cctv_scaler.transform(cctv_x)
            skt_x_scaled = skt_scaler.transform(skt_x)

            pred_scaled = model(cctv_x_scaled, skt_x_scaled, cctv_pos, skt_pos, edge_index_cctv, edge_index_skt2cctv)
            pred_real = cctv_scaler.inverse(pred_scaled)

            seen_eval = (target_is_real > 0.5) & (seen_mask > 0.5)
            ghost_eval = (target_is_real > 0.5) & (ghost_mask > 0.5)

            if seen_eval.any():
                p = pred_real[seen_eval]
                t = target[seen_eval]
                t_seen_abs += torch.abs(p - t).sum().item()
                t_seen_sq += ((p - t) ** 2).sum().item()
                t_seen_t_sum += t.sum().item()
                t_seen_cnt += seen_eval.sum().item()

            if ghost_eval.any():
                p = pred_real[ghost_eval]
                t = target[ghost_eval]
                t_ghost_abs += torch.abs(p - t).sum().item()
                t_ghost_sq += ((p - t) ** 2).sum().item()
                t_ghost_t_sum += t.sum().item()
                t_ghost_cnt += ghost_eval.sum().item()

            if i < args.max_frames:
                pred_np = pred_real.detach().cpu().numpy()[0, :, 0]
                target_np = target.detach().cpu().numpy()[0, :, 0]
                time_label = pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")
                frames.append({"pred": pred_np, "target": target_np, "time_label": time_label})

    seen_mae = t_seen_abs / t_seen_cnt if t_seen_cnt > 0 else float('nan')
    seen_rmse = np.sqrt(t_seen_sq / t_seen_cnt) if t_seen_cnt > 0 else float('nan')
    seen_wmape = t_seen_abs / t_seen_t_sum if t_seen_t_sum > 0 else float('nan')
    ghost_mae = t_ghost_abs / t_ghost_cnt if t_ghost_cnt > 0 else float('nan')
    ghost_rmse = np.sqrt(t_ghost_sq / t_ghost_cnt) if t_ghost_cnt > 0 else float('nan')
    ghost_wmape = t_ghost_abs / t_ghost_t_sum if t_ghost_t_sum > 0 else float('nan')

    print("\n" + "="*50)
    print(" 🎯 FINAL TEST SET EVALUATION")
    print("="*50)
    print(f"SEEN  NODES | MAE: {seen_mae:.2f}, RMSE: {seen_rmse:.2f}, WMAPE: {seen_wmape:.4f}")
    print(f"GHOST NODES | MAE: {ghost_mae:.2f}, RMSE: {ghost_rmse:.2f}, WMAPE: {ghost_wmape:.4f}")
    print("="*50 + "\n")

    if not frames:
        print("[!] No frames to render.")
        return

    print(f"[*] Rendering GIF animation with {len(frames)} frames. This might take a moment to download map tiles...")
    
    # ---------------------------------------------------------
    # 🔥 시야각(Bounds)을 SKT 기준으로 설정하고 여백 추가
    # ---------------------------------------------------------
    all_lons_for_bounds = np.concatenate([lons, skt_lons])
    all_lats_for_bounds = np.concatenate([lats, skt_lats])
    
    lon_min, lon_max = all_lons_for_bounds.min(), all_lons_for_bounds.max()
    lat_min, lat_max = all_lats_for_bounds.min(), all_lats_for_bounds.max()
    
    # 상하좌우 10% 여백 (Padding)
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), constrained_layout=True)
    
    def setup_panel(ax, title):
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
        
        # 실제 지도 배경 추가 (CartoDB Positron: 분석용으로 깔끔한 밝은 테마)
        try:
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, alpha=0.7)
        except Exception as e:
            print(f"[!] 지도 타일을 불러오지 못했습니다: {e}")

    setup_panel(axes[0], "Ground Truth (Real Crowd)")
    setup_panel(axes[1], "Prediction (Seen + Ghost)")
    setup_panel(axes[2], "Absolute Error (Prediction vs GT)")

    all_target_vals = np.concatenate([f["target"] for f in frames])
    all_pred_vals = np.concatenate([f["pred"] for f in frames])
    vmax_crowd = float(np.nanmax(np.concatenate([all_target_vals, all_pred_vals])))
    vmax_error = float(np.nanmax(np.abs(all_pred_vals - all_target_vals)))

    # ---------------------------------------------------------
    # 🔥 마커 사이즈 3배 이상 확대 및 테두리(edgecolors) 강화
    # ---------------------------------------------------------
    scat_gt = axes[0].scatter(lons, lats, c=frames[0]["target"], cmap="YlOrRd", vmin=0, vmax=vmax_crowd, 
                              s=150, edgecolors="black", linewidths=1.0, zorder=5, alpha=0.9)
    
    scat_pred_seen = axes[1].scatter(lons[seen_mask_np], lats[seen_mask_np], c=frames[0]["pred"][seen_mask_np], 
                                     cmap="YlOrRd", vmin=0, vmax=vmax_crowd, s=150, edgecolors="black", 
                                     linewidths=1.0, label="Seen", zorder=5, alpha=0.9)
    
    scat_pred_ghost = axes[1].scatter(lons[ghost_mask_np], lats[ghost_mask_np], c=frames[0]["pred"][ghost_mask_np], 
                                      cmap="YlOrRd", vmin=0, vmax=vmax_crowd, s=350, marker="*", edgecolors="blue", 
                                      linewidths=1.5, label="Ghost (Target)", zorder=6, alpha=1.0)
    axes[1].legend(loc="upper right", fontsize=12)

    abs_err = np.abs(frames[0]["pred"] - frames[0]["target"])
    scat_err = axes[2].scatter(lons, lats, c=abs_err, cmap="Reds", vmin=0, vmax=vmax_error, 
                               s=150, edgecolors="black", linewidths=1.0, zorder=5, alpha=0.9)

    fig.colorbar(scat_gt, ax=axes[0], label="Crowd Count")
    fig.colorbar(scat_pred_seen, ax=axes[1], label="Crowd Count")
    fig.colorbar(scat_err, ax=axes[2], label="Absolute Error")

    time_text = fig.suptitle(f"Crowd Inference | Time: {frames[0]['time_label']} | Ghost Ratio: {args.ghost_holdout_ratio:.2f}", fontsize=18, fontweight="bold")

    def update(frame_idx):
        frame = frames[frame_idx]
        scat_gt.set_array(frame["target"])
        scat_pred_seen.set_array(frame["pred"][seen_mask_np])
        scat_pred_ghost.set_array(frame["pred"][ghost_mask_np])
        
        current_err = np.abs(frame["pred"] - frame["target"])
        scat_err.set_array(current_err)
        
        time_text.set_text(f"Crowd Inference | Time: {frame['time_label']} | Ghost Ratio: {args.ghost_holdout_ratio:.2f}")
        return scat_gt, scat_pred_seen, scat_pred_ghost, scat_err, time_text

    gif_path = output_dir / f"crowd_inference_map_ghost_{args.ghost_holdout_ratio:.2f}.gif"
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    
    print(f"[*] Inference GIF with Map Background saved successfully at: {gif_path}")

if __name__ == "__main__":
    main()