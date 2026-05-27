#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))
from data import CrowdDataset
from model import CrowdMPNN
from normalizer import StandardScaler


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Seen/Ghost 유효 영역에 대해 직관적인 MAE, RMSE, WMAPE 지표만 정밀 계산 (MSE 제거)"""
    mask = mask.float()
    
    # 마스킹 처리 (유효 노드만 추출)
    active_preds = pred[mask > 0.5]
    active_targets = target[mask > 0.5]
    
    if len(active_targets) == 0:
        return {"mae": 0.0, "rmse": 0.0, "wmape": 0.0}
    
    # 1. MAE (명 단위)
    mae = torch.mean(torch.abs(active_preds - active_targets)).item()
    
    # 2. RMSE (명 단위)
    mse_val = torch.mean((active_preds - active_targets) ** 2).item()
    rmse = np.sqrt(mse_val)
    
    # 3. WMAPE (%)
    sum_abs_err = torch.sum(torch.abs(active_preds - active_targets)).item()
    sum_target = torch.sum(torch.abs(active_targets)).item()
    wmape = (sum_abs_err / (sum_target + 1e-5)) * 100 
    
    return {"mae": mae, "rmse": rmse, "wmape": wmape}


def main():
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal GNN with Live Plotting and Full Metrics")
    parser.add_argument("--cctv_nc", type=str, default='../sample/processed/cctv_processed.nc')
    parser.add_argument("--skt_nc", type=str, default='../sample/processed/skt_processed.nc')
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--name", type=str, default="exp_base", help="Experiment name")
    
    parser.add_argument("--back_steps", type=int, default=24)
    parser.add_argument("--lead_steps", type=int, default=24)
    parser.add_argument("--ghost_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--ghost_split_seed", type=int, default=42)
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_passing", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sensor_dropout", type=int, default=1)
    parser.add_argument("--sensor_dropout_ratio", type=float, default=0.2)
    args = parser.parse_args()
    args.sensor_dropout = args.sensor_dropout == 1

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(output_dir, 0o777)
    except Exception:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using Device: {device}")

    print("[*] Loading datasets...")
    dataset = CrowdDataset(
        args.cctv_nc,
        args.skt_nc,
        back_steps=args.back_steps,
        lead_steps=args.lead_steps,
        ghost_holdout_ratio=args.ghost_holdout_ratio,
        ghost_seed=args.ghost_split_seed,
    )

    cctv_vals, skt_vals = dataset.get_scaler_values()
    cctv_scaler = StandardScaler.fit(cctv_vals).to(device)
    skt_scaler = StandardScaler.fit(skt_vals).to(device)

    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    meta = dataset.get_meta()
    cctv_pos = torch.from_numpy(np.column_stack([meta.cctv_lon, meta.cctv_lat]).astype(np.float32)).to(device)
    skt_pos = torch.from_numpy(np.column_stack([meta.skt_lon, meta.skt_lat]).astype(np.float32)).to(device)
    edge_index_cctv = meta.edge_index_cctv.to(device)
    edge_index_skt2cctv = meta.edge_index_skt2cctv.to(device)

    model = CrowdMPNN(
        back_steps=args.back_steps,
        hidden_dim=args.hidden_dim,
        n_passing=args.n_passing,
        dropout_rate=args.dropout_rate,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 💡 실제 사람 수 오차인 Seen MAE를 추적하기 위해 무한대로 초기화
    best_val_loss = float('inf') 
    best_epoch = 0
    epochs_no_improve = 0
    
    history_logs = []
    train_loss_curve = []
    val_loss_curve = []

    print(f"[*] Starting training for {args.epochs} epochs... (Patience: {args.patience})")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            cctv_x = batch["cctv_x"].to(device)
            skt_x = batch["skt_x"].to(device)
            target = batch["target"].to(device)
            seen_mask = batch["seen_mask"].to(device)

            cctv_x_scaled = cctv_scaler.transform(cctv_x)
            skt_x_scaled = skt_scaler.transform(skt_x)

            optimizer.zero_grad()
            pred_scaled = model(cctv_x_scaled, skt_x_scaled, cctv_pos, skt_pos, edge_index_cctv, edge_index_skt2cctv)
            pred_real = cctv_scaler.inverse(pred_scaled)

            # Sensor Dropout 동적 마스킹
            train_mask = seen_mask > 0.5
            if args.sensor_dropout and args.sensor_dropout_ratio > 0:
                drop_mask = torch.rand_like(train_mask.float()) < args.sensor_dropout_ratio
                train_mask = train_mask & (~drop_mask)
                
            # 역전파(Backprop)용 손실 연산은 안정적인 학습을 위해 기존 수식 유지
            diff = pred_real - target
            diff = torch.where(train_mask, diff, torch.zeros_like(diff))
            loss = (diff ** 2).sum() / train_mask.float().sum().clamp(min=1.0)
            
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        epoch_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0

        # --- VALIDATION PHASE (완전 분리 다중 지표 검증) ---
        model.eval()
        all_preds, all_targets, all_seens, all_ghosts = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                cctv_x = batch["cctv_x"].to(device)
                skt_x = batch["skt_x"].to(device)
                target = batch["target"].to(device)
                seen_mask = batch["seen_mask"].to(device)
                ghost_mask = batch["ghost_mask"].to(device)

                cctv_x_scaled = cctv_scaler.transform(cctv_x)
                skt_x_scaled = skt_scaler.transform(skt_x)

                pred_scaled = model(cctv_x_scaled, skt_x_scaled, cctv_pos, skt_pos, edge_index_cctv, edge_index_skt2cctv)
                pred_real = cctv_scaler.inverse(pred_scaled)

                all_preds.append(pred_real)
                all_targets.append(target)
                all_seens.append(seen_mask)
                all_ghosts.append(ghost_mask)

        val_preds = torch.cat(all_preds, dim=0)
        val_targets = torch.cat(all_targets, dim=0)
        val_seens = torch.cat(all_seens, dim=0)
        val_ghosts = torch.cat(all_ghosts, dim=0)

        # Seen / Ghost 분리 3대 지표 연산 (MSE 완전 탈락)
        seen_metrics = calculate_metrics(val_preds, val_targets, val_seens)
        ghost_metrics = calculate_metrics(val_preds, val_targets, val_ghosts)

        # 💡 시각화 커브용 타겟을 가장 직관적인 'Seen MAE' 지표로 교체 매핑
        epoch_val_loss_for_monitor = seen_metrics["mae"]

        train_loss_curve.append(epoch_train_loss)
        val_loss_curve.append(epoch_val_loss_for_monitor)

        # 🔥 [출력 최적화] 터미널 구분선 내부에서 가독성 떨어지던 MSE 항목 전면 삭제
        print(f"\n==================== [ Epoch {epoch:03d} / {args.epochs:03d} ] ====================")
        print(f"[*] Train Loss (Scaled MSE) : {epoch_train_loss:.4f}")
        print(f"[-] VALIDATION SEEN  -> MAE: {seen_metrics['mae']:.2f}명 | RMSE: {seen_metrics['rmse']:.2f}명 | WMAPE: {seen_metrics['wmape']:.2f}%")
        print(f"[-] VALIDATION GHOST -> MAE: {ghost_metrics['mae']:.2f}명 | RMSE: {ghost_metrics['rmse']:.2f}명 | WMAPE: {ghost_metrics['wmape']:.2f}%")

        # --- BEST MODEL CHECK & SAVING ---
        # 💡 착시 및 역전 현상 원천 차단: 의사결정 기준을 'Seen MAE(명 수 오차)'로 고정 통일
        current_eval_metric = seen_metrics["mae"]
        
        if current_eval_metric < best_val_loss:
            best_val_loss = current_eval_metric
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"[*] >>> 🏆 New Best Model Saved at Epoch {epoch} (Seen MAE: {best_val_loss:.2f}명) <<<")
        else:
            if args.patience > 0:  # 안전 조건식 래핑
                epochs_no_improve += 1
                print(f"[*] No improvement in validation MAE for {epochs_no_improve} epoch(s).")

        # 🔥 [실시간 시각화] 축 라벨명 변경 및 Seen MAE 기반으로 실시간 플롯 업데이트
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_curve) + 1), train_loss_curve, label="Train Loss (Scaled)", color="blue", linewidth=1.5)
        plt.plot(range(1, len(val_loss_curve) + 1), val_loss_curve, label="Val Loss (Seen MAE)", color="red", linewidth=1.5)
        plt.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
        plt.title(f"Real-time Loss & Metric Curve Update [Epoch {epoch}]", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Value (Scaled / People Count)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        curve_img_path = output_dir / "loss_curve.png"
        plt.savefig(curve_img_path, dpi=150)
        plt.close()

        # CSV 파일 백업 데이터에서 불필요한 MSE 열 완벽 청소
        history_logs.append({
            "Epoch": epoch,
            "Train_Loss": epoch_train_loss,
            "Seen_MAE": seen_metrics["mae"], "Seen_RMSE": seen_metrics["rmse"], "Seen_WMAPE": seen_metrics["wmape"],
            "Ghost_MAE": ghost_metrics["mae"], "Ghost_RMSE": ghost_metrics["rmse"], "Ghost_WMAPE": ghost_metrics["wmape"],
            "Status": "Best" if epoch == best_epoch else "Running"
        })

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
            break

    end_time = time.time()
    print(f"\n[*] Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # --- CSV 최종 기록 및 최하단 6대 핵심 평가지표 요약행 결합 ---
    print("[*] Processing training logs to CSV...")
    df_logs = pd.DataFrame(history_logs)
    
    if best_epoch > 0:
        best_row = df_logs.loc[df_logs["Epoch"] == best_epoch].iloc[0]
        best_summary = {
            "Epoch": f"BEST_EPOCH_{best_epoch}",
            "Train_Loss": best_row["Train_Loss"],
            "Seen_MAE": best_row["Seen_MAE"], "Seen_RMSE": best_row["Seen_RMSE"], "Seen_WMAPE": best_row["Seen_WMAPE"],
            "Ghost_MAE": best_row["Ghost_MAE"], "Ghost_RMSE": best_row["Ghost_RMSE"], "Ghost_WMAPE": best_row["Ghost_WMAPE"],
            "Status": "FINAL_BEST_METRIC"
        }
    else:
        best_summary = {"Epoch": "NONE", "Status": "FINAL_BEST_METRIC"}
    
    df_logs = pd.concat([df_logs, pd.DataFrame([best_summary])], ignore_index=True)
    
    csv_output_path = output_dir / "learning_history.csv"
    df_logs.to_csv(csv_output_path, index=False)
    
    try:
        os.chmod(csv_output_path, 0o777)
        os.chmod(output_dir / "best.pt", 0o777)
        os.chmod(curve_img_path, 0o777)
    except Exception:
        pass
        
    print(f"[*] Success! Refined Learning history CSV saved at: {csv_output_path}")


if __name__ == "__main__":
    main()