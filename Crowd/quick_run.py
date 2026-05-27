from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from data import CrowdDataset
from model import CrowdMPNN


def main():
    parser = argparse.ArgumentParser(description="Crowd quick smoke test")
    parser.add_argument("--cctv_nc", type=str, required=True)
    parser.add_argument("--skt_nc", type=str, required=True)
    parser.add_argument("--back_steps", type=int, default=48)
    parser.add_argument("--lead_steps", type=int, default=12)
    args = parser.parse_args()

    dataset = CrowdDataset(
        args.cctv_nc,
        args.skt_nc,
        back_steps=args.back_steps,
        lead_steps=args.lead_steps,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid windows found after filtering missing/blackout periods")
    meta = dataset.get_meta()

    sample = dataset[0]
    cctv_x = sample["cctv_x"].unsqueeze(0)
    skt_x = sample["skt_x"].unsqueeze(0)

    cctv_pos = torch.from_numpy(np.column_stack([meta.cctv_lon, meta.cctv_lat]).astype(np.float32))
    skt_pos = torch.from_numpy(np.column_stack([meta.skt_lon, meta.skt_lat]).astype(np.float32))

    model = CrowdMPNN(back_steps=args.back_steps)
    pred = model(cctv_x, skt_x, cctv_pos, skt_pos, meta.edge_index_cctv, meta.edge_index_skt2cctv)
    print("cctv_x", cctv_x.shape)
    print("skt_x", skt_x.shape)
    print("pred", pred.shape)


if __name__ == "__main__":
    main()
