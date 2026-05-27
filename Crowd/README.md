# Crowd Prediction (GNN)

This folder contains a minimal crowd prediction model that mirrors the weather-model MPNN design:
- Internal message passing on CCTV nodes.
- External message passing from SKT nodes to CCTV nodes.
- Ghost-node holdout and initialization for generalization.

## Data
Uses the processed netCDF files in `sample/processed`:
- `cctv_processed.nc`
- `skt_processed.nc`

## Quick start
```bash
cd Crowd
python quick_run.py \
  --cctv_nc ../sample/processed/cctv_processed.nc \
  --skt_nc ../sample/processed/skt_processed.nc
```

## Train
```bash
cd Crowd
python train.py \
  --cctv_nc ../sample/processed/cctv_processed.nc \
  --skt_nc ../sample/processed/skt_processed.nc \
  --epochs 3
```

Outputs are saved under `Crowd/outputs/<run_name>/`.
