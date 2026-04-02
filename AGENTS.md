# LocalizedWeather Agent Guide

This document is for the next agent that needs to continue work in this repository without reconstructing the context from scratch.

## Scope

This project has two active workflows:

1. The original research training and evaluation code under `Source/`
2. The Triton inference and browser GUI path for real-data serving

Before making changes, decide which workflow you are touching. The research code and the Triton deployment path are related, but they are not the same system.

## Files To Check First

- [README.md](README.md): Main project documentation
- [TRITON_README_KR.md](TRITON_README_KR.md): Korean Triton-only runbook
- [gui/triton_gui.py](gui/triton_gui.py): Real-data MADIS/ERA5 browser GUI
- [triton_model_repository/weather_model/1/model.py](triton_model_repository/weather_model/1/model.py): Triton Python backend
- [triton_model_repository/weather_model/config.pbtxt](triton_model_repository/weather_model/config.pbtxt): Triton input/output contract
- [triton_model_repository/test_infer_weather.py](triton_model_repository/test_infer_weather.py): Synthetic Triton smoke test

If you only need to understand the deployment path, read the Triton files first. If you are changing the original model or data pipeline, start in `Source/`.

## Data Paths

The real-data GUI uses these files:

- `WindDataNE-US/madis/processed/Meta-2019-2023/madis_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc`
- `WindDataNE-US/ERA5/Processed/era5_2023_e2m_8_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.nc`
- `WindDataNE-US/Shapefiles/Regions/northeastern_buffered.shp`

If the repository is moved, update the hardcoded paths in [gui/triton_gui.py](gui/triton_gui.py) as part of the move. Do not assume the GUI will discover the new location automatically.

## Current Run Flow

### Triton

1. `cd triton_model_repository`
2. `docker build -t weather-triton-server:latest .`
3. `docker run --gpus=all --rm --name localized-weather-triton -p 18000:8000 -p 18001:8001 -p 18002:8002 -v "$PWD:/models" weather-triton-server:latest tritonserver --model-repository=/models`
4. `python triton_model_repository/test_infer_weather.py`

### GUI

1. `python gui/triton_gui.py`
2. Open `http://127.0.0.1:8080`
3. Choose a channel and click `Run inference`

If the GUI fails at startup, check `netCDF4`, the NetCDF file paths, and whether port `8080` is already in use.

## Operating Rules

- Use `apply_patch` for file edits.
- Do not overwrite user changes unless explicitly asked.
- Keep changes minimal and targeted.
- Prefer fixing the root cause over adding a temporary workaround.
- Keep README content separated by purpose. The main README should stay general; Triton-specific usage belongs in `TRITON_README_KR.md` or a separate Triton-only document.

## Implementation Notes

- The GUI uses real 2023 MADIS/ERA5 NetCDF files, not synthetic inputs.
- `geopandas` is optional at runtime. If it is unavailable, the GUI still runs, but boundary rendering is skipped.
- `netCDF4` is required for reading the processed NetCDF files with `xarray` in the current GUI path.
- The Triton backend loads `best_model.pt` and reconstructs the `MPNN` architecture from the checkpoint state dict.
- The Triton deployment path depends on the model repository layout remaining stable.

## Troubleshooting Checklist

### Triton does not respond

- Confirm the Triton container is running.
- Check `http://localhost:18000/v2/health/ready`.
- Run `test_infer_weather.py` before trying the GUI.

### GUI fails to start

- Verify `netCDF4` is installed in the environment used to launch the GUI.
- Verify the MADIS and ERA5 file paths exist.
- Check whether port `8080` is already occupied.

### Boundary outline is missing

- `geopandas` may not be installed.
- This is acceptable; the GUI should still function without the boundary overlay.

### Triton inference request fails from the GUI

- Make sure Triton was started before the GUI.
- Verify the Triton model repository path.
- Confirm the backend model and `config.pbtxt` still agree on the input names and shapes.

## Recommended Edit Order

1. Make the smallest possible code change in the relevant file.
2. Run `get_errors` on the file you changed.
3. If the change affects Triton or GUI behavior, verify it end-to-end.
4. Leave a short note in a repo memory file if you discover a recurring issue or a useful workaround.

## Reference Documents

- [TRITON_README_KR.md](TRITON_README_KR.md): Detailed Korean Triton runbook
- [README.md](README.md): General project overview and research workflow

## What To Preserve

If you need to modify the deployment path, preserve these conventions unless the user explicitly asks otherwise:

- Triton model name: `weather_model`
- GUI port: `8080`
- Triton ports: `18000`, `18001`, `18002`
- Real-data input files from `WindDataNE-US/`
- The existing distinction between the research workflow and the Triton workflow
