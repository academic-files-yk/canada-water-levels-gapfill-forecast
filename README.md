# Gap filling and one-day-ahead forecasting of daily water levels at Canadian hydrometric stations

Code and station-level results for the manuscript

> Kaya, Y. *Gap filling and one-day-ahead forecasting of daily water levels at 3,645 Canadian hydrometric stations: interpolation, model comparison under a common protocol, and trends.* Journal of Hydrology (under review, manuscript HYDROL65986).

The workflow starts from the daily water level archive of the Water Survey of Canada (3,645 stations, 21,839,295 station-days) and reproduces every number, table and figure of the manuscript.

## Contents

| Path | Content |
|---|---|
| `src/` | Analysis scripts (Python 3.13), numbered in the order in which they are run |
| `results/` | Station-level results used in the manuscript |
| `figures/` | Figures 1–8 of the manuscript (PNG) |
| `data/raw/` | Location of the raw archive snapshot (download from the release, see below) |
| `requirements.txt` | Pinned library versions |

| Script | Purpose | Main output (`src/output/`) |
|---|---|---|
| `config.py` | Paths and shared constants | |
| `01_prepare_series.py` | Screening, basin assignment, gap filling (gaps of up to 14 days, linear), segmentation | `station_flow.csv`, `station_metadata.csv`, `work/series/` |
| `02_gap_length_distribution.py` | Lengths of all real gaps | `gap_lengths.csv` |
| `03_interpolation_experiment.py` | Paired interpolation experiment (178 complete-record stations; random and realistic gaps; 5–20%) | `interpolation_runs.csv` |
| `04_interpolation_gap_length.py` | Interpolation error as a function of gap length | `gap_length_runs.csv` |
| `05_interpolation_statistics.py` | Paired Wilcoxon tests with Holm correction and paired bootstrap intervals | `interpolation_stats.csv`, `gap_length_stats.csv` |
| `06_forecast_pipeline.py` | One-day-ahead forecasts: persistence, ARIMA, RF, XGBoost, LightGBM, LSTM | `work/results*/`, `work/preds*/` |
| `07_collect_results.py` | Collects the station results | `forecast_metrics_long.csv`, `forecast_status.csv` |
| `08_data_flow_table.py` | Stations retained at each stage (Table 1) | `data_flow_table.csv`, `data_flow_basins.csv` |
| `09_forecast_statistics.py` | Friedman test, Holm-corrected Wilcoxon tests, paired bootstrap intervals | `forecast_*.csv`, `forecast_friedman.txt` |
| `10_trend_tests.py` | Mann–Kendall tests, Sen's slope, Benjamini–Hochberg and Hamed–Rao corrections | `trend_tests_stations.csv`, `trend_tests_basins.csv` |
| `11_hyperparameter_sensitivity.py` | Random-search tuning at a stratified subset of 154 stations | `hyperparameter_runs.csv` |
| `12_figures.py` | Figures 3–8 | `figures/` |
| `13_study_area_figures.py` | Figures 1–2: study area map and observation record of every station (`--download` fetches the map layers) | `figures/`, `observation_periods.csv` |

## Data

Large files are attached to the GitHub release `v1.0-r4` rather than stored in the repository:

| Release asset | Content |
|---|---|
| `wsc_snapshot_2024_part1…part4_*.parquet` | Snapshot of the Water Survey of Canada daily water level archive as downloaded in 2024 (all 3,645 stations, original fields), in four files; place all four in `data/raw/wsc_snapshot_2024/` |
| `processed_series_basin_XX*.zip` (15 files) | Screened daily series of the 3,564 analysed stations after gap filling, with flags for filled days and segment identifiers; one or more files per drainage basin. Unzip all into `work/series/` to skip step 01 |

Map layers for Fig. 1 are downloaded by `13_study_area_figures.py --download` into `work/gis/`: the major drainage areas of Canada (Atlas of Canada, Natural Resources Canada, Open Government Licence – Canada) and country, province and lake outlines from Natural Earth (public domain).

The raw data are published by the Water Survey of Canada (Environment and Climate Change Canada) under the Open Government Licence – Canada and are redistributed here unchanged with attribution. Current versions are available from the Water Survey of Canada (https://wateroffice.ec.gc.ca/).

## Reproducing the analysis

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the four wsc_snapshot_2024_part*.parquet files in data/raw/wsc_snapshot_2024/
cd src
python 01_prepare_series.py
python 02_gap_length_distribution.py
python 03_interpolation_experiment.py
python 04_interpolation_gap_length.py
python 05_interpolation_statistics.py
python 06_forecast_pipeline.py --workers 12                       # absolute-level formulation, persistence, ARIMA
WL_FORMULATION=relative WL_RESULTS=results_rel WL_PREDS=preds_rel \
  python 06_forecast_pipeline.py --models Persistence,RF,XGB,LGBM,LSTM --workers 12   # change-based formulation (main)
python 07_collect_results.py
python 10_trend_tests.py
python 09_forecast_statistics.py
python 08_data_flow_table.py
python 11_hyperparameter_sensitivity.py --workers 12
python 12_figures.py
python 13_study_area_figures.py --download
```

Intermediate files are written to `work/`, tables to `src/output/` and figures to `figures/`. All random seeds are fixed. The two forecasting runs together take about three hours on a 14-core laptop, and the hyperparameter analysis about one hour. On macOS, XGBoost and LightGBM need an OpenMP runtime (`libomp`).

## License

Code: MIT License (see `LICENSE`). Results tables: CC BY 4.0. Raw data: Open Government Licence – Canada.
