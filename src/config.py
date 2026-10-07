"""Shared paths and constants for all analysis scripts.

Paths can be overridden with environment variables:
  WL_RAW   raw archive snapshot (folder of parquet parts, a single parquet file or a pickle)
  WL_WORK  folder for intermediate files (processed series, model results, predictions)
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
OUTPUT = os.path.join(SRC, "output")


RAW = os.environ.get("WL_RAW", os.path.join(ROOT, "data", "raw", "wsc_snapshot_2024"))
WORK = os.environ.get("WL_WORK", os.path.join(ROOT, "work"))
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(OUTPUT, exist_ok=True)

MAX_FILL_GAP = 14     # gaps of up to this many days are filled by linear interpolation
MIN_OBS = 100         # minimum number of valid daily observations per station
SEED = 42

BASIN_NAMES = {"01": "Maritime Provinces", "02": "St. Lawrence", "03": "N. Quebec & Labrador",
               "04": "SW Hudson Bay", "05": "Nelson River", "06": "W & N Hudson Bay", "07": "Great Slave Lake",
               "08": "Pacific", "09": "Yukon River", "10": "Arctic", "11": "Mississippi River"}

MAIN_MODELS = ["Persistence", "ARIMA", "RF", "XGB", "LGBM", "LSTM"]


def out(name):
    """Path of a table in the output folder."""
    return os.path.join(OUTPUT, name)
