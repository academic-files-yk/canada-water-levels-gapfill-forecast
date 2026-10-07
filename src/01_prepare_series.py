"""
01 - Screening, basin assignment, gap filling and segmentation of the raw daily records.

Input : raw Water Survey of Canada snapshot (config.RAW), 21,839,295 rows, 3,645 stations.
Rules :
  1. The basin is taken from the first two digits of the station identifier.
  2. Non-numeric values, unreadable dates and duplicated station-dates are removed (first record kept).
  3. Stations with fewer than MIN_OBS valid observations are excluded.
  4. Each series is placed on a continuous daily axis between its first and last observation. Gaps of up to
     MAX_FILL_GAP days are filled by linear interpolation (filled = True). Longer gaps are left as NaN and split
     the series into independent segments.
Output: WORK/series/<station>.parquet (date, value, filled, segment, symbol)
        output/station_flow.csv (per-station counts), output/station_metadata.csv
"""
import os

import numpy as np
import pandas as pd

import config as C


def build_series(g):
    """g: cleaned (date, value, symbol) records of one station, sorted by date."""
    idx = pd.date_range(g["date"].iloc[0], g["date"].iloc[-1], freq="D")
    s = g.set_index("date").reindex(idx)
    obs = s["value"].notna().values
    miss = (~obs).astype(int)
    e = np.diff(np.r_[0, miss, 0])
    starts, ends = np.where(e == 1)[0], np.where(e == -1)[0]
    lengths = ends - starts
    value = s["value"].values.astype(float).copy()
    filled = np.zeros(len(s), bool)
    x = np.arange(len(s))
    for a, b, length in zip(starts, ends, lengths):
        if length <= C.MAX_FILL_GAP:
            value[a:b] = np.interp(x[a:b], x[obs], value[obs])
            filled[a:b] = True
    # unfilled gaps split the series into segments numbered 0..k-1 (-1 = missing)
    unfilled = np.isnan(value)
    seg = np.cumsum(np.r_[0, (unfilled[1:] & ~unfilled[:-1]) | (~unfilled[1:] & unfilled[:-1])])
    seg = np.where(unfilled, -1, seg)
    valid = seg >= 0
    if valid.any():
        _, seg[valid] = np.unique(seg[valid], return_inverse=True)
    series = pd.DataFrame({"date": idx, "value": value, "filled": filled, "segment": seg.astype(int),
                           "symbol": s["symbol"].fillna("").values})
    info = dict(n_days_span=len(s), n_obs=int(obs.sum()), n_gaps=len(lengths),
                n_gaps_filled=int((lengths <= C.MAX_FILL_GAP).sum()), n_days_filled=int(filled.sum()),
                n_days_unfilled=int(unfilled.sum()), n_segments=int(seg.max() + 1) if valid.any() else 0,
                max_gap=int(lengths.max()) if len(lengths) else 0)
    return series, info


def read_raw():
    if os.path.isdir(C.RAW) or C.RAW.endswith(".parquet"):
        return pd.read_parquet(C.RAW)
    return pd.read_pickle(C.RAW)


def main():
    os.makedirs(os.path.join(C.WORK, "series"), exist_ok=True)
    d = read_raw()
    n_raw_rows, n_raw_st = len(d), d["Station ID"].nunique()
    # In stations whose name contains a comma, the name was split over two fields when the CSV files were read and
    # all following metadata fields moved one column to the right. Metadata are read one field further for these
    # rows; values and dates are not affected.
    name = d["Station Name"].astype(str)
    shifted = name.str.startswith('"') & ~name.str.endswith('"')
    n_shifted_st = d.loc[shifted, "Station ID"].nunique()
    lat = pd.to_numeric(d["Latitude"].where(~shifted, d["Longitude"]), errors="coerce")
    lon = pd.to_numeric(d["Longitude"].where(~shifted, d["Operating Agency"]), errors="coerce")
    sched = d["Operation Schedule"].where(~shifted, d["Start Month"]).astype(str)
    full_name = name.where(~shifted, name + "," + d["Province"].astype(str))
    d = pd.DataFrame({"sid": d["Station ID"].astype(str).str.strip(),
                      "date": pd.to_datetime(d["Date"], errors="coerce"),
                      "value": pd.to_numeric(d["Value"], errors="coerce"),
                      "symbol": d["Symbol"].astype(str).str.strip(),
                      "name": full_name.str.strip().str.strip('"'),
                      "lat": lat, "lon": lon, "schedule": sched})
    print(f"rows with shifted metadata: {int(shifted.sum()):,} ({n_shifted_st} stations), corrected")
    n_bad_value, n_bad_date = int(d.value.isna().sum()), int(d.date.isna().sum())
    d = d.dropna(subset=["value", "date"])
    n_dup = int(d.duplicated(["sid", "date"]).sum())
    d = d.drop_duplicates(["sid", "date"], keep="first").sort_values(["sid", "date"])
    d["basin"] = d["sid"].str[:2]
    meta = d.groupby("sid").agg(basin=("basin", "first"), name=("name", "first"), lat=("lat", "first"),
                                lon=("lon", "first"), schedule=("schedule", "first"),
                                first=("date", "min"), last=("date", "max"), n_obs=("value", "size"))
    meta["basin_name"] = meta.basin.map(C.BASIN_NAMES)
    rows = []
    for sid, g in d.groupby("sid", sort=True):
        if len(g) < C.MIN_OBS:
            rows.append(dict(Station_id=sid, included=False, reason=f"n_obs<{C.MIN_OBS}", n_obs=len(g)))
            continue
        series, info = build_series(g[["date", "value", "symbol"]].reset_index(drop=True))
        series.to_parquet(os.path.join(C.WORK, "series", f"{sid}.parquet"), index=False)
        rows.append(dict(Station_id=sid, included=True, reason="", **info))
    flow = pd.DataFrame(rows).merge(meta[["basin", "basin_name"]], left_on="Station_id", right_index=True)
    flow.to_csv(C.out("station_flow.csv"), index=False)
    meta.to_csv(C.out("station_metadata.csv"))
    inc = flow[flow.included]
    print(f"raw: {n_raw_rows:,} rows, {n_raw_st} stations | non-numeric {n_bad_value:,}, "
          f"bad dates {n_bad_date}, duplicates {n_dup}")
    print(f"stations: {len(flow)} -> included {len(inc)} (excluded, n_obs<{C.MIN_OBS}: {(~flow.included).sum()})")
    print(f"days {inc.n_days_span.sum():,}; observed {inc.n_obs.sum():,}; filled {inc.n_days_filled.sum():,} "
          f"({100 * inc.n_days_filled.sum() / inc.n_days_span.sum():.2f}%); unfilled {inc.n_days_unfilled.sum():,}; "
          f"median segments per station {inc.n_segments.median():.0f}")


if __name__ == "__main__":
    main()
