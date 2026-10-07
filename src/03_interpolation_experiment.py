"""
03 - Paired comparison of interpolation methods with artificial gaps.

Design
- Stations: stations without any missing day between their first and last observation and with at least MIN_OBS
  observations (178 stations), taken from the output of step 01.
- Methods (each uses only the series of the station itself; x = day index):
    Linear    : linear interpolation (numpy.interp)
    Quadratic : piecewise second-order interpolation (scipy interp1d, kind="quadratic")
    Spline    : cubic spline with not-a-knot end conditions (scipy CubicSpline)
- Gap ratios: 5, 10, 15 and 20% of the days.
- Masking schemes:
    random : single interior days drawn at random
    block  : blocks of consecutive days whose lengths are drawn from the real gap lengths (step 02)
- Ten repetitions per station, scheme and ratio. Within a repetition all methods fill the same mask (paired design).
- The first and the last day are never removed (no extrapolation). Seeds are deterministic.
Output: output/interpolation_runs.csv - one row per station, scheme, ratio, repetition and method
"""
import os
import zlib

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

import config as C

RATES = (0.05, 0.10, 0.15, 0.20)
REPS = 10
SCHEMES = ("random", "block")
METHODS = ("Linear", "Quadratic", "Spline")


def load_stations():
    flow = pd.read_csv(C.out("station_flow.csv"), dtype={"Station_id": str})
    ids = sorted(flow[flow.included & (flow.n_gaps == 0)].Station_id)
    stations = {}
    for sid in ids:
        d = pd.read_parquet(os.path.join(C.WORK, "series", f"{sid}.parquet"))
        assert d.value.notna().all() and not d.filled.any()
        stations[sid] = d.value.values.astype(float)
    return stations


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def random_mask(n, k, rng):
    idx = rng.choice(np.arange(1, n - 1), size=k, replace=False)
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def block_mask(n, k, rng, gap_lengths):
    """Masks k days in non-touching blocks whose lengths are drawn from the real gap lengths."""
    m = np.zeros(n, bool)
    usable = gap_lengths[gap_lengths <= max(1, k)]
    if len(usable) == 0:
        usable = np.array([1])
    total, tries = 0, 0
    while total < k and tries < 10000:
        tries += 1
        length = int(min(rng.choice(usable), k - total))
        s = int(rng.integers(1, n - 1 - length + 1)) if n - 1 - length > 1 else 1
        e = s + length
        # the block and one day on each side must be free (at least one observed day between blocks)
        if e > n - 1 or m[max(0, s - 1):min(n, e + 1)].any():
            continue
        m[s:e] = True
        total += length
    return m


def fill(x_obs, y_obs, x_q, method):
    if method == "Linear":
        return np.interp(x_q, x_obs, y_obs)
    if method == "Quadratic":
        return interp1d(x_obs, y_obs, kind="quadratic", assume_sorted=True)(x_q)
    if method == "Spline":
        return CubicSpline(x_obs, y_obs)(x_q)
    raise ValueError(method)


def metrics(obs, est):
    err = est - obs
    ss = float(np.sum((obs - obs.mean()) ** 2))
    return (float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))),
            float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan)


def main():
    stations = load_stations()
    print(f"stations: {len(stations)}")
    gap_lengths = pd.read_csv(C.out("gap_lengths.csv")).iloc[:, 0].values.astype(int)
    rows = []
    for sid, y in stations.items():
        n = len(y)
        x = np.arange(n)
        for scheme in SCHEMES:
            for rate in RATES:
                k = max(1, int(round(rate * n)))
                for rep in range(REPS):
                    rng = np.random.default_rng(seed_of(sid, scheme, rate, rep))
                    m = random_mask(n, k, rng) if scheme == "random" else block_mask(n, k, rng, gap_lengths)
                    for method in METHODS:
                        rmse, mae, r2 = metrics(y[m], fill(x[~m], y[~m], x[m], method))
                        rows.append((sid, sid[:2], n, scheme, int(rate * 100), rep, method, rmse, mae, r2, int(m.sum())))
    df = pd.DataFrame(rows, columns=["Station_id", "Basin", "n_days", "scheme", "rate_pct", "rep",
                                     "method", "RMSE", "MAE", "R2", "n_masked"])
    df.to_csv(C.out("interpolation_runs.csv"), index=False)
    print("rows:", len(df))
    print(df.groupby(["scheme", "rate_pct", "method"]).RMSE.agg(["mean", "median"]).round(4).unstack("method"))


if __name__ == "__main__":
    main()
