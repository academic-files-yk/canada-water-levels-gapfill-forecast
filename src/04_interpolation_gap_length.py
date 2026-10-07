"""
04 - Interpolation error as a function of gap length.

For every reference station (step 03) and gap length L, a single block of L days is removed at 20 random interior
positions and filled with each method (paired: all methods fill the same block).
Output: output/gap_length_runs.csv (station, L, placement, method, RMSE, MAE, bias)
"""
import importlib.util
import os
import zlib

import numpy as np
import pandas as pd

import config as C

_spec = importlib.util.spec_from_file_location("exp", os.path.join(C.SRC, "03_interpolation_experiment.py"))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

LENGTHS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90)
N_PLACE = 20


def main():
    rows = []
    for sid, y in exp.load_stations().items():
        n = len(y)
        x = np.arange(n)
        for length in LENGTHS:
            if n < 3 * length + 2:
                continue
            rng = np.random.default_rng(zlib.crc32(f"{sid}|{length}".encode()))
            for k in range(N_PLACE):
                s = int(rng.integers(1, n - 1 - length))
                m = np.zeros(n, bool)
                m[s:s + length] = True
                for method in exp.METHODS:
                    err = exp.fill(x[~m], y[~m], x[m], method) - y[m]
                    rows.append((sid, sid[:2], length, k, method, float(np.sqrt(np.mean(err ** 2))),
                                 float(np.mean(np.abs(err))), float(np.mean(err))))
    df = pd.DataFrame(rows, columns=["Station_id", "Basin", "gap_len", "placement", "method", "RMSE", "MAE", "bias"])
    df.to_csv(C.out("gap_length_runs.csv"), index=False)
    t = df.groupby(["gap_len", "method"]).RMSE.median().unstack("method").round(4)
    print("median RMSE by gap length\n", t.to_string())


if __name__ == "__main__":
    main()
