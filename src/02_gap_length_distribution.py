"""
02 - Distribution of real gap lengths in the archive.

Input : WORK/series/<station>.parquet from step 01. A missing day is a filled day or an unfilled (NaN) day;
        a gap is a run of consecutive missing days between the first and the last observation.
Output: output/gap_lengths.csv - lengths (days) of all gaps of all stations. Used by step 03 to draw realistic
        block lengths.
"""
import glob
import os

import numpy as np
import pandas as pd

import config as C


def main():
    runs = []
    for f in sorted(glob.glob(os.path.join(C.WORK, "series", "*.parquet"))):
        d = pd.read_parquet(f, columns=["value", "filled"])
        miss = (d.filled.values | d.value.isna().values).astype(int)
        e = np.diff(np.r_[0, miss, 0])
        runs.extend(np.where(e == -1)[0] - np.where(e == 1)[0])
    r = pd.Series(runs, name="gap_length_days")
    r.to_csv(C.out("gap_lengths.csv"), index=False)
    print("gaps:", len(r), "| percentiles:", r.quantile([.25, .5, .75, .9, .95, .99]).to_dict(), "max", r.max())
    for c in (1, 2, 3, 7, 14, 30, 90, 180, 365):
        print(f"  <= {c:>3} days: {100 * (r <= c).mean():5.1f}% of gaps, "
              f"{100 * r[r <= c].sum() / r.sum():5.1f}% of missing days")


if __name__ == "__main__":
    main()
