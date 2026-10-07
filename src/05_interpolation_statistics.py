"""
05 - Paired statistics of the interpolation experiments.

The station is the unit of analysis: the RMSE of each station is first averaged over the ten repetitions.
For every scheme and gap ratio: median RMSE of each method; for every pair of methods the median of the
station-wise differences with a 95% paired bootstrap confidence interval (B = 5000), the share of stations where
the first method is better, and the Wilcoxon signed-rank test with Holm correction (within each scheme).
The same paired summary is computed for the gap-length experiment (Spline - Linear, Quadratic - Linear).
Output: output/interpolation_stats.csv, output/gap_length_stats.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import config as C

B = 5000
METHODS = ["Linear", "Quadratic", "Spline"]


def ci(d, rng):
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])


def main():
    rng = np.random.default_rng(2026)
    d = pd.read_csv(C.out("interpolation_runs.csv"), dtype={"Station_id": str})
    st = d.groupby(["scheme", "rate_pct", "Station_id", "method"]).RMSE.mean().unstack("method")
    rows = []
    for (scheme, rate), g in st.groupby(level=["scheme", "rate_pct"]):
        med = g.median()
        for a, b in itertools.combinations(METHODS, 2):
            diff = (g[a] - g[b]).values
            lo, hi = ci(diff, rng)
            rows.append(dict(scheme=scheme, rate_pct=rate, A=a, B=b, n=len(diff), median_RMSE_A=med[a],
                             median_RMSE_B=med[b], median_diff=np.median(diff), ci_low=lo, ci_high=hi,
                             A_better_frac=float((diff < 0).mean()), p_wilcoxon=wilcoxon(g[a], g[b]).pvalue))
    r = pd.DataFrame(rows)
    r["p_holm"] = r.groupby("scheme").p_wilcoxon.transform(lambda p: multipletests(p, method="holm")[1])
    r.to_csv(C.out("interpolation_stats.csv"), index=False)
    pd.set_option("display.width", 220)
    print(r.round(5).to_string(index=False))

    gl = pd.read_csv(C.out("gap_length_runs.csv"), dtype={"Station_id": str})
    w = gl.groupby(["gap_len", "Station_id", "method"]).RMSE.mean().unstack("method")
    rows = []
    for length, g in w.groupby(level="gap_len"):
        for a, b in (("Spline", "Linear"), ("Quadratic", "Linear")):
            diff = (g[a] - g[b]).values
            lo, hi = ci(diff, rng)
            rows.append(dict(gap_len=length, A=a, B=b, n=len(diff), median_RMSE_A=g[a].median(),
                             median_RMSE_B=g[b].median(), median_diff=np.median(diff), ci_low=lo, ci_high=hi,
                             A_better_frac=float((diff < 0).mean())))
    q = pd.DataFrame(rows)
    q.to_csv(C.out("gap_length_stats.csv"), index=False)
    print("\n", q.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
