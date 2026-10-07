"""
09 - Statistical comparison of the forecasting models.

Omnibus test: Friedman test with stations as blocks (repeated measures) over Persistence, ARIMA, RF, XGB, LGBM and
LSTM. Post hoc: pairwise Wilcoxon signed-rank tests with Holm correction. Effect sizes: median of the station-wise
RMSE differences with a 95% paired bootstrap confidence interval (B = 5000). Basin tables are descriptive only.
Output: output/forecast_friedman.txt, output/forecast_pairwise.csv, output/forecast_basins.csv,
        output/forecast_model_summary.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

import config as C

B = 5000
MODELS = C.MAIN_MODELS


def boot_median_ci(d, rng):
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return np.percentile(np.median(d[idx], axis=1), [2.5, 97.5])


def main():
    m = pd.read_csv(C.out("forecast_metrics_long.csv"), dtype={"Station_id": str})
    m = m[m.error.isna()]
    w = m.pivot(index="Station_id", columns="model", values="RMSE")
    full = w.dropna(subset=MODELS)
    n = len(full)
    chi, p = friedmanchisquare(*[full[c].values for c in MODELS])
    ranks = full[MODELS].rank(axis=1)
    lines = [f"Stations (all {len(MODELS)} models valid): {n}",
             f"Friedman chi2 = {chi:.1f}, df = {len(MODELS) - 1}, p = {p:.2e}, Kendall W = {chi / (n * (len(MODELS) - 1)):.3f}",
             "Mean rank: " + ", ".join(f"{k} {v:.2f}" for k, v in ranks.mean().sort_values().items())]
    rng = np.random.default_rng(2026)
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        d = (full[a] - full[b]).values
        lo, hi = boot_median_ci(d, rng)
        rows.append(dict(A=a, B=b, median_diff_A_minus_B=np.median(d), ci_low=lo, ci_high=hi,
                         A_better_frac=float((d < 0).mean()), p_wilcoxon=wilcoxon(full[a], full[b]).pvalue))
    pr = pd.DataFrame(rows)
    pr["p_holm"] = multipletests(pr.p_wilcoxon, method="holm")[1]
    pr.to_csv(C.out("forecast_pairwise.csv"), index=False)
    pers = full["Persistence"]
    summ = pd.DataFrame({"median_RMSE": full[MODELS].median(), "mean_RMSE": full[MODELS].mean(),
                         "median_skill_vs_persistence": (1 - full[MODELS].div(pers, axis=0)).median(),
                         "frac_beats_persistence": full[MODELS].lt(pers, axis=0).mean(),
                         "n_best": full[MODELS].idxmin(axis=1).value_counts().reindex(MODELS).fillna(0).astype(int)})
    for extra, label in [("ARIMA_MS", "ARIMA multi-step")] + \
                        [(f"{mm}_level", f"{mm} absolute level") for mm in ("RF", "XGB", "LGBM", "LSTM")]:
        if extra in w:
            x = w.loc[full.index, extra].dropna()
            summ.loc[label, ["median_RMSE", "mean_RMSE", "median_skill_vs_persistence", "frac_beats_persistence"]] = [
                x.median(), x.mean(), (1 - x / pers.loc[x.index]).median(), float((x < pers.loc[x.index]).mean())]
    summ.to_csv(C.out("forecast_model_summary.csv"))
    full = full.assign(basin=full.index.str[:2])
    hb = full.groupby("basin")[MODELS].median()
    hb.insert(0, "n", full.groupby("basin").size())
    hb.insert(0, "basin_name", [C.BASIN_NAMES.get(b, b) for b in hb.index])
    hb["best_median"] = full.groupby("basin")[MODELS].median().idxmin(axis=1)
    hb.to_csv(C.out("forecast_basins.csv"))
    open(C.out("forecast_friedman.txt"), "w").write("\n".join(lines) + "\n")
    pd.set_option("display.width", 220)
    print("\n".join(lines), "\n\n", summ.round(4).to_string(), "\n\n", pr.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
