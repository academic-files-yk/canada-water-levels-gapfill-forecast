"""
10 - Mann-Kendall trend tests on annual mean water level with Benjamini-Hochberg correction.

Input : WORK/series/<station>.parquet - observed days only (interpolated days are excluded).
Accepted year: number of observed days >= 0.8 x the station's typical annual count (median over all years except
the first and the last), so that seasonal stations are judged against their own operating season.
Tested stations: >= 10 accepted years. Tests: Mann-Kendall with Sen's slope; Hamed-Rao variance correction as a
check for serial correlation. Multiple testing: Benjamini-Hochberg at q = 0.05 over all tested stations.
Output: output/trend_tests_stations.csv, output/trend_tests_basins.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import pymannkendall as mk
from statsmodels.stats.multitest import multipletests

import config as C

MIN_YEARS = 10
COVER = 0.8
Q = 0.05


def annual_means(ser):
    s = ser[(~ser.filled) & ser.value.notna()]
    if s.empty:
        return None, 0
    yr = s.date.dt.year
    cnt = yr.value_counts().sort_index()
    inner = cnt.iloc[1:-1] if len(cnt) > 2 else cnt
    typical = float(inner.median())
    ok_years = cnt[cnt >= COVER * typical].index
    am = s[yr.isin(ok_years)].groupby(yr[yr.isin(ok_years)]).value.mean()
    return am, typical


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(C.WORK, "series", "*.parquet"))):
        sid = os.path.basename(f)[:-8]
        am, typical = annual_means(pd.read_parquet(f, columns=["date", "value", "filled"]))
        base = dict(Station_id=sid, basin=sid[:2], typical_days=typical)
        if am is None or len(am) < MIN_YEARS:
            rows.append(dict(**base, n_years=0 if am is None else len(am), tested=False))
            continue
        x = am.values
        r = mk.original_test(x)
        try:
            p_hr = mk.hamed_rao_modification_test(x).p
        except Exception:
            p_hr = np.nan
        rows.append(dict(**base, n_years=len(x), first_year=int(am.index.min()), last_year=int(am.index.max()),
                         span_years=int(am.index.max() - am.index.min() + 1), tested=True,
                         tau=r.Tau, p=r.p, sen_slope=r.slope, trend=r.trend, p_hamed_rao=p_hr))
    d = pd.DataFrame(rows)
    t = d[d.tested].copy()
    t["p_bh"] = multipletests(t.p, alpha=Q, method="fdr_bh")[1]
    t["sig_raw"] = t.p < 0.05
    t["sig_bh"] = t.p_bh < Q
    hr_ok = t.p_hamed_rao.notna()
    t.loc[hr_ok, "p_hr_bh"] = multipletests(t.loc[hr_ok, "p_hamed_rao"], alpha=Q, method="fdr_bh")[1]
    t["sig_hr_bh"] = t.p_hr_bh < Q
    t["direction"] = np.where(t.tau > 0, "increasing", np.where(t.tau < 0, "decreasing", "none"))
    t["big_slope"] = t.sen_slope.abs() > 1.0
    d = d.merge(t[["Station_id", "p_bh", "sig_raw", "sig_bh", "p_hr_bh", "sig_hr_bh", "direction", "big_slope"]],
                on="Station_id", how="left")
    d.to_csv(C.out("trend_tests_stations.csv"), index=False)
    print(f"stations: {len(d)} | tested (>= {MIN_YEARS} accepted years): {len(t)}")
    print(f"significant p < 0.05: {t.sig_raw.sum()} -> Benjamini-Hochberg q < 0.05: {t.sig_bh.sum()} "
          f"| Hamed-Rao + BH: {int(t.sig_hr_bh.sum())}")
    print("direction (BH significant):", t[t.sig_bh].direction.value_counts().to_dict())
    print(f"|Sen slope| > 1 m/yr: {t.big_slope.sum()} stations")
    g = t.groupby("basin")
    s = pd.DataFrame({"basin_name": [C.BASIN_NAMES.get(b, b) for b in g.size().index], "n_tested": g.size(),
                      "median_tau": g.tau.median(), "median_sen_m_per_yr": g.sen_slope.median(),
                      "sig_raw": g.sig_raw.sum(), "sig_bh": g.sig_bh.sum(),
                      "sig_bh_inc": g.apply(lambda x: int((x.sig_bh & (x.tau > 0)).sum())),
                      "sig_bh_dec": g.apply(lambda x: int((x.sig_bh & (x.tau < 0)).sum())),
                      "sig_hr_bh": g.sig_hr_bh.sum()})
    s.to_csv(C.out("trend_tests_basins.csv"))
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
