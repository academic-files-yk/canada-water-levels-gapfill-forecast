"""
08 - Number of stations retained at each analytical stage (Table 1).

Output: output/data_flow_table.csv (stage, rule, stations, days or samples), output/data_flow_basins.csv
"""
import pandas as pd

import config as C


def main():
    flow = pd.read_csv(C.out("station_flow.csv"), dtype={"Station_id": str, "basin": str})
    status = pd.read_csv(C.out("forecast_status.csv"), dtype={"Station_id": str})
    m = pd.read_csv(C.out("forecast_metrics_long.csv"), dtype={"Station_id": str})
    trend = pd.read_csv(C.out("trend_tests_stations.csv"), dtype={"Station_id": str})
    ok_models = m[m.error.isna() & m.model.isin(C.MAIN_MODELS)].groupby("Station_id").model.nunique()
    all_models = set(ok_models[ok_models == len(C.MAIN_MODELS)].index)
    inc = flow[flow.included]
    ok = status[status.status == "ok"]
    rows = [
        ("Archive", "all stations", len(flow), None),
        ("Screening", f">= {C.MIN_OBS} valid daily observations", len(inc), int(inc.n_obs.sum())),
        ("Gap filling", f"gaps <= {C.MAX_FILL_GAP} days filled linearly; longer gaps split segments", len(inc),
         int(inc.n_days_filled.sum())),
        ("Forecast samples", "window within one segment, observed target; >=100 train, >=10 val, >=30 test samples",
         len(ok), int(ok.n_samples.sum())),
        ("Model comparison", f"valid results for all {len(C.MAIN_MODELS)} models", len(all_models), None),
        ("Trend tests", ">= 10 accepted years", int(trend.tested.sum()), None),
        ("Significant trends", "Benjamini-Hochberg q < 0.05", int(trend.sig_bh.fillna(False).astype(bool).sum()), None),
    ]
    t = pd.DataFrame(rows, columns=["stage", "rule", "stations", "days_or_samples"])
    t.to_csv(C.out("data_flow_table.csv"), index=False)
    print(t.to_string(index=False))
    basins = pd.DataFrame({
        "archive": flow.groupby("basin").size(),
        "screened": inc.groupby("basin").size(),
        "forecast": pd.Series([s[:2] for s in all_models]).value_counts(),
        "trend_tests": trend[trend.tested].basin.astype(str).str.zfill(2).value_counts(),
    }).fillna(0).astype(int)
    basins.loc["Total"] = basins.sum()
    basins.to_csv(C.out("data_flow_basins.csv"))
    print("\n", basins.to_string())


if __name__ == "__main__":
    main()
