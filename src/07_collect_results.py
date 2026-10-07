"""
07 - Collects the station results of step 06 into one table.

Two runs are combined:
  WORK/results     : absolute-level formulation - Persistence, ARIMA, ARIMA_MS and the machine-learning models
  WORK/results_rel : change-based (relative) formulation - machine-learning models (main analysis)
In the combined table the machine-learning models come from the relative run (RF, XGB, LGBM, LSTM); the
absolute-level versions are kept as RF_level, XGB_level, LGBM_level and LSTM_level for comparison.
Output: output/forecast_metrics_long.csv (station x model), output/forecast_status.csv
"""
import glob
import json
import os

import pandas as pd

import config as C

ML = ("RF", "XGB", "LGBM", "LSTM")
EXTRA = ["ARIMA_MS"] + [f"{m}_level" for m in ML]


def load(folder):
    status, rows = [], []
    for f in sorted(glob.glob(os.path.join(C.WORK, folder, "*.json"))):
        r = json.load(open(f))
        status.append({k: r.get(k) for k in ("Station_id", "basin", "status", "n_days", "n_measured", "n_samples",
                                             "n_train", "n_val", "n_test", "test_start", "test_end",
                                             "test_input_fill_frac", "sec")})
        for m, v in (r.get("models") or {}).items():
            rows.append(dict(Station_id=r["Station_id"], basin=r.get("basin"), model=m,
                             RMSE=v.get("RMSE"), MAE=v.get("MAE"), R2=v.get("R2"), n=v.get("n"),
                             error=v.get("error"), order=str(v.get("order")) if v.get("order") else None,
                             best_iteration=v.get("best_iteration")))
    return pd.DataFrame(status), pd.DataFrame(rows)


def main():
    status, lv = load("results")
    if os.path.isdir(os.path.join(C.WORK, "results_rel")):
        _, rl = load("results_rel")
        lv.loc[lv.model.isin(ML), "model"] = lv.loc[lv.model.isin(ML), "model"] + "_level"
        m = pd.concat([lv, rl[rl.model.isin(ML)]], ignore_index=True)
    else:
        m = lv
    status.to_csv(C.out("forecast_status.csv"), index=False)
    m.to_csv(C.out("forecast_metrics_long.csv"), index=False)
    print("status:", status.status.value_counts().to_dict())
    err = m[m.error.notna()]
    if len(err):
        print("model errors:", err.groupby("model").size().to_dict())
    w = m[m.error.isna()].pivot(index="Station_id", columns="model", values="RMSE")
    core = [c for c in C.MAIN_MODELS if c in w]
    full = w.dropna(subset=core)
    cols = [c for c in core + EXTRA if c in full]
    s = pd.DataFrame({"median": full[cols].median(), "mean": full[cols].mean(),
                      "median_skill": (1 - full[cols].div(full["Persistence"], axis=0)).median()})
    print(f"{len(full)} stations with valid results for all main models\n", s.round(4).to_string())


if __name__ == "__main__":
    main()
