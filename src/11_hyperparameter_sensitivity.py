"""
11 - Hyperparameter sensitivity of the machine-learning models.

Subset: up to PER_BASIN stations per basin, divided equally among the thirds of the basin's distribution of sample
numbers (stations with status "ok" in step 06), fixed seed.
For every station and model, with exactly the samples, split and scaling of step 06:
  default : configuration of the main analysis (change-based formulation)
  tuned   : random search with a fixed budget, selected on validation RMSE only (change-based formulation);
            for the LSTM the search includes the number of epochs with early stopping (patience 5)
  level   : default configuration with the absolute-level formulation
Output: WORK/hp/<station>.json, output/hyperparameter_runs.csv, output/hyperparameter_subset.csv
Usage : python 11_hyperparameter_sensitivity.py [--workers N] [--per-basin 14] [--trials 20] [--summarize-only]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import glob
import importlib.util
import json
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as C

_spec = importlib.util.spec_from_file_location("pipe", os.path.join(C.SRC, "06_forecast_pipeline.py"))
pipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipe)

HP = os.path.join(C.WORK, "hp")
MODELS = ("RF", "XGB", "LGBM", "LSTM")
SPACE = {
    "RF": {"n_estimators": [100, 300, 500], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5, 10],
           "max_features": [1.0, 0.5, "sqrt"]},
    "XGB": {"n_estimators": [100, 300, 1000], "max_depth": [3, 5, 7, 9], "learning_rate": [0.01, 0.05, 0.1, 0.3],
            "subsample": [0.7, 1.0], "colsample_bytree": [0.7, 1.0], "min_child_weight": [1, 5]},
    "LGBM": {"n_estimators": [100, 300, 1000], "num_leaves": [15, 31, 63], "max_depth": [-1, 5, 10],
             "learning_rate": [0.01, 0.05, 0.1, 0.3], "min_child_samples": [10, 20, 50], "subsample": [0.7, 1.0],
             "subsample_freq": [1]},
    "LSTM": {"units": [25, 50, 100], "epochs": [10, 30, 60], "learning_rate": [0.001, 0.005],
             "batch_size": [32, 128]},
}
LSTM_TRIALS = 8


def sample_configs(model, n, rng):
    space = SPACE[model]
    seen, out = set(), []
    for _ in range(n * 20):
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) == n:
            break
    return out


def lstm_early_stopping(Xtr, ytr, Xva, yva, Xte, params):
    """LSTM with early stopping on the validation loss (same architecture as step 06)."""
    import keras
    keras.utils.set_random_seed(pipe.SEED)
    m = keras.Sequential([keras.Input((pipe.K, 1)), keras.layers.LSTM(int(params["units"])), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"])), loss="mean_squared_error")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    h = m.fit(Xtr[..., None], ytr, epochs=int(params["epochs"]), batch_size=int(params["batch_size"]),
              validation_data=(Xva[..., None], yva), verbose=0, callbacks=cb)
    pte = m.predict(Xte[..., None], verbose=0, batch_size=1024).ravel()
    pva = m.predict(Xva[..., None], verbose=0, batch_size=1024).ravel()
    keras.backend.clear_session()
    return pte, pva, {"epochs_run": len(h.history["loss"])}


def run_station(sid, n_trials):
    ser = pd.read_parquet(os.path.join(pipe.SERIES, f"{sid}.parquet"))
    S = pipe.make_samples(ser)
    n = len(S["y"])
    sl_tr, sl_va, sl_te = pipe.split(n)
    yva, yte = S["y"][sl_va], S["y"][sl_te]
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    res = {"Persistence": {"default": pipe.metrics(yte, S["X"][sl_te][:, -1])}}
    prep = {f: pipe.prepare(S, sl_tr, f) for f in ("relative", "level")}

    def evaluate(model, params, formulation="relative", early_stopping=False):
        Xs, ys, to_level = prep[formulation]
        if early_stopping:
            pte, pva, extra = lstm_early_stopping(Xs[sl_tr], ys[sl_tr], Xs[sl_va], ys[sl_va], Xs[sl_te], params)
        else:
            pte, pva, extra = pipe.fit_predict_ml(model, Xs[sl_tr], ys[sl_tr], Xs[sl_va], ys[sl_va], Xs[sl_te], params)
        val = float(np.sqrt(np.mean((to_level(pva, sl_va) - yva) ** 2)))
        return val, pipe.metrics(yte, to_level(pte, sl_te)), extra

    for model in MODELS:
        r = {}
        v, m, _ = evaluate(model, None)
        r["default"] = {**m, "val_RMSE": v}
        v, m, _ = evaluate(model, None, formulation="level")
        r["level"] = {**m, "val_RMSE": v}
        best = (np.inf, None, None)
        for c in sample_configs(model, LSTM_TRIALS if model == "LSTM" else n_trials, rng):
            try:
                v, m, extra = evaluate(model, c, early_stopping=(model == "LSTM"))
            except Exception:
                continue
            if v < best[0]:
                best = (v, m, {**c, **extra})
        if best[1] is not None:
            r["tuned"] = {**best[1], "val_RMSE": best[0], "params": best[2]}
        res[model] = r
    return dict(Station_id=sid, basin=sid[:2], n_train=sl_tr.stop, n_test=n - sl_va.stop, results=res)


def _worker(args):
    sid, n_trials = args
    t0 = time.time()
    try:
        r = run_station(sid, n_trials)
        r["status"] = "ok"
    except Exception as e:
        r = dict(Station_id=sid, status="crash", error=repr(e)[:300], trace=traceback.format_exc()[-800:])
    r["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(HP, f"{sid}.json"), "w") as fh:
        json.dump(r, fh, default=str)
    return sid


def choose_subset(per_basin, seed=2026):
    st = pd.read_csv(C.out("forecast_status.csv"), dtype={"Station_id": str})
    st = st[st.status == "ok"].copy()
    st["basin"] = st.Station_id.str[:2]
    st["tercile"] = st.groupby("basin").n_samples.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    rng = np.random.default_rng(seed)
    pick = []
    for _, g in st.groupby("basin"):
        k = min(per_basin, len(g))
        per = [k // 3 + (1 if i < k % 3 else 0) for i in range(3)]
        for t, kk in zip(range(3), per):
            gg = g[g.tercile == t]
            pick += list(rng.choice(gg.Station_id.values, size=min(kk, len(gg)), replace=False))
    return sorted(pick)


def summarize():
    rows = []
    for f in glob.glob(os.path.join(HP, "*.json")):
        r = json.load(open(f))
        if r.get("status") != "ok":
            continue
        for model, cfgs in r["results"].items():
            for cfg, m in cfgs.items():
                rows.append(dict(Station_id=r["Station_id"], basin=r["basin"], model=model, config=cfg,
                                 RMSE=m["RMSE"], val_RMSE=m.get("val_RMSE"), params=json.dumps(m.get("params"))))
    d = pd.DataFrame(rows).sort_values(["Station_id", "model", "config"]).reset_index(drop=True)
    d.to_csv(C.out("hyperparameter_runs.csv"), index=False)
    w = d.pivot_table(index="Station_id", columns=["model", "config"], values="RMSE")
    pers = w[("Persistence", "default")]
    tab = [dict(model=m, config=c, median_RMSE=w[(m, c)].median(), median_skill=(1 - w[(m, c)] / pers).median())
           for m in MODELS for c in ("level", "default", "tuned") if (m, c) in w]
    print(f"stations: {w.shape[0]}\n", pd.DataFrame(tab).round(4).to_string(index=False))
    from scipy.stats import spearmanr
    for cfg in ("tuned", "level"):
        a = pd.DataFrame({m: w[(m, "default")] for m in MODELS}).rank(axis=1)
        b = pd.DataFrame({m: w[(m, cfg)] for m in MODELS if (m, cfg) in w}).rank(axis=1)
        rho = [spearmanr(a.loc[i], b.loc[i])[0] for i in a.index if b.loc[i].notna().all()]
        print(f"default vs {cfg}: median within-station rank correlation {np.nanmedian(rho):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-basin", type=int, default=14)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(HP, exist_ok=True)
    if not a.summarize_only:
        sids = choose_subset(a.per_basin)
        pd.Series(sids, name="Station_id").to_csv(C.out("hyperparameter_subset.csv"), index=False)
        todo = [s for s in sids if not os.path.exists(os.path.join(HP, f"{s}.json"))]
        print(f"subset {len(sids)} stations, to run {len(todo)}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, f in enumerate(as_completed([ex.submit(_worker, (s, a.trials)) for s in todo]), 1):
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"{i}/{len(todo)} elapsed {el / 60:.1f} min remaining ~{el / i * (len(todo) - i) / 60:.1f} min",
                          flush=True)
    summarize()


if __name__ == "__main__":
    main()
