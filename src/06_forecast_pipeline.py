"""
06 - One-day-ahead forecasting experiment.

Input : WORK/series/<station>.parquet (step 01: gaps <= 14 days filled linearly, longer gaps split segments)
Samples:
  target y(t) : a day with an observed (not interpolated) value
  inputs      : y(t-10) ... y(t-1); the whole window lies in the same segment as the target (no unfilled gap is
                crossed); the window may contain interpolated values from short gaps (their share is recorded)
  split       : samples in time order; first 80% training, next 10% validation, last 10% test
Models (all one day ahead, same forecast origin and information):
  Persistence  y_hat(t) = y(t-1)
  RF           RandomForestRegressor(n_estimators=100, random_state=42)
  XGB          XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, early stopping 10 rounds on validation)
  LGBM         LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, early stopping 10 rounds on validation)
  LSTM         LSTM(50) -> Dense(1), Adam, MSE, 10 epochs, batch size 32
  ARIMA        order in p{0,1,2} x d{0,1} x q{0,1,2}, estimated on the training period, order selected by one-day-ahead
               RMSE on the validation period, re-estimated on training + validation; in the test period the parameters
               are fixed and the state is updated with every new observation (one-day-ahead). ARIMA_MS: the same model
               forecasting the whole test period from the end of the validation period (multi-step; for comparison).
Formulation of the machine-learning models (environment variable WL_FORMULATION):
  relative (main) : inputs relative to the last value of the window, target = y(t) - y(t-1)
  level           : absolute levels scaled to [0, 1] on the training period
Output: WORK/<results>/<station>.json (metrics) and WORK/<preds>/<station>.parquet (test predictions)
Usage : python 06_forecast_pipeline.py [--workers N] [--limit K] [--stations A,B,...] [--models RF,XGB,...]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import glob
import json
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as C

SERIES = os.path.join(C.WORK, "series")
RES = os.path.join(C.WORK, os.environ.get("WL_RESULTS", "results"))
PRED = os.path.join(C.WORK, os.environ.get("WL_PREDS", "preds"))
FORMULATION = os.environ.get("WL_FORMULATION", "level")
K = 10
SEED = C.SEED
MIN_TRAIN, MIN_VAL, MIN_TEST = 100, 10, 30
ALL_MODELS = ("Persistence", "RF", "XGB", "LGBM", "LSTM", "ARIMA")
ARIMA_GRID = [(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1, 2)]


# ---------------------------------------------------------------- samples
def make_samples(ser):
    v = ser["value"].values.astype(float)
    seg = ser["segment"].values
    filled = ser["filled"].values
    n = len(v)
    if n <= K:
        return None
    t = np.arange(K, n)
    same = np.ones(len(t), bool)
    for lag in range(1, K + 1):
        same &= seg[t - lag] == seg[t]
    ok = same & (seg[t] >= 0) & ~filled[t]
    t = t[ok]
    if len(t) == 0:
        return None
    X = np.stack([v[t - K + j] for j in range(K)], axis=1)
    Xf = np.stack([filled[t - K + j] for j in range(K)], axis=1)
    return dict(t=t, X=X, y=v[t], fill_frac=Xf.mean(axis=1), dates=ser["date"].values[t])


def split(n):
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def metrics(y, p):
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                R2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"), n=int(len(y)))


# ---------------------------------------------------------------- models
class Scaler:
    def __init__(self, a):
        self.lo, self.hi = float(np.min(a)), float(np.max(a))
        self.rng = self.hi - self.lo if self.hi > self.lo else 1.0

    def f(self, a):
        return (a - self.lo) / self.rng

    def inv(self, a):
        return a * self.rng + self.lo


def fit_predict_ml(model, Xtr, ytr, Xva, yva, Xte, params=None):
    params = dict(params or {})
    if model == "RF":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(**{"n_estimators": 100, "random_state": SEED, "n_jobs": 1, **params})
        m.fit(Xtr, ytr)
        return m.predict(Xte), m.predict(Xva), {}
    if model == "XGB":
        import xgboost as xgb
        kw = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": SEED, "n_jobs": 1,
              "early_stopping_rounds": 10, **params}
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict(Xte), m.predict(Xva), {"best_iteration": int(m.best_iteration)}
    if model == "LGBM":
        import lightgbm as lgb
        kw = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": SEED, "n_jobs": 1,
              "verbose": -1, **params}
        m = lgb.LGBMRegressor(**kw)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(10, verbose=False)])
        return m.predict(Xte), m.predict(Xva), {"best_iteration": int(m.best_iteration_ or kw["n_estimators"])}
    if model == "LSTM":
        import keras
        keras.utils.set_random_seed(SEED)
        units = int(params.get("units", 50))
        epochs = int(params.get("epochs", 10))
        batch = int(params.get("batch_size", 32))
        lr = float(params.get("learning_rate", 0.001))
        m = keras.Sequential([keras.Input((K, 1)), keras.layers.LSTM(units), keras.layers.Dense(1)])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error")
        h = m.fit(Xtr[..., None], ytr, epochs=epochs, batch_size=batch, validation_data=(Xva[..., None], yva),
                  verbose=0, shuffle=True)
        pte = m.predict(Xte[..., None], verbose=0, batch_size=1024).ravel()
        pva = m.predict(Xva[..., None], verbose=0, batch_size=1024).ravel()
        keras.backend.clear_session()
        return pte, pva, {"val_loss_last": float(h.history["val_loss"][-1])}
    raise ValueError(model)


def arima_block(ser, S, sl_tr, sl_va, sl_te):
    """ARIMA with the same information as the other models: the forecast for day t uses observations up to t-1."""
    from statsmodels.tsa.arima.model import ARIMA
    v = ser["value"].values.astype(float)
    t = S["t"]
    t_tr_end, t_va_end, t_te_end = t[sl_tr][-1], t[sl_va][-1], t[sl_te][-1]
    y_tr = v[: t_tr_end + 1]
    best = (np.inf, None, None)
    tried = []
    for order in ARIMA_GRID:
        trend = "c" if order[1] == 0 else "t"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = ARIMA(y_tr, order=order, trend=trend).fit()
                f = ARIMA(v[: t_va_end + 1], order=order, trend=trend).filter(r.params)
                pv = f.predict()[t[sl_va]]
            if not np.all(np.isfinite(pv)):
                continue
            rm = float(np.sqrt(np.mean((pv - S["y"][sl_va]) ** 2)))
            tried.append((order, rm))
            if rm < best[0]:
                best = (rm, order, trend)
        except Exception:
            continue
    if best[1] is None:
        return None
    _, order, trend = best
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ARIMA(v[: t_va_end + 1], order=order, trend=trend).fit()
        f = ARIMA(v[: t_te_end + 1], order=order, trend=trend).filter(r.params)
        p1 = f.predict()[t[sl_te]]
        fc = r.forecast(steps=int(t_te_end - t_va_end))          # multi-step protocol, for comparison
        pms = np.asarray(fc)[t[sl_te] - t_va_end - 1]
    return p1, pms, {"order": list(order), "trend": trend, "n_orders_ok": len(tried)}


def prepare(S, sl_tr, formulation=None):
    """Input/target transformation of the machine-learning models.
    Returns Xs, ys and to_level(prediction, slice) -> water level (m)."""
    formulation = formulation or FORMULATION
    last = S["X"][:, -1]
    if formulation == "relative":
        Xr, yr = S["X"] - last[:, None], S["y"] - last
        s = float(np.std(yr[sl_tr])) or 1.0
        return Xr / s, yr / s, (lambda p, sl: np.asarray(p, float) * s + last[sl])
    sc = Scaler(np.r_[S["X"][sl_tr].ravel(), S["y"][sl_tr]])
    return sc.f(S["X"]), sc.f(S["y"]), (lambda p, sl: sc.inv(np.asarray(p, float)))


# ---------------------------------------------------------------- station
def run_station(sid, models=ALL_MODELS, params_by_model=None, tag=""):
    t0 = time.time()
    params_by_model = params_by_model or {}
    ser = pd.read_parquet(os.path.join(SERIES, f"{sid}.parquet"))
    S = make_samples(ser)
    info = dict(Station_id=sid, basin=sid[:2], tag=tag, n_days=len(ser),
                n_measured=int((~ser.filled & ser.value.notna()).sum()))
    if S is None:
        return dict(**info, status="no_samples")
    n = len(S["y"])
    sl_tr, sl_va, sl_te = split(n)
    n_tr, n_va, n_te = sl_tr.stop, sl_va.stop - sl_va.start, n - sl_va.stop
    info.update(n_samples=n, n_train=n_tr, n_val=n_va, n_test=n_te,
                test_start=str(S["dates"][sl_te][0])[:10], test_end=str(S["dates"][-1])[:10],
                test_input_fill_frac=float(S["fill_frac"][sl_te].mean()))
    if n_tr < MIN_TRAIN or n_va < MIN_VAL or n_te < MIN_TEST:
        return dict(**info, status="too_few_samples")
    Xs, ys, to_level = prepare(S, sl_tr)
    info["formulation"] = FORMULATION
    yte = S["y"][sl_te]
    preds = {"date": S["dates"][sl_te], "y": yte, "input_fill_frac": S["fill_frac"][sl_te]}
    out = {}
    for model in models:
        tm = time.time()
        try:
            if model == "Persistence":
                p, extra = S["X"][sl_te][:, -1], {}
            elif model == "ARIMA":
                r = arima_block(ser, S, sl_tr, sl_va, sl_te)
                if r is None:
                    out[model] = {"error": "no_admissible_order"}
                    continue
                p, pms, extra = r
                preds["ARIMA_MS"] = pms
                out["ARIMA_MS"] = {**metrics(yte, pms)}
            else:
                pte, pva, extra = fit_predict_ml(model, Xs[sl_tr], ys[sl_tr], Xs[sl_va], ys[sl_va], Xs[sl_te],
                                                 params_by_model.get(model))
                p = to_level(pte, sl_te)
                extra["val_RMSE"] = float(np.sqrt(np.mean((to_level(pva, sl_va) - S["y"][sl_va]) ** 2)))
            preds[model] = p
            out[model] = {**metrics(yte, p), **extra, "sec": round(time.time() - tm, 2)}
        except Exception as e:
            out[model] = {"error": repr(e)[:300], "trace": traceback.format_exc()[-800:]}
    os.makedirs(PRED, exist_ok=True)
    pd.DataFrame(preds).to_parquet(os.path.join(PRED, f"{sid}{tag}.parquet"), index=False)
    return dict(**info, status="ok", models=out, sec=round(time.time() - t0, 1))


def _worker(args):
    sid, models, params, tag = args
    path = os.path.join(RES, f"{sid}{tag}.json")
    try:
        r = run_station(sid, models, params, tag)
    except Exception as e:
        r = dict(Station_id=sid, tag=tag, status="crash", error=repr(e)[:300], trace=traceback.format_exc()[-800:])
    with open(path, "w") as fh:
        json.dump(r, fh, default=float)
    return sid, r.get("status"), r.get("sec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stations", default="")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--seed-sample", type=int, default=0, help="random sample of stations (pilot run)")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    sids = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(SERIES, "*.parquet")))
    if a.stations:
        sids = a.stations.split(",")
    if a.seed_sample:
        sids = sorted(np.random.default_rng(a.seed_sample).choice(sids, size=a.limit, replace=False))
    elif a.limit:
        sids = sids[: a.limit]
    models = tuple(a.models.split(","))
    todo = [s for s in sids if a.redo or not os.path.exists(os.path.join(RES, f"{s}.json"))]
    print(f"stations {len(sids)}, to run {len(todo)}, workers {a.workers}, models {models}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(_worker, (s, models, None, "")) for s in todo]):
            f.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)}  elapsed {el / 60:.1f} min  remaining ~{el / done * (len(todo) - done) / 60:.1f} min",
                      flush=True)


if __name__ == "__main__":
    main()
