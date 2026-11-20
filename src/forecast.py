"""
Forecasting qubit stability with a time-aware (rolling-origin) evaluation.

Target 1: T1 of every qubit at the next calibration (h = 1 step = 3 h)
          and one day ahead (h = 8 steps = 24 h).
Target 2: readout fidelity one hour ahead.

Split: the first 7 days are the initial training window. Every later
calibration time is a forecast origin: models see only data up to the
origin and forecast origin + h. No shuffling (Hyndman & Athanasopoulos,
FPP3, sec. 5.10 "time series cross-validation").

Metrics: MAE and RMSE on the original scale, and skill against the
persistence forecast (1 - error_model / error_persistence).

Decision: before each 3-hour window, a qubit is excluded from circuit
layout when its forecast says T1 may fall below 15 us.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from pathlib import Path

from simulate import simulate, pilot_check

ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore")
SEED = 2026
TRAIN_DAYS = 7
T1_THRESHOLD_US = 15.0
Q_LOW = 0.10                 # lower quantile used by the cautious policy
REFIT_EVERY = 8              # global model is refit once a day (8 x 3 h)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


# --------------------------------------------------------------------------- #
# per-qubit models (work on log T1 or on fidelity)
# --------------------------------------------------------------------------- #
def f_persistence(hist, h):
    return hist[-1]


def f_moving_avg(hist, h, w=8):
    return hist[-w:].mean()


def f_mean(hist, h):
    return hist.mean()


def f_ses(hist, h):
    fit = SimpleExpSmoothing(hist, initialization_method="estimated").fit()
    return float(fit.forecast(1)[0])


def f_ar1(hist, h):
    """AR(1) with intercept, fitted by least squares; h-step recursion."""
    x, y = hist[:-1], hist[1:]
    xm, ym = x.mean(), y.mean()
    phi = np.clip(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum(), 0.0, 0.99)
    mu = ym - phi * xm
    level = mu / (1 - phi)
    return level + phi ** h * (hist[-1] - level)


# --------------------------------------------------------------------------- #
# global gradient-boosting model with lag features (pooled over qubits)
# --------------------------------------------------------------------------- #
N_LAGS = 6


def features(logv: np.ndarray, t: int) -> np.ndarray:
    """Features available at time t for every qubit (rows = qubits)."""
    lags = [logv[t - k] for k in range(N_LAGS)]
    win = logv[t - 7:t + 1]
    hist = logv[:t + 1]
    exp_mean = hist.mean(axis=0)
    return np.column_stack(lags + [win.mean(0), win.std(0), exp_mean, logv[t] - exp_mean])


def training_rows(logv, origin, h):
    X, y = [], []
    for t in range(N_LAGS + 7, origin - h + 1):
        X.append(features(logv, t))
        y.append(logv[t + h] - logv[t])          # target: change in log T1
    return np.vstack(X), np.concatenate(y)


def gbm(loss="squared_error", quantile=None):
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
              min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    if loss == "quantile":
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(**kw)


# --------------------------------------------------------------------------- #
# rolling-origin evaluation
# --------------------------------------------------------------------------- #
def evaluate_t1(t1_df: pd.DataFrame, h: int):
    logv = np.log(t1_df.to_numpy())
    n_t, n_q = logv.shape
    start = TRAIN_DAYS * 24 // 3                 # first origin = end of day 7
    per_qubit = {"Persistence (last calibration)": f_persistence,
                 "Moving average, 24 h": f_moving_avg,
                 "Exponential smoothing": f_ses,
                 "AR(1)": f_ar1}
    preds = {k: [] for k in per_qubit}
    preds["Gradient boosting, lag features"] = []
    q_low, truth, origin_idx = [], [], []
    model = model_q = None

    for k, origin in enumerate(range(start - 1, n_t - h)):
        if k % REFIT_EVERY == 0:                 # refit the global model daily
            X, y = training_rows(logv, origin, h)
            model = gbm().fit(X, y)
            model_q = gbm("quantile", Q_LOW).fit(X, y)
        Xo = features(logv, origin)
        preds["Gradient boosting, lag features"].append(logv[origin] + model.predict(Xo))
        q_low.append(logv[origin] + model_q.predict(Xo))
        for name, f in per_qubit.items():
            preds[name].append(np.array([f(logv[:origin + 1, q], h) for q in range(n_q)]))
        truth.append(logv[origin + h])
        origin_idx.append(origin)

    y_true = np.exp(np.array(truth))
    out = {}
    for name, p in preds.items():
        f = np.exp(np.array(p))
        out[name] = {"MAE_us": mae(y_true, f), "RMSE_us": rmse(y_true, f), "forecast": f}
    base = out["Persistence (last calibration)"]
    for v in out.values():
        v["skill_MAE"] = 1 - v["MAE_us"] / base["MAE_us"]
        v["skill_RMSE"] = 1 - v["RMSE_us"] / base["RMSE_us"]
    return out, y_true, np.exp(np.array(q_low)), np.array(origin_idx)


def leakage_demo(t1_df: pd.DataFrame, h: int = 1):
    """Same gradient-boosting model, but evaluated with shuffled 5-fold CV."""
    logv = np.log(t1_df.to_numpy())
    X, y = training_rows(logv, logv.shape[0] - 1, h)
    last = X[:, 0]
    errs = []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = gbm().fit(X[tr], y[tr])
        errs.append(np.abs(np.exp(last[te] + y[te]) - np.exp(last[te] + m.predict(X[te]))))
    return float(np.mean(np.concatenate(errs)))


def evaluate_readout(ro_df: pd.DataFrame):
    v = ro_df.to_numpy()
    start = TRAIN_DAYS * 24
    models = {"Persistence (last value)": f_persistence,
              "Mean of past values": f_mean,
              "Moving average, 24 h": lambda hst, h: hst[-24:].mean(),
              "Exponential smoothing": f_ses}
    preds = {k: [] for k in models}
    truth = []
    for origin in range(start - 1, v.shape[0] - 1, 3):   # every 3 h to keep it quick
        for name, f in models.items():
            preds[name].append([f(v[:origin + 1, q], 1) for q in range(v.shape[1])])
        truth.append(v[origin + 1])
    y = np.array(truth)
    out = {}
    for name, p in preds.items():
        p = np.array(p)
        out[name] = {"MAE_pp": 100 * mae(y, p), "RMSE_pp": 100 * rmse(y, p)}
    base = out["Persistence (last value)"]
    for o in out.values():
        o["skill_MAE"] = 1 - o["MAE_pp"] / base["MAE_pp"]
        o["skill_RMSE"] = 1 - o["RMSE_pp"] / base["RMSE_pp"]
    return out


# --------------------------------------------------------------------------- #
# operational decision: which qubits to keep out of the next layout
# --------------------------------------------------------------------------- #
def decision(y_true, last_cal, point, low):
    bad = y_true < T1_THRESHOLD_US
    policies = {
        "Last calibration below 15 us": last_cal < T1_THRESHOLD_US,
        "Forecast below 15 us": point < T1_THRESHOLD_US,
        "Forecast 10% bound below 15 us": low < T1_THRESHOLD_US,
    }
    # same number of exclusions as the last-calibration policy, chosen by forecast risk
    k = policies["Last calibration below 15 us"].sum(axis=1)
    order = np.argsort(low, axis=1)
    matched = np.zeros_like(bad)
    for w in range(bad.shape[0]):
        matched[w, order[w, :k[w]]] = True
    policies["Forecast risk, same number excluded"] = matched
    n_windows = y_true.shape[0]
    res = {"bad_qubit_windows": int(bad.sum()), "windows": n_windows,
           "qubit_windows": int(bad.size)}
    for name, excl in policies.items():
        res[name] = {
            "caught": int((excl & bad).sum()),
            "missed": int((~excl & bad).sum()),
            "false_exclusions": int((excl & ~bad).sum()),
            "recall": float((excl & bad).sum() / bad.sum()),
            "excluded_per_window": float(excl.sum() / n_windows),
            "missed_per_window": float((~excl & bad).sum() / n_windows),
        }
    return res


def main():
    t1_df, ro_df = simulate(SEED)
    results = {"seed": SEED, "pilot_check": pilot_check(t1_df, ro_df)}

    for h in (1, 8):
        out, y_true, low, origins = evaluate_t1(t1_df, h)
        results[f"t1_h{h}"] = {k: {m: v[m] for m in ("MAE_us", "RMSE_us", "skill_MAE", "skill_RMSE")}
                               for k, v in out.items()}
        results[f"t1_h{h}"]["n_forecasts"] = int(y_true.size)
        last_cal = out["Persistence (last calibration)"]["forecast"]
        point = out["Gradient boosting, lag features"]["forecast"]
        results[f"decision_h{h}"] = decision(y_true, last_cal, point, low)
        if h == 1:
            np.savez(ROOT / "results" / "forecasts_h1.npz", y_true=y_true, last=last_cal, gbm=point,
                     low=low, ar1=out["AR(1)"]["forecast"], origins=origins,
                     index=t1_df.index.astype("int64"), t1=t1_df.to_numpy())

    results["leakage_shuffled_cv_MAE_us"] = leakage_demo(t1_df)
    results["readout_h1"] = evaluate_readout(ro_df)

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
