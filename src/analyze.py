"""
Pilot analysis of the calibration history collected by oq_calibration.py.

Input : data/calibration_long.csv (one row per snapshot, qubit and metric)
Output: results/pilot_results.json and figures/pilot_*.png

Steps
 1. Versions and re-measurements. A document version is identified by its
    hash and stamped with the provider's capture time (captured_at). For a
    given metric, a version is a re-measurement only if the metric changed
    for at least one qubit; otherwise it is a repeat and is dropped.
 2. Device comparison: medians of T1, T2, readout and RB fidelity in the
    latest calibration of every device.
 3. Drift between consecutive re-measurements of T1 and T2: relative change
    per qubit, and whether |delta| > 2 sigma, where sigma combines the
    standard errors of both measurements.
 4. Qubits to avoid: the largest T1 jumps and the lowest readout fidelity.
 5. Time-aware back-test of the readout forecast: every re-measurement from
    the third on is forecast only from the earlier ones (last value, mean of
    past values); errors as MAE, RMSE and median absolute error.

Usage:  python src/analyze.py [--csv data/calibration_long.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GRAY, INK, MUTED, BG = "#1F6CAB", "#D9692B", "#8A939C", "#13202E", "#5F6A75", "#F6F5F1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": "#C9C4B8",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG,
    "axes.facecolor": BG, "axes.grid": True, "grid.color": "#E2DED4",
})


# --------------------------------------------------------------------------- #
# loading and metric names
# --------------------------------------------------------------------------- #
def canonical(metric: str, unit: str) -> str | None:
    """Map a provider's metric name to t1 / t2 / t2_echo / readout / readout_error / rb."""
    m = metric.lower()
    if m in ("t1",):
        return "t1"
    if m in ("t2", "t2_star", "t2_ramsey"):
        return "t2"
    if m in ("t2_echo", "t2e", "t2_hahn"):
        return "t2_echo"
    if "readout" in m or "assignment" in m or "measure" in m:
        return "readout_error" if "error" in m else "readout"
    if "rb" in m or "randomized" in m or "single_qubit_fidelity" in m:
        return "rb"
    return None


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["kind"] == "qubit"].copy()
    df["metric_c"] = [canonical(m, u) for m, u in zip(df["metric"], df["unit"].fillna(""))]
    df = df.dropna(subset=["metric_c"])
    # readout error -> readout fidelity
    err = df["metric_c"] == "readout_error"
    df.loc[err, "value"] = 1 - df.loc[err, "value"]
    df.loc[err, "metric_c"] = "readout"
    # seconds -> microseconds for coherence times
    coh = df["metric_c"].isin(["t1", "t2", "t2_echo"]) & (df["unit"] == "s")
    df.loc[coh, ["value", "stderr"]] = df.loc[coh, ["value", "stderr"]] * 1e6
    t = pd.to_datetime(df["doc_time"], utc=True, errors="coerce")
    df["time"] = t.fillna(pd.to_datetime(df["collected_at"], utc=True, errors="coerce"))
    # one row per (version, qubit, metric)
    return df.drop_duplicates(["backend", "doc_hash", "target", "metric_c"])


def remeasurements(df: pd.DataFrame, backend: str, metric: str) -> pd.DataFrame:
    """Wide table (rows = re-measurement times, columns = qubits) with repeats dropped."""
    sub = df[(df["backend"] == backend) & (df["metric_c"] == metric)]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    val = sub.pivot_table(index="time", columns="target", values="value", aggfunc="first").sort_index()
    se = sub.pivot_table(index="time", columns="target", values="stderr", aggfunc="first").reindex(val.index)
    keep = [0]
    for i in range(1, len(val)):
        prev, cur = val.iloc[keep[-1]], val.iloc[i]
        if not np.allclose(prev.to_numpy(), cur.to_numpy(), equal_nan=True, rtol=0, atol=0):
            keep.append(i)
    return val.iloc[keep], se.iloc[keep].reindex(columns=val.columns)


# --------------------------------------------------------------------------- #
# analysis steps
# --------------------------------------------------------------------------- #
def cadence(df: pd.DataFrame) -> dict:
    out = {}
    for b, sub in df.groupby("backend"):
        span = sub["time"].agg(["min", "max"])
        out[b] = {"versions": int(sub["doc_hash"].nunique()),
                  "first": str(span["min"]), "last": str(span["max"]),
                  "re_measurements": {m: int(len(remeasurements(df, b, m)[0]))
                                      for m in sorted(sub["metric_c"].unique())}}
    return out


def device_medians(df: pd.DataFrame) -> dict:
    out = {}
    for b, sub in df.groupby("backend"):
        last = sub[sub["time"] == sub["time"].max()]
        out[b] = {"qubits": int(last["target"].nunique())}
        for m in ("t1", "t2", "t2_echo", "readout", "rb"):
            v = last.loc[last["metric_c"] == m, "value"]
            if len(v):
                out[b][f"median_{m}"] = float(v.median())
                if m == "readout":
                    out[b]["min_readout"] = float(v.min())
    return out


def drift(df: pd.DataFrame, backend: str, metric: str) -> dict | None:
    val, se = remeasurements(df, backend, metric)
    if len(val) < 2:
        return None
    pairs = []
    for i in range(1, len(val)):
        a, b = val.iloc[i - 1], val.iloc[i]
        sa, sb = se.iloc[i - 1], se.iloc[i]
        ok = a.notna() & b.notna() & (a != b)        # a qubit whose value was not updated is skipped
        d = (b - a)[ok]
        rel = (d.abs() / a[ok])
        sigma = np.sqrt(sa[ok] ** 2 + sb[ok] ** 2)
        sig = (d.abs() > 2 * sigma) & sigma.notna()
        pairs.append({"from": str(val.index[i - 1]), "to": str(val.index[i]),
                      "qubits": int(ok.sum()),
                      "median_rel_change": float(rel.median()),
                      "p90_rel_change": float(rel.quantile(0.9)),
                      "significant_2sigma": int(sig.sum()),
                      "with_stderr": int(sigma.notna().sum()),
                      "increased": int((d > 0).sum())})
    return {"re_measurements": len(val), "pairs": pairs}


def qubits_to_avoid(df: pd.DataFrame, backend: str, n: int = 5) -> dict:
    t1, _ = remeasurements(df, backend, "t1")
    ro, _ = remeasurements(df, backend, "readout")
    out = {}
    if len(t1) >= 2:
        a, b = t1.iloc[-2], t1.iloc[-1]
        ratio = (b / a).dropna()
        worst = (np.log(ratio).abs()).sort_values(ascending=False).head(n).index
        out["unstable_t1"] = [{"qubit": q, "before_us": float(a[q]), "after_us": float(b[q]),
                               "ratio": float(ratio[q])} for q in worst]
    if len(ro):
        last = ro.iloc[-1].dropna().sort_values().head(n)
        t1_last = t1.iloc[-1] if len(t1) else pd.Series(dtype=float)
        out["low_readout"] = [{"qubit": q, "readout": float(v),
                               "t1_us": float(t1_last.get(q, np.nan)),
                               "low_in_all_remeasurements": bool((ro[q] < 0.9).all())}
                              for q, v in last.items()]
        out["chip_median_readout"] = float(ro.iloc[-1].median())
    return out


def readout_backtest(df: pd.DataFrame, backend: str) -> dict | None:
    ro, _ = remeasurements(df, backend, "readout")
    if len(ro) < 3:
        return None
    errs = {"Last value": [], "Mean of past values": []}
    for k in range(2, len(ro)):                       # forecast k from 0..k-1 only
        past, truth = ro.iloc[:k], ro.iloc[k]
        for name, f in (("Last value", past.iloc[-1]), ("Mean of past values", past.mean())):
            e = (truth - f).dropna()
            errs[name].append(e.to_numpy())
    out = {"re_measurements": len(ro)}
    for name, e in errs.items():
        e = np.concatenate(e)
        out[name] = {"n_forecasts": int(e.size),
                     "MAE_pp": float(100 * np.mean(np.abs(e))),
                     "RMSE_pp": float(100 * np.sqrt(np.mean(e ** 2))),
                     "median_abs_error_pp": float(100 * np.median(np.abs(e)))}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_devices(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=200)
    for ax, m, label in ((axes[0], "t1", "T1, µs"), (axes[1], "readout", "readout fidelity")):
        data, names = [], []
        for b, sub in df.groupby("backend"):
            last = sub[sub["time"] == sub["time"].max()]
            v = last.loc[last["metric_c"] == m, "value"].dropna()
            if len(v):
                data.append(v.to_numpy()); names.append(b.split(":")[-1])
        if data:
            ax.boxplot(data, tick_labels=names, medianprops={"color": ORANGE, "lw": 2},
                       boxprops={"color": BLUE}, whiskerprops={"color": GRAY},
                       capprops={"color": GRAY}, flierprops={"markeredgecolor": GRAY, "markersize": 4})
        ax.set_ylabel(label)
    fig.tight_layout(); fig.savefig(path, facecolor=BG); plt.close(fig)


def fig_drift(df: pd.DataFrame, backend: str, path: Path) -> None:
    val, se = remeasurements(df, backend, "t1")
    if len(val) < 2:
        return
    a, b = val.iloc[-2], val.iloc[-1]
    sigma = np.sqrt(se.iloc[-2] ** 2 + se.iloc[-1] ** 2)
    ok = a.notna() & b.notna() & (a != b)
    sig = ((b - a).abs() > 2 * sigma) & ok
    lim = float(np.nanmax([a.max(), b.max()])) * 1.05
    fig, ax = plt.subplots(figsize=(6.2, 5.5), dpi=200)
    x = np.linspace(0, lim, 50)
    ax.fill_between(x, 0.8 * x, 1.2 * x, color="#DCE6F0", lw=0, label="±20%")
    ax.plot(x, x, color=GRAY, lw=1)
    ax.scatter(a[ok & ~sig], b[ok & ~sig], s=28, color=BLUE, edgecolor=BG, lw=1, label="within 2σ")
    ax.scatter(a[sig], b[sig], s=28, color=ORANGE, edgecolor=BG, lw=1, label="change > 2σ")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("T1 before re-measurement, µs"); ax.set_ylabel("T1 after re-measurement, µs")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout(); fig.savefig(path, facecolor=BG); plt.close(fig)


def fig_readout(df: pd.DataFrame, backend: str, path: Path) -> None:
    ro, _ = remeasurements(df, backend, "readout")
    if len(ro) < 2:
        return
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    ax.fill_between(ro.index, ro.quantile(0.1, axis=1), ro.quantile(0.9, axis=1),
                    color="#DCE6F0", lw=0, label="10–90% of qubits")
    ax.plot(ro.index, ro.median(axis=1), color=INK, lw=2, label="median")
    for q in ro.iloc[-1].nsmallest(2).index:
        ax.plot(ro.index, ro[q], color=ORANGE, lw=1.5, marker="o", ms=5)
        ax.text(ro.index[-1], ro[q].iloc[-1], f"  {q}", color=INK, fontsize=12, va="center")
    ax.set_ylabel("readout fidelity"); ax.set_xlabel("calibration time, UTC")
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, facecolor=BG); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "calibration_long.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot_results.json"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()

    df = load(Path(args.csv))
    figs = Path(args.figures); figs.mkdir(parents=True, exist_ok=True)
    res = {"cadence": cadence(df), "devices": device_medians(df), "drift": {}, "avoid": {}, "readout_backtest": {}}
    for b in sorted(df["backend"].unique()):
        res["drift"][b] = {m: drift(df, b, m) for m in ("t1", "t2")}
        res["avoid"][b] = qubits_to_avoid(df, b)
        res["readout_backtest"][b] = readout_backtest(df, b)
        tag = b.replace(":", "_")
        fig_drift(df, b, figs / f"pilot_t1_drift_{tag}.png")
        fig_readout(df, b, figs / f"pilot_readout_{tag}.png")
    fig_devices(df, figs / "pilot_devices.png")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
