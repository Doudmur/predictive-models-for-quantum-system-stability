"""
Synthetic calibration history for one superconducting chip.

The generator is parameterised from the pilot measurements on Rigetti
Cepheus-1-108Q (Open Quantum calibration documents):
  * 107 qubits, median T1 = 27 us, median readout fidelity = 0.968;
  * median relative change of T1 between consecutive re-measurements = 12.8 %,
    90th percentile = 55 %;
  * readout fidelity changes by about 0.5 percentage points per hour;
  * a few qubits have very low readout fidelity (down to 0.59).

Physical picture (Klimov et al. 2018, Burnett et al. 2019, Carroll et al. 2022):
  log T1 = qubit baseline + slow mean-reverting drift + two-level-system (TLS)
  "dips" that switch on and off + fit noise of the T1 measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_QUBITS = 107
DAYS = 14
T1_STEP_H = 3           # T1 is re-measured every 3 hours
RO_STEP_H = 1           # readout fidelity is re-measured every hour

# T1 dynamics (per 3-hour step, natural-log units)
T1_MEDIAN_US = 31.0       # baseline; TLS dips pull the observed median to ~27 us
T1_BASE_SD = 0.45       # qubit-to-qubit spread of the baseline
DRIFT_PHI = 0.90        # AR(1) memory of the slow drift
DRIFT_SD = 0.13         # innovation of the slow drift
TLS_DEPTH = 0.90        # a TLS dip lowers T1 by ~x0.41
TLS_P_ON = 0.06         # chance a dip appears in a step (stable qubits)
TLS_P_ON_UNSTABLE = 0.20
TLS_P_OFF = 0.30        # chance a dip disappears in a step
UNSTABLE_SHARE = 0.12   # share of qubits with an active TLS neighbourhood
FIT_SD = 0.07           # measurement (fit) noise of T1

# readout dynamics (per 1-hour step, fidelity units)
RO_MEDIAN = 0.968
RO_LEVEL_PHI = 0.995
RO_LEVEL_SD = 0.0012
RO_NOISE_SD = 0.0052
RO_BAD_SHARE = 0.05     # qubits with poor readout (0.6-0.87)
RO_RECAL_P = 0.004      # chance per hour that a bad qubit is fixed by recalibration


def simulate(seed: int = 2026):
    rng = np.random.default_rng(seed)
    n_t1 = DAYS * 24 // T1_STEP_H
    n_ro = DAYS * 24 // RO_STEP_H

    # ---------------- T1 ----------------
    base = np.log(T1_MEDIAN_US) + rng.normal(0, T1_BASE_SD, N_QUBITS)
    unstable = rng.random(N_QUBITS) < UNSTABLE_SHARE
    p_on = np.where(unstable, TLS_P_ON_UNSTABLE, TLS_P_ON)
    drift = np.zeros(N_QUBITS)
    dip = np.zeros(N_QUBITS, dtype=bool)
    t1 = np.empty((n_t1, N_QUBITS))
    for t in range(n_t1):
        drift = DRIFT_PHI * drift + rng.normal(0, DRIFT_SD, N_QUBITS)
        turn_on = (~dip) & (rng.random(N_QUBITS) < p_on)
        turn_off = dip & (rng.random(N_QUBITS) < TLS_P_OFF)
        dip = (dip | turn_on) & ~turn_off
        log_t1 = base + drift - TLS_DEPTH * dip + rng.normal(0, FIT_SD, N_QUBITS)
        t1[t] = np.exp(log_t1)

    # ---------------- readout ----------------
    bad = rng.random(N_QUBITS) < RO_BAD_SHARE
    level = np.where(bad, rng.uniform(0.60, 0.87, N_QUBITS),
                     np.clip(RO_MEDIAN + rng.normal(0, 0.012, N_QUBITS), 0.90, 0.995))
    target = level.copy()
    dev = np.zeros(N_QUBITS)
    ro = np.empty((n_ro, N_QUBITS))
    for t in range(n_ro):
        fixed = bad & (rng.random(N_QUBITS) < RO_RECAL_P)
        target = np.where(fixed, np.clip(RO_MEDIAN + rng.normal(0, 0.008, N_QUBITS), 0.94, 0.99), target)
        bad = bad & ~fixed
        dev = RO_LEVEL_PHI * dev + rng.normal(0, RO_LEVEL_SD, N_QUBITS)
        ro[t] = np.clip(target + dev + rng.normal(0, RO_NOISE_SD, N_QUBITS), 0.5, 0.999)

    qubits = [f"q{i:03d}" for i in range(N_QUBITS)]
    t1_df = pd.DataFrame(t1, columns=qubits)
    t1_df.index = pd.date_range("2026-11-01", periods=n_t1, freq=f"{T1_STEP_H}h", tz="UTC")
    ro_df = pd.DataFrame(ro, columns=qubits)
    ro_df.index = pd.date_range("2026-11-01", periods=n_ro, freq=f"{RO_STEP_H}h", tz="UTC")
    return t1_df, ro_df


def pilot_check(t1_df: pd.DataFrame, ro_df: pd.DataFrame) -> dict:
    """Same statistics as in the pilot, computed on the synthetic data."""
    rel = (t1_df.diff().abs() / t1_df.shift()).iloc[1:].to_numpy().ravel()
    d_ro = ro_df.diff().abs().iloc[1:].to_numpy().ravel()
    return {
        "t1_median_us": float(np.median(t1_df.to_numpy())),
        "t1_rel_change_median": float(np.median(rel)),
        "t1_rel_change_p90": float(np.quantile(rel, 0.9)),
        "ro_median": float(np.median(ro_df.to_numpy())),
        "ro_min": float(ro_df.to_numpy().min()),
        "ro_abs_change_median": float(np.median(d_ro)),
    }


if __name__ == "__main__":
    t1, ro = simulate()
    for k, v in pilot_check(t1, ro).items():
        print(f"{k:24s} {v:.4f}")
