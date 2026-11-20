# Predictive Models for Quantum System Stability

Student report, topic 10: forecasting how superconducting qubits drift between calibrations, evaluating the forecasts with a time-aware split, and turning them into an operational decision.

Authors: Kaiyr Zhaniya, Kaiyr Beiimbet

## What is here

| Path | Content |
|---|---|
| `src/oq_calibration.py` | Collector for real calibration documents from the Open Quantum platform (read-only, no jobs submitted); keeps every value with its standard error and the provider's capture time |
| `src/analyze.py` | Pilot analysis of the collected data: re-measurements, device medians, 2σ drift test, qubits to avoid, time-aware readout back-test |
| `scripts/run_probe.command` | One-click launcher for the collector on macOS |
| `src/simulate.py` | Synthetic 14-day calibration history for 107 qubits, calibrated to the pilot statistics |
| `src/forecast.py` | Forecast models, rolling-origin evaluation, MAE/RMSE, and the qubit-exclusion decision |
| `src/figures.py` | Figure: forecast skill against horizon |
| `tests/test_pipeline.py` | End-to-end check of collector parsing and analysis on synthetic calibration documents |
| `results/` | `results.json` with every number of the forecasting part |
| `figures/` | Figures used on the slides |
| `slides/` | Presentation (PDF) |

## Part 1: real pilot data

Open Quantum serves only the current calibration document of each device, so the history is built by polling it and keeping every new version (`src/oq_calibration.py`). A version counts as a re-measurement only if a metric changed for at least one qubit, and a change is significant if it exceeds 2σ of the combined measurement error.

Pilot findings on Rigetti Cepheus-1-108Q, IQM Emerald and IQM Garnet (181 qubits with T1/T2):

- Rigetti re-measures readout fidelity about once an hour, but T1, T2 and RB far less often, so frequent polling does not give a dense T1 series.
- Between two calibrations T1 changed by a median of 12.8%; 66 of 106 Rigetti qubits changed by more than 2σ.
- Median T1: 27 µs (Rigetti), 51 µs (Emerald), 45 µs (Garnet).

## Part 2: forecast models

The pilot gave too few T1 re-measurements for a time series, so models are compared on a synthetic history whose statistics match the pilot (median T1 26.7 vs 27 µs, median change 12.9 vs 12.8%). The generator follows the published picture of T1 fluctuations: a qubit-specific level, slow mean-reverting drift, and switching two-level-system defects.

**Models:** persistence (last calibration), 24-hour moving average, exponential smoothing, AR(1) on log T1, and a gradient-boosting model pooled over all qubits with lag features (also fitted as a 10% quantile).

**Time-aware split:** days 1–7 are the initial training window; every later calibration is a forecast origin, models are refit on all data before it, and the origin moves forward by 3 hours. No shuffling.

**Metrics:** MAE and RMSE on the original scale, and skill = 1 − MAE ÷ MAE of persistence.

| Model | MAE 3 h, µs | RMSE 3 h, µs | MAE 24 h, µs | RMSE 24 h, µs | Skill 24 h |
|---|---|---|---|---|---|
| Persistence | 6.5 | 11.1 | 13.3 | 19.7 | — |
| Moving average | 9.4 | 14.0 | 12.4 | 18.3 | +6% |
| Exponential smoothing | 6.8 | 11.2 | 12.8 | 19.0 | +4% |
| AR(1) | 6.7 | 10.8 | 10.7 | 15.5 | +19% |
| Gradient boosting | 6.7 | 10.6 | 10.9 | 16.1 | +18% |

Three hours ahead the last calibration is hard to beat; one day ahead AR(1) lowers MAE by 19% and RMSE by 21%. For readout fidelity one hour ahead, exponential smoothing lowers MAE from 0.58 to 0.45 percentage points.

## Operational decision

Before a 24-hour batch, rank qubits by forecast risk (the 10% lower bound of T1 one day ahead) and exclude as many as the current rule based on the last calibration does (19 of 107 on average). With the same number of exclusions, the forecast catches 48% of qubits whose T1 falls below 15 µs, against 41% for the last calibration. A cautious rule ("exclude if the 10% bound is below 15 µs") catches 84% but removes 53 of 107 qubits. Three hours ahead the forecast gives no gain.

## Reproduce

```bash
pip install -r requirements.txt
python src/simulate.py        # prints the pilot statistics of the synthetic data
python src/forecast.py        # writes results/results.json
python src/figures.py         # writes figures/fig_skill_vs_horizon.png
python tests/test_pipeline.py # checks collector parsing + pilot analysis
```

All randomness uses the fixed seed 2026, so the numbers above are reproduced exactly.

To collect and analyse real data, put your Open Quantum SDK key in the repository root as `sdk_key.json` (it is git-ignored), then from the repository root:

```bash
bash scripts/run_probe.command                                   # probe + one poll
python src/oq_calibration.py collect --interval-min 10 --hours 168   # long run
python src/analyze.py                                            # results/pilot_results.json, figures/pilot_*.png
```

The raw calibration snapshots of the pilot run are not included in this repository; the pilot numbers above are from that run and are reproduced by `analyze.py` on newly collected data.

## Limitations

- Model comparisons use simulated data calibrated to the pilot, not two measured weeks.
- The real T1 series is short: two re-measurements on Rigetti, one calibration per IQM device.
- The 15 µs threshold and the exclusion budget are illustrative.
- AQT and IonQ do not publish per-qubit T1 and T2.

## References

1. Open Quantum. Platform and Python SDK documentation (openquantum-sdk 1.2.1). https://docs.openquantum.com
2. P. V. Klimov et al. Fluctuations of energy-relaxation times in superconducting qubits. Phys. Rev. Lett. 121, 090502 (2018).
3. J. J. Burnett et al. Decoherence benchmarking of superconducting qubits. npj Quantum Information 5, 54 (2019).
4. M. Carroll, S. Rosenblatt, P. Jurcevic, I. Lauer, A. Kandala. Dynamics of superconducting qubit relaxation times. npj Quantum Information 8, 132 (2022).
5. T. Proctor et al. Detecting and tracking drift in quantum information processors. Nature Communications 11, 5396 (2020).
6. J. Kelly, P. O'Malley, M. Neeley, H. Neven, J. M. Martinis. Physical qubit calibration on a directed acyclic graph. arXiv:1803.03226 (2018).
7. R. J. Hyndman, G. Athanasopoulos. Forecasting: Principles and Practice, 3rd ed. OTexts, 2021. https://otexts.com/fpp3
8. J.-Y. Liu, H.-J. Ding, C.-M. Zhang, S.-P. Xie, Q. Wang. Practical phase-modulation stabilization in quantum key distribution via machine learning. Phys. Rev. Applied 12, 014059 (2019).
9. Z. A. Dahi, J. F. Chicano, G. Luque, I. Delgado. Optimising quantum calculations reliability via machine learning: the IBM case study (2025). https://hdl.handle.net/10630/41362
