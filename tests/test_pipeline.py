"""
End-to-end check of collector parsing + pilot analysis on synthetic documents.

Writes oq-cal-like calibration documents (same envelope as oq_calibration.py
stores) for one 107-qubit chip, published every 30 minutes, where readout is
re-measured every hour and T1/T2 only twice; then runs
`oq_calibration.py flatten` and `analyze.py` and checks the counts.

Run:  python tests/test_pipeline.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import oq_calibration as oq  # noqa: E402

N = 107
VERSIONS = 8                       # one document every 30 min
RO_EVERY = 2                       # readout re-measured every second version (hourly)
T1_AT = (0, 1)                     # T1/T2 re-measured at versions 0 and 1 only


def make_docs(tmp: Path) -> None:
    rng = np.random.default_rng(7)
    t1 = np.exp(np.log(27e-6) + rng.normal(0, 0.45, N))
    ro = np.clip(0.968 + rng.normal(0, 0.012, N), 0.9, 0.995)
    ro[:3] = [0.59, 0.68, 0.84]
    labels = {f"${i}": f"q{i}" for i in range(N)}
    for v in range(VERSIONS):
        if v in T1_AT and v > 0:
            t1 = t1 * np.exp(rng.normal(0, 0.18, N))
        if v % RO_EVERY == 0 and v > 0:
            ro = np.clip(ro + rng.normal(0, 0.005, N), 0.5, 0.999)
        qubits = {f"${i}": {
            "t1": {"value": float(t1[i]), "unit": "s", "stderr": float(0.04 * t1[i])},
            "t2": {"value": float(0.4 * t1[i]), "unit": "s", "stderr": float(0.016 * t1[i])},
            "readout_fidelity": {"value": float(ro[i]), "unit": "dimensionless", "stderr": 0.002},
        } for i in range(N)}
        captured = f"2026-11-19T15:{0 if v % 2 == 0 else 30:02d}:00Z".replace("15:", f"{15 + v // 2}:")
        doc = {"schema": "oq-cal/1", "captured_at": captured,
               "device": {"qubit_labels": labels, "enabled_qubits": list(labels)},
               "qubits": qubits}
        env = {"backend": "test:chip-107q", "collected_at": captured, "source": "poll",
               "doc_hash": oq.doc_hash(doc), "document": doc}
        folder = tmp / "data" / "raw" / "test_chip-107q"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"v{v}.json").write_text(json.dumps(env))


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_docs(tmp)
        subprocess.run([sys.executable, str(SRC / "oq_calibration.py"), "flatten"], cwd=tmp, check=True)
        out = tmp / "pilot.json"
        subprocess.run([sys.executable, str(SRC / "analyze.py"), "--csv", str(tmp / "data" / "calibration_long.csv"),
                        "--out", str(out), "--figures", str(tmp / "figs")], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        r = json.loads(out.read_text())
        b = "test:chip-107q"
        rm = r["cadence"][b]["re_measurements"]
        assert r["cadence"][b]["versions"] == VERSIONS, r["cadence"]
        assert rm["t1"] == 2 and rm["t2"] == 2, rm
        assert rm["readout"] == 4, rm                      # versions 0, 2, 4, 6
        pair = r["drift"][b]["t1"]["pairs"][0]
        assert pair["qubits"] == N and pair["with_stderr"] == N, pair
        assert 0.05 < pair["median_rel_change"] < 0.3, pair
        bt = r["readout_backtest"][b]
        assert bt["Last value"]["n_forecasts"] == 2 * N, bt   # forecasts of re-measurements 3 and 4
        assert r["avoid"][b]["low_readout"][0]["qubit"] == "q0", r["avoid"][b]
        assert abs(r["devices"][b]["median_t1"] - 27) < 8, r["devices"][b]
        print("pipeline test passed:", json.dumps({"re_measurements": rm, "t1_pair": pair,
                                                   "readout_backtest": bt}, indent=1))


if __name__ == "__main__":
    main()
