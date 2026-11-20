#!/usr/bin/env python3
"""
Collect qubit calibration snapshots from Open Quantum and turn them into a
time series for forecasting (T1, T2, frequency, gate errors, readout, ...).

Open Quantum does not expose device history. It exposes:
  * the CURRENT public calibration document of each backend
    (SchedulerClient.get_backend_class(short_code)["calibration_data_url"]),
  * the calibration snapshot tagged on each of YOUR finished jobs
    (JobList.calibration_data_url).
So we build the history ourselves: poll the current document on a schedule,
keep every distinct version, and add the snapshots from past jobs.
Reading calibration documents only calls read endpoints; it submits no jobs.

Auth: uses your saved Open Quantum account (same as your other SDK scripts),
or OPENQUANTUM_CLIENT_ID / OPENQUANTUM_CLIENT_SECRET / OPENQUANTUM_SDK_KEY.

Usage
  python oq_calibration.py probe                 # what is in the documents?
  python oq_calibration.py jobs                  # snapshots from your past jobs
  python oq_calibration.py collect --once        # one poll (for cron / Task Scheduler)
  python oq_calibration.py collect --interval-min 30 --hours 168   # loop for 7 days
  python oq_calibration.py flatten               # rebuild data/calibration_long.csv
  python analyze.py                              # pilot analysis of the collected data

Output (folder ./data)
  raw/<backend>/<UTC>_<source>_<hash>.json   every distinct document (untouched,
                                              inside a small metadata envelope)
  calibration_long.csv                        one row per (snapshot, qubit/gate, metric)
  poll_log.csv                                every poll, even unchanged ones
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests

DATA = Path("data")
RAW = DATA / "raw"
LONG_CSV = DATA / "calibration_long.csv"
POLL_LOG = DATA / "poll_log.csv"
TIME_HINTS = ("time", "date", "timestamp", "created", "updated", "calibrated", "valid")


# --------------------------------------------------------------------------- #
# Open Quantum access
# --------------------------------------------------------------------------- #
def make_client():
    """Saved SDK account first; otherwise OPENQUANTUM_SDK_KEY / env credentials."""
    import os
    from openquantum_sdk.session import OpenQuantumClient

    if os.getenv("OPENQUANTUM_SDK_KEY") or os.getenv("OPENQUANTUM_CLIENT_ID") \
            or os.getenv("OPENQUANTUM_TOKEN"):
        client = OpenQuantumClient.from_env()
    else:
        client = OpenQuantumClient()  # autoloads the saved "default" account
    try:
        client.organizations()
    except Exception as e:
        sys.exit(
            f"Open Quantum authentication failed ({type(e).__name__}).\n"
            "Put your SDK key file (JSON with client_id and client_secret, from the\n"
            "Open Quantum dashboard) next to this script as sdk_key.json, or save an\n"
            "account with OpenQuantumClient.login(..., save=True)."
        )
    return client


def qpu_backends(client, only: Optional[List[str]]) -> List[Dict[str, Any]]:
    items = client.backends()
    out = [b for b in items if (b.get("type") or "").upper() == "QPU"]
    if only:
        wanted = {s.lower() for s in only}
        out = [b for b in out if (b.get("short_code") or "").lower() in wanted]
    return out


def fetch_json(url: str) -> Any:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def current_document(client, short_code: str):
    details = client.scheduler.get_backend_class(short_code)
    url = details.get("calibration_data_url") if isinstance(details, dict) else None
    if not url:
        return None, details
    return fetch_json(url), details


# --------------------------------------------------------------------------- #
# Storage helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def doc_hash(doc: Any) -> str:
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def known_hashes(backend: str) -> set:
    folder = RAW / safe(backend)
    if not folder.exists():
        return set()
    return {p.stem.rsplit("_", 1)[-1] for p in folder.glob("*.json")}


def save_raw(backend: str, doc: Any, stamp: str, source: str) -> Optional[Path]:
    """Save the document unless an identical one is already stored."""
    h = doc_hash(doc)
    if h in known_hashes(backend):
        return None
    folder = RAW / safe(backend)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe(stamp.replace(':', ''))}_{safe(source)}_{h}.json"
    envelope = {"backend": backend, "collected_at": stamp, "source": source,
                "doc_hash": h, "document": doc}
    path.write_text(json.dumps(envelope, indent=1, sort_keys=True))
    return path


def append_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# Document parsing (generic: we keep every {"value", "unit", stderr} we can find)
# --------------------------------------------------------------------------- #
def find_timestamps(doc: Any, depth: int = 0, prefix: str = "") -> Dict[str, str]:
    """Collect scalar fields whose name looks like a time (top 3 levels)."""
    found: Dict[str, str] = {}
    if depth > 2 or not isinstance(doc, Mapping):
        return found
    for k, v in doc.items():
        key = f"{prefix}{k}"
        if isinstance(v, (str, int, float)) and any(h in str(k).lower() for h in TIME_HINTS):
            found[key] = str(v)
        elif isinstance(v, Mapping) and k not in ("qubits",):
            found.update(find_timestamps(v, depth + 1, key + "."))
    return found


STDERR_KEYS = ("stderr", "std_err", "standard_error", "uncertainty", "sigma", "std", "err")


def datum(raw: Any):
    """(value, unit, standard error or None) from a {"value", "unit", ...} entry."""
    if isinstance(raw, Mapping) and "value" in raw:
        try:
            value = float(raw["value"])
        except (TypeError, ValueError):
            return None
        se = None
        for k in STDERR_KEYS:
            if k in raw:
                try:
                    se = float(raw[k])
                except (TypeError, ValueError):
                    se = None
                break
        return value, str(raw.get("unit", "")), se
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), "", None
    return None


def document_time(doc: Any) -> str:
    """The provider's capture time: captured_at if present, else the first time-like field."""
    ts = find_timestamps(doc)
    for key, value in ts.items():
        if key.split(".")[-1] == "captured_at":
            return value
    return next(iter(ts.values()), "")


def flatten_document(doc: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(doc, Mapping):
        return rows
    device = doc.get("device") if isinstance(doc.get("device"), Mapping) else {}
    labels = device.get("qubit_labels") if isinstance(device.get("qubit_labels"), Mapping) else {}

    qubits = doc.get("qubits")
    if isinstance(qubits, Mapping):
        for ref, body in qubits.items():
            if not isinstance(body, Mapping):
                continue
            for metric, raw in body.items():
                d = datum(raw)
                if d:
                    rows.append({"kind": "qubit", "target": labels.get(ref, ref),
                                 "metric": metric, "value": d[0], "unit": d[1], "stderr": d[2]})

    gates = doc.get("gates")
    if isinstance(gates, list):
        for g in gates:
            if not isinstance(g, Mapping):
                continue
            name = str(g.get("name") or "gate")
            refs = g.get("qubits") if isinstance(g.get("qubits"), list) else []
            target = "-".join(str(labels.get(r, r)) for r in refs)
            for metric, raw in g.items():
                if metric in ("name", "qubits"):
                    continue
                d = datum(raw)
                if d:
                    rows.append({"kind": f"gate:{name}", "target": target,
                                 "metric": metric, "value": d[0], "unit": d[1], "stderr": d[2]})
    return rows


LONG_FIELDS = ["collected_at", "source", "backend", "doc_hash", "doc_time",
               "kind", "target", "metric", "value", "unit", "stderr"]


def rows_for(doc: Mapping[str, Any], backend: str, stamp: str, source: str):
    doc_time = document_time(doc)
    h = doc_hash(doc)
    for r in flatten_document(doc):
        yield {"collected_at": stamp, "source": source, "backend": backend,
               "doc_hash": h, "doc_time": doc_time, **r}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_probe(args) -> None:
    client = make_client()
    backends = qpu_backends(client, args.backends)
    print(f"QPU backends visible: {len(backends)}\n")
    for b in backends:
        sc = b["short_code"]
        print(f"=== {sc}  ({b.get('name')}, status={b.get('status')}, "
              f"accepting_jobs={b.get('accepting_jobs')})")
        try:
            doc, _ = current_document(client, sc)
        except Exception as e:
            print(f"   error: {e}\n")
            continue
        if doc is None:
            print("   no public calibration document\n")
            continue
        stamp = utc_now()
        path = save_raw(sc, doc, stamp, "probe")
        rows = flatten_document(doc)
        metrics = sorted({(r["kind"].split(":")[0], r["metric"], r["unit"]) for r in rows})
        n_q = len({r["target"] for r in rows if r["kind"] == "qubit"})
        print(f"   top-level keys : {sorted(doc.keys()) if isinstance(doc, Mapping) else type(doc)}")
        print(f"   time fields    : {find_timestamps(doc) or 'none found'}")
        n_se = sum(r["stderr"] is not None for r in rows)
        print(f"   with stderr    : {n_se} of {len(rows)} values")
        print(f"   qubits w/ data : {n_q}")
        print(f"   metrics        : {metrics}")
        print(f"   saved          : {path or 'identical document already stored'}\n")


def cmd_jobs(args) -> None:
    client = make_client()
    sched = client.scheduler
    id_to_code = {b["id"]: b["short_code"] for b in client.backends()}
    cursor, n_jobs, n_new = None, 0, 0
    while True:
        page = sched.list_jobs(limit=50, cursor=cursor)
        for job in page.jobs:
            n_jobs += 1
            url = getattr(job, "calibration_data_url", None)
            if not url:
                continue
            bc_id = getattr(job, "backend_class_id", None)
            backend = id_to_code.get(bc_id) or getattr(job, "backend_name", None) or str(bc_id)
            stamp = getattr(job, "completed_at", None) or getattr(job, "submitted_at", None) or utc_now()
            try:
                doc = fetch_json(url)
            except Exception as e:
                print(f"job {job.id}: {e}")
                continue
            if save_raw(backend, doc, str(stamp), f"job-{str(job.id)[:8]}"):
                append_csv(LONG_CSV, rows_for(doc, backend, str(stamp), "job"), LONG_FIELDS)
                n_new += 1
        cursor = page.pagination.next_cursor
        if not cursor:
            break
    print(f"jobs scanned: {n_jobs}; new calibration snapshots saved: {n_new}")


def poll_once(client, backends: List[str]) -> None:
    stamp = utc_now()
    log = []
    for sc in backends:
        status, h = "ok", ""
        try:
            doc, details = current_document(client, sc)
            if doc is None:
                status = "no-calibration-url"
            else:
                h = doc_hash(doc)
                if save_raw(sc, doc, stamp, "poll"):
                    append_csv(LONG_CSV, rows_for(doc, sc, stamp, "poll"), LONG_FIELDS)
                    status = "NEW"
                else:
                    status = "unchanged"
        except Exception as e:
            status = f"error: {e}"
        log.append({"polled_at": stamp, "backend": sc, "status": status, "doc_hash": h})
        print(f"{stamp}  {sc:28s} {status}")
    append_csv(POLL_LOG, log, ["polled_at", "backend", "status", "doc_hash"])


def cmd_collect(args) -> None:
    client = make_client()
    backends = [b["short_code"] for b in qpu_backends(client, args.backends)]
    if not backends:
        sys.exit("No QPU backends found.")
    if args.once:
        poll_once(client, backends)
        return
    end = time.time() + args.hours * 3600
    while time.time() < end:
        poll_once(client, backends)
        time.sleep(args.interval_min * 60)


def cmd_flatten(args) -> None:
    if LONG_CSV.exists():
        LONG_CSV.unlink()
    n = 0
    for path in sorted(RAW.glob("*/*.json")):
        env = json.loads(path.read_text(encoding="utf-8"))
        append_csv(LONG_CSV, rows_for(env["document"], env["backend"],
                                      env["collected_at"], env["source"]), LONG_FIELDS)
        n += 1
    print(f"rebuilt {LONG_CSV} from {n} documents")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "jobs", "collect", "flatten"):
        s = sub.add_parser(name)
        s.add_argument("--backends", nargs="*", help="short codes, e.g. iqm:garnet iqm:emerald")
        if name == "collect":
            s.add_argument("--once", action="store_true", help="single poll, then exit")
            s.add_argument("--interval-min", type=float, default=30)
            s.add_argument("--hours", type=float, default=168)
    args = p.parse_args()
    {"probe": cmd_probe, "jobs": cmd_jobs, "collect": cmd_collect, "flatten": cmd_flatten}[args.cmd](args)


if __name__ == "__main__":
    main()
