#!/bin/bash
# Runs the Open Quantum calibration probe from the repository root. Output -> run_log.txt
cd "$(dirname "$0")/.."
exec > >(tee run_log.txt) 2>&1
echo "== started $(date -u +%FT%TZ)"
PY=""
for c in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 "$HOME/.pyenv/shims/python" \
         "$HOME"/PycharmProjects/*/.venv/bin/python "$HOME"/PycharmProjects/*/venv/bin/python; do
  if "$c" -c "import openquantum_sdk" 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  python3 -m venv .venv && .venv/bin/pip install -q openquantum-sdk keyring platformdirs && PY=.venv/bin/python
fi
echo "== python: $PY ($("$PY" --version 2>&1))"

# SDK key: sdk_key.json in the repository root (git-ignored; its contents are never printed)
KEY=""
if [ -f sdk_key.json ]; then KEY="$PWD/sdk_key.json"
else
  for f in ./*.json; do
    [ -f "$f" ] && grep -q '"client_secret"' "$f" 2>/dev/null && KEY="$PWD/${f#./}" && break
  done
fi
if [ -n "$KEY" ]; then
  export OPENQUANTUM_SDK_KEY="$KEY"; chmod 600 "$KEY"
  echo "== using SDK key file: $(basename "$KEY")"
else
  echo "== no SDK key file here; trying a saved SDK account"
fi

echo; echo "== PROBE";  "$PY" src/oq_calibration.py probe || exit 1
echo; echo "== JOBS";   "$PY" src/oq_calibration.py jobs
echo; echo "== COLLECT (one poll)"; "$PY" src/oq_calibration.py collect --once
echo "== finished $(date -u +%FT%TZ)"
