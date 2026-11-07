#!/usr/bin/env bash
# Keep the CaseFlow engine alive (restarts it if it exits). Run from anywhere.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=../.venv/Scripts/python
while true; do
  echo "[serve] starting $(date +%H:%M:%S)"
  "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port 8600
  echo "[serve] exited with $? at $(date +%H:%M:%S); restarting in 3s"
  sleep 3
done
