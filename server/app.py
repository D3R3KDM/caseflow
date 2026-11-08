"""CaseFlow local service: hardware readout, parts library, and simulation jobs.

Run from the caseflow folder:
    python -m uvicorn server.app:app --host 127.0.0.1 --port 8600
"""
from __future__ import annotations

import sys
import threading
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from engine import sim  # noqa: E402
from engine.hardware import detect, gpu_sensors  # noqa: E402
from engine.thermal import MIN_CAL_WATTS, calibrate_g  # noqa: E402

app = FastAPI(title="CaseFlow service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_lock = threading.Lock()          # one GPU simulation at a time
_hw: dict = {}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "caseflow"}


@app.get("/api/hardware")
def hardware(refresh: bool = False) -> dict:
    """CPU / board / RAM once (Windows CIM), GPU sensors (nvidia-smi) on every call."""
    if refresh or not _hw:
        _hw.clear()
        _hw.update(detect())
    else:
        _hw["gpus"] = gpu_sensors()
    return _hw


@app.get("/api/library")
def library() -> dict:
    return sim.load_library()


@app.get("/api/default_build")
def default_build() -> dict:
    return sim.default_build()


class SimQuery(BaseModel):
    build: dict
    calib_gpu: dict | None = None    # {"temp_c": .., "watts": ..} measured under load
    calib_cpu: dict | None = None


def _run(jid: str, q: SimQuery) -> None:
    job = _jobs[jid]
    try:
        with _lock:
            r = sim.simulate_build(q.build, callback=lambda p: job.__setitem__("progress", p))
        notes = []
        for key, cal in (("gpu", q.calib_gpu), ("cpu", q.calib_cpu)):
            if cal and cal.get("temp_c") and cal.get("watts"):
                g = calibrate_g(float(cal["temp_c"]), float(cal["watts"]), r["load"]["ambient_c"],
                                r["parts"][key]["intake_rise_c"])
                if g:
                    p = r["parts"][key]
                    p.update(g_w_per_k=g, calibrated=True,
                             temp_c=r["load"]["ambient_c"] + p["intake_rise_c"] + p["watts"] / g)
                    notes.append(f"{key.upper()} cooler calibrated: {g:.2f} W/K")
                elif float(cal["watts"]) < MIN_CAL_WATTS:
                    notes.append(f"{key.upper()} calibration skipped: needs a reading under load (>= {MIN_CAL_WATTS:.0f} W)")
        job.update(result=sim.package(r, notes), status="done", progress=1.0)
    except Exception as exc:  # pragma: no cover - reported to the client
        job.update(status="error", error=f"{exc!r} {traceback.format_exc()[-400:]}")


@app.post("/api/simulate")
def simulate(q: SimQuery) -> dict:
    jid = uuid.uuid4().hex[:10]
    _jobs[jid] = {"status": "running", "progress": 0.0}
    threading.Thread(target=_run, args=(jid, q), daemon=True).start()
    return {"job": jid}


@app.get("/api/job/{jid}")
def job(jid: str) -> dict:
    if jid not in _jobs:
        raise HTTPException(404, "no such job")
    return _jobs[jid]
