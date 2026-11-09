"""CaseFlow local service: parts library and simulation jobs.

CaseFlow is a simulator: every input (parts, fan layout, power, room temperature) is a setting, nothing is read from
the PC it runs on.

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

app = FastAPI(title="CaseFlow service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_lock = threading.Lock()          # one GPU simulation at a time


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "caseflow"}


@app.get("/api/library")
def library() -> dict:
    return sim.load_library()


@app.get("/api/default_build")
def default_build() -> dict:
    return sim.default_build()


class SimQuery(BaseModel):
    build: dict


def _run(jid: str, q: SimQuery) -> None:
    job = _jobs[jid]
    try:
        with _lock:
            r = sim.simulate_build(q.build, callback=lambda p: job.__setitem__("progress", p))
        job.update(result=sim.package(r), status="done", progress=1.0)
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
