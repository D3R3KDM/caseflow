"""CaseFlow local service: parts library, simulation jobs and the neural surrogate.

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

from engine import bank, sim, surrogate  # noqa: E402

app = FastAPI(title="CaseFlow service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_lock = threading.Lock()          # one GPU simulation at a time
MAX_JOBS = 24                     # finished jobs kept for polling (each holds a ~6 MB bank package)
_ai: surrogate.Surrogate | None = None
_ai_lock = threading.Lock()


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
    simulate: bool = True        # False: only a cached bank; nothing runs on the GPU for a layout never simulated


def _run_bank(jid: str, q: SimQuery) -> None:
    job = _jobs[jid]
    try:
        b = bank.load(bank.bank_key(q.build))        # a cached bank needs no GPU: never wait behind a simulation
        if b is None:
            with _lock:
                b = bank.get(q.build, callback=lambda p: job.__setitem__("progress", p))
        job.update(result=bank.package(b), status="done", progress=1.0)
    except Exception as exc:  # pragma: no cover - reported to the client
        job.update(status="error", error=f"{exc!r} {traceback.format_exc()[-400:]}")


@app.post("/api/bank")
def make_bank(q: SimQuery) -> dict:
    """A build's bank (airflow, heat and dust at three case-fan duties): from the cache in data/banks, or computed on
    the GPU (about 1.5 min). The browser then solves any room temperature, load and fan curve itself. With simulate
    false a build that was never simulated returns no job: the web shows the network's prediction until asked."""
    jid = uuid.uuid4().hex[:10]
    cached = (bank.CACHE / f"{bank.bank_key(q.build)}.json").exists()      # the .json is written last
    if not cached and not q.simulate:
        return {"job": None, "cached": False}
    for old in [k for k, v in _jobs.items() if v["status"] != "running"][:max(0, len(_jobs) - MAX_JOBS)]:
        del _jobs[old]                                                       # oldest finished jobs first
    _jobs[jid] = {"status": "running", "progress": 0.0, "cached": cached}
    threading.Thread(target=_run_bank, args=(jid, q), daemon=True).start()
    return {"job": jid, "cached": cached}


def _surrogate() -> surrogate.Surrogate:
    global _ai
    with _ai_lock:
        if _ai is None:
            if not surrogate.CHECKPOINT.exists():
                raise HTTPException(503, "no trained network yet (scripts/train_surrogate.py)")
            _ai = surrogate.Surrogate()
        return _ai


@app.get("/api/ai/info")
def ai_info() -> dict:
    """The network: architecture, what it was trained on and how it does on the held-out test layouts."""
    return {k: v for k, v in _surrogate().info.items() if k not in ("train_keys", "test_keys")}


@app.post("/api/ai")
def ai_bank(q: SimQuery) -> dict:
    """The neural surrogate's bank for a build, in the format of /api/bank, in well under a second. The app shows it
    at once and replaces it with the simulation's when that finishes. 422 for builds unlike the training data."""
    why = surrogate.out_of_domain(q.build)
    if why:
        raise HTTPException(422, why)
    net = _surrogate()
    with _ai_lock:
        b = net.predict(q.build)
    return bank.package(b)


@app.get("/api/job/{jid}")
def job(jid: str) -> dict:
    if jid not in _jobs:
        raise HTTPException(404, "no such job")
    return _jobs[jid]
