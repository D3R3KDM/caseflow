"""Bank: the airflow, heat and dust of one build at a few case-fan duties, precomputed once so that any room
temperature, CPU / GPU power and fan curve can be solved instantly (engine/scenario.py).

Per level (a duty shared by every case fan that follows the case-fan curve):
- the airflow (lattice Boltzmann; all levels in one GPU batch) and every fan's operating point;
- the air temperature rise per watt of CPU power and per watt of GPU power (one batched heat solve);
- dust: how much outside dust gets in and where it settles.

The summary is small JSON (what the solver needs); the 3D fields (air velocity, per-watt rise) are float16 arrays.
Both are cached in data/banks under a key of everything that shapes the airflow: case, GPU, cooler and fans, but
not the CPU, the curves or the load, which the solver handles.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from engine.sim import (DX, U_LAT_PER_MS, build_model, deposit_points, dust_by_part, find, load_library,
                        run_airflow_batch, simulate_dust)
from engine.thermal import solve_unit_fields

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "banks"
LEVELS = (0.12, 0.45, 1.0)        # case-fan duties: the curve's floor, a mid point, full speed
BANK_VERSION = 1
# library fields that do not change the airflow (cooling, power, labels): left out of the cache key
NOT_AIRFLOW = {"cooling", "idle_w", "t_max_c", "tdp_w", "r_air_full_k_per_w", "cooling_note", "noise_dba", "est",
               "sources", "aliases", "amazon_query", "note", "name", "brand", "detected_by"}


def curve_fans(build: dict) -> list[str]:
    """Slots of the case fans that follow the case-fan curve."""
    return [f["slot"] for f in build["fans"] if f.get("mode") in ("in", "out") and f.get("control", "curve") == "curve"]


def bank_key(build: dict, levels=LEVELS, steps: int = 8000, lib: dict | None = None) -> str:
    lib = lib or load_library()
    strip = lambda p: {k: v for k, v in p.items() if k not in NOT_AIRFLOW}  # noqa: E731
    fans = sorted((f["slot"], f["fan"], f["mode"], f.get("control", "curve"),
                   f.get("pct", 100) if f.get("control", "curve") == "fixed" else None)
                  for f in build["fans"] if f.get("mode") in ("in", "out"))
    cooler = find(lib, "coolers", build["cooler"])
    parts = {"case": strip(find(lib, "cases", build["case"])), "gpu": strip(find(lib, "gpus", build["gpu"])),
             "cooler": strip(cooler), "fans": [strip(find(lib, "fans", f)) for f in sorted({x[1] for x in fans} | {c["fan"] for c in cooler["fans"]})]}
    spec = {"v": BANK_VERSION, "fans": fans, "levels": list(levels) if curve_fans(build) else None, "steps": steps, "parts": parts}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def geometry(m: dict) -> dict:
    """What the 3D view draws: case, parts, mounts and fans (the same for every level)."""
    D, W, H = m["dims"]
    keep = ("slot", "fan", "mode", "control", "pct", "wall", "a", "b", "size", "name", "filtered")
    return {"shape": list(m["shape"]), "off": m["grid"].off, "dx_mm": DX, "case": {"D": D, "W": W, "H": H},
            "case_name": m["case"]["name"], "mounts": m["case"]["mounts"], "panels": m["case"].get("panels", {}),
            "boxes": m["boxes"], "checks": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in m["checks"].items()},
            "fans": [{k: f.get(k) for k in keep} for f in m["fans"]]}


def compute(build: dict, levels=LEVELS, steps: int = 8000, callback=None, with_dust: bool = True) -> dict:
    """Simulate the build at every level. Returns {"summary": JSON-able, "fields": [per level numpy arrays]}."""
    lib = load_library()
    t0 = time.time()
    cb = callback or (lambda f: None)
    lv = list(levels) if curve_fans(build) else [None]      # no fan on the curve: one level is all there is
    models = [build_model(build, lib, case_duty=d) for d in lv]
    m0 = models[0]
    fu = torch.from_numpy(np.stack([m["fu"] for m in models]))
    u_lat = run_airflow_batch(m0, fu, steps, callback=lambda f: cb(0.55 * f))
    cb(0.56)
    inside = ~(m0["solid"] | m0["reservoir"])
    geo = {"shape": m0["shape"], "solid": m0["solid"], "reservoir": m0["reservoir"], "heat": m0["heat"]}
    K, iters = solve_unit_fields(u_lat, geo, [m0["intake"]["cpu"], m0["intake"]["gpu"], inside],
                                 dx_mm=DX, u_lat_per_ms=U_LAT_PER_MS)
    cb(0.8)
    dev = K.device
    ci, gi, ins = (torch.from_numpy(a).to(dev) for a in (m0["intake"]["cpu"], m0["intake"]["gpu"], inside))
    out_levels, fields = [], []
    for k, (d, m) in enumerate(zip(lv, models)):
        kc, kg = K[k, 0], K[k, 1]
        u_ms = u_lat[k].float() / U_LAT_PER_MS
        dust = simulate_dust(u_ms, m) if with_dust else None
        speed = torch.sqrt((u_ms ** 2).sum(0))
        q_in = sum(f["q_m3s"] for f in m["fans"] if f["mode"] == "in")
        q_out = sum(f["q_m3s"] for f in m["fans"] if f["mode"] == "out")
        out_levels.append({
            "duty": d,
            "k_cpu_in": [float(kc[ci].mean()), float(kg[ci].mean())],
            "k_gpu_in": [float(kc[gi].mean()), float(kg[gi].mean())],
            "k_air_mean": [float(kc[ins].mean()), float(kg[ins].mean())],
            "k_air_max": [float(kc[ins].max()), float(kg[ins].max())],
            "flow_in_m3s": q_in, "flow_out_m3s": q_out, "case_pa": m["case_pa"],
            "pressure": "positive" if q_in > q_out * 1.05 else ("negative" if q_out > q_in * 1.05 else "balanced"),
            "dead_air_fraction": float((speed[ins] < 0.05).float().mean()),
            "fans": [{k2: f[k2] for k2 in ("slot", "q_m3s", "op_frac", "duty", "speed", "rpm")} for f in m["fans"]],
            "dust": None if not dust else {k2: v for k2, v in dust.items() if k2 != "dep"} | {"by_part": dust_by_part(dust["dep"], m)},
        })
        fields.append({"u": u_ms.half().cpu().numpy(), "kc": kc.half().cpu().numpy(), "kg": kg.half().cpu().numpy(),
                       "dust_points": np.asarray(deposit_points(dust["dep"], m) if dust else [], np.float32).reshape(-1, 3)})
        cb(0.8 + 0.19 * (k + 1) / len(lv))
    del u_lat, K
    torch.cuda.empty_cache()
    summary = {"version": BANK_VERSION, "key": bank_key(build, levels, steps, lib), "levels": out_levels,
               "curve_fans": curve_fans(build), "lbm_steps": steps, "thermal_iters": iters,
               "seconds": time.time() - t0, "geometry": geometry(m0)}
    return {"summary": summary, "fields": fields}


# ----------------------------------------------------------------------------- cache
def save(bank: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = bank["summary"]["key"]
    arrays = {}
    for i, f in enumerate(bank["fields"]):
        arrays.update({f"u{i}": f["u"], f"kc{i}": f["kc"], f"kg{i}": f["kg"], f"dp{i}": f["dust_points"]})
    np.savez_compressed(CACHE / f"{key}.npz", **arrays)
    (CACHE / f"{key}.json").write_text(json.dumps(bank["summary"]))       # written last: marks the entry complete


def load(key: str) -> dict | None:
    js, nz = CACHE / f"{key}.json", CACHE / f"{key}.npz"
    if not (js.exists() and nz.exists()):
        return None
    summary = json.loads(js.read_text())
    with np.load(nz) as z:
        fields = [{"u": z[f"u{i}"], "kc": z[f"kc{i}"], "kg": z[f"kg{i}"], "dust_points": z[f"dp{i}"]}
                  for i in range(len(summary["levels"]))]
    return {"summary": summary, "fields": fields}


def get(build: dict, callback=None, levels=LEVELS, steps: int = 8000, with_dust: bool = True) -> dict:
    """The build's bank: from the cache if it was computed before, otherwise computed (about 2 min) and cached."""
    b = load(bank_key(build, levels, steps))
    if b is None:
        b = compute(build, levels, steps, callback, with_dust)
        save(b)
    return b


def package(bank: dict) -> dict:
    """JSON for the web viewer: the summary plus the fields as base64 float16 per level."""
    def b64(a):
        return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
    return {**bank["summary"], "fields": [{"u": b64(f["u"]), "kc": b64(f["kc"]), "kg": b64(f["kg"]),
                                           "dust_points": np.round(f["dust_points"], 1).tolist()} for f in bank["fields"]]}
