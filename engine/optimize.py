"""Find the best layout for the user's own fans with the physics simulator.

Inventory (user's PC, 2026-11-07, corrected): the LANCOOL 207's four stock fans - 2 x 140 mm front intake and
2 x 120 mm reverse-blade bottom intake (under the GPU, behind the PSU) - plus the user's three 120 mm fans, which
they have on top x2 + rear x1, all exhaust. The stock fans stay where Lian Li put them (the bottom fans are
reverse-blade, the front ones are the 140 mm pair), so the free choice is where the three 120 mm fans go and
which way each one blows. There are four free 120 mm mounts (top-front, top-mid, top-rear, rear), so the whole
space is C(4,3) x 2^3 = 32 layouts. Every one is simulated in full (airflow + heat + dust): an exhaustive
search, not a hand-picked shortlist. Progress is saved after each layout, so a restart resumes where it stopped.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import default_build, simulate_build  # noqa: E402

OUT = ROOT / "data"
P = OUT / "optimize.json"
STOCK = [{"slot": "front_low", "fan": "lianli-207-front-140", "mode": "in", "pct": 70},
         {"slot": "front_high", "fan": "lianli-207-front-140", "mode": "in", "pct": 70},
         {"slot": "bottom_front", "fan": "lianli-207-bottom-120-rev", "mode": "in", "pct": 70},
         {"slot": "bottom_rear", "fan": "lianli-207-bottom-120-rev", "mode": "in", "pct": 70}]
LOAD = {"cpu_w": 88.0, "gpu_w": 300.0, "ambient_c": 25.0}
FREE = ["top_1", "top_2", "top_3", "rear"]
LABEL = {"top_1": "top-front", "top_2": "top-mid", "top_3": "top-rear", "rear": "rear"}
DUST_WEIGHT = 2.0   # deg C of CPU+GPU temperature traded for letting in 100 % more of the outside dust (judgement call)


def fans120(spec: list[tuple[str, str]]) -> list[dict]:
    return [{"slot": s, "fan": "generic-120", "mode": m, "pct": 80} for s, m in spec]


def name_for(spec) -> str:
    ins = [LABEL[s] for s in FREE for t, m in spec if t == s and m == "in"]
    outs = [LABEL[s] for s in FREE for t, m in spec if t == s and m == "out"]
    parts = []
    if ins:
        parts.append(" + ".join(ins) + " in")
    if outs:
        parts.append(" + ".join(outs) + " out")
    return " · ".join(parts)


def candidates() -> dict[str, list[tuple[str, str]]]:
    out = {}
    for slots in itertools.combinations(FREE, 3):
        for modes in itertools.product(["out", "in"], repeat=3):
            spec = list(zip(slots, modes))
            out[name_for(spec)] = spec
    return out


def build_for(spec) -> dict:
    return {"case": "lianli-lancool-207", "gpu": "nvidia-rtx-5080-fe", "cooler": "thermalright-pa120-se-argb",
            "fans": STOCK + fans120(spec), "load": dict(LOAD)}


def is_current(b) -> bool:
    key = lambda fs: sorted((f["slot"], f["fan"], f["mode"]) for f in fs)  # noqa: E731
    return key(b["fans"]) == key(default_build()["fans"])


def summarize(name, b, r) -> dict:
    p = r["parts"]
    out = {"name": name, "build": b, "current": is_current(b), "gpu_c": p["gpu"]["temp_c"], "cpu_c": p["cpu"]["temp_c"],
           "gpu_air": p["gpu"]["intake_rise_c"], "cpu_air": p["cpu"]["intake_rise_c"],
           "score": p["gpu"]["temp_c"] + p["cpu"]["temp_c"], "pressure": r["pressure"],
           "cfm_in": r["flow_in_m3s"] * 2119, "cfm_out": r["flow_out_m3s"] * 2119,
           "dead_air": r["dead_air_fraction"], "seconds": r["seconds"]}
    if r.get("dust"):
        d = r["dust"]
        out.update({"dust_in": d["dust_entering_rel"], "dust_settles": d["deposited_frac"],
                    "dust_top": [(x["name"], x["share"]) for x in d["by_part"][:4]]})
    return out


def pick(rows: list[dict]) -> dict:
    for x in rows:
        x["combined"] = x["score"] + DUST_WEIGHT * x["dust_in"]
    coolest = min(rows, key=lambda x: x["score"])
    rec = min(rows, key=lambda x: x["combined"])
    clean = min(rows, key=lambda x: (round(x["dust_in"], 2), x["score"]))
    cpu = min(rows, key=lambda x: (x["cpu_c"], x["combined"]))
    worst = max(rows, key=lambda x: x["combined"])
    return {"coolest": coolest["name"], "best": rec["name"],
            # alternatives are only worth a view when they differ by more than the noise between runs
            "cleanest": clean["name"] if clean["dust_in"] <= rec["dust_in"] - 0.05 else None,
            "cpu": cpu["name"] if cpu["cpu_c"] <= rec["cpu_c"] - 0.5 else None,
            "worst": worst["name"], "dust_weight": DUST_WEIGHT}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    done = {}
    if P.exists():
        old = json.loads(P.read_text())
        if old.get("version") == 4:
            done = {x["name"]: x for x in old["ranking"]}
    t0 = time.time()
    rows = []
    for i, (name, spec) in enumerate(cands.items()):
        if name in done:
            rows.append(done[name])
            continue
        b = build_for(spec)
        r = simulate_build(b, with_dust=True)
        rows.append(summarize(name, b, r))
        x = rows[-1]
        print(f"[opt {i + 1:2d}/{len(cands)}] {name:44s} GPU {x['gpu_c']:.1f}  CPU {x['cpu_c']:.1f}  sum {x['score']:.1f}"
              f"  dust {x['dust_in'] * 100:.0f}%  {x['pressure']}{'  <- yours' if x['current'] else ''}", flush=True)
        P.write_text(json.dumps({"version": 4, "partial": True, "ranking": rows}, indent=1))
    rows.sort(key=lambda x: x["score"] + DUST_WEIGHT * x["dust_in"])
    res = {"version": 4, "ranking": rows, **pick(rows),
           "space": "4 stock fans fixed (2x140 front in, 2x120 bottom in); the user's 3x120 fans over "
                    "{top-front, top-mid, top-rear, rear} x {in, out}: 32 layouts, all simulated",
           "objective": "recommended = lowest (CPU + GPU temperature) + 2 x (fraction of outside dust that gets in)",
           "minutes": (time.time() - t0) / 60}
    P.write_text(json.dumps(res, indent=1))
    print("OPT DONE best:", res["best"], "| coolest:", res["coolest"], "| cleanest:", res["cleanest"],
          "| lowest CPU:", res["cpu"], "| worst:", res["worst"], f"{res['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
