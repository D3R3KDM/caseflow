"""Find the best layout for the user's own fans with the bank model.

Inventory (user's PC, 2026-11-07, corrected): the LANCOOL 207's four stock fans - 2 x 140 mm front intake and
2 x 120 mm reverse-blade bottom intake (under the GPU, behind the PSU) - plus the user's three 120 mm fans, which
they have on top x2 + rear x1, all exhaust. The stock fans stay where Lian Li put them (the bottom fans are
reverse-blade, the front ones are the 140 mm pair), so the free choice is where the three 120 mm fans go and
which way each one blows. There are four free 120 mm mounts (top-front, top-mid, top-rear, rear), so the whole
space is C(4,3) x 2^3 = 32 layouts. Every case fan follows the board's case-fan curve.

Every layout gets a bank (engine/bank.py: airflow, heat and dust at three case-fan duties; about 1.5 min on the GPU,
cached in data/banks, so a restart resumes where it stopped). Their summaries go to data/optimize.json. With them
the 32 layouts are ranked for any scenario - room temperature, CPU / GPU load, fan curves - in milliseconds: below
for a set of reference scenarios, and in the web app for whatever the user sets.

Score of a layout in a scenario = CPU degC + GPU degC + DUST_WEIGHT x its dust settling rate relative to the user's
layout. Layouts within TIE of the top score count as a tie: no change is recommended for less.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import default_build, load_library  # noqa: E402

OUT = ROOT / "data"
P = OUT / "optimize.json"
VERSION = 7
STOCK = [{"slot": "front_low", "fan": "lianli-207-front-140", "mode": "in", "control": "curve", "pct": 70},
         {"slot": "front_high", "fan": "lianli-207-front-140", "mode": "in", "control": "curve", "pct": 70},
         {"slot": "bottom_front", "fan": "lianli-207-bottom-120-rev", "mode": "in", "control": "curve", "pct": 70},
         {"slot": "bottom_rear", "fan": "lianli-207-bottom-120-rev", "mode": "in", "control": "curve", "pct": 70}]
FREE = ["top_1", "top_2", "top_3", "rear"]
LABEL = {"top_1": "top-front", "top_2": "top-mid", "top_3": "top-rear", "rear": "rear"}
DUST_WEIGHT = 2.0   # degC of CPU+GPU temperature traded for dust settling 100 % faster than in your layout (judgement call)
TIE = 0.5           # degC: layouts closer than this to the top score are tied within the model's accuracy
# reference scenarios for the stored rankings: (load preset, room degC)
SCENARIOS = [("cpu_game", 23), ("office", 23), ("gpu_game", 23), ("stress", 23), ("gpu_game", 30), ("stress", 35)]


def fans120(spec: list[tuple[str, str]]) -> list[dict]:
    return [{"slot": s, "fan": "generic-120", "mode": m, "control": "curve", "pct": 80} for s, m in spec]


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
    b = default_build()
    b["fans"] = STOCK + fans120(spec)
    return b


def is_current(b) -> bool:
    key = lambda fs: sorted((f["slot"], f["fan"], f["mode"]) for f in fs)  # noqa: E731
    return key(b["fans"]) == key(default_build()["fans"])


def rank(layouts: list[dict], lib: dict, ambient_c: float, cpu_frac: float, gpu_frac: float, curves: dict | None = None) -> list[dict]:
    """Every layout's steady state in one scenario, best score first (web/src/App.tsx ranks the same way)."""
    rows = []
    for L in layouts:
        b = dict(L["build"], **({"curves": curves} if curves else {}))
        su = sc.setup_for(b, lib)
        pc, pg = sc.load_watts(su, cpu_frac, gpu_frac)
        r = sc.solve(L["bank"], su, ambient_c, pc, pg)
        rows.append({"name": L["name"], "current": L["current"], "cpu_c": r["cpu"]["temp_c"], "gpu_c": r["gpu"]["temp_c"],
                     "throttled": r["cpu"]["throttled"] or r["gpu"]["throttled"], "noise_dba": r["noise_dba"],
                     "case_duty": r["case_duty"], "cfm_in": r["flow_in_m3s"] * 2119, "dust_rate": r["dust_rate"]})
    ref = next((x["dust_rate"] for x in rows if x["current"]), None) or max(x["dust_rate"] for x in rows)
    for x in rows:
        x["dust_rel"] = x["dust_rate"] / ref
        x["score"] = x["cpu_c"] + x["gpu_c"] + DUST_WEIGHT * x["dust_rel"]
    rows.sort(key=lambda x: x["score"])
    top = rows[0]["score"]
    for x in rows:
        x["tied"] = x["score"] - top < TIE
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lib = load_library()
    cands = candidates()
    t0 = time.time()
    layouts = []
    for i, (name, spec) in enumerate(cands.items()):
        b = build_for(spec)
        cached = bk.load(bk.bank_key(b)) is not None
        t1 = time.time()
        bank = bk.get(b)["summary"]
        layouts.append({"name": name, "build": b, "current": is_current(b),
                        "bank": {k: v for k, v in bank.items() if k != "geometry"}})
        su = sc.setup_for(b, lib)
        r = sc.solve(bank, su, 23, *sc.load_watts(su, 0.97, 0.38))
        print(f"[opt {i + 1:2d}/{len(cands)}] {name:44s} CPU {r['cpu']['temp_c']:.1f}  GPU {r['gpu']['temp_c']:.1f}"
              f"  case fans {r['case_duty'] * 100:.0f}%  {'cached' if cached else f'{time.time() - t1:.0f} s'}"
              f"{'  <- yours' if layouts[-1]['current'] else ''}", flush=True)
    presets = {p["id"]: p for p in lib["defaults"]["scenarios"]}
    rankings = []
    for pid, room in SCENARIOS:
        p = presets[pid]
        rows = rank(layouts, lib, room, p["cpu"], p["gpu"])
        rankings.append({"scenario": pid, "name": p["name"], "ambient_c": room, "rows": rows})
        cur = next(j for j, x in enumerate(rows) if x["current"])
        print(f"{p['name']} at {room} degC: #1 {rows[0]['name']} (CPU {rows[0]['cpu_c']:.1f}, GPU {rows[0]['gpu_c']:.1f}); "
              f"yours #{cur + 1} ({rows[cur]['score'] - rows[0]['score']:+.2f}); worst {rows[-1]['name']}", flush=True)
    res = {"version": VERSION, "layouts": layouts, "rankings": rankings, "dust_weight": DUST_WEIGHT, "tie_margin": TIE,
           "space": "4 stock fans fixed (2x140 front in, 2x120 bottom in); the user's 3x120 fans over "
                    "{top-front, top-mid, top-rear, rear} x {in, out}: 32 layouts; every case fan follows the case-fan curve",
           "objective": "score = CPU + GPU temperature + 2 x dust settling rate relative to your layout; within 0.5 of the top counts as a tie",
           "minutes": (time.time() - t0) / 60}
    P.write_text(json.dumps(res, indent=1))
    print(f"OPT DONE: {len(layouts)} layouts, {res['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
