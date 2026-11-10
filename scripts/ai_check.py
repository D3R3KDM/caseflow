"""Check the network's search with the simulation.

The network ranks all 6561 layouts of the 8 mounts for a scenario (web/public/data/ai_space.json). This script takes
what it picks - its top layouts overall and its best layout for each number of fans - simulates each one (a bank,
~90 s on the GPU unless cached) and compares the simulated temperatures with the predicted ones and with your
layout. Writes data/surrogate/ai_check.json.

    python scripts/ai_check.py                       # CPU-heavy game, 23 degC room, top 3 + best for 4..8 fans
    python scripts/ai_check.py --scenario stress --room 35 --top 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import default_build, load_library  # noqa: E402
from engine.surrogate import build_from_code, code_of  # noqa: E402

SPACE = ROOT / "web" / "public" / "data" / "ai_space.json"
OUT = ROOT / "data" / "surrogate" / "ai_check.json"


def row_bank(space: dict, row: dict) -> dict:
    slots = [s for s, c in zip(space["mounts"], row["c"]) if c != "0"]
    return {"levels": [{"duty": space["levels"][j] if slots else None, "k_cpu_in": v[0:2], "k_gpu_in": v[2:4], "k_air_mean": v[4:6],
                        "k_air_max": v[6:8], "flow_in_m3s": v[8], "flow_out_m3s": v[9], "case_pa": v[10],
                        "fans": [{"slot": s, "q_m3s": q} for s, q in zip(slots, v[11])], "dust": None} for j, v in enumerate(row["L"])]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cpu_game")
    ap.add_argument("--room", type=float, default=23.0)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--fans", type=int, nargs="*", default=[4, 5, 6, 7, 8], help="also check the best layout with this many fans")
    args = ap.parse_args()
    lib = load_library()
    space = json.loads(SPACE.read_text())
    pr = next(p for p in lib["defaults"]["scenarios"] if p["id"] == args.scenario)
    ds = json.loads((ROOT / "data" / "surrogate" / "dataset.json").read_text())
    split = {r["key"]: r["split"] for r in ds["rows"]}
    t0 = time.time()
    rows = []
    for row in space["rows"]:
        b = build_from_code(tuple(int(c) for c in row["c"]))
        su = sc.setup_for(b, lib)
        pc, pg = sc.load_watts(su, pr["cpu"], pr["gpu"])
        r = sc.solve(row_bank(space, row), su, args.room, pc, pg)
        throttled = r["cpu"]["throttled"] or r["gpu"]["throttled"]
        rows.append({"code": row["c"], "fans": sum(c != "0" for c in row["c"]), "cpu_ai": r["cpu"]["temp_c"], "gpu_ai": r["gpu"]["temp_c"],
                     "score_ai": r["cpu"]["temp_c"] + r["gpu"]["temp_c"] + (1000 if throttled else 0)})
    rows.sort(key=lambda x: x["score_ai"])
    for i, x in enumerate(rows):
        x["rank_ai"] = i + 1
    print(f"ranked {len(rows)} predicted layouts in {time.time() - t0:.0f} s", flush=True)
    home = "".join(map(str, code_of(default_build())))
    picks = rows[:args.top] + [next(x for x in rows if x["fans"] == n) for n in args.fans] + [next(x for x in rows if x["code"] == home)]
    seen, out = set(), []
    for x in picks:
        if x["code"] in seen:
            continue
        seen.add(x["code"])
        b = build_from_code(tuple(int(c) for c in x["code"]))
        key = bk.bank_key(b)
        cached = bk.load(key) is not None
        t1 = time.time()
        B = bk.get(b)
        su = sc.setup_for(b, lib)
        pc, pg = sc.load_watts(su, pr["cpu"], pr["gpu"])
        r = sc.solve(B["summary"], su, args.room, pc, pg)
        x.update({"cpu_sim": r["cpu"]["temp_c"], "gpu_sim": r["gpu"]["temp_c"], "score_sim": r["cpu"]["temp_c"] + r["gpu"]["temp_c"],
                  "yours": x["code"] == home, "in_training": split.get(key) == "train", "held_out": split.get(key) == "test",
                  "lbm_limiter": bool(B["summary"].get("lbm_limiter_ms"))})
        out.append(x)
        print(f"{x['code']} AI #{x['rank_ai']:4d} ({x['fans']} fans): CPU {x['cpu_ai']:.1f} -> sim {x['cpu_sim']:.1f}, GPU {x['gpu_ai']:.1f} -> sim "
              f"{x['gpu_sim']:.1f}{'  [yours]' if x['yours'] else ''}{'  [trained on]' if x['in_training'] else ''}"
              f"  ({'cached' if cached else f'{time.time() - t1:.0f} s'})", flush=True)
    res = {"scenario": args.scenario, "room_c": args.room, "n_layouts": len(rows), "picks": out,
           "note": "picks = the network's top layouts for the scenario, its best layout for each fan count, and yours; *_sim from the simulation"}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"AI CHECK DONE: {len(out)} layouts, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
