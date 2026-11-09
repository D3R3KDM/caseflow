"""Print the steady state of a build for every load preset at a few room temperatures.

The build's bank (airflow at 3 case-fan duties) is computed on the GPU the first time (about 2 min) and cached in
data/banks; after that the table takes milliseconds per scenario.

    python scripts/scenario_table.py                  # the reference PC (engine/sim.py default_build)
    python scripts/scenario_table.py --rooms 18 23 30 35
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import default_build, load_library  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=float, nargs="+", default=[18, 23, 28, 35])
    args = ap.parse_args()
    lib = load_library()
    build = default_build()
    t0 = time.time()
    b = bk.get(build, callback=lambda f: print(f"\r  bank {f * 100:5.1f} %", end="", flush=True))
    s = b["summary"]
    print(f"\rbank {s['key']}: {len(s['levels'])} levels, {s['seconds']:.0f} s to compute "
          f"(loaded in {time.time() - t0:.1f} s), heat iterations {s['thermal_iters']}")
    for L in s["levels"]:
        print(f"  duty {L['duty']:.2f}: in {L['flow_in_m3s'] * 2119:5.1f} CFM, out {L['flow_out_m3s'] * 2119:5.1f} CFM, "
              f"{L['case_pa']:+.1f} Pa | K/W at CPU intake {L['k_cpu_in'][0]:.4f} (CPU) {L['k_cpu_in'][1]:.4f} (GPU), "
              f"at GPU intake {L['k_gpu_in'][0]:.4f} {L['k_gpu_in'][1]:.4f}")
    su = sc.setup_for(build, lib)
    print(f"\n{'load':16s} {'room':>4s} | {'CPU W':>5s} {'°C':>5s} {'fan':>4s} | {'GPU W':>5s} {'°C':>5s} {'fan':>4s} | {'case':>4s} {'CFM in':>6s} {'dBA':>4s}")
    for p in lib["defaults"]["scenarios"]:
        pc, pg = sc.load_watts(su, p["cpu"], p["gpu"])
        for room in args.rooms:
            r = sc.solve(s, su, room, pc, pg)
            c, g = r["cpu"], r["gpu"]
            flag = " throttled" if c["throttled"] or g["throttled"] else ""
            print(f"{p['name']:16s} {room:4.0f} | {c['watts']:5.0f} {c['temp_c']:5.1f} {c['fan_duty'] * 100:3.0f}% | "
                  f"{g['watts']:5.0f} {g['temp_c']:5.1f} {g['fan_duty'] * 100:3.0f}% | {r['case_duty'] * 100:3.0f}% "
                  f"{r['flow_in_m3s'] * 2119:6.1f} {r['noise_dba']:4.1f}{flag}")


if __name__ == "__main__":
    main()
