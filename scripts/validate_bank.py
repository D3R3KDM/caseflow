"""How much error does the instant scenario model add to the physics model?

The instant model (engine/bank.py + engine/scenario.py) makes two shortcuts:
1. case-fan airflow is interpolated between three precomputed duties;
2. the CPU cooler's and the GPU's own fans stir the case air at one representative speed (their real speed only
   changes their cooler's conductance).

The direct check removes both. Starting from the instant answer, it simulates the build in full with every fan at
exactly its speed - case fans at the solved duty, cooler fans and GPU fans at their solved speeds - computes the
temperatures, takes the fan speeds the curves give for them, and repeats until the speeds settle (change < 2 %).
The difference between the two answers is the error of the shortcuts. Results go to data/validation.json.

    python scripts/validate_bank.py          # ~10 min on an RTX 5080
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import (DX, GPU_FAN_MS, GPU_FAN_REF_SPEED, U_LAT_PER_MS, build_model, default_build,  # noqa: E402
                        load_library, run_airflow)
from engine.thermal import solve_temperature_converged  # noqa: E402

CASES = [("office", 23), ("cpu_game", 23), ("gpu_game", 23), ("stress", 35)]


def temps_in_airflow(su, amb, pc, pg, rise_c, rise_g):
    """Part temperatures for fixed intake rises, with each cooler's own fans following their curve."""
    cv = su["curves"]

    def cpu_t(t):
        s = sc.fan_speed(sc.curve_duty(cv["cpu"]["points"], t), su["cooler"]["s_min"])
        return amb + rise_c + pc / sc.conductance(su["cpu"]["r_base"], su["cooler"]["r_air_full"], s, sc.CPU_FAN_FLOOR) - t

    def gpu_t(t):
        s = sc.gpu_fan_speed(sc.curve_duty(cv["gpu"]["points"], t))
        return amb + rise_g + pg / sc.conductance(su["gpu"]["r_base"], su["gpu"]["r_air_full"], s, su["gpu"]["floor"]) - t
    return sc._bisect(cpu_t, amb, amb + 400), sc._bisect(gpu_t, amb, amb + 400)


def direct(build, lib, su, amb, pc, pg, start, steps=8000, max_runs=5, tol=0.02):
    d_case, s_cpu, s_gpu = start
    cv = su["curves"]
    runs = []
    for _ in range(max_runs):
        m = build_model(build, lib, case_duty=d_case, cooler_speed=s_cpu,
                        gpu_ms=GPU_FAN_MS * max(s_gpu, 0.05) / GPU_FAN_REF_SPEED)
        u, _ = run_airflow(m, steps)
        inside = ~(m["solid"] | m["reservoir"])
        geo = {"shape": m["shape"], "solid": m["solid"], "reservoir": m["reservoir"], "heat": m["heat"]}
        T, _ = solve_temperature_converged(u, geo, {"cpu_w": pc, "gpu_w": pg},
                                           probes=[m["intake"]["cpu"], m["intake"]["gpu"], inside], dx_mm=DX, u_lat_per_ms=U_LAT_PER_MS)
        dev = T.device
        rc = float(T[torch.from_numpy(m["intake"]["cpu"]).to(dev)].mean())
        rg = float(T[torch.from_numpy(m["intake"]["gpu"]).to(dev)].mean())
        tc, tg = temps_in_airflow(su, amb, pc, pg, rc, rg)
        src = cv["case"].get("source", "cpu")
        t_src = tc if src == "cpu" else tg if src == "gpu" else max(tc, tg)
        nxt = (sc.curve_duty(cv["case"]["points"], t_src),
               sc.fan_speed(sc.curve_duty(cv["cpu"]["points"], tc), su["cooler"]["s_min"]),
               sc.gpu_fan_speed(sc.curve_duty(cv["gpu"]["points"], tg)))
        runs.append({"case_duty": d_case, "cpu_fan_speed": s_cpu, "gpu_fan_speed": s_gpu, "cpu_c": tc, "gpu_c": tg,
                     "cpu_rise": rc, "gpu_rise": rg})
        print(f"    run {len(runs)}: case {d_case * 100:4.1f}%  cpu fan {s_cpu * 100:4.1f}%  gpu fan {s_gpu * 100:4.1f}%"
              f"  ->  CPU {tc:5.2f}  GPU {tg:5.2f}", flush=True)
        if max(abs(a - b) for a, b in zip(nxt, (d_case, s_cpu, s_gpu))) < tol:
            break
        d_case, s_cpu, s_gpu = (0.5 * (a + b) for a, b in zip(nxt, (d_case, s_cpu, s_gpu)))   # damped
    return runs


def main() -> None:
    lib = load_library()
    build = default_build()
    t0 = time.time()
    bank = bk.get(build)["summary"]
    su = sc.setup_for(build, lib)
    presets = {p["id"]: p for p in lib["defaults"]["scenarios"]}
    rows = []
    for pid, amb in CASES:
        p = presets[pid]
        pc, pg = sc.load_watts(su, p["cpu"], p["gpu"])
        inst = sc.solve(bank, su, amb, pc, pg)
        print(f"{p['name']} at {amb} degC ({pc:.0f} W / {pg:.0f} W): instant CPU {inst['cpu']['temp_c']:.2f}  "
              f"GPU {inst['gpu']['temp_c']:.2f}  case {inst['case_duty'] * 100:.1f}%", flush=True)
        runs = direct(build, lib, su, amb, pc, pg, (inst["case_duty"], inst["cpu"]["fan_speed"], inst["gpu"]["fan_speed"]))
        last = runs[-1]
        rows.append({"scenario": pid, "name": p["name"], "ambient_c": amb, "cpu_w": pc, "gpu_w": pg,
                     "instant": {"cpu_c": inst["cpu"]["temp_c"], "gpu_c": inst["gpu"]["temp_c"], "case_duty": inst["case_duty"],
                                 "cpu_fan_speed": inst["cpu"]["fan_speed"], "gpu_fan_speed": inst["gpu"]["fan_speed"]},
                     "direct": last, "runs": len(runs),
                     "error_cpu_c": inst["cpu"]["temp_c"] - last["cpu_c"], "error_gpu_c": inst["gpu"]["temp_c"] - last["gpu_c"]})
        print(f"  error (instant - direct): CPU {rows[-1]['error_cpu_c']:+.2f}  GPU {rows[-1]['error_gpu_c']:+.2f}", flush=True)
    out = {"build": "reference PC (engine/sim.py default_build)", "bank_key": bank["key"], "minutes": (time.time() - t0) / 60,
           "rows": rows, "max_abs_error_c": max(max(abs(r["error_cpu_c"]), abs(r["error_gpu_c"])) for r in rows)}
    (ROOT / "data" / "validation.json").write_text(json.dumps(out, indent=1))
    print(f"VALIDATION DONE: largest error {out['max_abs_error_c']:.2f} degC, {out['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
