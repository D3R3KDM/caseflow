"""Is the temperature converged? Run the airflow once per layout, then keep iterating the heat transport and
watch the CPU / GPU temperatures. Low-flow layouts (all intake) need the longest.

    python scripts/check_convergence.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.lbm import LBMConfig, LBMSolver  # noqa: E402
from engine.optimize import build_for, candidates  # noqa: E402
from engine.sim import DX, G_DEFAULT, U_LAT_PER_MS, build_model, default_build, load_library  # noqa: E402
from engine.thermal import solve_temperature  # noqa: E402

LAYOUTS = {
    "yours (#1)": default_build(),
    "all in (#25)": build_for(candidates()["top-mid + top-rear + rear in"]),
    "worst (#32)": build_for(candidates()["rear in · top-front + top-mid out"]),
}
CHECK = [int(x) for x in "5000,10000,20000,40000,80000".split(",")]
if len(sys.argv) > 2:
    CHECK = [int(x) for x in sys.argv[2].split(",")]
STEPS = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4000]


def temps(T, model, load, lib_cooler, lib_gpu):
    out = []
    for key, g in (("cpu", G_DEFAULT[lib_cooler]), ("gpu", G_DEFAULT[lib_gpu])):
        t_in = float(T[torch.from_numpy(model["intake"][key]).to(T.device)].mean())
        out.append((t_in, load["ambient_c"] + t_in + load[f"{key}_w"] / g))
    return out


lib = load_library()
for steps in STEPS:
    for name, b in LAYOUTS.items():
        t0 = time.time()
        model = build_model(b, lib)
        load = {"cpu_w": 88.0, "gpu_w": 300.0, "ambient_c": 25.0, **b.get("load", {})}
        solver = LBMSolver(torch.from_numpy(model["solid"])[None], LBMConfig(u_in=0.0, tau=0.52, smagorinsky=0.16, sponge=0),
                           open_inlet=False, forced_mask=torch.from_numpy(model["forced"]), forced_u=torch.from_numpy(model["fu"]),
                           reservoir_mask=torch.from_numpy(model["reservoir"]))
        u = solver.run(steps=steps, avg_from=steps // 2, report_every=10 ** 9, check_nan_every=1000).u[0]
        del solver
        q_in = sum(f["q_m3s"] for f in model["fans"] if f["mode"] == "in") * 2119
        print(f"\n{name}  LBM {steps} steps  in {q_in:.0f} CFM  ({time.time() - t0:.0f} s)", flush=True)
        T, done = None, 0
        mm = {"shape": model["shape"], "solid": model["solid"], "reservoir": model["reservoir"], "heat": model["heat"]}
        for it in CHECK:
            t1 = time.time()
            T = solve_temperature(u, mm, {"cpu_w": load["cpu_w"], "gpu_w": load["gpu_w"]}, iters=it - done, dx_mm=DX, u_lat_per_ms=U_LAT_PER_MS, T0=T)
            done = it
            (ci, ct), (gi, gt) = temps(T, model, load, model["cooler"]["type"], model["gpu"]["cooler"])
            inside = ~torch.from_numpy(model["solid"] | model["reservoir"]).to(T.device)
            print(f"  {it:6d} iters  CPU {ct:6.2f} (air +{ci:5.2f})  GPU {gt:6.2f} (air +{gi:5.2f})  mean air +{float(T[inside].mean()):5.2f}  [{time.time() - t1:.0f} s]", flush=True)
        torch.cuda.empty_cache()
