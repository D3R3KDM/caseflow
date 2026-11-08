"""Physical sanity checks of a simulation:
  - each case fan: flow it really delivers (flux one cell inside its forced layer) vs the flow it was set to
  - mass balance through the case shell (in = out)
  - case pressure from the LBM vs the fan operating-point model
  - energy balance: heat carried out through the openings vs CPU + GPU power; exhaust temperature

    python scripts/sanity_check.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import thermal  # noqa: E402
from engine.optimize import build_for, candidates  # noqa: E402
from engine.sim import DX, RHO_AIR, U_LAT_PER_MS, _wall_box, build_model, default_build, load_library, run_airflow  # noqa: E402

M3S_TO_CFM = 2118.88
LAYOUTS = {
    "yours": default_build(),
    "all in": build_for(candidates()["top-mid + top-rear + rear in"]),
    "worst": build_for(candidates()["rear in · top-front + top-mid out"]),
}


def check(name, b, lib, steps=8000):
    m = build_model(b, lib)
    u, p = run_airflow(m, steps)
    load = {"cpu_w": 88.0, "gpu_w": 300.0}
    mm = {"shape": m["shape"], "solid": m["solid"], "reservoir": m["reservoir"], "heat": m["heat"]}
    inside = ~(m["solid"] | m["reservoir"])
    T, iters = thermal.solve_temperature_converged(u, mm, load, probes=[m["intake"]["cpu"], m["intake"]["gpu"], inside],
                                                    dx_mm=DX, u_lat_per_ms=U_LAT_PER_MS)
    un = u.cpu().numpy() / U_LAT_PER_MS                      # m/s
    Tn = T.cpu().numpy()
    pn = p.cpu().numpy() * RHO_AIR / U_LAT_PER_MS ** 2        # Pa = p_lat * rho * (dx/dt)^2
    A = (DX / 1000.0) ** 2
    X, Y, Z = m["shape"]
    D, W, H = m["dims"]
    print(f"\n== {name}  ({iters} heat iterations)")
    # 1. what each fan really delivers
    tot_set = tot_got = 0.0
    for f in m["fans"]:
        bx, n_in = _wall_box(f["wall"], f["a"], f["b"], f["size"], D, W, H)
        i0, i1, j0, j1, k0, k1 = m["grid"].box(np.zeros(m["shape"], bool), *bx)
        ax = int(np.argmax(np.abs(n_in)))
        sl = [slice(i0, i1), slice(j0, j1), slice(k0, k1)]
        lo, hi = (i0, i1) if ax == 0 else (k0, k1)
        inner = hi if n_in[ax] > 0 else lo - 1
        sl[ax] = slice(inner, inner + 1)
        got = float((un[ax][tuple(sl)] * n_in[ax]).sum()) * A * (1 if f["mode"] == "in" else -1)
        tot_set += f["q_m3s"]
        tot_got += got
        print(f"  fan {f['slot']:12s} {f['mode']:3s}  set {f['q_m3s'] * M3S_TO_CFM:5.1f}  delivered {got * M3S_TO_CFM:5.1f} CFM ({got / f['q_m3s'] * 100:3.0f}%)")
    # 2. + 4. fluxes through the faces between the room layer and the case (face velocity, upwind temperature)
    q_out = q_in = heat_out = 0.0
    for dim, n in ((0, X), (1, Y), (2, Z)):
        for res_i, in_i, sgn in ((0, 1, -1), (n - 1, n - 2, 1)):
            sr = [slice(None)] * 3
            si = [slice(None)] * 3
            sr[dim], si[dim] = res_i, in_i
            open_ = ~m["solid"][tuple(si)]
            uf = 0.5 * (un[dim][tuple(sr)] + un[dim][tuple(si)]) * sgn * open_     # outward m/s
            out = np.clip(uf, 0, None)
            q_out += float(out.sum()) * A
            q_in += float(np.clip(-uf, 0, None).sum()) * A
            heat_out += float((out * Tn[tuple(si)]).sum()) * A * thermal.RHO_CP
    P = sum(load.values())
    p_case = float(np.median(pn[inside & ~m["forced"]]))
    print(f"  fans delivered {tot_got / tot_set * 100:.0f}% of their set flow in total")
    print(f"  shell: in {q_in * M3S_TO_CFM:6.1f} CFM, out {q_out * M3S_TO_CFM:6.1f} CFM   -> out/in = {q_out / max(q_in, 1e-12):.3f}")
    print(f"  case pressure: LBM median {p_case:+.2f} Pa   vs fan model {m['case_pa']:+.2f} Pa")
    print(f"  energy: carried out {heat_out:6.1f} W  vs {P:.0f} W in  ({heat_out / P * 100:.1f}%)")
    print(f"  exhaust +{heat_out / max(q_out * thermal.RHO_CP, 1e-9):.2f} K (flow-weighted)   bulk P/(rho cp Q) = +{P / (thermal.RHO_CP * max(q_out, 1e-9)):.2f} K")
    print(f"  air inside: mean +{float(Tn[inside].mean()):.2f} K, max +{float(Tn[inside].max()):.1f} K")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    lib = load_library()
    for name, b in LAYOUTS.items():
        check(name, b, lib)
