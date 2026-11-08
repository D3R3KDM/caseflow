"""How much do the conclusions depend on the uncertain parameters?

For a few key layouts, re-run the simulation with one parameter changed at a time and print CPU / GPU
temperature and the score CPU + GPU + 2 x dust. The ranking between the layouts is what matters.

    python scripts/sensitivity.py            # all variants (about 25 min)
    python scripts/sensitivity.py kappa      # only the named variants
    python scripts/sensitivity.py fans       # case-fan speed: the board sets it, and we cannot read it
    python scripts/sensitivity.py airflow    # the GPU's and CPU cooler's own fans
"""
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine.sim as sim  # noqa: E402
from engine import thermal  # noqa: E402
from engine.optimize import STOCK, build_for, candidates  # noqa: E402

BOTTOM_REAR_OUT = copy.deepcopy(sim.default_build())
for f in BOTTOM_REAR_OUT["fans"]:
    if f["slot"] == "bottom_rear":
        f["mode"] = "out"
LAYOUTS = {
    "yours": sim.default_build(),
    "all three on top out": build_for(candidates()["top-front + top-mid + top-rear out"]),
    "#2 all out, top-front": build_for(candidates()["top-front + top-rear + rear out"]),
    "top-front+rear in": build_for(candidates()["top-front + top-rear in · rear out"]),
    "all in": build_for(candidates()["top-mid + top-rear + rear in"]),
    "rear in (worst)": build_for(candidates()["rear in · top-front + top-mid out"]),
    "bottom-rear out": BOTTOM_REAR_OUT,
}


def with_lib(edit):
    lib = sim.load_library()
    edit(lib)
    return lambda: copy.deepcopy(lib)


def case(lib):
    return next(c for c in lib["cases"] if c["id"] == "lianli-lancool-207")


def gpu_up(lib):
    case(lib)["board"]["gpu_slot_top_z"] += 40


def front_filter(lib):
    case(lib).setdefault("filters", {})["front"] = True


STOCK_SLOTS = {f["slot"] for f in STOCK}
KEY = ["yours", "all three on top out", "#2 all out, top-front", "all in", "rear in (worst)"]
VARIANTS = {
    "baseline": {},
    "kappa 0.01": {"kappa": 0.01},
    "kappa 0.08": {"kappa": 0.08},
    "derate 0.5": {"DERATE": 0.5},
    "derate 0.7": {"DERATE": 0.7},
    "porosity x0.5": {"POROSITY": {"open": 0.3, "vent": 0.2}},
    "porosity x1.5": {"POROSITY": {"open": 0.9, "vent": 0.6}},
    "GPU 40 mm higher": {"lib": gpu_up},
    "front filter": {"lib": front_filter},
    "grid 4 mm": {"DX": 4.0, "steps": 12000, "kappa": 0.06},   # finer grid; same physical time and diffusivity
    # load scenarios (CPU W, GPU W): typical gaming (TechPowerUp: 9700X ~70 W in games), CPU-only stress, everything maxed
    "load gaming 70+300": {"load": {"cpu_w": 70.0, "gpu_w": 300.0}},
    "load CPU stress 88+60": {"load": {"cpu_w": 88.0, "gpu_w": 60.0}},
    "load full 88+360": {"load": {"cpu_w": 88.0, "gpu_w": 360.0}},
    # case-fan speed (% of max rpm; Q ~ rpm, p ~ rpm^2). Default: stock 70 %, spare 80 %. The real speed follows the
    # board's fan curve, which Windows cannot read without a sensor driver, so test slow, fast and lopsided.
    "fans all 50%": {"pct": {"stock": 50, "spare": 50}, "layouts": KEY},
    "fans all 100%": {"pct": {"stock": 100, "spare": 100}, "layouts": KEY},
    "fans stock 100% / spare 50%": {"pct": {"stock": 100, "spare": 50}, "layouts": KEY},
    # the GPU's and the CPU cooler's own fans are fixed in the model (1.6 m/s, 70 % of rated flow); both scaled together
    "GPU+cooler airflow x0.6": {"GPU_FAN_MS": 1.6 * 0.6, "COOLER_FAN_FRAC": 0.7 * 0.6, "layouts": KEY},
    "GPU+cooler airflow x1.4": {"GPU_FAN_MS": 1.6 * 1.4, "COOLER_FAN_FRAC": 0.7 * 1.4, "layouts": KEY},
}


def run(variant: dict):
    saved = {k: getattr(sim, k) for k in ("DERATE", "POROSITY", "load_library", "DX", "GPU_FAN_MS", "COOLER_FAN_FRAC")}
    saved_kappa = thermal.KAPPA
    try:
        for k in ("DERATE", "POROSITY", "DX", "GPU_FAN_MS", "COOLER_FAN_FRAC"):
            if k in variant:
                setattr(sim, k, variant[k])
        if "lib" in variant:
            sim.load_library = with_lib(variant["lib"])
        if "kappa" in variant:
            thermal.KAPPA = variant["kappa"]
        rows = {}
        for name, b in LAYOUTS.items():
            if "layouts" in variant and name not in variant["layouts"]:
                continue
            bb = copy.deepcopy(b)
            if "load" in variant:
                bb["load"] = {**bb.get("load", {}), **variant["load"]}
            if "pct" in variant:
                for f in bb["fans"]:
                    f["pct"] = variant["pct"]["stock" if f["slot"] in STOCK_SLOTS else "spare"]
            r = sim.simulate_build(bb, steps=variant.get("steps", 8000), with_dust=True)
            p = r["parts"]
            score = p["cpu"]["temp_c"] + p["gpu"]["temp_c"] + 2 * r["dust"]["dust_entering_rel"]
            rows[name] = (p["cpu"]["temp_c"], p["gpu"]["temp_c"], r["dust"]["dust_entering_rel"], score)
        return rows
    finally:
        for k, v in saved.items():
            setattr(sim, k, v)
        thermal.KAPPA = saved_kappa


if __name__ == "__main__":
    pick = [a for a in sys.argv[1:] if not a.startswith("--only=")]
    only = next((a[7:].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    if only:                                   # e.g. --only=yours,all three on top out
        LAYOUTS = {k: v for k, v in LAYOUTS.items() if k in only}
    names = [v for v in VARIANTS if not pick or any(p in v for p in pick)]
    for v in names:
        t0 = time.time()
        rows = run(VARIANTS[v])
        order = sorted(rows, key=lambda k: rows[k][3])
        print(f"\n== {v}  ({time.time() - t0:.0f} s)   order: " + " < ".join(order), flush=True)
        for k in (k for k in LAYOUTS if k in rows):
            c, g, d, s = rows[k]
            print(f"  {k:24s} CPU {c:5.1f}  GPU {g:5.1f}  dust {d * 100:3.0f}%  score {s:6.2f}", flush=True)
