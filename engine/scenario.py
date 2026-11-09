"""Fan curves and the steady state of a build in a scenario (room temperature, CPU power, GPU power).

Real fans follow temperature curves: a hotter part spins its fans faster, which moves more air and makes its cooler
work better, which cools the part again. This module finds the state where temperatures and fan speeds agree.

It works on a *bank*: the airflow and heat of one build, precomputed by engine/bank.py at a few case-fan duties.
For a fixed airflow the air's temperature rise is proportional to the power, so the bank stores the rise per watt
of CPU power and per watt of GPU power at each duty; any power and room temperature is a weighted sum of those, and
a scenario solves in milliseconds. Between the precomputed duties the per-watt rise is interpolated in 1 / speed
(it scales like 1 / airflow), and flows in speed.

    part temperature = room + intake air rise + P / G(fan speed)
    G = 1 / (r_base + r_air_full * speed^-0.7)      fin heat transfer grows like air speed^0.7

Simplifications, on top of those of the airflow model:
- all case fans that follow a curve share one curve (one duty); fans set to "fixed" keep their duty;
- the CPU cooler's and GPU's own fans change their cooler's conductance G, but the airflow they drive inside the
  case stays as precomputed (a fixed representative speed);
- a part that would pass its temperature limit is throttled: its power drops until it sits at the limit.

Units: °C, W, m^3/s; duties and fan speeds are fractions 0..1 (speed = rpm / rpm_max).
web/src/scenario.ts is a line-by-line port of this file for the browser; tests/test_scenario.py checks they agree.
"""
from __future__ import annotations

import math

H_EXP = 0.7               # fin heat transfer ~ air speed^0.7
S_MIN_REF = 0.2           # speed at 0 % duty that places the bank's levels on the speed axis
CPU_FAN_FLOOR = 0.05      # lowest speed in the CPU cooler's conductance (its fans never stop)
BISECT_ITERS = 50
OUTER_ITERS = 60


# ----------------------------------------------------------------------------- fans and coolers
def curve_duty(points: list, t: float) -> float:
    """Duty 0..1 at temperature t from [[t_c, duty_pct], ...]: linear between points, flat outside them."""
    pts = sorted(points)
    if t <= pts[0][0]:
        return pts[0][1] / 100.0
    for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return (d0 + f * (d1 - d0)) / 100.0
    return pts[-1][1] / 100.0


def fan_speed(duty: float, s_min: float) -> float:
    """rpm / rpm_max of a PWM case or cooler fan: most keep turning at 0 % (s_min), then rise linearly."""
    return s_min + (1.0 - s_min) * min(max(duty, 0.0), 1.0)


def gpu_fan_speed(duty: float) -> float:
    """GPU fans stop at 0 % (zero-RPM mode); their duty is their speed."""
    return min(max(duty, 0.0), 1.0)


def conductance(r_base: float, r_air_full: float, speed: float, floor: float) -> float:
    """Cooler conductance G (W/K) at a fan speed; floor stands in for the air that still crosses the fins."""
    return 1.0 / (r_base + r_air_full * max(speed, floor) ** -H_EXP)


def noise_dba(sources: list) -> float:
    """Total sound level of fans [(dBA at full speed, speed, count)]: each fan -50 dB per decade of speed, summed
    as power. A rough comparison between settings, not a sound meter reading."""
    e = sum(n * 10 ** ((lw + 50.0 * math.log10(s)) / 10.0) for lw, s, n in sources if s > 1e-3 and n > 0)
    return 10.0 * math.log10(e) if e > 0 else 0.0


# ----------------------------------------------------------------------------- bank interpolation
RISE_KEYS = ("k_cpu_in", "k_gpu_in", "k_air_mean", "k_air_max")    # K per W, interpolated in 1 / speed
FLOW_KEYS = ("flow_in_m3s", "flow_out_m3s", "case_pa")              # interpolated in speed


def _interp(xs: list, ys: list, x: float) -> float:
    """Piecewise linear through (xs, ys), xs increasing; linear extrapolation past the ends."""
    if len(xs) == 1:
        return ys[0]
    i = 0
    while i < len(xs) - 2 and x > xs[i + 1]:
        i += 1
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + f * (ys[i + 1] - ys[i])


def bank_state(bank: dict, duty: float) -> dict:
    """Per-watt rises, flows and fan flows of the bank at a case-fan duty."""
    lv = bank["levels"]
    if len(lv) == 1:
        return dict(lv[0])
    s = [fan_speed(L["duty"], S_MIN_REF) for L in lv]
    sn = fan_speed(duty, S_MIN_REF)
    inv = [1.0 / v for v in s][::-1]                       # increasing 1 / speed
    out = {"duty": duty}
    for k in RISE_KEYS:
        out[k] = [max(_interp(inv, [L[k][j] for L in lv][::-1], 1.0 / sn), 0.0) for j in range(2)]
    for k in FLOW_KEYS:
        out[k] = _interp(s, [L[k] for L in lv], sn)
    out["fans"] = [{"slot": f["slot"], "q_m3s": max(_interp(s, [L["fans"][i]["q_m3s"] for L in lv], sn), 0.0)}
                   for i, f in enumerate(lv[0]["fans"])]
    return out


def dust_rate(bank: dict, duty: float) -> float | None:
    """How fast outside dust settles inside (m^3/s of outside air whose dust stays in the case): intake flow x the
    unfiltered share x the share of what enters that settles. The two shares are interpolated linearly in duty."""
    lv = [L for L in bank["levels"] if L.get("dust")]
    if not lv:
        return None
    if len(lv) == 1:
        d = lv[0]["dust"]
        return lv[0]["flow_in_m3s"] * d["dust_entering_rel"] * d["deposited_frac"]
    xs = [L["duty"] for L in lv]
    unfiltered = _interp(xs, [L["dust"]["dust_entering_rel"] for L in lv], duty)
    settles = _interp(xs, [L["dust"]["deposited_frac"] for L in lv], duty)
    return bank_state(bank, duty)["flow_in_m3s"] * min(max(unfiltered, 0.0), 1.0) * min(max(settles, 0.0), 1.0)


# ----------------------------------------------------------------------------- setup from a build
def setup_for(build: dict, lib: dict) -> dict:
    """The numbers the solver needs from a build and the parts library (no airflow)."""
    def find(cat, pid):
        return next(p for p in lib[cat] if p["id"] == pid)
    cpu = find("cpus", build.get("cpu", lib["cpus"][0]["id"]))
    cooler = find("coolers", build["cooler"])
    cfan = find("fans", cooler["fans"][0]["fan"])
    gpu = find("gpus", build["gpu"])
    gc = gpu["cooling"]
    d = lib["defaults"]["curves"]
    cv = build.get("curves") or {}
    curves = {"case": {"source": (cv.get("case") or d["case"]).get("source", "cpu"), "points": (cv.get("case") or d["case"])["points"]},
              "cpu": {"points": (cv.get("cpu") or d["cpu"])["points"]},
              "gpu": {"points": (cv.get("gpu") or {"points": gc["fan_curve"]})["points"]}}
    fans = []
    for f in build["fans"]:
        if f.get("mode") not in ("in", "out"):
            continue
        p = find("fans", f["fan"])
        fans.append({"slot": f["slot"], "curve": f.get("control", "curve") == "curve", "pct": f.get("pct", 100),
                     "s_min": p.get("s_min", S_MIN_REF), "noise": p.get("noise_dba", 30.0), "rpm_max": p["rpm_max"]})
    return {
        "cpu": {"r_base": cpu["r_die_k_per_w"], "t_max": cpu["t_max_c"], "idle_w": cpu["idle_w"], "max_w": cpu["max_w"]},
        "cooler": {"r_air_full": cooler["r_air_full_k_per_w"], "s_min": cfan.get("s_min", S_MIN_REF),
                   "fans": sum(x.get("count", 1) for x in cooler["fans"]), "noise": cfan.get("noise_dba", 30.0), "rpm_max": cfan["rpm_max"]},
        "gpu": {"r_base": gc["r_base_k_per_w"], "r_air_full": gc["r_air_full_k_per_w"], "floor": gc["passive_speed"],
                "t_max": gpu["t_max_c"], "idle_w": gpu["idle_w"], "max_w": gpu["tdp_w"], "fans": gc["fans"],
                "noise": gc["fan_noise_dba"], "rpm_max": gc["fan_rpm_max"]},
        "case_fans": fans,
        "curves": curves,
    }


def load_watts(setup: dict, cpu_frac: float, gpu_frac: float) -> tuple[float, float]:
    """Power of a load given as fractions between idle (0) and the power limit (1)."""
    c, g = setup["cpu"], setup["gpu"]
    return c["idle_w"] + cpu_frac * (c["max_w"] - c["idle_w"]), g["idle_w"] + gpu_frac * (g["max_w"] - g["idle_w"])


# ----------------------------------------------------------------------------- the coupled solve
def _implied(bank: dict, su: dict, amb: float, pc: float, pg: float, tc: float, tg: float) -> dict:
    """Everything that follows from assumed CPU / GPU temperatures: fan duties, conductances, and the temperatures
    those would produce."""
    cv = su["curves"]
    src = cv["case"].get("source", "cpu")
    t_src = tc if src == "cpu" else tg if src == "gpu" else max(tc, tg)
    d_case = curve_duty(cv["case"]["points"], t_src)
    b = bank_state(bank, d_case)
    d_cpu = curve_duty(cv["cpu"]["points"], tc)
    s_cpu = fan_speed(d_cpu, su["cooler"]["s_min"])
    d_gpu = curve_duty(cv["gpu"]["points"], tg)
    s_gpu = gpu_fan_speed(d_gpu)
    g_cpu = conductance(su["cpu"]["r_base"], su["cooler"]["r_air_full"], s_cpu, CPU_FAN_FLOOR)
    g_gpu = conductance(su["gpu"]["r_base"], su["gpu"]["r_air_full"], s_gpu, su["gpu"]["floor"])
    rise_c = b["k_cpu_in"][0] * pc + b["k_cpu_in"][1] * pg
    rise_g = b["k_gpu_in"][0] * pc + b["k_gpu_in"][1] * pg
    return {"tc": amb + rise_c + pc / g_cpu, "tg": amb + rise_g + pg / g_gpu, "b": b, "d_case": d_case,
            "d_cpu": d_cpu, "s_cpu": s_cpu, "d_gpu": d_gpu, "s_gpu": s_gpu, "g_cpu": g_cpu, "g_gpu": g_gpu,
            "rise_c": rise_c, "rise_g": rise_g}


def _bisect(fn, lo: float, hi: float) -> float:
    """Root of a decreasing function on [lo, hi] (clamped to the ends)."""
    if fn(lo) <= 0:
        return lo
    if fn(hi) >= 0:
        return hi
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if fn(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def equilibrium(bank: dict, su: dict, amb: float, pc: float, pg: float) -> tuple[float, float]:
    """CPU and GPU temperatures where fan speeds and temperatures agree. Each temperature's implied value falls as
    the temperature rises (hotter -> faster fans), so each is found by bisection, alternating until both settle."""
    tc = tg = amb + 20.0
    hi = amb + 400.0
    for _ in range(OUTER_ITERS):
        tc_n = _bisect(lambda t: _implied(bank, su, amb, pc, pg, t, tg)["tc"] - t, amb, hi)
        tg_n = _bisect(lambda t: _implied(bank, su, amb, pc, pg, tc_n, t)["tg"] - t, amb, hi)
        done = abs(tc_n - tc) < 1e-5 and abs(tg_n - tg) < 1e-5
        tc, tg = tc_n, tg_n
        if done:
            break
    return tc, tg


def _throttle(bank: dict, su: dict, amb: float, pc: float, pg: float, part: str) -> float:
    """Highest power of `part` (cpu / gpu) that keeps it at its temperature limit."""
    lim = su[part]["t_max"]
    idx = 0 if part == "cpu" else 1

    def over(p):
        t = equilibrium(bank, su, amb, p if part == "cpu" else pc, p if part == "gpu" else pg)[idx]
        return lim - t                                      # decreasing in p
    return _bisect(over, 0.0, pc if part == "cpu" else pg)


def solve(bank: dict, su: dict, ambient_c: float, cpu_w: float, gpu_w: float) -> dict:
    """Steady state of a build (bank + setup) at a room temperature and CPU / GPU power."""
    amb, pc, pg = float(ambient_c), float(cpu_w), float(gpu_w)
    tc, tg = equilibrium(bank, su, amb, pc, pg)
    thr = {"cpu": False, "gpu": False}
    for _ in range(3):                                      # a throttled part changes the other's air: repeat
        changed = False
        if tc > su["cpu"]["t_max"] + 1e-3:
            pc, thr["cpu"], changed = _throttle(bank, su, amb, pc, pg, "cpu"), True, True
            tc, tg = equilibrium(bank, su, amb, pc, pg)
        if tg > su["gpu"]["t_max"] + 1e-3:
            pg, thr["gpu"], changed = _throttle(bank, su, amb, pc, pg, "gpu"), True, True
            tc, tg = equilibrium(bank, su, amb, pc, pg)
        if not changed:
            break
    s = _implied(bank, su, amb, pc, pg, tc, tg)
    b = s["b"]
    fans = []
    for f in su["case_fans"]:
        duty = s["d_case"] if f["curve"] else f["pct"] / 100.0
        sp = fan_speed(duty, f["s_min"])
        q = next((x["q_m3s"] for x in b["fans"] if x["slot"] == f["slot"]), 0.0)
        fans.append({"slot": f["slot"], "curve": f["curve"], "duty": duty, "speed": sp, "rpm": sp * f["rpm_max"], "q_m3s": q})
    co, gp = su["cooler"], su["gpu"]
    noise = noise_dba([(f["noise"], x["speed"], 1) for f, x in zip(su["case_fans"], fans)]
                      + [(co["noise"], s["s_cpu"], co["fans"]), (gp["noise"], s["s_gpu"], gp["fans"])])
    return {
        "ambient_c": amb,
        "cpu": {"temp_c": tc, "watts": pc, "watts_set": float(cpu_w), "throttled": thr["cpu"], "intake_rise_c": s["rise_c"],
                "g_w_per_k": s["g_cpu"], "fan_duty": s["d_cpu"], "fan_speed": s["s_cpu"], "fan_rpm": s["s_cpu"] * co["rpm_max"],
                "t_max_c": su["cpu"]["t_max"]},
        "gpu": {"temp_c": tg, "watts": pg, "watts_set": float(gpu_w), "throttled": thr["gpu"], "intake_rise_c": s["rise_g"],
                "g_w_per_k": s["g_gpu"], "fan_duty": s["d_gpu"], "fan_speed": s["s_gpu"], "fan_rpm": s["s_gpu"] * gp["rpm_max"],
                "t_max_c": su["gpu"]["t_max"]},
        "case_duty": s["d_case"], "fans": fans,
        "flow_in_m3s": b["flow_in_m3s"], "flow_out_m3s": b["flow_out_m3s"], "case_pa": b["case_pa"],
        "air_mean_rise_c": b["k_air_mean"][0] * pc + b["k_air_mean"][1] * pg,
        "air_max_rise_c": b["k_air_max"][0] * pc + b["k_air_max"][1] * pg,
        "noise_dba": noise,
        "dust_rate": dust_rate(bank, s["d_case"]),
    }
