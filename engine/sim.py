"""CaseFlow simulator: airflow, heat and dust in a PC case built from the local parts library.

Frame (mm, case interior): x = front (0) -> rear (D), y = glass side (0) -> tray side (W), z = bottom -> top.
Airflow: our LBM with fan openings as imposed-velocity cells, ambient air outside the shell. Each case fan's
flow is its operating point: linear fan curve against one common case pressure, balanced by the passive openings.
Mesh panels: empty fan positions are left open (real mesh is mostly open area); glass is closed.
Heat: CPU / GPU power released in their heatsink zones; air temperature by advection-diffusion;
part temperature = intake air + P / G (G = cooler conductance, W/K).
Dust: particles released at the openings air comes in through (weighted by inflow, reduced by a
filter if the panel has one), tracked through the time-averaged flow with eddy diffusion and
settling; deposition where they hit a surface.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from engine.lbm import LBMConfig, LBMSolver
from engine.scenario import fan_speed
from engine.thermal import solve_temperature_converged

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "data" / "parts" / "library.json"
DX = 6.0                       # mm per cell
U_LAT_PER_MS = 0.05 / 1.5      # lattice velocity per m/s
CFM_TO_M3S = 0.000471947
DERATE = 0.6                   # free-air CFM -> in-case flow (system impedance, filters); typical 0.5-0.7
FILTER_EFF = 0.85              # fraction of ~20 um dust stopped by a dust filter (assumption)
G_DEFAULT = {"dual_tower": 2.0, "dual_flow_through": 8.5, "open_air": 8.0}   # W/K
# fan operating point (case as one plenum): every case fan works against the same case gauge pressure p
RHO_AIR = 1.2                  # kg/m^3
CD_ORIFICE = 0.6               # discharge coefficient of a passive opening (sharp-edged grille)
POROSITY = {"open": 0.6, "vent": 0.4}   # open-area fraction: empty fan grille on mesh / vented slot covers (est.)
FILTER_OPEN = 0.7              # extra open-area factor of a dust filter
ASSIST_CAP = 1.25              # a fan helped by the case pressure delivers at most 1.25x its free flow
# The GPU's and the CPU cooler's own fans move air inside the case at a representative speed: 1.6 m/s is about the
# 5080 FE's speed in games (~41 %). Their real speed follows their own curve; engine/scenario.py turns that into the
# cooler's conductance, while the air they stir in the case stays at this speed.
GPU_FAN_MS = 1.6               # air speed through the GPU's own fans
GPU_FAN_REF_SPEED = 0.41       # the GPU fan speed (rpm / rpm_max) that GPU_FAN_MS stands for
COOLER_FAN_SPEED = 0.7         # CPU cooler fans: share of their rated flow
MMH2O_TO_PA = 9.80665


def fan_flow(q_free: float, p_max: float, dp: float) -> float:
    """Linear fan curve: Q = Q_free (1 - dp / p_max), dp = pressure rise the fan must make."""
    return q_free * min(max(1.0 - dp / p_max, 0.0), ASSIST_CAP)


def operating_point(fans: list[tuple[str, float, float]], a_eff: float) -> tuple[float, list[float]]:
    """Case gauge pressure p (Pa) where the fans' mass balance closes through the passive openings.
    fans: (mode, Q_free m^3/s, p_max Pa). Intake fans push against +p, exhaust fans are helped by it;
    passive openings pass Cd * A_eff * sqrt(2 |p| / rho) (out when p > 0). Net flow falls monotonically with p,
    so bisection finds the single root."""
    def net(p):
        q = sum(fan_flow(qf, pm, p) if md == "in" else -fan_flow(qf, pm, -p) for md, qf, pm in fans)
        return q - math.copysign(CD_ORIFICE * a_eff * math.sqrt(2.0 * abs(p) / RHO_AIR), p)
    lo, hi = -300.0, 300.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if net(mid) > 0 else (lo, mid)
    p = 0.5 * (lo + hi)
    return p, [fan_flow(qf, pm, p if md == "in" else -p) for md, qf, pm in fans]


def load_library() -> dict:
    return json.loads(LIB_PATH.read_text(encoding="utf-8"))


def find(lib: dict, cat: str, pid: str) -> dict:
    for p in lib[cat]:
        if p["id"] == pid:
            return p
    raise KeyError(f"{cat}:{pid}")


def search(lib: dict, q: str, cat: str | None = None, limit: int = 12) -> list[dict]:
    q = q.strip().lower()
    out = []
    for c in ([cat] if cat else ["cases", "gpus", "coolers", "fans"]):
        for p in lib.get(c, []):
            hay = " ".join([p["name"], p.get("brand", "")] + p.get("aliases", [])).lower()
            if not q or all(tok in hay for tok in q.split()):
                score = (0 if p.get("brand") == "generic" else 1) + (2 if q and q in p["name"].lower() else 0)
                out.append((score, c, p))
    out.sort(key=lambda t: -t[0])
    return [{"category": c, **p} for _, c, p in out[:limit]]


def default_build() -> dict:
    """The user's actual PC (2026-11-07): the case's four stock fans - 2 x 140 mm front intake and 2 x 120 mm
    reverse-blade bottom intake under the GPU - plus their own 120 mm fans on top x2 + rear (exhaust).
    Every case fan follows the board's case-fan curve (control "curve"); pct is the duty it runs at when set to
    "fixed". Load: the CPU-heavy game measured on this PC (86 W CPU, 146 W GPU) in a 23 degC room."""
    return {
        "case": "lianli-lancool-207", "gpu": "nvidia-rtx-5080-fe", "cooler": "thermalright-pa120-se-argb",
        "cpu": "amd-ryzen-7-9700x",
        "fans": [
            {"slot": "front_low", "fan": "lianli-207-front-140", "mode": "in", "control": "curve", "pct": 70},
            {"slot": "front_high", "fan": "lianli-207-front-140", "mode": "in", "control": "curve", "pct": 70},
            {"slot": "bottom_front", "fan": "lianli-207-bottom-120-rev", "mode": "in", "control": "curve", "pct": 70},
            {"slot": "bottom_rear", "fan": "lianli-207-bottom-120-rev", "mode": "in", "control": "curve", "pct": 70},
            {"slot": "top_2", "fan": "generic-120", "mode": "out", "control": "curve", "pct": 80},
            {"slot": "top_3", "fan": "generic-120", "mode": "out", "control": "curve", "pct": 80},
            {"slot": "rear", "fan": "generic-120", "mode": "out", "control": "curve", "pct": 80},
        ],
        "load": {"cpu_w": 86.0, "gpu_w": 146.0, "ambient_c": 23.0},
    }


# ----------------------------------------------------------------------------- geometry
class Grid:
    def __init__(self, D, W, H):
        self.off = 2                                   # reservoir layer + wall layer
        self.nx = int(round(D / DX)) + 2 * self.off
        self.ny = int(round(W / DX)) + 2 * self.off
        self.nz = int(round(H / DX)) + 2 * self.off
        self.shape = (self.nx, self.ny, self.nz)

    def idx(self, v):
        return int(round(v / DX)) + self.off

    def box(self, mask, x0, x1, y0, y1, z0, z1, value=True):
        i0, i1 = max(self.idx(x0), 0), min(self.idx(x1), self.nx)
        j0, j1 = max(self.idx(y0), 0), min(self.idx(y1), self.ny)
        k0, k1 = max(self.idx(z0), 0), min(self.idx(z1), self.nz)
        mask[i0:i1, j0:j1, k0:k1] = value
        return (i0, i1, j0, j1, k0, k1)


def _wall_box(wall, a, b, s, D, W, H):
    """Opening box (mm) for a square fan of size s centred at (a, b) on a wall, spanning the shell."""
    h = s / 2
    if wall == "front":
        return (-DX, DX, a - h, a + h, b - h, b + h), np.array([1.0, 0, 0])      # inward normal
    if wall == "rear":
        return (D - DX, D + DX, a - h, a + h, b - h, b + h), np.array([-1.0, 0, 0])
    if wall == "top":
        return (a - h, a + h, b - h, b + h, H - DX, H + DX), np.array([0, 0, -1.0])
    if wall == "bottom":
        return (a - h, a + h, b - h, b + h, -DX, DX), np.array([0, 0, 1.0])
    raise ValueError(wall)


def build_model(build: dict, lib: dict | None = None, case_duty: float | None = None,
                cooler_speed: float = COOLER_FAN_SPEED, gpu_ms: float = GPU_FAN_MS) -> dict:
    """Voxel model of a build. case_duty: PWM duty (0..1) of the case fans that follow the case-fan curve
    (control "curve"); None runs every fan at its own pct. Fans set to "fixed" always run at their pct. A fan's
    speed (rpm / rpm_max) is s_min + (1 - s_min) * duty: most PWM fans keep turning at 0 %."""
    lib = lib or load_library()
    case = find(lib, "cases", build["case"])
    gpu = find(lib, "gpus", build["gpu"])
    cooler = find(lib, "coolers", build["cooler"])
    D, W, H = case["dims_mm"]["D"], case["dims_mm"]["W"], case["dims_mm"]["H"]
    g = Grid(D, W, H)
    shape = g.shape
    solid = np.zeros(shape, bool)
    reservoir = np.zeros(shape, bool)
    reservoir[0, :, :] = reservoir[-1, :, :] = True
    reservoir[:, 0, :] = reservoir[:, -1, :] = True
    reservoir[:, :, 0] = reservoir[:, :, -1] = True
    solid[1:-1, 1:-1, 1:-1] = True
    solid[2:-2, 2:-2, 2:-2] = False
    forced = np.zeros(shape, bool)
    fu = np.zeros((3,) + shape, np.float32)
    boxes = []                                              # for drawing (mm)
    openings = []                                           # for dust sources
    filters = case.get("filters", {})
    used = {f["slot"] for f in build["fans"]}
    mounts = {m["slot"]: m for m in case["mounts"]}
    # empty mounts on mesh walls and passive vents stay open; their effective area sets the case's system curve
    a_eff = 0.0
    for m in case["mounts"]:
        if m["slot"] in used:
            continue
        if "mesh" not in case.get("panels", {}).get(m["wall"], "") and m["wall"] != "rear":
            continue
        s = max(m["sizes"])
        bx, _ = _wall_box(m["wall"], m["a"], m["b"], s, D, W, H)
        c = g.box(np.zeros(shape, bool), *bx)
        solid[c[0]:c[1], c[2]:c[3], c[4]:c[5]] = False
        filt = bool(filters.get(m["wall"], False))
        a_eff += (s / 1000.0) ** 2 * POROSITY["open"] * (FILTER_OPEN if filt else 1.0)
        openings.append({"box": c, "kind": "open", "filtered": filt, "wall": m["wall"]})
        boxes.append({"name": f"open mount ({m['slot']})", "kind": "open", "b": list(bx)})
    for v in case.get("vents", []):
        y0, y1, z0, z1 = v["box"]
        c = g.box(np.zeros(shape, bool), D - DX, D + DX, y0, y1, z0, z1)
        solid[c[0]:c[1], c[2]:c[3], c[4]:c[5]] = False
        a_eff += (y1 - y0) * (z1 - z0) / 1e6 * POROSITY["vent"]
        openings.append({"box": c, "kind": "vent", "filtered": False, "wall": "rear"})
        boxes.append({"name": v["name"], "kind": "vent", "b": [D - 2, D + 2, y0, y1, z0, z1]})
    # case fans at their operating point against the common case pressure
    active = []
    for f in build["fans"]:
        m = mounts.get(f["slot"])
        if m is None or f.get("mode") not in ("in", "out"):
            continue
        fan = find(lib, "fans", f["fan"])
        follows = f.get("control", "curve") == "curve" and case_duty is not None
        duty = case_duty if follows else f.get("pct", 100) / 100.0
        frac = fan_speed(duty, fan.get("s_min", 0.2))               # rpm / rpm_max
        q_free = fan["cfm"] * CFM_TO_M3S * frac * DERATE            # fan laws: Q ~ rpm, p ~ rpm^2
        p_max = fan.get("static_mmh2o", 1.6) * MMH2O_TO_PA * frac ** 2
        active.append((f, m, fan, q_free, p_max, duty, frac))
    case_pa, q_ops = operating_point([(a[0]["mode"], a[3], a[4]) for a in active], a_eff)
    fan_info = []
    for (f, m, fan, q_free, p_max, duty, frac), q in zip(active, q_ops):
        s = fan["size_mm"] if fan["size_mm"] in m["sizes"] else max(m["sizes"])
        bx, n_in = _wall_box(m["wall"], m["a"], m["b"], s, D, W, H)
        c = g.box(np.zeros(shape, bool), *bx)
        solid[c[0]:c[1], c[2]:c[3], c[4]:c[5]] = False
        v = q / (s / 1000.0) ** 2                              # m/s face velocity
        d = n_in * (1 if f["mode"] == "in" else -1) * v * U_LAT_PER_MS
        forced[c[0]:c[1], c[2]:c[3], c[4]:c[5]] = True
        for k in range(3):
            fu[k][c[0]:c[1], c[2]:c[3], c[4]:c[5]] = d[k]
        fan_info.append({**f, "wall": m["wall"], "a": m["a"], "b": m["b"], "size": s, "name": fan["name"],
                         "q_m3s": q, "q_free_m3s": q_free, "op_frac": q / q_free if q_free else 0.0, "v_ms": v,
                         "duty": duty, "speed": frac, "rpm": frac * fan["rpm_max"],
                         "filtered": bool(filters.get(m["wall"], False))})
        openings.append({"box": c, "kind": "fan", "mode": f["mode"], "filtered": bool(filters.get(m["wall"], False)), "wall": m["wall"]})
    # PSU
    p = case["psu"]["box"]
    g.box(solid, *p)
    boxes.append({"name": "PSU", "kind": "psu", "b": p})
    # motherboard
    bd = case["board"]
    py = bd["plane_y"]
    bx0, bx1, bz0, bz1 = bd["box_xz"]
    g.box(solid, bx0, bx1, py, py + 8, bz0, bz1)
    boxes.append({"name": "motherboard", "kind": "board", "b": [bx0, bx1, py, py + 8, bz0, bz1]})
    heat = {"cpu": np.zeros(shape, bool), "gpu": np.zeros(shape, bool)}
    intake = {"cpu": np.zeros(shape, bool), "gpu": np.zeros(shape, bool)}
    # GPU: bracket at the rear wall, card hangs from the board towards the glass
    L, Hh, T = gpu["length_mm"], gpu["height_mm"], gpu["thickness_mm"]
    gx1, gx0 = D - 8, D - 8 - L
    gy0, gy1 = py - Hh, py
    gz1 = bd["gpu_slot_top_z"]
    gz0 = gz1 - T
    v_gpu = gpu_ms * U_LAT_PER_MS
    if gpu["cooler"] == "dual_flow_through":
        zl = 0.3 * L
        zones = [(gx0, gx0 + zl), (gx1 - zl, gx1)]
        g.box(solid, gx0 + zl, gx1 - zl, gy0, gy1, gz0, gz1)          # PCB / vapour chamber block
        boxes.append({"name": "GPU (PCB + vapour chamber)", "kind": "gpu", "b": [gx0 + zl, gx1 - zl, gy0, gy1, gz0, gz1]})
        for (a0, a1) in zones:
            g.box(heat["gpu"], a0, a1, gy0, gy1, gz0 + 8, gz1)
            c = g.box(forced, a0 + 6, a1 - 6, gy0 + 6, gy1 - 6, gz0, gz0 + 8)
            fu[2][c[0]:c[1], c[2]:c[3], c[4]:c[5]] = v_gpu
            g.box(intake["gpu"], a0 + 6, a1 - 6, gy0 + 6, gy1 - 6, gz0 - 20, gz0 - 4)
            boxes.append({"name": "GPU flow-through fin stack", "kind": "gpu_hs", "b": [a0, a1, gy0, gy1, gz0 + 8, gz1]})
            boxes.append({"name": "GPU fan", "kind": "gpu_fan", "b": [a0 + 6, a1 - 6, gy0 + 6, gy1 - 6, gz0, gz0 + 8]})
    else:
        g.box(solid, gx0, gx1, gy0, gy1, gz1 - 7, gz1)                 # PCB + backplate on top
        g.box(heat["gpu"], gx0, gx1, gy0, gy1, gz0 + 8, gz1 - 7)
        c = g.box(forced, gx0 + 15, gx1 - 15, gy0 + 8, gy1 - 8, gz0, gz0 + 8)
        fu[2][c[0]:c[1], c[2]:c[3], c[4]:c[5]] = v_gpu
        g.box(intake["gpu"], gx0 + 15, gx1 - 15, gy0 + 8, gy1 - 8, gz0 - 20, gz0 - 4)
        boxes += [{"name": "GPU PCB + backplate", "kind": "gpu", "b": [gx0, gx1, gy0, gy1, gz1 - 7, gz1]},
                  {"name": "GPU heatsink", "kind": "gpu_hs", "b": [gx0, gx1, gy0, gy1, gz0 + 8, gz1 - 7]},
                  {"name": "GPU fans", "kind": "gpu_fan", "b": [gx0 + 15, gx1 - 15, gy0 + 8, gy1 - 8, gz0, gz0 + 8]}]
    # CPU cooler: dual tower, fans push front -> rear
    sx, sz = bd["socket_xz"]
    ch, cw, cd = cooler["height_mm"], cooler["width_mm"], cooler["depth_mm"]
    cx0 = sx - cd / 2
    cz0, cz1 = sz - cw / 2, sz + cw / 2
    cy0, cy1 = py - ch, py
    cfan = find(lib, "fans", cooler["fans"][0]["fan"])
    v_cool = cfan["cfm"] * CFM_TO_M3S * cooler_speed / (cfan["size_mm"] / 1000.0) ** 2 * U_LAT_PER_MS
    segs = [("fan", 0, 25), ("tower", 25, 67), ("fan", 67, 92), ("tower", 92, cd)] if cooler["type"] == "dual_tower" else [("fan", 0, 25), ("tower", 25, cd)]
    for kind, a, b in segs:
        if kind == "fan":
            c = g.box(forced, cx0 + a, cx0 + b, cy0 + 10, cy1 - 10, cz0 + 2, cz1 - 2)
            fu[0][c[0]:c[1], c[2]:c[3], c[4]:c[5]] = v_cool
            boxes.append({"name": "CPU cooler fan", "kind": "cpu_fan", "b": [cx0 + a, cx0 + b, cy0 + 10, cy1 - 10, cz0 + 2, cz1 - 2]})
        else:
            g.box(heat["cpu"], cx0 + a, cx0 + b, cy0, cy1 - 20, cz0, cz1)
            boxes.append({"name": "CPU cooler fin tower", "kind": "cpu_hs", "b": [cx0 + a, cx0 + b, cy0, cy1 - 20, cz0, cz1]})
    g.box(intake["cpu"], cx0 - 20, cx0 - 4, cy0 + 10, cy1 - 10, cz0 + 2, cz1 - 2)
    solid &= ~reservoir
    forced &= ~reservoir
    checks = {
        "gpu_fits": L <= case.get("max_gpu_len_mm", 1e9), "gpu_len": L, "gpu_max": case.get("max_gpu_len_mm"),
        "cooler_fits": ch <= case.get("max_cooler_h_mm", 1e9), "cooler_h": ch, "cooler_max": case.get("max_cooler_h_mm"),
    }
    return {"grid": g, "shape": shape, "solid": solid, "reservoir": reservoir, "forced": forced, "fu": fu,
            "heat": heat, "intake": intake, "boxes": boxes, "fans": fan_info, "openings": openings,
            "case": case, "gpu": gpu, "cooler": cooler, "dims": (D, W, H), "checks": checks,
            "case_pa": case_pa, "a_eff_m2": a_eff}


# ----------------------------------------------------------------------------- dust
def simulate_dust(u_ms: torch.Tensor, model: dict, n: int = 40000, steps: int = 2500, d_um: float = 20.0, seed: int = 0):
    """Lagrangian dust particles in the time-averaged flow. Returns deposition per cell + summary."""
    dev = u_ms.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    shape = model["shape"]
    solid = torch.from_numpy(model["solid"]).to(dev)
    res = torch.from_numpy(model["reservoir"]).to(dev)
    X, Y, Z = shape
    # inflow-weighted sources at openings
    src_pts, src_w, src_filtered = [], [], []
    for o in model["openings"]:
        i0, i1, j0, j1, k0, k1 = o["box"]
        sub = u_ms[:, i0:i1, j0:j1, k0:k1]
        w = o["wall"]
        comp = {"front": (0, 1), "rear": (0, -1), "top": (2, -1), "bottom": (2, 1)}[w]
        inflow = (sub[comp[0]] * comp[1]).clamp(min=0)               # inward velocity component
        flux = float(inflow.sum())
        if flux <= 1e-6:
            continue
        ii, jj, kk = torch.nonzero(inflow > 0.02 * inflow.max(), as_tuple=True)
        src_pts.append(torch.stack([ii + i0, jj + j0, kk + k0], 1).float())
        wgt = flux * (1 - FILTER_EFF if o["filtered"] else 1.0)
        src_w.append((wgt, o["filtered"], flux))
    if not src_pts:
        return None
    weights = torch.tensor([w for w, _, _ in src_w], device=dev)
    counts = torch.distributions.Multinomial(n, probs=weights / weights.sum()).sample().long() if weights.sum() > 0 else None
    pos = []
    for pts, c in zip(src_pts, counts.tolist()):
        if c == 0:
            continue
        sel = torch.randint(0, len(pts), (c,), generator=gen, device=dev)
        pos.append(pts[sel] + torch.rand((c, 3), generator=gen, device=dev))
    pos = torch.cat(pos)
    N = len(pos)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    dep = torch.zeros(shape, device=dev)
    # physics (cell units): dt chosen so the fastest air moves <= 0.5 cell per step
    umax = float(u_ms.abs().max()) + 1e-6
    dt = 0.5 * (DX / 1000.0) / umax
    rho_p, mu = 1500.0, 1.8e-5
    v_settle = rho_p * 9.81 * (d_um * 1e-6) ** 2 / (18 * mu)          # Stokes settling, m/s
    D_eddy = 2e-3                                                      # m^2/s, turbulent mixing in the case
    sig = math.sqrt(2 * D_eddy * dt) / (DX / 1000.0)
    exited = 0
    for _ in range(steps):
        if not bool(alive.any()):
            break
        p = pos[alive]
        ci = p.long()
        ci[:, 0].clamp_(0, X - 1); ci[:, 1].clamp_(0, Y - 1); ci[:, 2].clamp_(0, Z - 1)
        uu = u_ms[:, ci[:, 0], ci[:, 1], ci[:, 2]].T                      # [n, 3] m/s
        step = (uu * dt) / (DX / 1000.0)
        step[:, 2] -= v_settle * dt / (DX / 1000.0)
        step += torch.randn(step.shape, generator=gen, device=dev) * sig
        q = p + step
        qi = q.long()
        qi[:, 0].clamp_(0, X - 1); qi[:, 1].clamp_(0, Y - 1); qi[:, 2].clamp_(0, Z - 1)
        hit = solid[qi[:, 0], qi[:, 1], qi[:, 2]]
        out = res[qi[:, 0], qi[:, 1], qi[:, 2]]
        # deposit on the fluid cell next to the surface it hit
        if bool(hit.any()):
            hc = ci[hit]
            dep.index_put_((hc[:, 0], hc[:, 1], hc[:, 2]), torch.ones(len(hc), device=dev), accumulate=True)
        q = torch.where((hit | out).unsqueeze(1), p, q)
        idx = torch.nonzero(alive, as_tuple=True)[0]
        pos[idx] = q
        exited += int(out.sum())
        gone = hit | out
        alive[idx[gone]] = False
    deposited = float(dep.sum())
    still = int(alive.sum())
    total_in = sum(f for _, _, f in src_w)
    filtered_in = sum(f for _, fl, f in src_w if fl)
    return {
        "dep": dep.cpu().numpy(), "n": N, "deposited_frac": deposited / N, "exited_frac": exited / N,
        "airborne_frac": still / N, "inflow_through_filters_frac": filtered_in / max(total_in, 1e-9),
        "dust_entering_rel": float(weights.sum()) / max(total_in, 1e-9), "d_um": d_um,
    }


def dust_by_part(dep: np.ndarray, model: dict) -> list[dict]:
    """Share of deposited dust on each drawn part (cells within one cell of the part box)."""
    g = model["grid"]
    total = dep.sum() + 1e-9
    out = []
    for b in model["boxes"]:
        if b["kind"] in ("open", "vent"):
            continue
        x0, x1, y0, y1, z0, z1 = b["b"]
        c = (max(g.idx(x0) - 1, 0), g.idx(x1) + 1, max(g.idx(y0) - 1, 0), g.idx(y1) + 1, max(g.idx(z0) - 1, 0), g.idx(z1) + 1)
        out.append({"name": b["name"], "kind": b["kind"], "share": float(dep[c[0]:c[1], c[2]:c[3], c[4]:c[5]].sum() / total)})
    # case walls (cells next to the shell)
    o0 = g.off
    walls = {"front panel": dep[o0, :, :], "rear wall": dep[g.nx - o0 - 1, :, :], "glass side": dep[:, o0, :],
             "tray side": dep[:, g.ny - o0 - 1, :], "case floor": dep[:, :, o0], "top panel": dep[:, :, g.nz - o0 - 1]}
    for name, sl in walls.items():
        out.append({"name": name, "kind": "wall", "share": float(sl.sum() / total)})
    # merge duplicates by name
    merged = {}
    for o in out:
        merged.setdefault(o["name"], {"name": o["name"], "kind": o["kind"], "share": 0.0})["share"] += o["share"]
    return sorted(merged.values(), key=lambda o: -o["share"])


# ----------------------------------------------------------------------------- run
def run_airflow(model: dict, steps: int = 8000, callback=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Time-averaged velocity [3, X, Y, Z] and gauge pressure [X, Y, Z], both in lattice units.
    The room layer around the case is an open pressure boundary (see LBMSolver open_shell)."""
    cfg = LBMConfig(u_in=0.0, tau=0.52, smagorinsky=0.16, sponge=0)
    solver = LBMSolver(torch.from_numpy(model["solid"])[None], cfg, open_inlet=False,
                       forced_mask=torch.from_numpy(model["forced"]), forced_u=torch.from_numpy(model["fu"]),
                       reservoir_mask=torch.from_numpy(model["reservoir"]), open_shell=True)
    res = solver.run(steps=steps, avg_from=int(steps * 0.5), report_every=200,
                     callback=(lambda s, n, cd: callback(s / n)) if callback else None, check_nan_every=1000)
    u, p = res.u[0], res.p[0]
    del solver
    return u, p


def run_airflow_batch(model: dict, fu: torch.Tensor, steps: int = 8000, callback=None) -> torch.Tensor:
    """Time-averaged velocity [L, 3, X, Y, Z] (lattice units) for L sets of fan velocities fu [L, 3, X, Y, Z] in the
    same geometry, solved together as one batch on the GPU (one run per level would cost L times the kernel calls)."""
    n = fu.shape[0]
    cfg = LBMConfig(u_in=0.0, tau=0.52, smagorinsky=0.16, sponge=0)
    solid = torch.from_numpy(model["solid"])[None].expand(n, -1, -1, -1).contiguous()
    solver = LBMSolver(solid, cfg, open_inlet=False, forced_mask=torch.from_numpy(model["forced"]), forced_u=fu,
                       reservoir_mask=torch.from_numpy(model["reservoir"]), open_shell=True)
    res = solver.run(steps=steps, avg_from=int(steps * 0.5), report_every=200,
                     callback=(lambda s, n_, cd: callback(s / n_)) if callback else None, check_nan_every=1000)
    u = res.u
    del solver
    return u


def simulate_build(build: dict, steps: int = 8000, callback=None, with_dust: bool = True) -> dict:
    lib = load_library()
    t0 = time.time()
    model = build_model(build, lib)
    load = {"cpu_w": 88.0, "gpu_w": 300.0, "ambient_c": 25.0, **build.get("load", {})}
    u_lat, _ = run_airflow(model, steps, callback=(lambda f: callback(0.6 * f)) if callback else None)
    if callback:
        callback(0.62)
    parts_thermal = {"cpu_w": load["cpu_w"], "gpu_w": load["gpu_w"]}
    inside = ~(model["solid"] | model["reservoir"])
    T, thermal_iters = solve_temperature_converged(
        u_lat, {"shape": model["shape"], "solid": model["solid"], "reservoir": model["reservoir"], "heat": model["heat"]},
        parts_thermal, probes=[model["intake"]["cpu"], model["intake"]["gpu"], inside], dx_mm=DX, u_lat_per_ms=U_LAT_PER_MS)
    if callback:
        callback(0.8)
    u_ms = u_lat / U_LAT_PER_MS
    dust = simulate_dust(u_ms, model) if with_dust else None
    if callback:
        callback(0.97)
    Tn = T.cpu().numpy()
    g_cpu = build.get("g_cpu") or G_DEFAULT.get(model["cooler"]["type"], 2.0)
    g_gpu = build.get("g_gpu") or G_DEFAULT.get(model["gpu"]["cooler"], 8.0)
    parts = {}
    for key, gval in (("cpu", g_cpu), ("gpu", g_gpu)):
        im = model["intake"][key]
        t_in = float(Tn[im].mean()) if im.any() else 0.0
        parts[key] = {"intake_rise_c": t_in, "temp_c": load["ambient_c"] + t_in + load[f"{key}_w"] / gval,
                      "watts": load[f"{key}_w"], "g_w_per_k": gval}
    q_in = sum(f["q_m3s"] for f in model["fans"] if f["mode"] == "in")
    q_out = sum(f["q_m3s"] for f in model["fans"] if f["mode"] == "out")
    um = u_ms.float().cpu().numpy()
    speed = np.sqrt((um ** 2).sum(0))
    out = {
        "model": model, "u": um, "T": Tn, "parts": parts, "load": load,
        "flow_in_m3s": q_in, "flow_out_m3s": q_out, "case_pa": model["case_pa"],
        "pressure": "positive" if q_in > q_out * 1.05 else ("negative" if q_out > q_in * 1.05 else "balanced"),
        "dead_air_fraction": float((speed[inside] < 0.05).mean()), "air_max_rise_c": float(Tn[inside].max()),
        "seconds": time.time() - t0, "dust": None, "thermal_iters": thermal_iters, "lbm_steps": steps,
    }
    if dust:
        dust["by_part"] = dust_by_part(dust["dep"], model)
        out["dust"] = dust
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- packaging for the web
def deposit_points(dep: np.ndarray, model: dict, n: int = 2500) -> list:
    g = model["grid"]
    idx = np.argwhere(dep > 0)
    if len(idx) == 0:
        return []
    w = dep[dep > 0].astype(np.float64)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), size=min(n, int(w.sum())), p=w / w.sum())
    pts = (idx[pick] - g.off + rng.random((len(pick), 3))) * DX
    return np.round(pts, 1).tolist()
