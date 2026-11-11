"""Figures for docs/GUIDE.md, made from the engine, the cached banks (data/banks) and data/optimize.json (no GPU run).

    python docs/make_figures.py        # from the caseflow folder; writes docs/media/*.png

Chart colours were checked with the dataviz palette validator on the #0b1119 surface: CPU blue #3987e5 / GPU orange
#d95926 / case aqua #199e70 (all checks pass, all pairs), the older categorical cyan #1f9fd0 / amber #d9772b for
intake / exhaust, and the ordinal cyan ramp #a9e2f7 #5cc0e8 #1f8fbf #17698d (all checks pass).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.optimize import build_for, candidates  # noqa: E402
from engine.sim import (CD_ORIFICE, CFM_TO_M3S, DERATE, MMH2O_TO_PA, RHO_AIR, build_model,  # noqa: E402
                        default_build, fan_flow, find, load_library)

OUT = ROOT / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
BG, INK, MUTED, GRID = "#0b1119", "#dde9f3", "#8b98a9", "#1c2735"
CY, AM = "#1f9fd0", "#d9772b"                      # intake / exhaust
CPU_C, GPU_C, CASE_C = "#3987e5", "#d95926", "#199e70"
ORD = ["#a9e2f7", "#5cc0e8", "#1f8fbf", "#17698d"]
M3S_TO_CFM = 1 / CFM_TO_M3S
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 150,
})
LIB = load_library()
HOME = default_build()
OPT = json.loads((ROOT / "data" / "optimize.json").read_text())
DEFAULT = ("cpu_game", 23)                         # the reference PC's measured game, its room


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def preset(pid):
    return next(p for p in LIB["defaults"]["scenarios"] if p["id"] == pid)


def steady(b, summary, room, pid, su=None):
    su = su or sc.setup_for(b, LIB)
    p = preset(pid)
    pc, pg = sc.load_watts(su, p["cpu"], p["gpu"])
    return sc.solve(summary, su, room, pc, pg), su


def ranking(pid, room):
    return next(r for r in OPT["rankings"] if r["scenario"] == pid and r["ambient_c"] == room)


def layout(name):
    return next(L for L in OPT["layouts"] if L["name"] == name)


def level_weights(levels, duty):
    """The two levels and weights scenario.bank_state uses at a duty (linear in 1 / speed)."""
    if len(levels) == 1:
        return [(0, 1.0)]
    inv = [1 / sc.fan_speed(L["duty"], sc.S_MIN_REF) for L in levels][::-1]
    x = 1 / sc.fan_speed(duty, sc.S_MIN_REF)
    i = 0
    while i < len(inv) - 2 and x > inv[i + 1]:
        i += 1
    f = (x - inv[i]) / (inv[i + 1] - inv[i])
    n = len(levels)
    return [(n - 1 - i, 1 - f), (n - 2 - i, f)]


def fields_at(b, pid, room):
    """Air velocity (nearest level) and temperature rise (composed for the scenario) of a build, plus its result."""
    B = bk.load(bk.bank_key(b))
    s = B["summary"]
    r, _ = steady(b, s, room, pid)
    lv = s["levels"]
    T = sum(w * (r["cpu"]["watts"] * B["fields"][k]["kc"].astype(np.float32) + r["gpu"]["watts"] * B["fields"][k]["kg"].astype(np.float32))
            for k, w in level_weights(lv, r["case_duty"]))
    near = min(range(len(lv)), key=lambda k: abs(sc.fan_speed(lv[k]["duty"], sc.S_MIN_REF) - sc.fan_speed(r["case_duty"], sc.S_MIN_REF)))
    return s, B["fields"][near]["u"].astype(np.float32), np.clip(T, 0, None), r, near


def gpu_slice_index(geo):
    gb = next(b for b in geo["boxes"] if b["kind"] == "gpu")
    return int(round((gb["b"][2] + gb["b"][3]) / 2 / geo["dx_mm"] + geo["off"]))


def plain_axes(ax, X):
    ax.invert_xaxis()                                                # through the glass: front on the right
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------- 1. how the grid sees the case
def fig_grid():
    m = build_model(HOME)
    geo = bk.geometry(m)
    j = gpu_slice_index(geo)
    solid, forced, res, fu = m["solid"], m["forced"], m["reservoir"], m["fu"]
    X, Y, Z = m["shape"]
    cls = np.zeros((X, Z), np.int8)                                  # 0 air
    cls[solid[:, j, :]] = 1                                          # solid
    cls[res[:, j, :]] = 2                                            # room outside
    hz = (m["heat"]["cpu"] | m["heat"]["gpu"])[:, j, :] & ~solid[:, j, :]
    cls[hz] = 6
    xs, zs = np.meshgrid(np.arange(X), np.arange(Z), indexing="ij")
    wall = (xs <= 3) | (xs >= X - 4) | (zs <= 3) | (zs >= Z - 4)
    f = forced[:, j, :]
    toward = fu[0][:, j, :] * (X / 2 - xs) + fu[2][:, j, :] * (Z / 2 - zs)
    cls[f & wall & (toward > 0)] = 3                                 # case fan in
    cls[f & wall & (toward <= 0)] = 4                                # case fan out
    cls[f & ~wall] = 5                                               # cooler / GPU fans
    shell = (xs == 1) | (xs == X - 2) | (zs == 1) | (zs == Z - 2)
    cls[shell & ~solid[:, j, :] & ~f & ~res[:, j, :]] = 7            # open to the room
    cmap = ListedColormap(["#121c28", "#4a5564", BG, CY, AM, "#9fb0c3", "#a8445a", "#6b8aa5"])
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(cls.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=7.5, interpolation="nearest", extent=[-0.5, X - 0.5, -0.5, Z - 0.5])
    plain_axes(ax, X)
    ax.set_title(f"What the solver sees: one slice of the {X} × {Y} × {Z} grid (6 mm cells)")
    ax.text(X - 3, -3, "REAR", color=MUTED, fontsize=9, va="top")
    ax.text(3, -3, "FRONT", color=MUTED, fontsize=9, va="top", ha="right")
    names = ["air", "solid (PSU, board, GPU, walls)", None, "case fan · in", "case fan · out", "cooler / GPU fans",
             "heat zone (fin stacks)", "open to the room"]
    cols = ["#121c28", "#4a5564", None, CY, AM, "#9fb0c3", "#a8445a", "#6b8aa5"]
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(cols, names) if n], loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9, handlelength=1.2, labelcolor=INK)
    save(fig, "fig_grid.png")


# ----------------------------------------------------------------------------- 2. D3Q19
def fig_d3q19():
    c = [(0, 0, 0)]
    c += [(s * (i == 0), s * (i == 1), s * (i == 2)) for i in range(3) for s in (1, -1)]
    c += [v for v in {(a, b, 0) for a in (1, -1) for b in (1, -1)} | {(a, 0, b) for a in (1, -1) for b in (1, -1)}
          | {(0, a, b) for a in (1, -1) for b in (1, -1)}]
    fig = plt.figure(figsize=(5.6, 5.2))
    ax = fig.add_subplot(projection="3d", facecolor=BG)
    ax.set_box_aspect((1, 1, 1))
    for v in c[1:]:
        col = CY if sum(map(abs, v)) == 1 else AM
        ax.quiver(0, 0, 0, *v, color=col, linewidth=1.6, arrow_length_ratio=0.12)
    ax.scatter([0], [0], [0], color=INK, s=28)
    r = [-1, 1]
    for s in r:
        for t in r:
            ax.plot(r, [s, s], [t, t], color=GRID, lw=0.8)
            ax.plot([s, s], r, [t, t], color=GRID, lw=0.8)
            ax.plot([s, s], [t, t], r, color=GRID, lw=0.8)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=35)
    ax.set_title("D3Q19: every cell keeps 19 particle populations", pad=4)
    fig.legend(handles=[Patch(color=INK, label="1 at rest"), Patch(color=CY, label="6 to the faces (speed 1)"),
                        Patch(color=AM, label="12 to the edges (speed √2)")], loc="lower center", ncol=3, fontsize=9, labelcolor=INK)
    save(fig, "fig_d3q19.png")


# ----------------------------------------------------------------------------- 3. fan operating point
def fig_operating_point():
    pid, room = DEFAULT
    cases = [("Your layout: 3 exhaust fans", HOME, (100, -20)),
             ("All three on top, blowing in: no exhaust fan", build_for(candidates()["top-front + top-mid + top-rear in"]), (20, 40))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    ps = np.linspace(-1, 14, 600)
    for ax, (title, b, off) in zip(axes, cases):
        summary = bk.load(bk.bank_key(b))["summary"]
        r, _ = steady(b, summary, room, pid)
        duty = r["case_duty"]
        m = build_model(b, LIB, case_duty=duty)
        fans = []
        for f in b["fans"]:
            s = find(LIB, "fans", f["fan"])
            sp = sc.fan_speed(duty, s.get("s_min", 0.2))
            fans.append((f["mode"], s["cfm"] * CFM_TO_M3S * sp * DERATE, s["static_mmh2o"] * MMH2O_TO_PA * sp ** 2))
        a = m["a_eff_m2"]
        q_in = np.array([sum(fan_flow(q, pm, p) for md, q, pm in fans if md == "in") for p in ps]) * M3S_TO_CFM
        q_out = np.array([sum(fan_flow(q, pm, -p) for md, q, pm in fans if md == "out")
                          + np.sign(p) * CD_ORIFICE * a * np.sqrt(2 * abs(p) / RHO_AIR) for p in ps]) * M3S_TO_CFM
        p0 = m["case_pa"]
        q0 = np.interp(p0, ps, q_in)
        ax.plot(ps, q_in, color=CY, lw=2)
        ax.plot(ps, q_out, color=AM, lw=2)
        ax.scatter([p0], [q0], s=70, color=INK, zorder=5, edgecolor=BG, linewidth=2)
        ax.annotate(f"operating point\np = +{p0:.1f} Pa, {q0:.0f} CFM in", (p0, q0), xytext=off,
                    textcoords="offset points", color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_title(f"{title}\ncase fans at {duty * 100:.0f} % (CPU-heavy game, 23 °C)", fontsize=10.5)
        ax.set_xlabel("case pressure above the room, p (Pa)")
        ax.set_xlim(ps[0], ps[-1])
        ax.set_ylim(0, None)
    axes[0].set_ylabel("air flow (CFM)")
    fig.legend(handles=[Patch(color=CY, label="air pushed in by the intake fans"), Patch(color=AM, label="air let out: exhaust fans + passive openings")],
               loc="upper right", ncol=2, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(1, 1.08))
    save(fig, "fig_operating_point.png")


# ----------------------------------------------------------------------------- 4. the bank: rise per watt against fan speed
def fig_bank():
    s = bk.load(bk.bank_key(HOME))["summary"]
    duties = np.linspace(0.08, 1.0, 120)
    st = [sc.bank_state(s, d) for d in duties]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "k_cpu_in", "Air entering the CPU cooler"), (axes[1], "k_gpu_in", "Air entering the GPU fans")):
        for j, (lab, col) in enumerate((("from 100 W of CPU heat", CPU_C), ("from 100 W of GPU heat", GPU_C))):
            ax.plot(duties * 100, [x[key][j] * 100 for x in st], color=col, lw=2, label=lab)
            ax.scatter([L["duty"] * 100 for L in s["levels"]], [L[key][j] * 100 for L in s["levels"]], s=64, color=col, edgecolor=BG, linewidth=2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("case-fan duty (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("temperature rise (°C per 100 W)")
    axes[0].annotate("simulated levels\n(12, 45, 100 %)", (12, s["levels"][0]["k_cpu_in"][1] * 100), xytext=(30, 6), textcoords="offset points",
                     color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.legend(handles=[Line2D([], [], color=CPU_C, lw=2, label="from 100 W of CPU heat"), Line2D([], [], color=GPU_C, lw=2, label="from 100 W of GPU heat")],
               loc="upper right", ncol=2, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(1, 1.07))
    fig.suptitle("The bank: three airflow simulations, interpolated in 1 / speed (your layout)", x=0.02, ha="left", y=1.08, fontsize=12.5, fontweight="bold")
    save(fig, "fig_bank.png")


# ----------------------------------------------------------------------------- 5. fan curves and the steady state
def fig_curves():
    pid, room = DEFAULT
    s = bk.load(bk.bank_key(HOME))["summary"]
    r, su = steady(HOME, s, room, pid)
    cv = su["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ts = np.linspace(20, 100, 400)
    now = {"case": (r["cpu"]["temp_c"], r["case_duty"]), "cpu": (r["cpu"]["temp_c"], r["cpu"]["fan_duty"]), "gpu": (r["gpu"]["temp_c"], r["gpu"]["fan_duty"])}
    for k, lab, col in (("case", "case fans (follow the CPU)", CASE_C), ("cpu", "CPU cooler fans", CPU_C), ("gpu", "GPU fans (follow the GPU)", GPU_C)):
        ax.plot(ts, [sc.curve_duty(cv[k]["points"], t) * 100 for t in ts], color=col, lw=2, label=lab)
        t0, d0 = now[k]
        ax.scatter([t0], [d0 * 100], s=70, color=col, edgecolor=BG, linewidth=2, zorder=5)
    ax.set_xlabel("temperature of the part the curve follows (°C)")
    ax.set_ylabel("duty (%)")
    ax.set_title("Default fan curves, and where they sit\nin the CPU-heavy game at 23 °C", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 105)
    ax = axes[1]
    pc, pg = r["cpu"]["watts"], r["gpu"]["watts"]
    tg = r["gpu"]["temp_c"]
    tt = np.linspace(40, 95, 300)
    implied = [sc._implied(s, su, room, pc, pg, t, tg)["tc"] for t in tt]
    held = [room + r["cpu"]["intake_rise_c"] + pc / sc.conductance(su["cpu"]["r_base"], su["cooler"]["r_air_full"], r["cpu"]["fan_speed"], sc.CPU_FAN_FLOOR)] * len(tt)
    ax.plot(tt, tt, color=MUTED, lw=1, label="assumed = result")
    ax.plot(tt, implied, color=CPU_C, lw=2, label="CPU temperature the fans would produce")
    ax.plot(tt, held, color=CPU_C, lw=1.5, ls=(0, (4, 3)), alpha=0.75, label="fans held at today's speed")
    tc = r["cpu"]["temp_c"]
    ax.scatter([tc], [tc], s=80, color=INK, edgecolor=BG, linewidth=2, zorder=5)
    ax.annotate(f"steady state {tc:.1f} °C\n(the PC measured 65.3 °C)", (tc, tc), xytext=(18, -34), textcoords="offset points",
                color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("assumed CPU temperature (°C)")
    ax.set_ylabel("resulting CPU temperature (°C)")
    ax.set_title("Hotter → faster fans → cooler: the solver finds\nwhere the two agree", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK)
    save(fig, "fig_curves.png")


# ----------------------------------------------------------------------------- 6. trends: room temperature and load
def fig_trends():
    pid, room0 = DEFAULT
    s = bk.load(bk.bank_key(HOME))["summary"]
    r0, su = steady(HOME, s, room0, pid)
    held = json.loads(json.dumps(su))
    for k, d in (("case", r0["case_duty"]), ("cpu", r0["cpu"]["fan_duty"]), ("gpu", r0["gpu"]["fan_duty"])):
        held["curves"][k]["points"] = [[0, d * 100], [100, d * 100]]
    p = preset(pid)
    rooms = np.arange(10, 41)
    loads = np.arange(0, 101, 5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    runs = [("room", rooms, lambda S, x: sc.solve(s, S, x, *sc.load_watts(S, p["cpu"], p["gpu"]))),
            ("load", loads, lambda S, x: sc.solve(s, S, room0, *sc.load_watts(S, p["cpu"], x / 100)))]
    for ax, (kind, xs, f) in zip(axes, runs):
        for S, ls, alpha, tag in ((su, "-", 1.0, ""), (held, (0, (4, 3)), 0.75, " (fans held)")):
            rs = [f(S, x) for x in xs]
            for part, col in (("cpu", CPU_C), ("gpu", GPU_C)):
                ax.plot(xs, [x[part]["temp_c"] for x in rs], color=col, lw=2 if not tag else 1.5, ls=ls, alpha=alpha)
        now = room0 if kind == "room" else p["gpu"] * 100
        ax.axvline(now, color=MUTED, lw=0.8)
        ax.set_xlabel("room temperature (°C)" if kind == "room" else f"GPU load (%), CPU at {p['cpu'] * 100:.0f} %")
        ax.set_title("Temperature against room temperature" if kind == "room" else "Temperature against GPU load", fontsize=11)
    a = sc.solve(s, su, 28, *sc.load_watts(su, p["cpu"], p["gpu"]))
    axes[0].annotate(f"room +5 °C: CPU +{a['cpu']['temp_c'] - r0['cpu']['temp_c']:.1f} °C\n(fans held: +5.0 °C)", (28, a["cpu"]["temp_c"]),
                     xytext=(-150, 22), textcoords="offset points", color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[0].set_ylabel("temperature (°C)")
    fig.legend(handles=[Line2D([], [], color=CPU_C, lw=2, label="CPU"), Line2D([], [], color=GPU_C, lw=2, label="GPU"),
                        Line2D([], [], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="fans held at the speed they have at 23 °C / now")],
               loc="upper right", ncol=3, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(1, 1.07))
    fig.suptitle("Your layout, CPU-heavy game: the fans absorb part of a warmer room or a heavier load", x=0.02, ha="left", y=1.08, fontsize=12.5, fontweight="bold")
    save(fig, "fig_trends.png")


# ----------------------------------------------------------------------------- 7. air speed and temperature
def fig_fields():
    pid, room = DEFAULT
    rows = ranking(pid, room)["rows"]
    rk = {x["name"]: i + 1 for i, x in enumerate(rows)}
    cur = next(x for x in rows if x["current"])
    worst = rows[-1]
    cmap_v = LinearSegmentedColormap.from_list("v", ["#0d1824", "#17698d", CY, "#d6f2ff"])
    cmap_t = LinearSegmentedColormap.from_list("t", ["#120d0a", "#7a3c14", AM, "#ffe2c2"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.6))
    for row, (name, title) in enumerate([(cur["name"], f"Your layout (#{rk[cur['name']]} of {len(rows)})"), (worst["name"], f"Worst layout (#{len(rows)}): {worst['name']}")]):
        b = layout(name)["build"]
        s, u, T, r, _ = fields_at(b, pid, room)
        geo = s["geometry"]
        j = gpu_slice_index(geo)
        m = build_model(b, LIB)
        X, Y, Z = geo["shape"]
        u = u.reshape(3, X, Y, Z)
        T = T.reshape(X, Y, Z)
        mask = (m["solid"] | m["reservoir"])[:, j, :].T
        spd = np.sqrt((u[:, :, j, :] ** 2).sum(0)).T
        tt = T[:, j, :].T
        ext = [-0.5, X - 0.5, -0.5, Z - 0.5]
        for col, (arr, cm, vmax, lab) in enumerate([(spd, cmap_v, 1.5, "air speed (m/s)"), (tt, cmap_t, 12, "air temperature above room (°C)")]):
            ax = axes[row, col]
            im = ax.imshow(np.ma.masked_where(mask, arr), origin="lower", cmap=cm, vmin=0, vmax=vmax, extent=ext, interpolation="bilinear")
            ax.imshow(np.ma.masked_where(~mask, np.ones_like(arr)), origin="lower", cmap=ListedColormap(["#3a4452"]), extent=ext, interpolation="nearest")
            if col == 0:
                st = 4
                xs, zs = np.meshgrid(np.arange(0, X, st), np.arange(0, Z, st))
                ux, uz = u[0, ::st, j, ::st].T, u[2, ::st, j, ::st].T
                ok = ~mask[::st, ::st] & (np.hypot(ux, uz) > 0.05)
                ax.quiver(xs[ok], zs[ok], ux[ok], uz[ok], color="#e8f6ff", alpha=0.55, scale=22, width=0.0025, headwidth=4)
            plain_axes(ax, X)
            ax.set_title(f"{title}\n{lab}" if col == 0 else f"\n{lab}", fontsize=10.5)
            ax.text(X - 3, -2, "REAR", color=MUTED, fontsize=8, va="top")
            ax.text(3, -2, "FRONT", color=MUTED, fontsize=8, va="top", ha="right")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            if col == 1:
                ax.text(0.02, 0.98, f"CPU {r['cpu']['temp_c']:.1f} °C   GPU {r['gpu']['temp_c']:.1f} °C   case fans {r['case_duty'] * 100:.0f} %",
                        transform=ax.transAxes, color=INK, fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec=GRID, alpha=0.85))
    fig.suptitle("Slice through the graphics card, seen through the glass (CPU-heavy game, 23 °C room)", x=0.02, ha="left", fontsize=13, fontweight="bold")
    save(fig, "fig_fields.png")


# ----------------------------------------------------------------------------- 8. all 32 layouts in three scenarios
def fig_ranking():
    scen = [("cpu_game", 23), ("gpu_game", 23), ("stress", 35)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (pid, room) in zip(axes, scen):
        rows = ranking(pid, room)["rows"]
        n_in = [sum(1 for f in layout(x["name"])["build"]["fans"] if f["fan"] == "generic-120" and f["mode"] == "in") for x in rows]
        for k in range(4):
            sel = [i for i, v in enumerate(n_in) if v == k]
            ax.scatter([rows[i]["cpu_c"] for i in sel], [rows[i]["gpu_c"] for i in sel], s=56, color=ORD[k], edgecolor=BG, linewidth=1.5,
                       label=f"{k} of your 3 fans blowing in", zorder=3)
        cur = next(i for i, x in enumerate(rows) if x["current"])
        for i, text, off in ((cur, "yours", (14, 14)), (0, "best", (-62, 14)), (len(rows) - 1, "worst", (-82, -4))):
            x = rows[i]
            ax.scatter([x["cpu_c"]], [x["gpu_c"]], s=140, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
            ax.annotate(f"#{i + 1} {text}", (x["cpu_c"], x["gpu_c"]), xytext=off, textcoords="offset points", color=INK, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        p = preset(pid)
        ax.set_title(f"{p['name']}, {room} °C room", fontsize=11)
        ax.set_xlabel("CPU temperature (°C)")
    axes[0].set_ylabel("GPU temperature (°C)")
    axes[0].legend(loc="upper left", fontsize=8.5, labelcolor=INK)
    fig.suptitle("All 32 placements of your three fans, fans following their curves (lower-left is better)", x=0.02, ha="left", fontsize=13, fontweight="bold")
    save(fig, "fig_ranking.png")


# ----------------------------------------------------------------------------- 9. where the dust goes
def fig_dust():
    pid, room = DEFAULT
    rows = ranking(pid, room)["rows"]
    cur = next(x for x in rows if x["current"])
    worst = rows[-1]
    stats = []
    for x in (cur, worst):
        s, _, _, r, near = fields_at(layout(x["name"])["build"], pid, room)
        stats.append(s["levels"][near]["dust"])
    a, b = stats
    names = [p["name"] for p in a["by_part"][:6]]
    va = [p["share"] * 100 for p in a["by_part"][:6]]
    bmap = {p["name"]: p["share"] * 100 for p in b["by_part"]}
    vb = [bmap.get(nm, 0) for nm in names]
    y = np.arange(len(names))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4))
    ax.barh(y + h / 2 + 0.01, va, height=h, color=CY, label="your layout (dust settling ×1.00)")
    ax.barh(y - h / 2 - 0.01, vb, height=h, color=AM, label=f"worst layout (dust settling ×{worst['dust_rel']:.2f})")
    for yy, v in zip(y + h / 2, va):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", color=INK, fontsize=8.5)
    for yy, v in zip(y - h / 2, vb):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlabel("share of the dust that settles inside (%)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Where 20 µm dust settles (CPU-heavy game, 23 °C)")
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK)
    save(fig, "fig_dust.png")


# ----------------------------------------------------------------------------- 10. annotated interface
UI_MARKS = [(1, 374, 88), (2, 392, 168), (3, 402, 520), (4, 1004, 152), (5, 1006, 516), (6, 1284, 330), (7, 584, 28), (8, 266, 862), (9, 742, 380), (10, 104, 402)]


def fig_ui():
    from PIL import Image, ImageDraw, ImageFont
    src = ROOT / "docs" / "screens" / "caseflow_main.png"
    im = Image.open(src).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans:bold"), 17)
    for k, x, y in UI_MARKS:
        dr.ellipse([x - 15, y - 15, x + 15, y + 15], fill=(236, 236, 236, 245), outline=(22, 22, 22, 255), width=2)
        dr.text((x, y), str(k), font=font, fill=(20, 20, 20, 255), anchor="mm")
    im.save(OUT / "ui_tour.png")
    print("wrote ui_tour.png")


if __name__ == "__main__":
    for f in (fig_grid, fig_d3q19, fig_operating_point, fig_bank, fig_curves, fig_trends, fig_fields, fig_ranking, fig_dust, fig_ui):
        f()
