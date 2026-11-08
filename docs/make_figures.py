"""Figures for docs/GUIDE.md, made from the engine and the precomputed results (no GPU run).

    python docs/make_figures.py        # from the caseflow folder; writes docs/media/*.png

Chart colours were checked with the dataviz palette validator on the #0b1119 surface:
categorical cyan #1f9fd0 / amber #d9772b (all checks pass), ordinal cyan ramp
#a9e2f7 #5cc0e8 #1f8fbf #17698d (all checks pass).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.optimize import build_for, candidates  # noqa: E402
from engine.sim import (CD_ORIFICE, CFM_TO_M3S, DERATE, DX, MMH2O_TO_PA, RHO_AIR, build_model,  # noqa: E402
                        default_build, fan_flow, find, load_library)

OUT = ROOT / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "web" / "public" / "data"

BG, INK, MUTED, GRID = "#0b1119", "#dde9f3", "#8b98a9", "#1c2735"
CY, AM = "#1f9fd0", "#d9772b"
ORD = ["#a9e2f7", "#5cc0e8", "#1f8fbf", "#17698d"]
M3S_TO_CFM = 1 / CFM_TO_M3S
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def load_view(key):
    d = json.loads((DATA / f"{key}.json").read_text())
    X, Y, Z = d["shape"]
    u = np.frombuffer(base64.b64decode(d["u"]), dtype=np.float16).astype(np.float32).reshape(3, X, Y, Z)
    T = np.frombuffer(base64.b64decode(d["T"]), dtype=np.float16).astype(np.float32).reshape(X, Y, Z)
    return d, u, T


def gpu_slice_index(d):
    gb = next(b for b in d["boxes"] if b["kind"] == "gpu")
    return int(round((gb["b"][2] + gb["b"][3]) / 2 / d["dx_mm"] + d["off"]))


# ----------------------------------------------------------------------------- 1. how the grid sees the case
def fig_grid():
    b = default_build()
    m = build_model(b)
    d, _, _ = load_view("current")
    j = gpu_slice_index(d)
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
    ax.imshow(cls.T, origin="lower", cmap=cmap, vmin=-0.5, vmax=7.5, interpolation="nearest",
              extent=[-0.5, X - 0.5, -0.5, Z - 0.5])
    ax.invert_xaxis()                                                # through the glass: front on the right
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
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
                        Patch(color=AM, label="12 to the edges (speed √2)")], loc="lower center", ncol=3, fontsize=9,
               labelcolor=INK)
    save(fig, "fig_d3q19.png")


# ----------------------------------------------------------------------------- 3. fan operating point
def fig_operating_point():
    lib = load_library()
    cases = [("Your layout: 3 exhaust fans", default_build(), (100, -20)),
             ("All three on top, blowing in: no exhaust fan", build_for(candidates()["top-front + top-mid + top-rear in"]), (20, 40))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    ps = np.linspace(-1, 14, 600)
    for ax, (title, b, off) in zip(axes, cases):
        m = build_model(b, lib)
        fans = []
        for f in b["fans"]:
            s = find(lib, "fans", f["fan"])
            fr = f["pct"] / 100
            fans.append((f["mode"], s["cfm"] * CFM_TO_M3S * fr * DERATE, s["static_mmh2o"] * MMH2O_TO_PA * fr ** 2))
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
        pi = 12.5
        ax.text(pi, np.interp(pi, ps, q_in) + 10, "pushed in", color=CY, fontsize=9, va="bottom", ha="center")
        po = pi if off[1] < 0 else 3.0
        ax.text(po, np.interp(po, ps, q_out) - 6, "let out", color=AM, fontsize=9, va="top", ha="center")
        ax.set_title(title)
        ax.set_xlabel("case pressure above the room, p (Pa)")
        ax.set_xlim(ps[0], ps[-1])
        ax.set_ylim(0, None)
    axes[0].set_ylabel("air flow (CFM)")
    fig.legend(handles=[Patch(color=CY, label="air pushed in by the intake fans"), Patch(color=AM, label="air let out: exhaust fans + passive openings")],
               loc="upper right", ncol=2, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(1, 1.04))
    save(fig, "fig_operating_point.png")


# ----------------------------------------------------------------------------- 4. air speed and temperature
def fig_fields():
    lib = load_library()
    cmap_v = LinearSegmentedColormap.from_list("v", ["#0d1824", "#17698d", CY, "#d6f2ff"])
    cmap_t = LinearSegmentedColormap.from_list("t", ["#120d0a", "#7a3c14", AM, "#ffe2c2"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.6))
    for row, (key, title) in enumerate([("current", "Your layout (#1 of 32)"), ("worst", "Worst layout (#32): rear fan blowing in")]):
        d, u, T = load_view(key)
        j = gpu_slice_index(d)
        m = build_model(d["build"], lib)
        X, Y, Z = d["shape"]
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
            ax.invert_xaxis()
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{title}\n{lab}" if col == 0 else f"\n{lab}", fontsize=10.5)
            ax.text(X - 3, -2, "REAR", color=MUTED, fontsize=8, va="top")
            ax.text(3, -2, "FRONT", color=MUTED, fontsize=8, va="top", ha="right")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            p = d["parts"]
            if col == 1:
                ax.text(0.02, 0.98, f"CPU {p['cpu']['temp_c']:.1f} °C   GPU {p['gpu']['temp_c']:.1f} °C", transform=ax.transAxes,
                        color=INK, fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec=GRID, alpha=0.85))
    fig.suptitle("Slice through the graphics card, seen through the glass", x=0.02, ha="left", fontsize=13, fontweight="bold")
    save(fig, "fig_fields.png")


# ----------------------------------------------------------------------------- 5. all 32 layouts
def fig_ranking():
    opt = json.loads((ROOT / "data" / "optimize.json").read_text())
    rows = opt["ranking"]
    n_in = [sum(1 for f in r["build"]["fans"] if f["fan"] == "generic-120" and f["mode"] == "in") for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 6))
    for k in range(4):
        sel = [i for i, v in enumerate(n_in) if v == k]
        ax.scatter([rows[i]["cpu_c"] for i in sel], [rows[i]["gpu_c"] for i in sel], s=70, color=ORD[k], edgecolor=BG, linewidth=1.5,
                   label=f"{k} of your 3 fans blowing in", zorder=3)
    lo, hi, ylo, yhi = 74.0, 77.6, 62.9, 66.3
    for s in (138, 139, 140, 141, 142):
        xs = np.linspace(lo, hi, 2)
        ax.plot(xs, s - xs, color=GRID, lw=0.9, ls=(0, (4, 3)), zorder=1)
        if s - lo < yhi - 0.1:                                       # label where the line leaves the left edge ...
            ax.text(lo + 0.03, s - lo + 0.03, f"Σ {s}", color=MUTED, fontsize=8, ha="left", va="bottom")
        else:                                                        # ... or the top edge
            ax.text(s - yhi + 0.06, yhi - 0.05, f"Σ {s}", color=MUTED, fontsize=8, ha="left", va="top")
    marks = {r["name"]: i for i, r in enumerate(rows)}
    notes = [(next(i for i, r in enumerate(rows) if r.get("current")), "#1  your layout: all three out", (-12, -30)),
             (marks[opt["cpu"]], "#5  lowest CPU: 2 top fans in", (-12, -100)),
             (marks["top-front + top-mid + top-rear in"], "#14  all three in (fans stall)", (14, 22)),
             (marks[opt["worst"]], "#32  worst: rear fan in", (-120, -46))]
    for i, text, off in notes:
        r = rows[i]
        ax.scatter([r["cpu_c"]], [r["gpu_c"]], s=150, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
        ax.annotate(f"{text}\n{r['cpu_c']:.1f} / {r['gpu_c']:.1f} °C · dust {r['dust_in'] * 100:.0f}%", (r["cpu_c"], r["gpu_c"]),
                    xytext=off, textcoords="offset points", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("CPU temperature (°C)")
    ax.set_ylabel("GPU temperature (°C)")
    ax.set_title("All 32 placements of your three fans (lower-left is better; dashed = equal CPU + GPU)")
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK)
    save(fig, "fig_ranking.png")


# ----------------------------------------------------------------------------- 6. where the dust goes
def fig_dust():
    a = json.loads((DATA / "current.json").read_text())["dust"]
    b = json.loads((DATA / "worst.json").read_text())["dust"]
    names = [p["name"] for p in a["by_part"][:6]]
    va = [p["share"] * 100 for p in a["by_part"][:6]]
    bmap = {p["name"]: p["share"] * 100 for p in b["by_part"]}
    vb = [bmap.get(nm, 0) for nm in names]
    y = np.arange(len(names))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4))
    ax.barh(y + h / 2 + 0.01, va, height=h, color=CY, label=f"your layout · {a['dust_entering_rel'] * 100:.0f}% of outside dust gets in")
    ax.barh(y - h / 2 - 0.01, vb, height=h, color=AM, label=f"worst layout · {b['dust_entering_rel'] * 100:.0f}% gets in")
    for yy, v in zip(y + h / 2, va):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", color=INK, fontsize=8.5)
    for yy, v in zip(y - h / 2, vb):
        ax.text(v + 0.4, yy, f"{v:.0f}%", va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlabel("share of the dust that settles inside (%)")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_title("Where 20 µm dust settles")
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK)
    save(fig, "fig_dust.png")


# ----------------------------------------------------------------------------- 7. annotated interface
def fig_ui():
    from PIL import Image, ImageDraw, ImageFont
    src = ROOT / "docs" / "screens" / "caseflow_main.png"
    im = Image.open(src).convert("RGB")
    dr = ImageDraw.Draw(im, "RGBA")
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans:bold"), 17)
    pts = [(1, 92, 290), (2, 388, 188), (3, 398, 543), (4, 1000, 148), (5, 1000, 538), (6, 1278, 336), (7, 562, 28), (8, 340, 860), (9, 742, 385), (10, 985, 470)]
    for k, x, y in pts:
        dr.ellipse([x - 15, y - 15, x + 15, y + 15], fill=(31, 159, 208, 235), outline=(232, 246, 255, 255), width=2)
        dr.text((x, y), str(k), font=font, fill=(5, 8, 12, 255), anchor="mm")
    im.save(OUT / "ui_tour.png")
    print("wrote ui_tour.png")


if __name__ == "__main__":
    fig_grid()
    fig_d3q19()
    fig_operating_point()
    fig_fields()
    fig_ranking()
    fig_dust()
    fig_ui()
