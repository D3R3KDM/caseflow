"""Figures of the neural surrogate for docs/GUIDE.md: training, accuracy on held-out layouts, learning curve, the
predicted fields next to the simulated ones, what the network finds over all 6561 layouts, and the dust tracked in its
airflow against the simulation's (after scripts/ai_dust_check.py).

    python docs/make_ai_figures.py        # from the caseflow folder, after scripts/train_surrogate.py and scripts/ai_space.py

Colours: CPU blue #3987e5 / GPU orange #d95926 (validated pair, as in make_figures.py); train / test from the
validated ordinal cyan ramp (#5cc0e8 dashed / #1f8fbf solid); differences on the reference diverging pair blue <->
red with the dark grey #383835 at zero.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
from make_figures import (BG, CPU_C, CY, AM, GPU_C, GRID, INK, LIB, MUTED, ORD, gpu_slice_index, level_weights,  # noqa: E402
                          plain_axes, preset, save)

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import build_model  # noqa: E402
from engine.surrogate import Surrogate, build_from_code  # noqa: E402

SUR = ROOT / "data" / "surrogate"
DIV = LinearSegmentedColormap.from_list("div", ["#9cc3f5", "#3987e5", "#383835", "#e66767", "#f5b0b0"])
TRAIN_C, TEST_C = ORD[1], ORD[2]


def load(name):
    p = SUR / name
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- 1. training
def fig_training():
    hist = load("caseflow_unet_history.json")
    if not hist:
        return print("skip fig_ai_training (no history)")
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    total = [h["train_u"] + h["train_k"] + h["train_probe"] for h in hist]
    a.plot(ep, total, color=CY, lw=2)
    a.set_yscale("log")
    a.set_xlabel("epoch")
    a.set_ylabel("training loss (log scale)")
    a.set_title("Training loss")
    ev = [h for h in hist if "test" in h]
    for key, color, ls, lab in (("train_fields", TRAIN_C, "--", "training layouts"), ("test", TEST_C, "-", "held-out test layouts")):
        xs = [h["epoch"] for h in ev if key in h]
        ys = [h[key]["intake_mae_kw"] * 1000 for h in ev if key in h]
        b.plot(xs, ys, color=color, lw=2, ls=ls, label=lab)
        b.annotate(f"{lab}  {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    b.set_xlabel("epoch")
    b.set_ylabel("mean error at the CPU / GPU intakes (mK/W)")
    b.set_ylim(bottom=0)
    b.set_xlim(0, ep[-1] * 1.32)
    b.set_title("Error of the per-watt rise at the intakes")
    b.legend(loc="upper right")
    save(fig, "fig_ai_training.png")


# ----------------------------------------------------------------------------- 2. accuracy on held-out layouts
def fig_parity():
    ev = load("caseflow_unet_eval.json")
    if not ev or not ev["final"]["temps"]:
        return print("skip fig_ai_parity (no eval)")
    t = ev["final"]["temps"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    pts = {"cpu": ([], []), "gpu": ([], [])}
    for L in t["layouts"]:
        for s in L["scenarios"]:
            for part in ("cpu", "gpu"):
                pts[part][0].append(s[f"{part}_sim"])
                pts[part][1].append(s[f"{part}_ai"])
    lo = min(min(v[0] + v[1]) for v in pts.values()) - 2
    hi = max(max(v[0] + v[1]) for v in pts.values()) + 2
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
    for part, color, lab in (("cpu", CPU_C, "CPU"), ("gpu", GPU_C, "GPU")):
        a.scatter(*pts[part], s=34, color=color, edgecolor=BG, linewidth=1.2, label=lab, zorder=3)
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.set_aspect("equal")
    a.set_xlabel("simulation (°C)")
    a.set_ylabel("neural network (°C)")
    a.set_title(f"{t['n_layouts']} held-out layouts × 6 scenarios")
    a.legend(loc="upper left")
    errs = {p: np.array(pts[p][1]) - np.array(pts[p][0]) for p in pts}
    bins = np.linspace(-max(abs(e).max() for e in errs.values()) - 0.05, max(abs(e).max() for e in errs.values()) + 0.05, 25)
    for part, color, lab in (("cpu", CPU_C, "CPU"), ("gpu", GPU_C, "GPU")):
        b.hist(errs[part], bins=bins, histtype="step", color=color, linewidth=2,
               label=f"{lab}: mean |error| {np.abs(errs[part]).mean():.2f} °C, worst {np.abs(errs[part]).max():.2f}")
    b.axvline(0, color=MUTED, lw=1, ls=":")
    b.set_xlabel("network − simulation (°C)")
    b.set_ylabel("layout × scenario cases")
    b.set_title("Error")
    b.set_ylim(0, b.get_ylim()[1] * 1.4)              # room for the legend above the bars
    b.legend(loc="upper left")
    base = t.get("baseline")
    if base:
        b.text(0.98, 0.62, f"for comparison, guessing\nthe average layout:\nCPU {base['cpu_mae_c']:.2f} °C, GPU {base['gpu_mae_c']:.2f} °C",
               transform=b.transAxes, color=MUTED, fontsize=9, va="top", ha="right")
    save(fig, "fig_ai_parity.png")


# ----------------------------------------------------------------------------- 3. learning curve
def fig_learning():
    runs = []
    for p in sorted(SUR.glob("caseflow_unet*_eval.json")):
        e = json.loads(p.read_text())
        if e.get("final", {}).get("temps"):
            runs.append((e["trained"]["layouts"], e["final"]["temps"], e["final"]["test"]))
    if len(runs) < 2:
        return print("skip fig_ai_learning (fewer than 2 runs)")
    runs.sort(key=lambda r: r[0])
    n = [r[0] for r in runs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for part, color, lab in (("cpu", CPU_C, "CPU"), ("gpu", GPU_C, "GPU")):
        ys = [r[1][f"{part}_mae_c"] for r in runs]
        a.plot(n, ys, color=color, lw=2, marker="o", ms=7, mec=BG, mew=1.5, label=f"{lab}, network")
        a.annotate(f"{lab} {ys[-1]:.2f} °C", (n[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
        base = [r[1].get("baseline", {}).get(f"{part}_mae_c") for r in runs]
        if all(v is not None for v in base):
            a.plot(n, base, color=color, lw=1.4, ls="--", alpha=0.8, label=f"{lab}, guessing the average layout")
    a.set_xlabel("training layouts (× 3 fan speeds)")
    a.set_ylabel("mean error on the 16 held-out layouts (°C)")
    a.set_ylim(bottom=0)
    a.set_xticks(n)
    a.set_title("Part temperatures")
    a.legend(loc="lower left", fontsize=9)
    a.margins(x=0.18)
    ys = [r[2]["intake_rel"] * 100 for r in runs]
    b.plot(n, ys, color=CY, lw=2, marker="o", ms=7, mec=BG, mew=1.5)
    for x, y in zip(n, ys):
        b.annotate(f"{y:.0f} %", (x, y), xytext=(0, 9), textcoords="offset points", color=INK, fontsize=9, ha="center")
    b.set_xlabel("training layouts (× 3 fan speeds)")
    b.set_ylabel("error of the air rise at the CPU / GPU intakes (%)")
    b.set_ylim(0, max(ys) * 1.3)
    b.set_xticks(n)
    b.set_title("Intake air rise (what the temperatures come from)")
    b.margins(x=0.18)
    fig.suptitle("More simulated layouts, better predictions (same 16 test layouts; one training run each)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.28)
    save(fig, "fig_ai_learning.png")


# ----------------------------------------------------------------------------- 4. fields: simulation vs network
def composed(summary, fields, r):
    lv = summary["levels"]
    T = sum(w * (r["cpu"]["watts"] * np.asarray(fields[k]["kc"], np.float32) + r["gpu"]["watts"] * np.asarray(fields[k]["kg"], np.float32))
            for k, w in level_weights(lv, r["case_duty"]))
    near = min(range(len(lv)), key=lambda k: abs(sc.fan_speed(lv[k]["duty"] or 0, sc.S_MIN_REF) - sc.fan_speed(r["case_duty"], sc.S_MIN_REF)))
    return np.asarray(fields[near]["u"], np.float32), np.clip(T, 0, None)


def fig_fields():
    ev = load("caseflow_unet_eval.json")
    ds = load("dataset.json")
    if not ev or not ds:
        return print("skip fig_ai_fields")
    test = [r for r in ds["rows"] if r["split"] == "test" and bk.load(r["key"]) is not None]
    if not test:
        return print("skip fig_ai_fields (no test layout simulated)")
    # the held-out layout with the median CPU error in the CPU-heavy game
    errs = {L["key"]: abs(next(s for s in L["scenarios"] if s["scenario"] == "cpu_game")["cpu_ai"]
                          - next(s for s in L["scenarios"] if s["scenario"] == "cpu_game")["cpu_sim"]) for L in ev["final"]["temps"]["layouts"]}
    test.sort(key=lambda r: errs.get(r["key"], 9))
    row = test[len(test) // 2]
    build = build_from_code(tuple(row["code"]))
    sim = bk.load(row["key"])
    ai = Surrogate().predict(build)
    su = sc.setup_for(build, LIB)
    p = preset("cpu_game")
    pc, pg = sc.load_watts(su, p["cpu"], p["gpu"])
    res = {}
    for name, B in (("sim", sim), ("ai", ai)):
        r = sc.solve(B["summary"], su, 23, pc, pg)
        res[name] = (r,) + composed(B["summary"], B["fields"], r)
    geo = sim["summary"]["geometry"]
    j = gpu_slice_index(geo)
    X, Y, Z = geo["shape"]
    m = build_model(build, LIB)
    mask = (m["solid"] | m["reservoir"])[:, j, :].T
    cmap_v = LinearSegmentedColormap.from_list("v", ["#0d1824", "#17698d", CY, "#d6f2ff"])
    cmap_t = LinearSegmentedColormap.from_list("t", ["#120d0a", "#7a3c14", AM, "#ffe2c2"])
    ext = [-0.5, X - 0.5, -0.5, Z - 0.5]
    code = "".join("-io"[c] for c in row["code"])
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 15.2))
    sl = {}
    for name in ("sim", "ai"):
        r, u, T = res[name]
        u = u.reshape(3, X, Y, Z)
        sl[name] = (np.sqrt((u[:, :, j, :] ** 2).sum(0)).T, T.reshape(X, Y, Z)[:, j, :].T, u)
    rows = [("Simulation (lattice Boltzmann + heat transport, ~90 s)", "sim"), ("Neural network (3D U-Net, ~0.1 s)", "ai"), ("Network − simulation", None)]
    for i, (title, name) in enumerate(rows):
        for col in range(2):
            ax = axes[i, col]
            if name:
                arr = sl[name][col]
                cm, vmin, vmax = (cmap_v, 0, 1.5) if col == 0 else (cmap_t, 0, 12)
            else:
                arr = sl["ai"][col] - sl["sim"][col]
                lim = 0.6 if col == 0 else 3.0
                cm, vmin, vmax = DIV, -lim, lim
            im = ax.imshow(np.ma.masked_where(mask, arr), origin="lower", cmap=cm, vmin=vmin, vmax=vmax, extent=ext, interpolation="bilinear")
            ax.imshow(np.ma.masked_where(~mask, np.ones_like(arr)), origin="lower", cmap=ListedColormap(["#3a4452"]), extent=ext, interpolation="nearest")
            if name and col == 0:
                u = sl[name][2]
                st = 4
                xs, zs = np.meshgrid(np.arange(0, X, st), np.arange(0, Z, st))
                ux, uz = u[0, ::st, j, ::st].T, u[2, ::st, j, ::st].T
                ok = ~mask[::st, ::st] & (np.hypot(ux, uz) > 0.05)
                ax.quiver(xs[ok], zs[ok], ux[ok], uz[ok], color="#e8f6ff", alpha=0.55, scale=22, width=0.0025, headwidth=4)
            plain_axes(ax, X)
            lab = ["air speed (m/s)", "air temperature above room (°C)"][col]
            if not name:
                lab += " · red: network higher, blue: lower"
            ax.set_title(f"{title}\n{lab}" if col == 0 else f"\n{lab}", fontsize=10.5)
            ax.text(X - 3, -2, "REAR", color=MUTED, fontsize=8, va="top")
            ax.text(3, -2, "FRONT", color=MUTED, fontsize=8, va="top", ha="right")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            if name and col == 1:
                r = res[name][0]
                ax.text(0.02, 0.98, f"CPU {r['cpu']['temp_c']:.1f} °C   GPU {r['gpu']['temp_c']:.1f} °C", transform=ax.transAxes, color=INK,
                        fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec=GRID, alpha=0.85))
    fig.suptitle(f"A held-out layout the network never trained on: {code}\n(front low/high, bottom front/rear, top front/mid/rear, rear; "
                 "i in, o out, - off)\nslice through the graphics card, CPU-heavy game, 23 °C room", x=0.02, y=0.995, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.9, hspace=0.32)
    save(fig, "fig_ai_fields.png")


# ----------------------------------------------------------------------------- 5. the whole design space
def fig_space():
    p = ROOT / "web" / "public" / "data" / "ai_space.json"
    if not p.exists():
        return print("skip fig_ai_space (no ai_space.json)")
    space = json.loads(p.read_text())
    pr = preset("cpu_game")
    out = []
    for row in space["rows"]:
        code = tuple(int(c) for c in row["c"])
        build = build_from_code(code)
        slots = [s for s, c in zip(space["mounts"], code) if c]
        levels = []
        for j, v in enumerate(row["L"]):
            levels.append({"duty": space["levels"][j] if slots else None, "k_cpu_in": v[0:2], "k_gpu_in": v[2:4], "k_air_mean": v[4:6],
                           "k_air_max": v[6:8], "flow_in_m3s": v[8], "flow_out_m3s": v[9], "case_pa": v[10],
                           "fans": [{"slot": s, "q_m3s": q} for s, q in zip(slots, v[11])], "dust": None})
        su = sc.setup_for(build, LIB)
        pc, pg = sc.load_watts(su, pr["cpu"], pr["gpu"])
        r = sc.solve({"levels": levels}, su, 23, pc, pg)
        out.append((sum(code[i] > 0 for i in range(8)), r["cpu"]["temp_c"], r["gpu"]["temp_c"], row["c"]))
    n = np.array([o[0] for o in out])
    cpu = np.array([o[1] for o in out])
    gpu = np.array([o[2] for o in out])
    home = "11110222"
    ih = next(i for i, o in enumerate(out) if o[3] == home)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    seq = plt.get_cmap("Blues_r")          # one hue; on the dark surface few fans recede, many fans stand out
    for k in range(9):
        sel = n == k
        a.scatter(cpu[sel], gpu[sel], s=7, color=seq(0.12 + 0.73 * k / 8), alpha=0.8, linewidth=0, label=f"{k}" if k in (0, 2, 4, 6, 8) else None)
    a.scatter([cpu[ih]], [gpu[ih]], s=110, facecolor="none", edgecolor=INK, linewidth=1.8, zorder=4)
    a.annotate("your layout", (cpu[ih], gpu[ih]), xytext=(cpu.max() - 0.1, gpu.min() + 0.5), textcoords="data", color=INK, fontsize=9.5,
               ha="right", va="bottom", arrowprops=dict(arrowstyle="-", color=INK, lw=1), bbox=dict(boxstyle="square,pad=0.3", fc=BG, ec=GRID), zorder=5)
    a.set_xlabel("CPU (°C)")
    a.set_ylabel("GPU (°C)")
    a.set_title(f"All {len(out)} layouts, predicted (CPU-heavy game, 23 °C room)")
    a.legend(title="fans", loc="upper left", markerscale=3, title_fontsize=9)
    ks = list(range(9))
    best = [min((i for i in range(len(out)) if n[i] == k), key=lambda i: cpu[i] + gpu[i]) for k in ks]
    top = min(range(len(out)), key=lambda i: cpu[i] + gpu[i])
    for arr, color, lab in ((cpu, CPU_C, "CPU"), (gpu, GPU_C, "GPU")):
        ys = [arr[i] - arr[top] for i in best]
        b.plot(ks, ys, color=color, lw=2, marker="o", ms=6, mec=BG, mew=1.5, label=lab)
        b.annotate(f"{lab} +{ys[-1]:.1f}", (ks[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    b.axhline(0, color=MUTED, lw=1, ls=":")
    b.set_xlabel("number of case fans")
    b.set_ylabel("best layout with that many fans,\n°C above the best of all 6561")
    b.set_title("Diminishing returns")
    b.set_xticks(ks)
    b.legend(loc="upper right")
    b.margins(x=0.12)
    save(fig, "fig_ai_space.png")
    order = sorted(range(len(out)), key=lambda i: cpu[i] + gpu[i])
    print("best overall:", out[order[0]], "| yours: #", order.index(ih) + 1, out[ih])
    for k in (6, 7):
        i = best[k]
        print(f"best with {k} fans: {out[i]}")


# ----------------------------------------------------------------------------- 6. dust in the network's airflow
def fig_dust():
    d = load("ai_dust_check.json")
    if not d:
        return print("skip fig_ai_dust (run scripts/ai_dust_check.py)")
    s, rows = d["summary"], d["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.32)
    L = [x for r in rows for x in r["levels"]]
    xs, ys = [x["settles_sim"] * 100 for x in L], [x["settles_ai"] * 100 for x in L]
    lo, hi = min(xs + ys) - 2, max(xs + ys) + 2
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
    a.scatter(xs, ys, s=34, color=TEST_C, edgecolor=BG, linewidth=1.2, zorder=3)
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.set_aspect("equal")
    a.set_xlabel("simulation (%)")
    a.set_ylabel("tracked in the network's airflow (%)")
    a.set_title(f"Share of the dust that settles\n{s['samples']} held-out cases, mean |diff| {s['settles_mae_pts']:.1f} pts")
    rs, ra = np.array([r["rate_sim"] for r in rows]), np.array([r["rate_ai"] for r in rows])
    ref = rs.mean()
    lo, hi = min(rs.min(), ra.min()) / ref - 0.1, max(rs.max(), ra.max()) / ref + 0.1
    b.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
    b.scatter(rs / ref, ra / ref, s=40, color=TEST_C, edgecolor=BG, linewidth=1.2, zorder=3)
    b.set_xlim(lo, hi)
    b.set_ylim(lo, hi)
    b.set_aspect("equal")
    b.set_xlabel("simulation (× the average layout)")
    b.set_ylabel("network's airflow (× the average layout)")
    b.set_title(f"Dust rate, CPU-heavy game, 23 °C\nmean |diff| {s['rate_rel_err_mean_pct']:.0f}%, rank corr. {s['rate_rank_spearman']:.2f}")
    save(fig, "fig_ai_dust.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["training", "parity", "learning", "fields", "space", "dust"]
    for w in which:
        globals()[f"fig_{w}"]()
