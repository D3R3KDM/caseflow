"""Builds the interactive explainer page, docs/presentation/explainer.html, from explainer_template.html and the
project's own numbers (no GPU: everything comes from files the project already has):

- every layout of the 8 mounts: the network's CPU / GPU temperature in the CPU-heavy game at 23 degC, solved from
  web/public/data/ai_space.json exactly like the browser does;
- the simulation's temperature for every layout the engine has cached (data/banks);
- the training / held-out split, and the network's error on each of those 150 layouts in six scenarios;
- your layout's bank and setup, for the fan-curve demo (the page runs the same steady-state solve);
- one slice of your layout's voxel model, for the U-Net demo.

    python docs/presentation/make_explainer.py          # from the caseflow folder
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import FILTER_EFF, build_model, default_build, load_library  # noqa: E402
from engine.surrogate import MOUNT_FAN, MOUNTS, all_codes, build_from_code  # noqa: E402

HERE = Path(__file__).parent
LIB = load_library()
SPACE = json.loads((ROOT / "web" / "public" / "data" / "ai_space.json").read_text())
DS = json.loads((ROOT / "data" / "surrogate" / "dataset.json").read_text())
SCEN = [("cpu_game", 23.0), ("office", 23.0), ("gpu_game", 23.0), ("gpu_game", 30.0), ("stress", 23.0), ("stress", 35.0)]
PRESET = {p["id"]: p for p in LIB["defaults"]["scenarios"]}
FAN_MM = {f["id"]: f["size_mm"] for f in LIB["fans"]}
ZH = {"idle": "待机", "office": "日常", "cpu_game": "吃 CPU 的游戏", "gpu_game": "吃显卡的游戏", "stress": "满载"}


def row_bank(row: dict) -> dict:
    """A layout's bank summary from its ai_space row (as scripts/ai_check.py and the browser rebuild it)."""
    slots = [s for s, c in zip(SPACE["mounts"], row["c"]) if c != "0"]
    return {"levels": [{"duty": SPACE["levels"][j] if slots else None, "k_cpu_in": v[0:2], "k_gpu_in": v[2:4], "k_air_mean": v[4:6],
                        "k_air_max": v[6:8], "flow_in_m3s": v[8], "flow_out_m3s": v[9], "case_pa": v[10],
                        "fans": [{"slot": s, "q_m3s": q} for s, q in zip(slots, v[11])], "dust": None} for j, v in enumerate(row["L"])]}


def temps(summary: dict, build: dict, pid: str, room: float) -> tuple[float, float, bool]:
    su = sc.setup_for(build, LIB)
    pc, pg = sc.load_watts(su, PRESET[pid]["cpu"], PRESET[pid]["gpu"])
    r = sc.solve(summary, su, room, pc, pg)
    return r["cpu"]["temp_c"], r["gpu"]["temp_c"], r["cpu"]["throttled"] or r["gpu"]["throttled"]


def main() -> None:
    codes = ["".join(map(str, c)) for c in all_codes()]
    rows = {r["c"]: r for r in SPACE["rows"]}
    builds = {c: build_from_code(tuple(int(x) for x in c)) for c in codes}
    ai_c, ai_g, thr = [], [], []
    for i, c in enumerate(codes):
        t = temps(row_bank(rows[c]), builds[c], "cpu_game", 23.0)
        ai_c.append(round(t[0] * 10))
        ai_g.append(round(t[1] * 10))
        if t[2]:
            thr.append(i)
        if i % 1000 == 0:
            print(f"ai {i}/{len(codes)}", flush=True)
    sim, sim_sum = {}, {}
    for c in codes:
        js = bk.CACHE / f"{bk.bank_key(builds[c])}.json"
        if js.exists():
            sim_sum[c] = json.loads(js.read_text())
            t = temps(sim_sum[c], builds[c], "cpu_game", 23.0)
            sim[c] = [round(t[0], 2), round(t[1], 2)]
    split = {"train": [], "test": []}
    err = {"train": [], "test": []}
    for r in DS["rows"]:
        c = "".join(map(str, r["code"]))
        split[r["split"]].append(c)
        if c not in sim_sum:
            continue
        e = []
        for pid, room in SCEN:
            a = temps(row_bank(rows[c]), builds[c], pid, room)
            s = temps(sim_sum[c], builds[c], pid, room)
            e.append([round(a[0] - s[0], 2), round(a[1] - s[1], 2)])
        err[r["split"]].append({"c": c, "e": e})
    # your layout: its simulated bank and setup, for the page's own steady-state solve
    home = default_build()
    hb = json.loads((bk.CACHE / f"{bk.bank_key(home)}.json").read_text())
    su = sc.setup_for(home, LIB)
    # one x-z slice of your layout's voxel model through the graphics card: 0 air, 1 solid, 2 outside, 3 fan, 4 heat zone
    m = build_model(home, LIB, case_duty=0.45)
    gb = next(b for b in m["boxes"] if b["kind"] == "gpu")
    j = m["grid"].idx((gb["b"][2] + gb["b"][3]) / 2)
    X, Y, Z = m["shape"]
    cells = []
    for k in range(Z - 1, -1, -1):
        for i in range(X):
            v = 2 if m["reservoir"][i, j, k] else 3 if m["forced"][i, j, k] else 4 if (m["heat"]["cpu"][i, j, k] or m["heat"]["gpu"][i, j, k]) else 1 if m["solid"][i, j, k] else 0
            cells.append(str(v))
    case = next(c for c in LIB["cases"] if c["id"] == home["case"])
    data = {
        "case": case["dims_mm"],
        # a mount's fan size (the fan the network's layouts put there) and the opening when it is empty
        "mountsGeo": [{"slot": mt["slot"], "wall": mt["wall"], "a": mt["a"], "b": mt["b"],
                       "size": FAN_MM[MOUNT_FAN[mt["slot"]]] if FAN_MM[MOUNT_FAN[mt["slot"]]] in mt["sizes"] else max(mt["sizes"]),
                       "open": max(mt["sizes"])} for mt in case["mounts"]],
        "parts": [{"kind": b["kind"], "b": b["b"]} for b in m["boxes"] if b["kind"] in ("psu", "board", "gpu", "gpu_hs", "gpu_fan", "cpu_hs", "cpu_fan", "vent")],
        # empty mounts on mesh walls (and the rear) stay open; the bottom has the dust filter
        "open_walls": [w for w, p in case["panels"].items() if "mesh" in p] + ["rear"],
        "filters": case.get("filters", {}), "filter_eff": FILTER_EFF,
        "mounts": SPACE["mounts"], "codes_order": "itertools.product(range(3), repeat=8): 0 off, 1 in, 2 out",
        "ai_cpu10": ai_c, "ai_gpu10": ai_g, "throttled": thr, "sim": sim, "split": split,
        "err": {"scenarios": [f"{ZH[p]} · {room:g} °C" for p, room in SCEN], **err},
        "home": "".join(map(str, (1, 1, 1, 1, 0, 2, 2, 2))),
        "bank": {"levels": [{"duty": L["duty"], "k_cpu_in": L["k_cpu_in"], "k_gpu_in": L["k_gpu_in"]} for L in hb["levels"]]},
        "setup": {k: su[k] for k in ("cpu", "cooler", "gpu", "curves")},
        "presets": {p: {"name": ZH.get(p, PRESET[p]["name"]), "cpu": PRESET[p]["cpu"], "gpu": PRESET[p]["gpu"]} for p in PRESET},
        "consts": {"S_MIN_REF": sc.S_MIN_REF, "H_EXP": sc.H_EXP, "CPU_FAN_FLOOR": sc.CPU_FAN_FLOOR},
        "slice": {"nx": X, "nz": Z, "cells": "".join(cells), "dx_mm": 6},
    }
    assert data["home"] == "".join(map(str, sc_code(home))), "your layout's code"
    blob = json.dumps(data, separators=(",", ":"))
    tpl = (HERE / "explainer_template.html").read_text(encoding="utf-8")
    assert tpl.count("__CF_DATA__") == 1
    (HERE / "explainer.html").write_text(tpl.replace("__CF_DATA__", blob), encoding="utf-8")
    print(f"explainer.html: {len(blob) / 1e3:.0f} kB of data, {len(sim)} simulated layouts, {len(thr)} throttle")


def sc_code(build: dict) -> tuple[int, ...]:
    from engine.surrogate import code_of
    return code_of(build)


if __name__ == "__main__":
    main()
