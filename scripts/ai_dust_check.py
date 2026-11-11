"""Dust in the network's airflow against the simulation's, on the 16 held-out test layouts.

The network predicts air velocity and heat, not dust. The app gets the AI's dust by running the simulation's own
particle tracking (engine/sim.simulate_dust: 40 000 particles of 20 um, same seed) in the predicted air velocity
(Surrogate.dust, /api/ai/dust). So the two differ only through the airflow. For every test layout and case-fan level this
compares how much of the dust that gets in settles, where it settles, and - in a scenario - the dust rate the app ranks
layouts by. Needs the test layouts' simulated banks (data/banks, from scripts/gen_surrogate_data.py) and the GPU,
about 8 s per layout. Writes data/surrogate/ai_dust_check.json.

    python scripts/ai_dust_check.py                  # CPU-heavy game, 23 degC room
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine import scenario as sc  # noqa: E402
from engine.sim import load_library  # noqa: E402
from engine.surrogate import Surrogate, build_from_code  # noqa: E402

OUT = ROOT / "data" / "surrogate" / "ai_dust_check.json"


def ranks(x: list[float]) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cpu_game")
    ap.add_argument("--room", type=float, default=23.0)
    args = ap.parse_args()
    lib = load_library()
    pr = next(p for p in lib["defaults"]["scenarios"] if p["id"] == args.scenario)
    ds = json.loads((ROOT / "data" / "surrogate" / "dataset.json").read_text())
    tests = [r for r in ds["rows"] if r["split"] == "test"]
    net = Surrogate()
    t0 = time.time()
    rows = []
    for r in tests:
        b = build_from_code(tuple(r["code"]))
        sim = bk.load(bk.bank_key(b))
        if sim is None:
            print("no simulated bank for", r["code"], "- run scripts/gen_surrogate_data.py")
            continue
        ai = net.dust(b)
        s_sum = sim["summary"]
        a_sum = copy.deepcopy(s_sum)                      # the same bank, with the dust tracked in the AI's airflow
        for L, d in zip(a_sum["levels"], ai["levels"]):
            L["dust"] = d["dust"]
        su = sc.setup_for(b, lib)
        pc, pg = sc.load_watts(su, pr["cpu"], pr["gpu"])
        rate = [sc.solve(x, su, args.room, pc, pg)["dust_rate"] for x in (s_sum, a_sum)]
        levels = []
        for Ls, La in zip(s_sum["levels"], a_sum["levels"]):
            ds_, da = Ls["dust"], La["dust"]
            if not ds_ or not da:
                continue
            share_s = {p["name"]: p["share"] for p in ds_["by_part"]}
            share_a = {p["name"]: p["share"] for p in da["by_part"]}
            top_s, top_a = ds_["by_part"][0]["name"], da["by_part"][0]["name"]
            levels.append({"duty": Ls["duty"], "settles_sim": ds_["deposited_frac"], "settles_ai": da["deposited_frac"],
                           "enter_sim": ds_["dust_entering_rel"], "enter_ai": da["dust_entering_rel"],
                           "top_sim": top_s, "top_ai": top_a, "top_share_sim": share_s[top_s], "top_share_ai": share_a.get(top_s, 0.0),
                           # how far apart the two "where it lands" lists are: half the summed share difference (0 = same)
                           "where_diff": 0.5 * sum(abs(share_s.get(k, 0) - share_a.get(k, 0)) for k in set(share_s) | set(share_a))})
        code = "".join(str(c) for c in r["code"])
        rows.append({"code": code, "fans": sum(c != 0 for c in r["code"]), "rate_sim": rate[0], "rate_ai": rate[1], "levels": levels})
        lv = [L for L in levels if L["duty"] in (None, 0.45)] or levels
        print(f"{code}  settles sim {lv[0]['settles_sim']:.3f} ai {lv[0]['settles_ai']:.3f}  top {lv[0]['top_sim']:<12} {lv[0]['top_ai']:<12}"
              f"  rate sim {rate[0]:.4f} ai {rate[1]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    L = [x for r in rows for x in r["levels"]]
    d_settle = np.array([x["settles_ai"] - x["settles_sim"] for x in L]) * 100
    d_where = np.array([x["where_diff"] for x in L]) * 100
    rs, ra = [r["rate_sim"] for r in rows], [r["rate_ai"] for r in rows]
    rel = np.abs(np.array(ra) / np.array(rs) - 1) * 100
    rho = float(np.corrcoef(ranks(rs), ranks(ra))[0, 1])
    # the app shows dust relative to a reference layout: how often do two layouts compare the same way?
    pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    same = sum((rs[i] - rs[j]) * (ra[i] - ra[j]) > 0 for i, j in pairs) / len(pairs)
    summary = {
        "layouts": len(rows), "samples": len(L), "scenario": args.scenario, "room_c": args.room,
        "settles_mae_pts": float(np.abs(d_settle).mean()), "settles_max_pts": float(np.abs(d_settle).max()),
        "where_diff_mean_pts": float(d_where.mean()), "where_diff_max_pts": float(d_where.max()),
        "top_part_same": float(np.mean([x["top_sim"] == x["top_ai"] for x in L])),
        "rate_rel_err_mean_pct": float(rel.mean()), "rate_rel_err_max_pct": float(rel.max()),
        "rate_rank_spearman": rho, "rate_pairs_same_order": float(same), "seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
