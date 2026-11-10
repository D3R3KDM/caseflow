"""Training data for the neural surrogate (engine/surrogate.py): banks of random fan layouts.

Design space: all 8 mounts of the reference case, each off / blowing in / blowing out - 3^8 = 6561 layouts, the one
with no case fan at all included (then only the CPU cooler's and the GPU's own fans stir the air, and the bank has a
single level). Fan models stay as on the reference PC (stock 140 mm on the front, stock 120 mm reverse-blade on the
bottom, generic 120 mm on top and at the rear) and every fan follows the case-fan curve.

The dataset is the 32 layouts of the search (engine/optimize.py, already cached) and the no-fan layout, all for
training, plus the first N layouts of a fixed random order (seed 0). Every 7th random layout is held out as a test
layout the network never sees in training, so its accuracy on new layouts can be measured; growing the dataset later
keeps the same test layouts.

Each layout's bank (airflow + per-watt heat + dust at three case-fan duties, ~90 s on the GPU) is cached in
data/banks, so the run can be stopped at any time (Ctrl+C, or end the python process) and resumes where it stopped.

    python scripts/gen_surrogate_data.py --target 150     # 32 + the no-fan layout + 117 random layouts
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine.optimize import build_for, candidates  # noqa: E402
from engine.sim import default_build  # noqa: E402

OUT = ROOT / "data" / "surrogate"
MOUNT_FAN = {"front_low": "lianli-207-front-140", "front_high": "lianli-207-front-140",
             "bottom_front": "lianli-207-bottom-120-rev", "bottom_rear": "lianli-207-bottom-120-rev",
             "top_1": "generic-120", "top_2": "generic-120", "top_3": "generic-120", "rear": "generic-120"}
MOUNTS = list(MOUNT_FAN)
MODES = (None, "in", "out")          # code per mount: 0 off, 1 in, 2 out
TEST_EVERY = 7


def build_from_code(code: tuple[int, ...]) -> dict:
    b = default_build()
    b["fans"] = [{"slot": s, "fan": MOUNT_FAN[s], "mode": MODES[c], "control": "curve", "pct": 70}
                 for s, c in zip(MOUNTS, code) if c]
    return b


def code_of(build: dict) -> tuple[int, ...]:
    m = {f["slot"]: f["mode"] for f in build["fans"] if f.get("mode") in ("in", "out")}
    return tuple(MODES.index(m[s]) if s in m else 0 for s in MOUNTS)


def random_order() -> list[tuple[int, ...]]:
    codes = list(itertools.product(range(3), repeat=len(MOUNTS)))
    return random.Random(0).sample(codes, len(codes))


def dataset(n_random: int) -> list[dict]:
    """The 32 search layouts and the no-fan layout (all training) + the first n_random random ones (every 7th held
    out as test)."""
    search = [code_of(build_for(spec)) for spec in candidates().values()]
    rows = [{"code": list(c), "source": "search", "split": "train"} for c in search]
    none = (0,) * len(MOUNTS)
    if none not in search:
        rows.append({"code": list(none), "source": "no fans", "split": "train"})
    seen = {tuple(r["code"]) for r in rows}
    k = 0
    for c in random_order():
        if k >= n_random:
            break
        if c in seen:
            continue
        seen.add(c)
        rows.append({"code": list(c), "source": "random", "split": "test" if k % TEST_EVERY == TEST_EVERY - 1 else "train"})
        k += 1
    for r in rows:
        r["key"] = bk.bank_key(build_from_code(tuple(r["code"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150, help="layouts in the dataset, the 32 search layouts included")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = dataset(0)
    rows = dataset(max(args.target - len(rows), 0))
    (OUT / "dataset.json").write_text(json.dumps({"mounts": MOUNTS, "mount_fan": MOUNT_FAN, "rows": rows}, indent=1))
    todo = [r for r in rows if bk.load(r["key"]) is None]
    print(f"dataset: {len(rows)} layouts ({sum(r['split'] == 'test' for r in rows)} test), {len(todo)} to simulate "
          f"(~{len(todo) * 90 / 3600:.1f} h)", flush=True)
    t0, failed = time.time(), 0
    for i, r in enumerate(todo):
        t1 = time.time()
        code = "".join("-io"[c] for c in r["code"])
        try:
            b = bk.get(build_from_code(tuple(r["code"])))
            assert b["summary"]["key"] == r["key"]
            lv = b["summary"]["levels"]
            info = f"k_cpu_in {lv[0]['k_cpu_in'][0]:.4f}..{lv[-1]['k_cpu_in'][0]:.4f} K/W, {b['summary']['thermal_iters']} heat iters"
            if b["summary"].get("lbm_limiter_ms"):
                info += f", velocity limiter {b['summary']['lbm_limiter_ms']:.1f} m/s"
        except Exception as e:                       # one bad layout must not stop the run
            failed += 1
            info = f"FAILED: {type(e).__name__}: {e}"
        left = (len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 60
        print(f"[gen {i + 1:3d}/{len(todo)}] {code} {r['split']:5s} {time.time() - t1:4.0f} s, ~{left:.0f} min left | {info}", flush=True)
    print(f"GEN DONE: {len(rows) - failed} of {len(rows)} layouts with banks, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
