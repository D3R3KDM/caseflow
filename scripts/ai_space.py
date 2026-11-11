"""The neural surrogate over the whole design space: all 3^8 = 6561 layouts of the reference case (each of the 8
mounts off / in / out, engine/surrogate.py) in a few minutes. Simulating them all would take 6561 x ~90 s, about a week
of GPU time.

Writes web/public/data/ai_space.json: per layout, at each case-fan duty, the per-watt rises at the CPU and GPU intakes
and in the case air, the flows and each fan's flow - what the browser's scenario solver (web/src/scenario.ts) needs to
rank every layout for any room temperature, load and fan curves. Dust is left out (it needs particle tracking). Also
lists the layouts the network was trained on and the held-out test layouts (data/surrogate/dataset.json), so the app
can say whether the network has seen a layout and offer the test layouts.

    python scripts/ai_space.py
    python scripts/ai_space.py --splits-only     # only refresh the training / test lists in the existing file
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine.sim import build_model, load_library  # noqa: E402
from engine.surrogate import MOUNT_FAN, MOUNTS, Surrogate, all_codes, build_from_code  # noqa: E402

OUT = ROOT / "web" / "public" / "data" / "ai_space.json"
RISES = ("k_cpu_in", "k_gpu_in", "k_air_mean", "k_air_max")


def sig(x: float, n: int = 4) -> float:
    return float(f"{x:.{n}g}")


def splits() -> dict:
    """Codes of the layouts the network was trained on and of the held-out test layouts."""
    ds = json.loads((ROOT / "data" / "surrogate" / "dataset.json").read_text())
    return {k: sorted("".join(map(str, r["code"])) for r in ds["rows"] if r["split"] == k) for k in ("train", "test")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-only", action="store_true", help="only refresh the training / test lists in the existing file")
    if ap.parse_args().splits_only:
        out = json.loads(OUT.read_text())
        out.update(splits())
        OUT.write_text(json.dumps(out, separators=(",", ":")))
        return print(f"wrote the training / test lists ({len(out['train'])} / {len(out['test'])}) into {OUT.name}")
    t0 = time.time()
    net = Surrogate()
    lib = load_library()
    codes = all_codes()
    jobs = []                                   # (layout index, duty) per network sample
    for i, c in enumerate(codes):
        for d in (bk.LEVELS if any(c) else (None,)):
            jobs.append((i, d))
    rows = [{"c": "".join(map(str, c)), "L": []} for c in codes]
    batch = 24
    for s in range(0, len(jobs), batch):
        part = jobs[s:s + batch]
        models = [build_model(build_from_code(codes[i]), lib, case_duty=d) for i, d in part]
        for (i, d), m, (u, kc, kg) in zip(part, models, net.fields(models)):
            L = bk.level_summary(d, m, u, kc, kg, None)
            rows[i]["L"].append([sig(v) for k in RISES for v in L[k]]
                                + [sig(L["flow_in_m3s"]), sig(L["flow_out_m3s"]), sig(L["case_pa"]), [sig(f["q_m3s"]) for f in L["fans"]]])
        if (s // batch) % 100 == 0:
            done = s + len(part)
            print(f"[ai {done:5d}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    torch.cuda.empty_cache()
    info = {k: net.info.get(k) for k in ("name", "params", "config", "trained", "test", "baseline")}
    out = {"version": 1, "model": info, "mounts": MOUNTS, "mount_fan": MOUNT_FAN, "levels": list(bk.LEVELS),
           "row": "per level: k_cpu_in[2], k_gpu_in[2], k_air_mean[2], k_air_max[2] (K/W; CPU heat, GPU heat), "
                  "flow_in_m3s, flow_out_m3s, case_pa, [q_m3s of each fan in mount order]; code digit per mount: 0 off, 1 in, 2 out",
           "seconds": round(time.time() - t0, 1), **splits(), "rows": rows}
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"AI SPACE DONE: {len(codes)} layouts, {len(jobs)} network samples in {time.time() - t0:.0f} s, "
          f"{OUT.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
