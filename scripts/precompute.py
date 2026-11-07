"""Precompute the web views from data/optimize.json (written by engine/optimize.py):
current (the user's own layout), best (recommended), coolest and cleanest (least dust) - the last two only when
they differ from the views before them. Stale view files are removed so the page never shows an old result."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.sim import LIB_PATH, default_build, package, simulate_build  # noqa: E402

OUT = ROOT / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
opt = json.loads((ROOT / "data" / "optimize.json").read_text())
assert not opt.get("partial"), "optimize.json is partial - run engine/optimize.py to the end first"
row = {x["name"]: x for x in opt["ranking"]}
cur = next(x for x in opt["ranking"] if x.get("current"))
jobs = [("current", f"Your layout: {cur['name']}", default_build(), cur["name"])]
seen = {cur["name"]}
for key, title in (("best", "Recommended"), ("coolest", "Coolest"), ("cleanest", "Least dust"),
                   ("cpu", "Lowest CPU"), ("worst", f"Worst of {len(opt['ranking'])}")):
    name = opt.get(key)
    if key != "best" and (not name or name in seen):
        (OUT / f"{key}.json").unlink(missing_ok=True)
        continue
    seen.add(name)
    jobs.append((key, f"{title}: {name}", row[name]["build"], name))
cache = {}
for key, label, b, name in jobs:
    r = cache.get(name) or simulate_build(b, with_dust=True)      # best == current: reuse the run
    cache[name] = r
    data = package(r, label=label)
    data["build"] = b
    (OUT / f"{key}.json").write_text(json.dumps(data))
    print(key, round(r["parts"]["cpu"]["temp_c"], 1), round(r["parts"]["gpu"]["temp_c"], 1),
          f"dust {r['dust']['dust_entering_rel'] * 100:.0f}%", flush=True)
(OUT / "optimize.json").write_text(json.dumps(opt))
(OUT / "library.json").write_text(LIB_PATH.read_text(encoding="utf-8"), encoding="utf-8")   # static fallback for the web app
print("PRECOMPUTE PC DONE", flush=True)
