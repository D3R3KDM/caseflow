"""Write what the web app needs to run without the engine (web/public/data):

- library.json        the parts library
- default_build.json  the reference PC (the user's layout)
- bank_current.json   its bank with the 3D fields (airflow, per-watt heat, dust at three case-fan duties) and its
                      voxels (for the dust animation), so the room, load and curve sliders work offline in 3D
- optimize.json       the 32-layout search: every layout's bank summary, so the ranking is redone in the browser
                      for any scenario (other layouts open in 3D when the engine is running; it has them cached)

The banks come from data/banks (engine/optimize.py computes them); a missing one is computed here (~1.5 min).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import bank as bk  # noqa: E402
from engine.sim import LIB_PATH, default_build  # noqa: E402

OUT = ROOT / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
for old in ("current", "best", "cpu", "worst", "coolest", "cleanest"):     # fixed-fan views of the old model
    (OUT / f"{old}.json").unlink(missing_ok=True)
(OUT / "library.json").write_text(LIB_PATH.read_text(encoding="utf-8"), encoding="utf-8")
home = default_build()
(OUT / "default_build.json").write_text(json.dumps(home, indent=1))
b = bk.get(home, callback=lambda f: print(f"\r  bank {f * 100:5.1f} %", end="", flush=True))
(OUT / "bank_current.json").write_text(json.dumps(bk.package(b) | {"voxels": bk.voxels(home)}))
print(f"\rbank_current.json: {(OUT / 'bank_current.json').stat().st_size / 1e6:.1f} MB")
opt = ROOT / "data" / "optimize.json"
if opt.exists() and json.loads(opt.read_text()).get("version", 0) >= 7:
    (OUT / "optimize.json").write_text(json.dumps(json.loads(opt.read_text())))
    print(f"optimize.json: {(OUT / 'optimize.json').stat().st_size / 1e3:.0f} kB")
else:
    print("data/optimize.json is missing or from the old model: run engine/optimize.py first")
print("PRECOMPUTE DONE", flush=True)
