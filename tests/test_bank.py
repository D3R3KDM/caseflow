"""Bank checks that need no GPU: what the cache key depends on, and how build_model applies the fan duty."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.bank import bank_key, curve_fans  # noqa: E402
from engine.sim import build_model, default_build  # noqa: E402


def test_key_ignores_what_the_solver_handles():
    """CPU, curves and load do not change the airflow, so they must not force a new bank."""
    b = default_build()
    b2 = copy.deepcopy(b)
    b2.update(cpu="amd-ryzen-9-9950x", load={"cpu_w": 200, "gpu_w": 50, "ambient_c": 35},
              curves={"case": {"source": "gpu", "points": [[30, 20], [80, 100]]}})
    b2["fans"][0]["pct"] = 40                     # unused while the fan follows the curve
    assert bank_key(b2) == bank_key(b)


def test_key_follows_the_airflow():
    b = default_build()
    flipped = copy.deepcopy(b)
    flipped["fans"][-1]["mode"] = "in"
    fixed = copy.deepcopy(b)
    fixed["fans"][0]["control"] = "fixed"
    fixed_faster = copy.deepcopy(fixed)
    fixed_faster["fans"][0]["pct"] = 90
    keys = {bank_key(x) for x in (b, flipped, fixed, fixed_faster)}
    assert len(keys) == 4


def test_duty_reaches_curve_fans_only():
    b = default_build()
    b["fans"][0].update(control="fixed", pct=50)
    m = build_model(b, case_duty=1.0)
    f0, f1 = m["fans"][0], m["fans"][1]
    assert f0["duty"] == 0.5 and abs(f0["speed"] - (0.26 + 0.74 * 0.5)) < 1e-9     # its own 50 %, s_min 0.26
    assert f1["duty"] == 1.0 and f1["speed"] == 1.0
    assert curve_fans(b) == [f["slot"] for f in b["fans"][1:]]
