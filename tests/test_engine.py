"""Fast checks of the CaseFlow engine (no GPU run): fan operating point and model assembly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.optimize import build_for, candidates, is_current  # noqa: E402
from engine.sim import build_model, default_build, fan_flow, operating_point  # noqa: E402


def test_fan_curve_limits():
    assert fan_flow(1.0, 10.0, 0.0) == 1.0
    assert fan_flow(1.0, 10.0, 10.0) == 0.0        # stalled at its static pressure
    assert fan_flow(1.0, 10.0, -100.0) == 1.25     # helped, but capped


def test_operating_point_mass_balance():
    import math
    fans = [("in", 0.03, 15.0), ("in", 0.03, 15.0), ("out", 0.02, 10.0)]
    a = 0.01
    p, q = operating_point(fans, a)
    passive = math.copysign(0.6 * a * math.sqrt(2 * abs(p) / 1.2), p)
    assert p > 0                                     # more intake than exhaust -> positive pressure
    assert abs(q[0] + q[1] - q[2] - passive) < 1e-6


def test_all_intake_is_throttled():
    m_cur = build_model(default_build())
    m_in = build_model(build_for(candidates()["top-front + top-mid + top-rear in"]))
    q = lambda m, mode: sum(f["q_m3s"] for f in m["fans"] if f["mode"] == mode)  # noqa: E731
    assert m_in["case_pa"] > 3 * m_cur["case_pa"] > 0
    assert q(m_in, "in") < q(m_cur, "in")


def test_search_space():
    c = candidates()
    assert len(c) == 32
    assert sum(is_current(build_for(s)) for s in c.values()) == 1
