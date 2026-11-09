"""Scenario model checks (no GPU): fan curves, bank interpolation, and the coupled fan / temperature solve."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import scenario as sc  # noqa: E402
from engine.sim import default_build, load_library  # noqa: E402


def synthetic_bank(duties=(0.12, 0.45, 1.0), dust=False) -> dict:
    """A bank whose per-watt rises scale exactly like 1 / speed and flows like speed (the physical scaling)."""
    levels = []
    for d in duties:
        s = sc.fan_speed(d, sc.S_MIN_REF)
        levels.append({
            "duty": d, "k_cpu_in": [0.040 / s, 0.008 / s], "k_gpu_in": [0.002 / s, 0.012 / s],
            "k_air_mean": [0.010 / s, 0.012 / s], "k_air_max": [0.06 / s, 0.05 / s],
            "flow_in_m3s": 0.060 * s, "flow_out_m3s": 0.055 * s, "case_pa": 3.0 * s * s,
            "fans": [{"slot": "front_low", "q_m3s": 0.03 * s}, {"slot": "rear", "q_m3s": 0.027 * s}],
            "dust": {"dust_entering_rel": 0.75 - 0.1 * s, "deposited_frac": 0.25 + 0.05 * s, "d_um": 20, "by_part": []} if dust else None,
        })
    return {"levels": levels}


@pytest.fixture(scope="module")
def setup():
    lib = load_library()
    b = default_build()
    return sc.setup_for(b, lib)


def flat(su, pct=60.0):
    """Same setup with every curve flat: fans at a fixed duty whatever the temperature."""
    s = copy.deepcopy(su)
    for k in ("case", "cpu", "gpu"):
        s["curves"][k]["points"] = [[30, pct], [90, pct]]
    return s


def test_curve_duty():
    pts = [[40, 14], [60, 40], [75, 100], [85, 100]]
    assert sc.curve_duty(pts, 20) == pytest.approx(0.14)       # flat below the first point
    assert sc.curve_duty(pts, 50) == pytest.approx(0.27)       # halfway between 40 and 60
    assert sc.curve_duty(pts, 95) == pytest.approx(1.0)        # flat above the last point


def test_bank_interpolation_is_exact_for_one_over_speed():
    bank = synthetic_bank()
    for d in (0.12, 0.3, 0.45, 0.7, 1.0):
        st = sc.bank_state(bank, d)
        s = sc.fan_speed(d, sc.S_MIN_REF)
        assert st["k_cpu_in"][0] == pytest.approx(0.040 / s, rel=1e-9)
        assert st["flow_in_m3s"] == pytest.approx(0.060 * s, rel=1e-9)


def test_presets_reproduce_the_measured_loads(setup):
    assert sc.load_watts(setup, 0.3, 0.0) == pytest.approx((41.8, 15.0))      # everyday use: measured 42 / 14 W
    pc, pg = sc.load_watts(setup, 0.97, 0.38)                                    # CPU-heavy game: measured 86 / 145 W
    assert pc == pytest.approx(86.0, abs=0.1) and pg == pytest.approx(146.1, abs=0.1)


def test_fixed_fans_shift_with_the_room_exactly(setup):
    """With fans that do not react, a warmer room warms everything by exactly the same amount."""
    bank, su = synthetic_bank(), flat(setup)
    a = sc.solve(bank, su, 23, 86, 145)
    b = sc.solve(bank, su, 28, 86, 145)
    assert b["cpu"]["temp_c"] - a["cpu"]["temp_c"] == pytest.approx(5.0, abs=1e-3)
    assert b["gpu"]["temp_c"] - a["gpu"]["temp_c"] == pytest.approx(5.0, abs=1e-3)


def test_fan_curves_absorb_part_of_a_warmer_room(setup):
    bank = synthetic_bank()
    a = sc.solve(bank, setup, 23, 86, 145)
    b = sc.solve(bank, setup, 28, 86, 145)
    rise = b["cpu"]["temp_c"] - a["cpu"]["temp_c"]
    assert 0.5 < rise < 4.9                                    # warmer, but by less than the room
    assert b["case_duty"] > a["case_duty"] and b["cpu"]["fan_duty"] > a["cpu"]["fan_duty"]
    assert b["noise_dba"] > a["noise_dba"]


def test_steady_state_is_self_consistent(setup):
    """At the solution the curves, applied to the solved temperatures, give back the same temperatures."""
    bank = synthetic_bank()
    r = sc.solve(bank, setup, 23, 86, 145)
    imp = sc._implied(bank, setup, 23, 86, 145, r["cpu"]["temp_c"], r["gpu"]["temp_c"])
    assert imp["tc"] == pytest.approx(r["cpu"]["temp_c"], abs=1e-3)
    assert imp["tg"] == pytest.approx(r["gpu"]["temp_c"], abs=1e-3)


def test_more_power_is_hotter_and_louder(setup):
    bank = synthetic_bank()
    lo = sc.solve(bank, setup, 23, 60, 150)
    hi = sc.solve(bank, setup, 23, 88, 330)
    assert hi["cpu"]["temp_c"] > lo["cpu"]["temp_c"] and hi["gpu"]["temp_c"] > lo["gpu"]["temp_c"]
    assert hi["gpu"]["fan_duty"] > lo["gpu"]["fan_duty"] and hi["noise_dba"] > lo["noise_dba"]


def test_gpu_fans_stop_at_idle(setup):
    r = sc.solve(synthetic_bank(), setup, 23, 30, 15)
    assert r["gpu"]["fan_speed"] == 0.0 and r["gpu"]["temp_c"] < 40.0


def test_a_part_past_its_limit_is_throttled(setup):
    su = copy.deepcopy(setup)
    su["cooler"]["r_air_full"] = 0.6                           # a far too small cooler
    r = sc.solve(synthetic_bank(), su, 35, 88, 145)
    assert r["cpu"]["throttled"] and r["cpu"]["watts"] < 88
    assert r["cpu"]["temp_c"] == pytest.approx(su["cpu"]["t_max"], abs=0.05)
    assert not r["gpu"]["throttled"]


NODE = shutil.which("node")
WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_typescript_port_agrees(setup):
    """web/src/scenario.ts (what the browser runs) must give the same answers as this module."""
    lib = load_library()
    hot = copy.deepcopy(setup)
    hot["cooler"]["r_air_full"] = 0.6                          # throttles
    by_gpu = copy.deepcopy(setup)
    by_gpu["curves"]["case"]["source"] = "gpu"
    cases = [{"bank": synthetic_bank(), "setup": su, "ambient_c": amb, "cpu_w": pc, "gpu_w": pg}
             for su in (setup, flat(setup), hot, by_gpu)
             for amb, pc, pg in ((18, 22, 15), (23, 86, 146), (28, 68, 326), (35, 88, 360))]
    cases += [{"bank": synthetic_bank(dust=True), "setup": setup, "ambient_c": amb, "cpu_w": 86, "gpu_w": 146} for amb in (18, 30)]
    mixed = default_build()
    mixed["fans"][0].update(control="fixed", pct=55)
    mixed["curves"] = {"case": {"source": "max", "points": [[30, 20], [70, 90]]}}
    builds = [default_build(), mixed]
    p = subprocess.run([NODE, str(WEB / "scripts" / "check_scenario.mjs")], cwd=WEB, capture_output=True, text=True,
                       timeout=180, input=json.dumps({"cases": cases, "setups": [{"build": b, "lib": lib} for b in builds]}))
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout)
    for c, ts in zip(cases, out["results"]):
        py = sc.solve(c["bank"], c["setup"], c["ambient_c"], c["cpu_w"], c["gpu_w"])
        for part in ("cpu", "gpu"):
            for k in ("temp_c", "watts", "fan_duty", "fan_speed", "intake_rise_c", "g_w_per_k"):
                assert ts[part][k] == pytest.approx(py[part][k], abs=1e-6), (part, k)
            assert ts[part]["throttled"] == py[part]["throttled"]
        for k in ("case_duty", "flow_in_m3s", "flow_out_m3s", "case_pa", "air_mean_rise_c", "noise_dba"):
            assert ts[k] == pytest.approx(py[k], abs=1e-6), k
        assert (ts["dust_rate"] is None) == (py["dust_rate"] is None)
        if py["dust_rate"] is not None:
            assert ts["dust_rate"] == pytest.approx(py["dust_rate"], abs=1e-12)
        assert [f["rpm"] for f in ts["fans"]] == pytest.approx([f["rpm"] for f in py["fans"]], abs=1e-6)
    for b, ts in zip(builds, out["setups"]):
        assert ts == json.loads(json.dumps(sc.setup_for(b, lib)))


def test_dust_settles_faster_with_faster_fans():
    bank = synthetic_bank(dust=True)
    assert sc.dust_rate(bank, 1.0) > sc.dust_rate(bank, 0.3) > 0
    assert sc.dust_rate(synthetic_bank(), 0.5) is None          # no dust run: no rate
