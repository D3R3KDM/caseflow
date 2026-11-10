"""The neural surrogate's plumbing (no GPU, no trained network needed): encoding, network shapes, design space."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine.sim import default_build  # noqa: E402
from engine.surrogate import (EPS_K, IN_CH, OUT_CH, CaseUNet, all_codes, build_from_code, code_of, decode,  # noqa: E402
                              encode, out_of_domain, target)


def test_target_decodes_back():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, (3, 6, 5, 4)).astype(np.float32)
    kc, kg = rng.uniform(0, 0.08, (2, 6, 5, 4)).astype(np.float32)
    u2, kc2, kg2 = decode(torch.from_numpy(target(u, kc, kg)))
    assert np.allclose(u2.numpy(), u, atol=1e-6)
    assert np.allclose(kc2.numpy(), kc, rtol=1e-5, atol=1e-9) and np.allclose(kg2.numpy(), kg, rtol=1e-5, atol=1e-9)
    assert float(decode(torch.full((5, 2, 2, 2), -3.0))[1].min()) == 0.0          # a negative output means no rise
    assert abs(float(target(u, kc, kg)[3].max()) - float(np.log1p(kc.max() / EPS_K))) < 1e-5


def test_network_keeps_any_grid_size():
    net = CaseUNet(base=8, depth=4).eval()
    for shape in ((16, 8, 16), (20, 10, 18), (17, 9, 13)):                         # odd sizes too (ceil pooling)
        with torch.no_grad():
            y = net(torch.zeros((2, IN_CH) + shape))
        assert y.shape == (2, OUT_CH) + shape


def test_encode_marks_fans_and_heat():
    from engine.sim import build_model, load_library
    m = build_model(default_build(), load_library(), case_duty=1.0)
    x = encode(m)
    assert x.shape == (IN_CH,) + tuple(m["shape"])
    assert (x[2] > 0).sum() == m["forced"].sum()                                  # fan cells
    assert np.abs(x[3:6]).max() > 0 and np.all(x[3:6][:, ~m["forced"]] == 0)     # imposed velocity only on fan cells
    assert x[6].sum() == m["heat"]["cpu"].sum() and x[7].sum() == m["heat"]["gpu"].sum()


def test_design_space():
    codes = all_codes()
    assert len(codes) == 3 ** 8 == len(set(codes)) and (0,) * 8 in codes            # the no-fan layout included
    for c in codes[::97]:
        assert code_of(build_from_code(c)) == c
        assert out_of_domain(build_from_code(c)) is None
    assert code_of(default_build()) == (1, 1, 1, 1, 0, 2, 2, 2)
    assert out_of_domain(default_build()) is None


def test_out_of_domain():
    b = default_build()
    b["gpu"] = "something-else"
    assert "gpu" in out_of_domain(b)
    b = default_build()
    b["fans"][0]["fan"] = "generic-120"
    assert "fan" in out_of_domain(b)
    b = default_build()
    b["fans"][0]["control"] = "fixed"
    assert "curve" in out_of_domain(b)


def test_test_layouts_stay_fixed():
    from gen_surrogate_data import TEST_EVERY, dataset
    small, big = dataset(40), dataset(80)
    assert [r["key"] for r in big[:len(small)]] == [r["key"] for r in small]       # growing keeps the first ones
    assert [r["split"] for r in big[:len(small)]] == [r["split"] for r in small]
    rnd = [r for r in big if r["source"] == "random"]
    assert all((r["split"] == "test") == (i % TEST_EVERY == TEST_EVERY - 1) for i, r in enumerate(rnd))
    assert all(r["split"] == "train" for r in big if r["source"] != "random")
