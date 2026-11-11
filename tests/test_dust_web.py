"""The browser's dust (web/src/dust.ts) runs the engine's particles: on your layout's bank it must settle the same share
of the dust, on the same surfaces, as engine.sim.simulate_dust did when the bank was made. The 3D view animates these
particles, so this is what keeps the animation honest. No GPU: node runs the TypeScript directly."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine.sim import build_model, default_build, dust_by_part

NODE = shutil.which("node")
WEB = Path(__file__).resolve().parents[1] / "web"
PKG = WEB / "public" / "data" / "bank_current.json"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_dust_settles_like_the_engine():
    pkg = json.loads(PKG.read_text())
    if "voxels" not in pkg:
        pytest.skip("bank_current.json has no voxels: run scripts/precompute.py")
    p = subprocess.run([NODE, str(WEB / "scripts" / "check_dust.mjs"), "40000"], cwd=WEB, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout)
    shape = tuple(out["shape"])
    for level, eng in zip(out["levels"], pkg["levels"]):
        e = eng["dust"]
        assert level["settled_frac"] == pytest.approx(e["deposited_frac"], abs=0.03)
        dep = np.zeros(int(np.prod(shape)))
        for i, c in level["cells"]:
            dep[i] = c
        shares = {q["name"]: q["share"] for q in dust_by_part(dep.reshape(shape), build_model(default_build(), case_duty=eng["duty"]))}
        for q in e["by_part"][:5]:                    # where most of it lands
            assert shares.get(q["name"], 0.0) == pytest.approx(q["share"], abs=0.03), q["name"]
