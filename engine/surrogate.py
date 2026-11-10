"""Neural surrogate: a 3D U-Net that predicts a build's airflow and heat fields from its voxel model in milliseconds,
trained on CaseFlow's own simulations (data: scripts/gen_surrogate_data.py, training: scripts/train_surrogate.py).

One sample is one build at one case-fan duty (engine/sim.build_model), on the 80 x 40 x 80 grid of 6 mm cells.
Input  [B, 8, X, Y, Z] (+ 3 coordinate channels the network adds itself):
       solid, outside air, fan cells, the fans' imposed air velocity (3 channels, m/s / U0), CPU and GPU heat zones.
Output [B, 5, X, Y, Z]: air velocity (3 channels, m/s / U0) and the air temperature rise per watt of CPU power and per
       watt of GPU power, as log1p(k / EPS_K) so that a cool corner and a hot heat sink both count.
The fans' operating points (flow against the case pressure) stay analytical, as in the simulation; the network learns
where that air goes and how it carries the heat. The physics (LBM + heat transport, about 90 s per build) stays the
reference: the app shows the network's answer at once and the simulation's, with the difference, when it finishes.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sim import U_LAT_PER_MS, build_model, default_build, load_library

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "surrogate" / "caseflow_unet.pt"
U0 = 2.0                  # m/s: velocity scale of the network's input and output
EPS_K = 0.005             # K/W: rises are learned as log1p(k / EPS_K)
IN_CH, OUT_CH = 8, 5

# ----------------------------------------------------------------------------- the design space it was trained on
# The reference PC's parts; each of the 8 mounts off / in / out with the fan model that sits there on that PC (the
# case's stock 140 mm on the front, its stock 120 mm reverse-blade on the bottom, a generic 120 mm on top and rear),
# every fan on the case-fan curve: 3^8 = 6561 layouts.
DOMAIN = {"case": "lianli-lancool-207", "gpu": "nvidia-rtx-5080-fe", "cooler": "thermalright-pa120-se-argb"}
MOUNT_FAN = {"front_low": "lianli-207-front-140", "front_high": "lianli-207-front-140",
             "bottom_front": "lianli-207-bottom-120-rev", "bottom_rear": "lianli-207-bottom-120-rev",
             "top_1": "generic-120", "top_2": "generic-120", "top_3": "generic-120", "rear": "generic-120"}
MOUNTS = list(MOUNT_FAN)
MODES = (None, "in", "out")          # code per mount: 0 off, 1 in, 2 out


def build_from_code(code) -> dict:
    """The build of a layout code (one digit per mount in MOUNTS order: 0 off, 1 in, 2 out)."""
    b = default_build()
    b["fans"] = [{"slot": s, "fan": MOUNT_FAN[s], "mode": MODES[c], "control": "curve", "pct": 70}
                 for s, c in zip(MOUNTS, code) if c]
    return b


def code_of(build: dict) -> tuple[int, ...]:
    m = {f["slot"]: f["mode"] for f in build["fans"] if f.get("mode") in ("in", "out")}
    return tuple(MODES.index(m[s]) if s in m else 0 for s in MOUNTS)


def all_codes() -> list[tuple[int, ...]]:
    return list(itertools.product(range(3), repeat=len(MOUNTS)))


def out_of_domain(build: dict) -> str | None:
    """Why the network cannot be trusted for this build (None if it was trained on builds like it)."""
    for k, v in DOMAIN.items():
        if build.get(k) != v:
            return f"the network was trained on the reference PC's {k} ({v}) only"
    for f in build["fans"]:
        if f.get("mode") not in ("in", "out"):
            continue
        if f["slot"] not in MOUNT_FAN:
            return f"no training data for a fan at {f['slot']}"
        if f["fan"] != MOUNT_FAN[f["slot"]]:
            return f"the network was trained with the {MOUNT_FAN[f['slot']]} fan at {f['slot']}"
        if f.get("control", "curve") != "curve":
            return "the network was trained with every case fan on the case-fan curve"
    return None


# ----------------------------------------------------------------------------- encoding
def encode(m: dict) -> np.ndarray:
    """Network input of one voxel model: [8, X, Y, Z] float32."""
    fu = m["fu"] / U_LAT_PER_MS / U0
    return np.concatenate([np.stack([m["solid"], m["reservoir"], m["forced"]]).astype(np.float32), fu.astype(np.float32),
                           np.stack([m["heat"]["cpu"], m["heat"]["gpu"]]).astype(np.float32)])


def target(u_ms: np.ndarray, kc: np.ndarray, kg: np.ndarray) -> np.ndarray:
    """Training target of one simulated level: [5, X, Y, Z] float32."""
    k = np.stack([kc, kg]).astype(np.float32).clip(0, None)
    return np.concatenate([u_ms.astype(np.float32) / U0, np.log1p(k / EPS_K)])


def decode(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Network output [5, X, Y, Z] (or [B, 5, ...]) -> air velocity m/s [3, ...], CPU and GPU rise per watt K/W."""
    out = out.float()
    k = EPS_K * torch.expm1(out[..., 3:5, :, :, :].clamp(min=0))
    return out[..., 0:3, :, :, :] * U0, k[..., 0, :, :, :], k[..., 1, :, :, :]


# ----------------------------------------------------------------------------- network
class Block(nn.Module):
    """Two 3x3x3 convolutions with group norm and GELU, plus a residual path."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(h)) + self.skip(x))


class CaseUNet(nn.Module):
    """3D U-Net. Four halvings take the 80 x 40 x 80 grid to 5 x 2 x 5, so the deepest features see the whole case
    (a fan on the top changes the flow at the front); the skip connections keep the 6 mm detail."""

    def __init__(self, cin: int = IN_CH, cout: int = OUT_CH, base: int = 24, depth: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.config = {"cin": cin, "cout": cout, "base": base, "depth": depth}
        self.inc = Block(cin + 3, chs[0])
        self.downs = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(depth))])
        self.out = nn.Conv3d(chs[0], cout, 1)

    @staticmethod
    def coords(x: torch.Tensor) -> torch.Tensor:
        B, _, X, Y, Z = x.shape
        ax = [torch.linspace(-1, 1, n, device=x.device, dtype=x.dtype) for n in (X, Y, Z)]
        return torch.stack(torch.meshgrid(*ax, indexing="ij"))[None].expand(B, 3, X, Y, Z)

    def forward(self, x):
        h = self.inc(torch.cat([x, self.coords(x)], 1))
        skips = []
        for down in self.downs:
            skips.append(h)
            h = down(F.max_pool3d(h, 2, ceil_mode=True))
        for up in self.ups:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[2:], mode="trilinear", align_corners=False)
            h = up(torch.cat([h, s], 1))
        return self.out(h)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------------------- inference
class Surrogate:
    """The trained network, ready to predict banks (engine/bank.py format) for any build of the reference case."""

    def __init__(self, path: Path = CHECKPOINT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = CaseUNet(**ck["config"])
        self.net.load_state_dict({k: v.float() for k, v in ck["state"].items()})
        self.net.to(self.device).eval()
        self.info = {k: v for k, v in ck.items() if k not in ("state",)}

    @torch.no_grad()
    def fields(self, models: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """(u m/s [3, X, Y, Z], k_cpu K/W, k_gpu K/W) for each voxel model, in one batch."""
        x = torch.from_numpy(np.stack([encode(m) for m in models])).to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            out = self.net(x)
        res = []
        for k, m in enumerate(models):
            u, kc, kg = decode(out[k])
            air = torch.from_numpy(~m["solid"]).to(self.device)
            inner = air & ~torch.from_numpy(m["reservoir"]).to(self.device)
            res.append((u * air, kc * inner, kg * inner))
        return res

    def predict(self, build: dict, with_dust: bool = False) -> dict:
        """The build's bank as the network sees it: same summary and fields as engine.bank.compute, marked "ai"."""
        from engine import bank as bk
        from engine.sim import deposit_points, simulate_dust
        t0 = time.time()
        lib = load_library()
        lv = list(bk.LEVELS) if bk.curve_fans(build) else [None]
        models = [build_model(build, lib, case_duty=d) for d in lv]
        levels, fields = [], []
        for d, m, (u, kc, kg) in zip(lv, models, self.fields(models)):
            dust = simulate_dust(u, m) if with_dust else None
            levels.append(bk.level_summary(d, m, u, kc, kg, dust))
            fields.append({"u": u.half().cpu().numpy(), "kc": kc.half().cpu().numpy(), "kg": kg.half().cpu().numpy(),
                           "dust_points": np.asarray(deposit_points(dust["dep"], m) if dust else [], np.float32).reshape(-1, 3)})
        key = bk.bank_key(build)
        seen = "train" if key in self.info.get("train_keys", ()) else "test" if key in self.info.get("test_keys", ()) else "new"
        summary = {"version": bk.BANK_VERSION, "key": key, "levels": levels, "curve_fans": bk.curve_fans(build),
                   "source": "ai", "seen": seen, "seconds": time.time() - t0, "geometry": bk.geometry(models[0]),
                   "model": {k: self.info.get(k) for k in ("name", "params", "trained", "test")}}
        return {"summary": summary, "fields": fields}
