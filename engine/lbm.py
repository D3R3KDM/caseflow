"""D3Q19 lattice Boltzmann solver for a virtual wind tunnel (PyTorch, GPU).

Written from scratch for this project. Everything is in lattice units.

Coordinates: x = streamwise (inlet at x=0), y = lateral, z = vertical.
Tensor layout: f[B, 19, X, Y, Z] so that several geometries can run as a batch.

Collision: BGK with a Smagorinsky sub-grid model (Hou et al. 1996).
Boundaries:
  * inlet  (x=0):    equilibrium with prescribed velocity, density from x=1
  * outlet (x=X-1):  zero-gradient copy, plus a sponge layer of increased viscosity
  * sides (y, z):    periodic (falls out of torch.roll)
  * obstacle:        halfway bounce-back; force by momentum exchange
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

# --- lattice -----------------------------------------------------------------
C_LIST = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
]
W_LIST = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
OPP_LIST = [C_LIST.index((-a, -b, -c)) for (a, b, c) in C_LIST]
Q = 19


@dataclass
class LBMConfig:
    u_in: float = 0.06          # inlet velocity (lattice units)
    tau: float = 0.55           # molecular relaxation time -> nu = (tau - 0.5) / 3
    smagorinsky: float = 0.1    # Smagorinsky constant (0 disables the LES model)
    sponge: int = 16            # sponge layer thickness at the outlet (cells)
    tau_sponge: float = 1.0     # relaxation time reached at the outlet
    u_max: float | None = None  # velocity limiter (lattice units): faster cells relax towards u_max (off by default)
    dtype: torch.dtype = torch.float32

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0


@dataclass
class LBMResult:
    u: torch.Tensor             # [B, 3, X, Y, Z] time-averaged velocity
    p: torch.Tensor             # [B, X, Y, Z]   time-averaged gauge pressure (rho - 1) / 3
    cd: torch.Tensor            # [B] time-averaged drag coefficient
    cl: torch.Tensor            # [B, 2] time-averaged lateral / vertical force coefficients
    cd_history: list = field(default_factory=list)   # list of (step, [B] Cd)
    area: torch.Tensor | None = None                  # [B] frontal area (cells)
    steps: int = 0
    seconds: float = 0.0
    mlups: float = 0.0
    mass_drift: torch.Tensor | None = None            # [B] relative change of total mass (fluid cells)


class LBMSolver:
    def __init__(self, solid: torch.Tensor, cfg: LBMConfig | None = None, device: str | torch.device = "cuda",
                 inlet_profile=None, top_lid: bool = False, forced_mask=None, forced_u=None,
                 reservoir_mask=None, open_inlet: bool = True, open_shell: bool = False):
        """Optional extras (all off by default, the free-stream tunnel is unchanged):
        inlet_profile  [Z] streamwise inlet velocity per height (atmospheric boundary layer)
        top_lid        top layer held at the inlet's top velocity (instead of wrapping to the ground)
        forced_mask/u  cells whose velocity is imposed every step (fans), u: [3, X, Y, Z] or [B, 3, X, Y, Z]
        reservoir_mask cells held at rest at rho = 1 (ambient air around a closed box)
        open_shell     the reservoir is the outer layer of the box: hold it at rho = 1 but give it the velocity of
                       the cell inside (pressure outlet / inlet). Without this, air entering or leaving through an
                       opening meets populations at rest, which halves the flux a fan can push and makes passive
                       openings far more resistive than real grilles.
        open_inlet     False disables the inlet / outlet planes (closed domains)"""
        cfg = cfg or LBMConfig()
        self.cfg = cfg
        dev = torch.device(device)
        self.device = dev
        if solid.dim() == 3:
            solid = solid.unsqueeze(0)
        solid = solid.to(dev, torch.bool)
        self.solid = solid
        B, X, Y, Z = solid.shape
        self.shape = (B, X, Y, Z)
        dt = cfg.dtype

        self.c_int = torch.tensor(C_LIST, dtype=torch.long)
        self.c = torch.tensor(C_LIST, dtype=dt, device=dev)                 # [19, 3]
        self.w = torch.tensor(W_LIST, dtype=dt, device=dev)                 # [19]
        self.w5 = self.w.view(1, Q, 1, 1, 1)
        self.opp = torch.tensor(OPP_LIST, dtype=torch.long, device=dev)
        # second-moment tensor for the non-equilibrium stress, order xx yy zz xy xz yz
        cc = []
        for (cx, cy, cz) in C_LIST:
            cc.append([cx * cx, cy * cy, cz * cz, cx * cy, cx * cz, cy * cz])
        self.cc = torch.tensor(cc, dtype=dt, device=dev)                    # [19, 6]
        self.qw = torch.tensor([1, 1, 1, 2, 2, 2], dtype=dt, device=dev).view(1, 6, 1, 1, 1)

        # bounce-back links: fluid cell x whose neighbour x + c_i is solid
        fluid = ~solid
        bb = []
        for (cx, cy, cz) in C_LIST:
            nb_solid = torch.roll(solid, shifts=(-cx, -cy, -cz), dims=(1, 2, 3))
            bb.append(nb_solid & fluid)
        self.bb = torch.stack(bb, 1)                                        # [B, 19, X, Y, Z]
        self.bb[:, 0] = False
        self.bb_opp = self.bb[:, self.opp].contiguous()                     # bb_opp[:, j] = bb[:, opp[j]]
        self.bb_f = self.bb.to(dt)
        self.solid5 = solid.unsqueeze(1)
        self.fluid_f = fluid.to(dt)

        # relaxation-time field with outlet sponge
        xs = torch.arange(X, device=dev, dtype=dt)
        ramp = ((xs - (X - cfg.sponge)) / max(cfg.sponge, 1)).clamp(min=0.0) ** 2
        self.tau0 = (cfg.tau + (cfg.tau_sponge - cfg.tau) * ramp).view(1, X, 1, 1)

        # frontal area (projected on the y-z plane)
        self.area = solid.any(dim=1).sum(dim=(1, 2)).to(dt)                # [B]

        # initial state: uniform flow outside the body, rest inside (impulsive start)
        rho = torch.ones((B, X, Y, Z), dtype=dt, device=dev)
        u = torch.zeros((B, 3, X, Y, Z), dtype=dt, device=dev)
        u[:, 0] = cfg.u_in
        u = u * fluid.unsqueeze(1)
        self.f = self.equilibrium(rho, u).contiguous()
        u_in = torch.zeros((1, 3, 1, 1), dtype=dt, device=dev)
        u_in[:, 0] = cfg.u_in
        self.u_in_vec = u_in
        self.open_inlet = open_inlet
        self.top_lid = top_lid
        if inlet_profile is not None:
            prof = torch.as_tensor(inlet_profile, dtype=dt, device=dev).view(1, 1, 1, Z)
            self.u_in_vec = torch.zeros((1, 3, 1, Z), dtype=dt, device=dev)
            self.u_in_vec[:, 0:1] = prof
            u0 = torch.zeros((B, 3, X, Y, Z), dtype=dt, device=dev)
            u0[:, 0] = prof.view(1, 1, 1, Z)
            self.f = self.equilibrium(rho, u0 * fluid.unsqueeze(1)).contiguous()
        if not open_inlet:
            self.f = self.equilibrium(rho, torch.zeros_like(u)).contiguous()
        self.forced_mask = None
        if forced_mask is not None:
            fm = torch.as_tensor(forced_mask, device=dev).bool()
            self.forced_mask = (fm if fm.dim() == 4 else fm.unsqueeze(0)).unsqueeze(1)
            fu = torch.as_tensor(forced_u, dtype=dt, device=dev)
            self.forced_u = fu if fu.dim() == 5 else fu.unsqueeze(0)
        self.reservoir_mask = None
        if reservoir_mask is not None:
            rm = torch.as_tensor(reservoir_mask, device=dev).bool()
            self.reservoir_mask = (rm if rm.dim() == 4 else rm.unsqueeze(0)).unsqueeze(1)
            self.f_rest = self.equilibrium(torch.ones((1, 1, 1, 1), dtype=dt, device=dev),
                                           torch.zeros((1, 3, 1, 1, 1), dtype=dt, device=dev))
        self.open_shell = open_shell
        self.step_count = 0
        self.mass0 = (self.f.sum(1) * self.fluid_f).sum(dim=(1, 2, 3))
        self._graph = None

    # --- kernels -------------------------------------------------------------
    def equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """rho [B, ...], u [B, 3, ...] -> feq [B, 19, ...]"""
        cu = torch.einsum("id,bd...->bi...", self.c, u)
        usq = (u * u).sum(1, keepdim=True)
        w = self.w.view((1, Q) + (1,) * (u.dim() - 2))
        return w * rho.unsqueeze(1) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    def macroscopic(self, f: torch.Tensor):
        rho = f.sum(1)
        u = torch.einsum("id,bixyz->bdxyz", self.c, f) / rho.unsqueeze(1)
        return rho, u

    def limit(self, u: torch.Tensor) -> torch.Tensor:
        """Scale velocities above cfg.u_max down to it. Only for the rare flows that would otherwise leave the
        lattice's low-Mach range and blow up; below u_max nothing changes."""
        if not self.cfg.u_max:
            return u
        sp = torch.sqrt((u * u).sum(1, keepdim=True))
        return u * (self.cfg.u_max / sp.clamp(min=self.cfg.u_max))

    def collide(self, f: torch.Tensor) -> torch.Tensor:
        rho, u = self.macroscopic(f)
        feq = self.equilibrium(rho, self.limit(u))
        fneq = f - feq
        tau0 = self.tau0
        if self.cfg.smagorinsky > 0:
            pi = torch.einsum("ik,bixyz->bkxyz", self.cc, fneq)
            qn = torch.sqrt((pi * pi * self.qw).sum(1))
            cs2 = self.cfg.smagorinsky ** 2
            tau = 0.5 * (tau0 + torch.sqrt(tau0 * tau0 + 18.0 * math.sqrt(2.0) * cs2 * qn / rho))
        else:
            tau = tau0.expand_as(rho)
        return feq + fneq * (1.0 - 1.0 / tau).unsqueeze(1)

    def stream(self, f: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(f)
        out[:, 0] = f[:, 0]
        for i in range(1, Q):
            cx, cy, cz = C_LIST[i]
            out[:, i] = torch.roll(f[:, i], shifts=(cx, cy, cz), dims=(1, 2, 3))
        return out

    def _step(self) -> torch.Tensor:
        fpost = self.collide(self.f)
        fs = self.stream(fpost)
        # halfway bounce-back: population leaving x along i towards a wall returns along opp(i)
        fs = torch.where(self.bb_opp, fpost[:, self.opp], fs)
        if self.open_inlet:
            # inlet: equilibrium at prescribed velocity, density extrapolated from x=1
            rho_in = fs[:, :, 1].sum(1)                                      # [B, Y, Z]
            fs[:, :, 0] = self.equilibrium(rho_in, self.u_in_vec.expand(rho_in.shape[0], 3, *rho_in.shape[1:]))
            # outlet: zero gradient
            fs[:, :, -1] = fs[:, :, -2]
        if self.top_lid:
            rho_top = fs[:, :, :, :, -2].sum(1)                              # [B, X, Y]
            u_top = torch.zeros((rho_top.shape[0], 3) + tuple(rho_top.shape[1:]), dtype=fs.dtype, device=fs.device)
            u_top[:, 0] = self.u_in_vec[0, 0, 0, -1]
            fs[:, :, :, :, -1] = self.equilibrium(rho_top, u_top)
        if self.forced_mask is not None:   # fans: impose velocity, keep local density
            rho_l = fs.sum(1)
            feq_f = self.equilibrium(rho_l, self.forced_u.expand(fs.shape[0], -1, -1, -1, -1))
            fs = torch.where(self.forced_mask, feq_f, fs)
        if self.reservoir_mask is not None:
            if self.open_shell:              # ambient pressure, velocity of the cell inside
                fs = self._open_shell(fs)
            else:                            # ambient air at rest
                fs = torch.where(self.reservoir_mask, self.f_rest, fs)
        # keep solid interior at rest (never read by fluid cells, keeps values finite)
        fs = torch.where(self.solid5, self.w5, fs)
        self.f.copy_(fs)
        return fpost

    def _open_shell(self, fs: torch.Tensor) -> torch.Tensor:
        """Outer layer of the box = equilibrium at rho = 1 with the velocity of the adjacent inner cell."""
        a = slice(None)
        for dim, n in ((2, fs.shape[2]), (3, fs.shape[3]), (4, fs.shape[4])):
            for res_i, in_i in ((0, 1), (n - 1, n - 2)):
                sr = [a] * 5
                si = [a] * 5
                sr[dim], si[dim] = res_i, in_i
                f_in = fs[tuple(si)]                                          # [B, 19, face...]
                rho = f_in.sum(1)
                u = torch.einsum("id,bi...->bd...", self.c, f_in) / rho.unsqueeze(1)
                fs[tuple(sr)] = self.equilibrium(torch.ones_like(rho), self.limit(u))
        return fs

    def momentum_exchange(self, fpost: torch.Tensor) -> torch.Tensor:
        """Force on the body [B, 3] from post-collision populations crossing wall links."""
        s = (fpost * self.bb_f).sum(dim=(2, 3, 4))                         # [B, 19]
        return 2.0 * s @ self.c                                              # [B, 3]

    # --- driver --------------------------------------------------------------
    def run(
        self,
        steps: int = 5000,
        avg_from: int | None = None,
        sample_every: int = 10,
        report_every: int = 100,
        callback=None,
        check_nan_every: int = 500,
    ) -> LBMResult:
        cfg = self.cfg
        B, X, Y, Z = self.shape
        if avg_from is None:
            avg_from = int(steps * 0.6)
        dev = self.device
        u_acc = torch.zeros((B, 3, X, Y, Z), dtype=cfg.dtype, device=dev)
        rho_acc = torch.zeros((B, X, Y, Z), dtype=cfg.dtype, device=dev)
        f_acc = torch.zeros((B, 3), dtype=cfg.dtype, device=dev)
        f_window = torch.zeros((B, 3), dtype=cfg.dtype, device=dev)
        n_acc = 0
        n_fwin = 0
        hist = []
        q = 0.5 * cfg.u_in ** 2 * self.area.clamp(min=1.0)                 # dynamic pressure x area
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        t0 = time.perf_counter()
        for s in range(1, steps + 1):
            fpost = self._step()
            self.step_count += 1
            # sample on two consecutive steps: bounce-back between walls one cell apart can drive a
            # period-2 oscillation, and sampling only every 10th step would alias it into a bias
            if s % sample_every in (0, 1):
                F = self.momentum_exchange(fpost)
                f_window += F
                n_fwin += 1
                if s > avg_from:
                    rho, u = self.macroscopic(self.f)
                    u_acc += u
                    rho_acc += rho
                    f_acc += F
                    n_acc += 1
            if s % report_every == 0 and n_fwin > 0:
                cd_now = (f_window[:, 0] / n_fwin) / q
                hist.append((s, cd_now.detach().cpu().tolist()))
                f_window.zero_()
                n_fwin = 0
                if callback is not None:
                    callback(s, steps, cd_now)
            if check_nan_every and s % check_nan_every == 0:
                if not torch.isfinite(self.f).all():
                    raise FloatingPointError(f"LBM diverged at step {s}")
        torch.cuda.synchronize(dev) if dev.type == "cuda" else None
        secs = time.perf_counter() - t0
        n_acc = max(n_acc, 1)
        u_avg = (u_acc / n_acc) * self.fluid_f.unsqueeze(1)
        rho_avg = rho_acc / n_acc
        p_avg = (rho_avg - 1.0) / 3.0 * self.fluid_f
        F_avg = f_acc / n_acc
        cd = F_avg[:, 0] / q
        cl = F_avg[:, 1:] / q.unsqueeze(1)
        mass = (self.f.sum(1) * self.fluid_f).sum(dim=(1, 2, 3))
        drift = (mass - self.mass0) / self.mass0
        mlups = B * X * Y * Z * steps / secs / 1e6
        return LBMResult(
            u=u_avg, p=p_avg, cd=cd, cl=cl, cd_history=hist, area=self.area,
            steps=steps, seconds=secs, mlups=mlups, mass_drift=drift,
        )


def reynolds(cfg: LBMConfig, area_cells: float) -> float:
    """Re based on the frontal-area-equivalent diameter D = sqrt(4 A / pi)."""
    d = math.sqrt(4.0 * area_cells / math.pi)
    return cfg.u_in * d / cfg.nu
