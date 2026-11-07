"""Air temperature in the case and cooler calibration.

The LBM gives the time-averaged air velocity; heat is then carried by that air (explicit upwind
advection + diffusion, walls adiabatic, outside air held at ambient). A part's temperature is
intake air rise + P / G, where G (W/K) is the cooler's conductance.
"""
from __future__ import annotations

import torch

RHO_CP = 1200.0           # J / (m^3 K), air


def solve_temperature(u_lat: torch.Tensor, model: dict, parts: dict, iters: int = 5000, kappa: float = 0.04,
                      dx_mm: float = 6.0, u_lat_per_ms: float = 0.05 / 1.5) -> torch.Tensor:
    """Steady air temperature rise (K above ambient). model: shape, solid, reservoir, heat{cpu,gpu} masks;
    parts: cpu_w, gpu_w (W)."""
    dev = u_lat.device
    solid = torch.from_numpy(model["solid"]).to(dev)
    ambient = torch.from_numpy(model["reservoir"]).to(dev)
    fluid = (~solid).float()
    dt_phys = (dx_mm / 1000.0) * u_lat_per_ms              # s per lattice step (u_phys = u_lat / u_lat_per_ms)
    v_cell = (dx_mm / 1000.0) ** 3
    q = torch.zeros(model["shape"], device=dev)
    for key, watts in (("cpu", parts["cpu_w"]), ("gpu", parts["gpu_w"])):
        m = torch.from_numpy(model["heat"][key]).to(dev) & ~solid
        n = int(m.sum())
        if n:
            q[m] = watts / n / (RHO_CP * v_cell) * dt_phys     # K per lattice step per cell
    T = torch.zeros(model["shape"], device=dev)
    ux, uy, uz = u_lat[0], u_lat[1], u_lat[2]
    for _ in range(iters):
        adv = torch.zeros_like(T)
        for d, uc in ((0, ux), (1, uy), (2, uz)):
            fwd = torch.roll(T, -1, d) - T
            bwd = T - torch.roll(T, 1, d)
            adv = adv + torch.where(uc > 0, uc * bwd, uc * fwd)
        lap = torch.zeros_like(T)                          # diffusion only between fluid cells
        for d in range(3):
            for sh in (1, -1):
                nb = torch.roll(T, sh, d)
                nbf = torch.roll(fluid, sh, d)
                lap = lap + (nb - T) * nbf
        T = (T - adv + kappa * lap + q) * fluid
        T = torch.where(ambient, torch.zeros_like(T), T)
    return T


def calibrate_g(measured_c: float, watts: float, ambient_c: float, intake_rise_c: float) -> float | None:
    """Cooler conductance from one measured temperature at a known power."""
    d = measured_c - ambient_c - intake_rise_c
    return watts / d if d > 1.0 and watts > 20 else None
