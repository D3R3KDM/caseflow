"""Air temperature in the case and cooler calibration.

The LBM gives the time-averaged air velocity; heat is then carried by that air (explicit upwind
advection + diffusion, walls adiabatic, outside air held at ambient). A part's temperature is
intake air rise + P / G, where G (W/K) is the cooler's conductance.
"""
from __future__ import annotations

import torch

RHO_CP = 1200.0           # J / (m^3 K), air


def solve_temperature(u_lat: torch.Tensor, model: dict, parts: dict, iters: int = 5000, kappa: float = 0.04,
                      dx_mm: float = 6.0, u_lat_per_ms: float = 0.05 / 1.5, T0: torch.Tensor | None = None) -> torch.Tensor:
    """Steady air temperature rise (K above ambient). model: shape, solid, reservoir, heat{cpu,gpu} masks;
    parts: cpu_w, gpu_w (W). T0 continues from an earlier field (for convergence checks)."""
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
    T = T0.clone() if T0 is not None else torch.zeros(model["shape"], device=dev)
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


def solve_temperature_converged(u_lat: torch.Tensor, model: dict, parts: dict, probes: list, tol: float = 0.005,
                                chunk: int = 2500, min_iters: int = 10000, max_iters: int = 80000, **kw) -> tuple[torch.Tensor, int]:
    """Iterate the heat transport until the probe temperatures (mean over each boolean mask, e.g. the CPU and GPU
    intakes) change by less than tol (K) over one chunk. A fixed 5000 iterations was not enough: low-flow layouts
    (all intake, 75 CFM) still warmed by 2-3 K, which made them look better than they are."""
    T, done, last = None, 0, None
    masks = [torch.as_tensor(m, device=u_lat.device) for m in probes]
    while done < max_iters:
        T = solve_temperature(u_lat, model, parts, iters=chunk, T0=T, **kw)
        done += chunk
        now = [float(T[m].mean()) for m in masks]
        if last is not None and done >= min_iters and max(abs(a - b) for a, b in zip(now, last)) < tol:
            break
        last = now
    return T, done


def calibrate_g(measured_c: float, watts: float, ambient_c: float, intake_rise_c: float) -> float | None:
    """Cooler conductance from one measured temperature at a known power."""
    d = measured_c - ambient_c - intake_rise_c
    return watts / d if d > 1.0 and watts > 20 else None
