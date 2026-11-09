"""Air temperature in the case.

The LBM gives the time-averaged air velocity; heat is then carried by that air (explicit upwind
advection + diffusion, walls adiabatic, outside air held at ambient). A part's temperature is
intake air rise + P / G, where G (W/K) is the cooler's conductance.
"""
from __future__ import annotations

import torch

RHO_CP = 1200.0           # J / (m^3 K), air
KAPPA = 0.04              # thermal (eddy) diffusivity in lattice units: cells^2 per step


def solve_temperature(u_lat: torch.Tensor, model: dict, parts: dict, iters: int = 5000, kappa: float | None = None,
                      dx_mm: float = 6.0, u_lat_per_ms: float = 0.05 / 1.5, T0: torch.Tensor | None = None) -> torch.Tensor:
    """Steady air temperature rise (K above ambient). model: shape, solid, reservoir, heat{cpu,gpu} masks;
    parts: cpu_w, gpu_w (W). T0 continues from an earlier field (for convergence checks).

    Flux form: heat moves between cells only through shared faces (upwind value times face velocity), so the
    heat the parts release can leave the case only by being carried out through an opening. Diffusion (turbulent
    mixing) acts only between air cells inside the case: walls are adiabatic and there is no diffusive leak into
    the room air at the openings."""
    kappa = KAPPA if kappa is None else kappa
    dev = u_lat.device
    solid = torch.from_numpy(model["solid"]).to(dev)
    ambient = torch.from_numpy(model["reservoir"]).to(dev)
    fluid = (~solid).float()                                  # air, including the room layer around the case
    inner = (~solid & ~ambient).float()                       # air inside the case
    dt_phys = (dx_mm / 1000.0) * u_lat_per_ms                 # s per lattice step (u_phys = u_lat / u_lat_per_ms)
    v_cell = (dx_mm / 1000.0) ** 3
    q = torch.zeros(model["shape"], device=dev)
    for key, watts in (("cpu", parts["cpu_w"]), ("gpu", parts["gpu_w"])):
        m = torch.from_numpy(model["heat"][key]).to(dev) & ~solid & ~ambient
        n = int(m.sum())
        if n:
            q[m] = watts / n / (RHO_CP * v_cell) * dt_phys     # K per lattice step per cell
    T = T0.clone() if T0 is not None else torch.zeros(model["shape"], device=dev)
    # face velocity between cell i and i+1 along each axis; zero across walls
    faces = []
    for d in range(3):
        uc = u_lat[d]
        faces.append((d, 0.5 * (uc + torch.roll(uc, -1, d)) * fluid * torch.roll(fluid, -1, d)))
    neighbours = [(d, sh, torch.roll(inner, sh, d) * inner) for d in range(3) for sh in (1, -1)]
    for _ in range(iters):
        div = torch.zeros_like(T)
        for d, uf in faces:
            F = torch.where(uf > 0, uf * T, uf * torch.roll(T, -1, d))   # flux through each cell's + face
            div = div + F - torch.roll(F, 1, d)
        lap = torch.zeros_like(T)
        for d, sh, m in neighbours:
            lap = lap + (torch.roll(T, sh, d) - T) * m
        T = (T - div + kappa * lap + q) * fluid
        T = torch.where(ambient, torch.zeros_like(T), T)     # room air stays at ambient
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


def solve_unit_fields(u_lat: torch.Tensor, model: dict, probes: list, p_ref: float = 100.0, tol: float = 0.005,
                      chunk: int = 2500, min_iters: int = 10000, max_iters: int = 80000, kappa: float | None = None,
                      dx_mm: float = 6.0, u_lat_per_ms: float = 0.05 / 1.5) -> tuple[torch.Tensor, int]:
    """Air temperature rise per watt of CPU power and per watt of GPU power, for each airflow u_lat [L, 3, X, Y, Z].
    Returns [L, 2, X, Y, Z] in K/W (channel 0: CPU heat, 1: GPU heat) and the iterations used.

    Same scheme as solve_temperature, run on 2L fields at once. The equation is linear in the heat sources, so for
    any CPU and GPU power the rise is P_cpu * T[:, 0] + P_gpu * T[:, 1]. Each channel is solved with p_ref watts
    (so the tolerance means the same as in solve_temperature_converged) and divided by p_ref at the end."""
    kappa = KAPPA if kappa is None else kappa
    dev = u_lat.device
    n_lv = u_lat.shape[0]
    shape = tuple(model["shape"])
    solid = torch.from_numpy(model["solid"]).to(dev)
    ambient = torch.from_numpy(model["reservoir"]).to(dev)
    fluid = (~solid).float()
    inner = (~solid & ~ambient).float()
    dt_phys = (dx_mm / 1000.0) * u_lat_per_ms
    v_cell = (dx_mm / 1000.0) ** 3
    q1 = torch.zeros((2,) + shape, device=dev)
    for c, key in enumerate(("cpu", "gpu")):
        m = torch.from_numpy(model["heat"][key]).to(dev) & ~solid & ~ambient
        n = int(m.sum())
        if n:
            q1[c][m] = p_ref / n / (RHO_CP * v_cell) * dt_phys
    q = q1.repeat(n_lv, 1, 1, 1)                                      # [2L, X, Y, Z]: per level CPU, then GPU
    u = u_lat.float().repeat_interleave(2, dim=0)                     # [2L, 3, X, Y, Z]
    faces = []
    for d in range(3):
        uc = u[:, d]
        faces.append((d + 1, 0.5 * (uc + torch.roll(uc, -1, d + 1)) * fluid * torch.roll(fluid, -1, d)))
    del u
    neighbours = [(d + 1, sh, torch.roll(inner, sh, d) * inner) for d in range(3) for sh in (1, -1)]
    masks = [torch.as_tensor(m, device=dev) for m in probes]
    T = torch.zeros_like(q)
    zero = torch.zeros((), device=dev)
    done, last = 0, None
    while done < max_iters:
        for _ in range(chunk):
            div = torch.zeros_like(T)
            for dd, uf in faces:
                F = torch.where(uf > 0, uf * T, uf * torch.roll(T, -1, dd))
                div = div + F - torch.roll(F, 1, dd)
            lap = torch.zeros_like(T)
            for dd, sh, m in neighbours:
                lap = lap + (torch.roll(T, sh, dd) - T) * m
            T = (T - div + kappa * lap + q) * fluid
            T = torch.where(ambient, zero, T)
        done += chunk
        now = torch.stack([T[:, m].mean(dim=1) for m in masks])          # [probes, 2L]
        if last is not None and done >= min_iters and float((now - last).abs().max()) < tol:
            break
        last = now
    return (T / p_ref).view((n_lv, 2) + shape), done
