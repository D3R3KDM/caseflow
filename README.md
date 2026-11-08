# CaseFlow

**Airflow, heat and dust inside your own PC case: simulated, ranked, and shown in 3D.**

![CaseFlow](docs/screens/caseflow_main.png)

**Walkthrough with figures, diagrams and a video (Chinese): [docs/GUIDE.md](docs/GUIDE.md)** · [30 s demo video](docs/media/caseflow_tour.mp4)

You tell CaseFlow what is in your PC: case, graphics card, CPU cooler, and where the fans are and which way they blow. It builds the case on a 6 mm grid and then:

- solves the air flow with its own GPU lattice-Boltzmann solver;
- carries the CPU and GPU heat through that flow;
- tracks dust particles from the openings to where they settle.

It then tries **every** way to place your spare fans and tells you whether your layout is worth changing.

## Result for the reference build

The reference build is a Lian Li LANCOOL 207 with an RTX 5080 Founders Edition, a Thermalright Peerless Assassin 120 SE and a Ryzen 7 9700X:

- 4 stock fans: 2 × 140 mm front intake, 2 × 120 mm bottom intake.
- 3 extra 120 mm fans: top-mid, top-rear and rear, all exhaust.

| # | Your 3 fans | CPU | GPU | Dust let in |
|---|---|---|---|---|
| **1** | **top-mid + top-rear + rear out (the current layout)** | **75.6 °C** | **63.4 °C** | **69 %** |
| 2 | top-front + top-rear + rear out | 75.8 °C | 63.4 °C | 68 % |
| 5 | top-front + top-rear in · rear out (lowest CPU) | 74.4 °C | 65.1 °C | 81 % |
| 14 | all three on top, blowing in (no exhaust fan) | 74.7 °C | 65.4 °C | 88 % |
| 32 | rear in · top-front + top-mid out (worst) | 77.2 °C | 64.9 °C | 76 % |

All results are at CPU 88 W, GPU 300 W and a room at 25 °C. The current layout ranks first of 32, so keep it.

- **Direction matters more than position.** The top-three layouts differ only in which top holes are used, and are within 0.4 °C of each other.
- **Avoid intakes on the top.** Turning any of the three fans into an intake warms the GPU by 1–2 °C and lets in 6–19 points more dust, because the top has no filter.
- **Never use the rear fan as an intake.** It blows straight into the CPU cooler's exhaust: CPU +1.6 °C.

## Run

**Just look:** the web app runs without Python. The views and the 32-layout ranking are precomputed:

```bash
cd web && npm install && npm run dev        # http://localhost:5190
```

**Full setup** (live hardware readout and new simulations). You need Python 3.11+, Node 18+, and an NVIDIA GPU for speed:

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build of PyTorch
.venv\Scripts\pip install -r requirements.txt
cd web && npm install
```

Then on Windows double-click **`start.bat`**. It starts the engine on port 8600 and the web app on port 5190, then opens <http://localhost:5190>.

```bash
# engine (from this folder)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8600     # or: bash scripts/serve.sh (auto-restart)

# web app
cd web && npm install && npm run dev                              # http://localhost:5190
```

The views (your layout, lowest CPU, worst) and the full 32-layout ranking are precomputed in `web/public/data/`, so the app also works with the engine off. With the engine running you can also:

- read live GPU temperature and power (`nvidia-smi`);
- detect the CPU, motherboard and RAM;
- edit the build and run new simulations (about 40 s each);
- calibrate the cooler with a measured GPU temperature.

Python: see `requirements.txt` (PyTorch with CUDA for speed). `start.bat` uses `.venv` in this folder, or `../.venv`.

```bash
python engine/optimize.py        # search all 32 layouts (≈18 min on an RTX 5080), resumable
python scripts/precompute.py     # write the web views from the search result
python -m pytest tests           # fast engine checks (no GPU run)
```

## How it works

| Stage | Model |
|---|---|
| **Geometry** | Parts come from `data/parts/library.json`: dimensions, fan mounts, fan specs and source links. They are placed in the case on a 6 mm grid. <br>Solid: walls, PSU, motherboard, GPU PCB and CPU fin towers (as heat zones). <br>Empty mesh mounts and the vented slot covers are open to the room. |
| **Air** | Our own D3Q19 lattice-Boltzmann solver (`engine/lbm.py`, PyTorch on the GPU) with BGK collision, Smagorinsky turbulence and bounce-back walls. <br>Fans are cells with a set air velocity. A layer of "room" cells around the case holds outside air at rest. |
| **Fans** | Each case fan runs at its **operating point**. <br>Fan curve: Q = Q₀ (1 − Δp / p_max), from the rated flow and static pressure, scaled by the fan laws. <br>All fans see one case pressure p. The passive openings balance the flow as orifices: Q = C_d A √(2\|p\| / ρ). <br>p is found by bisection. Without this step, "all intake, no exhaust" ranked first. With it, those fans stall against +7.8 Pa and that layout drops to 14th. |
| **Heat** | CPU and GPU power enters the air in their fin stacks and is carried by the solved flow (upwind advection–diffusion). <br>Part temperature = intake air rise + P / G, where G is the cooler's conductance. One measured temperature calibrates G. |
| **Dust** | 40 000 particles of 20 µm are released where air comes in, weighted by inflow; the bottom filter stops 85 %. <br>They drift with the flow plus turbulent diffusion and settling, and stick where they touch a surface. |
| **Search** | Stock fans stay where they are. The spare fans go on every free mount, each blowing in or out (C(4,3) × 2³ = 32 layouts), and every layout gets the full simulation. <br>Score = CPU + GPU °C + 2 °C × the fraction of outside dust let in. |

**Limits.**
- Mount positions, opening porosity and cooler conductance are estimates from specs and reviews (marked `est` in the library).
- Fans are modelled as uniform jets, without swirl.
- Trust the differences between layouts more than the absolute temperatures.

## Layout

```
engine/
  lbm.py          D3Q19 LBM solver (GPU)
  sim.py          build → grid, fan operating point, airflow + heat + dust, web packaging
  thermal.py      air temperature (advection–diffusion), cooler calibration
  optimize.py     exhaustive layout search
  hardware.py     CPU / board / RAM (Windows CIM), GPU sensors (nvidia-smi)
server/app.py     FastAPI: /api/health, /hardware, /library, /default_build, /simulate, /job/{id}
scripts/          precompute.py (web views), serve.sh (engine with auto-restart)
data/             parts library, search result (optimize.json)
web/              Vite + React + three.js app (floating HUD windows over the 3D case)
tests/            engine checks
docs/screens/     screenshots
```

## Interface

The 3D case fills the screen. Information sits in floating glass windows that you can drag, collapse and reset (⟲):

- **01 System**: the parts.
- **02 Layout search**: all 32 layouts, ranked; click one to view it.
- **03 Telemetry**: temperatures, airflow, case pressure, dust.
- **04 Compare**: your layout against the alternative, as side views through the glass.
- **05 Configure**: edit the build and run a simulation.
- **06 Method**: how the numbers are made.

The bottom dock switches the layers: moving flow particles, streamlines, dust deposits, and a heat slice through the GPU. The 3D view is seen through the glass, front on the right, as the PC stands on a desk.
