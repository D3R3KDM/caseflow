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

| # | Your 3 fans | CPU | GPU | Dust settling |
|---|---|---|---|---|
| **1** | **all three on top, blowing out (rear mount empty)** | **75.6 °C** | **62.2 °C** | ×0.97 |
| 2 | top-front + top-rear + rear out | 76.0 °C | 63.0 °C | ×0.93 |
| 3 | top-mid + top-rear + rear out (the current layout) | 75.8 °C | 63.4 °C | ×1.00 |
| 8 | all three on top, blowing in (lowest CPU) | 74.6 °C | 66.6 °C | ×0.88 |
| 22 | rear in · top-front + top-mid out | 79.5 °C | 65.9 °C | ×0.94 |
| 32 | all seven fans blowing in (worst) | 79.9 °C | 69.5 °C | ×0.90 |

All results are at CPU 88 W, GPU 300 W and a room at 25 °C, with the case fans at 70 % (stock) and 80 % (spare). These are settings, not readings: see [Inputs](#inputs-what-is-set-and-what-is-measured). Dust settling is relative to the current layout.

- **Keep all three spare fans as exhaust.** The top four layouts are all-exhaust.
- **Optional improvement:** move the rear fan to the empty top-front mount (still exhaust). The GPU runs about 1.3 °C cooler (0.2–2.5 °C across the sensitivity, load and fan-speed runs: the slower the case fans, the bigger the gain) and the CPU about the same. With 40 % slower GPU and cooler fans the three all-exhaust layouts tie.
- **Avoid intakes on the top.** Each extra intake warms the GPU; three of them cost about 3 °C.
- **Never use the rear fan as an intake.** It blows straight into the CPU cooler's exhaust. The bottom twelve layouts all do this.
- **Never make all fans intakes.** The heat has nowhere to go: this is the worst of the 32 layouts.
- **Dust barely depends on the layout** (×0.88–1.09), because only the bottom has a filter. The optional front filter (LAN207-1X) cuts the unfiltered share from 69 % to 17 %.

## Inputs: what is set and what is measured

Fan speeds and power are **settings**, not readings from the PC, and utilisation is not used:

| Input | Value used | Source |
|---|---|---|
| Stock fan speed | 70 %: ≈1330 rpm (front 140), ≈1365 rpm (bottom 120) | Set. The board drives the fans from its Q-Fan curve, and Windows cannot read case-fan rpm without a sensor driver. Configure shows the rpm next to each slider, so it can be matched to Armoury Crate or the BIOS. |
| Spare fans | 80 %: ≈1200 rpm | Model unknown: a typical 120 mm fan (1500 rpm, 55 CFM, 1.6 mmH₂O). |
| Fan flow | rated CFM × speed × 0.6, then the operating point against the case pressure | Rated flow and static pressure from the spec sheets; 0.6 is an estimate. The current layout takes in 130 CFM. |
| CPU cooler and GPU fans | fixed: ≈53 CFM through the cooler, 1.6 m/s (≈66 CFM) through the GPU | Set. The real 5080 FE stops its fans at light load and runs them at about 1,500 rpm in games. |
| CPU power | 88 W | The 9700X's power limit (PPT): all cores busy. Games use about 70 W. Measured live, with no driver, from Windows' Energy Meter counters (RAPL); *use my last 10 min of gaming* fills in your real average. |
| GPU power | 300 W | Typical in games (limit 360 W). Read live with `nvidia-smi`; *use my last 10 min of gaming* fills in your real average. |
| Utilisation | not used | Heat follows watts, not % busy. |
| Room | 25 °C | Set. |

Room temperature and cooler conductance shift every layout by the same amount, so they never change the ranking. Power and fan speed can, so both were tested on five key layouts (CPU / GPU °C; "order" compares the five layouts' scores with the default):

| Scenario | Current (#3) | All three top out (#1) | #2 | Rear in | All in (#32) | Order |
|---|---|---|---|---|---|---|
| Default: 88 W + 300 W, fans 70 / 80 % | 75.8 / 63.4 | 75.6 / 62.2 | 76.0 / 63.0 | 79.5 / 65.9 | 79.9 / 69.5 | — |
| Gaming: 70 W + 300 W | 66.8 / 63.4 | 66.5 / 62.2 | 67.0 / 63.0 | 70.3 / 65.7 | 70.7 / 69.2 | same |
| CPU stress: 88 W + 60 W | 70.6 / 32.8* | 70.5 / 32.5* | 70.6 / 32.7* | 72.1 / 33.9* | 72.2 / 34.9* | same |
| Everything maxed: 88 W + 360 W | 77.2 / 71.1 | 76.8 / 69.6 | 77.4 / 70.6 | 81.4 / 73.9 | 81.9 / 78.1 | same |
| Case fans all 50 % | 78.5 / 65.6 | 77.0 / 63.1 | 78.5 / 64.9 | 81.5 / 67.7 | 83.0 / 70.9 | same |
| Case fans all 100 % | 74.2 / 61.8 | 74.1 / 61.6 | 74.2 / 61.6 | 77.2 / 64.6 | 76.7 / 66.9 | same |
| Stock fans 100 %, spare fans 50 % | 75.4 / 64.6 | 74.3 / 63.4 | 75.3 / 63.7 | 76.3 / 65.8 | 76.7 / 66.8 | same |
| GPU and CPU cooler fans × 0.6 | 76.1 / 62.5 | 76.5 / 62.3 | 76.4 / 62.2 | 80.5 / 66.2 | 79.3 / 70.9 | #2 < current < #1 < rear in < all in |
| GPU and CPU cooler fans × 1.4 | 75.5 / 63.4 | 74.6 / 61.8 | 75.6 / 63.0 | 77.8 / 64.9 | 79.4 / 68.0 | same |

\* At 60 W the real GPU stops its fans and runs warmer; the model keeps them spinning.

- **The bottom two never move:** rear intake and all-intake are the worst in every test.
- **Among the three all-exhaust layouts, the gap depends on airflow:**
  - The slower the case fans, the more "all three on top out" helps: −1.5 / −2.5 °C (CPU / GPU) at 50 %. At 100 % the three are tied within 0.2 °C.
  - With weak spare fans (stock 100 %, spare 50 %) it still gains 1.1 / 1.2 °C.
  - Faster GPU and cooler fans (× 1.4) widen the gap to 0.9 / 1.6 °C. Slower ones (× 0.6) close it: the three tie within 0.1 °C on score, and the current layout's CPU is 0.4 °C cooler. These two rows change only how those fans move the air in the case. The coolers' own conductance G is kept, because it shifts every layout equally.
- **So "all three on top out" is never meaningfully worse than the current layout:** at worst it ties, and it gains 0–2.5 °C. It stays an optional small change.
- **Not knowing the fan speed costs about ±2 °C in absolute terms:** from 50 % to 100 %, the current layout goes from 78.5 / 65.6 °C to 74.2 / 61.8 °C.

**Absolute temperatures.**
- CPU: within the range of reviews (about 57–72 °C at 88 W) once the in-case pre-heat and the room temperature are taken out.
- GPU: probably about 5 °C low. Reviews give about 66 °C (64–68) at a room of about 20–23 °C; the model gives 60 °C at 22 °C. On this PC at 154 W the GPU read 49.9 °C with its fans at 30 %, where the model gives about 45 °C at 25 °C.
- A gaming session removes this: Configure → *use my last 10 min of gaming* fills in the real CPU and GPU power and calibrates the GPU cooler. Then set the room temperature and the fan speeds you see in Armoury Crate.

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

- read live GPU temperature and power (`nvidia-smi`) and CPU package power (Windows' Energy Meter counters), and keep the last 30 minutes so a gaming session can be used;
- detect the CPU, motherboard and RAM;
- edit the build and run new simulations (about 40 s each);
- calibrate the cooler with a measured GPU temperature.

Python: see `requirements.txt` (PyTorch with CUDA for speed). `start.bat` uses `.venv` in this folder, or `../.venv`.

```bash
python engine/optimize.py        # search all 32 layouts (≈25 min on an RTX 5080), resumable
python scripts/precompute.py     # write the web views from the search result
python -m pytest tests           # fast engine checks (no GPU run)
```

## How it works

| Stage | Model |
|---|---|
| **Geometry** | Parts come from `data/parts/library.json`: dimensions, fan mounts, fan specs and source links. They are placed in the case on a 6 mm grid. <br>Solid: walls, PSU, motherboard, GPU PCB and CPU fin towers (as heat zones). <br>Empty mesh mounts and the vented slot covers are open to the room. |
| **Air** | Our own D3Q19 lattice-Boltzmann solver (`engine/lbm.py`, PyTorch on the GPU) with BGK collision, Smagorinsky turbulence and bounce-back walls. <br>Fans are cells with a set air velocity. <br>The layer of "room" cells around the case is an open boundary: ambient pressure, with the velocity of the cell inside. So fans deliver their set flow (≈ 95 %) and the openings are not artificially resistive. |
| **Fans** | Each case fan runs at its **operating point**. <br>Fan curve: Q = Q₀ (1 − Δp / p_max), from the rated flow and static pressure, scaled by the fan laws. <br>All fans see one case pressure p. The passive openings balance the flow as orifices: Q = C_d A √(2\|p\| / ρ). <br>p is found by bisection. Without this step, "all intake, no exhaust" ranked first. With it, those fans stall against +7.8 Pa and that layout drops to 14th. |
| **Heat** | CPU and GPU power enters the air in their fin stacks and is carried by the solved flow. <br>The transport is conservative: upwind fluxes through cell faces, and mixing only inside the case. So all the heat leaves through the openings (energy balance 99.9 %). <br>It is iterated until the intake temperatures change by less than 0.005 K. <br>Part temperature = intake air rise + P / G, where G is the cooler's conductance. One measured temperature calibrates G. |
| **Dust** | 40 000 particles of 20 µm are released where air comes in, weighted by inflow; the bottom filter stops 85 %. <br>They drift with the flow plus turbulent diffusion and settling, and stick where they touch a surface. |
| **Search** | Stock fans stay where they are. The spare fans go on every free mount, each blowing in or out (C(4,3) × 2³ = 32 layouts), and every layout gets the full simulation. <br>Score = CPU + GPU °C + 2 °C × dust settling rate (current layout = 1), where dust settling = unfiltered intake × share that settles. <br>Layouts within 0.5 °C of the top count as a tie: no change is recommended for less. |

**Checks** (scripts in `scripts/`):

| Check | Result |
|---|---|
| `check_convergence.py` | Heat iterated to 80 000 steps and airflow to 16 000 steps: temperatures change by less than 0.1 °C. |
| `sanity_check.py` | Fans deliver about 95 % of their set flow. Shell out/in flow is 0.95–0.99. Heat carried out is 99.9 % of the 388 W, and exhaust air rise matches P / (ρ c_p Q). |
| `sensitivity.py` | Varied one at a time: heat diffusivity ÷4 / ×2, fan derate 0.5 / 0.7, opening porosity ×0.5 / ×1.5, GPU slot 40 mm higher, front filter, 4 mm grid; three loads, three case-fan speeds, slower and faster GPU and cooler fans ([Inputs](#inputs-what-is-set-and-what-is-measured)). <br>The ranking is unchanged in every run except among near-tied layouts: one swap in the parameter runs, and a three-way tie (within 0.1 °C) with 40 % slower GPU and cooler fans. Absolute temperatures move by 2–3 °C for the parameters and about ±2 °C for the case-fan speed. |
| Reviews | CPU: within the review range once the in-case pre-heat and room temperature are taken out. GPU: probably about 5 °C low ([Inputs](#inputs-what-is-set-and-what-is-measured)). |

These checks found two real bugs, both fixed and the search re-run:
- The heat solve was stopped after 5 000 iterations, before it converged.
- A room boundary held at rest delivered only about 70 % of each fan's flow and lost heat. This made "all intake" look better than it is.

**Limits.**
- Mount positions, opening porosity and cooler conductance are estimates from specs and reviews (marked `est` in the library).
- Fans are modelled as uniform jets, without swirl, at fixed speeds: real fans follow temperature curves, and the GPU stops its fans at light load. The grid uses a higher viscosity than real air, for stability. Buoyancy is not modelled.
- The room outside stays at 25 °C, so hot exhaust is never drawn back in.
- Trust the differences between layouts more than the absolute temperatures. Calibrate with a GPU reading taken under load (150 W or more), best with a gaming session; idle readings are rejected.

## Layout

```
engine/
  lbm.py          D3Q19 LBM solver (GPU)
  sim.py          build → grid, fan operating point, airflow + heat + dust, web packaging
  thermal.py      air temperature (advection–diffusion), cooler calibration
  optimize.py     exhaustive layout search
  hardware.py     CPU / board / RAM (Windows CIM), GPU sensors (nvidia-smi), CPU power (Energy Meter)
server/app.py     FastAPI: /api/health, /hardware, /hardware/session, /library, /default_build, /simulate, /job/{id}
scripts/          precompute.py (web views), serve.sh (engine with auto-restart),
                  check_convergence.py, sanity_check.py, sensitivity.py (checks)
data/             parts library, search result (optimize.json)
web/              Vite + React + three.js app (3D case with floating holographic screens)
tests/            engine checks
docs/screens/     screenshots
```

## Interface

The 3D case sits in the middle of six floating holographic screens. Each screen is wired to the part it describes by a glowing leader line:

- **CPU** and **GPU**: temperature gauge, intake air rise, load; lines to the CPU cooler and the graphics card.
- **Airflow**: in / out CFM, case pressure, and every fan at its operating point; line to the front intake fans.
- **Dust**: share of outside dust that gets in, and where it settles; line to the glass.
- **Layout search** and **Verdict**: angled side screens with the 32-layout ranking and your layout against the alternative.

The screens are CSS3D objects in the same scene as the case. They turn with the camera, so they stay readable from any angle, while the lines and anchors stay fixed on the parts.

**Click any case fan** in 3D to flip it or turn it off, or click an empty mount to add a fan. If the new layout is a stored one, the view switches to it; otherwise the engine simulates it (about 40 s) and the result appears as *Custom*.

The bottom dock switches the layers: the screens, moving flow particles, streamlines, dust deposits, and a heat slice through the GPU. **Configure** (edit the build, run a simulation) and **Method** open as glass windows. The view is through the glass, front on the right, as the PC stands on a desk.
