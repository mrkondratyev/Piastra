# Piastra

*A small, readable finite-volume laboratory for astrophysical fluid dynamics.*

Piastra solves the equations that shocks, jets, instabilities, and magnetized
flows live by — in **pure Python and NumPy**, with nothing hidden behind a
compiled black box. Every Riemann solver, every reconstruction stencil, every
geometric source term is written out in plain array operations you can read,
break, and rebuild. It is meant for two kinds of people: a student meeting
Godunov's method for the first time, and a researcher who wants to prototype a
scheme this afternoon without fighting a build system.

If you can read a `for` loop and an einsum-free NumPy slice, you can read all of
Piastra.

---

## What it solves

| Mode   | Physics                                            | `solver_type`                         |
|--------|----------------------------------------------------|---------------------------------------|
| `adv`  | Linear scalar advection                            | `adv` (upwind Godunov), `LW` (Lax–Wendroff) |
| `HD`   | Compressible (Euler) hydrodynamics                 | `LLF`, `HLL`, `HLLC`, `Roe`, `Exact`  |
| `rHD`  | Special-relativistic hydrodynamics                 | `LLF`, `HLL`, `HLLC`                  |
| `MHD`  | Ideal magnetohydrodynamics                         | `LLF`, `HLL`, `HLLC`, `HLLD`          |
| `rMHD` | Special-relativistic MHD                           | `LLF`, `HLL`                          |
| `SWE`  | Shallow-water equations (with bathymetry, Coriolis)| `LLF`, `HLL`, `Exact`                 |
| `diff` | Thermal diffusion ∂ₜT = ∇·(κ∇T) + S                | `expl` (explicit Euler), `rkl2` (super-time-stepping) |

All of it runs on **1D and 2D structured grids** in four geometries —
Cartesian `(x, y)`, cylindrical `(R, z)`, polar `(R, φ)`, and spherical-polar
`(r, θ)` — with face areas, cell volumes, and curvilinear source terms handled
consistently so the same solver code works in every coordinate system
(the SWE solver is Cartesian only).

On top of the hyperbolic solvers sit **body forces** (external and
self-gravity) and a general **per-step physics hook**, `par.before_step`, for
anything else a problem needs once per time step — gravity, cooling, damping
zones.

---

## How it's built (the numerics)

***HYPERBOLIC***  
- **Godunov finite volumes.** States are reconstructed to cell faces, a Riemann
  solver returns the interface flux, and the conservative update is the
  divergence of those fluxes — the textbook recipe, applied uniformly.
- **Reconstruction:** `PCM` (1st order), `PLM` (2nd, van Leer-limited slopes),
  `PPMorig` (Colella & Woodward 1984), `PPM` (Mignone 2014, Cartesian part only),
  `WENO` (central WENO, 3rd order), `MP5` (Suresh & Huynh 1997). The number of
  ghost cells is chosen automatically from `rec_type`.
- **Time integration:** SSP (TVD) Runge–Kutta `RK1`/`RK2`/`RK3`
  (Shu & Osher 1988).
- **Divergence control for MHD:** Constrained Transport (`CT`), hyperbolic
  divergence cleaning (`GLM`, Dedner et al. 2002), and Powell's 8-wave method
  (`8wave`).
- **Relativity:** reconstruction on the 4-velocity `u = W v` (so |v| < 1 at
  faces), Newton–Raphson conservative-to-primitive inversion. Relativistic MHD
  uses flux-CT (arithmetic averaging of the corner EMF; Balsara & Spicer 1999,
  Tóth 2000) for a divergence-free magnetic field.

***PARABOLIC***  
- **Diffusion:** explicit Euler (1st order in time) or RKL2
  super-time-stepping (Meyer, Balsara & Aslam 2012, 2nd order in time). One
  RKL2 super-step with `s` stages covers `(s² + s − 2)/4` explicit time
  steps at the cost of `s` operator evaluations.

***ELLIPTIC***  
- **Poisson solver:** second-order finite-volume `div(grad(phi)) = rhs` on any
  1D/2D grid and geometry, via matrix-free, diagonally-preconditioned Conjugate
  Gradient. Periodic, zero-gradient, and Dirichlet boundaries are supported on
  each face independently. Without any Dirichlet face the mean of `rhs` is
  removed (solvability) and `phi` is returned with zero mean. Stateless and
  grid-agnostic, so it is called from other modules — self-gravity,
  divergence cleaning — not just used standalone.

***SOURCES AND EXTRA PHYSICS***  
- **Gravity** (`src/gravity.py`): star + orbiting planet on a polar grid,
  spherical monopole self-gravity, and general self-gravity through the
  Poisson solver. The acceleration lives in `state.F1`, `state.F2`.
- **`par.before_step(grid, state, par, dt)`**: an optional function called
  once per time step by the HD, rHD, MHD and rMHD solvers, after `dt` is known
  and before the hyperbolic update (first-order operator splitting). This is
  where gravity is recomputed, and where cooling, damping zones or any other
  local source go.

---

## Requirements

- Python 3.9+
- NumPy
- matplotlib
- IPython / Jupyter (optional: only for `main.ipynb` and live plots in a notebook)

```bash
pip install numpy matplotlib ipython
```

No compilation, no external solver libraries, no configuration files.

---

## Quickstart

Piastra is a package rooted at `src/`. Run it from the repository root.

**The fast path** — edit the run-control block and the `Parameters` block
at the top of `main()` in `main.py` and run it:

```bash
python main.py
```

or open `main.ipynb` for the same workflow with live, re-runnable cells.
Both write snapshots to `output/` and can restart from any of them — see
[Output and restart](#output-and-restart).

**The explicit path** — drive it yourself in a few lines:

```python
from src.parameters import Parameters
from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.misc.helpers import initial_model, run_simulation
from src.models.HD.HD_step import HD2D

# 1. Configure the run
par = Parameters(mode="HD", problem="KHI2D", Nx1=128, Nx2=128,
                 solver_type="HLLC", rec_type="PPM", RK_order="RK3", CFL=0.7)

# 2. Build the grid, allocate state, and load the initial condition
grid  = Grid(par.Nx1, par.Nx2, par.Ngc)
state = SimState(grid, par)
grid, state, par, eos = initial_model(grid, state, par)   # sets geometry, ICs, BCs, t_fin

# 3. Pick a solver and march in time (plotting every n_plot steps)
solver = HD2D(grid, state, eos, par)
state, par.timenow = run_simulation(grid, state, par, solver,
                                    lambda: state.dens, n_plot=20)
```

The variable to plot is passed as a function (re-evaluated before every
plot), so derived quantities work too, e.g. `lambda: state.h + state.b` for
the SWE free surface.

The initial-condition function chooses the geometry, fills the primitive
variables, sets boundary conditions and the final time, and installs
`par.before_step` when the problem needs it (e.g. gravity) — so a single
`problem` string fully specifies a test case.

---

## Output and restart

A snapshot is one compressed NumPy archive (`.npz`). It holds the full state
(primitive, conservative, and the staggered CT / GLM fields), the grid
description, the scheme settings, the boundary conditions, the EOS and the
current time — so the **same file** is used to restart a run and to analyse
it.

**From `main.py` / `main.ipynb`.** The run-control block at the top of
`main()` (a separate cell in the notebook):

```python
OUTPUT_DIR    = "output"   # folder for snapshots; None = write nothing
DT_OUTPUT     = None       # physical time between snapshots; None = initial + final only
SAVE_ASCII_1D = True       # 1D runs: also write each snapshot as a .dat text table

RESTART       = False      # True = continue from RESTART_FILE
RESTART_FILE  = "output/MHD_disk2D_final.npz"
TIMEFIN_NEW   = None       # restart only: new final time (None = keep)
```

- **Files**: `output/<mode>_<problem>_0000.npz` is the initial condition,
  `_0001`, `_0002`, ... are written at `t = k·DT_OUTPUT`, and `_final.npz` at
  the end of the run. Output times are hit **exactly**: `run_simulation`
  shortens the step that would jump over one.
- **Restart**: set `RESTART = True` and point `RESTART_FILE` at any snapshot.
  The `Parameters` block is then ignored — grid, state, scheme, BCs, EOS and
  time all come from the file. To continue a finished run, restart from
  `_final.npz` with a larger `TIMEFIN_NEW`. Snapshot numbering continues from
  where it was (it is computed from `t / DT_OUTPUT`).
- **Per-step physics survives a restart.** `par.before_step` (e.g. gravity)
  is a Python function and can't be stored in a file; `restart_simulation`
  rebuilds it by re-running the problem's initial condition on scratch
  objects.
- A restarted run reproduces the uninterrupted one **bit for bit** (checked
  for every mode by the `restart` suite of the testbed).

**Analysis** needs only the file:

```python
from src.misc.io_utils import load_data

d = load_data("output/HD_sod1Dcart_final.npz")
Ngc = d['Ngc']
rho = d['dens'][Ngc:-Ngc, Ngc:-Ngc]    # fields are stored with ghost cells
x1, x2 = d['cx1'], d['cx2']            # interior cell centres
print(d['mode'], d['problem'], "t =", d['timenow'])
```

The last cells of `main.ipynb` do exactly this and plot the result.

**In your own script** — the same functions, plus two optional arguments of
`run_simulation`:

```python
from src.misc.io_utils import save_data, restart_simulation

# write a snapshot every 0.1 time units
save = lambda st, p, k: save_data(f"runs/khi_{k:04d}", grid, st, p, eos)
state, par.timenow = run_simulation(grid, state, par, solver, lambda: state.dens,
                                    n_plot=20, dt_output=0.1, on_output=save)

# later: continue from one of them
grid, state, par, eos = restart_simulation("runs/khi_0005")
```

Two things cannot change on restart: the resolution and `rec_type` (the
number of ghost cells depends on it). The CFL number and `timefin` can.

---

## Configuration reference

Everything lives in the `Parameters` object. Required: `mode`, `problem`,
`Nx1`, `Nx2`. Optional knobs (with defaults):

| Parameter      | Default  | Meaning                                                            |
|----------------|----------|-------------------------------------------------------------------|
| `CFL`          | `0.7`    | Courant number (capped at 0.4 for `rHD`/`rMHD`)                    |
| `rec_type`     | `"PLM"`  | `PCM`, `PLM`, `PPMorig`, `PPM`, `WENO`, `MP5` (not used by `diff`) |
| `RK_order`     | `"RK2"`  | `RK1`, `RK2`, `RK3` (not used by `diff`)                           |
| `solver_type`  | per mode | Riemann solver, or time integrator for `diff` (table below)        |
| `divb_tr`      | `"GLM"`  | MHD divergence control: `CT`, `GLM`, `8wave`; rMHD always uses `CT` |
| `rkl2_stages`  | `10`     | RKL2 stage count `s >= 2` (`diff` with `solver_type="rkl2"`)       |

Set by the initial-condition function (rarely by hand):

| Attribute        | Meaning |
|------------------|---------|
| `BC`, `BCm`      | boundary types `[x1_inner, x2_inner, x1_outer, x2_outer]`, each `'free'`, `'wall'`, `'peri'` or `'axis'`; `BCm` is the magnetic-field set (MHD/rMHD) |
| `BC_fixed`       | fixed ghost states (inflow patches) on parts of a face — HD, rHD, MHD (GLM/8wave), adv, diff |
| `before_step`    | per-step physics hook `f(grid, state, par, dt)`, or `None` |
| `timenow`, `timefin` | current and final time |

**Solver options per mode** (default first)

```
adv  : adv, LW
HD   : HLLC, LLF, HLL, Roe, Exact
rHD  : HLLC, LLF, HLL
MHD  : HLLD, LLF, HLL, HLLC          divb_tr: GLM (default), CT, 8wave
rMHD : HLL, LLF                      divb_tr: CT (only)
SWE  : HLL, LLF, Exact
diff : rkl2, expl                    rkl2_stages: int >= 2
```

Use `RK_order="RK2"` or `"RK3"` with `rec_type="MP5"`, not `"RK1"`: forward
Euler has the smallest stability margin of the three, and MP5 is
deliberately the least dissipative of the high-order reconstructions, so
the pairing can lose positivity on a strong shock (verified in
`tests/test_robustness.py`; MP5 is fine with RK1 on smooth data, just not
discontinuities). This is the standard reason high-order shock-capturing
schemes are paired with SSP-RK2/RK3 rather than RK1.

---

## The test-problem catalogue

Pass one of these as `problem`. Pass `"user_defined"` in any mode to start
from a template you fill in yourself. The authoritative list is `PROBLEMS`
in `src/misc/helpers.py` (problem name → initial-condition function).

- **`adv`** — `smooth1D`, `disc1D`, `smooth2D`, `disc2D`
- **`HD`** — `sod1Dcart`, `sod1Dcyl`, `sod1Dsph`, `strong1D`, `DBW1D`,
  `shuosher1D`, `einfeldt1D`, `sod2Dcart`, `sod2Dsph`, `sod2Dpol`,
  `sedov2Dcart`, `sedov2Dcyl`, `RP2D`, `gresho2D`, `vortex2D`, `KHI2D`,
  `RTI2D`, `shock-cloud`, `jet2Dcyl`, and with gravity: `gap-opening`
  (star + planet), `jeans1D`, `collapse_sph`, `merger2D` (self-gravity)
- **`rHD`** — `RP1`, `RP3`, `RP4`, `RP5`, `RP2D`, `RTI`, `jet2Dcart`, `jet2Dcyl`
- **`MHD`** — `BW1D`, `toth1D`, `RJ1D`, `alfven1D`, `alfven2D`, `blast2Dcart`,
  `blast2Dcyl`, `blast2Dsph`, `rotor2D`, `OT2D`, `current-sheet`,
  `field-loop`, `disk2D`, `jet2Dcyl`, `shock-cloud`
- **`rMHD`** — `BW1D`, `RP2`, `RP3`, `RP4`, `blast2D`, `rotor2D`, `OT2D`
- **`SWE`** — `dam1D`, `dam2D`, `rotdam2D`, `bathtub2D`, `kelvin2D`,
  `jet2D`, `KHI2D`, `lake2D`
- **`diff`** — `gauss2D`, `step1D`, `sine1D`, `cyl2D`, `gauss2Dpol`, `gauss2Dsph`

---

## Project layout

```
Piastra/
├── main.py                 # script entry point — edit parameters, run
├── main.ipynb              # notebook entry point — mirrors main.py
├── run_testbed.py          # testbed entry point — sanity / conservation /
│                            #   convergence / robustness / restart, see Testbed below
├── tests/                  # the testbed itself (plain test_* functions)
└── src/
│   ├── parameters.py       # Parameters: configuration and defaults
│   ├── sim_state.py        # SimState: per-mode variable storage
│   ├── gravity.py          # body forces: planet, monopole and Poisson self-gravity
│   ├── grid/
│   │   ├── grid_setup.py   # Grid: cart / cyl / pol / sph geometries
│   │   └── grid_misc.py    # divergence, gradient, curl, interpolation, norms
│   ├── common/
│   │   ├── boundaries.py   # scalar / vector / fixed ghost-cell fillers
│   │   ├── high_order_rec.py  # PCM / PLM / PPM / WENO / MP5
│   │   ├── eos_setup.py    # EOSdata (ideal-gas equation of state)
│   │   └── poisson_solver.py  # FV Poisson solve via preconditioned CG
│   ├── misc/
│   │   ├── helpers.py      # PROBLEMS catalogue, initial_model, run_simulation
│   │   ├── io_visual.py    # live matplotlib visualization
|   │   └── io_utils.py     # snapshots: save / load / restart (.npz), 1D text dump
│   └── models/             # one self-contained package per physics mode
│       └── adv/  HD/  rHD/  MHD/  rMHD/  SWE/  diff/
└── notebooks/              # pedagogical notebooks (TBD)
```

Every physics package follows the same rhythm: `*_step.py` (the
time-stepping class and CFL condition), `*_phys.py` (conserved↔primitive maps,
boundary fills, the Riemann-solver dispatcher), `*_init_cond.py` (the test
problems), and `*_riemann_*.py` (the Riemann solvers themselves; not needed
for `adv` and `diff`). MHD has one `*_step` file per divergence-control
scheme. Learn one package and you can read them all.

---

## Testbed

```bash
python run_testbed.py                          # everything (~1-2 minutes)
python run_testbed.py --suite sanity            # one suite
python run_testbed.py --suite sanity,convergence
python run_testbed.py -q                        # only print failures
```

Five suites under `tests/`, the standard checks for a computational
(astrophysical) fluid-dynamics code:

| Suite          | What it checks                                                                  |
|----------------|----------------------------------------------------------------------------------|
| `sanity`       | every `(mode, problem)` in `PROBLEMS` builds and runs a few steps without crashing, NaNs, or unphysical (negative density/pressure/height) values — on a deliberately non-square grid, since a square one hides axis-swap bugs |
| `conservation` | mass / momentum / total energy are constant to round-off on closed domains (periodic HD and MHD, wall-bounded Sedov blast, zero-flux diffusion, a closed SWE basin); CT keeps div(B) at round-off |
| `convergence`  | measured error shrinks at close to the design order under grid refinement: periodic advection in 1D and 2D (return to the initial state), diffusion of a sine mode (exact solution), the Gresho vortex (exact steady state), and circularly-polarized Alfvén waves in 1D and 2D |
| `robustness`   | every `solver_type` × `rec_type` × `RK_order` combination a mode supports stays finite and positivity-preserving, checked after *every* step, on the field's standard strong-shock benchmarks (Woodward-Colella double blast wave, Brio-Wu MHD shock tube, a shallow-water dam break, ...) |
| `restart`      | a run stopped halfway and continued from its snapshot gives the same final state as the uninterrupted run, **bit for bit** — every mode, every MHD div(B) treatment, every problem with gravity; snapshots land exactly on their output times |

Each suite is a plain module of `test_*` functions with plain `assert`
statements — no test-framework dependency — so `tests/` also works with
`pytest tests/` for anyone who prefers that runner.

---

## Selected references

- Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. (2009) — Godunov-type solvers
- Balsara (2017), *Living Rev. Comput. Astrophys.* **3**, 2 — high-order methods and models
- Zingale (2021) *Tutorial on Computational Astrophysics*, https://zingale.github.io/comp_astro_tutorial/intro.html
- Shu & Osher (1988), *JCP* **77**, 439 — TVD (SSP) Runge–Kutta
- Colella & Woodward (1984), *JCP* **54**, 174 — PPM
- Mignone (2014), *JCP* **270**, 784 — high-order curvilinear reconstruction
- Suresh & Huynh (1997), *JCP* **136**, 83 — MP5 reconstruction
- Miyoshi & Kusano (2005), *JCP* **208**, 315 — HLLD for MHD
- Tóth (2000), *JCP* **161**, 605 — div(B) treatments for MHD
- Dedner et al. (2002), *JCP* **175**, 645 — GLM divergence cleaning
- Balsara & Spicer (1999), *JCP* **149**, 270 — constrained transport
- Mignone & Bodo (2005), *MNRAS* **364**, 126 — relativistic HLLC
- Mignone & Bodo (2006), *MNRAS* **368**, 1040 — relativistic MHD
- Meyer, Balsara & Aslam (2012), *MNRAS* **422**, 2102 — RKL2 super-time-stepping
- Shewchuk (1994), *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain* — preconditioned CG

---

**Author:** mrkondratyev · [github.com/mrkondratyev/Piastra](https://github.com/mrkondratyev/Piastra)
