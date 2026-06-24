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

| Mode   | Physics                                            | Solvers                      |
|--------|----------------------------------------------------|------------------------------|
| `adv`  | Linear scalar advection                            | upwind, Lax–Wendroff         |
| `HD`   | Compressible (Euler) hydrodynamics                 | LLF, HLL, HLLC, Roe, Exact   |
| `rHD`  | Special-relativistic hydrodynamics                 | LLF, HLL, HLLC               |
| `MHD`  | Ideal magnetohydrodynamics                         | LLF, HLL, HLLC, HLLD         |
| `rMHD` | Special-relativistic MHD                           | LLF, HLL                     |
| `SWE`  | Shallow-water equations (geophysical flows)        | LLF, HLL, Exact              |
| `diff` | Thermal diffusion ∂ₜT = ∇·(κ∇T)                    | — (parabolic)                |

All of it runs on **1D and 2D structured grids** in four geometries —
Cartesian `(x, y)`, cylindrical `(R, z)`, polar `(R, φ)`, and spherical-polar
`(r, θ)` — with face areas, cell volumes, and curvilinear source terms handled
consistently so the same solver code works in every coordinate system.

---

## How it's built (the numerics)
***HYPERBOLIC***  

- **Godunov finite volumes.** States are reconstructed to cell faces, a Riemann
  solver returns the interface flux, and the conservative update is the
  divergence of those fluxes — the textbook recipe, applied uniformly.
- **High-order reconstruction:** `PCM` (1st), `PLM` (2nd, slope-limited),
  `PPMorig`, `PPM` (Mignone 2014), `WENO5` (Jiang & Shu 1996), `MP5`
  (Suresh & Huynh 1997). Ghost-cell count is chosen automatically (2 for
  PCM/PLM, 3 for the rest).
- **Time integration:** TVD Runge–Kutta `RK1`/`RK2`/`RK3` (Shu & Osher 1988).
- **Divergence control for MHD:** Constrained Transport (`CT`), hyperbolic
  divergence cleaning (`GLM`), and Powell's 8-wave method (`8wave`).
- **Relativity:** reconstruction on the 4-velocity (guaranteeing |v| < 1 at
  faces) with a Newton–Raphson conservative-to-primitive inversion.

***PARABOLIC***  
- **Diffusion:** explicit Euler or RKL2 super-time-stepping (Meyer, Balsara &
  Aslam 2014), which buys an ~s²/4 speed-up at second-order accuracy.

---

## Quickstart

Piastra is a package rooted at `src/`. Run it from the repository root.

**The fast path** — edit the parameters at the top of `main.py` and run it:

```bash
python main.py
```

or open `main.ipynb` for the same workflow with live, re-runnable cells.

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

# 3. Pick a solver and march in time (plotting every Nplot steps)
solver = HD2D(grid, state, eos, par)
state, par.timenow = run_simulation(grid, state, par, solver, state.dens, n_plot=20)
```

The initial-condition function chooses the geometry, fills the primitive
variables, sets boundary conditions, and returns the final time — so a single
`problem` string fully specifies a test case.

---

## Configuration reference

Everything lives in the `Parameters` object. Required: `mode`, `problem`,
`Nx1`, `Nx2`. Optional knobs (with defaults):

| Parameter      | Default  | Meaning                                                            |
|----------------|----------|-------------------------------------------------------------------|
| `CFL`          | `0.7`    | Courant number (auto-capped at 0.4 for relativistic modes)        |
| `rec_type`     | `"PLM"`  | `PCM`, `PLM`, `PPMorig`, `PPM`, `WENO`, `MP5`                      |
| `RK_order`     | `"RK2"`  | `RK1`, `RK2`, `RK3`                                                |
| `solver_type`  | per mode | Riemann solver (or time-stepping for diffusion) (see table below)  |
| `divb_tr`      | `"GLM"`  | MHD divergence control: `CT`, `GLM`, `8wave` (MHD); `CT` (rMHD)    |
| `rkl2_stages`  | `10`     | RKL2 stage count (diffusion, `solver_type="rkl2"`)                |

**Solver options per mode**

```
adv  : adv, LW
HD   : LLF, HLL, HLLC, Roe, Exact
rHD  : LLF, HLL, HLLC
MHD  : LLF, HLL, HLLC, HLLD          divb_tr: CT, GLM, 8wave
rMHD : LLF, HLL                      divb_tr: CT (only)
SWE  : LLF, HLL, Exact
diff : expl, rkl2                    rkl2_stages: int >= 2
```

---

## The test-problem catalogue

Pass one of these as `problem`. Pass `"user_defined"` in any mode to drop into a
template you fill in yourself.

- **`adv`** — `smooth1D`, `disc1D`, `smooth2D`, `disc2D`
- **`HD`** — `sod1Dcart`, `sod1Dcyl`, `sod1Dsph`, `strong1D`, `DBW1D`,
  `shuosher1D`, `einfeldt1D`, `sod2Dcart`, `sod2Dsph`, `sod2Dpol`,
  `sedov2Dcart`, `sedov2Dcyl`, `RP2D`, `gresho2D`, `KHI2D`, `RTI2D`,
  `shock-cloud`, `gap-opening`, `jet2Dcyl`
- **`rHD`** — `RP1`, `RP3`, `RP4`, `RP5`, `RP2D`, `RTI`, `jet2Dcart`, `jet2Dcyl`
- **`MHD`** — `BW1D`, `toth1D`, `RJ1D`, `alfven1D`, `blast2Dcart`, `blast2Dcyl`,
  `blast2Dsph`, `rotor2D`, `OT2D`, `current-sheet`, `field-loop`, `disk2D`,
  `shock-cloud`
- **`rMHD`** — `BW1D`, `RP2`, `RP3`, `RP4`, `blast2D`, `rotor2D`
- **`SWE`** — `dam1D`, `bump1D`, `bathtub2D`, `expl2D`, `tsunami2D`, `ocean2D`,
  `atmo2D`, `dam2D`, `jet2D`, `KHI2D`
- **`diff`** — `gauss1D`, `gauss2D`, `step1D`, `sine1D`, `cross2D`, `ring2D`,
  `cyl2D`

---

## Project layout

```
Piastra/
├── main.py                 # script entry point — edit parameters, run
├── main.ipynb              # notebook entry point — mirrors main.py
├── src/
│   ├── parameters.py       # Parameters: config, defaults, validation
│   ├── sim_state.py        # SimState: unified per-mode variable storage
│   ├── grid/
│   │   ├── grid_setup.py   # Grid: cart / cyl / pol / sph geometries
│   │   └── grid_misc.py    # divergence, gradient, interpolation, norms
│   ├── common/
│   │   ├── boundaries.py   # scalar / vector / fixed ghost-cell fillers
│   │   ├── high_order_rec.py  # PCM / PLM / PPM / WENO / MP5
│   │   └── eos_setup.py    # EOSdata (ideal-gas equation of state)
│   ├── misc/
│   │   ├── helpers.py      # initial_model dispatch + run_simulation loop
│   │   ├── io_visual.py    # live matplotlib visualization
│   │   └── io_utils.py     # snapshot save/load, 1D ASCII export
│   └── models/             # one self-contained package per physics mode
│       ├── adv/  HD/  rHD/  MHD/  rMHD/  SWE/  diff/
│       └── ...             # each: *_step, *_phys, *_init_cond + optionals
└── notebooks/              # standalone pedagogical notebooks
```

Every physics package follows the same four-file rhythm: `*_step.py` (the
time-stepping class and CFL condition), `*_phys.py` (conserved↔primitive maps,
boundary fills, the flux driver), `*_riemann_*.py` (the solvers), and
`*_init_cond.py` (the test problems). Learn one package and you can read them all.

---

## Requirements

- Python 3.9+
- NumPy
- matplotlib
- IPython (for the notebooks)

```bash
pip install numpy matplotlib ipython
```

No compilation, no external solver libraries, no configuration files.

---

## Selected references

- Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. (2009) — Godunov-type solvers basics 
- Balsara (2017) Living Reviews in Computational Astrophysics, 3:2 — high-order methods and models
- Toth (2000), *JCP* — divB tretments for MHD
- Dedner et al (2002), *JCP* — GLM divB cleaning for MHD
- Mignone & Bodo (2005), *MNRAS* **364**, 126 — relativistic HLLC
- Miyoshi & Kusano (2005), *JCP* **208**, 315 — HLL-type solvers for MHD
- Shu & Osher (1988), *JCP* **77**, 439 — TVD Runge–Kutta
- Mignone (2014), *JCP* **270**, 784 — high-order curvilinear reconstruction
- Meyer, Balsara & Aslam (2014), *MNRAS* **422**, 2102 — RKL2 super-time-stepping for diffusion

---

**Author:** mrkondratyev · [github.com/mrkondratyev/Piastra](https://github.com/mrkondratyev/Piastra)
