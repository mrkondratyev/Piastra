# Piastra — Developer Documentation

This is the deep-dive companion to [`README.md`](README.md). The README sells
the project and gets you running a simulation in five minutes; this document
is for when you want to know exactly what an object contains, how a piece of
the solver actually works, or how to bend the framework toward a problem it
wasn't shipped with. It assumes you've read the README once.

## Table of contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [Core objects](#2-core-objects)
3. [The ghost-cell / interior-cell convention](#3-the-ghost-cell--interior-cell-convention)
4. [Numerical building blocks](#4-numerical-building-blocks)
5. [Boundary conditions](#5-boundary-conditions)
6. [The Poisson solver](#6-the-poisson-solver)
7. [Physics modes, file by file](#7-physics-modes-file-by-file)
8. [Gravity and other body forces](#8-gravity-and-other-body-forces)
9. [Saving, loading, and restarting](#9-saving-loading-and-restarting)
10. [Visualization](#10-visualization)
11. [The testbed](#11-the-testbed)
12. [Use cases / recipes](#12-use-cases--recipes) — including
    [the astrophysics problems](#128-the-astrophysics-problems)
13. [Extending the framework](#13-extending-the-framework)
14. [Conventions cheat-sheet](#14-conventions-cheat-sheet)

---

## 1. Architecture at a glance

Every run — whether launched from `main.py`, a notebook, or a hand-rolled
script — follows the same five-stage pipeline:

```
Parameters  ──►  Grid  ──►  SimState  ──►  initial_model()  ──►  Solver.step_RK()
 (config)      (mesh +      (field       (dispatches to the      (one Godunov /
                metric)      storage)     chosen IC function,     RKL2 / CG step,
                                          which also builds        called in a loop
                                          the geometry and EOS)     by run_simulation)
```

- **`Parameters`** (`src/parameters.py`) is a plain config object: mode,
  problem name, resolution, reconstruction/solver/RK choices, CFL, boundary
  condition arrays. It validates its own inputs and fills in per-mode
  defaults (e.g. `Ngc` from `rec_type`, `divb_tr` for MHD).
- **`Grid`** (`src/grid/grid_setup.py`) allocates the metric arrays (face
  areas, cell volumes, coordinates) for a given resolution, then one of its
  four geometry methods (`CartesianGrid`, `CylindricalGrid`, `PolarGrid`,
  `SphericalPolarGrid`) fills them in. Nothing downstream branches on
  geometry again — every solver and helper reads `grid.fS1/fS2/cVol/hx2`
  and gets the right answer in all four coordinate systems for free.
- **`SimState`** (`src/sim_state.py`) allocates the per-mode array set
  (primitive, conservative, magnetic, ...). It is pure storage — no
  behaviour.
- **`initial_model`** (`src/misc/helpers.py`) dispatches `(par.mode,
  par.problem)` to the matching `IC_*` function, which is the single place
  that decides the domain, geometry, primitive fields, boundary conditions,
  and `par.timefin` for that test problem. This is also where the `EOSdata`
  object is built for the modes that need one.
- The **solver class** (one per mode, e.g. `HD2D`, `MHD2D_CT`,
  `rHD2D`, `SWE2D`, `Diff2D`) owns `step_RK()`: reconstruct → Riemann
  solve → flux-difference update → boundary refill, wrapped in a TVD
  Runge-Kutta stage loop (or RKL2 sub-stepping for `diff`). `main.py`'s
  `SOLVER_DISPATCH` dict maps a mode string to the right constructor; reuse
  it instead of hard-coding the class name if you're writing something
  mode-generic (the testbed does exactly this).
- **`run_simulation`** (`src/misc/helpers.py`) is the `while par.timenow <
  par.timefin: state = solver.step_RK()` loop with periodic plotting bolted
  on. You don't have to use it — see [§12.4](#124-drive-the-time-loop-by-hand)
  for driving `step_RK()` yourself.

## 2. Core objects

### `Grid` (`src/grid/grid_setup.py`)

| Attribute | Shape | Meaning |
|---|---|---|
| `Nx1, Nx2` | scalar | real (non-ghost) cell counts |
| `Ngc` | scalar | ghost-cell count per side (2 for PCM/PLM, 3 otherwise, 1 for `diff`) |
| `Nx1r, Nx2r` | scalar | last real-cell index + 1 (`= Nx + Ngc`) |
| `grid_shape` | tuple | `(Nx1+2*Ngc, Nx2+2*Ngc)` — the shape of every ghost-inclusive array |
| `fx1, fx2` | `(Nx1+2Ngc+1, ·)`, `(·, Nx2+2Ngc+1)` | face coordinates |
| `cx1, cx2` | `grid_shape` | cell-centre coordinates (ghost-inclusive) |
| `dx1, dx2` | `grid_shape` | local cell widths |
| `ax1, ax2` | `grid_shape` | volumetric centroids (used where the geometric and arithmetic cell centre differ, e.g. radial cells) |
| `fS1, fS2` | `(Nx1+1, Nx2)`, `(Nx1, Nx2+1)` | face areas, **interior-only, no ghosts** |
| `fS3` | `(Nx1, Nx2)` | face area ⟂ the third (out-of-plane) direction, used by CT |
| `cVol` | `(Nx1, Nx2)` | cell volumes, **interior-only, no ghosts** |
| `edg1, edg2, edg3` | various | edge lengths, used by CT's Stokes-theorem curl |
| `hx2` | `grid_shape` | Lamé/metric factor for the x2 direction: `1` for cart/cyl, `cx1` (=R or r) for pol/sph |
| `geom` | str | `'cart'`, `'cyl'`, `'pol'`, `'sph'` |

Build a grid in two steps — allocate, then pick a geometry:

```python
grid = Grid(Nx1=128, Nx2=64, Ngc=2)
grid.CartesianGrid(x1ini=0.0, x1fin=1.0, x2ini=0.0, x2fin=0.5)
# or grid.CylindricalGrid / grid.PolarGrid / grid.SphericalPolarGrid(...)
```

In practice you never call this yourself for a catalogue problem — the
`IC_*` function does it (that's *why* `problem` alone is enough to fully
specify a case). You only build a `Grid` by hand when writing a new IC or
using the framework as a library (see [§12](#12-use-cases--recipes)).

`grid_setup.py` also exposes `reconstruct_grid(Nx1, Nx2, Ngc, geom, x1ini,
x1fin, x2ini, x2fin)` — a module-level function, not a `Grid` method — that
rebuilds a grid
deterministically from just those seven numbers. `io_utils.restart_simulation`
uses it so a saved run doesn't need to serialize any metric arrays.

### `Parameters` (`src/parameters.py`)

Holds everything that configures *how* a run advances, as opposed to *what
physical state* it's in (that's `SimState`'s job). Required: `mode`,
`problem`, `Nx1`, `Nx2`. See the README's
[Configuration reference](README.md#configuration-reference) for the full
option table — that table is authoritative, this document won't duplicate
it. Two things worth calling out that aren't in the README:

- `par.BC_fixed` is a `{0,1,2,3: [(start, end, {field: value}), ...]}` dict
  for pinning a *sub-range* of one boundary to fixed values (e.g. an inflow
  patch on part of a wall) — see `boundaries.apply_bc_fixed`. It's empty by
  default; most `IC_*` functions never touch it.
- `par.BCm` exists only for MHD/rMHD (magnetic-field boundary conditions,
  independent of `par.BC` for the hydro variables) — `None` for every other
  mode.

### `SimState` (`src/sim_state.py`)

Allocates a different attribute set per `par.mode` — see the module
docstring for the full per-mode attribute list, or just read
`src/sim_state.py` directly, it's short and literal (no metaprogramming).
The one thing you must internalize before touching any array on this object
is [§3](#3-the-ghost-cell--interior-cell-convention) below.

### `EOSdata` (`src/common/eos_setup.py`)

A one-parameter ideal-gas EOS: `EOSdata(GAMMA)`. Exposes
`sound_speed_nr(dens, pres)` / `sound_speed_sr(dens, pres)` (non-relativistic
and special-relativistic sound speed) and `eint(dens, pres)` /
`pres(dens, eint)` (internal-energy ↔ pressure, both trivial for an ideal
gas but kept as methods so the modes' `*_phys.py` files never hard-code
`GAMMA - 1`). `None` for `adv`, `diff`, `SWE` — those modes have no equation
of state.

## 3. The ghost-cell / interior-cell convention

This is the single most important thing to know before writing code against
this framework, and the source of every non-obvious bug fixed during this
project's development. Two array shapes coexist:

- **Ghost-inclusive**, shape `grid.grid_shape = (Nx1+2*Ngc, Nx2+2*Ngc)`:
  every *primitive* field — `dens`, `pres`, `vel1/2/3`, `bfi1/2/3`, `T`, `h`
  — plus geometric coordinate arrays (`cx1`, `cx2`, `dx1`, `dx2`, `ax1`,
  `ax2`, `hx2`). Indexing the real domain out of one of these needs the
  ghost offset: `field[Ngc:-Ngc, Ngc:-Ngc]`.
- **Interior-only**, shape `(Nx1, Nx2)`, **no ghost cells at all**: every
  *conservative* field — `mass`, `mom1/2/3`, `etot`, `bcon1/2/3`, `F1`,
  `F2`, `divB`, `glmcon` — and the geometric **face/volume** arrays `fS1`,
  `fS2`, `fS3`, `cVol`. These must be indexed directly, `field[:, :]` or
  `field[i, j]` — slicing them with a ghost offset silently produces
  garbage (wrong cells) or an `IndexError` on a non-square grid.

Every real bug caught by this project's testbed traced back to applying the
wrong one of these two slicing patterns to `grid.cVol` (it looks exactly
like a ghost-inclusive array from the shape alone if your grid happens to be
square — which is why the testbed deliberately uses non-square resolutions
everywhere). When you write new code that touches `cVol`, `fS1`, or `fS2`,
slice them plainly; when you touch `dens`, `cx1`, or `pres`, use the
`Ngc:-Ngc` offset.

## 4. Numerical building blocks

`src/grid/grid_misc.py` holds the geometry-aware finite-volume operators
every solver (hyperbolic, parabolic, elliptic) is built from. None of them
branch on `grid.geom` — the geometry lives entirely in the grid's metric
arrays (`fS1`, `fS2`, `cVol`, `hx2`), so these functions are correct in all
four coordinate systems by construction.

| Function | Signature | What it does |
|---|---|---|
| `interp_face_to_cell` | `(grid, fV1, fV2) -> V1, V2` | distance-weighted interpolation of a staggered vector field to cell centres |
| `div_face_vector` | `(grid, fV1, fV2) -> divV` | Gauss's-theorem divergence of a face-centred vector field (used by CT and the Poisson solver) |
| `div_cell_vector` | `(grid, V1, V2) -> divV` | divergence of a cell-centred vector field via face-averaging |
| `cell_gradient` | `(grid, f) -> g1, g2` | second-order gradient of a cell-centred scalar, **on cell centres**, metric-corrected via `hx2` |
| `face_gradient` | `(grid, f) -> g1, g2` | same, but evaluated **on faces** (shapes `(Nx1+1,Nx2)` / `(Nx1,Nx2+1)`) — what the Poisson operator and the diffusion solver use |
| `edge_to_face_curl` | `(grid, edg_var) -> fV1, fV2` | discrete Stokes-theorem curl of an out-of-plane edge scalar, producing a solenoidal-by-construction staggered field — used to seed CT's initial `fb1/fb2` from a vector potential, and inside the CT update itself |
| `Ln_norm` | `(grid, n, var_num, var_ref) -> float` | volume-weighted `sum(cVol * (num-ref)^n)`, for convergence testing |
| `integral_over_grid` | `(grid, var) -> float` | volume integral `sum(cVol * var)` of a ghost-inclusive field |

`cell_gradient` vs. `face_gradient`: use `cell_gradient` when you need a
gradient sampled at the same locations as the input field (e.g. converting
a potential to a cell-centred body-force — see `gravity.selfgravity_poisson`);
use `face_gradient` when you need it at the same staggered locations a
finite-volume flux divergence expects (e.g. `poisson_operator`'s
`-div(grad(phi))`, or projecting a face-normal correction for divergence
cleaning).

`src/common/high_order_rec.py` implements the reconstruction stencils
(`PCM`, `PLM` with 6 limiter choices, `PPMorig`, `PPM`, `WENO`, `MP5`)
behind one dispatcher, `VarReconstruct(var, rec_type, limiter_type, axis)`,
called once per characteristic variable per stage by every hyperbolic
solver's flux routine.

## 5. Boundary conditions

Two independent BC systems exist, with **different vocabularies** — don't
mix them up:

- **Hyperbolic solvers** (`par.BC`, `par.BCm`): each of the 4 faces is one
  of `'free'` (zero-gradient), `'wall'` (reflective, normal component
  flips), `'peri'` (periodic), `'axis'` (reflective with an azimuthal-sign
  flip, for the coordinate-singularity face of cylindrical/spherical
  grids). Filled by `boundaries.apply_bc_scalar` / `apply_bc_vector`,
  called once per RK stage by each mode's `*_phys.py`. Fills **all** `Ngc`
  ghost layers, since high-order reconstruction stencils read more than one.
- **The Poisson solver** (`BC` argument to `solve_poisson`): each face is
  one of `'peri'`, `'free'`, or `'dirichlet'` (a genuine fixed value —
  there is no `'wall'`/`'axis'` here, since the Laplacian doesn't
  distinguish a reflecting wall from a zero-gradient face). Filled by
  `boundaries.apply_bc_scalar_Ngc1`, which only touches the **one** ghost
  layer bordering the real domain — the natural width of a 3-point
  second-order stencil, and deliberately not more, so a deeper-ghosted
  hydro/MHD grid (Ngc=2 or 3) can be handed straight to the Poisson solver
  without disturbing its own ghost layers.

Face indexing convention, shared by both systems: `BC[0]` = x1 inner,
`BC[1]` = x2 inner, `BC[2]` = x1 outer, `BC[3]` = x2 outer.

`boundaries.apply_bc_fixed(state_fields, Ngc, N1, N2, face, patches)` is a
third, narrower tool: it pins ghost cells on a *sub-range* of one face to
literal values, for problems needing a partial inflow boundary rather than
a uniform one across the whole face — driven by `par.BC_fixed`.

## 6. The Poisson solver

`src/common/poisson_solver.py` solves `div(grad(phi)) = rhs` on any grid
(1D or 2D, any of the four geometries) with matrix-free,
diagonally-preconditioned Conjugate Gradient. It's deliberately generic —
not tied to gravity or divergence-cleaning — so any module can call it for
an elliptic sub-problem.

```python
phi, info = solve_poisson(grid, rhs, BC, BC_value=None, phi0=None,
                           tol=1e-10, maxiter=None, verbose=False)
```

- `rhs` — interior-only `(Nx1, Nx2)` source term.
- `BC` — 4-entry list, `'peri'` / `'free'` / `'dirichlet'` per face (see
  [§5](#5-boundary-conditions)); opposite faces must agree on `'peri'`.
- `BC_value` — `{face_index: value}` dict for `'dirichlet'` faces (defaults
  to 0.0 if omitted).
- Returns `phi` as a full `grid.grid_shape` array (ghost layer
  already filled, ready for `cell_gradient`/`face_gradient`) and an `info`
  dict `{'niter', 'residual', 'converged'}`.

Implementation notes, if you're extending it or debugging a convergence
failure:

- The operator `A(phi) = -div(grad(phi))` (`poisson_operator`) reuses
  `face_gradient` + `div_face_vector` — the same building blocks as the
  diffusion solver — so it's automatically consistent with the grid metric.
- CG's inner product is volume-weighted, `sum(cVol*u*v)` (`_dot`), which is
  what makes the finite-volume operator self-adjoint — a plain unweighted
  dot product would break CG's convergence guarantee on a non-uniform grid.
- The Jacobi (diagonal) preconditioner (`_diag_operator`) is built from
  closed-form face conductances (`_face_conductance`), not by
  finite-differencing `A` — cheap, and exact.
- Pure-Neumann / pure-periodic problems (no `'dirichlet'` face anywhere)
  only fix `phi` up to an additive constant and are only solvable if
  `rhs`'s volume integral is zero; `solve_poisson` enforces this itself by
  subtracting the volume-weighted mean of `rhs` before solving.
- Inhomogeneous Dirichlet BCs use the standard CG "lifting" trick:
  `poisson_operator` is called with the real (inhomogeneous) `BC_value` only
  when evaluating the residual `b - A(phi_int)`; every other application of
  `A` inside the CG loop (the search-direction matrix-vector products) uses
  the homogeneous form (`BC_value=None`), which is what makes `A`
  linear/self-adjoint on the vectors CG actually iterates over.

## 7. Physics modes, file by file

Every mode's package follows the same four-file rhythm (the README already
states this; here's what's actually in each file):

| File | Contents |
|---|---|
| `*_step.py` | The solver class (`step_RK()`, the CFL/`dt` calculation, the RK-stage loop) |
| `*_phys.py` | Primitive ↔ conservative conversion, the boundary-condition call, the flux/residual driver called once per stage |
| `*_init_cond.py` | Every `IC_*` test-problem function for that mode, plus `user_defined` |
| `*_riemann_*.py` | The Riemann solver(s) (approximate and, for HD/SWE, exact) |

Solver class → file → mode, for `SOLVER_DISPATCH` lookups:

| Mode | Class | File |
|---|---|---|
| `adv` | `Adv2D` | `src/models/adv/adv_step.py` |
| `HD` | `HD2D` | `src/models/HD/HD_step.py` |
| `rHD` | `rHD2D` | `src/models/rHD/rHD_step.py` |
| `MHD` | `MHD2D_CT` / `MHD2D_GLM` / `MHD2D_8wave` | `src/models/MHD/MHD_step_{CT,GLM,8wave}.py` (picked by `par.divb_tr`) |
| `rMHD` | `rMHD2D_CT` | `src/models/rMHD/rMHD_step.py` (CT only) |
| `SWE` | `SWE2D` | `src/models/SWE/SWE_step.py` |
| `diff` | `Diff2D` | `src/models/diff/diff_step.py` |

Every solver constructor takes `(grid, state, eos, par)` except `adv`, `SWE`,
and `diff`, which have no equation of state and take `(grid, state, par)` —
`SOLVER_DISPATCH` in `main.py` already absorbs this asymmetry behind a
uniform `lambda grid, state, eos, par: ...` per entry, so callers never need
to special-case it.

The full test-problem catalogue (which `problem` string maps to which
`IC_*` function, for every mode) is in the README and mirrored verbatim in
`main.py`'s module docstring — not repeated a third time here.

## 8. Gravity and other body forces

`src/gravity.py` provides the body-force sources that feed every
HD/MHD-family solver's momentum and energy residual through `state.F1` and
`state.F2`.

### Static vs. per-stage forces

A force that never changes — a fixed central point mass — can be written
into `state.F1`/`F2` once by the IC function; the arrays persist and the RK
integrators' deep copies carry them along.

A force that depends on the **evolving solution** (self-gravity: ρ changes
every stage; Coriolis: **v** changes every stage) or explicitly on **time**
(an orbiting perturber) must be recomputed every Runge-Kutta stage. That's
what the optional `state.body_force` hook is for: a callable
`body_force(grid, state, par)` invoked at the top of each stage's residual
evaluation, on *that stage's* state. Build one with the `*_hook` factories:

```python
state.body_force = selfgravity_poisson_hook(G=1.0, BC=['peri'] * 4)
```

The hook is a plain closure, which `copy.deepcopy` treats as atomic, so the
per-stage deep copy shares it rather than duplicating what it captured. It
is **not** serialized by `save_data` (only arrays and scalars are), so a run
resumed via `restart_simulation` must re-install its hook — `save_data`
prints a warning when the state carries one.

### The gravitational timestep limit

An ordinary CFL condition limits `dt` by the *signal* speed `|v| + c_s`. In a
cold, self-gravitating flow released from rest both terms are ≈ 0, so the
hydrodynamic CFL imposes essentially **no limit** — a pressureless collapse
will take one enormous step and integrate the whole run in a single forward
Euler update, producing a smooth, plausible, completely wrong answer. Gravity
accelerates the gas without any wave carrying information about it, so it
must supply its own constraint:

```
(1/2)|a| dt² ≤ CFL·dx   ⟹   dt ≤ sqrt(2·CFL·dx / |a|)
```

`gravity.body_force_dt(grid, state, CFL)` evaluates this per cell, and every
`CFLcondition_*` routine takes the `min` of it and the hydrodynamic limit.
It returns `inf` when the force is zero, so problems without gravity are
unaffected.

**Sign convention** (spelled out in full at the top of `gravity.py`): every
solver applies the body force as `Res += -dens*F`, and the update is
`U_new = U_old - dt*Res`, so the net effect is `+dt*dens*F`. For that to be
the physical force per unit mass, **`F1`/`F2` must equal the acceleration
directly** (`F = a = -grad(Phi)`), not its negative. All three functions
below follow this convention; if you write a fourth one, follow it too — an
accidental sign flip here turns attraction into repulsion silently (no
crash, no NaN, just a slowly-unbinding "self-gravitating" cloud).

### The routines

| Function | Use case | Method |
|---|---|---|
| `planet_gravity_polar(...)` | star + orbiting planet, **lab frame**, polar grid | closed-form softened point-mass + indirect term, no elliptic solve. Time-dependent → needs the hook |
| `corotating_planet_disk(...)` | disk-planet / gap opening, **co-rotating frame**, polar grid | static planet potential + centrifugal + Coriolis. Velocity-dependent → needs the hook |
| `selfgravity_monopole_spherical(grid, state, par)` | nearly-spherical self-gravitating object, spherical-polar grid | angular-averaged (l=0) enclosed-mass integration, no elliptic solve |
| `selfgravity_poisson(grid, state, par, G=1.0, BC=None, BC_value=None, tol=1e-10, maxiter=None)` | general self-gravity, any density field, any geometry | actual elliptic solve via `solve_poisson`, `Phi` differentiated with `cell_gradient` |
| `body_force_dt(grid, state, CFL)` | timestep limit from any body force | `sqrt(2·CFL·dx/|a|)`, called by every `CFLcondition_*` |

Each of the first four has a matching `*_hook(...)` factory returning a
`state.body_force` callable.

### Choosing the Poisson boundary condition

This is the part that is easy to get silently wrong, because both choices
run without complaint:

- **Periodic box** (`['peri']*4`): a pure-Neumann problem is solvable only if
  the source has zero volume integral, so `solve_poisson` subtracts the
  volume-weighted mean of `rhs`. For self-gravity that subtraction *is* the
  Jeans swindle — exactly right, and not a fudge.
- **Isolated body**: that same mean subtraction is *wrong* — it adds a
  uniform negative background density and changes the enclosed mass. Use a
  `'dirichlet'` face carrying the exterior potential instead: `-G·M/r` for a
  point mass (`collapse1D`), or the monopole `-G·M/sqrt(R²+z²)` evaluated on
  each face (`collapse2D`). `'free'` is exact on a symmetry axis (`r=0`),
  where `dΦ/dr = 0`.

On a strongly graded mesh (spherical `cVol ~ r²dr`) the operator is poorly
conditioned near the origin and CG needs more than its default cap of
`Nx1*Nx2` iterations — pass `maxiter` explicitly, or an unconverged
potential will be returned silently.

`selfgravity_poisson` solves `div(grad(Phi)) = 4*pi*G*rho` and sets
`F1, F2 = -grad(Phi)` **on interior cells only** — matching `state.F1`/`F2`'s
existing `(Nx1, Nx2)` shape, so no ghost-cell values are ever computed or
needed for it. It defaults to `BC=['free']*4`, which is safe even though a
pure-Neumann Poisson problem leaves `Phi`'s normalization undetermined,
because only `Phi`'s gradient is ever used. See [§12.5](#125-self-gravity-in-a-manual-time-stepping-loop)
for a worked example wiring it into a loop.

## 9. Saving, loading, and restarting

`src/misc/io_utils.py` is one writer, one reader, and a restart helper —
mode-agnostic, since it dumps every `SimState` attribute rather than
special-casing per mode.

```python
save_data(filepath, grid, state, par, eos=None)       # -> filepath actually written (.npz appended if missing)
load_data(filepath) -> dict                             # flat dict, e.g. d['dens'], for offline analysis
restart_simulation(filepath) -> (grid, state, par, eos)  # live objects, ready for a solver
save_1d_ascii(filepath, grid, state, par, eos=None)      # optional plain-text 1D profile dump
```

A single `.npz` archive stores run metadata (mode, problem, time, CFL,
reconstruction/RK/solver choices, BCs), the grid's construction parameters
(geometry + bounds + resolution — the grid is *rebuilt*, not serialized, via
`grid_setup.reconstruct_grid`), and every array/scalar on `SimState`
(including CT's staggered `fb1`/`fb2` and GLM's `bglm`, which can't be
recovered from cell-centred primitives alone — an exact restart needs them).
`load_data` mirrors `save_data`'s `.npz` auto-append (it tries the bare path
first, then `path + ".npz"`), so `load_data(p)` always works with the same
`p` you originally passed to `save_data(p, ...)`, whether or not you typed
the extension.

## 10. Visualization

`src/misc/io_visual.py` is deliberately small: `plot_setup(grid, var, time)`
builds the right kind of matplotlib figure for the grid (a line plot in 1D;
`imshow` for 2D Cartesian/cylindrical; `pcolormesh` on the mapped `(x,y)` or
`(R,z)` vertices for 2D polar/spherical-polar, since those geometries don't
live on a rectangular pixel grid), and `plotting(...)` updates it in place
each call. `run_simulation` calls both for you; call them yourself only if
you're driving the time loop by hand (see [§12.4](#124-drive-the-time-loop-by-hand)).

## 11. The testbed

See the README's [Testbed](README.md#testbed) section for the day-to-day
`python run_testbed.py` usage and the four-suite table. This section is for
extending it.

`tests/testbed_common.py` is the shared plumbing: `build_case(mode, problem,
Nx1, Nx2, **par_kwargs)` builds a full `(grid, state, par, eos, solver)`
tuple the same way `main.py` does (reusing `SOLVER_DISPATCH`, so a new mode
registered there is automatically usable by the testbed too), and
`run_steps`/`run_to_tfin` advance it. Every case in the testbed deliberately
uses `Nx1 != Nx2` — a square grid hides axis-swap bugs, which is exactly how
the `cVol` mis-slicing bugs fixed during this project's development went
unnoticed for as long as they did.

To add a new sanity case: add an entry to the `CATALOGUE` dict at the top of
`test_sanity.py` — every `(mode, problem)` pair listed there is picked up
automatically. Robustness cases are opted in explicitly instead (a stress
sweep is expensive, so not every catalogue problem gets one): call
`_register_cross_product(_REGISTRY, prefix, mode, problem, Nx1, Nx2,
solver_types, ...)` at module level in `test_robustness.py`, and it
generates one `test_*` function per `solver_type × rec_type × RK_order`
combination for that problem (skipping the documented `MP5`+`RK1`
exception) — you still name the problem once, but not the ~90-combination
cross product.

To add a new convergence check: write a `test_*` function in
`test_convergence.py` that runs the same problem at 2-3 increasing
resolutions, measures `testbed_common.l2_error` against a known solution (or
against itself at `t=0` for an exact-return-to-initial-state problem like
periodic advection), and asserts `testbed_common.observed_order(errors)` is
close to the reconstruction's design order. The MHD Alfvén-wave convergence
tests (`test_mhd_alfven_1d_convergence`, `test_mhd_alfven_2d_convergence` in
`test_convergence.py`) are a template: a circularly-polarized Alfvén wave on
a periodic domain returns exactly to its initial state after one period, so
the error is just the L2 distance from `t=0`.

Every suite is plain `test_*` functions with plain `assert` — no pytest
dependency to run `run_testbed.py`, but `pytest tests/` works too if you
prefer that runner or want `-k`/`-x` filtering.

## 12. Use cases / recipes

### 12.1. Run a built-in problem

The two ways to do this (fast path via `main.py`, explicit path in Python)
are in the [README's Quickstart](README.md#quickstart) — not repeated here.

### 12.2. Write a custom initial condition

Every mode ships a `user_defined` problem as a fill-in-the-blanks template —
pick the one for your mode (e.g. `src/models/HD/HD_init_cond.py`'s
`IC_HD_user_defined`), copy it under a new name, register it in the
matching dispatch dict in `src/misc/helpers.py`'s `initial_model`, and add
the new problem string to that mode's list in `main.py`'s module docstring
(and the README's catalogue, if it's meant to be permanent). A minimal HD
example:

```python
# src/models/HD/HD_init_cond.py
def IC_HD2D_my_blob(grid, fluid, par):
    print("my custom HD blob problem")

    grid.CartesianGrid(0.0, 1.0, 0.0, 1.0)
    par.timenow, par.timefin = 0.0, 0.3

    fluid.dens[:, :] = 1.0
    fluid.pres[:, :] = 1.0
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0

    x0, y0, r0 = 0.5, 0.5, 0.1
    rad = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    fluid.dens[:, :] = np.where(rad < r0, 10.0, 1.0)   # dense blob

    par.BC[:] = 'free'          # all four faces zero-gradient

    eos = EOSdata(5.0/3.0)
    return grid, fluid, par, eos
```

```python
# src/misc/helpers.py, inside initial_model()'s hd_dispatch dict
"my-blob": IC_HD2D_my_blob,
```

Note the pattern every `IC_*` function follows: build the grid geometry
first (`grid.<Geometry>Grid(...)`), then set `par.timenow`/`par.timefin`,
then fill primitive fields **on the full ghost-inclusive array** (`[:, :]`
— the solver's own boundary-condition call fills the ghosts consistently
with `par.BC` before the first flux evaluation, so you never need to set
ghost values yourself), then set `par.BC`, then return.

### 12.3. Solve a custom elliptic problem

`solve_poisson` doesn't care why you need an elliptic solve — here's the
manufactured-solution check from the module's own docstring, verifying
`phi = sin(pi x) sin(pi y)` against its own Laplacian on a unit square:

```python
import numpy as np
from src.grid.grid_setup import Grid
from src.common.poisson_solver import solve_poisson

g = Grid(64, 64, 2)
g.CartesianGrid(0.0, 1.0, 0.0, 1.0)

x = g.cx1[g.Ngc:-g.Ngc, g.Ngc:-g.Ngc]
y = g.cx2[g.Ngc:-g.Ngc, g.Ngc:-g.Ngc]
rhs = -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

BC = ['dirichlet', 'dirichlet', 'dirichlet', 'dirichlet']
phi, info = solve_poisson(g, rhs, BC, verbose=True)
print(info)   # {'niter': ..., 'residual': ..., 'converged': True}
```

Swap `BC` for `['peri', 'free', 'peri', 'free']` (periodic in x1, Neumann in
x2) or any other per-face mix, and `rhs` for whatever source you actually
have (`div(B)` for a divergence-cleaning correction, a charge density, ...).
`phi` comes back ghost-padded and ready for `grid_misc.cell_gradient`/
`face_gradient` immediately.

### 12.4. Drive the time loop by hand

`run_simulation` is a convenience wrapper, not a requirement — everything it
does is three lines:

```python
from src.parameters import Parameters
from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.misc.helpers import initial_model
from src.models.HD.HD_step import HD2D

par = Parameters(mode="HD", problem="sod1Dcart", Nx1=200, Nx2=1)
grid = Grid(par.Nx1, par.Nx2, par.Ngc)
state = SimState(grid, par)
grid, state, par, eos = initial_model(grid, state, par)

solver = HD2D(grid, state, eos, par)
while par.timenow < par.timefin:
    state = solver.step_RK()
    # inspect/plot/checkpoint state here on your own terms
print(f"reached t = {par.timenow}")
```

This is the shape every testbed case and every gravity/IO recipe below
builds on.

### 12.5. Self-gravity in a manual time-stepping loop

`selfgravity_poisson` (and the other two `gravity.py` functions) are meant
to be called once per step, before the solver's residual evaluation reads
`state.F1`/`F2`:

Install a hook and the solver re-solves the potential every RK stage — you
do **not** call it yourself in the loop, and you must not compute it once
before the loop, because the density it depends on changes every stage:

```python
from src.gravity import selfgravity_poisson_hook

# inside an IC function, after the density is set:
state.body_force = selfgravity_poisson_hook(G=1.0, BC=['peri'] * 4)
```

Then the ordinary time loop of [§12.4](#124-drive-the-time-loop-by-hand)
needs no change at all — `step_RK()` invokes the hook internally, once per
stage, on that stage's state.

Boundary conditions decide correctness here, so pick deliberately:

```python
# periodic box (Jeans problem): mean subtraction IS the Jeans swindle
BC = ['peri'] * 4

# isolated sphere, spherical grid: exact exterior potential on the outer face
BC     = ['free', 'free', 'dirichlet', 'free']   # 'free' at r=0 is exact
BC_val = {2: -G * M_total / r_out}

# isolated body, cylindrical (R,z): monopole potential on each outer face
BC     = ['free', 'dirichlet', 'dirichlet', 'dirichlet']
```

Using `['free']*4` for an *isolated* body is the trap: with no Dirichlet
face the solver must enforce solvability by subtracting the mean source,
which quietly adds a uniform negative background density and changes the
enclosed mass. It runs, and it is wrong. See
[§8](#8-gravity-and-other-body-forces).

Three worked examples ship as test problems — `collapse1D` (isolated,
Dirichlet), `jeans2D` (periodic), `collapse2D` (isolated, monopole faces).

### 12.6. Save, restart, and post-process a run

```python
from src.misc.io_utils import save_data, load_data, restart_simulation

# --- during/after a run ---
save_data("runs/khi_0100", grid, state, par, eos)   # -> runs/khi_0100.npz

# --- offline analysis, no live objects needed ---
d = load_data("runs/khi_0100")
rho = d['dens']                 # ghost-inclusive ndarray, exactly as stored
print("t =", d['timenow'], "mode =", d['mode'])

# --- resume the simulation ---
grid, state, par, eos = restart_simulation("runs/khi_0100")
from main import SOLVER_DISPATCH
solver = SOLVER_DISPATCH[par.mode](grid, state, eos, par)
while par.timenow < par.timefin:
    state = solver.step_RK()
```

`restart_simulation` rebuilds `Grid` via `reconstruct_grid` and `Parameters`
via its normal constructor (so derived fields like `Ngc` stay consistent
with `rec_type`), then allocates a fresh `SimState` and overwrites every
field the archive actually has — including fields an `__init__` might not
pre-allocate for every mode (e.g. `adv`'s constant `vel1`/`vel2`).

### 12.7. Run and extend the testbed

```bash
python run_testbed.py                       # all four suites
python run_testbed.py --suite convergence    # one suite
python run_testbed.py -q                     # only print failures
```

To sanity-check a new problem you just added in §12.2 without writing a
bespoke test, add it to the `CATALOGUE` dict in `tests/test_sanity.py`:

```python
# tests/test_sanity.py
CATALOGUE["HD"].append("my-blob")
```

and it's automatically built, stepped, and checked for NaNs/positivity by
the sanity suite. If it's also meant to stress-test every solver/
reconstruction/RK-order combination (robustness), register it explicitly in
`test_robustness.py` instead (see [§11](#11-the-testbed)) — that suite
opts problems in one call per problem, not from the sanity catalogue.

## 12.8. The astrophysics problems

Six problems in the catalogue are set up as end-to-end astrophysical
applications rather than numerical benchmarks. They are the place to look
for how the pieces (curvilinear geometry, inlet boundaries, CT, self-gravity,
rotating frames) combine in practice.

| Problem | Mode | Geometry | What it demonstrates |
|---|---|---|---|
| `gap-opening` | HD | polar (R,φ) | A planet carving an annular gap in a protoplanetary disk, in the frame **co-rotating** with the planet: static planet potential plus centrifugal and Coriolis terms, a smooth planet-mass ramp, and radial boundaries pinned to the analytic power-law equilibrium |
| `jet2Dcyl` (HD) | HD | cylindrical (R,z) | A Mach-6 light jet injected through a fixed-state nozzle boundary, developing a bow shock, Mach disk and cocoon |
| `jet2Dcyl` (MHD) | MHD | cylindrical (R,z) | The magnetized counterpart, on a uniform axial field. **Run with `divb_tr='CT'`** |
| `disk2D` | MHD | cylindrical (R,z) | A constant-angular-momentum accretion torus in an exactly hydrostatic atmosphere, seeded with a divergence-free poloidal field for the MRI |
| `collapse1D` | HD | spherical (r,θ) | Pressureless collapse of a uniform sphere, checked against the exact free-fall cycloid |
| `jeans2D` | HD | Cartesian | Gravitational instability at the exact linear growth rate |
| `collapse2D` | HD | cylindrical (R,z) | A rotating cloud collapsing and **flattening into a disk** — angular momentum conserved to ~1e-4 |

Two practical notes, both learned the hard way and easy to hit again:

- **The MHD jet and torus want `divb_tr='CT'`.** A uniform axial field is
  exactly divergence-free on the staggered mesh, and CT holds it at round-off
  (measured `max|divB|/(B0/dR) ~ 4e-15`). Under GLM cleaning the same setup
  reaches 4–12 and the divergence error drives the gas pressure into its
  floor near the nozzle. The jet IC prints a warning if handed anything else.
- **An ambient medium in a point-mass potential must be hydrostatic.** A
  constant-density floor is not a solution of anything: it free-falls from
  the first step and pins the pressure floor across the grid. The torus uses
  `rho ~ r^(-n)` with the matching `p(r)`, which is exactly static for any
  `n`.

## 13. Extending the framework

- **New reconstruction scheme**: add it to `src/common/high_order_rec.py`'s
  `VarReconstruct` dispatcher, and to `Parameters.__init__`'s `Ngc` rule if
  it needs a nonstandard ghost-cell count (2 for PCM/PLM-width stencils, 3
  otherwise, currently).
- **New Riemann solver**: add it alongside the existing ones in that mode's
  `*_riemann_*.py`, and register the new `solver_type` string in the
  mode's `*_step.py` flux driver (a plain `if solver_type == ...` dispatch,
  no registry indirection to fight).
- **New physics mode**: the four-file rhythm in [§7](#7-physics-modes-file-by-file)
  is the template — a new `SimState` branch, a `Parameters` mode entry, a
  `*_init_cond.py` with at least a `user_defined` problem, and a
  `SOLVER_DISPATCH` entry in `main.py`. `gravity.py` and `poisson_solver.py`
  are already mode-agnostic (they only touch `state.dens`/`F1`/`F2` and
  `grid`), so a new HD-family mode gets self-gravity for free by calling
  `selfgravity_poisson` the same way HD does.
- **New elliptic use of `solve_poisson`** (e.g. magnetic divergence
  cleaning as a post-step projection): follow the pattern in
  `poisson_solver.py`'s own "Designed to be called from other modules"
  docstring section — solve for a scalar potential with the right BCs, then
  differentiate it with `face_gradient`/`cell_gradient` as your use case
  needs a face- or cell-centred correction.

## 14. Conventions cheat-sheet

- **Ghost vs. interior**: primitive fields and coordinate arrays are
  ghost-inclusive (`grid.grid_shape`); conservative fields and `fS1/fS2/cVol`
  are interior-only `(Nx1, Nx2)`, **no ghosts**. See [§3](#3-the-ghost-cell--interior-cell-convention).
- **Body-force sign**: `state.F1`/`F2` must hold the acceleration itself
  (`F = a`), never its negative — every solver's residual already applies
  the correct sign. See [§8](#8-gravity-and-other-body-forces).
- **BC vocabulary differs by system**: hyperbolic solvers use `'free'` /
  `'wall'` / `'peri'` / `'axis'`; the Poisson solver uses `'free'` /
  `'peri'` / `'dirichlet'`. See [§5](#5-boundary-conditions).
- **Face index convention**: `0`=x1 inner, `1`=x2 inner, `2`=x1 outer,
  `3`=x2 outer — shared by `par.BC`, `par.BC_fixed`, and `solve_poisson`'s
  `BC`/`BC_value`.
- **1D is 2D with `Nx2==1` (or `Nx1==1`)**, not a separate code path — every
  operator in `grid_misc.py` and every solver checks `grid.Nx1 > 1`/
  `grid.Nx2 > 1` rather than assuming both are active. Writing a 1D-only
  helper means guarding it the same way.
- **`rec_type` fixes `Ngc`**: 2 for `PCM`/`PLM`, 3 for `PPMorig`/`PPM`/
  `WENO`/`MP5` (1 for `diff`, which doesn't use `rec_type` at all). Don't
  hard-code `Ngc=2` anywhere that might see a higher-order run.
- **Don't pair `MP5` with `RK1`**: forward Euler's stability margin is too
  small for MP5's low dissipation on a strong shock (loses positivity;
  fine on smooth data). Verified and documented in
  `tests/test_robustness.py`; use `RK2`/`RK3` with `MP5`.

---

For the mode/solver/geometry option tables, the quickstart, the full
test-problem catalogue, and the reference list, see [`README.md`](README.md).
