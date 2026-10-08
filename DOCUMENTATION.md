# Piastra — Developer Documentation

This is the deep-dive companion to [`README.md`](README.md). The README
introduces the project and gets you running a simulation in five minutes;
this document is for when you want to know exactly what an object contains,
how a piece of the solver actually works, or how to bend the framework toward
a problem it wasn't shipped with. It assumes you've read the README once.

## Table of contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [Core objects](#2-core-objects)
3. [The ghost-cell / interior-cell convention](#3-the-ghost-cell--interior-cell-convention)
4. [Numerical building blocks](#4-numerical-building-blocks)
5. [Boundary conditions](#5-boundary-conditions)
6. [The Poisson solver](#6-the-poisson-solver)
7. [Physics modes, file by file](#7-physics-modes-file-by-file)
8. [Per-step physics: `par.before_step`](#8-per-step-physics-parbefore_step)
9. [Gravity and other body forces](#9-gravity-and-other-body-forces)
10. [Saving, loading, and restarting](#10-saving-loading-and-restarting)
11. [Visualization](#11-visualization)
12. [The testbed](#12-the-testbed)
13. [Use cases / recipes](#13-use-cases--recipes)
14. [Extending the framework](#14-extending-the-framework)
15. [Known limitations](#15-known-limitations)
16. [Conventions cheat-sheet](#16-conventions-cheat-sheet)

---

## 1. Architecture at a glance

Every run — whether launched from `main.py`, a notebook, or a hand-rolled
script — follows the same five-stage pipeline:

```
Parameters  ──►  Grid  ──►  SimState  ──►  initial_model()  ──►  Solver.step_RK()
 (config)      (mesh +      (field       (looks the problem      (one explicit or
                metric)      storage)     up in PROBLEMS and      RKL2 step, called
                                          calls its IC function,  in a loop by
                                          which also builds the   run_simulation)
                                          geometry and EOS)
```

- **`Parameters`** (`src/parameters.py`) is a plain config object: mode,
  problem name, resolution, reconstruction/solver/RK choices, CFL, boundary
  condition arrays, the optional per-step hook `before_step`. It checks
  `mode` (and `divb_tr` for MHD) and fills in per-mode defaults (e.g. `Ngc`
  from `rec_type`, the default `solver_type`). Other options (`rec_type`,
  `RK_order`, `solver_type`) are checked where they are used, so a typo
  raises inside the solver rather than here.
- **`Grid`** (`src/grid/grid_setup.py`) allocates the metric arrays (face
  areas, cell volumes, coordinates) for a given resolution, then one of its
  four geometry methods (`CartesianGrid`, `CylindricalGrid`, `PolarGrid`,
  `SphericalPolarGrid`) fills them in. Nothing downstream branches on
  geometry again — every solver and helper reads `grid.fS1/fS2/cVol/hx2`
  and gets the right answer in all four coordinate systems.
- **`SimState`** (`src/sim_state.py`) allocates the per-mode array set
  (primitive, conservative, magnetic, ...). It is pure storage — no
  behaviour.
- **`initial_model`** (`src/misc/helpers.py`) looks `(par.mode, par.problem)`
  up in the catalogue `PROBLEMS` and calls the matching `IC_*` function,
  which is the single place that decides the domain, geometry, primitive
  fields, boundary conditions, `par.timefin`, and (if the problem needs one)
  `par.before_step`. This is also where the `EOSdata` object is built for the
  modes that need one.
- The **solver class** (one per mode, e.g. `HD2D`, `MHD2D_CT`, `rHD2D`,
  `SWE2D`, `Diff2D`) owns `step_RK()`: compute `dt` from the CFL condition,
  call `par.before_step` (HD-family modes), then reconstruct → Riemann solve
  → flux-difference update → boundary refill, wrapped in an SSP Runge-Kutta
  stage loop (or RKL2 sub-stepping for `diff`). `main.py`'s `SOLVER_DISPATCH`
  dict maps a mode string to the right constructor; reuse it instead of
  hard-coding the class name if you're writing something mode-generic (the
  testbed does exactly this).
- **`run_simulation`** (`src/misc/helpers.py`) is the `while par.timenow <
  par.timefin: state = solver.step_RK()` loop with periodic plotting and,
  optionally, periodic output at exact times ([§10](#10-saving-loading-and-restarting)).
  You don't have to use it — see [§13.4](#134-drive-the-time-loop-by-hand).

## 2. Core objects

### `Grid` (`src/grid/grid_setup.py`)

| Attribute | Shape | Meaning |
|---|---|---|
| `Nx1, Nx2` | scalar | real (non-ghost) cell counts |
| `Ngc` | scalar | ghost-cell count per side (2 for PCM/PLM, 3 otherwise, 1 for `diff`) |
| `Nx1r, Nx2r` | scalar | `Nx + Ngc`: one past the last real cell, i.e. the exclusive end of `field[Ngc:Nx1r]` |
| `grid_shape` | tuple | `(Nx1+2*Ngc, Nx2+2*Ngc)` — the shape of every ghost-inclusive array |
| `fx1, fx2` | `(Nx1+2Ngc+1, ·)`, `(·, Nx2+2Ngc+1)` | face coordinates |
| `cx1, cx2` | `grid_shape` | cell-centre coordinates (ghost-inclusive); for `sph`, `cx1` is the volume-weighted radius `2(r₊³−r₋³)/(3(r₊²−r₋²))`, which balances the `2p/r` geometric source exactly |
| `dx1, dx2` | `grid_shape` | local cell widths (coordinate widths; `dx2` is an angle for `pol`/`sph`) |
| `dx1uc, dx2uc` | scalar | uniform cell widths (Cartesian grids; used by Lax–Wendroff advection) |
| `ax1, ax2` | `grid_shape` | volumetric centroids |
| `fS1, fS2` | `(Nx1+1, Nx2)`, `(Nx1, Nx2+1)` | face areas, **interior-only, no ghosts** |
| `fS3` | `(Nx1, Nx2)` | face area ⟂ the third (out-of-plane) direction, used by CT |
| `cVol` | `(Nx1, Nx2)` | cell volumes, **interior-only, no ghosts** |
| `edg1, edg2, edg3` | various | edge lengths, used by CT's Stokes-theorem curl |
| `hx2` | `grid_shape` | Lamé factor of the x2 direction: `1` for cart/cyl, `cx1` (R or r) for pol/sph; the physical width in x2 is `dx2*hx2` |
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
using the framework as a library (see [§13](#13-use-cases--recipes)).

`grid_setup.py` also exposes `reconstruct_grid(Nx1, Nx2, Ngc, geom, x1ini,
x1fin, x2ini, x2fin)` — a module-level function, not a `Grid` method — that
rebuilds a grid deterministically from just these numbers.
`io_utils.restart_simulation` uses it so a saved run doesn't need to
serialize any metric arrays.

### `Parameters` (`src/parameters.py`)

Holds everything that configures *how* a run advances, as opposed to *what
physical state* it's in (that's `SimState`'s job). Required: `mode`,
`problem`, `Nx1`, `Nx2`. See the README's
[Configuration reference](README.md#configuration-reference) for the full
option table. Things worth calling out that the README only lists:

- `par.BC`, `par.BCm` are 4-entry arrays `[x1_inner, x2_inner, x1_outer,
  x2_outer]` of boundary types (see [§5](#5-boundary-conditions)). They are
  stored with `dtype=object`, so a longer (mistyped) name is not silently
  truncated to 4 characters but rejected by the boundary routines.
  `par.BCm` (magnetic field) exists only for MHD/rMHD; `None` otherwise.
- `par.BC_fixed` is a `{0,1,2,3: [(start, end, {field: value}), ...]}` dict
  for pinning a *sub-range* of one boundary to fixed values (e.g. an inflow
  nozzle on part of a wall) — see `boundaries.apply_bc_fixed` and
  [§5](#5-boundary-conditions) for which modes support it. It's empty by
  default; most `IC_*` functions never touch it.
- `par.before_step` is the per-step physics hook, `None` by default — see
  [§8](#8-per-step-physics-parbefore_step).

### `SimState` (`src/sim_state.py`)

Allocates a different attribute set per `par.mode` — see the module
docstring for the full per-mode attribute list, or just read
`src/sim_state.py` directly, it's short and literal (no metaprogramming).
The one thing you must internalize before touching any array on this object
is [§3](#3-the-ghost-cell--interior-cell-convention) below.

### `EOSdata` (`src/common/eos_setup.py`)

A one-parameter ideal-gas EOS: `EOSdata(GAMMA)`. Exposes
`sound_speed_nr(dens, pres)` / `sound_speed_sr(dens, pres)` (non-relativistic
and special-relativistic sound speed), `enthalpy_sr(dens, pres)`, and
`eint(dens, pres)` / `pres(dens, eint)` (internal-energy ↔ pressure, kept as
methods so the modes' `*_phys.py` files never hard-code `GAMMA - 1`). `None`
for `adv`, `diff`, `SWE` — those modes have no equation of state.

## 3. The ghost-cell / interior-cell convention

This is the single most important thing to know before writing code against
this framework, and the source of most non-obvious bugs. Two array shapes
coexist:

- **Ghost-inclusive**, shape `grid.grid_shape = (Nx1+2*Ngc, Nx2+2*Ngc)`:
  every *primitive* field — `dens`, `pres`, `vel1/2/3`, `bfi1/2/3`, `bglm`,
  `T`, `h`, `b`, `f_c` — plus geometric coordinate arrays (`cx1`, `cx2`,
  `dx1`, `dx2`, `ax1`, `ax2`, `hx2`). Indexing the real domain out of one of
  these needs the ghost offset: `field[Ngc:-Ngc, Ngc:-Ngc]`.
- **Interior-only**, shape `(Nx1, Nx2)`, **no ghost cells at all**: every
  *conservative* field — `mass`, `mom1/2/3`, `etot`, `bcon1/2/3`, `glmcon` —
  the body force `F1`, `F2`, the diffusion source `ST`, `divB`, and the
  geometric **face/volume** arrays `fS1`, `fS2`, `fS3`, `cVol`. These must be
  indexed directly, `field[:, :]` or `field[i, j]` — slicing them with a ghost
  offset silently produces garbage (wrong cells) or an `IndexError` on a
  non-square grid.
- The CT face fields `fb1`, `fb2` have shapes `(Nx1+1, Nx2)` and
  `(Nx1, Nx2+1)`, also without ghosts. `fb2` lives on x2-faces, i.e. at
  x1 = `cx1` (cell centre), x2 = face; `fb1` at x1 = face, x2 = `cx2`.

Every real bug caught by this project's testbed traced back to applying the
wrong one of these slicing patterns to `grid.cVol` (it looks exactly like a
ghost-inclusive array from the shape alone if your grid happens to be square
— which is why the testbed uses non-square resolutions). When you write new
code that touches `cVol`, `fS1`, `fS2`, `F1`, `F2`, slice them plainly; when
you touch `dens`, `cx1`, or `pres`, use the `Ngc:-Ngc` offset.

## 4. Numerical building blocks

`src/grid/grid_misc.py` holds the geometry-aware finite-volume operators
every solver (hyperbolic, parabolic, elliptic) is built from. None of them
branch on `grid.geom` — the geometry lives entirely in the grid's metric
arrays (`fS1`, `fS2`, `cVol`, `hx2`), so these functions are correct in all
four coordinate systems by construction.

| Function | Signature | What it does |
|---|---|---|
| `interp_face_to_cell` | `(grid, fV1, fV2) -> V1, V2` | distance-weighted interpolation of a staggered vector field (`fV1` of shape `(Nx1+1, Nx2)`, `fV2` of shape `(Nx1, Nx2+1)`) to cell centres |
| `div_face_vector` | `(grid, fV1, fV2) -> divV` | Gauss's-theorem divergence of a face-centred vector field (used by CT, diffusion and the Poisson solver) |
| `div_cell_vector` | `(grid, V1, V2) -> divV` | divergence of a cell-centred vector field via face-averaging |
| `cell_gradient` | `(grid, f) -> g1, g2` | second-order gradient of a cell-centred scalar, **on cell centres**, metric-corrected via `hx2` |
| `face_gradient` | `(grid, f) -> g1, g2` | same, but evaluated **on faces** (shapes `(Nx1+1,Nx2)` / `(Nx1,Nx2+1)`) — what the Poisson operator and the diffusion solver use |
| `edge_to_face_curl` | `(grid, edg_var) -> fV1, fV2` | discrete Stokes-theorem curl of an out-of-plane edge scalar, producing a solenoidal-by-construction staggered field — used to seed CT's initial `fb1/fb2` from a vector potential |
| `Ln_norm` | `(grid, n, var_num, var_ref) -> float` | `(sum(cVol * abs(num - ref)^n))^(1/n)` over the real cells |
| `integral_over_grid` | `(grid, var) -> float` | volume integral `sum(cVol * var)` of a ghost-inclusive field over the real cells |

`cell_gradient` vs. `face_gradient`: use `cell_gradient` when you need a
gradient sampled at the same locations as the input field (e.g. converting
a potential to a cell-centred body force — see `gravity.selfgravity_poisson`);
use `face_gradient` when you need it at the faces a finite-volume flux
divergence expects (e.g. `poisson_operator`'s `-div(grad(phi))`, or the
diffusive flux).

`src/common/high_order_rec.py` implements the reconstruction stencils behind
one dispatcher,

```python
var_L, var_R = VarReconstruct(var, grid, rec_type, dim, limiter_type=None)
```

called per primitive variable and per direction by every hyperbolic solver's
flux routine. `rec_type` is one of `PCM`, `PLM`, `PPMorig`, `PPM`, `WENO`,
`MP5`. `PLM` uses the van Leer limiter by default; `limiter_type` can select
`'MM'` (minmod), `'MC'`, `'KOR'` (Koren), `'PCM'` or `'NO'` (unlimited) —
it is not exposed through `Parameters`, so change it in code if you need it.
The relativistic solvers fall back to PLM (van Leer) at faces where a
higher-order reconstruction produces an unphysical state or meets a strong
pressure jump (`_swap_troubled`).

## 5. Boundary conditions

Two independent BC systems exist, with **different vocabularies** — don't
mix them up:

- **Hyperbolic solvers** (`par.BC`, `par.BCm`): each of the 4 faces is one of
  - `'free'` — outflow: the ghost cells are the mirror image of the interior,
    i.e. zero gradient at the boundary face;
  - `'wall'` — reflective: mirror image with the normal vector component
    flipped;
  - `'peri'` — periodic (both opposite faces must be `'peri'`);
  - `'axis'` — symmetry axis: mirror image with the normal and azimuthal
    components flipped. Valid only at x1_inner (R = 0) and on the x2 faces
    of a spherical grid (θ = 0, π). The flipped "azimuthal" component is
    `V3`, i.e. the cylindrical `(R, z, φ)` / spherical `(r, θ, φ)` ordering;
    on a polar `(R, φ)` grid the azimuthal velocity is `V2`, so `'axis'` at
    R = 0 there is only correct for flows with `v_φ = 0`.

  Any other string raises `ValueError`. Filled by
  `boundaries.apply_bc_scalar` / `apply_bc_vector`, called once per RK stage
  by each mode's `*_phys.py`. Fills **all** `Ngc` ghost layers, since
  high-order reconstruction stencils read more than one.
- **The Poisson solver** (`BC` argument to `solve_poisson`): each face is
  one of `'peri'`, `'free'`, or `'dirichlet'` (a genuine fixed value —
  there is no `'wall'`/`'axis'` here, since the Laplacian doesn't
  distinguish a reflecting wall from a zero-gradient face). Filled by
  `boundaries.apply_bc_scalar_Ngc1`, which only touches the **one** ghost
  layer bordering the real domain — the natural width of a 3-point
  second-order stencil — so a deeper-ghosted hydro/MHD grid (Ngc=2 or 3)
  can be handed straight to the Poisson solver.

Face indexing convention, shared by both systems: `BC[0]` = x1 inner,
`BC[1]` = x2 inner, `BC[2]` = x1 outer, `BC[3]` = x2 outer.

`boundaries.apply_bc_fixed(state_fields, Ngc, N1, N2, face, patches)` is a
third, narrower tool: it pins ghost cells on a *sub-range* of one face to
prescribed values, after the regular fill — for a partial inflow boundary
such as a jet nozzle. It is driven by `par.BC_fixed`. Because the prescribed
state lives in the ghost cells, the Riemann solver at the boundary face
produces the inflow flux without any change to the flux routine. A field
name that the mode doesn't have raises `ValueError`.

`BC_fixed` support by mode:

| Mode | `BC_fixed` |
|---|---|
| `HD`, `rHD`, `adv`, `diff` | supported |
| `MHD` with `divb_tr='GLM'` or `'8wave'` | supported (cell-centred B, so the ghost B enters through the Riemann solver like ρ and v) |
| `MHD` with `divb_tr='CT'`, `rMHD` | **rejected** (`ValueError` in the solver constructor): with CT the face fields are advanced by the corner EMF, and at a boundary corner half of it comes from the mirrored ghost-face fluxes rather than from the inflow state — a correct CT inflow would need the boundary EMF prescribed |
| `SWE` | not implemented (ignored) |

## 6. The Poisson solver

`src/common/poisson_solver.py` solves `div(grad(phi)) = rhs` on any grid
(1D or 2D, any of the four geometries) with matrix-free,
diagonally-preconditioned Conjugate Gradient. It's deliberately generic —
not tied to gravity or divergence cleaning — so any module can call it for
an elliptic sub-problem.

```python
phi, info = solve_poisson(grid, rhs, BC, BC_value=None, phi0=None,
                          tol=1e-10, maxiter=None, verbose=False)
```

- `rhs` — interior-only `(Nx1, Nx2)` source term.
- `BC` — 4-entry list, `'peri'` / `'free'` / `'dirichlet'` per face (see
  [§5](#5-boundary-conditions)); opposite faces must agree on `'peri'`.
- `BC_value` — `{face_index: value}` dict for `'dirichlet'` faces; a value
  may be a scalar or an array along the face (defaults to 0.0 if omitted).
- `phi0` — optional initial guess (full `grid_shape` array), e.g. the
  previous step's potential.
- `tol` — relative residual `||r|| / ||rhs||` (volume-weighted L2 norm).
- Returns `phi` as a full `grid.grid_shape` array (boundary ghost layer
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
  closed-form face conductances. On a uniform Cartesian grid the diagonal is
  constant and the preconditioner does nothing; it helps only through the
  metric factors of curvilinear grids. CG needs O(N) iterations on an N×N
  grid, so the solve is the expensive part of a self-gravity run.
- **Problems without any `'dirichlet'` face** (pure periodic / pure
  zero-gradient) fix `phi` only up to an additive constant and are solvable
  only if the volume integral of `rhs` vanishes. `solve_poisson` therefore
  subtracts the volume-weighted mean of `rhs` before solving, and fixes the
  constant by returning `phi` with zero volume-weighted mean. For gravity
  this means that only `rho - <rho>` gravitates — correct for a periodic box
  (the Jeans swindle), wrong for an isolated object (see [§9](#9-gravity-and-other-body-forces)).
- Inhomogeneous Dirichlet BCs use the standard CG "lifting" trick:
  `poisson_operator` is called with the real `BC_value` only when evaluating
  the residual `b - A(phi_int)`; every other application of `A` inside the CG
  loop uses the homogeneous form (`BC_value=None`), which is what keeps `A`
  linear and self-adjoint on the vectors CG iterates over.

## 7. Physics modes, file by file

Every mode's package follows the same rhythm:

| File | Contents |
|---|---|
| `*_step.py` | The solver class (`step_RK()`, the CFL/`dt` calculation, the RK-stage loop, the flux/residual driver called once per stage) |
| `*_phys.py` | Primitive ↔ conservative conversion, the boundary-condition call, and the Riemann-solver dispatcher (`Riemann_HD`, `Riemann_MHD`, ...: rotates the states for the x2 direction, picks the solver by `solver_type`, rotates the fluxes back) |
| `*_init_cond.py` | Every `IC_*` test-problem function for that mode, plus `user_defined` |
| `*_riemann_*.py` | The Riemann solvers themselves (approximate and, for HD/SWE, exact); not needed for `adv` and `diff` |

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
`SOLVER_DISPATCH` in `main.py` absorbs this asymmetry behind a uniform
`lambda grid, state, eos, par: ...` per entry.

Mode-specific notes:

- **SWE.** Source terms (bed slope `-g grad(b)` and Coriolis `f`) are applied
  by Strang splitting around the hyperbolic RK step. Each source half-step is
  an explicit Euler step: exact for the bed slope, but only first order for
  Coriolis, which multiplies the kinetic energy of inertial motions by
  `1 + (f dt)^2/2` per step. This is harmless when `f^2 * dt * t_fin << 1`;
  for long rotating runs an exact rotation of `(v1, v2)` by the angle
  `f dt/2` is the remedy. The bed source is a cell-centred gradient, applied
  separately from the face fluxes, so the scheme is **not well-balanced**:
  the `lake2D` problem (lake at rest over a seamount) shows the resulting
  spurious currents.
- **Diffusion.** `diff.kappa` may be a scalar or a ghost-inclusive array (face
  values are arithmetic means); `diff.ST` is an interior-only source term
  added to the operator. RKL2 takes super-steps of `(s² + s − 2)/4` explicit
  steps at the cost of `s` operator evaluations.
- **Relativistic modes.** The conservative-to-primitive inversion is
  Newton–Raphson; cells that do not converge are reported
  (`[rMHD] Newton: N cell(s) unconverged ...`) but not corrected. In strongly
  magnetized, cold regions (`σ = b²/ρh ≫ 1`, gas pressure a tiny fraction of
  the total energy) such messages are expected — the Komissarov blast test
  (`rMHD/blast2D`) is the standard example.

The full test-problem catalogue (which `problem` string maps to which
`IC_*` function, for every mode) is the `PROBLEMS` dict in
`src/misc/helpers.py`, summarized in the README.

## 8. Per-step physics: `par.before_step`

`par.before_step` is an optional function

```python
def before_step(grid, state, par, dt):
    ...
```

that the HD, rHD, MHD (all three schemes) and rMHD solvers call **once per
time step**, inside `step_RK`, after `dt` has been computed and before the
hyperbolic update. It is the place for any physics that is not a flux:
recomputing gravity, optically thin cooling, relaxation (damping) zones,
mass or energy injection. The IC function of a problem that needs it
installs it (`gap-opening`, `jeans1D`, `collapse_sph`, `merger2D`), so a
catalogue problem works without any extra setup.

Rules:

- **Change primitive variables (and `F1`, `F2`) only.** Every step starts
  by recomputing the conservative variables from the primitives, so a
  change to `mass`, `mom1`, ... inside the hook is lost.
- **`par.timenow` is the time at the start of the step**, `dt` the step that
  is about to be taken. A time-dependent source (an orbiting planet) is
  evaluated at `timenow`; `dt` is there for sources integrated over the step,
  e.g. exact relaxation `q ← q₀ + (q − q₀) exp(−dt/τ)`.
- **Time accuracy.** This is first-order operator splitting: whatever the
  hook sets (e.g. `F1`, `F2`) is held fixed through all RK stages of the
  step, so the coupling is first order in time even with RK3.
- **Several sources** go into one function. The gravity routines *assign*
  `F1`, `F2` (`=`), so to combine two of them, keep a copy of the first
  result and add it after calling the second.
- **Not stored, but restored.** A function can't go into an `.npz` file;
  `restart_simulation` rebuilds the hook by re-running the problem's IC
  function on scratch objects (see [§10](#10-saving-loading-and-restarting)).
  A hook you installed by hand, outside the IC function, has to be installed
  again after a restart.
- **Modes without the hook:** `adv`, `SWE` and `diff` do not call it.

Example — a planet's gravity plus an exponential damping zone near the inner
boundary of a disk:

```python
def disk_physics(grid, state, par, dt):
    planet_gravity_polar(grid, state, par, M_star=1.0, M_planet=1e-3,
                         r_planet=1.0, phi0=np.pi, soft=0.03)
    R = grid.cx1
    rate = np.where(R < R_damp, ((R - R_damp) / (R_in - R_damp))**2 / T_damp, 0.0)
    k = np.exp(-rate * dt)                       # exact relaxation over dt
    state.dens[:, :] = dens0 + (state.dens - dens0) * k
    state.vel1[:, :] =          state.vel1          * k

par.before_step = disk_physics
```

## 9. Gravity and other body forces

`src/gravity.py` provides three body-force routines. Each fills
`state.F1`, `state.F2` (interior-only `(Nx1, Nx2)`) and is meant to be called
from `par.before_step` ([§8](#8-per-step-physics-parbefore_step)); the
solver then reads `F1`, `F2` as a source term in every RK stage.

**Sign convention.** `F1`/`F2` hold the **acceleration** itself,
`F = a = -grad(Phi)`, never its negative. The HD and MHD solvers add
`rho F` to the momentum and `rho F·v` to the energy equation; the
relativistic solvers weight the force by the inertia `rho h W²` instead
of `rho`. An accidental sign flip turns attraction into repulsion silently
(no crash, no NaN, just a slowly unbinding "self-gravitating" cloud).

| Function | Use case | Method |
|---|---|---|
| `planet_gravity_polar(grid, state, par, M_star, M_planet, r_planet, phi0=0.0, indirect=True, soft=0.05)` | star + planet on a circular orbit, polar grid | softened point masses, planet azimuth `phi0 + Omega_p * timenow`, optional indirect term of the star-centred frame; no elliptic solve |
| `selfgravity_monopole_spherical(grid, state, par)` | nearly spherical self-gravitating object, spherical-polar grid | angle-averaged (l = 0) enclosed mass; no elliptic solve |
| `selfgravity_poisson(grid, state, par, G=1.0, BC=None, BC_value=None, tol=1e-10, maxiter=None)` | general self-gravity, any density field, any geometry | `div(grad(Phi)) = 4 pi G rho` via `solve_poisson`, then `F = -cell_gradient(Phi)` |

**Choosing the Poisson boundary conditions** for `selfgravity_poisson` is
the physics decision, not a detail:

- **Periodic box** (`BC=['peri']*4`): the mean density is removed
  automatically, so only `rho - <rho>` gravitates — the Jeans swindle.
  Correct for `jeans1D` and `merger2D`.
- **Isolated object**: use `'dirichlet'` faces with the exterior potential.
  For a spherical mass distribution on a spherical grid the monopole
  `-G M / r_out` at the outer radius is exact (`collapse_sph` does this);
  on other grids a monopole `-G M / |x_b - x_cm|` per boundary face is
  accurate to the quadrupole order. `BC_value` accepts arrays along a face.
- **`'free'` faces are wrong for an isolated object**: zero normal gradient
  means zero gravitational flux through the boundary, and with no
  Dirichlet face the mean density is removed as well. The default
  `BC=['free']*4` is therefore only a placeholder — always pass `BC`
  explicitly.

**What a 2D Poisson solve means physically** depends on the geometry:
on `cyl` `(R, z)` and `sph` `(r, θ)` grids it is axisymmetric 3D gravity; on
`cart` `(x, y)` and `pol` `(R, φ)` grids it is the gravity of infinitely long
cylinders (force ∝ 1/d), *not* of a thin disk.

**Time step.** The CFL condition sees flow and sound speeds, not the
acceleration. For cold or initially static gas under strong gravity the
first steps can be too long; `collapse_sph` uses a hot, dilute ambient
medium for exactly this reason.

## 10. Saving, loading, and restarting

`src/misc/io_utils.py` is one writer, one reader, and a restart helper —
mode-agnostic, since it dumps every `SimState` attribute rather than
special-casing per mode.

```python
save_data(filepath, grid, state, par, eos=None)        # -> path written (.npz appended if missing)
load_data(filepath) -> dict                             # flat dict, e.g. d['dens'], for analysis
restart_simulation(filepath, restore_hooks=True)        # -> (grid, state, par, eos), ready for a solver
save_1d_ascii(filepath, grid, state, par, eos=None)     # optional plain-text 1D profile (.dat)
```

**What is in a file.** Run metadata (mode, problem, `timenow`, `timefin`,
CFL, reconstruction / RK / solver / `divb_tr` / `rkl2_stages`, `BC`, `BCm`,
`BC_fixed`, `GAMMA`), the grid's construction parameters (geometry, bounds,
resolution, `Ngc` — the grid is *rebuilt*, not serialized, via
`grid_setup.reconstruct_grid`), the interior cell centres `cx1`, `cx2` for
analysis, and every array or scalar on `SimState` — including CT's
staggered `fb1`/`fb2`, GLM's `bglm` and the body forces `F1`/`F2`, none of
which can be recovered from cell-centred primitives alone. Fields are
stored exactly as in memory, i.e. ghost-inclusive for primitives and
interior-only for conservative variables ([§3](#3-the-ghost-cell--interior-cell-convention)).
`load_data` mirrors `save_data`'s `.npz` auto-append, so `load_data(p)`
works with the same `p` you passed to `save_data(p, ...)`.

**What `restart_simulation` does.**

1. Rebuilds the `Grid` with `reconstruct_grid`.
2. Rebuilds `Parameters` with its normal constructor (so derived fields like
   `Ngc` stay consistent with `rec_type`), then sets `timenow`, `timefin`,
   `BC`, `BCm`, `BC_fixed` from the file.
3. Allocates a fresh `SimState` and overwrites every field the file has —
   including fields an `__init__` might not pre-allocate (e.g. `adv`'s
   constant `vel1`/`vel2`).
4. With `restore_hooks=True` (default), restores `par.before_step`. A
   function can't be stored, so the problem's IC function is run once more on
   a scratch grid and state with the same settings, and its `before_step` is
   taken; the scratch objects are discarded and the restarted state is not
   touched. This works because the IC hooks capture only constants (masses,
   `G`, radii) and receive `grid`, `state`, `par` as arguments. The hook comes
   from the **current** IC code: if you edited, say, the planet mass after
   writing the file, the restarted run uses the new value.

The result is an exact continuation: a run restarted at `T/2` ends at `T`
with the same bits as the uninterrupted run (the `restart` testbed suite
checks this for every mode). What you may change between the restart and
the solver construction: `par.timefin` (to extend a run) and `par.CFL`.
What you may not: the resolution and `rec_type`, since the stored arrays
carry the original ghost-cell count.

**Periodic output.** `run_simulation` takes two optional arguments:

```python
run_simulation(grid, state, par, solver, var_to_plot, n_plot,
               dt_output=None, on_output=None)
```

Output number `k` is due at `t_k = k·dt_output`, with `k` counted from
`t = 0`, so after a restart the numbering continues. When `t_k` is reached,
`run_simulation` calls `on_output(state, par, k)` — any function; usually
one that calls `save_data`. Output times are hit exactly, with no change
to the solvers: each solver already shortens its last step as
`dt = min(dt_CFL, par.timefin - par.timenow)`, so `run_simulation` puts
`min(t_end, t_k)` into `par.timefin` for the duration of one step and the
true final time back right after it. The cost is at most one short step per
output. Neither `on_output` nor anything after the loop ever sees the
temporary value.

**In `main.py` / `main.ipynb`** this is wired up by a run-control block
(`OUTPUT_DIR`, `DT_OUTPUT`, `SAVE_ASCII_1D`, `RESTART`, `RESTART_FILE`,
`TIMEFIN_NEW`) and two small helpers, `snapshot_path` and `write_snapshot`.
Files are named `<OUTPUT_DIR>/<mode>_<problem>_<kkkk>.npz`, with `0000` the
initial condition, plus `<mode>_<problem>_final.npz` at the end of the
run. Restarting from snapshot `k` writes snapshots `k+1, k+2, ...` again,
overwriting the old ones — identical files if nothing was changed, new ones
if, say, the CFL number was.

## 11. Visualization

`src/misc/io_visual.py` is deliberately small: `plot_setup(grid, var, time)`
builds the right kind of matplotlib figure for the grid (a line plot in 1D;
`imshow` for 2D Cartesian/cylindrical; `pcolormesh` on the mapped `(x,y)` or
`(R,z)` vertices for 2D polar/spherical-polar), and `plotting(...)` updates
it in place each call. In a Jupyter notebook the figure is redrawn in the
cell; in plain Python (`python main.py`) it is redrawn in an interactive
window. IPython is only needed for the notebook case.

`run_simulation(grid, state, par, solver, var_to_plot, n_plot, dt_output=None,
on_output=None)` calls both for you (the last two arguments are for output,
see [§10](#10-saving-loading-and-restarting)). `var_to_plot` is either an array that the solver updates in place
(e.g. `state.dens`) or a function with no arguments returning the array,
e.g. `lambda: state.h + state.b` for the SWE free surface. A function is
re-evaluated before every plot, so it is the safe choice: it works for
derived quantities and for solvers that rebind a state array instead of
updating it in place. `main.py` and `main.ipynb` use functions.

## 12. The testbed

See the README's [Testbed](README.md#testbed) section for the day-to-day
`python run_testbed.py` usage and the suite table. This section is for
extending it.

`tests/testbed_common.py` is the shared plumbing: `build_case(mode, problem,
Nx1, Nx2, **par_kwargs)` builds a full `(grid, state, par, eos, solver)`
tuple the same way `main.py` does (reusing `SOLVER_DISPATCH`, so a new mode
registered there is automatically usable by the testbed too), and
`run_steps`/`run_to_tfin` advance it. The sanity and robustness suites use
non-square grids on purpose — a square grid hides axis-swap bugs.

- **Sanity** builds its catalogue directly from `PROBLEMS` in
  `src/misc/helpers.py` (every problem except `user_defined`), so a problem
  registered there is smoke-tested automatically — no test code to write.
- **Robustness** cases are opted in explicitly (a stress sweep is expensive,
  so not every catalogue problem gets one): call
  `_register_cross_product(_REGISTRY, prefix, mode, problem, Nx1, Nx2,
  solver_types, ...)` at module level in `test_robustness.py`, and it
  generates one `test_*` function per `solver_type × rec_type × RK_order`
  combination for that problem (skipping the documented `MP5`+`RK1`
  exception).
- **Convergence**: write a `test_*` function in `test_convergence.py` that
  runs the same problem at 2-3 increasing resolutions, measures
  `testbed_common.l2_error` against a known solution (or against itself at
  `t=0` for an exact-return-to-initial-state problem like periodic
  advection), and asserts `testbed_common.observed_order(errors)` is close
  to the design order. The MHD Alfvén-wave tests
  (`test_mhd_alfven_1d_convergence`, `test_mhd_alfven_2d_convergence`) are a
  template: a circularly-polarized Alfvén wave on a periodic domain returns
  exactly to its initial state after one period.

- **Restart**: add a line to `CASES` in `test_restart.py` —
  `(name, mode, problem, Nx1, Nx2, fraction of timefin, Parameters kwargs)`.
  The case is run uninterrupted to `T` and, separately, restarted from its
  snapshot at `T/2`; the two final states must agree **exactly** (zero
  difference, not round-off). Exact agreement is the right criterion: the
  restarted run starts from the same bits and takes the same time steps, so
  any difference means something was not stored or not restored. A control
  test checks that a self-gravitating run restarted *without* its hook does
  differ, so the suite can't pass for a trivial reason.

Every suite is plain `test_*` functions with plain `assert` — no pytest
dependency to run `run_testbed.py`, but `pytest tests/` works too if you
prefer that runner or want `-k`/`-x` filtering.

## 13. Use cases / recipes

### 13.1. Run a built-in problem

The two ways to do this (fast path via `main.py`, explicit path in Python)
are in the [README's Quickstart](README.md#quickstart).

### 13.2. Write a custom initial condition

Every mode ships a `user_defined` problem as a fill-in-the-blanks template —
pick the one for your mode (e.g. `src/models/HD/HD_init_cond.py`'s
`IC_HD_user_defined`), copy it under a new name, import it in
`src/misc/helpers.py` and add it to the mode's dict there (`hd_dispatch`
for HD), which is part of `PROBLEMS`. A minimal HD example:

```python
# src/models/HD/HD_init_cond.py
def IC_HD2D_my_blob(grid, fluid, par):
    print("my custom HD blob problem")

    grid.CartesianGrid(0.0, 1.0, 0.0, 1.0)
    par.timenow, par.timefin = 0.0, 0.3

    eos = EOSdata(5.0/3.0)

    x0, y0, r0 = 0.5, 0.5, 0.1
    rad = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    fluid.dens[:, :] = np.where(rad < r0, 10.0, 1.0)   # dense blob
    fluid.pres[:, :] = 1.0
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0

    par.BC[:] = 'free'          # all four faces: outflow

    return grid, fluid, par, eos
```

```python
# src/misc/helpers.py, in the hd_dispatch dict
"my-blob": IC_HD2D_my_blob,
```

The new problem is now available as `problem="my-blob"` and is picked up by
the sanity suite automatically. Note the pattern every `IC_*` function
follows: build the grid geometry first (`grid.<Geometry>Grid(...)`), then set
`par.timenow`/`par.timefin`, then fill primitive fields **on the full
ghost-inclusive array** (`[:, :]` — the solver's boundary-condition call
fills the ghosts before the first flux evaluation), then set `par.BC` (and
`par.before_step` if the problem needs per-step physics), then return.

### 13.3. Solve a custom elliptic problem

`solve_poisson` doesn't care why you need an elliptic solve — here's the
manufactured-solution check from the module's own docstring, verifying
`phi = sin(pi x) sin(pi y)` against its Laplacian on a unit square:

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
have. `phi` comes back ghost-padded and ready for `grid_misc.cell_gradient`/
`face_gradient`.

### 13.4. Drive the time loop by hand

`run_simulation` is a convenience wrapper, not a requirement:

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

Per-step physics set in `par.before_step` is called inside `step_RK`, so a
hand-written loop needs nothing extra for it. To write snapshots from your own
loop, call `save_data` where you like; note that without the `timefin` trick
of `run_simulation` ([§10](#10-saving-loading-and-restarting)) they land at
whatever time the step ends on.

### 13.5. Add self-gravity to a problem

Install the gravity call as the per-step hook (in your IC function, or
after `initial_model`):

```python
from src.gravity import selfgravity_poisson

# periodic box: only rho - <rho> gravitates (Jeans swindle)
par.before_step = lambda grid, state, par, dt: selfgravity_poisson(
    grid, state, par, G=1.0, BC=['peri'] * 4)
```

For an isolated object on a spherical grid, use a Dirichlet outer face with
the monopole potential of the total mass, recomputed every step — this is
what `collapse_sph` does:

```python
def gravity(grid, state, par, dt):
    Ngc = grid.Ngc
    M = np.sum(grid.cVol * state.dens[Ngc:-Ngc, Ngc:-Ngc])     # total mass
    return selfgravity_poisson(grid, state, par, G=1.0,
                               BC=['free', 'free', 'dirichlet', 'free'],
                               BC_value={2: -1.0 * M / r_out})
par.before_step = gravity
```

Here `'free'` at r = 0 and on the θ-axes is a symmetry condition, not an
outer boundary. Do not call gravity yourself before `solver.step_RK()` in
addition to the hook — that would compute it twice. See
[§9](#9-gravity-and-other-body-forces) for the choice of boundary conditions.

### 13.6. Save, restart, and post-process a run

With `main.py` or the notebook, set `DT_OUTPUT` and run; to continue, set
`RESTART = True`, `RESTART_FILE` and, for a finished run, `TIMEFIN_NEW`
(see the README's [Output and restart](README.md#output-and-restart)). The
same in your own script:

```python
from src.misc.io_utils import save_data, load_data, restart_simulation
from src.misc.helpers import run_simulation
from main import SOLVER_DISPATCH

# --- run with a snapshot every 0.5 time units ---
save = lambda st, p, k: save_data(f"runs/khi_{k:04d}", grid, st, p, eos)
state, par.timenow = run_simulation(grid, state, par, solver, lambda: state.dens,
                                    20, dt_output=0.5, on_output=save)

# --- offline analysis, no live objects needed ---
d = load_data("runs/khi_0002")
Ngc = d['Ngc']
rho = d['dens'][Ngc:-Ngc, Ngc:-Ngc]   # interior; d['cx1'], d['cx2'] match it
print("t =", d['timenow'], "mode =", d['mode'])

# --- resume from a snapshot and run 2 time units longer ---
grid, state, par, eos = restart_simulation("runs/khi_0002")   # before_step restored
par.timefin += 2.0
solver = SOLVER_DISPATCH[par.mode](grid, state, eos, par)
state, par.timenow = run_simulation(grid, state, par, solver, lambda: state.dens,
                                    20, dt_output=0.5, on_output=save)
```

The second `run_simulation` call continues the numbering at `khi_0003`.
Note that `save` refers to the `grid` and `eos` variables, so after the
restart it uses the restored ones.

### 13.7. Run and extend the testbed

```bash
python run_testbed.py                       # all five suites
python run_testbed.py --suite convergence    # one suite
python run_testbed.py -q                     # only print failures
```

A problem registered in `PROBLEMS` (§13.2) is already in the sanity suite.
If it should also stress-test every solver/reconstruction/RK-order
combination, register it in `test_robustness.py` (see
[§12](#12-the-testbed)).

## 14. Extending the framework

- **New reconstruction scheme**: add it to `src/common/high_order_rec.py`'s
  `VarReconstruct` dispatcher, and to `Parameters.__init__`'s `Ngc` rule if
  it needs a nonstandard ghost-cell count (2 for PCM/PLM-width stencils, 3
  otherwise, currently).
- **New Riemann solver**: add the flux function to the mode's
  `*_riemann_*.py` (same calling convention as its neighbours: left/right
  primitive states with x the normal direction, returning the fluxes), and
  add a branch for the new `solver_type` string to the dispatcher in the
  mode's `*_phys.py` (`Riemann_HD`, `Riemann_MHD`, `Riemann_rMHD`, ...).
  The dispatcher already handles the rotation for the x2 direction.
- **New per-step physics** (cooling, heating, damping zones, external
  potentials): write a function `f(grid, state, par, dt)` and assign it to
  `par.before_step` — see [§8](#8-per-step-physics-parbefore_step). No
  solver code needs to change.
- **New physics mode**: the file rhythm in [§7](#7-physics-modes-file-by-file)
  is the template — a new `SimState` branch, a `Parameters` mode entry, a
  `*_init_cond.py` with at least a `user_defined` problem, its dict in
  `PROBLEMS`, and a `SOLVER_DISPATCH` entry in `main.py`. `gravity.py` and
  `poisson_solver.py` are already mode-agnostic (they only touch
  `state.dens`/`F1`/`F2` and `grid`), so a new HD-family mode gets
  self-gravity by calling `par.before_step` in its `step_RK` the same way
  HD does.
- **New elliptic use of `solve_poisson`** (e.g. divergence cleaning as a
  post-step projection): follow the pattern in `poisson_solver.py`'s
  "Designed to be called from other modules" docstring section — solve for
  a scalar potential with the right BCs, then differentiate it with
  `face_gradient`/`cell_gradient`.

## 15. Known limitations

Deliberate simplifications of an educational code; each is a good exercise.

- **Source coupling is first order in time.** `par.before_step` freezes
  `F1`, `F2` over all RK stages; per-stage gravity would need a hook inside
  the stage loop.
- **SWE Coriolis term** is integrated with explicit Euler half-steps
  (energy of inertial motions grows like `exp(f² dt t / 2)`), and the bed
  source is not well-balanced (see [§7](#7-physics-modes-file-by-file)).
- **Relativistic con2prim** has no fallback: non-converged cells are only
  reported.
- **CT (MHD and rMHD) does not support `BC_fixed` inflow** (see
  [§5](#5-boundary-conditions)).
- **Poisson solver cost**: plain diagonally-preconditioned CG, O(N)
  iterations per solve; an FFT (periodic boxes) or multigrid solver would be
  much faster.
- **Restart keeps the resolution and `rec_type`** (the stored arrays have
  the ghost-cell count of the original run); there is no regridding.
- **Isolated self-gravity** needs user-supplied Dirichlet values (monopole
  or multipole); there is no built-in multipole expansion.

## 16. Conventions cheat-sheet

- **Ghost vs. interior**: primitive fields and coordinate arrays are
  ghost-inclusive (`grid.grid_shape`); conservative fields, `F1/F2`, `ST` and
  `fS1/fS2/cVol` are interior-only `(Nx1, Nx2)`, **no ghosts**. See [§3](#3-the-ghost-cell--interior-cell-convention).
- **Body-force sign**: `state.F1`/`F2` hold the acceleration itself
  (`F = a`), never its negative. See [§9](#9-gravity-and-other-body-forces).
- **Per-step physics** goes into `par.before_step(grid, state, par, dt)`:
  primitives (and `F1`, `F2`) only, first order in time; not stored in a
  snapshot, but rebuilt from the IC function by `restart_simulation`. See
  [§8](#8-per-step-physics-parbefore_step).
- **BC vocabulary differs by system**: hyperbolic solvers use `'free'` /
  `'wall'` / `'peri'` / `'axis'` (anything else raises); the Poisson solver
  uses `'free'` / `'peri'` / `'dirichlet'`. See [§5](#5-boundary-conditions).
- **Face index convention**: `0`=x1 inner, `1`=x2 inner, `2`=x1 outer,
  `3`=x2 outer — shared by `par.BC`, `par.BC_fixed`, and `solve_poisson`'s
  `BC`/`BC_value`.
- **Problem catalogue**: `PROBLEMS` in `src/misc/helpers.py` is the single
  source of truth; the sanity tests are built from it.
- **1D is 2D with `Nx2==1` (or `Nx1==1`)**, not a separate code path — the
  operators in `grid_misc.py` and the flux routines check `grid.Nx1 > 1`/
  `grid.Nx2 > 1` rather than assuming both are active.
- **`rec_type` fixes `Ngc`**: 2 for `PCM`/`PLM`, 3 for `PPMorig`/`PPM`/
  `WENO`/`MP5` (1 for `diff`, which doesn't use `rec_type`). Don't hard-code
  `Ngc=2` anywhere that might see a higher-order run.
- **Don't pair `MP5` with `RK1`**: forward Euler's stability margin is too
  small for MP5's low dissipation on a strong shock (loses positivity;
  fine on smooth data). Use `RK2`/`RK3` with `MP5`.

---

For the mode/solver/geometry option tables, the quickstart, the full
test-problem catalogue, and the reference list, see [`README.md`](README.md).
