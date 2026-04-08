# -*- coding: utf-8 -*-
"""
===============================================================================
diffusion_one_step.py
===============================================================================

2D thermal diffusion solver for the Piastra framework.

Solves the parabolic equation

    ∂T/∂t = ∇·(κ ∇T)

on the structured 2D grid provided by grid_setup.Grid, using a
finite-volume discretisation that is consistent with all geometries
supported by that class (Cartesian, cylindrical, polar).

Two time-integration methods are available, selected via the ``solver``
argument of Diffusion2D:

``'expl'`` – Explicit forward-Euler.
    Stable for

        dt  ≤  dx1² · dx2² / (2 · κ_max · (dx1² + dx2²))

``'rkl2'`` – RKL2 Super Time Stepping (Meyer, Balsara & Aslam 2014,
    MNRAS 422, 2102).  Each super-step spans

        dt_super  =  dt_expl · (s² + s − 2) / 4

    with ``s`` stages, giving an effective speed-up of ∼ s²/4 over the
    explicit scheme at the same second-order temporal accuracy.

Spatial discretisation
----------------------
Diffusive fluxes are computed at cell faces and accumulated into the
finite-volume residual:

    L(T)[i,j] = [ F1_{i+½,j}·S1_{i+½,j} − F1_{i-½,j}·S1_{i-½,j}
                + F2_{i,j+½}·S2_{i,j+½} − F2_{i,j-½}·S2_{i,j-½} ]
                / V_{i,j}

where  F = κ · ∂T/∂n  at the face.  Face areas ``fS1``, ``fS2`` and
cell volumes ``cVol`` are taken directly from the Grid object, so the
operator is automatically geometry-aware.

Variable diffusivity
--------------------
``kappa`` in SimState may be a scalar or a 2-D cell-centred array
(shape grid.grid_shape).  Face-centred values are obtained by arithmetic
averaging of the two neighbouring cells.

Boundary conditions
-------------------
Boundary conditions are applied through ``apply_bc_scalar`` from
``boundaries.py``.  The four-element array ``par.BC`` encodes:

    BC[0] : x1 inner (left / bottom-R)
    BC[1] : x2 inner (bottom / inner-Z)
    BC[2] : x1 outer (right / top-R)
    BC[3] : x2 outer (top / outer-Z)

Supported types: ``'free'`` (zero-gradient), ``'wall'`` (identical to
``'free'`` for scalars), ``'peri'`` (periodic).

Usage
-----
>>> from grid_setup         import Grid
>>> from sim_state          import SimState
>>> from diffusion_one_step import Diffusion2D
>>>
>>> g   = Grid(128, 128, 2)
>>> g.CartesianGrid(0.0, 1.0, 0.0, 1.0)
>>> state = SimState(g, par)          # par.mode == 'diff'
>>> # … fill state.T with initial condition …
>>>
>>> par.timenow = 0.0
>>> par.timefin = 1.0
>>> par.CFL     = 0.9
>>> par.BC      = np.array(['free', 'free', 'free', 'free'])
>>>
>>> solver = Diffusion2D(g, state, par, solver='rkl2', rkl2_stages=16)
>>> while par.timenow < par.timefin:
...     state = solver.step_RK()

Author: mrkondratyev
"""

import numpy as np
import copy

from src.common.boundaries import apply_bc_scalar


# ============================================================================
#   Top-level solver class
# ============================================================================

class Diffusion2D:
    """
    Container class for 2D thermal-diffusion routines.

    Provides a ``step_RK()`` method that advances the temperature field
    by one (super-)timestep using either the explicit Euler scheme or
    the RKL2 super time-stepping algorithm.  The interface is intentionally
    kept identical to the other Piastra solver classes (Advection2D,
    Hydro2D, …) so that the same ``run_simulation`` loop can drive it.

    Parameters
    ----------
    g : Grid
        Grid object with attributes Nx1, Nx2, Ngc, dx1uc, dx2uc,
        fS1, fS2, cVol.
    diff : SimState
        State container with arrays T and kappa (mode='diff').
    par : object
        Parameters object.  Must expose:
        ``timenow`` (float), ``timefin`` (float),
        ``CFL`` (float), ``BC`` (array of 4 strings).
    solver : {'expl', 'rkl2'}, optional
        Time-integration method.  Default ``'expl'``.
    rkl2_stages : int, optional
        Number of RKL2 stages s ≥ 2.  Only used when
        ``solver='rkl2'``.  Larger values give a proportionally larger
        super-step but increase cost per step.  Default 10.

    Attributes
    ----------
    g    : Grid
    diff : SimState
    par  : parameters
    """

    def __init__(self, g, diff, par, solver='expl', rkl2_stages=10):
        self.g      = g
        self.diff   = diff
        self.par    = par
        self.solver = solver
        self.s      = int(rkl2_stages)

        if solver not in ('expl', 'rkl2'):
            raise ValueError(
                f"Unknown solver '{solver}'. Choose 'expl' or 'rkl2'."
            )
        if solver == 'rkl2':
            if self.s < 2:
                raise ValueError("rkl2_stages must be >= 2.")
            self._b, self._mu, self._nu, self._gamma = _rkl2_coefs(self.s)

    # ------------------------------------------------------------------
    def step_RK(self):
        """
        Advance the diffusion state by one (super-)timestep.

        Computes the CFL-limited timestep, advances ``diff.T`` in place,
        updates ``par.timenow``, and returns the updated state.

        Returns
        -------
        diff : SimState
            Updated diffusion state.
        """
        dt_cfl = _CFL_condition_diff(self.g, self.diff, self.par.CFL)

        if self.solver == 'expl':
            dt = min(dt_cfl, self.par.timefin - self.par.timenow)
            self.diff = _explicit_step_diff(self.g, self.diff, self.par, dt)

        else:  # rkl2
            dt_super = dt_cfl * (self.s**2 + self.s - 2) / 4.0
            dt = min(dt_super, self.par.timefin - self.par.timenow)
            self.diff = _rkl2_step_diff(
                self.g, self.diff, self.par, dt,
                self._b, self._mu, self._nu, self._gamma, self.s
            )

        self.par.timenow += dt
        return self.diff


# ============================================================================
#   CFL condition
# ============================================================================

def _CFL_condition_diff(g, diff, CFL):
    """
    Compute the CFL-limited explicit timestep for diffusion.

    For a uniform grid the von Neumann stability condition is

        dt  ≤  dx1² · dx2² / (2 · κ_max · (dx1² + dx2²))

    Parameters
    ----------
    g    : Grid
    diff : SimState
    CFL  : float
        Safety factor (< 1).

    Returns
    -------
    dt : float
        Maximum stable explicit timestep multiplied by the CFL factor.
    """
    kappa_max = (
        float(diff.kappa)
        if np.isscalar(diff.kappa)
        else float(np.max(diff.kappa))
    )
    dx1 = g.dx1uc
    dx2 = g.dx2uc
    dt = CFL * dx1**2 * dx2**2 / (2.0 * kappa_max * (dx1**2 + dx2**2))
    return dt


# ============================================================================
#   Boundary conditions
# ============================================================================

def _apply_bc_diff(g, diff, par):
    """
    Fill ghost cells of diff.T using par.BC.

    Parameters
    ----------
    g    : Grid
    diff : SimState  (modified in place)
    par  : parameters  (par.BC[0..3])
    """
    Ngc = g.Ngc
    diff.T = apply_bc_scalar(diff.T, Ngc, par.BC[0], axis=1, side='inner')
    diff.T = apply_bc_scalar(diff.T, Ngc, par.BC[1], axis=2, side='inner')
    diff.T = apply_bc_scalar(diff.T, Ngc, par.BC[2], axis=1, side='outer')
    diff.T = apply_bc_scalar(diff.T, Ngc, par.BC[3], axis=2, side='outer')
    return diff


# ============================================================================
#   Spatial operator  L(T) = ∇·(κ ∇T)
# ============================================================================

def _face_kappa(kappa, Ngc, Nx1r, Nx2r, direction):
    """
    Compute the arithmetic-mean face-centred diffusivity.

    Parameters
    ----------
    kappa     : float or ndarray of shape (Nx1+2*Ngc, Nx2+2*Ngc)
    Ngc       : int
    Nx1r, Nx2r: int  (= Nx1+Ngc, Nx2+Ngc)
    direction : int  (1 → x1 faces, 2 → x2 faces)

    Returns
    -------
    kf : float or ndarray of the appropriate face shape
    """
    if np.isscalar(kappa):
        return kappa

    if direction == 1:
        # shape (Nx1+1, Nx2)
        return 0.5 * (kappa[Ngc:Nx1r + 1, Ngc:Nx2r] +
                      kappa[Ngc - 1:Nx1r,  Ngc:Nx2r])
    else:
        # shape (Nx1, Nx2+1)
        return 0.5 * (kappa[Ngc:Nx1r, Ngc:Nx2r + 1] +
                      kappa[Ngc:Nx1r, Ngc - 1:Nx2r])


def _spatial_operator_diff(g, diff):
    """
    Evaluate the finite-volume diffusion operator on the interior cells.

    Computes the discrete divergence of the diffusive flux:

        L(T)[i,j] = [ F1_{i+½} · S1_{i+½} − F1_{i-½} · S1_{i-½}
                    + F2_{j+½} · S2_{j+½} − F2_{j-½} · S2_{j-½} ]
                    / V_{i,j}

    where

        F1_{i+½,j} = κ_{i+½,j} · (T[i+1,j] − T[i,j]) / dx1
        F2_{i,j+½} = κ_{i,j+½} · (T[i,j+1] − T[i,j]) / dx2

    Parameters
    ----------
    g    : Grid
    diff : SimState

    Returns
    -------
    LT : ndarray, shape (Nx1, Nx2)
        Rate-of-change contribution from diffusion on each interior cell.
    """
    Ngc  = g.Ngc
    Nx1r = g.Nx1r   # = Nx1 + Ngc
    Nx2r = g.Nx2r   # = Nx2 + Ngc
    T    = diff.T

    # Face-centred diffusivities
    kf1 = _face_kappa(diff.kappa, Ngc, Nx1r, Nx2r, direction=1)  # (Nx1+1, Nx2)
    kf2 = _face_kappa(diff.kappa, Ngc, Nx1r, Nx2r, direction=2)  # (Nx1, Nx2+1)

    # Diffusive fluxes (gradient × face diffusivity)
    # x1: Nx1+1 faces for Nx2 real cells
    flux1 = kf1 * (T[Ngc:Nx1r + 1, Ngc:Nx2r] -
                   T[Ngc - 1:Nx1r,  Ngc:Nx2r]) / g.dx1uc   # (Nx1+1, Nx2)

    # x2: Nx1 real cells by Nx2+1 faces
    flux2 = kf2 * (T[Ngc:Nx1r, Ngc:Nx2r + 1] -
                   T[Ngc:Nx1r, Ngc - 1:Nx2r]) / g.dx2uc   # (Nx1, Nx2+1)

    # Finite-volume divergence
    LT = ((flux1[1:, :] * g.fS1[1:, :] - flux1[:-1, :] * g.fS1[:-1, :]) +
          (flux2[:, 1:] * g.fS2[:, 1:] - flux2[:, :-1] * g.fS2[:, :-1])
          ) / g.cVol                                         # (Nx1, Nx2)

    return LT


# ============================================================================
#   Explicit Euler step
# ============================================================================

def _explicit_step_diff(g, diff, par, dt):
    """
    One explicit forward-Euler timestep for the diffusion equation.

        T^{n+1} = T^n + dt · L(T^n)

    Parameters
    ----------
    g    : Grid
    diff : SimState
    par  : parameters
    dt   : float

    Returns
    -------
    diff : SimState
        Updated state (T modified in place).
    """
    Ngc = g.Ngc
    diff = _apply_bc_diff(g, diff, par)

    LT = _spatial_operator_diff(g, diff)
    diff.T[Ngc:-Ngc, Ngc:-Ngc] += dt * LT

    return diff


# ============================================================================
#   RKL2 super time-stepping
# ============================================================================

def _rkl2_coefs(s):
    """
    Pre-compute the RKL2 recursion coefficients for s stages.

    Reference: Meyer, Balsara & Aslam (2014), MNRAS 422, 2102,
    equations (17)–(20).

    Parameters
    ----------
    s : int
        Number of RKL2 stages (>= 2).

    Returns
    -------
    b, mu, nu, gamma : ndarray, each of length s+1
    """
    w1    = 4.0 / (s**2 + s - 2.0)
    b     = np.zeros(s + 1)
    mu    = np.zeros(s + 1)
    nu    = np.zeros(s + 1)
    gamma = np.zeros(s + 1)

    b[0] = b[1] = 1.0 / 3.0
    for j in range(2, s + 1):
        b[j]     = (j**2 + j - 2.0) / (2.0 * j * (j + 1.0))
        mu[j]    = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
        nu[j]    = -(j - 1.0) / j   * b[j] / b[j - 2]
        gamma[j] = -(1.0 - b[j - 1]) * mu[j] * w1

    return b, mu, nu, gamma


def _rkl2_step_diff(g, diff, par, dt, b, mu, nu, gamma, s):
    """
    One RKL2 super-step for the diffusion equation.

    Implements the second-order Runge-Kutta-Legendre scheme with s stages.
    The super-step size is  dt_super = dt_expl · (s² + s − 2) / 4,
    yielding an effective speed-up of ≈ s²/4 versus the explicit scheme.

    Recursion (Meyer, Balsara & Aslam 2014, eq. 14):

        Y_1 = T^n + (w1/3) · dt · L(T^n)

        Y_j = μ_j · Y_{j-1} + ν_j · Y_{j-2}
              + w1·μ_j·dt · L(Y_{j-1})
              + (1 − μ_j − ν_j) · T^n
              + γ_j · dt · L(T^n)

    ``Y0`` in the loop carries Y_{j-2} (the two-stage-back value).
    ``Tn`` is the original T^n, kept fixed throughout.

    Parameters
    ----------
    g              : Grid
    diff           : SimState
    par            : parameters
    dt             : float    (super-step size)
    b, mu, nu,
    gamma          : ndarray  (RKL2 coefficients from _rkl2_coefs)
    s              : int      (number of stages)

    Returns
    -------
    diff : SimState
        Updated state.
    """
    Ngc = g.Ngc
    w1  = 4.0 / (s**2 + s - 2.0)

    diff = _apply_bc_diff(g, diff, par)

    # Keep a reference to the original T array so the final result can be
    # written back in-place.  This makes RKL2 consistent with the explicit
    # scheme (which also updates diff.T in-place) and ensures that any
    # external reference to the T array (e.g. T = state.T captured before
    # the time loop) sees the correct updated values.
    T_out = diff.T

    # Save T^n; pre-compute L(T^n) once — both are reused at every stage.
    Tn  = diff.T.copy()                         # T^n, never modified
    LT0 = _spatial_operator_diff(g, diff)       # L(T^n), shape (Nx1, Nx2)

    # ---- stage 1 ----
    # Pre-allocate three rotating buffers (avoids per-stage allocation)
    Y0 = Tn.copy()                              # Y_{j-2} initialised to T^n
    Y1 = Tn.copy()
    Y1[Ngc:-Ngc, Ngc:-Ngc] = Tn[Ngc:-Ngc, Ngc:-Ngc] + (w1 / 3.0) * dt * LT0
    Y2 = np.empty_like(Tn)                      # scratch buffer, reused each stage

    # ---- stages 2 … s ----
    for j in range(2, s + 1):

        # Apply BCs to Y_{j-1} before evaluating L(Y_{j-1})
        diff.T = Y1
        diff   = _apply_bc_diff(g, diff, par)
        Y1     = diff.T

        LT1 = _spatial_operator_diff(g, diff)   # L(Y_{j-1})

        # Copy ghost cells from Y1 into scratch buffer, then overwrite interior
        np.copyto(Y2, Y1)
        Y2[Ngc:-Ngc, Ngc:-Ngc] = (
              mu[j]                      * Y1[Ngc:-Ngc, Ngc:-Ngc]   # μ_j · Y_{j-1}
            + nu[j]                      * Y0[Ngc:-Ngc, Ngc:-Ngc]   # ν_j · Y_{j-2}
            + w1 * mu[j] * dt            * LT1                       # w1·μ_j·dt·L(Y_{j-1})
            + (1.0 - mu[j] - nu[j])     * Tn[Ngc:-Ngc, Ngc:-Ngc]   # (1−μ−ν)·T^n
            + gamma[j] * dt              * LT0                       # γ_j·dt·L(T^n)
        )

        Y0, Y1, Y2 = Y1, Y2, Y0   # rotate buffers (no allocation)

    # Write the final stage result back into the original T array in-place,
    # then restore diff.T to point to that array.
    np.copyto(T_out, Y1)
    diff.T = T_out
    return diff
