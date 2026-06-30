# -*- coding: utf-8 -*-
"""
===============================================================================
poisson_solver.py
===============================================================================

Finite-volume Poisson solver for the Piastra framework.

Solves the elliptic equation

    div( grad(phi) ) = rhs

on the structured grid provided by grid_setup.Grid, in 1D (Nx1==1 or
Nx2==1) or 2D, and in every geometry supported by that class (Cartesian,
cylindrical, polar, spherical-polar), using a matrix-free,
diagonally-preconditioned Conjugate-Gradient (CG) iteration.

The discretisation reuses the same second-order, geometry-aware face
operators as the rest of Piastra (``grid_misc.face_gradient`` /
``grid_misc.div_face_vector`` -- the same building blocks used by the
thermal-diffusion solver, diff_step.py), so the Poisson operator is
automatically consistent with the grid metric in every supported geometry,
with no geometry branching in this module.

Boundary conditions
--------------------
Each of the four domain faces independently takes one of:

    'peri'      - periodic
    'free'      - zero-gradient (homogeneous Neumann)
    'dirichlet' - fixed boundary VALUE (second-order accurate)

via the single-ghost-layer filler ``apply_bc_scalar_Ngc1`` in
boundaries.py -- the natural stencil width of a second-order Laplacian.
This makes the solver reusable on grids with more ghost cells than it
needs (e.g. an Ngc=2/3 hydro or MHD grid borrowed for self-gravity or
magnetic-divergence cleaning): only the ghost layer touching the real
domain is filled; any deeper ghost layers are left untouched, which is
all that grid_misc's second-order operators ever read.

Pure-Neumann / pure-periodic domains (no 'dirichlet' face) only
determine phi up to an additive constant; the discrete right-hand side
must then satisfy the solvability condition (volume-integral of rhs
equal to zero). ``solve_poisson`` enforces this automatically by
subtracting the volume-weighted mean of `rhs` in that case, and the CG
iteration (started from a zero or mean-free initial guess) returns the
mean-free solution.

Designed to be called from other modules
-----------------------------------------
``solve_poisson`` is a small, stateless function: pass a grid, an
interior-cell source array and a BC description, get back a
ghost-padded potential. Typical callers:

  * self-gravity:        phi, info = solve_poisson(grid, 4*pi*G*rho, BC)
                          F1, F2 = [-g for g in grid_misc.cell_gradient(grid, phi)]
  * divergence cleaning:  phi, info = solve_poisson(grid, divB, BC)
                          fb1, fb2 = grid_misc.face_gradient(grid, phi)
                          B_clean_face = B_face - fb1, B_face - fb2

Usage
-----
>>> from src.grid.grid_setup import Grid
>>> from src.common.poisson_solver import solve_poisson
>>> import numpy as np
>>>
>>> g = Grid(64, 64, 2)
>>> g.CartesianGrid(0.0, 1.0, 0.0, 1.0)
>>> rhs = -2.0 * np.pi**2 * np.sin(np.pi * g.cx1[g.Ngc:-g.Ngc, g.Ngc:-g.Ngc]) \\
...       * np.sin(np.pi * g.cx2[g.Ngc:-g.Ngc, g.Ngc:-g.Ngc])
>>> BC = ['dirichlet', 'dirichlet', 'dirichlet', 'dirichlet']
>>> phi, info = solve_poisson(g, rhs, BC)
>>> info['converged']
True

Author: mrkondratyev
"""

import numpy as np

from src.grid.grid_misc import face_gradient, div_face_vector
from src.common.boundaries import apply_bc_scalar_Ngc1


# Face convention shared with par.BC / par.BC_fixed throughout Piastra:
# 0 = x1 inner, 1 = x2 inner, 2 = x1 outer, 3 = x2 outer.
_FACE_AXIS = {0: 1, 1: 2, 2: 1, 3: 2}
_FACE_SIDE = {0: 'inner', 1: 'inner', 2: 'outer', 3: 'outer'}
_VALID_BC  = ('peri', 'free', 'dirichlet')


def _check_BC(BC):
    """Validate the 4-face BC list and periodic-pairing consistency."""
    if len(BC) != 4:
        raise ValueError("BC must have exactly 4 entries "
                          "[x1_inner, x2_inner, x1_outer, x2_outer].")
    for bc in BC:
        if bc not in _VALID_BC:
            raise ValueError(
                f"Invalid BC entry: '{bc}'. Expected one of {_VALID_BC}.")
    if (BC[0] == 'peri') != (BC[2] == 'peri'):
        raise ValueError("x1 boundaries must both be 'peri' if either is.")
    if (BC[1] == 'peri') != (BC[3] == 'peri'):
        raise ValueError("x2 boundaries must both be 'peri' if either is.")



def boundCond_poisson(grid, phi, BC, BC_value=None):
    """
    Fill the single ghost layer bordering the domain on all 4 faces.

    Parameters
    ----------
    grid : Grid
    phi : ndarray, shape grid.grid_shape
    BC : sequence of 4 str
        [x1_inner, x2_inner, x1_outer, x2_outer], each in
        {'peri', 'free', 'dirichlet'}.
    BC_value : dict, optional
        Maps face index (0..3, same convention as BC) -> prescribed
        boundary value for 'dirichlet' faces (scalar or array
        broadcastable to the tangential dimension). A 'dirichlet' face
        absent from the dict -- or BC_value=None entirely -- defaults to
        0.0 (the homogeneous form used internally by the CG iteration).

    Returns
    -------
    phi : ndarray
        Field with the boundary-adjacent ghost layer updated.
    """
    Ngc = grid.Ngc
    for face in range(4):
        value = 0.0
        if BC_value is not None and BC[face] == 'dirichlet':
            value = BC_value.get(face, 0.0)
        phi = apply_bc_scalar_Ngc1(
            phi, Ngc, BC[face], axis=_FACE_AXIS[face], side=_FACE_SIDE[face],
            bc_value=value)
    return phi



def poisson_operator(grid, phi_int, BC, BC_value=None):
    """
    Apply the discrete operator  A(phi) = -div(grad(phi))  to an
    interior-cell field.

    This is the matrix-free matrix-vector product driving the CG
    iteration: pads `phi_int` with ghosts, fills the boundary-adjacent
    layer via ``boundCond_poisson``, and evaluates the same second-order
    face-gradient / face-divergence pair the diffusion solver uses
    (``grid_misc.face_gradient``, ``grid_misc.div_face_vector``), so the
    operator is automatically consistent with the grid's geometry.

    Parameters
    ----------
    grid : Grid
    phi_int : ndarray, shape (Nx1, Nx2)
        Field on interior cells only.
    BC : sequence of 4 str
    BC_value : dict, optional
        Dirichlet boundary values (see ``boundCond_poisson``). Leave as
        None to apply the HOMOGENEOUS form of the same boundary types --
        what CG uses internally for its search directions.

    Returns
    -------
    Aphi : ndarray, shape (Nx1, Nx2)
    """
    Ngc = grid.Ngc
    phi = np.zeros(grid.grid_shape, dtype=np.double)
    phi[Ngc:-Ngc, Ngc:-Ngc] = phi_int
    phi = boundCond_poisson(grid, phi, BC, BC_value)

    g1, g2 = face_gradient(grid, phi)
    return -div_face_vector(grid, g1, g2)



def _face_conductance(grid):
    """
    Geometric face conductances C1, C2 such that the kappa=1 diffusive
    flux at a face equals C * (value on the + side - value on the - side)
    -- the same quantity implicitly built by grid_misc.face_gradient,
    isolated here for the diagonal (Jacobi) preconditioner below.

    Returns
    -------
    C1 : ndarray, shape (Nx1+1, Nx2)
    C2 : ndarray, shape (Nx1, Nx2+1)
    """
    Ngc, Nx1, Nx2, Nx1r, Nx2r = grid.Ngc, grid.Nx1, grid.Nx2, grid.Nx1r, grid.Nx2r

    if Nx1 > 1:
        dist1 = (grid.cx1[Ngc:Nx1r + 1, Ngc:Nx2r] -
                 grid.cx1[Ngc - 1:Nx1r, Ngc:Nx2r])
        C1 = grid.fS1 / dist1
    else:
        C1 = np.zeros((Nx1 + 1, Nx2))

    if Nx2 > 1:
        dist2 = (grid.cx2[Ngc:Nx1r, Ngc:Nx2r + 1] -
                 grid.cx2[Ngc:Nx1r, Ngc - 1:Nx2r])
        h2 = grid.hx2[Ngc:Nx1r, Ngc]               # hx2 depends on x1 only
        C2 = grid.fS2 / dist2 / h2[:, None]
    else:
        C2 = np.zeros((Nx1, Nx2 + 1))

    return C1, C2



def _diag_operator(grid, BC):
    """
    Diagonal of the operator A = -div(grad(.)), used for Jacobi
    (diagonal) preconditioning.

    A boundary face contributes: 0 (no coupling at all) for 'free' --
    the mirrored ghost makes that face's flux vanish identically, for
    any value of the boundary cell; the bare conductance for 'peri' --
    it behaves exactly like an interior face, just wrapped around; and
    twice the bare conductance for 'dirichlet' -- the ghost is mirrored
    about the fixed face value, doubling the effective gradient there.

    Parameters
    ----------
    grid : Grid
    BC : sequence of 4 str

    Returns
    -------
    diagA : ndarray, shape (Nx1, Nx2)
    """
    C1, C2 = _face_conductance(grid)
    diagA = np.zeros((grid.Nx1, grid.Nx2))
    bc_weight = {'free': 0.0, 'peri': 1.0, 'dirichlet': 2.0}

    if grid.Nx1 > 1:
        w1 = np.ones_like(C1)
        w1[0, :]  = bc_weight[BC[0]]
        w1[-1, :] = bc_weight[BC[2]]
        Cw1 = C1 * w1
        diagA += (Cw1[1:, :] + Cw1[:-1, :]) / grid.cVol

    if grid.Nx2 > 1:
        w2 = np.ones_like(C2)
        w2[:, 0]  = bc_weight[BC[1]]
        w2[:, -1] = bc_weight[BC[3]]
        Cw2 = C2 * w2
        diagA += (Cw2[:, 1:] + Cw2[:, :-1]) / grid.cVol

    return diagA



def _dot(grid, u, v):
    """
    Volume-weighted inner product, consistent with grid_misc.Ln_norm /
    integral_over_grid -- the inner product under which the
    finite-volume operator `poisson_operator` is self-adjoint, since the
    FV equation at cell i is naturally weighted by cVol[i].
    """
    return np.sum(grid.cVol * u * v)



def solve_poisson(grid, rhs, BC, BC_value=None, phi0=None,
                   tol=1e-10, maxiter=None, verbose=False):
    """
    Solve the finite-volume Poisson equation  div(grad(phi)) = rhs  on
    `grid`, using diagonally-preconditioned Conjugate Gradient.

    Works in 1D (Nx1==1 or Nx2==1) and 2D, and in every geometry
    supported by Grid (Cartesian, cylindrical, polar, spherical-polar):
    the discretisation is expressed entirely through grid metric
    quantities (fS1, fS2, cVol, cx1, cx2, hx2), with no geometry
    branching in this module.

    Parameters
    ----------
    grid : Grid
        Any grid (Ngc >= 1). Only the ghost layer immediately bordering
        the real domain is ever touched, so a grid built for a
        hyperbolic solver (Ngc=2 or 3) can be reused directly -- e.g. to
        solve for the self-gravity potential of an HD/MHD density field,
        or for a magnetic-divergence-cleaning scalar.
    rhs : ndarray, shape (Nx1, Nx2)
        Source term on interior cells (e.g. 4*pi*G*rho for self-gravity,
        or div(B) for divergence cleaning).
    BC : sequence of 4 str
        [x1_inner, x2_inner, x1_outer, x2_outer], each one of 'peri',
        'free' (zero-gradient), 'dirichlet' (fixed value). Opposite
        faces must agree on 'peri' (periodicity is a property of the
        whole direction, not of one face).
    BC_value : dict, optional
        Maps face index (0..3) -> prescribed boundary value for
        'dirichlet' faces (scalar or array broadcastable to the
        tangential direction). Defaults to 0.0 on any 'dirichlet' face
        not given explicitly.
    phi0 : ndarray, optional
        Initial guess, full grid_shape (with ghosts). Defaults to zero.
    tol : float, optional
        Relative residual tolerance ||r|| / ||rhs|| (volume-weighted L2
        norm). Default 1e-10.
    maxiter : int, optional
        Maximum number of CG iterations. Defaults to Nx1*Nx2 (CG is
        exact within that many iterations in infinite precision; the
        diagonal preconditioner makes practical convergence far faster).
    verbose : bool, optional
        Print final iteration count and residual. Default False.

    Returns
    -------
    phi : ndarray, shape grid.grid_shape
        Solution, with the boundary-adjacent ghost layer filled
        consistently with BC -- ready for direct use in
        grid_misc.cell_gradient / face_gradient by the caller.
    info : dict
        {'niter': int, 'residual': float, 'converged': bool} -- number
        of CG iterations taken, final relative residual, and whether
        `tol` was reached.
    """
    _check_BC(BC)

    Ngc = grid.Ngc
    Nx1, Nx2 = grid.Nx1, grid.Nx2
    shape = (Nx1, Nx2)

    if maxiter is None:
        maxiter = Nx1 * Nx2

    rhs = np.asarray(rhs, dtype=np.double)

    # Pure-Neumann / pure-periodic problems only fix phi up to a constant,
    # and are solvable only if the volume integral of rhs vanishes -- the
    # discrete analogue of the boundary flux balancing the source. Project
    # rhs onto that compatible subspace automatically.
    no_dirichlet = 'dirichlet' not in BC
    if no_dirichlet:
        mean_rhs = _dot(grid, rhs, np.ones(shape)) / np.sum(grid.cVol)
        if abs(mean_rhs) > 1e-300:
            rhs = rhs - mean_rhs

    b = -rhs

    phi_int = (np.array(phi0[Ngc:-Ngc, Ngc:-Ngc], dtype=np.double)
               if phi0 is not None else np.zeros(shape, dtype=np.double))

    diagA = _diag_operator(grid, BC)
    diagA_safe = np.where(np.abs(diagA) > 1e-300, diagA, 1.0)

    r = b - poisson_operator(grid, phi_int, BC, BC_value)
    z = r / diagA_safe
    p = z.copy()
    rz_old = _dot(grid, r, z)

    bnorm = np.sqrt(_dot(grid, b, b))
    bnorm_safe = bnorm if bnorm > 1e-300 else 1.0

    niter = 0
    converged = np.sqrt(_dot(grid, r, r)) <= tol * bnorm_safe

    while not converged and niter < maxiter:
        Ap = poisson_operator(grid, p, BC, None)        # homogeneous BC
        pAp = _dot(grid, p, Ap)
        if abs(pAp) < 1e-300:
            break
        alpha = rz_old / pAp

        phi_int += alpha * p
        r -= alpha * Ap
        niter += 1

        resnorm = np.sqrt(_dot(grid, r, r))
        converged = resnorm <= tol * bnorm_safe
        if converged:
            break

        z = r / diagA_safe
        rz_new = _dot(grid, r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    residual = np.sqrt(_dot(grid, r, r)) / bnorm_safe

    if verbose:
        print(f"solve_poisson: niter = {niter}, residual = {residual:.3e}, "
              f"converged = {converged}")

    phi = np.zeros(grid.grid_shape, dtype=np.double)
    phi[Ngc:-Ngc, Ngc:-Ngc] = phi_int
    phi = boundCond_poisson(grid, phi, BC, BC_value)

    info = {'niter': niter, 'residual': residual, 'converged': bool(converged)}
    return phi, info
