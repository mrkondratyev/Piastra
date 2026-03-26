# -*- coding: utf-8 -*-
"""
===============================================================================
elliptic_solver.py
===============================================================================

Iterative solver for second-order elliptic boundary-value problems on the
structured 2D grids provided by grid_setup.Grid.

The module targets equations of the form

    ∇·(α ∇u) + γ u = f          (*)

where α (scalar or cell-centred field), γ (scalar or field), and f are
known, and u is the unknown.  Special cases:

- Poisson equation:  α = 1, γ = 0   →   ∇²u = f
- Helmholtz equation: α = 1          →   ∇²u + γ u = f
- Variable-coefficient diffusion: γ = 0  →  ∇·(α ∇u) = f

Two iterative solvers are provided (selectable via the ``method``
parameter):

- ``'multigrid'``: Geometric multigrid V-cycle with weighted-Jacobi
  smoothing and standard coarsening (factor 2 in each active direction).
- ``'cg'``: Conjugate Gradient with volume-weighted inner product.

Supported geometries: Cartesian (cart), Cylindrical (cyl), Polar (pol).
Metric terms are handled through face areas and cell volumes from
``Grid``, following the same finite-volume conventions as the rest of
the Piastra framework.

Author: mrkondratyev
"""

import numpy as np
from grid_setup import Grid


# ============================================================================
# Data container for the elliptic problem
# ============================================================================

class EllipticProblem:
    """
    Descriptor for a scalar elliptic BVP on a 2D structured grid.

    Parameters
    ----------
    grid : Grid
        Grid object from grid_setup (geometry must already be initialised).
    alpha : ndarray or float, optional
        Diffusion coefficient α(x1, x2).  Default is 1.0.
    gamma : ndarray or float, optional
        Reaction coefficient γ(x1, x2).  Default is 0.0.
    rhs : ndarray, optional
        Right-hand-side source f(x1, x2), shape ``(Nx1, Nx2)`` or
        ``grid.grid_shape``.  Default is 0.0.
    bc_type : dict, optional
        Boundary-condition type for each face.  Keys: ``'x1lo'``,
        ``'x1hi'``, ``'x2lo'``, ``'x2hi'``.  Values: ``'dirichlet'``,
        ``'neumann'``, or ``'periodic'``.  Default: Dirichlet everywhere.
    bc_val : dict, optional
        Boundary values for each face (float or 1-D array along the face).
        For Dirichlet: prescribed u.  For Neumann: prescribed ∂u/∂n
        (outward normal).  Default is 0.0 on every face.
    """

    _faces = ('x1lo', 'x1hi', 'x2lo', 'x2hi')

    def __init__(self, grid, *,
                 alpha=1.0, gamma=0.0,
                 rhs=None,
                 bc_type=None, bc_val=None):

        self.grid = grid
        shape = grid.grid_shape

        self.alpha = np.broadcast_to(
            np.asarray(alpha, dtype=np.float64), shape).copy()
        self.gamma = np.broadcast_to(
            np.asarray(gamma, dtype=np.float64), shape).copy()

        if rhs is None:
            self.rhs = np.zeros(shape, dtype=np.float64)
        else:
            self.rhs = np.asarray(rhs, dtype=np.float64).copy()
            if self.rhs.shape != shape:
                tmp = np.zeros(shape, dtype=np.float64)
                Ngc = grid.Ngc
                tmp[Ngc:-Ngc, Ngc:-Ngc] = self.rhs
                self.rhs = tmp

        if bc_type is None:
            bc_type = {f: 'dirichlet' for f in self._faces}
        if bc_val is None:
            bc_val = {f: 0.0 for f in self._faces}
        self.bc_type = bc_type
        self.bc_val  = bc_val


# ============================================================================
# Elliptic solver
# ============================================================================

class EllipticSolver:
    """
    Iterative solver for :class:`EllipticProblem`.

    Parameters
    ----------
    problem : EllipticProblem
        Fully specified elliptic BVP.
    method : str, optional
        ``'multigrid'`` (geometric V-cycle) or ``'cg'`` (Conjugate Gradient).
        Default is ``'multigrid'``.
    tol : float, optional
        Convergence tolerance on the volume-weighted L2 residual norm.
        Default is 1e-6.
    max_iter : int, optional
        Maximum number of outer iterations.  Default is 500.
    omega_smooth : float, optional
        Damping factor for the weighted-Jacobi smoother (multigrid only).
        Default is 2/3, which is optimal for high-frequency damping.
    mg_pre : int, optional
        Number of pre-smoothing sweeps per V-cycle.  Default is 2.
    mg_post : int, optional
        Number of post-smoothing sweeps per V-cycle.  Default is 2.
    mg_bottom_iter : int, optional
        Number of smoother iterations on the coarsest grid.  Default is 50.
    verbose : bool, optional
        Print convergence information.  Default is False.
    print_every : int, optional
        Reporting interval when ``verbose`` is True.  Default is 10.

    Attributes
    ----------
    u : ndarray, shape grid.grid_shape
        Solution field (including ghost zones).
    residual_history : list of float
        Residual norm after each outer iteration.
    converged : bool
        Whether the solver reached the requested tolerance.
    """

    _known_methods = ('cg', 'multigrid')

    def __init__(self, problem, *,
                 method='multigrid', tol=1e-6, max_iter=500,
                 omega_smooth=2.0/3.0,
                 mg_pre=2, mg_post=2, mg_bottom_iter=50,
                 verbose=False, print_every=10):

        if method not in self._known_methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Expected one of {self._known_methods}.")

        self.problem    = problem
        self.method     = method
        self.tol        = tol
        self.max_iter   = max_iter
        self.omega_smooth = omega_smooth
        self.mg_pre     = mg_pre
        self.mg_post    = mg_post
        self.mg_bottom_iter = mg_bottom_iter
        self.verbose    = verbose
        self.print_every = print_every

        self.u = np.zeros(problem.grid.grid_shape, dtype=np.float64)
        self.residual_history = []
        self.converged = False

        # Build stencil on the finest grid
        self._grids    = [problem.grid]
        self._stencils = [self._build_stencil(
            problem.grid, problem.alpha, problem.gamma)]

        # Build multigrid hierarchy (coarser levels use Poisson stencil)
        if method == 'multigrid':
            self._build_hierarchy()

    # ================================================================
    # Public interface
    # ================================================================

    def solve(self, u0=None):
        """
        Solve the elliptic BVP.

        Parameters
        ----------
        u0 : ndarray, optional
            Initial guess.  If ``None``, the current ``self.u`` is used.

        Returns
        -------
        u : ndarray, shape grid.grid_shape
            Converged (or best-effort) solution including ghost zones.
        """
        if u0 is not None:
            self.u[:] = u0
        self.residual_history.clear()
        self.converged = False

        {'cg': self._solve_cg,
         'multigrid': self._solve_multigrid}[self.method]()
        return self.u

    def residual(self):
        """Return r = f − L[u] on real cells, shape (Nx1, Nx2)."""
        grid = self._grids[0]
        self._apply_bc(self.u, grid, homogeneous=False)
        return self._compute_residual(self.u, self.problem.rhs,
                                      grid, self._stencils[0])

    def residual_norm(self):
        """Volume-weighted L2 norm of the residual."""
        return self._norm(self.residual(), self._grids[0])

    # ================================================================
    # Stencil assembly
    # ================================================================

    @staticmethod
    def _build_stencil(grid, alpha_full=None, gamma_full=None):
        """
        Build the 5-point stencil for ∇·(α ∇u) + γ u using FV
        face-areas and cell volumes.

        Returns (aP, aE, aW, aN, aS), each shape (Nx1, Nx2).
        """
        Ngc = grid.Ngc
        Nx1, Nx2 = grid.Nx1, grid.Nx2
        Nx1r, Nx2r = grid.Nx1r, grid.Nx2r

        aE = np.zeros((Nx1, Nx2))
        aW = np.zeros((Nx1, Nx2))
        aN = np.zeros((Nx1, Nx2))
        aS = np.zeros((Nx1, Nx2))

        V   = grid.cVol                          # (Nx1, Nx2)
        dx1 = grid.dx1[Ngc:Nx1r, Ngc:Nx2r]      # (Nx1, Nx2)
        dx2 = grid.dx2[Ngc:Nx1r, Ngc:Nx2r]

        # Default: Poisson (α = 1 everywhere)
        if alpha_full is None:
            alpha_full = np.ones(grid.grid_shape)
        if gamma_full is None:
            gamma_full = np.zeros(grid.grid_shape)

        ac = alpha_full[Ngc:Nx1r, Ngc:Nx2r]      # centre

        if Nx1 > 1:
            ae = alpha_full[Ngc+1:Nx1r+1, Ngc:Nx2r]   # east neighbour
            aw = alpha_full[Ngc-1:Nx1r-1, Ngc:Nx2r]   # west neighbour
            alpha_e = 0.5 * (ac + ae)
            alpha_w = 0.5 * (ac + aw)
            aE[:] = alpha_e * grid.fS1[1:, :]  / (V * dx1)
            aW[:] = alpha_w * grid.fS1[:-1, :] / (V * dx1)

        if Nx2 > 1:
            an = alpha_full[Ngc:Nx1r, Ngc+1:Nx2r+1]   # north
            a_s = alpha_full[Ngc:Nx1r, Ngc-1:Nx2r-1]   # south
            alpha_n = 0.5 * (ac + an)
            alpha_s = 0.5 * (ac + a_s)
            aN[:] = alpha_n * grid.fS2[:, 1:]  / (V * dx2)
            aS[:] = alpha_s * grid.fS2[:, :-1] / (V * dx2)

        gc = gamma_full[Ngc:Nx1r, Ngc:Nx2r]
        aP = -(aE + aW + aN + aS) + gc
        return (aP, aE, aW, aN, aS)

    # ================================================================
    # Grid hierarchy for multigrid
    # ================================================================

    def _build_hierarchy(self):
        """Create coarser grids by factor-2 coarsening until too small."""
        g0 = self.problem.grid
        nx1, nx2 = int(g0.Nx1), int(g0.Nx2)
        Ngc = int(g0.Ngc)
        is_2d = (nx2 > 1)

        while True:
            if nx1 % 2 != 0 or nx1 < 4:
                break
            if is_2d and (nx2 % 2 != 0 or nx2 < 4):
                break

            nx1 //= 2
            if is_2d:
                nx2 //= 2

            coarse = Grid(nx1, nx2, Ngc)
            if g0.geom == 'cart':
                coarse.CartesianGrid(g0.x1ini, g0.x1fin, g0.x2ini, g0.x2fin)
            elif g0.geom == 'cyl':
                coarse.CylindricalGrid(g0.x1ini, g0.x1fin, g0.x2ini, g0.x2fin)
            elif g0.geom == 'pol':
                coarse.PolarGrid(g0.x1ini, g0.x1fin, g0.x2ini, g0.x2fin)

            self._grids.append(coarse)
            self._stencils.append(self._build_stencil(coarse))

            if nx1 <= 4 and (nx2 <= 4 or not is_2d):
                break

    # ================================================================
    # Boundary conditions
    # ================================================================

    def _apply_bc(self, u, grid, homogeneous=False):
        """
        Fill ghost zones of *u* on *grid*.

        For ``homogeneous=True`` (error equation in multigrid), all
        boundary values are set to zero while keeping the same BC type.
        """
        Ngc  = int(grid.Ngc)
        Nx1r = int(grid.Nx1r)
        Nx2r = int(grid.Nx2r)
        bt   = self.problem.bc_type
        bv   = ({f: 0.0 for f in EllipticProblem._faces}
                if homogeneous else self.problem.bc_val)

        # --- x1 boundaries ---
        if grid.Nx1 > 1:
            dx1 = float(grid.dx1[Ngc, Ngc])
            self._fill_bc_1d(u, bt.get('x1lo', 'dirichlet'),
                             float(bv.get('x1lo', 0.0)),
                             Ngc, Nx1r, dx1, axis=0, side='lo')
            self._fill_bc_1d(u, bt.get('x1hi', 'dirichlet'),
                             float(bv.get('x1hi', 0.0)),
                             Ngc, Nx1r, dx1, axis=0, side='hi')

        # --- x2 boundaries ---
        if grid.Nx2 > 1:
            dx2 = float(grid.dx2[Ngc, Ngc])
            self._fill_bc_1d(u, bt.get('x2lo', 'dirichlet'),
                             float(bv.get('x2lo', 0.0)),
                             Ngc, Nx2r, dx2, axis=1, side='lo')
            self._fill_bc_1d(u, bt.get('x2hi', 'dirichlet'),
                             float(bv.get('x2hi', 0.0)),
                             Ngc, Nx2r, dx2, axis=1, side='hi')

    @staticmethod
    def _fill_bc_1d(u, btype, bval, Ngc, Nxr, dx, axis, side):
        """Fill ghost cells for one boundary face."""
        for k in range(1, Ngc + 1):
            if side == 'lo':
                ig = Ngc - k          # ghost index
                ir = Ngc + k - 1      # mirror interior index
            else:
                ig = Nxr + k - 1
                ir = Nxr - k

            if axis == 0:
                if btype == 'dirichlet':
                    u[ig, :] = 2.0 * bval - u[ir, :]
                elif btype == 'neumann':
                    if k == 1:
                        u[ig, :] = u[ir, :] + dx * bval
                    else:
                        u[ig, :] = u[Ngc - 1 if side == 'lo' else Nxr, :]
                elif btype == 'periodic':
                    src = (Nxr - k) if side == 'lo' else (Ngc + k - 1)
                    u[ig, :] = u[src, :]
            else:
                if btype == 'dirichlet':
                    u[:, ig] = 2.0 * bval - u[:, ir]
                elif btype == 'neumann':
                    if k == 1:
                        u[:, ig] = u[:, ir] + dx * bval
                    else:
                        u[:, ig] = u[:, Ngc - 1 if side == 'lo' else Nxr]
                elif btype == 'periodic':
                    src = (Nxr - k) if side == 'lo' else (Ngc + k - 1)
                    u[:, ig] = u[:, src]

    # ================================================================
    # Discrete operator and residual
    # ================================================================

    @staticmethod
    def _apply_operator(u, grid, stencil):
        """Compute L[u] on real cells.  Returns array (Nx1, Nx2)."""
        Ngc  = grid.Ngc
        Nx1r = grid.Nx1r
        Nx2r = grid.Nx2r
        aP, aE, aW, aN, aS = stencil

        Lu = aP * u[Ngc:Nx1r, Ngc:Nx2r]
        if grid.Nx1 > 1:
            Lu += aE * u[Ngc+1:Nx1r+1, Ngc:Nx2r]
            Lu += aW * u[Ngc-1:Nx1r-1, Ngc:Nx2r]
        if grid.Nx2 > 1:
            Lu += aN * u[Ngc:Nx1r, Ngc+1:Nx2r+1]
            Lu += aS * u[Ngc:Nx1r, Ngc-1:Nx2r-1]
        return Lu

    @staticmethod
    def _compute_residual(u, rhs, grid, stencil):
        """r = f − L[u] on real cells."""
        Ngc = grid.Ngc
        f = rhs[Ngc:grid.Nx1r, Ngc:grid.Nx2r]
        return f - EllipticSolver._apply_operator(u, grid, stencil)

    @staticmethod
    def _norm(field, grid):
        """Volume-weighted L2 norm of a real-cell array."""
        return np.sqrt(np.sum(grid.cVol * field**2))

    # ================================================================
    # Weighted-Jacobi smoother
    # ================================================================

    def _smooth(self, u, rhs, grid, stencil):
        """One damped-Jacobi sweep: u ← u + ω (f − L[u]) / aP."""
        self._apply_bc(u, grid,
                       homogeneous=(grid is not self._grids[0]))
        res = self._compute_residual(u, rhs, grid, stencil)
        aP = stencil[0]
        Ngc = grid.Ngc
        u[Ngc:grid.Nx1r, Ngc:grid.Nx2r] += (
            self.omega_smooth * res / aP)

    # ================================================================
    # Transfer operators for multigrid
    # ================================================================

    @staticmethod
    def _restrict(fine_real, fine_grid, coarse_grid):
        """
        Volume-weighted restriction from fine real-cell array to
        coarse real-cell array.
        """
        Nc1 = int(coarse_grid.Nx1)
        Nc2 = int(coarse_grid.Nx2)
        is_2d = (int(fine_grid.Nx2) > 1)

        if is_2d:
            v00 = fine_grid.cVol[0::2, 0::2][:Nc1, :Nc2]
            v10 = fine_grid.cVol[1::2, 0::2][:Nc1, :Nc2]
            v01 = fine_grid.cVol[0::2, 1::2][:Nc1, :Nc2]
            v11 = fine_grid.cVol[1::2, 1::2][:Nc1, :Nc2]
            f00 = fine_real[0::2, 0::2][:Nc1, :Nc2]
            f10 = fine_real[1::2, 0::2][:Nc1, :Nc2]
            f01 = fine_real[0::2, 1::2][:Nc1, :Nc2]
            f11 = fine_real[1::2, 1::2][:Nc1, :Nc2]
            return (v00*f00 + v10*f10 + v01*f01 + v11*f11) / coarse_grid.cVol
        else:
            v0 = fine_grid.cVol[0::2, :][:Nc1, :Nc2]
            v1 = fine_grid.cVol[1::2, :][:Nc1, :Nc2]
            f0 = fine_real[0::2, :][:Nc1, :Nc2]
            f1 = fine_real[1::2, :][:Nc1, :Nc2]
            return (v0*f0 + v1*f1) / coarse_grid.cVol

    @staticmethod
    def _prolongate(coarse_u, coarse_grid, fine_grid):
        """
        Bilinear prolongation of a full coarse-grid array (with ghost
        zones filled) to fine real-cell array.
        """
        Ngc  = int(coarse_grid.Ngc)
        Nc1r = int(coarse_grid.Nx1r)
        Nc2r = int(coarse_grid.Nx2r)
        Nf1  = int(fine_grid.Nx1)
        Nf2  = int(fine_grid.Nx2)
        is_2d = (int(fine_grid.Nx2) > 1)

        # Shortcuts for coarse neighbours
        cc = coarse_u[Ngc:Nc1r,     Ngc:Nc2r]
        cW = coarse_u[Ngc-1:Nc1r-1, Ngc:Nc2r]
        cE = coarse_u[Ngc+1:Nc1r+1, Ngc:Nc2r]

        fine_real = np.zeros((Nf1, Nf2))

        if is_2d:
            cS  = coarse_u[Ngc:Nc1r,     Ngc-1:Nc2r-1]
            cN  = coarse_u[Ngc:Nc1r,     Ngc+1:Nc2r+1]
            cSW = coarse_u[Ngc-1:Nc1r-1, Ngc-1:Nc2r-1]
            cSE = coarse_u[Ngc+1:Nc1r+1, Ngc-1:Nc2r-1]
            cNW = coarse_u[Ngc-1:Nc1r-1, Ngc+1:Nc2r+1]
            cNE = coarse_u[Ngc+1:Nc1r+1, Ngc+1:Nc2r+1]

            fine_real[0::2, 0::2] = (9*cc + 3*cW + 3*cS + cSW) / 16.0
            fine_real[1::2, 0::2] = (9*cc + 3*cE + 3*cS + cSE) / 16.0
            fine_real[0::2, 1::2] = (9*cc + 3*cW + 3*cN + cNW) / 16.0
            fine_real[1::2, 1::2] = (9*cc + 3*cE + 3*cN + cNE) / 16.0
        else:
            fine_real[0::2, :] = 0.75 * cc + 0.25 * cW
            fine_real[1::2, :] = 0.75 * cc + 0.25 * cE

        return fine_real

    # ================================================================
    # Multigrid V-cycle
    # ================================================================

    def _vcycle(self, level, u, rhs):
        """Recursive V-cycle.  *u* and *rhs* are full-grid arrays."""
        grid    = self._grids[level]
        stencil = self._stencils[level]
        Ngc     = int(grid.Ngc)
        homo    = (level > 0)

        # Coarsest level — smooth extensively
        if level == len(self._grids) - 1:
            for _ in range(self.mg_bottom_iter):
                self._smooth(u, rhs, grid, stencil)
            self._apply_bc(u, grid, homogeneous=homo)
            return

        # Pre-smoothing
        for _ in range(self.mg_pre):
            self._smooth(u, rhs, grid, stencil)
        self._apply_bc(u, grid, homogeneous=homo)

        # Residual
        res = self._compute_residual(u, rhs, grid, stencil)

        # Restrict residual to coarser grid
        cg = self._grids[level + 1]
        res_c = self._restrict(res, grid, cg)

        # Coarse error and RHS
        e_c   = np.zeros(cg.grid_shape)
        rhs_c = np.zeros(cg.grid_shape)
        rhs_c[cg.Ngc:cg.Nx1r, cg.Ngc:cg.Nx2r] = res_c

        # Recurse
        self._vcycle(level + 1, e_c, rhs_c)

        # Prolongate correction
        self._apply_bc(e_c, cg, homogeneous=True)
        correction = self._prolongate(e_c, cg, grid)
        u[Ngc:grid.Nx1r, Ngc:grid.Nx2r] += correction

        # Post-smoothing
        self._apply_bc(u, grid, homogeneous=homo)
        for _ in range(self.mg_post):
            self._smooth(u, rhs, grid, stencil)
        self._apply_bc(u, grid, homogeneous=homo)

    # ================================================================
    # Outer solver: multigrid
    # ================================================================

    def _solve_multigrid(self):
        """Multigrid V-cycle iteration until convergence."""
        grid    = self._grids[0]
        stencil = self._stencils[0]
        rhs     = self.problem.rhs

        self._apply_bc(self.u, grid, homogeneous=False)

        for it in range(self.max_iter):
            self._vcycle(0, self.u, rhs)

            self._apply_bc(self.u, grid, homogeneous=False)
            res   = self._compute_residual(self.u, rhs, grid, stencil)
            rnorm = self._norm(res, grid)
            self.residual_history.append(rnorm)

            if self.verbose and (it + 1) % self.print_every == 0:
                print(f'  MG iter {it+1:4d}: ||res|| = {rnorm:.6e}')

            if rnorm < self.tol:
                self.converged = True
                if self.verbose:
                    print(f'  MG converged in {it+1} iterations '
                          f'(||res|| = {rnorm:.6e})')
                return

        if self.verbose:
            print(f'  MG did NOT converge in {self.max_iter} iterations '
                  f'(||res|| = {self.residual_history[-1]:.6e})')

    # ================================================================
    # Outer solver: Conjugate Gradient
    # ================================================================

    def _solve_cg(self):
        """
        Conjugate Gradient with volume-weighted inner product.

        The discrete Laplacian L is negative (semi-)definite, so CG is
        applied to the system  (−L) u = −f  where −L is SPD.
        """
        grid    = self._grids[0]
        stencil = self._stencils[0]
        Ngc     = int(grid.Ngc)
        Nx1r    = int(grid.Nx1r)
        Nx2r    = int(grid.Nx2r)
        rhs     = self.problem.rhs
        V       = grid.cVol            # (Nx1, Nx2)

        self._apply_bc(self.u, grid, homogeneous=False)

        # Initial CG-residual: r = (−f) − (−L)u = Lu − f = −(f − Lu)
        r = -self._compute_residual(self.u, rhs, grid, stencil)
        p = r.copy()
        rsold = np.sum(V * r * r)

        for k in range(self.max_iter):
            # Apply (−L) to p:  Ap = −L[p]
            p_full = np.zeros(grid.grid_shape)
            p_full[Ngc:Nx1r, Ngc:Nx2r] = p
            self._apply_bc(p_full, grid, homogeneous=True)
            Ap = -self._apply_operator(p_full, grid, stencil)

            pAp = np.sum(V * p * Ap)
            if abs(pAp) < 1e-30:
                break
            alpha = rsold / pAp

            self.u[Ngc:Nx1r, Ngc:Nx2r] += alpha * p
            self._apply_bc(self.u, grid, homogeneous=False)

            r -= alpha * Ap
            rsnew = np.sum(V * r * r)

            rnorm = np.sqrt(rsnew)
            self.residual_history.append(rnorm)

            if self.verbose and (k + 1) % self.print_every == 0:
                print(f'  CG iter {k+1:4d}: ||res|| = {rnorm:.6e}')

            if rnorm < self.tol:
                self.converged = True
                if self.verbose:
                    print(f'  CG converged in {k+1} iterations '
                          f'(||res|| = {rnorm:.6e})')
                return

            p = r + (rsnew / (rsold + 1e-30)) * p
            rsold = rsnew

        if self.verbose:
            print(f'  CG did NOT converge in {self.max_iter} iterations '
                  f'(||res|| = {self.residual_history[-1]:.6e})')
