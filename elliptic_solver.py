# -*- coding: utf-8 -*-
"""
===============================================================================
elliptic_solver.py
===============================================================================

Iterative solver for a general second-order elliptic boundary-value problem
on the structured 2D grids provided by grid_setup.Grid.

The module targets equations of the form

    ∇·(α ∇u) + β·∇u + γ u = f          (*)

where α (scalar), β = (β1, β2) (vector), γ (scalar), and f (scalar) are
known coefficient fields defined at cell centres, and u is the unknown.
Special cases include:

- Poisson equation:  α = 1,  β = γ = 0   →   ∇²u = f
- Helmholtz equation: α = 1, β = 0        →   ∇²u + γ u = f
- Variable-coefficient diffusion:  β = γ = 0  →  ∇·(α ∇u) = f

Supported geometries: Cartesian (cart), Cylindrical (cyl), Polar (pol).
The metric terms follow the same conventions as grid_misc.py.

Author: mrkondratyev
"""

import numpy as np


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
        Diffusion coefficient α(x1, x2).  If scalar, broadcast to all cells.
        Default is 1.0.
    beta1, beta2 : ndarray or float, optional
        Advection-like coefficient vector β = (β1, β2).  Default is 0.0.
    gamma : ndarray or float, optional
        Reaction coefficient γ(x1, x2).  Default is 0.0.
    rhs : ndarray, optional
        Right-hand-side source f(x1, x2), shape (grid.Nx1, grid.Nx2) or
        grid.grid_shape. Default is 0.0.
    bc_type : dict, optional
        Boundary condition type for each face.  Keys are ``'x1lo'``,
        ``'x1hi'``, ``'x2lo'``, ``'x2hi'``.  Values are one of
        ``'dirichlet'``, ``'neumann'``, ``'periodic'``.
        Default: Dirichlet on all faces.
    bc_val : dict, optional
        Boundary values for each face (float or 1-D array along that face).
        For Dirichlet: value of u.  For Neumann: value of ∂u/∂n.
        Default is 0.0 on every face.

    Attributes
    ----------
    grid : Grid
    alpha, beta1, beta2, gamma : ndarray, shape grid.grid_shape
    rhs : ndarray, shape grid.grid_shape
    bc_type : dict
    bc_val : dict
    """

    _faces = ('x1lo', 'x1hi', 'x2lo', 'x2hi')

    def __init__(self, grid, *,
                 alpha=1.0, beta1=0.0, beta2=0.0, gamma=0.0,
                 rhs=None,
                 bc_type=None, bc_val=None):

        self.grid = grid
        shape = grid.grid_shape

        # --- Broadcast scalar coefficients to full grid arrays ---
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64),
                                     shape).copy()
        self.beta1 = np.broadcast_to(np.asarray(beta1, dtype=np.float64),
                                     shape).copy()
        self.beta2 = np.broadcast_to(np.asarray(beta2, dtype=np.float64),
                                     shape).copy()
        self.gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64),
                                     shape).copy()

        if rhs is None:
            self.rhs = np.zeros(shape, dtype=np.float64)
        else:
            self.rhs = np.asarray(rhs, dtype=np.float64).copy()
            if self.rhs.shape != shape:
                # Allow passing (Nx1, Nx2) — embed into full array
                tmp = np.zeros(shape, dtype=np.float64)
                Ngc = grid.Ngc
                tmp[Ngc:-Ngc, Ngc:-Ngc] = self.rhs
                self.rhs = tmp

        # --- Boundary conditions ---
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

    Planned algorithms
    ------------------
    - Jacobi iteration
    - Gauss–Seidel iteration (red-black ordering)
    - Successive Over-Relaxation (SOR)
    - Multigrid V-cycle (geometric)

    Parameters
    ----------
    problem : EllipticProblem
        Fully specified elliptic BVP.
    method : str, optional
        Solver algorithm.  One of ``'jacobi'``, ``'gauss_seidel'``,
        ``'sor'``, ``'multigrid'``.  Default is ``'sor'``.
    tol : float, optional
        Convergence tolerance on the L2 residual norm.  Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations.  Default is 10000.
    omega : float, optional
        SOR relaxation parameter (used only for ``method='sor'``).
        Default is 1.5.  Optimal ω depends on the problem and grid size;
        for a Poisson equation on an N×N grid, ω_opt ≈ 2/(1+sin(π/N)).
    verbose : bool, optional
        Print convergence information every ``print_every`` iterations.
        Default is False.
    print_every : int, optional
        Reporting interval when ``verbose`` is True.  Default is 100.

    Attributes
    ----------
    u : ndarray, shape grid.grid_shape
        Solution field (initialised to zero; updated by ``solve``).
    residual_history : list of float
        L2 residual norm after each iteration.
    converged : bool
        Whether the solver reached the requested tolerance.
    """

    _known_methods = ('jacobi', 'gauss_seidel', 'sor', 'multigrid')

    def __init__(self, problem, *,
                 method='sor', tol=1e-6, max_iter=10000,
                 omega=1.5, verbose=False, print_every=100):

        if method not in self._known_methods:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Expected one of {self._known_methods}.")

        self.problem = problem
        self.method  = method
        self.tol     = tol
        self.max_iter = max_iter
        self.omega   = omega
        self.verbose = verbose
        self.print_every = print_every

        # Solution array (full grid including ghost zones)
        self.u = np.zeros(problem.grid.grid_shape, dtype=np.float64)
        self.residual_history = []
        self.converged = False

    # ----------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------

    def solve(self, u0=None):
        """
        Solve the elliptic BVP.

        Parameters
        ----------
        u0 : ndarray, optional
            Initial guess for the solution.  If ``None``, the current
            value of ``self.u`` is used (zero by default).

        Returns
        -------
        u : ndarray, shape grid.grid_shape
            Converged (or best-effort) solution including ghost zones.

        Raises
        ------
        NotImplementedError
            Until the individual solver kernels are implemented.
        """
        if u0 is not None:
            self.u[:] = u0

        self._apply_bc()

        dispatch = {
            'jacobi':       self._solve_jacobi,
            'gauss_seidel': self._solve_gauss_seidel,
            'sor':          self._solve_sor,
            'multigrid':    self._solve_multigrid,
        }
        dispatch[self.method]()
        return self.u

    def residual(self):
        """
        Compute the algebraic residual r = f − L[u] on real cells.

        Returns
        -------
        res : ndarray, shape (grid.Nx1, grid.Nx2)
            Pointwise residual.
        """
        raise NotImplementedError("residual computation — coming soon")

    def residual_norm(self):
        """
        L2 norm of the residual over the computational domain.

        Returns
        -------
        float
        """
        res = self.residual()
        vol = self.problem.grid.cVol
        return np.sqrt(np.sum(vol * res**2))

    # ----------------------------------------------------------------
    # Boundary-condition application
    # ----------------------------------------------------------------

    def _apply_bc(self):
        """
        Fill ghost zones of ``self.u`` according to ``problem.bc_type``
        and ``problem.bc_val``.
        """
        raise NotImplementedError("boundary-condition application — coming soon")

    # ----------------------------------------------------------------
    # Solver kernels (stubs)
    # ----------------------------------------------------------------

    def _solve_jacobi(self):
        """Jacobi iteration kernel."""
        raise NotImplementedError(
            "Jacobi solver not yet implemented.  Contributions welcome!")

    def _solve_gauss_seidel(self):
        """Gauss–Seidel (red-black) iteration kernel."""
        raise NotImplementedError(
            "Gauss–Seidel solver not yet implemented.  Contributions welcome!")

    def _solve_sor(self):
        """Successive Over-Relaxation kernel."""
        raise NotImplementedError(
            "SOR solver not yet implemented.  Contributions welcome!")

    def _solve_multigrid(self):
        """Geometric multigrid V-cycle."""
        raise NotImplementedError(
            "Multigrid solver not yet implemented.  Contributions welcome!")

    # ----------------------------------------------------------------
    # Multigrid helpers (stubs)
    # ----------------------------------------------------------------

    def _restrict(self, fine):
        """
        Restrict a fine-grid array to the next coarser level
        (full-weighting restriction).
        """
        raise NotImplementedError

    def _prolongate(self, coarse):
        """
        Prolongate (interpolate) a coarse-grid correction to the finer level
        (bilinear interpolation).
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Operator assembly helpers (stubs)
    # ----------------------------------------------------------------

    def _build_stencil(self):
        """
        Assemble the 5-point stencil coefficients for the discrete
        elliptic operator L on the current grid and geometry.

        Returns arrays (aP, aE, aW, aN, aS) of shape (Nx1, Nx2) such
        that  L[u]_{i,j} = aP*u_{i,j} + aE*u_{i+1,j} + aW*u_{i-1,j}
                          + aN*u_{i,j+1} + aS*u_{i,j-1}.
        """
        raise NotImplementedError(
            "Stencil assembly not yet implemented.  Contributions welcome!")
