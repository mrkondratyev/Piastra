"""
===============================================================================
high_order_rec.py
===============================================================================

High-order reconstruction routines for finite-volume fluid solvers in 2D.

This module provides a collection of spatial reconstruction methods for
finite-volume schemes used in computational astrophysics and fluid dynamics.
These routines compute the left and right states of a fluid variable at cell
faces, ready for flux evaluation by a Riemann solver. The module supports
uniform Cartesian grids and includes the following reconstruction schemes:

    1. PCM    : Piecewise-constant (1st order, no slope limiting)
    2. PLM    : Piecewise-linear (2nd order, monotonicity-limited)
    3. WENO   : Weighted Essentially Non-Oscillatory (3rd or 5th order)
    4. PPMorig: Standard Piecewise Parabolic Method (3rd order)
    5. PPM    : Fifth-order improved PPM (Mignone 2014)
    6. MP5    : Fifth-order monotonicity preserving scheme (Suresh & Huynh 1997)

Key routines
------------
- VarReconstruct : Unified interface for selecting reconstruction type
- rec_PLM        : Limited piecewise-linear reconstruction
- limiter        : Slope limiter for PLM and PPM schemes
- rec_WENO       : WENO/CWENO reconstruction for high-order accuracy
- rec_PPMorig    : Standard PPM reconstruction (3rd order)
- rec_PPM5       : Fifth-order PPM reconstruction
- rec_MP5       : Fifth-order MP5 reconstruction

Notes
-----
- All routines assume the presence of ghost cells (Ngc ≥ 2 for PLM/PCM,
  Ngc ≥ 3 for PPM/WENO/MP5).
- Designed for modular integration into finite-volume fluid solvers.

References
----------
- Colella, P. & Woodward, P. R. (1984). The Piecewise Parabolic Method (PPM)
  for gas-dynamical simulations. JCP 54, 174.
- Mignone, A. (2014). High-order conservative reconstruction schemes for
  finite volume methods in cylindrical and spherical coordinates.
  JCP 270, 784.
- Balsara, D. S. (2017). Higher-order accurate space-time schemes for
  computational astrophysics—Part I: finite volume methods.
  Living Rev Comput Astrophys 3:2.
- Suresh, A., & Huynh, H.T. (1997). Accurate Monotonicity-Preserving Schemes
  with Runge-Kutta Time Stepping. Journal of Computational Physics, 136, 83-99.

Author: mrkondratyev
===============================================================================
"""


import numpy as np


# ─── Module-level constants ───────────────────────────────────────────────────

# Fifth-order PPM/MP5 interpolation coefficients (Mignone 2014)
_PPM5_COEFFS = np.array([2.0, -13.0, 47.0, 27.0, -3.0]) / 60.0

# Legendre polynomial shift for cell-average-preserving reconstruction:
#   (0.5)**2 - 1.0/12.0 = 1/6
_LEG_SHIFT = 1.0 / 6.0


# ─── Public interface ─────────────────────────────────────────────────────────

def VarReconstruct(var, grid, rec_type, dim, limiter_type=None):
    """
    High-order reconstruction of a fluid variable for use in finite-volume schemes.
    This routine reconstructs the state variable at the **faces of each cell** in a desired dimension
    for input to a Riemann solver.
    
    Parameters
    ----------
    var : ndarray
        2D array of the fluid variable including ghost cells.
    grid : object
        Grid class object containing cell centers and face coordinates (e.g., grid.cx1, grid.cx2, grid.fx1, grid.fx2).
    rec_type : str
        Type of reconstruction. Supported options:
        - 'PCM'   : Piecewise-constant (1st order in space)
        - 'PLM'   : Piecewise-linear (2nd order in space)
        - 'WENO'  : Weighted ENO (3rd order for CWENO or 5th order for WENO5)
        - 'PPMorig': Standard PPM (3rd order)
        - 'PPM'   : Fifth-order PPM (Mignone 2014)
        - 'MP5'   : Fifth-order MP5
    dim : int
    
    limiter_type : str, optional
        PLM slope limiter ('VL', 'MM', 'MC', 'KOR'). If None, uses 'VL' (default).
        Dimension along which to perform the reconstruction (1 or 2).
    
    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the **left side** of each cell face.
    var_rec_R : ndarray
        Reconstructed variable at the **right side** of each cell face.
    """
    if rec_type == 'PCM':
        # First-order piecewise-constant: copy cell averages to faces
        if dim == 1:
            var_rec_L = var[grid.Ngc - 1 : grid.Nx1r,     grid.Ngc : -grid.Ngc]
            var_rec_R = var[grid.Ngc     : grid.Nx1r + 1,  grid.Ngc : -grid.Ngc]
        else:
            var_rec_L = var[grid.Ngc : -grid.Ngc,  grid.Ngc - 1 : grid.Nx2r]
            var_rec_R = var[grid.Ngc : -grid.Ngc,  grid.Ngc     : grid.Nx2r + 1]
        return var_rec_L, var_rec_R

    elif rec_type == 'PLM':
        return rec_PLM(grid, var, dim, limiter_type=None)

    elif rec_type == 'WENO':
        Nr = grid.Nx1r if dim == 1 else grid.Nx2r
        return rec_WENO(grid.Ngc, Nr, var, dim)

    elif rec_type == 'PPMorig':
        Nr = grid.Nx1r if dim == 1 else grid.Nx2r
        return rec_PPMorig(grid.Ngc, Nr, var, dim)

    elif rec_type == 'PPM':
        Nr = grid.Nx1r if dim == 1 else grid.Nx2r
        return rec_PPM5(grid.Ngc, Nr, var, dim)
    
    elif rec_type == 'MP5':
        Nr = grid.Nx1r if dim == 1 else grid.Nx2r
        return rec_MP5(grid.Ngc, Nr, var, dim)

    else:
        raise ValueError(
            f"Unknown rec_type: {rec_type}. "
            f"Expected one of ['PCM', 'PLM', 'WENO', 'PPMorig', 'PPM']."
        )


# ─── PLM reconstruction ──────────────────────────────────────────────────────

def rec_PLM(grid, var, dim, limiter_type=None):
    """
    Limited second-order piecewise linear method (PLM) for finite volume solvers.

    Parameters
    ----------
    grid : GRID class object
        Grid object containing cell-centered coordinates, face coordinates, and ghost cell information.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).
    limiter_type : str, optional
        PLM slope limiter type 
        
    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    The PLM method reconstructs the solution at cell faces up to the second order in space
    using linear extensions of cell-averaged values. Slopes are computed using neighboring cells,
    and a slope limiter is applied to maintain monotonicity and the TVD property.

    References
    ----------
    - D.S. Balsara, Living Rev Comput Astrophys (2017) 3:2.
    - van Leer, B. (1974)
    - Sweby, P. K. (1984)
    """
    Ngc = grid.Ngc

    # Select dimension-specific grid data; transpose dim-2 to reuse dim-1 logic
    if dim == 1:
        Nr = grid.Nx1r
        cx, fx = grid.cx1, grid.fx1
    else:
        Nr = grid.Nx2r
        cx, fx = grid.cx2.T, grid.fx2.T
        var = var.T

    if limiter_type is None:
        limiter_type = 'VL'

    # Named stencil slices for the cell-centered variable
    v_im1 = var[Ngc - 2 : Nr,     Ngc : -Ngc]   # cell i-1
    v_i   = var[Ngc - 1 : Nr + 1, Ngc : -Ngc]   # cell i
    v_ip1 = var[Ngc     : Nr + 2, Ngc : -Ngc]   # cell i+1

    # Corresponding cell-center coordinates
    cx_im1 = cx[Ngc - 2 : Nr,     Ngc : -Ngc]
    cx_i   = cx[Ngc - 1 : Nr + 1, Ngc : -Ngc]
    cx_ip1 = cx[Ngc     : Nr + 2, Ngc : -Ngc]

    # Left and right gradients across each cell
    grad_L = (v_i - v_im1) / (cx_i - cx_im1)
    grad_R = (v_ip1 - v_i) / (cx_ip1 - cx_i)

    # Slope-limited gradient (ensures monotonicity)
    lim_grad = limiter(grad_L, grad_R, limiter_type)

    # Face coordinates and adjacent cell centers for extrapolation
    fx_face  = fx[Ngc : -Ngc,     Ngc : -Ngc]
    cx_left  = cx[Ngc - 1 : -Ngc, Ngc : -Ngc]   # center of cell left of face
    cx_right = cx[Ngc : -Ngc + 1, Ngc : -Ngc]   # center of cell right of face

    # Left state: extrapolate from the left cell to the face
    var_rec_L = v_i[:-1, :] + (fx_face - cx_left) * lim_grad[:-1, :]

    # Right state: extrapolate from the right cell to the face
    var_rec_R = v_i[1:, :]  + (fx_face - cx_right) * lim_grad[1:, :]

    # Transpose back for dim-2
    if dim == 2:
        var_rec_L = var_rec_L.T
        var_rec_R = var_rec_R.T

    return var_rec_L, var_rec_R


# ─── Slope limiter ────────────────────────────────────────────────────────────

def limiter(x, y, limiter_type):
    """
    Slope limiter for second-order monotonic piecewise linear reconstruction.
    
    Parameters
    ----------
    x : ndarray
        Left gradient (or left difference).
    y : ndarray
        Right gradient (or right difference).
    limiter_type : str
        Limiter type: 'VL', 'MM', 'MC', 'KOR', 'PCM', or 'NO'.
    
    Returns
    -------
    df : ndarray
        Limited slope to ensure monotonicity.
    
    Description
    -----------
    The limiter enforces monotonicity on the linear reconstruction in PLM/PPM schemes.
    When gradients x and y have opposite signs (local extremum), the slope is zeroed.
    When they differ greatly in magnitude (near discontinuity), the slope is reduced.

    Notes
    -----
    - VL  : van Leer — smooth harmonic-mean limiter
    - MM  : minmod — most diffusive but robust
    - MC  : monotonized central — good balance of accuracy and stability
    - KOR : Koren third-order — accurate on uniform grids
    - PCM : piecewise constant (zero slope, for testing)
    - NO  : unlimited — may produce oscillations (Lax-Wendroff-like)
    """
    if limiter_type == 'VL':
        # van Leer limiter: harmonic mean 2|x||y|/(|x|+|y|) with correct sign,
        # zero when x and y have opposite signs
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        df = (np.sign(x) + np.sign(y)) * (abs_x * abs_y) / (abs_x + abs_y + 1e-30)

    elif limiter_type == 'MM':
        # Minmod limiter: smallest magnitude when same sign, zero otherwise
        df = 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))

    elif limiter_type == 'MC':
        # Monotonized central limiter (beta = 2):
        #   minmod(2*x, 2*y, (x+y)/2) — zero when signs differ
        same_sign = 0.5 * (np.sign(x) + np.sign(y))
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        df = same_sign * np.minimum(
            np.minimum(2.0 * abs_x, 2.0 * abs_y),
            0.5 * (abs_x + abs_y)
        )

    elif limiter_type == 'KOR':
        # Koren third-order limiter (third order only for uniform Cartesian grids)
        r = y / (x + np.copysign(1e-30, x))
        df = x * np.maximum(0.0, np.minimum(2.0 * r, np.minimum(1.0 / 3.0 + 2.0 * r / 3.0, 2.0)))

    elif limiter_type == 'PCM':
        # First-order scheme (for tests)
        df = 0.0

    elif limiter_type == 'NO':
        # Unlimited second-order reconstruction (may produce oscillations)
        df = x  # Lax-Wendroff-like scheme

    else:
        raise ValueError(
            f"Unknown limiter_type: {limiter_type}. "
            f"Expected one of ['VL', 'MM', 'MC', 'KOR', 'PCM', 'NO']."
        )

    return df


# ─── WENO reconstruction ─────────────────────────────────────────────────────

def rec_WENO(Ngc, Nr, var, dim):
    """
    WENO (Weighted Essentially Non-Oscillatory) reconstruction for finite volume solvers.

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    WENO uses higher-order polynomial extensions based on several candidate stencils.
    Three stencils (left, central, right) each provide second-order reconstructions.
    Their smoothness indicators (IS) determine weights: discontinuous stencils are
    suppressed while smooth ones dominate, yielding up to fifth-order accuracy.

    Currently restricted to uniform Cartesian grids.

    References
    ----------
    - D.S. Balsara, Living Rev Comput Astrophys (2017) 3:2.
    - Shu, C.-W. (1998)
    """
    # Transpose dim-2 to reuse dim-1 logic
    if dim == 2:
        var = var.T

    # Named stencil slices: 5-cell stencil centered on cell i
    v_im2 = var[Ngc - 3 : Nr - 1, Ngc : -Ngc]   # cell i-2
    v_im1 = var[Ngc - 2 : Nr,     Ngc : -Ngc]   # cell i-1
    v_i   = var[Ngc - 1 : Nr + 1, Ngc : -Ngc]   # cell i
    v_ip1 = var[Ngc     : Nr + 2, Ngc : -Ngc]   # cell i+1
    v_ip2 = var[Ngc + 1 : Nr + 3, Ngc : -Ngc]   # cell i+2

    # --- First and second derivatives for each candidate stencil ---

    # Left stencil: cells [i-2, i-1, i]
    uxl  = -2.0 * v_im1 + 1.5 * v_i + 0.5 * v_im2
    uxxl =  0.5 * (v_im2 - 2.0 * v_im1 + v_i)

    # Central stencil: cells [i-1, i, i+1]
    uxc  = 0.5 * v_ip1 - 0.5 * v_im1
    uxxc = 0.5 * (v_im1 - 2.0 * v_i + v_ip1)

    # Right stencil: cells [i, i+1, i+2]
    uxr  = 2.0 * v_ip1 - 1.5 * v_i - 0.5 * v_ip2
    uxxr = 0.5 * (v_i - 2.0 * v_ip1 + v_ip2)

    # --- Smoothness indicators ---
    ISl = uxl ** 2 + (13.0 / 3.0) * uxxl ** 2
    ISc = uxc ** 2 + (13.0 / 3.0) * uxxc ** 2
    ISr = uxr ** 2 + (13.0 / 3.0) * uxxr ** 2

    # --- WENO5 nonlinear weights ---
    # Linear weights for fifth-order accuracy
    gammal, gammac, gammar = 0.1, 0.6, 0.3
    IS_deg = 2  # exponent in weight denominator

    # Unnormalized weights (smooth stencils get large weights)
    wl = gammal / (ISl + 1e-12) ** IS_deg
    wc = gammac / (ISc + 1e-12) ** IS_deg
    wr = gammar / (ISr + 1e-12) ** IS_deg

    # Normalize to convex combination
    w_sum_inv = 1.0 / (wl + wc + wr)
    wwl = wl * w_sum_inv
    wwc = wc * w_sum_inv
    wwr = wr * w_sum_inv

    # --- Weighted derivatives ---
    ux  = wwl * uxl  + wwc * uxc  + wwr * uxr
    uxx = wwl * uxxl + wwc * uxxc + wwr * uxxr

    # --- Final face states via Legendre expansion ---
    # Left state at face i+1/2 (extrapolated from cell i)
    var_rec_L = v_i[:-1, :] + ux[:-1, :] * 0.5 + uxx[:-1, :] * _LEG_SHIFT

    # Right state at face i+1/2 (extrapolated from cell i+1)
    var_rec_R = v_i[1:, :]  - ux[1:, :]  * 0.5 + uxx[1:, :]  * _LEG_SHIFT

    if dim == 2:
        var_rec_L = var_rec_L.T
        var_rec_R = var_rec_R.T

    return var_rec_L, var_rec_R


# ─── Standard PPM reconstruction ─────────────────────────────────────────────

def rec_PPMorig(Ngc, Nr, var, dim):
    """
    Standard third-order Piecewise Parabolic Method (PPM) following Collela & Woodward (1984).

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    PPM constructs a parabolic profile within each cell. The procedure is:
      1. Compute limited slopes between neighboring cells.
      2. Construct preliminary face values using the parabolic profile.
      3. Enforce monotonicity (face values within allowed bounds).
      4. Regulate curvature to prevent new extrema inside cells.

    Works only for uniform Cartesian grids.

    References
    ----------
    - Collela, P., & Woodward, P.R. (1984). The PPM for Gas-Dynamical Simulations.
    - D.S. Balsara, Living Rev Comput Astrophys (2017) 3:2.
    """
    lim_type = 'MC'

    # Transpose dim-2 to reuse dim-1 logic
    if dim == 2:
        var = var.T

    # --- Step 1: limited differences over extended stencil ---
    # Extended stencil for slope computation (real cells + 1 ghost on each side)
    dv_right = var[Ngc - 1 : Nr + 3, Ngc : -Ngc] - var[Ngc - 2 : Nr + 2, Ngc : -Ngc]
    dv_left  = var[Ngc - 2 : Nr + 2, Ngc : -Ngc] - var[Ngc - 3 : Nr + 1, Ngc : -Ngc]
    deltaU = limiter(dv_right, dv_left, lim_type)

    # --- Step 2: preliminary face values ---
    # Face value at i+1/2, using cells i and i+1 plus limited slope correction
    v_left  = var[Ngc - 2 : Nr + 1, Ngc : -Ngc]   # cell to the left of each face
    v_right = var[Ngc - 1 : Nr + 2, Ngc : -Ngc]   # cell to the right of each face
    fvar0 = v_left + 0.5 * (v_right - v_left) - (deltaU[1:, :] - deltaU[:-1, :]) / 6.0

    # Cell-centered values for monotonicity checks
    v_i = var[Ngc - 1 : Nr + 1, Ngc : -Ngc]

    # --- Step 3: enforce face values within allowed interval ---
    # If (fvar0_right - v_i) * (v_i - fvar0_left) < 0, cell is at an extremum: flatten
    product = (fvar0[1:, :] - v_i) * (v_i - fvar0[:-1, :])
    is_extremum = product < 0.0

    fvar0_L = np.where(is_extremum, v_i, fvar0[:-1, :])
    fvar0_R = np.where(is_extremum, v_i, fvar0[1:, :])

    # --- Step 4: regulate curvature to prevent internal extrema ---
    diff = fvar0_R - fvar0_L
    avg  = 0.5 * (fvar0_R + fvar0_L)
    diff_sq_sixth = diff ** 2 / 6.0

    var_rec_L = np.where(
        diff * (v_i - avg) > diff_sq_sixth,
        3.0 * v_i - 2.0 * fvar0_R,
        fvar0_L
    )
    var_rec_R = np.where(
        diff * (v_i - avg) < -diff_sq_sixth,
        3.0 * v_i - 2.0 * fvar0_L,
        fvar0_R
    )

    # --- Final face states via Legendre expansion ---
    ux  = var_rec_R - var_rec_L
    uxx = 3.0 * var_rec_R - 6.0 * v_i + 3.0 * var_rec_L

    var_rec_L = v_i[:-1, :] + ux[:-1, :] * 0.5 + uxx[:-1, :] * _LEG_SHIFT
    var_rec_R = v_i[1:, :]  - ux[1:, :]  * 0.5 + uxx[1:, :]  * _LEG_SHIFT

    if dim == 2:
        var_rec_L = var_rec_L.T
        var_rec_R = var_rec_R.T

    return var_rec_L, var_rec_R


# ─── Fifth-order PPM reconstruction ──────────────────────────────────────────

def rec_PPM5(Ngc, Nr, var, dim):
    """
    Fifth-order PPM reconstruction following A. Mignone (JCP, 2014).

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    Improves on standard PPM by using a five-point stencil for fifth-order accuracy:
      1. Compute preliminary face values via weighted 5-cell combination.
      2. Clamp face values to lie between neighboring cell averages (monotonicity).
      3. Adjust reconstructed states based on local extrema and slope ratios.

    Works only for uniform Cartesian grids.

    References
    ----------
    - Mignone, A. (2014). JCP.
    - Collela, P., & Woodward, P.R. (1984). PPM for Gas-Dynamical Simulations.
    """
    c = _PPM5_COEFFS  # [c0, c1, c2, c3, c4] = [2, -13, 47, 27, -3] / 60

    # Transpose dim-2 to reuse dim-1 logic
    if dim == 2:
        var = var.T

    # Named stencil slices: 5-cell stencil centered on cell i
    v_im2 = var[Ngc - 3 : Nr - 1, Ngc : -Ngc]
    v_im1 = var[Ngc - 2 : Nr,     Ngc : -Ngc]
    v_i   = var[Ngc - 1 : Nr + 1, Ngc : -Ngc]
    v_ip1 = var[Ngc     : Nr + 2, Ngc : -Ngc]
    v_ip2 = var[Ngc + 1 : Nr + 3, Ngc : -Ngc]

    # --- Step 1: fifth-order face value estimates ---
    # var_L biased toward the left, var_R biased toward the right
    var_L = v_im2 * c[4] + v_im1 * c[3] + v_i * c[2] + v_ip1 * c[1] + v_ip2 * c[0]
    var_R = v_im2 * c[0] + v_im1 * c[1] + v_i * c[2] + v_ip1 * c[3] + v_ip2 * c[4]

    # --- Step 2: clamp to neighboring cell range (monotonicity) ---
    var_L = np.clip(var_L, np.minimum(v_im1, v_i), np.maximum(v_im1, v_i))
    var_R = np.clip(var_R, np.minimum(v_i, v_ip1), np.maximum(v_i, v_ip1))

    # --- Step 3: adjust near extrema ---
    dvar_R = var_R - v_i
    dvar_L = var_L - v_i

    # When dvar_R and dvar_L have the same sign, cell i is a local extremum: flatten
    same_sign = dvar_R * dvar_L >= 0.0

    var_rec_L = np.where(
        same_sign, v_i,
        np.where(np.abs(dvar_L) >= 2.0 * np.abs(dvar_R),
                 v_i - 2.0 * dvar_R,
                 v_i + dvar_L)
    )

    var_rec_R = np.where(
        same_sign, v_i,
        np.where(np.abs(dvar_R) >= 2.0 * np.abs(dvar_L),
                 v_i - 2.0 * dvar_L,
                 v_i + dvar_R)
    )

    # --- Final face states ---
    # Left state at face i+1/2 comes from the right side of cell i
    # Right state at face i+1/2 comes from the left side of cell i+1
    var_rec_L, var_rec_R = var_rec_R[:-1, :], var_rec_L[1:, :]

    if dim == 2:
        var_rec_L = var_rec_L.T
        var_rec_R = var_rec_R.T

    return var_rec_L, var_rec_R




# ─── MP5 reconstruction ───────────────────────────────────────────────────────

def rec_MP5(Ngc, Nr, var, dim):
    """
    Fifth-order Monotonicity-Preserving (MP5) reconstruction following Suresh & Huynh (1997).

    Parameters
    ----------
    Ngc : int
        Number of ghost cells in each dimension.
    Nr : int
        Number of real cells in the desired dimension.
    var : ndarray
        2D array of the state variable to reconstruct (including ghost cells).
    dim : int
        Dimension along which to perform the reconstruction (1 or 2).

    Returns
    -------
    var_rec_L : ndarray
        Reconstructed variable at the left side of the cell faces.
    var_rec_R : ndarray
        Reconstructed variable at the right side of the cell faces.

    Description
    -----------
    MP5 uses an upwind-biased five-point stencil to achieve fifth-order accuracy
    on smooth flows while preserving monotonicity near discontinuities.
    The procedure follows Appendix A of Suresh & Huynh (1997) exactly:
      1. Compute the original (unlimited) fifth-order interface value VOR (Eq. 2.1).
      2. Compute VMP, the simple monotonicity-preserving bound (Eq. 2.12).
      3. If VOR already satisfies the MP constraint (Eq. 2.29), accept it as-is.
      4. Otherwise compute the accuracy-preserving bounds using the four-argument
         minmod d^M4 (Eq. 2.24), then UL, MD, and LC estimates (Eqs. 2.8, 2.25, 2.26),
         and finally clamp VOR into [vmin, vmax] via median (Eq. 2.28).

    The alpha parameter (alpha = 4) and tolerance epsilon (1e-10) are taken
    directly from the paper.

    Works only for uniform Cartesian grids.

    References
    ----------
    - Suresh, A., & Huynh, H.T. (1997). Accurate Monotonicity-Preserving Schemes
      with Runge-Kutta Time Stepping. Journal of Computational Physics, 136, 83-99.
    """
    alpha = 4.0
    epsm  = 1.0e-10

    # Transpose dim-2 to reuse dim-1 logic
    if dim == 2:
        var = var.T

    # Named stencil slices: 5-cell stencil around the face i+1/2
    # Left-biased reconstruction uses cells [i-2, i-1, i, i+1, i+2]
    # for right-biased reconstruction we simply inverse the order of the cells
    v_im2 = var[Ngc - 3 : Nr - 1, Ngc : -Ngc]   # cell i-2
    v_im1 = var[Ngc - 2 : Nr,     Ngc : -Ngc]   # cell i-1
    v_i   = var[Ngc - 1 : Nr + 1, Ngc : -Ngc]   # cell i
    v_ip1 = var[Ngc     : Nr + 2, Ngc : -Ngc]   # cell i+1
    v_ip2 = var[Ngc + 1 : Nr + 3, Ngc : -Ngc]   # cell i+2
    
    # ── Inline helpers (operate on arrays) ───────────────────────────────────
    def MM(x, y):
        """Usual two-argument minmod function"""
        return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))

    def MM4(w, x, y, z):
        """Four-argument minmod function"""
        s = (np.sign(w) + np.sign(x)) * np.abs(np.sign(w) + np.sign(y)) * np.abs(np.sign(w) + np.sign(z))
        res = 0.125 * s * np.minimum(np.minimum(np.abs(w), np.abs(x)), np.minimum(np.abs(y), np.abs(z)))
        return res

    def mp5_interface(vj, vjm1, vjp1, vjm2, vjp2):
        """
        Compute the MP5-limited left interface value v^L_{j+1/2}
        from cell averages centered on cell j, following the algorithm
        summarised on p. 10 of Suresh & Huynh (1997) and the Fortran
        listing in Appendix A.
        """
        # Step 1 — original fifth-order upwind-biased value (Eq. 2.1)
        # Coefficients: B1 = 1/60, weights [2, -13, 47, 27, -3]
        vor = (2.0 * vjm2 - 13.0 * vjm1 + 47.0 * vj + 27.0 * vjp1 - 3.0 * vjp2) / 60.0

        # Step 2 — simple MP bound (Eq. 2.12)
        vmp = vj + MM(vjp1 - vj, alpha * (vj - vjm1))

        # Step 3 — bypass test (Eq. 2.29): accept VOR when already in [vj, vmp]
        bypass = (vor - vj) * (vor - vmp) <= epsm

        # Step 4 — accuracy-preserving bounds (only needed where bypass is False)
        # Second differences (Eq. 2.19)
        djm1 = vjm2 - 2.0 * vjm1 + vj       # d_{j-1}
        dj   = vjm1 - 2.0 * vj   + vjp1     # d_j
        djp1 = vj   - 2.0 * vjp1 + vjp2     # d_{j+1}

        # Four-argument minmod d^M4 (Eq. 2.24)
        dm4_jph  = MM4(4.0 * dj - djp1, 4.0 * djp1 - dj, dj, djp1)   # at j+1/2
        dm4_jmh  = MM4(4.0 * dj - djm1, 4.0 * djm1 - dj, dj, djm1)   # at j-1/2

        # Upper limit (Eq. 2.8)
        vul = vj + alpha * (vj - vjm1)

        # Average (Eq. 2.16)
        vav = 0.5 * (vj + vjp1)

        # Median-based MD estimate (Eq. 2.25): v^MD = v^AV - (1/2) d^M4_{j+1/2}
        vmd = vav - 0.5 * dm4_jph

        # Large-curvature LC estimate (Eq. 2.26):
        # v^LC = v_j + (1/2)(v_j - v_{j-1}) + (4/3) d^M4_{j-1/2}
        # Fortran: B2 = 4/3
        vlc = vj + 0.5 * (vj - vjm1) + (4.0 / 3.0) * dm4_jmh

        # MP bounds (Eqs. 2.27a, 2.27b)
        vmin = np.maximum(np.minimum(vj, np.minimum(vjp1, vmd)),
                          np.minimum(vj, np.minimum(vul,  vlc)))
        vmax = np.minimum(np.maximum(vj, np.maximum(vjp1, vmd)),
                          np.maximum(vj, np.maximum(vul,  vlc)))

        # Clamp into [vmin, vmax] via two-argument minmod (Eq. 2.28)
        vlim = vor + MM(vmin - vor, vmax - vor)

        return np.where(bypass, vor, vlim)

    # ── Left state at face i+1/2: upwind from cell i ─────────────────────────
    vL = mp5_interface(v_i, v_im1, v_ip1, v_im2, v_ip2)

    # ── Right state at face i+1/2: upwind from cell i+1 (mirror stencil) ─────
    vR = mp5_interface(v_i, v_ip1, v_im1, v_ip2, v_im2)

    # ── Assemble face arrays ──────────────────────────────────────────────────
    var_rec_L = vL[:-1, :]   # left  state at face i+1/2 from cell i
    var_rec_R = vR[1:, :]    # right state at face i+1/2 from cell i+1

    if dim == 2:
        var_rec_L = var_rec_L.T
        var_rec_R = var_rec_R.T

    return var_rec_L, var_rec_R
